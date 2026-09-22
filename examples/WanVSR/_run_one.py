#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Env-driven single-input runner for FlashVSR v1.1 Tiny (2x super-resolution).

Adapted from infer_flashvsr_v1.1_tiny.py to run on CPU (Sapphire Rapids) as well
as CUDA, driven entirely by environment variables so it is easy to benchmark.

Env vars
--------
RUN_INPUT           : path to input video/dir  (required)
RUN_DEVICE          : cpu | cuda               (default cpu)
RUN_DTYPE           : fp32 | bf16              (default: fp32 on cpu, bf16 on cuda)
RUN_OUTDIR          : output directory         (default ./results_run)
RUN_TAG             : tag appended to output    (default <device>_<attn>)
RUN_SCALE           : upscale factor            (default 2.0)
RUN_SEED            : seed                       (default 0)
FLASHVSR_ATTN_IMPL  : dense | flex | lcsa       (default dense)  -> read by the DiT
OMP_NUM_THREADS     : (optional) CPU threads    -> also fed to torch.set_num_threads
"""

import sys, os, re, time
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
FLASHVSR_ROOT = os.path.dirname(os.path.dirname(ROOT_DIR))
# Resolve all relative inputs and local imports from the project root, regardless
# of the caller's current working directory.
os.chdir(ROOT_DIR)
# Insert FLASHVSR_ROOT first to use local diffsynth development version
sys.path.insert(0, FLASHVSR_ROOT)
sys.path.insert(1, ROOT_DIR)
import numpy as np
from PIL import Image
import imageio
from tqdm import tqdm
import torch
from einops import rearrange

from diffsynth import ModelManager, FlashVSRTinyPipeline
from utils.utils import Causal_LQ4x_Proj
from utils.TCDecoder import build_tcdecoder


# ----------------------------- config from env -----------------------------
RUN_INPUT  = os.environ.get("RUN_INPUT", "./inputs/33frames_input1_540p.mp4")
RUN_DEVICE = os.environ.get("RUN_DEVICE", "cpu").lower()
_default_dtype = "fp32" if RUN_DEVICE == "cpu" else "bf16"
RUN_DTYPE  = os.environ.get("RUN_DTYPE", _default_dtype).lower()
# torch dynamic INT8 quant (quantized.linear_dynamic) only accepts fp32
# activations -> it errors ("expected scalar type Float but found BFloat16") if
# the model runs bf16. Force fp32 whenever RUN_INT8=1 so the quantized Linear
# layers get fp32 inputs. NOTE: this also runs everything else (attention/VAE)
# in fp32, so compare the per-bucket [PROFILE] ffn= time (which isolates the
# quantized FFN GEMM), not the confounded total fps.
if RUN_DEVICE == "cpu" and os.environ.get("RUN_INT8", "0") == "1" and RUN_DTYPE == "bf16":
    print("[init] RUN_INT8=1: forcing RUN_DTYPE=fp32 (dynamic int8 needs fp32 activations)")
    RUN_DTYPE = "fp32"

RUN_OUTDIR = os.environ.get("RUN_OUTDIR", "./results_run")
ATTN_IMPL  = os.environ.get("FLASHVSR_ATTN_IMPL", "dense").lower()
RUN_TAG    = os.environ.get("RUN_TAG", f"{RUN_DEVICE}_{ATTN_IMPL}")
RUN_SCALE  = float(os.environ.get("RUN_SCALE", "2.0"))
RUN_SEED   = int(os.environ.get("RUN_SEED", "0"))
# Optional exact output size (multiples of 128). When both are set the frame is
# resized to fill exactly this size (whole picture kept, no crop, no padding).
RUN_OUT_W  = int(os.environ.get("RUN_OUT_W", "0"))
RUN_OUT_H  = int(os.environ.get("RUN_OUT_H", "0"))
# Tile-local window selection for localwin mode. The benchmark wrapper passes
# these values through env; OpenVINO and IPEX both honor the same settings.
LOCAL_TH = int(os.environ.get("LOCAL_TH", "8"))
LOCAL_TW = int(os.environ.get("LOCAL_TW", "15"))
if ATTN_IMPL not in {"dense", "localwin", "flex", "lcsa"}:
    raise ValueError(f"Unsupported FLASHVSR_ATTN_IMPL={ATTN_IMPL!r}; expected dense/localwin/flex/lcsa")
if ATTN_IMPL == "localwin" and (LOCAL_TH <= 0 or LOCAL_TW <= 0):
    raise ValueError(f"localwin mode requires LOCAL_TH>0 and LOCAL_TW>0, got TH={LOCAL_TH} TW={LOCAL_TW}")

DTYPE = torch.bfloat16 if RUN_DTYPE == "bf16" else torch.float32
DEVICE = RUN_DEVICE

_omp = os.environ.get("OMP_NUM_THREADS")
if _omp:
    try:
        torch.set_num_threads(int(_omp))
    except Exception:
        pass


# ----------------------------- IO helpers (from reference) -----------------------------
def tensor2video(frames):
    frames = rearrange(frames, "C T H W -> T H W C")
    frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
    return [Image.fromarray(frame) for frame in frames]


def deseam_video(frames, tw, th):
    """Suppress the FIXED localwin tile seams (nohalo tiling).

    Seams sit at tile boundaries: vertical every LOCAL_TW*128 px, horizontal
    every LOCAL_TH*128 px (block = 128 output px). Modes via DESEAM_MODE:
      deblock- DEFAULT, RECOMMENDED: an H.264/HEVC-style ADAPTIVE deblocking
               filter run ONLY on the known seam lines. First cancels the
               low-freq DC step, then a clamped 2-px-each-side weak filter
               that is GATED on local flatness -> it smooths the residual
               boundary edge but leaves genuine detail/texture untouched.
               No wide blur band. Knobs: DESEAM_ALPHA/BETA/TC.
      step   - gentle: cancel only the low-freq brightness/colour STEP across
               the seam (texture preserved; may leave a faint residual line).
      blur   - step-correct, THEN box-blur the whole seam band (LOSES detail;
               causes visible blurred bands -> not recommended).
      blend  - replace the seam band with a linear cross-fade (flattest;
               blurs all band detail -> not recommended).

    Cost is negligible (<1% of inference).

    Env:
      RUN_DESEAM=1     enable (default off)
      DESEAM_MODE      deblock | step | blur | blend   (default deblock)
      DESEAM_BAND      seam half-width in px (DC-step ramp) (default 8)
      DESEAM_SMOOTH    along-seam step smoothing            (default 15)
      DESEAM_ALPHA     deblock: max cross-edge jump to filter (default 40)
      DESEAM_BETA      deblock: max within-side variation=flat (default 15)
      DESEAM_TC        deblock: max px a sample may move (clip) (default 8)
      DESEAM_BLUR      blur mode kernel px (default 2*BAND+1, forced odd)
    """
    mode = os.environ.get("DESEAM_MODE", "strong").lower()
    band = max(1, int(os.environ.get("DESEAM_BAND", "8")))
    smooth = max(1, int(os.environ.get("DESEAM_SMOOTH", "15")))
    _kraw = int(os.environ.get("DESEAM_BLUR", "0") or "0")
    ksize = (_kraw if _kraw > 0 else 2 * band + 1) | 1  # 0/unset -> auto (2*band+1), forced odd
    alpha = float(os.environ.get("DESEAM_ALPHA", "40"))
    beta = float(os.environ.get("DESEAM_BETA", "15"))
    tc = float(os.environ.get("DESEAM_TC", "8"))
    taps = max(1, int(os.environ.get("DESEAM_TAPS", "4")))  # strong: px each side
    do_strong = mode == "strong"
    do_deblock = mode == "deblock"
    do_step = mode in ("step", "blur", "deblock", "strong")  # DC-step first
    do_blur = mode == "blur"
    do_blend = mode == "blend"
    Hb = max(1, th // 128)
    Wb = max(1, tw // 128)
    TW = int(os.environ.get("LOCAL_TW", "5"))
    TH = int(os.environ.get("LOCAL_TH", str(Hb)))
    nWj = Wb // TW if TW > 0 else 1
    nHi = Hb // TH if TH > 0 else 1
    xs = list(range(TW * 128, tw, TW * 128)) if nWj > 1 else []
    ys = list(range(TH * 128, th, TH * 128)) if nHi > 1 else []
    if not xs and not ys:
        print(f"[deseam] no interior tile seams (nHi={nHi} nWj={nWj}); skipped")
        return frames
    print(f"[deseam] mode={mode} band={band} smooth={smooth} "
          f"alpha={alpha} beta={beta} tc={tc} taps={taps} | "
          f"vseams={len(xs)}@{xs[:3]}... hseams={len(ys)}@{ys}")

    def _hblur(a, k):
        # box blur along axis 1 (across the seam), edge-padded, vectorized
        if k <= 1:
            return a
        pad = k // 2
        ap = np.pad(a, ((0, 0), (pad, pad), (0, 0)), mode="edge")
        cs = np.cumsum(ap, axis=1)
        cs = np.concatenate([np.zeros_like(cs[:, :1, :]), cs], axis=1)
        return (cs[:, k:, :] - cs[:, :-k, :]) / k

    def _smooth1d(a, k):
        # box filter along axis 0 (along the seam) of an (N, C) array
        if k <= 1:
            return a
        kern = np.ones(k, np.float32) / k
        return np.stack([np.convolve(a[:, c], kern, mode="same")
                         for c in range(a.shape[1])], axis=1)

    _lw = np.array([0.299, 0.587, 0.114], np.float32)  # BT.601 luma weights

    def _deblock_col(arr, c):
        """H.264-style adaptive weak deblock across vertical boundary at col c.

        Samples across the boundary:  p2 p1 p0 | q0 q1 q2   (p=left, q=right).
        Filter a row ONLY if the boundary looks like an artifact (small jump)
        AND both sides are locally flat -> genuine edges/texture are skipped.
        Corrections are clamped to +-tc so nothing moves far (no blurred band).
        """
        H, W, _ = arr.shape
        if c - 3 < 0 or c + 2 >= W:
            return
        p2, p1, p0 = arr[:, c - 3], arr[:, c - 2], arr[:, c - 1]     # (H, C)
        q0, q1, q2 = arr[:, c], arr[:, c + 1], arr[:, c + 2]
        Lp2, Lp1, Lp0 = p2 @ _lw, p1 @ _lw, p0 @ _lw                 # (H,)
        Lq0, Lq1, Lq2 = q0 @ _lw, q1 @ _lw, q2 @ _lw
        m = ((np.abs(Lp0 - Lq0) < alpha) &
             (np.abs(Lp1 - Lp0) < beta) &
             (np.abs(Lq1 - Lq0) < beta))                             # (H,) row mask
        if not m.any():
            return
        m1 = m[:, None]
        # weak filter: nudge the two boundary samples p0,q0 (clamped)
        delta = np.clip((4.0 * (q0 - p0) + (p1 - q1)) / 8.0, -tc, tc)
        arr[:, c - 1] = np.where(m1, p0 + delta, p0)
        arr[:, c]     = np.where(m1, q0 - delta, q0)
        # optionally soften p1/q1 too, only where that side is extra-flat
        mid = (p0 + q0) * 0.5
        ap = (np.abs(Lp2 - Lp0) < beta)[:, None] & m1
        aq = (np.abs(Lq2 - Lq0) < beta)[:, None] & m1
        dp1 = np.clip((p2 + mid - 2.0 * p1) * 0.5, -tc, tc)
        dq1 = np.clip((q2 + mid - 2.0 * q1) * 0.5, -tc, tc)
        arr[:, c - 2] = np.where(ap, p1 + dp1, p1)
        arr[:, c + 1] = np.where(aq, q1 + dq1, q1)

    def _strong_col(arr, c):
        """STRONG UNCONDITIONAL deblock: triangular low-pass over `taps` px
        each side of the seam (2*taps samples), applied to EVERY row (no gate).
        Each modified sample = weighted average of a +-taps window taken from
        the ORIGINAL neighbourhood, so the hard boundary is smoothed out over a
        narrow strip. Wider than the weak filter -> stronger, but bounded to
        +-taps px (no full-band blur)."""
        H, W, _ = arr.shape
        N = taps
        R = taps
        if c - N - R < 0 or c + N + R >= W:
            return
        kk = np.arange(-R, R + 1)
        w = (R + 1 - np.abs(kk)).astype(np.float32)   # triangular
        w /= w.sum()
        orig = arr[:, c - N - R:c + N + R, :].copy()  # (H, 2N+2R, C)
        base = R                                       # local idx of col (c-N)
        for i in range(2 * N):                         # cols c-N .. c+N-1
            acc = np.zeros_like(orig[:, 0, :])
            centre = base + i
            for j, wj in zip(kk, w):
                acc += wj * orig[:, centre + j, :]
            arr[:, c - N + i, :] = acc

    def _fix(arr, positions):
        H, W, _ = arr.shape
        for c in positions:
            lo, hi = max(1, c - band), min(W - 1, c + band)
            if hi - lo < 2:
                continue
            if do_step:
                la = arr[:, c - 2:c, :].mean(1)   # (H, C) just LEFT
                ra = arr[:, c:c + 2, :].mean(1)   # (H, C) just RIGHT
                half = _smooth1d(ra - la, smooth) * 0.5
                for i in range(band):
                    w = (band - i) / band
                    if c - 1 - i >= 0:
                        arr[:, c - 1 - i, :] += half * w
                    if c + i < W:
                        arr[:, c + i, :] -= half * w
            if do_deblock:
                _deblock_col(arr, c)
            if do_strong:
                _strong_col(arr, c)
            if do_blur:
                region = arr[:, lo:hi, :]
                blurred = _hblur(region, ksize)
                x = np.arange(hi - lo, dtype=np.float32)
                wgt = 0.5 * (1.0 + np.cos(np.pi * (x - (c - lo)) / band))
                wgt = np.clip(wgt, 0.0, 1.0)[None, :, None]
                arr[:, lo:hi, :] = (1.0 - wgt) * region + wgt * blurred
            if do_blend:
                left = arr[:, lo - 1:lo, :]
                right = arr[:, hi:hi + 1, :]
                n = hi - lo
                t = ((np.arange(n) + 1) / (n + 1)).astype(np.float32)[None, :, None]
                arr[:, lo:hi, :] = left * (1.0 - t) + right * t
        return arr

    out = []
    for im in frames:
        arr = np.asarray(im, np.float32)  # (H, W, C)
        arr = _fix(arr, xs)
        if ys:
            arr = _fix(arr.swapaxes(0, 1), ys).swapaxes(0, 1)
        out.append(Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)))
    return out


def natural_key(name: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'([0-9]+)', os.path.basename(name))]


def list_images_natural(folder: str):
    exts = ('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG')
    fs = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(exts)]
    fs.sort(key=natural_key)
    return fs


def largest_8n1_leq(n):
    return 0 if n < 1 else ((n - 1) // 8) * 8 + 1


def is_video(path):
    return os.path.isfile(path) and path.lower().endswith(('.mp4', '.mov', '.avi', '.mkv'))


def pil_to_tensor_neg1_1(img: Image.Image, dtype, device):
    t = torch.from_numpy(np.asarray(img, np.uint8)).to(device=device, dtype=torch.float32)
    t = t.permute(2, 0, 1) / 255.0 * 2.0 - 1.0
    return t.to(dtype)


def save_video(frames, save_path, fps=30, quality=6):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    w = imageio.get_writer(save_path, fps=fps, quality=quality)
    for f in tqdm(frames, desc=f"Saving {os.path.basename(save_path)}"):
        w.append_data(np.array(f))
    w.close()


def compute_scaled_and_target_dims(w0, h0, scale=2.0, multiple=128):
    sW = int(round(w0 * scale)); sH = int(round(h0 * scale))
    tW = ((sW + multiple - 1) // multiple) * multiple
    tH = ((sH + multiple - 1) // multiple) * multiple
    if tW == 0 or tH == 0:
        raise ValueError(f"Scaled size too small ({sW}x{sH}) for multiple={multiple}.")
    return sW, sH, tW, tH


def upscale_then_center_pad(img, scale, tW, tH):
    w0, h0 = img.size
    sW = int(round(w0 * scale)); sH = int(round(h0 * scale))
    up = img.resize((sW, sH), Image.BICUBIC)
    out = Image.new(img.mode, (tW, tH))
    l = (tW - sW) // 2; t = (tH - sH) // 2
    out.paste(up, (l, t))
    return out


def resize_to_fill(img, tW, tH):
    # Stretch the whole frame to exactly (tW, tH): no crop, no padding.
    return img.resize((tW, tH), Image.BICUBIC)


def prepare_input_tensor(path, scale, dtype, device):
    if not is_video(path):
        raise ValueError(f"Unsupported / missing input: {path}")
    rdr = imageio.get_reader(path)
    first = Image.fromarray(rdr.get_data(0)).convert('RGB')
    w0, h0 = first.size
    meta = {}
    try: meta = rdr.get_meta_data()
    except Exception: pass
    fps_val = meta.get('fps', 30)
    fps = int(round(fps_val)) if isinstance(fps_val, (int, float)) else 30

    def count_frames(r):
        nf = meta.get('nframes', None)
        if isinstance(nf, int) and nf > 0: return nf
        try: return r.count_frames()
        except Exception:
            n = 0
            try:
                while True: r.get_data(n); n += 1
            except Exception:
                return n

    total = count_frames(rdr)
    if total <= 0:
        rdr.close(); raise RuntimeError(f"Cannot read frames from {path}")

    print(f"[{os.path.basename(path)}] Resolution: {w0}x{h0} | Frames: {total} | FPS: {fps}")
    fill_mode = RUN_OUT_W > 0 and RUN_OUT_H > 0
    if fill_mode:
        tW = (RUN_OUT_W // 128) * 128
        tH = (RUN_OUT_H // 128) * 128
        if tW == 0 or tH == 0:
            rdr.close(); raise ValueError(f"RUN_OUT_W/H too small ({RUN_OUT_W}x{RUN_OUT_H}) for multiple=128.")
        print(f"[{os.path.basename(path)}] Exact output (fill, no crop): {w0}x{h0} -> Target: {tW}x{tH}")
    else:
        sW, sH, tW, tH = compute_scaled_and_target_dims(w0, h0, scale=scale, multiple=128)
        print(f"[{os.path.basename(path)}] Scaled (x{scale:.2f}): {sW}x{sH} -> Target: {tW}x{tH}")

    idx = list(range(total)) + [total - 1] * 4
    F = largest_8n1_leq(len(idx))
    if F == 0:
        rdr.close(); raise RuntimeError(f"Not enough frames in {path}.")
    idx = idx[:F]
    print(f"[{os.path.basename(path)}] Target Frames (8n-3): {F - 4}")

    frames = []
    try:
        for i in idx:
            img = Image.fromarray(rdr.get_data(i)).convert('RGB')
            if fill_mode:
                img_out = resize_to_fill(img, tW=tW, tH=tH)
            else:
                img_out = upscale_then_center_pad(img, scale=scale, tW=tW, tH=tH)
            frames.append(pil_to_tensor_neg1_1(img_out, dtype, device))
    finally:
        try: rdr.close()
        except Exception: pass

    vid = torch.stack(frames, 0).permute(1, 0, 2, 3).unsqueeze(0)  # 1 C F H W
    return vid, tH, tW, F, fps


# --------------------- AMX-int8 FFN (IPEX static quant) ---------------------
class _Int8Island(torch.nn.Module):
    """Let a traced/frozen fp32 int8 module live inside a bf16 DiT graph: cast
    the incoming activations to fp32 for the (fp32-I/O) oneDNN AMX-int8 kernel,
    then cast the result back to the surrounding dtype (bf16). The cast cost is
    negligible vs the FFN GEMM it replaces.

    Chunking was previously the default (INT8_FFN_CHUNK=4096) to keep the fp32
    intermediate LLC-resident, but it introduced per-chunk Python/JIT dispatch
    overhead (5-12 calls per forward at real T) that cost 3.5× the FFN time.
    Default is now one-shot (chunk=0); the fp32 intermediate at T=46080 is ~1.6 GB
    which the oneDNN brgemm kernel streams through efficiently since it's already
    bandwidth-bound, and the L3 cache (1008 MB/socket) is used for weight reuse
    across the M dimension rather than fitting the whole intermediate.
    Set INT8_FFN_CHUNK > 0 to re-enable chunking for memory-constrained machines."""
    _chunk = int(os.environ.get("INT8_FFN_CHUNK", "0"))

    def __init__(self, traced):
        super().__init__()
        self.m = traced

    def forward(self, x, *args, **kwargs):
        cs = _Int8Island._chunk
        if cs <= 0 or x.shape[1] <= cs:
            return self.m(x.float()).to(x.dtype)
        outs = []
        for i in range(0, x.shape[1], cs):
            outs.append(self.m(x[:, i:i + cs, :].float()).to(x.dtype))
        return torch.cat(outs, dim=1)



def _rebuild_ffn_fp32(ffn_module):
    """Reconstruct a plain fp32 nn.Sequential(Linear, GELU(tanh), Linear) from a
    DiTBlock.ffn whose Linears may be AutoWrappedLinear (vram management).
    AutoWrappedLinear subclasses nn.Linear and exposes the original .weight /
    .bias, so we copy them out (upcasting bf16 -> fp32 for the quant master)."""
    lin1, lin2 = ffn_module[0], ffn_module[2]
    dim, ffn_dim = lin1.in_features, lin1.out_features
    new = torch.nn.Sequential(
        torch.nn.Linear(dim, ffn_dim, bias=lin1.bias is not None),
        torch.nn.GELU(approximate="tanh"),
        torch.nn.Linear(ffn_dim, dim, bias=lin2.bias is not None),
    )
    with torch.no_grad():
        new[0].weight.copy_(lin1.weight.float())
        if lin1.bias is not None:
            new[0].bias.copy_(lin1.bias.float())
        new[2].weight.copy_(lin2.weight.float())
        if lin2.bias is not None:
            new[2].bias.copy_(lin2.bias.float())
    new.eval()
    return new


def _quantize_and_swap_ffn(dit, ffn_fp32_list, calib_iters=2):
    """Static-quantize each fp32 FFN to AMX-int8 (IPEX prepare/calibrate/convert
    -> jit trace+freeze) and swap it into the block wrapped in an _Int8Island.

    Calibration and tracing use the real streaming chunk sizes:
      - INT8_FFN_CALIB_T (env, default 15360): primary calib/trace T — matches the
        recurring 2-frame streaming chunks (2 × 120 × 64 = 15,360 tokens at 1920×1024).
    The per-tensor scale converges in 2 synthetic randn passes (FFN input after
    RMSNorm+modulate is ~N(0,1); more passes add noise but not information).
    oneDNN brgemm handles variable M after freeze, so the model traced at 15360
    runs correctly (and fast) for the first 46080-token chunk too.
    Returns the number of blocks swapped."""
    import intel_extension_for_pytorch as ipex
    from intel_extension_for_pytorch.quantization import prepare, convert
    qcfg = ipex.quantization.default_static_qconfig
    calib_T = int(os.environ.get("INT8_FFN_CALIB_T", "15360"))
    n = 0
    for blk, ffn32 in zip(dit.blocks, ffn_fp32_list):
        dim = ffn32[0].in_features
        ex = torch.randn(1, calib_T, dim)
        prepared = prepare(ffn32, qcfg, example_inputs=ex, inplace=False)
        with torch.no_grad():
            for _ in range(calib_iters):
                prepared(torch.randn(1, calib_T, dim))
        converted = convert(prepared)
        with torch.no_grad():
            traced = torch.jit.freeze(torch.jit.trace(converted, ex))
            traced(ex); traced(ex)  # trigger oneDNN int8 kernel selection
        blk.ffn = _Int8Island(traced)
        n += 1
    return n


# ----------------------------- pipeline (CPU/CUDA) -----------------------------
def init_pipeline():
    t0 = time.time()
    mm = ModelManager(torch_dtype=DTYPE, device="cpu")
    mm.load_models(["./FlashVSR-v1.1/diffusion_pytorch_model_streaming_dmd.safetensors"])
    pipe = FlashVSRTinyPipeline.from_model_manager(mm, device=DEVICE)

    pipe.denoising_model().LQ_proj_in = Causal_LQ4x_Proj(in_dim=3, out_dim=1536, layer_num=1).to(DEVICE, dtype=DTYPE)
    lq_path = "./FlashVSR-v1.1/LQ_proj_in.ckpt"
    if os.path.exists(lq_path):
        pipe.denoising_model().LQ_proj_in.load_state_dict(torch.load(lq_path, map_location="cpu"), strict=True)
    pipe.denoising_model().LQ_proj_in.to(DEVICE)

    # NOTE: pass device/dtype explicitly (build_tcdecoder defaults to cuda/bf16).
    pipe.TCDecoder = build_tcdecoder(new_channels=[512, 256, 128, 128],
                                     new_latent_channels=16 + 768,
                                     device=DEVICE, dtype=DTYPE)
    mis = pipe.TCDecoder.load_state_dict(torch.load("./FlashVSR-v1.1/TCDecoder.ckpt", map_location="cpu"), strict=False)
    print("TCDecoder load:", mis)

    pipe.to(DEVICE)

    # Optional INT8 dynamic quantization of the DiT's Linear layers (attention
    # q/k/v/o + FFN). Applied BEFORE enable_vram_management so the plain
    # nn.Linear modules are matched (vram mgmt later swaps them to
    # AutoWrappedLinear). Dynamic quant needs no calibration data. Gated by
    # RUN_INT8=1. Lossy -> validate with UVQ. Not combined with IPEX/bf16.
    _run_int8 = RUN_DEVICE == "cpu" and os.environ.get("RUN_INT8", "0") == "1"
    if _run_int8:
        import torch.ao.quantization as tq
        n_lin = sum(1 for m in pipe.dit.modules() if type(m) is torch.nn.Linear)
        pipe.dit.eval()
        pipe.dit = tq.quantize_dynamic(pipe.dit, {torch.nn.Linear}, dtype=torch.qint8)
        n_q = sum(1 for m in pipe.dit.modules()
                  if m.__class__.__name__ == "Linear" and "quantized" in m.__class__.__module__)
        print(f"[init] INT8 dynamic quant: {n_lin} nn.Linear -> {n_q} quantized Linear")

    pipe.enable_vram_management(num_persistent_param_in_dit=None)
    pipe.init_cross_kv()
    pipe.load_models_to_device(["dit", "vae"])

    # Optional OpenVINO graph compilation via the torch.compile "openvino"
    # backend (openvino.torch). Captures the DiT fx graph and lowers it to an
    # OpenVINO CPU model (same oneDNN/AMX-bf16 kernels, plus OV graph fusion /
    # memory planning). Gated by RUN_OVCOMPILE=1. suppress_errors falls back to
    # eager on unsupported subgraphs; dynamic=True avoids recompiles as the
    # streaming KV-cache shapes grow. Not combined with IPEX/INT8.
    _run_ov = RUN_DEVICE == "cpu" and os.environ.get("RUN_OVCOMPILE", "0") == "1"
    # RUN_IPEX_INT8_FFN's quantize+swap only runs inside the RUN_IPEX block
    # below, which is itself skipped whenever _run_ov is set -> combining the
    # two would silently apply OV only and never quantize the FFN. Fail loudly
    # instead of producing a misleading "no benefit from stacking" result.
    if _run_ov and os.environ.get("RUN_IPEX_INT8_FFN", "0") == "1":
        raise RuntimeError(
            "RUN_IPEX_INT8_FFN=1 has no effect when RUN_OVCOMPILE=1 (OV compile "
            "bypasses the IPEX block that performs the int8 FFN swap). Run them "
            "separately."
        )
    if _run_ov:
        try:
            import openvino.torch  # noqa: F401  registers the "openvino" backend
            # The streaming KV-cache shapes change every forward, so dynamo hits
            # the DEFAULT cache_size_limit (8) and aborts mid-compile (looks like
            # a silent crash). Raise it unconditionally whenever OV is on.
            torch._dynamo.config.cache_size_limit = 256
            # RUN_OV_DEBUG=1 surfaces the real dynamo/OV compile error (and adds
            # verbose logging) instead of silently falling back to eager -> use
            # to diagnose crashes. Default: suppress + fall back to eager.
            _ov_debug = os.environ.get("RUN_OV_DEBUG", "0") == "1"
            torch._dynamo.config.suppress_errors = not _ov_debug
            if _ov_debug:
                torch._dynamo.config.verbose = True

            # Optionally unwrap the vram-management wrappers (AutoWrappedLinear /
            # AutoWrappedModule) back into plain nn.Linear / original modules.
            # In this all-resident single-device config their onload/offload are
            # no-ops and their forward only adds a Python if-branch that forces
            # a dynamo graph break at every layer -> OV can only compile
            # fragments. Unwrapping removes those breaks so OV captures one clean
            # graph. Gated by RUN_OV_UNWRAP=1 (default on under OVCOMPILE).
            if os.environ.get("RUN_OV_UNWRAP", "1") == "1":
                def _unwrap(mod):
                    n = 0
                    for name, child in list(mod.named_children()):
                        cn = child.__class__.__name__
                        if cn == "AutoWrappedLinear":
                            lin = torch.nn.Linear(child.in_features, child.out_features,
                                                  bias=child.bias is not None)
                            lin.weight = child.weight
                            lin.bias = child.bias
                            setattr(mod, name, lin)
                            n += 1
                        elif cn == "AutoWrappedModule":
                            setattr(mod, name, child.module)
                            n += 1 + _unwrap(child.module)
                        else:
                            n += _unwrap(child)
                    return n
                nu = _unwrap(pipe.dit)
                print(f"[init] OV unwrap: replaced {nu} AutoWrapped* modules with plain modules")

            _ov_cfg = {"device": "CPU"}
            if RUN_DTYPE == "bf16":
                _ov_cfg["config"] = {"INFERENCE_PRECISION_HINT": "bf16"}
            pipe.dit.eval()
            pipe.dit = torch.compile(pipe.dit, backend="openvino",
                                     dynamic=True, options=_ov_cfg)
            print(f"[init] torch.compile(backend=openvino) applied to dit ({_ov_cfg})")
        except Exception as e:
            print(f"[init] OV compile skipped: {e}")

        # Under OV only the DiT is torch.compiled; the VAE (TCDecoder) is left
        # untouched. Without this it runs as un-optimized eager code and the VAE
        # bucket regresses ~3x (108s -> 346s on cat). Keep the VAE on IPEX/
        # AMX-bf16 so the OV-vs-baseline comparison isolates the DiT.
        if RUN_DEVICE == "cpu" and getattr(pipe, "TCDecoder", None) is not None:
            try:
                import intel_extension_for_pytorch as ipex
                _ipex_dtype = torch.bfloat16 if RUN_DTYPE == "bf16" else torch.float32
                pipe.TCDecoder.eval()
                pipe.TCDecoder = ipex.optimize(pipe.TCDecoder, dtype=_ipex_dtype, inplace=True)
                print(f"[init] IPEX optimize applied to TCDecoder/VAE under OV (dtype={_ipex_dtype})")
            except Exception as e:
                print(f"[init] IPEX VAE optimize under OV skipped: {e}")

    # Optional lossless CPU graph optimization via Intel Extension for PyTorch.
    # Fuses conv/linear/norm and prepacks weights for the CPU backend; fp32 math
    # is preserved (near bit-exact). Gated by RUN_IPEX=1. Skipped under INT8/OV.
    if RUN_DEVICE == "cpu" and os.environ.get("RUN_IPEX", "0") == "1" and not _run_int8 and not _run_ov:
        try:
            import intel_extension_for_pytorch as ipex
            _ipex_dtype = torch.bfloat16 if RUN_DTYPE == "bf16" else torch.float32
            # Optional: replace each DiTBlock.ffn with a REAL AMX-int8 kernel
            # (IPEX static quant -> oneDNN brgemm_amx_int8) while the rest of the
            # DiT stays bf16/AMX. Microbench: int8 FFN ~2x faster than bf16,
            # ~1.4% error, stable across token counts. Distinct from RUN_INT8
            # (torch dynamic quant, which forces fp32 everywhere and is slower).
            # Snapshot the fp32 FFN weights BEFORE ipex.optimize prepacks them.
            # Gated by RUN_IPEX_INT8_FFN=1. Validate quality with UVQ.
            _int8_ffn = os.environ.get("RUN_IPEX_INT8_FFN", "0") == "1"
            _ffn_snap = None
            if _int8_ffn and getattr(pipe.dit, "blocks", None) is not None:
                _ffn_snap = [_rebuild_ffn_fp32(b.ffn) for b in pipe.dit.blocks]
            if pipe.dit is not None:
                pipe.dit.eval()
                pipe.dit = ipex.optimize(pipe.dit, dtype=_ipex_dtype, inplace=True)
            if getattr(pipe, "TCDecoder", None) is not None:
                pipe.TCDecoder.eval()
                pipe.TCDecoder = ipex.optimize(pipe.TCDecoder, dtype=_ipex_dtype, inplace=True)
            print(f"[init] IPEX optimize applied (dtype={_ipex_dtype})")
            if _int8_ffn and _ffn_snap is not None:
                _n = _quantize_and_swap_ffn(pipe.dit, _ffn_snap)
                print(f"[init] IPEX AMX-int8 FFN: quantized+swapped {_n} DiTBlock.ffn "
                      f"(rest stays {_ipex_dtype})")
        except Exception as e:
            import traceback
            print(f"[init] IPEX optimize skipped: {e}")
            traceback.print_exc()

    backend_name = "openvino" if RUN_DEVICE == "cpu" and os.environ.get("RUN_OVCOMPILE", "0") == "1" else ("ipex" if RUN_DEVICE == "cpu" and os.environ.get("RUN_IPEX", "0") == "1" else "eager")
    print(f"[init] pipeline ready in {time.time() - t0:.1f}s (device={DEVICE}, dtype={RUN_DTYPE}, attn={ATTN_IMPL}, backend={backend_name}, local_th={LOCAL_TH}, local_tw={LOCAL_TW})")
    return pipe


def main():
    os.makedirs(RUN_OUTDIR, exist_ok=True)
    sparse_ratio = 2.0
    pipe = init_pipeline()

    LQ, th, tw, F, fps = prepare_input_tensor(RUN_INPUT, scale=RUN_SCALE, dtype=DTYPE, device=DEVICE)
    out_frames = F - 4

    t_inf = time.time()
    video = pipe(
        prompt="", negative_prompt="", cfg_scale=1.0, num_inference_steps=1, seed=RUN_SEED,
        LQ_video=LQ, num_frames=F, height=th, width=tw, is_full_block=False, if_buffer=True,
        topk_ratio=sparse_ratio * 768 * 1280 / (th * tw),
        kv_ratio=3.0, local_range=11, color_fix=True,
    )
    inf_s = time.time() - t_inf
    fps_out = out_frames / inf_s if inf_s > 0 else 0.0
    print(f"[{RUN_TAG}] inference done in {inf_s:.1f}s ({fps_out:.4f} fps) for {out_frames} frames")

    if os.environ.get("FLASHVSR_PROFILE", "0") == "1":
        try:
            from diffsynth.models.wan_video_dit import _PROF
            tot = _PROF["self_attn"] + _PROF["cross_attn"] + _PROF["ffn"]
            rest = inf_s - tot                      # VAE decode + patch embed/head + IO
            nf = max(out_frames, 1)
            xa = _PROF["xattn_est"] + _PROF["xattn_realize"]
            _vae = _PROF.get("vae", 0.0)
            _rest_other = rest - _vae                # patch embed/head + reshapes
            print(f"[PROFILE] self_attn={_PROF['self_attn']:.1f}s cross_attn={_PROF['cross_attn']:.1f}s "
                  f"ffn={_PROF['ffn']:.1f}s (DiT sum={tot:.1f}s) rest={rest:.1f}s "
                  f"(vae={_vae:.1f}s + patch/IO={_rest_other:.1f}s) | "
                  f"sdpa={_PROF['sdpa']:.1f}s xattn_est={_PROF['xattn_est']:.1f}s "
                  f"xattn_realize={_PROF['xattn_realize']:.1f}s rope={_PROF['rope']:.1f}s "
                  f"partition={_PROF['partition']:.1f}s "
                  f"calls={_PROF['sdpa_calls']} avg_qlen={_PROF['sdpa_qlen']//max(_PROF['sdpa_calls'],1)} "
                  f"avg_klen={_PROF['sdpa_klen']//max(_PROF['sdpa_calls'],1)}")
            print(f"[PROFILE/frame] total={inf_s/nf:.3f}s ffn={_PROF['ffn']/nf:.3f}s "
                  f"self_attn={_PROF['self_attn']/nf:.3f}s (xattn={xa/nf:.3f}s) "
                  f"cross_attn={_PROF['cross_attn']/nf:.3f}s rest={rest/nf:.3f}s "
                  f"(vae={_vae/nf:.3f}s + patch/IO={_rest_other/nf:.3f}s) "
                  f"| ffn%={100*_PROF['ffn']/max(inf_s,1e-9):.1f} rest%={100*rest/max(inf_s,1e-9):.1f} "
                  f"vae%={100*_vae/max(inf_s,1e-9):.1f} "
                  f"self_attn%={100*_PROF['self_attn']/max(inf_s,1e-9):.1f}")
        except Exception as e:
            print(f"[PROFILE] unavailable: {e}")

    video = tensor2video(video)
    # Optional: dump the RAW (pre-deseam) frames losslessly so a deseam
    # parameter sweep can be run later WITHOUT re-running inference. The frames
    # are the seamed nohalo output; deseam is a pure image-space post-pass.
    _raw_npy = os.environ.get("RUN_RAW_NPY", "").strip()
    if _raw_npy:
        os.makedirs(os.path.dirname(os.path.abspath(_raw_npy)), exist_ok=True)
        stack = np.stack([np.asarray(f, np.uint8) for f in video], 0)  # (T,H,W,C)
        np.savez_compressed(_raw_npy if _raw_npy.endswith(".npz") else _raw_npy + ".npz",
                            frames=stack, tw=tw, th=th, fps=fps)
        print(f"[raw] saved {stack.shape} pre-deseam frames -> "
              f"{_raw_npy if _raw_npy.endswith('.npz') else _raw_npy + '.npz'}")
    if os.environ.get("RUN_DESEAM", "0") == "1":
        video = deseam_video(video, tw, th)
    base = os.path.splitext(os.path.basename(RUN_INPUT))[0]
    out_path = os.path.join(RUN_OUTDIR, f"{base}_{RUN_TAG}.mp4")
    save_video(video, out_path, fps=fps, quality=6)
    print(f"[{RUN_TAG}] saved: {out_path}")
    print(f"[{RUN_TAG}] TOTAL inference={inf_s:.1f}s frames={out_frames} fps={fps_out:.4f} "
          f"res={tw}x{th} device={DEVICE} dtype={RUN_DTYPE} attn={ATTN_IMPL}")


if __name__ == "__main__":
    main()
