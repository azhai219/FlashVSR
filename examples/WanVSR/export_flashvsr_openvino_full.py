#!/usr/bin/env python3
"""
Offline OpenVINO IR exporter for FlashVSR Tiny v1.1 major compute modules.

This is intended to evaluate a more complete OpenVINO path than torch.compile
backend lowering of only DiT. It exports three groups of modules:

1) LQ projection network (Causal_LQ4x_Proj)
2) DiT denoiser forward variants (chunk-0 and chunk-N style temporal shapes)
3) TCDecoder decode path (parallel decode)

Notes
-----
- The exported DiT models are NON-streaming forward variants (no per-layer
  Python-side KV cache handoff), because the current streaming helper path uses
  Python list state that is not directly representable as one static graph.
- TCDecoder export uses parallel=True decode to avoid stateful sequential queue
  execution that is difficult to capture in a single static graph.
- This script is for producing IR artifacts to benchmark feasibility and measure
  how much of end-to-end runtime can benefit from pure OV runtime execution.
"""

import argparse
import os
import sys
import shutil
from pathlib import Path

import openvino as ov
import torch

ROOT_DIR = Path(__file__).resolve().parent
FLASHVSR_ROOT = ROOT_DIR.parent.parent
os.chdir(ROOT_DIR)
sys.path.insert(0, str(FLASHVSR_ROOT))
sys.path.insert(1, str(ROOT_DIR))

from diffsynth.models.model_manager import ModelManager
from diffsynth.pipelines.flashvsr_tiny import FlashVSRTinyPipeline
from diffsynth.pipelines.flashvsr_tiny import model_fn_wan_video
from utils.utils import Causal_LQ4x_Proj
from utils.TCDecoder import build_tcdecoder


class LQProjWrapper(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        outs = self.module(video)
        return torch.stack(outs, dim=0)


class DitForwardWrapper(torch.nn.Module):
    def __init__(
        self,
        dit: torch.nn.Module,
        context: torch.Tensor,
        t_mod: torch.Tensor,
        t: torch.Tensor,
        cur_process_idx: int,
        layer_num: int = 1,
    ):
        super().__init__()
        self.dit = dit
        self.register_buffer("context", context, persistent=False)
        self.register_buffer("t_mod", t_mod, persistent=False)
        self.register_buffer("t", t, persistent=False)
        self.cur_process_idx = int(cur_process_idx)
        self.layer_num = layer_num

    def forward(self, x: torch.Tensor, timestep: torch.Tensor, lq_latents_stacked: torch.Tensor) -> torch.Tensor:
        # Unstack to list; model checks block_id < len(LQ_latents) so no None padding needed.
        lq_list = [lq_latents_stacked[i] for i in range(self.layer_num)]
        
        pre_cache_k = [None] * len(self.dit.blocks)
        pre_cache_v = [None] * len(self.dit.blocks)
        denoised, _, _ = model_fn_wan_video(
            self.dit,
            x=x,
            timestep=timestep,
            context=self.context,
            tea_cache=None,
            use_unified_sequence_parallel=False,
            LQ_latents=lq_list,
            topk_ratio=2.0,
            kv_ratio=3.0,
            is_full_block=False,
            is_stream=True,
            pre_cache_k=pre_cache_k,
            pre_cache_v=pre_cache_v,
            cur_process_idx=self.cur_process_idx,
            t_mod=self.t_mod,
            t=self.t,
            local_range=9,
        )
        return denoised


class TCDecoderWrapper(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module

    def forward(self, latents_ntchw: torch.Tensor, cond_bcfhw: torch.Tensor) -> torch.Tensor:
        self.module.clean_mem()
        return self.module.decode_video(
            latents_ntchw,
            parallel=True,
            show_progress_bar=False,
            cond=cond_bcfhw,
        )


def _can_run_dit_shape(module: torch.nn.Module, x: torch.Tensor, t: torch.Tensor, lq_latents: torch.Tensor = None) -> tuple[bool, str]:
    try:
        with torch.no_grad():
            args = (x, t) if lq_latents is None else (x, t, lq_latents)
            _ = module(*args)
        return True, ""
    except Exception as e:
        return False, str(e)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export FlashVSR Tiny modules to OpenVINO IR")
    p.add_argument("--output-dir", default="./openvino_ir/flashvsr_tiny_v11", help="Where to save IR files")
    p.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
    p.add_argument("--height", type=int, default=1024, help="Output frame height (multiple of 128 recommended)")
    p.add_argument("--width", type=int, default=1920, help="Output frame width (multiple of 128 recommended)")
    p.add_argument("--lq-frames", type=int, default=153, help="LQ input frames for LQ projector export sample")
    p.add_argument("--latent-frames-first", type=int, default=6, help="Latent frames for first denoise chunk")
    p.add_argument("--latent-frames-next", type=int, default=2, help="Latent frames for next denoise chunks")
    p.add_argument("--decode-latent-frames", type=int, default=34, help="Latent frames for decoder export sample")
    p.add_argument("--decode-cond-frames", type=int, default=149, help="Conditioning frames for decoder export sample")
    p.add_argument("--device", default="CPU", help="OpenVINO target device string")
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip exporting stages whose .xml and .bin already exist",
    )
    p.add_argument(
        "--min-free-gb",
        type=float,
        default=5.0,
        help="Fail fast if output filesystem has less than this much free space",
    )
    return p.parse_args()


def build_pipeline(dtype: torch.dtype) -> FlashVSRTinyPipeline:
    mm = ModelManager(torch_dtype=dtype, device="cpu")
    mm.load_models(["./FlashVSR-v1.1/diffusion_pytorch_model_streaming_dmd.safetensors"])
    pipe = FlashVSRTinyPipeline.from_model_manager(mm, device="cpu")

    pipe.denoising_model().LQ_proj_in = Causal_LQ4x_Proj(in_dim=3, out_dim=1536, layer_num=1).to("cpu", dtype=dtype)
    pipe.denoising_model().LQ_proj_in.load_state_dict(
        torch.load("./FlashVSR-v1.1/LQ_proj_in.ckpt", map_location="cpu"), strict=True
    )

    pipe.TCDecoder = build_tcdecoder(
        new_channels=[512, 256, 128, 128],
        new_latent_channels=16 + 768,
        device="cpu",
        dtype=dtype,
    )
    pipe.TCDecoder.load_state_dict(torch.load("./FlashVSR-v1.1/TCDecoder.ckpt", map_location="cpu"), strict=False)

    pipe.to("cpu")
    pipe.load_models_to_device(["dit", "vae"])
    pipe.init_cross_kv()
    pipe.eval()
    pipe.dit.eval()
    pipe.TCDecoder.eval()
    pipe.denoising_model().LQ_proj_in.eval()
    return pipe


def export_model(module: torch.nn.Module, example_inputs, xml_path: Path) -> None:
    ov_model = ov.convert_model(module, example_input=example_inputs)
    ov.save_model(ov_model, str(xml_path))


def _ir_exists(xml_path: Path) -> bool:
    return xml_path.exists() and xml_path.with_suffix(".bin").exists()


def main() -> None:
    args = parse_args()
    if args.latent_frames_first != 6:
        raise ValueError(
            "latent-frames-first must be 6 for FlashVSR streaming DiT export "
            "(model asserts first chunk f==6)."
        )
    if args.latent_frames_next != 2:
        raise ValueError(
            "latent-frames-next must be 2 for FlashVSR streaming DiT export "
            "(subsequent chunk shape expected by model_fn path)."
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(out_dir)
    free_gb = usage.free / (1024 ** 3)
    if free_gb < args.min_free_gb:
        raise RuntimeError(
            f"Not enough free disk space for IR export: {free_gb:.2f} GiB free, "
            f"requires at least {args.min_free_gb:.2f} GiB. "
            "Clear space or choose an output directory on a larger filesystem."
        )

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    h8 = args.height // 8
    w8 = args.width // 8
    if h8 <= 0 or w8 <= 0:
        raise ValueError(f"Invalid output size {args.height}x{args.width}")

    print(f"[export] building pipeline (dtype={args.dtype}, free={free_gb:.2f} GiB)")
    pipe = build_pipeline(dtype)

    # 1) LQ projector - export to process chunks of LQ frames
    lq_proj = LQProjWrapper(pipe.denoising_model().LQ_proj_in)
    # Export with example matching a typical chunk size (e.g., 6 frames for first chunk)
    lq_in_chunk = torch.randn(1, 3, args.latent_frames_first, args.height, args.width, dtype=dtype)
    lq_xml = out_dir / "lq_proj.xml"
    if args.skip_existing and _ir_exists(lq_xml):
        print(f"[export] lq_proj exists, skipping: {lq_xml}")
    else:
        print(f"[export] lq_proj -> {lq_xml}")
        print(f"[export] LQ proj will process chunks dynamically at inference time")
        export_model(lq_proj, lq_in_chunk, lq_xml)

    # 2) DiT denoiser chunk variants
    # context shape comes from prompt tensor loaded by init_cross_kv
    context = pipe.prompt_emb_posi["context"].detach().to(dtype)

    dit0 = DitForwardWrapper(
        pipe.dit,
        context=context,
        t_mod=pipe.t_mod.detach().to(dtype),
        t=pipe.t.detach().to(dtype),
        cur_process_idx=0,
        layer_num=1,  # Only first DiT layer uses LQ conditioning
    )
    x0 = torch.randn(1, 16, args.latent_frames_first, h8, w8, dtype=dtype)
    t0 = torch.tensor([1000.0], dtype=dtype)
    # Generate sample LQ latents for the first chunk by emulating the pipeline's
    # stream_forward loop: 7 calls of 4-frame slices, first returns None (priming),
    # then 6 non-None outputs are merged along the sequence dimension.
    # This matches the actual inference shape: 6 * (H/16 * W/16) = 6 * 7680 = 46080.
    # Using forward() with 6 frames is WRONG: it only produces 1 output chunk (7680 tokens).
    print(f"[export] generating sample LQ latents for first chunk via stream_forward (7 calls)...")
    lq_proj_in = pipe.denoising_model().LQ_proj_in
    with torch.no_grad():
        lq_proj_in.clear_cache()
        inner_loop_num_0 = 7  # matches pipeline: 7 stream_forward calls for first chunk
        lq_chunks_0 = []
        sample_lq_video = torch.randn(1, 3, inner_loop_num_0 * 4, args.height, args.width, dtype=dtype)
        for inner_idx in range(inner_loop_num_0):
            clip = sample_lq_video[:, :, inner_idx * 4:(inner_idx + 1) * 4, :, :]
            cur = lq_proj_in.stream_forward(clip)
            if cur is not None:
                lq_chunks_0.append(cur)
    # Merge: each chunk is a list of layer_num tensors; cat along seq dim
    lq_latents_list_0 = []
    for layer_idx in range(len(lq_chunks_0[0])):
        lq_latents_list_0.append(torch.cat([c[layer_idx] for c in lq_chunks_0], dim=1))
    lq_latents_stacked_0 = torch.stack(lq_latents_list_0, dim=0).detach().to(dtype)
    print(f"[export] LQ latents first chunk: {len(lq_latents_list_0)} layers, shape {tuple(lq_latents_stacked_0.shape)}")
    
    dit0_xml = out_dir / "dit_forward_first.xml"
    if args.skip_existing and _ir_exists(dit0_xml):
        print(f"[export] dit_forward_first exists, skipping: {dit0_xml}")
    else:
        print(f"[export] dit_forward_first -> {dit0_xml}")
        export_model(dit0, (x0, t0, lq_latents_stacked_0), dit0_xml)

    ditn = DitForwardWrapper(
        pipe.dit,
        context=context,
        t_mod=pipe.t_mod.detach().to(dtype),
        t=pipe.t.detach().to(dtype),
        cur_process_idx=1,
        layer_num=1,
    )
    # Model enforces f==6 when pre_cache is None (stream start). Both chunks use 6 frames for tracing.
    xn = torch.randn(1, 16, 6, h8, w8, dtype=dtype)
    tn = torch.tensor([1000.0], dtype=dtype)
    # Regenerate LQ latents from fresh state for 6-frame next-chunk trace (same 7-call pattern as first).
    print(f"[export] generating sample LQ latents for next chunk via stream_forward (7 calls, fresh state)...")
    with torch.no_grad():
        lq_proj_in.clear_cache()
        lq_chunks_n = []
        sample_lq_video_n = torch.randn(1, 3, 7 * 4, args.height, args.width, dtype=dtype)
        for inner_idx in range(7):
            clip = sample_lq_video_n[:, :, inner_idx * 4:(inner_idx + 1) * 4, :, :]
            cur = lq_proj_in.stream_forward(clip)
            if cur is not None:
                lq_chunks_n.append(cur)
    lq_latents_list_n = []
    for layer_idx in range(len(lq_chunks_n[0])):
        lq_latents_list_n.append(torch.cat([c[layer_idx] for c in lq_chunks_n], dim=1))
    lq_latents_stacked_n = torch.stack(lq_latents_list_n, dim=0).detach().to(dtype)
    print(f"[export] LQ latents next chunk: {len(lq_latents_list_n)} layers, shape {tuple(lq_latents_stacked_n.shape)}")

    ditn_xml = out_dir / "dit_forward_next.xml"
    if args.skip_existing and _ir_exists(ditn_xml):
        print(f"[export] dit_forward_next exists, skipping: {ditn_xml}")
    else:
        print(f"[export] dit_forward_next -> {ditn_xml}")
        export_model(ditn, (xn, tn, lq_latents_stacked_n), ditn_xml)

    # 3) TCDecoder decode path
    dec = TCDecoderWrapper(pipe.TCDecoder)
    lat_ntchw = torch.randn(1, args.decode_latent_frames, 16, h8, w8, dtype=dtype)
    # TCDecoder concatenates pixel-shuffled cond (time reduced by ~4x) with
    # latent along channel dim, so their time lengths must match.
    cond_frames = args.decode_cond_frames
    expected_t = args.decode_latent_frames
    cond_t = (cond_frames + 3) // 4
    if cond_t != expected_t:
        auto_cond_frames = max(1, expected_t * 4 - 3)
        print(
            f"[export] tcdecoder cond-frame align: requested {cond_frames} -> "
            f"{auto_cond_frames} so ceil(F/4) matches latent T={expected_t}"
        )
        cond_frames = auto_cond_frames
    cond = torch.randn(1, 3, cond_frames, args.height, args.width, dtype=dtype)
    dec_xml = out_dir / "tcdecoder_decode.xml"
    ok_dec, err_dec = _can_run_dit_shape(dec, lat_ntchw, cond)
    if not ok_dec and "Sizes of tensors must match" in err_dec:
        auto_cond_frames = max(1, expected_t * 4 - 3)
        print(
            f"[export] tcdecoder dry-run fallback due to shape mismatch; "
            f"retry cond_frames={auto_cond_frames}"
        )
        cond = torch.randn(1, 3, auto_cond_frames, args.height, args.width, dtype=dtype)
    elif not ok_dec:
        raise RuntimeError(f"tcdecoder dry-run failed: {err_dec}")

    if args.skip_existing and _ir_exists(dec_xml):
        print(f"[export] tcdecoder_decode exists, skipping: {dec_xml}")
    else:
        print(f"[export] tcdecoder_decode -> {dec_xml}")
        export_model(dec, (lat_ntchw, cond), dec_xml)

    # Save minimal manifest with shape assumptions used for export.
    manifest = out_dir / "manifest.txt"
    manifest.write_text(
        "\n".join(
            [
                "FlashVSR Tiny v1.1 OpenVINO offline export with per-chunk LQ conditioning",
                f"dtype={args.dtype}",
                f"height={args.height}",
                f"width={args.width}",
                f"lq_frames={args.lq_frames}",
                f"lq_conditioning_layers=1",
                f"latent_frames_first={args.latent_frames_first}",
                f"latent_frames_next={args.latent_frames_next}",
                f"decode_latent_frames={args.decode_latent_frames}",
                f"decode_cond_frames={cond.shape[2]}",
                f"device={args.device}",
                "NOTE: lq_proj IR processes LQ frame chunks dynamically at inference time",
                "NOTE: DiT IRs (first + next) accept (latents, timestep, lq_latents_chunk) for full SR conditioning",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[export] wrote manifest: {manifest}")
    print("[export] done")


if __name__ == "__main__":
    main()
