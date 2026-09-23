"""Reproduce FlashVSR's block-attention mask with real streaming tensor SHAPES.

The default 1920x1024 demo uses synthetic Q/K values, not features from the
model; for a real mask, call make_attention_block() with actual q_w/k_w tensors
captured after RoPE and WindowPartition3D.partition in SelfAttention.forward.
The current compatibility-mode OpenVINO IR does NOT use this mask.
"""

import argparse
import math

import torch
from einops import rearrange


@torch.no_grad()
def build_local_block_mask_shifted_vec_normal_slide(
	block_h: int, block_w: int, win_h: int = 6, win_w: int = 6,
	include_self: bool = True, device=None,
) -> torch.Tensor:
	"""Copy of the spatial candidate mask used by SelfAttention."""
	device = device or torch.device("cpu")
	H, W = block_h, block_w
	r = torch.arange(H, device=device)
	c = torch.arange(W, device=device)
	YY, XX = torch.meshgrid(r, c, indexing="ij")
	r_all = YY.reshape(-1)
	c_all = XX.reshape(-1)
	r_half = win_h // 2
	c_half = win_w // 2
	start_r = r_all - r_half
	end_r = start_r + win_h - 1
	start_c = c_all - c_half
	end_c = start_c + win_w - 1
	in_row = (r_all[None, :] >= start_r[:, None]) & (r_all[None, :] <= end_r[:, None])
	in_col = (c_all[None, :] >= start_c[:, None]) & (c_all[None, :] <= end_c[:, None])
	mask = in_row & in_col
	if not include_self:
		mask.fill_diagonal_(False)
	return mask


@torch.no_grad()
def generate_draft_block_mask(
	batch_size: int, nheads: int, seqlen: int,
	q_w: torch.Tensor, k_w: torch.Tensor, topk: int = 10,
	local_attn_mask: torch.Tensor | None = None,
) -> torch.Tensor:
	"""Copy of the Q/K window pooling, locality, softmax and top-k selection."""
	assert batch_size == 1, "Only batch_size=1 supported for now"
	assert local_attn_mask is not None, "local_attn_mask must be provided"
	avgpool_q = torch.mean(q_w, dim=1)
	avgpool_k = torch.mean(k_w, dim=1)
	avgpool_q = rearrange(avgpool_q, "s (h d) -> s h d", h=nheads)
	avgpool_k = rearrange(avgpool_k, "s (h d) -> s h d", h=nheads)
	q_heads = avgpool_q.permute(1, 0, 2)
	k_heads = avgpool_k.permute(1, 0, 2)
	D = avgpool_q.shape[-1]
	scores = torch.einsum("hld,hmd->hlm", q_heads, k_heads) / math.sqrt(D)

	repeat_head = scores.shape[0]
	repeat_len = scores.shape[1] // local_attn_mask.shape[0]
	repeat_num = scores.shape[2] // local_attn_mask.shape[1]
	local_attn_mask = local_attn_mask.unsqueeze(1).unsqueeze(0).repeat(repeat_len, 1, repeat_num, 1)
	local_attn_mask = rearrange(local_attn_mask, "x a y b -> (x a) (y b)")
	local_attn_mask = local_attn_mask.unsqueeze(0).repeat(repeat_head, 1, 1)
	local_attn_mask = local_attn_mask.to(torch.float32)
	local_attn_mask = local_attn_mask.masked_fill(local_attn_mask == False, -float("inf"))
	local_attn_mask = local_attn_mask.masked_fill(local_attn_mask == True, 0)
	scores = scores + local_attn_mask

	attn_map = torch.softmax(scores, dim=-1)
	attn_map = rearrange(attn_map, "h (it s1) s2 -> (h it) s1 s2", it=seqlen)
	loop_num, s1, s2 = attn_map.shape
	flat = attn_map.reshape(loop_num, -1)
	apply_topk = min(flat.shape[1] - 1, topk)
	thresholds = torch.topk(flat, k=apply_topk + 1, dim=1, largest=True).values[:, -1]
	thresholds = thresholds.unsqueeze(1)
	mask_new = (flat > thresholds).reshape(loop_num, s1, s2)
	mask_new = rearrange(mask_new, "(h it) s1 s2 -> h (it s1) s2", it=seqlen)
	return mask_new.unsqueeze(0).repeat(batch_size, 1, 1, 1)


@torch.no_grad()
def make_attention_block(
	q_w: torch.Tensor, k_w: torch.Tensor, *, height: int, width: int,
	num_heads: int = 12, local_range: int = 11, topk_ratio: float | None = None,
) -> torch.Tensor:
	"""Accept actual windowed Q/K [blocks,128,1536], return [1,heads,Qblocks,Kblocks]."""
	if height % 128 or width % 128 or height <= 0 or width <= 0:
		raise ValueError("Output height and width must be positive multiples of 128")
	block_h, block_w = height // 128, width // 128  # patchify /8, window /8
	spatial_blocks = block_h * block_w
	if q_w.ndim != 3 or k_w.ndim != 3 or q_w.shape[1:] != (128, 1536) or k_w.shape[1:] != (128, 1536):
		raise ValueError("Expected Q/K window tensors of shape [block_count, 128, 1536]")
	if q_w.shape[0] % spatial_blocks or k_w.shape[0] % spatial_blocks:
		raise ValueError("Q/K window counts must be multiples of the spatial block count")
	if q_w.device != k_w.device or q_w.dtype != k_w.dtype:
		raise ValueError("Q and K must have the same device and dtype")
	if topk_ratio is None:
		# _run_one.py passes sparse_ratio * 768 * 1280 / (height * width), sparse_ratio=2.
		topk_ratio = 2.0 * 768 * 1280 / (height * width)
	window_size = 2 * (height // 16) * (width // 16) // 128
	topk = int(window_size * window_size * topk_ratio) - 1
	local = build_local_block_mask_shifted_vec_normal_slide(
		block_h, block_w, local_range, local_range, include_self=True, device=q_w.device,
	)
	return generate_draft_block_mask(
		1, num_heads, q_w.shape[0] // spatial_blocks, q_w, k_w, topk=topk,
		local_attn_mask=local,
	)


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--height", type=int, default=1024, help="Output frame height")
	parser.add_argument("--width", type=int, default=1920, help="Output frame width")
	parser.add_argument("--chunk", choices=("first", "next"), default="first")
	parser.add_argument("--local-range", type=int, default=11)
	parser.add_argument("--seed", type=int, default=0)
	parser.add_argument("--save", help="Optional path to save the bool mask with torch.save")
	args = parser.parse_args()
	if args.height <= 0 or args.width <= 0 or args.height % 128 or args.width % 128:
		parser.error("height/width must be positive multiples of 128")
	torch.manual_seed(args.seed)
	n = (args.height // 128) * (args.width // 128)
	q_frames = 6 if args.chunk == "first" else 2
	q_blocks = (q_frames // 2) * n
	# First: K is the same 6-frame chunk. Next: 3 cached 2-frame groups + 1 new.
	new_k = torch.randn(q_blocks, 128, 1536)
	k_w = new_k if args.chunk == "first" else torch.cat((torch.randn(3 * n, 128, 1536), new_k), dim=0)
	q_w = torch.randn(q_blocks, 128, 1536)
	mask = make_attention_block(
		q_w, k_w, height=args.height, width=args.width, local_range=args.local_range,
	)
	print(f"Synthetic values (real streaming SHAPES), {args.chunk} chunk at {args.width}x{args.height}")
	print(f"Q windows: {tuple(q_w.shape)}, K windows: {tuple(k_w.shape)}")
	print(f"Local mask: {(n, n)}, topk: {int(n * n * 2.0 * 768 * 1280 / (args.height * args.width)) - 1}")
	print(f"Attention block (bool): {tuple(mask.shape)}, selected: {int(mask.sum())}/{mask.numel()}")
	if args.save:
		torch.save(mask, args.save)
		print(f"Saved: {args.save}")


if __name__ == "__main__":
	main()
