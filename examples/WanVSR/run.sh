#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
out_w="${RUN_OUT_W:-1920}"
out_h="${RUN_OUT_H:-1024}"
# Bind CPU execution to node 1, but allow memory allocation to fall back to
# other NUMA nodes: FP32 dense attention at 1920x1024 may exceed one node's RAM.
# Use 29 input frames to exercise the first and one recurrent chunk; set
# RUN_MAX_INPUT_FRAMES=0 to run the complete video.
numactl -C 24-31 env \
RUN_INPUT="${RUN_INPUT:-./inputs/example0.mp4}" \
RUN_MAX_INPUT_FRAMES="${RUN_MAX_INPUT_FRAMES:-29}" \
RUN_DEVICE=cpu \
RUN_DTYPE=fp32 \
RUN_IPEX=0 \
RUN_OUTDIR=./results_run RUN_TAG="${RUN_TAG:-dense_stateful_ir_${out_w}x${out_h}}" \
RUN_SCALE=2.25 \
RUN_SEED=0 \
RUN_OUT_W="$out_w" \
RUN_OUT_H="$out_h" \
RUN_STATEFUL_DIT_IR=./openvino_ir/flashvsr_tiny_v11_256x256_dense_compat_stateful_fp32 \
RUN_OVCOMPILE=0 \
RUN_INT8=0 \
RUN_IPEX_INT8_FFN=0 \
FLASHVSR_ATTN_IMPL=dense \
/home/xiuchuan/workspace/dev/venv/dev/bin/python -u _run_one.py
