numactl -N 1 --membind=1 env \
    FLASHVSR_ATTN_IMPL=dense \
    OMP_NUM_THREADS=32 \
    python export_flashvsr_openvino_full.py \
    --output-dir ./openvino_ir/flashvsr_tiny_v11_1920x1024_dense_stateful_fp32 \
    --dtype fp32 \
    --height 1024 --width 1920 \
    --lq-frames 153 --latent-frames-first 6 --latent-frames-next 2 \
    --decode-latent-frames 34 --decode-cond-frames 133 \
    --min-free-gb 20
