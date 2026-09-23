numactl -N 1 --membind=1 env \
    OMP_NUM_THREADS=32 \
    /home/xiuchuan/workspace/dev/venv/dev/bin/python export_flashvsr_openvino_full.py \
    --output-dir ./openvino_ir/flashvsr_tiny_v11_256x256_dense_compat_stateful_fp32 \
    --dtype fp32 \
    --compatibility-mode \
    --stateful \
    --height 256 --width 256 \
    --lq-frames 153 --latent-frames-first 6 --latent-frames-next 2 \
    --decode-latent-frames 34 --decode-cond-frames 133 \
    --min-free-gb 20
