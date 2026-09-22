numactl -N 1 --membind=1 env \
RUN_INPUT=./inputs/480p-5s-cat.mp4 \
RUN_DEVICE=cpu \
RUN_DTYPE=bf16 \
RUN_IPEX=1 \
RUN_OUTDIR=./results_run RUN_TAG=dense_cat_bf16_ov_1920x1024 \
RUN_SCALE=2.25 \
RUN_SEED=0 \
RUN_OUT_W=1920 \
RUN_OUT_H=1024 \
RUN_OVCOMPILE=1 \
RUN_INT8=0 \
RUN_IPEX_INT8_FFN=0 \
FLASHVSR_ATTN_IMPL=dense \
OMP_NUM_THREADS=16 \
python _run_one.py
