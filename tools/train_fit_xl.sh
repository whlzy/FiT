JOB_NAME = "train_fit_xl"
NNODES = 1
GPUS_PER_NODE = 8
MASTER_ADDR = "localhost"
export MASTER_PORT=60563

CMD=" \
    projects/FiT/FiT/train_fit.py \
    --project_name ${JOB_NAME} \
    --main_project_name image_generation \
    --seed 0 \
    --scale_lr \
    --allow_tf32 \
    --resume_from_checkpoint latest \
    --workdir workdir/fit_xl \
    --cfgdir "projects/FiT/FiT/configs/fit/config_fit_xl.yaml" \
    --use_ema
    "
TORCHLAUNCHER="torchrun \
    --nnodes $NNODES \
    --nproc_per_node $GPUS_PER_NODE \
    --rdzv_id $RANDOM \
    --rdzv_backend c10d \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    "
bash -c "$TORCHLAUNCHER $CMD"