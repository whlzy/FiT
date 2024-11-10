JOB_NAME = "train_fitv2_hr_3B"
NNODES = 1
GPUS_PER_NODE = 8
MASTER_ADDR = "localhost"
export MASTER_PORT=60563

CMD=" \
    projects/FiT/FiT/train_fitv2.py \
    --project_name ${JOB_NAME} \
    --main_project_name image_generation \
    --seed 0 \
    --scale_lr \
    --allow_tf32 \
    --resume_from_checkpoint latest \
    --workdir workdir/fitv2_hr_3B \
    --cfgdir "projects/FiT/FiT/configs/fitv2/config_fitv2_hr_3B.yaml" \
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