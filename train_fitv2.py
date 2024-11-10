import os
import torch
import argparse
import datetime
import time
import torchvision
import wandb
import logging
import math
import shutil
import accelerate
import torch
import torch.utils.checkpoint
import diffusers
import numpy as np
import torch.nn.functional as F

from functools import partial
from torch.cuda import amp
from omegaconf import OmegaConf
from accelerate import Accelerator, skip_first_batches
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed, save, FullyShardedDataParallelPlugin
from tqdm.auto import tqdm
from diffusers import AutoencoderKL, DDIMScheduler, DDPMScheduler
from safetensors import safe_open
from safetensors.torch import load_file
from copy import deepcopy
from einops import rearrange
from fit.schedulers.transport import create_transport
from fit.utils.utils import (
    instantiate_from_config,
    default,
    get_obj_from_str,
    update_ema,
    
)
from fit.utils.eval_utils import init_from_ckpt
from fit.utils.lr_scheduler import get_scheduler

logger = get_logger(__name__, log_level="INFO")

# For Omegaconf Tuple
def resolve_tuple(*args):
    return tuple(args)
OmegaConf.register_new_resolver("tuple", resolve_tuple)

def parse_args():
    parser = argparse.ArgumentParser(description="Argument.")
    parser.add_argument(
        "--project_name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="if setting, the logdir will be like: project_name",
    )
    parser.add_argument(
        "--main_project_name",
        type=str,
        default="image_generation",
    )
    parser.add_argument(
        "--workdir",
        type=str,
        default="workdir",
        help="workdir",
    )
    parser.add_argument( # if resume, you change it none. i will load from the resumedir
        "--cfgdir",
        nargs="*",
        help="paths to base configs. Loaded from left-to-right. "
        "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=0,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default='latest',
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--load_model_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be loaded from a pretrained model checkpoint."
            "Or you can set diffusion.pretrained_model_path in Config for loading!!!"
        ),
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--use_ema",
        action="store_true",
        default=True,
        help="Whether to use EMA model."
    )
    parser.add_argument(
        "--ema_decay",
        type=float,
        default=0.9999,
        help="The decay rate for ema."
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    return args


def main():
    args = parse_args()
    
    datenow = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    project_name = None
    workdir = None
    workdirnow = None
    cfgdir = None
    ckptdir = None
    logging_dir = None
    imagedir = None
    
    if args.project_name:
        project_name = args.project_name
        if os.path.exists(os.path.join(args.workdir, project_name)): #open resume
            workdir=os.path.join(args.workdir, project_name)
        else: # new a workdir
            workdir = os.path.join(args.workdir, project_name)
            # if accelerator.is_main_process:
            os.makedirs(workdir, exist_ok=True)
        workdirnow = workdir

        cfgdir = os.path.join(workdirnow, "configs")
        ckptdir = os.path.join(workdirnow, "checkpoints")
        logging_dir = os.path.join(workdirnow, "logs")
        imagedir = os.path.join(workdirnow, "images")

        # if accelerator.is_main_process:
        os.makedirs(cfgdir, exist_ok=True)
        os.makedirs(ckptdir, exist_ok=True)
        os.makedirs(logging_dir, exist_ok=True)
        os.makedirs(imagedir, exist_ok=True)
    if args.cfgdir:
        load_cfgdir = args.cfgdir
    
    # setup config
    configs_list = load_cfgdir # read config from a config dir
    configs = [OmegaConf.load(cfg) for cfg in configs_list]
    config = OmegaConf.merge(*configs)
    accelerate_cfg = config.accelerate
    diffusion_cfg = config.diffusion
    data_cfg = config.data
    grad_accu_steps = accelerate_cfg.gradient_accumulation_steps
    
    train_strtg_cfg = getattr(config, 'training_strategy', None)
    if train_strtg_cfg != None:
        warp_pos_idx = hasattr(train_strtg_cfg, 'warp_pos_idx')
        if warp_pos_idx:
            warp_pos_idx_fn = partial(warp_pos_idx_from_grid, 
                shift=train_strtg_cfg.warp_pos_idx.shift, 
                scale=train_strtg_cfg.warp_pos_idx.scale,
                max_len=train_strtg_cfg.warp_pos_idx.max_len
            )

    accelerator_project_cfg = ProjectConfiguration(project_dir=workdirnow, logging_dir=logging_dir)
    
    if getattr(accelerate_cfg, 'fsdp_config', None) != None:
        import functools
        from torch.distributed.fsdp.fully_sharded_data_parallel import (
            BackwardPrefetch, CPUOffload, ShardingStrategy, MixedPrecision, StateDictType, FullStateDictConfig, FullOptimStateDictConfig,
        )
        from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
        fsdp_cfg = accelerate_cfg.fsdp_config
        if accelerate_cfg.mixed_precision == "fp16":
            dtype = torch.float16
        elif accelerate_cfg.mixed_precision == "bf16":
            dtype = torch.bfloat16
        else:
            dtype = torch.float32   
        fsdp_plugin = FullyShardedDataParallelPlugin(
            sharding_strategy = {
                'FULL_SHARD': ShardingStrategy.FULL_SHARD,
                'SHARD_GRAD_OP': ShardingStrategy.SHARD_GRAD_OP,
                'NO_SHARD': ShardingStrategy.NO_SHARD,
                'HYBRID_SHARD': ShardingStrategy.HYBRID_SHARD,
                'HYBRID_SHARD_ZERO2': ShardingStrategy._HYBRID_SHARD_ZERO2,
            }[fsdp_cfg.sharding_strategy],
            backward_prefetch = {
                'BACKWARD_PRE': BackwardPrefetch.BACKWARD_PRE,
                'BACKWARD_POST': BackwardPrefetch.BACKWARD_POST,
            }[fsdp_cfg.backward_prefetch],
            mixed_precision_policy = MixedPrecision(
                param_dtype=dtype,
                reduce_dtype=dtype,
            ),
            auto_wrap_policy = functools.partial(
                size_based_auto_wrap_policy, min_num_params=fsdp_cfg.min_num_params
            ),
            cpu_offload = CPUOffload(offload_params=fsdp_cfg.cpu_offload),
            state_dict_type = {
                'FULL_STATE_DICT': StateDictType.FULL_STATE_DICT,
                'LOCAL_STATE_DICT': StateDictType.LOCAL_STATE_DICT,
                'SHARDED_STATE_DICT': StateDictType.SHARDED_STATE_DICT
            }[fsdp_cfg.state_dict_type],
            state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
            optim_state_dict_config = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True),
            limit_all_gathers = fsdp_cfg.limit_all_gathers, # False
            use_orig_params = fsdp_cfg.use_orig_params, # True
            sync_module_states = fsdp_cfg.sync_module_states,   #True
            forward_prefetch = fsdp_cfg.forward_prefetch,   # False
            activation_checkpointing = fsdp_cfg.activation_checkpointing,   # False
        )
    else:
        fsdp_plugin = None
    accelerator = Accelerator(
        gradient_accumulation_steps=grad_accu_steps,
        mixed_precision=accelerate_cfg.mixed_precision,
        fsdp_plugin=fsdp_plugin,
        log_with=getattr(accelerate_cfg, 'logger', 'wandb'),
        project_config=accelerator_project_cfg,
    )
    device = accelerator.device
    
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    
    if accelerator.is_local_main_process:
        File_handler = logging.FileHandler(os.path.join(logging_dir, project_name+"_"+datenow+".log"), encoding="utf-8")
        File_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
        File_handler.setLevel(logging.INFO)
        logger.logger.addHandler(File_handler)
        
        diffusers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        diffusers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()
    
    if args.seed is not None:
        set_seed(args.seed)

    if args.allow_tf32: # for A100
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if args.scale_lr:
        learning_rate = (
            accelerate_cfg.learning_rate * 
            grad_accu_steps * 
            data_cfg.params.train.loader.batch_size *   # local batch size per device
            accelerator.num_processes / accelerate_cfg.learning_rate_base_batch_size    # global batch size
        )
    else:
        learning_rate = accelerate_cfg.learning_rate

    
    model = instantiate_from_config(diffusion_cfg.network_config).to(device=device)
    # update ema
    if args.use_ema:
        # ema_dtype = torch.float32
        if hasattr(model, 'module'):
            ema_model = deepcopy(model.module).to(device=device)
        else:
            ema_model = deepcopy(model).to(device=device)
        if getattr(diffusion_cfg, 'pretrain_config', None) != None: # transfer to larger reolution
            if getattr(diffusion_cfg.pretrain_config, 'ema_ckpt', None) != None:
                init_from_ckpt(
                    ema_model, checkpoint_dir=diffusion_cfg.pretrain_config.ema_ckpt, 
                    ignore_keys=diffusion_cfg.pretrain_config.ignore_keys, verbose=True
                )
        for p in ema_model.parameters():
            p.requires_grad = False
    
    if args.use_ema:
        model = accelerator.prepare_model(model, device_placement=False)
        ema_model = accelerator.prepare_model(ema_model, device_placement=False)
    else:
        model = accelerator.prepare_model(model, device_placement=False)
        
    # In SiT, we use transport instead of diffusion
    transport = create_transport(**OmegaConf.to_container(diffusion_cfg.transport))  # default: velocity; 
    # schedule_sampler = create_named_schedule_sampler()

    # Setup Dataloader
    total_batch_size = data_cfg.params.train.loader.batch_size * accelerator.num_processes * grad_accu_steps
    global_steps = 0
    if args.resume_from_checkpoint:
        # normal read with safety check
        if args.resume_from_checkpoint != "latest":
            resume_from_path = os.path.basename(args.resume_from_checkpoint)
        else:   # Get the most recent checkpoint
            dirs = os.listdir(ckptdir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            resume_from_path = dirs[-1] if len(dirs) > 0 else None

        if resume_from_path is None:
            logger.info(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            global_steps = int(resume_from_path.split("-")[1]) # gs not calculate the gradient_accumulation_steps
            logger.info(f"Resuming from steps: {global_steps}")

    get_train_dataloader = instantiate_from_config(data_cfg)
    train_len = get_train_dataloader.train_len()
    train_dataloader = get_train_dataloader.train_dataloader(
        global_batch_size=total_batch_size, max_steps=accelerate_cfg.max_train_steps, 
        resume_step=global_steps, seed=args.seed
    )

    # Setup optimizer and lr_scheduler
    if accelerator.is_main_process:
        for name, param in model.named_parameters():
            print(name, param.requires_grad)
    if getattr(diffusion_cfg, 'pretrain_config', None) != None: # transfer to larger reolution     
        params = filter(lambda p: p.requires_grad, model.parameters())
    else:
        params = list(model.parameters())
    optimizer_cfg = default(
        accelerate_cfg.optimizer, {"target": "torch.optim.AdamW"}
    )
    optimizer = get_obj_from_str(optimizer_cfg["target"])(
        params, lr=learning_rate, **optimizer_cfg.get("params", dict())
    )
    lr_scheduler = get_scheduler(
        accelerate_cfg.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=accelerate_cfg.lr_warmup_steps,
        num_training_steps=accelerate_cfg.max_train_steps,
    )
    
    # Prepare Accelerate
    optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        optimizer, train_dataloader, lr_scheduler
    )
    
    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process and getattr(accelerate_cfg, 'logger', 'wandb') != None:
        os.environ["WANDB_DIR"] = os.path.join(os.getcwd(), workdirnow)
        accelerator.init_trackers(
            project_name=args.main_project_name, 
            config=config, 
            init_kwargs={"wandb": {"group": args.project_name}}
        )
    
    # Train!
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {train_len}")
    logger.info(f"  Instantaneous batch size per device = {data_cfg.params.train.loader.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Learning rate = {learning_rate}")
    logger.info(f"  Gradient Accumulation steps = {grad_accu_steps}")
    logger.info(f"  Total optimization steps = {accelerate_cfg.max_train_steps}")
    logger.info(f"  Current optimization steps = {global_steps}")
    logger.info(f"  Train dataloader length = {len(train_dataloader)} ")
    logger.info(f"  Training Mixed-Precision = {accelerate_cfg.mixed_precision}")

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        # normal read with safety check
        error_times=0
        while(True):
            if error_times >= 100:
                raise
            try:
                logger.info(f"Resuming from checkpoint {resume_from_path}")
                accelerator.load_state(os.path.join(ckptdir, resume_from_path))
                break
            except (RuntimeError, Exception) as err:
                error_times+=1
                if accelerator.is_local_main_process:
                    logger.warning(err)
                    logger.warning(f"Failed to resume from checkpoint {resume_from_path}")
                    shutil.rmtree(os.path.join(ckptdir, resume_from_path))
                else:
                    time.sleep(2)
    
    # save config
    OmegaConf.save(config=config, f=os.path.join(cfgdir, "config.yaml"))
    
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(0, accelerate_cfg.max_train_steps), 
        disable = not accelerator.is_main_process
    )
    progress_bar.set_description("Optim Steps")
    progress_bar.update(global_steps)
    
    if args.use_ema:
        # ema_model = ema_model.to(ema_dtype)
        ema_model.eval()
    # Training Loop
    model.train()
    train_loss = 0.0
    for step, batch in enumerate(train_dataloader, start=global_steps):
        for batch_key in batch.keys():
            if not isinstance(batch[batch_key], list):
                batch[batch_key] = batch[batch_key].to(device=device)
        x = batch['feature']        # (B, N, C)
        grid = batch['grid']        # (B, 2, N)
        mask = batch['mask']        # (B, N)
        y = batch['label']          # (B, 1)
        size = batch['size']        # (B, N_pack, 2), order: h, w. When pack is not used, N_pack=1.
        with accelerator.accumulate(model):
            # save memory for x, grid, mask
            N_batch = int(torch.max(torch.sum(size[..., 0] * size[..., 1], dim=-1)))
            x, grid, mask = x[:, : N_batch], grid[..., : N_batch], mask[:, : N_batch]
            
            # prepare other parameters
            y = y.squeeze(dim=-1).to(torch.int)
            model_kwargs = dict(y=y, grid=grid.long(), mask=mask, size=size)
            # forward model and compute loss
            with accelerator.autocast():
                loss_dict = transport.training_losses(model, x, model_kwargs)
            loss = loss_dict["loss"].mean()
            # Backpropagate
            optimizer.zero_grad()
            accelerator.backward(loss)
            if accelerator.sync_gradients and accelerate_cfg.max_grad_norm > 0.:
                all_norm = accelerator.clip_grad_norm_(
                    model.parameters(), accelerate_cfg.max_grad_norm
                )
            optimizer.step()
            lr_scheduler.step()
            # Gather the losses across all processes for logging (if we use distributed training).
            avg_loss = accelerator.gather(loss.repeat(data_cfg.params.train.loader.batch_size)).mean()
            train_loss += avg_loss.item() / grad_accu_steps
            
        # Checks if the accelerator has performed an optimization step behind the scenes; Check gradient accumulation
        if accelerator.sync_gradients: 
            if args.use_ema:
                # update_ema(ema_model, deepcopy(model).type(ema_dtype), args.ema_decay)
                update_ema(ema_model, model, args.ema_decay)
                
            progress_bar.update(1)
            global_steps += 1
            if getattr(accelerate_cfg, 'logger', 'wandb') != None:
                accelerator.log({"train_loss": train_loss}, step=global_steps)
                accelerator.log({"lr": lr_scheduler.get_last_lr()[0]}, step=global_steps)
                if accelerate_cfg.max_grad_norm != 0.0:
                    accelerator.log({"grad_norm": all_norm.item()}, step=global_steps)
            train_loss = 0.0
            if global_steps % accelerate_cfg.checkpointing_steps == 0:
                if accelerate_cfg.checkpoints_total_limit is not None:
                    checkpoints = os.listdir(ckptdir)
                    checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                    # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                    if accelerator.is_main_process and len(checkpoints) >= accelerate_cfg.checkpoints_total_limit:
                        num_to_remove = len(checkpoints) - accelerate_cfg.checkpoints_total_limit + 1
                        removing_checkpoints = checkpoints[0:num_to_remove]

                        logger.info(
                            f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                        )
                        logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                        for removing_checkpoint in removing_checkpoints:
                            removing_checkpoint = os.path.join(ckptdir, removing_checkpoint)
                            shutil.rmtree(removing_checkpoint)

                save_path = os.path.join(ckptdir, f"checkpoint-{global_steps}")
                if accelerator.is_main_process:
                    os.makedirs(save_path)
                accelerator.wait_for_everyone()
                accelerator.save_state(save_path)
                logger.info(f"Saved state to {save_path}")
                accelerator.wait_for_everyone()
                
            if global_steps in accelerate_cfg.checkpointing_steps_list:
                save_path = os.path.join(ckptdir, f"save-checkpoint-{global_steps}")
                accelerator.wait_for_everyone()
                accelerator.save_state(save_path)
                logger.info(f"Saved state to {save_path}")
                accelerator.wait_for_everyone()
            
        logs = {"step_loss": loss.detach().item(), 
                "lr": lr_scheduler.get_last_lr()[0]}
        progress_bar.set_postfix(**logs)
        if global_steps % accelerate_cfg.logging_steps == 0:
            if accelerator.is_main_process:
                logger.info("step="+str(global_steps)+" / total_step="+str(accelerate_cfg.max_train_steps)+", step_loss="+str(logs["step_loss"])+', lr='+str(logs["lr"]))

        if global_steps >= accelerate_cfg.max_train_steps:
            logger.info(f'global step ({global_steps}) >= max_train_steps ({accelerate_cfg.max_train_steps}), stop training!!!')
            break
    accelerator.wait_for_everyone()
    accelerator.end_training()

if __name__ == "__main__":
    main()