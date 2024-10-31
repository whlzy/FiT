# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Samples a large number of images from a pre-trained DiT model using DDP.
Subsequently saves a .npz file that can be used to compute FID and other
evaluation metrics via the ADM repo: https://github.com/openai/guided-diffusion/tree/main/evaluations

For a simple single-GPU/CPU sampling script, see sample.py.
"""
import os
import math
import torch
import argparse
import numpy as np
import torch.distributed as dist
import re

from omegaconf import OmegaConf
from tqdm import tqdm
from PIL import Image
from diffusers.models import AutoencoderKL
from fit.scheduler.improved_diffusion import create_diffusion
rom fit.utils.eval_utils import create_npz_from_sample_folder, init_from_ckpt
from fit.utils.utils import instantiate_from_config
f

def ntk_scaled_init(head_dim, base=10000, alpha=8):
    #The method is just these two lines
    dim_h = head_dim // 2 # for x and y
    base = base * alpha ** (dim_h / (dim_h-2)) #Base change formula
    return base

def main(args):
    """
    Run sampling.
    """
    torch.backends.cuda.matmul.allow_tf32 = args.tf32  # True: fast but may lead to some small numerical differences
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    # Setup DDP:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    if args.mixed == "fp32":
        weight_dtype = torch.float32
    elif args.mixed == "bf16":
        weight_dtype = torch.bfloat16
    
    if args.cfgdir == "":
        args.cfgdir = os.path.join(args.ckpt.split("/")[0], args.ckpt.split("/")[1], "configs/config.yaml")
    print("config dir: ",args.cfgdir)
    config = OmegaConf.load(args.cfgdir)
    config_diffusion = config.diffusion
    
    
    H, W = args.image_height // 8, args.image_width // 8
    patch_size = config_diffusion.network_config.params.patch_size
    n_patch_h, n_patch_w = H // patch_size, W // patch_size
    
    if args.interpolation != 'no':
        # sqrt(256) or sqrt(512), we set max PE length for inference, in fact some PE has been seen in the training stage.
        ori_max_pe_len = int(config_diffusion.network_config.params.context_size ** 0.5) 
        if args.interpolation == 'linear':    # 这个就是positional index interpolation，原来叫normal，现在叫linear
            config_diffusion.network_config.params['custom_freqs'] = 'linear'
        elif args.interpolation == 'dynntk':    # 这个就是ntk-aware
            config_diffusion.network_config.params['custom_freqs'] = 'ntk-aware'
        elif args.interpolation == 'partntk':   # 这个就是ntk-by-parts
            config_diffusion.network_config.params['custom_freqs'] = 'ntk-by-parts'
        elif args.interpolation == 'yarn':
            config_diffusion.network_config.params['custom_freqs'] = 'yarn'
        else:
            raise NotImplementedError
        config_diffusion.network_config.params['max_pe_len_h'] = n_patch_h
        config_diffusion.network_config.params['max_pe_len_w'] = n_patch_w
        config_diffusion.network_config.params['decouple'] = args.decouple
        config_diffusion.network_config.params['ori_max_pe_len'] = int(ori_max_pe_len)
        
    else:   # there is no need to do interpolation!
        pass
    
    model = instantiate_from_config(config_diffusion.network_config).to(device, dtype=weight_dtype)
    init_from_ckpt(model, checkpoint_dir=args.ckpt, ignore_keys=None, verbose=True)
    model.eval() # important
    
    # prepare first stage model
    if args.vae_decoder == 'sd-ft-mse':
        vae_model = 'stabilityai/sd-vae-ft-mse'
    elif args.vae_decoder == 'sd-ft-ema':
        vae_model = 'stabilityai/sd-vae-ft-ema'
    vae = AutoencoderKL.from_pretrained(vae_model, local_files_only=True).to(device, dtype=weight_dtype)
    vae.eval() # important
    
    
    config_diffusion.improved_diffusion.timestep_respacing = str(args.num_sampling_steps)
    diffusion = create_diffusion(**OmegaConf.to_container(config_diffusion.improved_diffusion))


    workdir_name = 'official_fit'
    folder_name = f'{args.ckpt.split("/")[-1].split(".")[0]}'

    sample_folder_dir = f"{args.sample_dir}/{workdir_name}/{folder_name}"
    if rank == 0:
        os.makedirs(os.path.join(args.sample_dir, workdir_name), exist_ok=True)
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    dist.barrier()
    args.cfg_scale = float(args.cfg_scale)
    assert args.cfg_scale >= 1.0, "In almost all cases, cfg_scale be >= 1.0"
    using_cfg = args.cfg_scale > 1.0

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = args.per_proc_batch_size
    global_batch_size = n * dist.get_world_size()
    # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples:
    total_samples = int(math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size)
    if rank == 0:
        print(f"Total number of images that will be sampled: {total_samples}")
    assert total_samples % dist.get_world_size() == 0, "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // n)
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar
    total = 0
    index = 0
    all_images = []
    while len(all_images) * n < int(args.num_fid_samples):
        print(device, "device: ", index, flush=True)
        index+=1
        # Sample inputs:
        z = torch.randn(
            (n, (patch_size**2)*model.in_channels, n_patch_h*n_patch_w)
        ).to(device=device, dtype=weight_dtype)
        y = torch.randint(0, args.num_classes, (n,), device=device)
        
        # prepare for x
        grid_h = torch.arange(n_patch_h, dtype=torch.float32)
        grid_w = torch.arange(n_patch_w, dtype=torch.float32)
        grid = torch.meshgrid(grid_w, grid_h, indexing='xy')
        grid = torch.cat(
            [grid[0].reshape(1,-1), grid[1].reshape(1,-1)], dim=0
        ).repeat(n,1,1).to(device=device, dtype=weight_dtype)
        mask = torch.ones(n, n_patch_h*n_patch_w).to(device=device, dtype=weight_dtype)
        size = torch.tensor((n_patch_h, n_patch_w)).repeat(n,1).to(device=device, dtype=torch.long)
        size = size[:, None, :]
        
       
        
        # Setup classifier-free guidance:
        if using_cfg:
            z = torch.cat([z, z], 0)            # (B, patch_size**2 * C, N) -> (2B, patch_size**2 * C, N)
            y_null = torch.tensor([1000] * n, device=device)
            y = torch.cat([y, y_null], 0)       # (B,) -> (2B, )
            grid = torch.cat([grid, grid], 0)   # (B, 2, N) -> (2B, 2, N)
            mask = torch.cat([mask, mask], 0)   # (B, N) -> (2B, N)
            model_kwargs = dict(y=y, grid=grid.long(), mask=mask, size=size, cfg_scale=args.cfg_scale)
            sample_fn = model.forward_with_cfg
        else:
            model_kwargs = dict(y=y, grid=grid.long(), mask=mask, size=size)
            sample_fn = model.forward
        
        # Sample images:
        samples = diffusion.p_sample_loop(
            sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device
        )
        if using_cfg:
            samples, _ = samples.chunk(2, dim=0)  # Remove null class samples

        samples = samples[..., : n_patch_h*n_patch_w]
        samples = model.unpatchify(samples, (H, W))
        samples = vae.decode(samples / vae.config.scaling_factor).sample
        samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to(torch.uint8).contiguous()

        # gather samples
        gathered_samples = [torch.zeros_like(samples) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, samples)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        torch.cuda.empty_cache()
        # Save samples to disk as individual .png files
        for i, sample in enumerate(samples.cpu().numpy()):
            index = i * dist.get_world_size() + rank + total
            Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")
        total += global_batch_size
        if rank == 0:
            pbar.update()

    # Make sure all processes have finished saving their samples before attempting to convert to .npz
    dist.barrier()
    if rank == 0:
        arr = np.concatenate(all_images, axis=0)
        arr = arr[: int(args.num_fid_samples)]
        npz_path = f"{sample_folder_dir}.npz"
        np.savez(npz_path, arr_0=arr)
        print(f"Saved .npz file to {npz_path} [shape={arr.shape}].")
        print("Done.")
    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfgdir",  type=str, default="")
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--sample-dir", type=str, default="workdir/eval")
    parser.add_argument("--per-proc-batch-size", type=int, default=32)
    parser.add_argument("--num-fid-samples", type=int, default=50_000)
    parser.add_argument("--image-height", type=int, default=256)
    parser.add_argument("--image-width", type=int, default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--vae-decoder", type=str, choices=['sd-ft-mse', 'sd-ft-ema'], default='sd-ft-ema')
    parser.add_argument("--cfg-scale",  type=str, default='1.5')
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--interpolation", type=str, choices=['no', 'linear', 'yarn', 'dynntk', 'partntk'], default='no') # interpolation
    parser.add_argument("--decouple", default=False, action="store_true") # interpolation
    parser.add_argument("--tf32", action='store_true', default=True)
    parser.add_argument("--mixed", type=str, default="fp32")
    args = parser.parse_args()
    main(args)
