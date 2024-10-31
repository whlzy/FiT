![Figure](assets/figure.png)

# FiT: Flexible Vision Transformer for Diffusion Model

<p align="center">
📃 <a href="https://arxiv.org/pdf/2402.12376.pdf" target="_blank">FiT Paper</a> • 
📦 <a href="https://huggingface.co/InfImagine/FiT" target="_blank">FiT Checkpoint</a> <br> • 
📃 <a href="https://arxiv.org/pdf/2410.13925" target="_blank">FiTv2 Paper</a> • 
📦 <a href="https://huggingface.co/InfImagine/FiTv2" target="_blank">FiTv2 Checkpoint</a> <br> 
</p>

This is the official repo which contains PyTorch model definitions, pre-trained weights and sampling code for our flexible vision transformer (FiT).
FiT is a diffusion transformer based model which can generate images at unrestricted resolutions and aspect ratios.

The core features will include:
* Pre-trained class-conditional FiT-XL-2-16 (2000K) model weight trained on ImageNet ($H\times W \le 256\times256$).
* Pre-trained class-conditional FiTv2-XL-2-16 (2000K) and FiTv2-3B-2-16 (1000K) model weight trained on ImageNet ($H\times W \le 256\times256$).
* High-resolution Fine-tuned FiTv2-XL-2-32 (400K) and FiTv2-3B-2-32 (200K) model weitht trained on ImageNet ($H\times W \le 512\times512$).
* A pytorch sample code for running pre-trained FiT and FiTv2 models to generate images at unrestricted resolutions and aspect ratios.

Why we need FiT?
* 🧐 Nature is infinitely resolution-free. FiT, like <a href="https://openai.com/sora" target="_blank">Sora</a>, was trained on the unrestricted resolution or aspect ratio. FiT is capable of generating images at unrestricted resolutions and aspect ratios.
* 🤗 FiT exhibits remarkable flexibility in resolution extrapolation generation.

Stay tuned for this project! 😆


## Setup
First, download and setup the repo:
```
git clone https://github.com/whlzy/FiT.git
cd FiT
```
## Installation
```
conda create -n fit_env python=3.10
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu118
pip install xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install -e .
```

## Sample
### Basic Sampling

Basically, the model is trained with images whose $H\times W\leqslant 256\times256$.
Our FiTv1-XL/2 model and FiTv2-XL/2 model are trained with batch size of 256 for 2000K steps. 
Our FiTv2-3B/2 model is trained with batch size of 256 for 1000K steps.

The pre-trained FiT models can be downloaded directyl from huggingface:
| FiT Model     | Checkpoint | FID-256x256 | FID-320x320 | Model Size | GFlOPS |
|---------------|------------|---------|-----------------|------------| ------ |
| [FiTv1-XL/2](https://huggingface.co/InfImagine/FiT/tree/main/FiTv1_xl) | [CKPT](https://huggingface.co/InfImagine/FiT/blob/main/FiTv1_xl/model_ema.bin) | 4.21 | 5.11 | 824M | 153 |
| [FiTv2-XL/2](https://huggingface.co/InfImagine/FiTv2/tree/main/FiTv2_XL) | [CKPT](https://huggingface.co/InfImagine/FiTv2/resolve/main/FiTv2_XL/model_ema.safetensors?download=true) | 2.26 | 3.55 | 671M | 147 |
| [FiTv2-3B/2](https://huggingface.co/InfImagine/FiT/tree/main/FiTv1_xl) | [CKPT](https://huggingface.co/InfImagine/FiTv2/resolve/main/FiTv2_3B/model_ema.bin?download=true) | 2.15 | 3.22 | 3B | 653 |


#### Downloading the checkpoints
Downloading via wget:
```
mkdir checkpoints

wget -c "https://huggingface.co/InfImagine/FiT/blob/main/FiTv1_xl/model_ema.bin" -O checkpoints/fitv1_xl.bin

wget -c "https://huggingface.co/InfImagine/FiTv2/resolve/main/FiTv2_XL/model_ema.safetensors?download=true" -O checkpoints/fitv2_xl.safetensors

wget -c "https://huggingface.co/InfImagine/FiTv2/resolve/main/FiTv2_3B/model_ema.bin?download=true" -O checkpoints/fitv2_3B.bin
```

#### Sampling 256x256 Images
Sampling with FiTv1-XL/2 for $256\times 256$ Images:
```
python -m torch.distributed.run --nnodes=${NUM_NODE} --nproc_per_node=${NUM_GPU} --rdzv_endpoint localhost:$MASTER_PORT sample_fit_ddp.py --num-fid-samples 50000 --cfgdir configs/fit/config_fit_xl.yaml --ckpt checkpoints/fitv1_xl.bin --image-height 256 --image-width 256 --num-sampling-steps 250 --cfg-scale 1.5 --global-seed 0 --per-proc-batch-size 32
```
Sampling with FiTv2-XL/2 for $256\times 256$ Images:
```
python -m torch.distributed.run --nnodes=${NUM_NODE} --nproc_per_node=${NUM_GPU} --rdzv_endpoint localhost:$MASTER_PORT sample_fitv2_ddp.py --num-fid-samples 50000 --cfgdir configs/fitv2/config_fitv2_xl.yaml --ckpt checkpoints/fitv2_xl.safetensors --image-height 256 --image-width 256 --num-sampling-steps 250 --cfg-scale 1.5 --global-seed 0 --per-proc-batch-size 32  --sampler-mode ODE
```
Sampling with FiTv2-3B/2 for $256\times 256$ Images:
```
python -m torch.distributed.run --nnodes=${NUM_NODE} --nproc_per_node=${NUM_GPU} --rdzv_endpoint localhost:$MASTER_PORT sample_fitv2_ddp.py --num-fid-samples 50000 --cfgdir configs/fitv2/config_fitv2_3B.yaml --ckpt checkpoints/fitv2_3B.bin --image-height 256 --image-width 256 --num-sampling-steps 250 --cfg-scale 1.5 --global-seed 0 --per-proc-batch-size 32  --sampler-mode ODE
```
Note that *NUM_NODE*, *NUM_GPU* and *MASTER_PORT* need to be specified.


#### Sampling Images with arbitrary resolutions
We can assign the *image-height* and *image-width* with any value we want. And we need to specify the original maximum positional embedding length (*ori-max-pe-len*) and the interpolation method. 
We show some examples as follows.

Sampling with FiTv2-XL/2 for $160\times 320$ images:
```
python -m torch.distributed.run --nnodes=${NUM_NODE} --nproc_per_node=${NUM_GPU} --rdzv_endpoint localhost:$MASTER_PORT sample_fitv2_ddp.py --num-fid-samples 50000 --cfgdir configs/fitv2/config_fitv2_xl.yaml --ckpt checkpoints/fitv2_xl.safetensors --image-height 160 --image-width 320 --num-sampling-steps 250 --cfg-scale 1.5 --global-seed 0 --per-proc-batch-size 32  --sampler-mode ODE --ori-max-pe-len 16 --interpolation dynntk --decouple
```
Sampling with FiTv2-XL/2 for $320\times 320$ images:
```
python -m torch.distributed.run --nnodes=${NUM_NODE} --nproc_per_node=${NUM_GPU} --rdzv_endpoint localhost:$MASTER_PORT sample_fitv2_ddp.py --num-fid-samples 50000 --cfgdir configs/fitv2/config_fitv2_xl.yaml --ckpt checkpoints/fitv2_xl.safetensors --image-height 320 --image-width 320 --num-sampling-steps 250 --cfg-scale 1.5 --global-seed 0 --per-proc-batch-size 32  --sampler-mode ODE --ori-max-pe-len 16 --interpolation ntkpro2 --decouple
```
Note that *NUM_NODE*, *NUM_GPU* and *MASTER_PORT* need to be specified.


### High-resolution Sampling

For high-resolution image generation, we use images whose $H\times W \leqslant 512\times512$. 
Our FiTv2-XL/2 is finetuned with batch size of 256 for 400K steps, while
FiTv2-3B/2 is finetuned with batch size of 256 for 200K steps.

The high-resolution fine-tuned FiT models can be downloaded directyl from huggingface:
| FiT Model     | Checkpoint | FID-512x512 | FID-320x640 | Model Size | GFlOPS |
|---------------|------------|---------|-----------------|------------| ------ |
| [FiTv2-HR-XL/2](https://huggingface.co/InfImagine/FiTv2/tree/main/FiTv2_XL_HRFT) | [CKPT](https://huggingface.co/InfImagine/FiTv2/resolve/main/FiTv2_XL_HRFT/model_ema.safetensors?download=true) | 2.90 | 4.87 | 671M | 147 |
| [FiTv2-HR-3B/2](https://huggingface.co/InfImagine/FiTv2/tree/main/FiTv2_3B_HRFT) | [CKPT](https://huggingface.co/InfImagine/FiTv2/resolve/main/FiTv2_3B_HRFT/model_ema.safetensors?download=true) | 2.41 | 4.54 | 3B | 653 |


#### Downloading
Downloading via wget:
```
mkdir checkpoints

wget -c "https://huggingface.co/InfImagine/FiTv2/resolve/main/FiTv2_XL_HRFT/model_ema.safetensors?download=true" -O checkpoints/fitv2_hr_xl.safetensors

wget -c "https://huggingface.co/InfImagine/FiTv2/resolve/main/FiTv2_3B_HRFT/model_ema.safetensors?download=true" -O checkpoints/fitv2_hr_3B.safetensors
```

#### Sampling 512x512 Images
Sampling with FiTv2-HR-XL/2 for $512\times 512$ Images:
```
python -m torch.distributed.run --nnodes=${NUM_NODE} --nproc_per_node=${NUM_GPU} --rdzv_endpoint localhost:$MASTER_PORT sample_fitv2_ddp.py --num-fid-samples 50000 --cfgdir configs/fitv2/config_fitv2_hr_xl.yaml --ckpt checkpoints/fitv2_hr_xl.safetensors --image-height 512 --image-width 512 --num-sampling-steps 250 --cfg-scale 1.65 --global-seed 0 --per-proc-batch-size 32  --sampler-mode ODE --ori-max-pe-len 16 --interpolation dynntk --decouple
```
Sampling with FiTv2-HR-3B/2 for $512\times 512$ Images:
```
python -m torch.distributed.run --nnodes=${NUM_NODE} --nproc_per_node=${NUM_GPU} --rdzv_endpoint localhost:$MASTER_PORT sample_fitv2_ddp.py --num-fid-samples 50000 --cfgdir configs/fitv2/config_fitv2_hr_3B.yaml --ckpt checkpoints/fitv2_hr_3B.safetensors --image-height 512 --image-width 512 --num-sampling-steps 250 --cfg-scale 1.5 --global-seed 0 --per-proc-batch-size 32  --sampler-mode ODE --ori-max-pe-len 16 --interpolation dynntk --decouple
```
Note that *NUM_NODE*, *NUM_GPU* and *MASTER_PORT* need to be specified.

#### Sampling Images with arbitrary resolutions
Sampling with FiTv2-HR-XL/2 for $320\times 640$ images:
```
python -m torch.distributed.run --nnodes=${NUM_NODE} --nproc_per_node=${NUM_GPU} --rdzv_endpoint localhost:$MASTER_PORT sample_fitv2_ddp.py --num-fid-samples 50000 --cfgdir configs/fitv2/config_fitv2_hr_xl.yaml --ckpt checkpoints/fitv2_hr_xl.safetensors --image-height 320 --image-width 640 --num-sampling-steps 250 --cfg-scale 1.65 --global-seed 0 --per-proc-batch-size 32  --sampler-mode ODE --ori-max-pe-len 16 --interpolation dynntk --decouple
```
Note that *NUM_NODE*, *NUM_GPU* and *MASTER_PORT* need to be specified.

## Evaluations
The sampling generates a folder of samples as well as a `.npz` file which can be directly used with [ADM's TensorFlow
evaluation suite](https://github.com/openai/guided-diffusion/tree/main/evaluations) to compute FID, Inception Score and
other metrics. 



## Acknowledgments
This codebase borrows from <a href="https://github.com/facebookresearch/DiT/tree/main" target="_blank">DiT</a>.

## BibTeX
```bibtex
@article{Lu2024FiT,
  title={FiT: Flexible Vision Transformer for Diffusion Model},
  author={Zeyu Lu and Zidong Wang and Di Huang and Chengyue Wu and Xihui Liu and Wanli Ouyang and Lei Bai},
  year={2024},
  journal={arXiv preprint arXiv:2402.12376},
}
```
```bibtex
@article{wang2024fitv2,
  title={Fitv2: Scalable and improved flexible vision transformer for diffusion model},
  author={Wang, ZiDong and Lu, Zeyu and Huang, Di and Zhou, Cai and Ouyang, Wanli and others},
  journal={arXiv preprint arXiv:2410.13925},
  year={2024}
}
```