![Figure](assets/figure.png)

# FiT: Flexible Vision Transformer for Diffusion Model

<p align="center">
📃 <a href="https://arxiv.org/pdf/2402.12376.pdf" target="_blank">Paper</a> • 📦 <a href="https://huggingface.co/whlzy/FiT-XL-2-16" target="_blank">Checkpoint</a> <br>

</p>

This repo contains PyTorch model definitions, pre-trained weights and sampling code for our flexible vision transformer (FiT).
FiT is a diffusion transformer based model which can generate images at unrestricted resolutions and aspect ratios.

The core features will include:
* Pre-trained class-conditional FiT-XL-2-16 (1800K) model weight trained on ImageNet ($H\times W \le 256\times256$).
* A pytorch sample code for running pre-trained DiT-XL/2 models to generate images at unrestricted resolutions and aspect ratios.

Why we need FiT?
* 🧐 Nature is infinitely resolution-free. FiT, like <a href="https://openai.com/sora" target="_blank">Sora</a>, was trained on the unrestricted resolution or aspect ratio. FiT is capable of generating images at unrestricted resolutions and aspect ratios.
* 🤗 FiT exhibits remarkable flexibility in resolution extrapolation generation.

Stay tuned for this project! 😆

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