mkdir datasets
cd datasets
mkdir imagenet1k_latents_1024_sd_vae_ft_ema
cd imagenet1k_latents_1024_sd_vae_ft_ema

wget -c "https://huggingface.co/datasets/InfImagine/imagenet_features_1024_sd_vae_ft_ema/resolve/main/from_16_to_1024.tar.gz.part_aa?download=true" -O from_16_to_1024.tar.gz.part_aa

tar -xzvf from_16_to_1024.tar.gz.part_aa

wget -c "https://huggingface.co/datasets/InfImagine/imagenet_features_1024_sd_vae_ft_ema/resolve/main/from_16_to_1024.tar.gz.part_ab?download=true" -O from_16_to_1024.tar.gz.part_ab

tar -xzvf from_16_to_1024.tar.gz.part_aa

wget -c "https://huggingface.co/datasets/InfImagine/imagenet_features_1024_sd_vae_ft_ema/resolve/main/from_16_to_1024.tar.gz.part_ac?download=true" -O from_16_to_1024.tar.gz.part_ac

tar -xzvf from_16_to_1024.tar.gz.part_aa

wget -c "https://huggingface.co/datasets/InfImagine/imagenet_features_1024_sd_vae_ft_ema/resolve/main/greater_than_1024_crop.tar.gz?download=true" -O greater_than_1024_crop.tar.gz

tar -xzvf greater_than_1024_crop.tar.gz

wget -c "https://huggingface.co/datasets/InfImagine/imagenet_features_1024_sd_vae_ft_ema/resolve/main/greater_than_1024_resize.tar.gz?download=true" -O greater_than_1024_resize.tar.gz

tar -xzvf greater_than_1024_resize.tar.gz

wget -c "https://huggingface.co/datasets/InfImagine/imagenet1k_features_256_sd_vae_ft_ema/resolve/main/less_than_16.tar.gz?download=true" -O less_than_16.tar.gz

tar -xzvf less_than_16.tar.gz
