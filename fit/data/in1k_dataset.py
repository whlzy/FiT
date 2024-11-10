import os
import datetime
import torchvision
import numpy as np

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms
from accelerate.logging import get_logger
logger = get_logger(__name__, log_level="INFO")


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.Resampling.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.Resampling.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

def resize_arr(pil_image, image_size, vq_vae_down=8, patch_size=2):
    w, h = pil_image.size
    min_len = int(vq_vae_down*patch_size)  # 8*2=16 -> 256/16=16, 512/16=32
    if w * h >= image_size ** 2:
        new_w = np.sqrt(w/h) * image_size
        new_h = new_w * h / w        
    elif w < min_len:   # upsample, this case only happens twice in ImageNet1k_256
        new_w = min_len
        new_h = min_len * h / w
    elif h < min_len:   # upsample, this case only happens once in ImageNet1k_256
        new_h = min_len
        new_w = min_len * w / h
    else:
        new_w, new_h = w, h
    new_w, new_h = int(new_w/min_len)*min_len, int(new_h/min_len)*min_len
    
    if new_w == w and new_h == h:
        return pil_image
    else:
        return pil_image.resize((new_w, new_h), resample=Image.Resampling.BICUBIC)

class ImagenetDataDictWrapper(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def __getitem__(self, i):
        x, y = self.dataset[i]
        return {"jpg": x, "cls": y}

    def __len__(self):
        return len(self.dataset)

class ImagenetLoader():
    def __init__(self, train, rescale='crop'):
        super().__init__()

        self.train_config = train

        self.batch_size = self.train_config.loader.batch_size
        self.num_workers = self.train_config.loader.num_workers
        self.shuffle = self.train_config.loader.shuffle

        if rescale == 'crop':
            transform = transforms.Compose([
                transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, self.train_config.resize)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
            ])
        elif rescale == 'resize':
            transform = transforms.Compose([
                transforms.Lambda(lambda pil_image: resize_arr(pil_image, self.train_config.resize)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
            ])
        elif rescale == 'keep':
            transform = transforms.Compose([
                transforms.Lambda(lambda pil_image: resize_arr(pil_image, self.train_config.resize)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
            ])
        else:
            raise NotImplementedError
        
        self.train_dataset = ImagenetDataDictWrapper(ImageFolder(self.train_config.data_path, transform=transform))
        
        self.test_dataset = None
        self.val_dataset = None

    def train_len(self):
        return len(self.train_dataset)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )

    def test_dataloader(self):
        return None

    def val_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )

if __name__ == "__main__":
    from omegaconf import OmegaConf
    conf = OmegaConf.load('/home/luzeyu/projects/workspace/generative-models/configs/example_training/dataset/imagenet-256-streaming.yaml')
    indataloader=ImagenetLoader(train=conf.data.params.train).train_dataloader()
    from tqdm import tqdm
    for i in tqdm(indataloader):
        # print(i)
        # print("*"*20)
        pass