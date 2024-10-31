from PIL import Image
import numpy as np
from tqdm import tqdm
import torch
import re
import os

from safetensors.torch import load_file


def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    imgs = sorted(os.listdir(sample_dir), key=lambda x: int(x.split('.')[0]))
    print(len(imgs))
    assert len(imgs) >= num
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{imgs[i]}")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path

def init_from_ckpt(
    model, checkpoint_dir, ignore_keys=None, verbose=False
) -> None: 
    if checkpoint_dir.endswith(".safetensors"):
        try:
            model_state_dict=load_file(checkpoint_dir)
        except: # 历史遗留问题，千万别删
            model_state_dict=torch.load(checkpoint_dir,  map_location="cpu")
    else:
        model_state_dict=torch.load(checkpoint_dir,  map_location="cpu")
    model_new_ckpt=dict()
    for i in model_state_dict.keys():
        model_new_ckpt[i] = model_state_dict[i]
    keys = list(model_new_ckpt.keys())
    for k in keys:
        if ignore_keys:
            for ik in ignore_keys:
                if re.match(ik, k):
                    print("Deleting key {} from state_dict.".format(k))
                    del model_new_ckpt[k]
    missing, unexpected = model.load_state_dict(model_new_ckpt, strict=False)
    if verbose:
        print(
            f"Restored with {len(missing)} missing and {len(unexpected)} unexpected keys"
        )
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")
    if verbose:
        print("")
