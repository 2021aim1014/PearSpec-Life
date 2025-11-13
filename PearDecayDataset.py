import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

def zero_pad_to_size(img, target_size=(224, 224)):
    """Zero-pad image to target size (no resizing)."""
    h, w = img.shape[:2]
    th, tw = target_size
    pad_top = (th - h) // 2 if h < th else 0
    pad_bottom = th - h - pad_top if h < th else 0
    pad_left = (tw - w) // 2 if w < tw else 0
    pad_right = tw - w - pad_left if w < tw else 0

    if img.ndim == 3:  #HSI/RGB
        return np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant')
    else:              # Grey Scale
        return np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant')

def normalize_rgb(img):
    """Normalize RGB image to [0, 1]."""
    return img.astype(np.float32) / 255.0

def normalize_hsi(hsi):
    """Normalize HSI cube channel-wise to [0, 1]."""
    hsi = hsi.astype(np.float32)
    hsi_min = hsi.min(axis=(0, 1), keepdims=True)
    hsi_max = hsi.max(axis=(0, 1), keepdims=True)
    return (hsi - hsi_min) / (hsi_max - hsi_min)

class PearDecayDataset(Dataset):
    def __init__(self, root_dir, data_type='RGB'):
        self.root_dir = root_dir
        self.data_pairs = []
        self.data_type = data_type

        print(f"📂 Scanning dataset in {root_dir}...")
        pear_names = [name for name in os.listdir(root_dir) if not name.startswith('.')]
        for pears_name in tqdm(pear_names):
            pear_path = os.path.join(root_dir, pears_name)
            pear_views = [name for name in os.listdir(pear_path) if not name.startswith('.')]
            for pear_view in pear_views:
                view_path = os.path.join(pear_path, pear_view)
                day_indexs = [name for name in os.listdir(view_path) if not name.startswith('.')]
                day_indexs = sorted(day_indexs, lambda x: int(x))   
                temp = 1 if len(day_indexs) > 10 else 0         
                for i, day_index in enumerate(day_indexs):
                    day_path = os.path.join(view_path, day_index)
                    day_left = len(day_index)-i-1
                    if self.data_type == 'RGB':
                        data_path = os.path.join(day_path, f"{day_index}_rgb_seg.png")
                    else: 
                        data_path = os.path.join(day_path, f"{day_index}_hsi_seg.np")

                    if os.path.exists(data_path):
                        self.data_pairs.append((data_path, temp, day_left))
        
        # print(self.data_pairs)

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        sample = self.data_pairs[idx]
        temperature = torch.tensor(sample[1], dtype=torch.float32)
        days_left = torch.tensor(sample[2], dtype=torch.float32)

        # --- Load RGB ---
        if self.data_type == 'RGB':
            rgb_img = cv2.imread(sample[0])
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
            rgb_img = zero_pad_to_size(rgb_img, target_size=(224, 224))
            rgb_img = normalize_rgb(rgb_img)
            rgb_img = torch.from_numpy(rgb_img).permute(2, 0, 1)  # [3, H, W]
            return rgb_img, temperature, days_left
        else:
            hsi = np.load(sample["hsi_path"])
            hsi = np.rot90(hsi, k=3, axes=(0, 1))
            hsi = zero_pad_to_size(hsi)
            hsi = normalize_hsi(hsi)
            hsi = torch.from_numpy(hsi).permute(2, 0, 1)  # [204, H, W]
            return hsi, temperature, days_left
