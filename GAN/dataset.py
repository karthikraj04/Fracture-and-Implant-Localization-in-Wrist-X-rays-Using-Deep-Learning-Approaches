import os
import torch
import numpy as np
from torch.utils.data import Dataset
# dataset.py
class MURADataset(Dataset):
    def __init__(self, root_dir, sliding_mask=False):
        self.image_paths = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.npy')]
        self.sliding_mask = sliding_mask

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = np.load(image_path)
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)

        mask = torch.ones_like(image)
        if self.sliding_mask:
            x = torch.randint(0, 192, (1,)).item()
            y = torch.randint(0, 192, (1,)).item()
            mask[:, y:y+64, x:x+64] = 0
        else:
            mask[:, 96:160, 96:160] = 0

        masked_image = image * mask
        return masked_image, image
