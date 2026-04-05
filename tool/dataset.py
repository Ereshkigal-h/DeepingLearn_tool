import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import tool


class dataset(Dataset):
    def __init__(self,pair_path,transform=None,image_size=112,accuracy="float32"):
        self.pair_path = pair_path
        self.images_size = image_size
        self.samples=self.read_path(self.pair_path)
        self.transform = transform
        dtype_mapping = {
            "float16": (np.float16, torch.float16),
            "float32": (np.float32, torch.float32),
            "float64": (np.float64, torch.float64)
        }
        self.numpy_type,self.torch_type = dtype_mapping[accuracy]
    def read_path(self,pair_path):
        return
    def read_image(self, path):
        img = Image.open(os.path.join(path)).resize((self.images_size, self.images_size))
        img = np.array(img, dtype=self.numpy_type)
        img = np.transpose(img, (2, 0, 1))
        img = tool.np_to_tensor(img.copy())
        return img
    def read_single_data(self, path):
        return
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, index):
        single_sample=self.samples[index]
        data,label=single_sample
        return data,torch.tensor(label,dtype=self.torch_type)


