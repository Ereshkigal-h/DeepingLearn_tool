import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class ImageDataset(Dataset):

    def __init__(self, data_list, transform=None, image_size=112, accuracy="float32"):
        self.data_list = data_list
        self.transform = transform
        self.image_size = image_size
        dtype_mapping = {
            "float16": (np.float16, torch.float16),
            "float32": (np.float32, torch.float32),
            "float64": (np.float64, torch.float64)
        }
        self.numpy_type, self.torch_type = dtype_mapping.get(accuracy, (np.float32, torch.float32))
    def read_image(self, path):
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
            return img
        else:
            img = img.resize((self.image_size, self.image_size))
            img = np.array(img, dtype=self.numpy_type)
            # HWC -> CHW
            img = np.transpose(img, (2, 0, 1))
            return torch.tensor(img, dtype=self.torch_type)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        img_path, label = self.data_list[index]
        img_tensor = self.read_image(img_path)
        label_tensor = torch.tensor(label, dtype=self.torch_type)
        return img_tensor, label_tensor

class NLPDataset(Dataset):

    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        # 仅仅返回字符串，不做任何转换
        # 数据会在 DataLoader 的 collate_fn 阶段交由 Tokenizer 处理
        item = self.data_list[index]
        return item