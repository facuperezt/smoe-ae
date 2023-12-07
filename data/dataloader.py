import pickle
from typing import Optional
import torch 
from torchvision.transforms import v2, ToTensor, Grayscale
import os
from PIL import Image


def initialize_transforms(img_size: int = 512):
    transforms = v2.Compose([
        ToTensor(),
        Grayscale(),
        v2.RandomResizedCrop(size=(img_size, img_size), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        v2.ToDtype(torch.float32, scale=False),
    ])

    return transforms

class DataLoader:
    def __init__(self, img_size: int = 512, block_size: int = 16):
        self.block_size = block_size
        self.transforms = initialize_transforms(img_size)
        self.training_data = []
        self.validation_data = []
        self.training_data_path = os.path.join(os.path.realpath(__file__).split("dataloader.py")[0], "train")
        self.validation_data_path = os.path.join(os.path.realpath(__file__).split("dataloader.py")[0], "valid")
        self.initialized = False

    def get(self, data: str = "train", limit_to: int = None, batch_size: int = 1):
        perm = torch.randperm(len(self.training_data))
        self.training_data = self.training_data[perm, :, :]
        if data == "train":
            data = torch.cat(self.training_data[:limit_to], dim=0)
        elif data == "valid":
            data = torch.cat(self.validation_data[:limit_to], dim=0)
        else:
            raise ValueError("train or valid expected")
        if batch_size == 0:
            batch_size = len(data)
        elif batch_size < 0:
            batch_size = int(len(data)/(-batch_size))
        for i in range(0, len(data), batch_size):
            yield data[i:i+batch_size], data[i:i+batch_size]
        yield data[i+batch_size:], data[i+batch_size:]

    def initialize(self, force_reinitialize: bool = False) -> None:
        train_pkl_not_found = not os.path.exists(f"{self.training_data_path}/train.pkl")
        valid_pkl_not_found = not os.path.exists(f"{self.validation_data_path}/valid.pkl")
        if force_reinitialize:
            self.fill_training_data(use_saved=False)
            self.fill_validation_data(use_saved=False)
            self.initialized = True
            return
        
        if train_pkl_not_found:
            self.fill_training_data(use_saved=False)
        else:
            self.fill_training_data()
        if valid_pkl_not_found:
            self.fill_validation_data(use_saved=False)
        else:
            self.fill_validation_data()
        self.initialized = True
        

    def fill_training_data(self, use_saved: bool = True, n_repeats: int = 3) -> None:
        if use_saved:
            with open(f"{self.training_data_path}/train.pkl", "rb") as f:
                self.training_data = pickle.load(f)
                return
            
        for image_path in os.listdir(self.training_data_path):
            if image_path.startswith(".") or image_path.endswith(".pkl"):
                continue
            img = Image.open(os.path.join(self.training_data_path, image_path))
            self.training_data.extend([self.transforms(_img) for _img in n_repeats*[img]])

        if not use_saved:
            with open(f"{self.training_data_path}/train.pkl", "wb") as f:
                pickle.dump(self.training_data, f)

    def fill_validation_data(self, use_saved: bool = True, n_repeats: int = 3) -> None:
        if use_saved:
            with open(f"{self.validation_data_path}/valid.pkl", "rb") as f:
                self.validation_data = pickle.load(f)
                return
        for image_path in os.listdir(self.validation_data_path):
            if image_path.startswith(".") or image_path.endswith(".pkl"):
                continue
            img = Image.open(os.path.join(self.validation_data_path, image_path))
            self.validation_data.extend([self.transforms(_img) for _img in n_repeats*[img]])

        if not use_saved:
            with open(f"{self.validation_data_path}/valid.pkl", "wb") as f:
                pickle.dump(self.validation_data, f)

    def get_epoch_training_data(self):
        for x, y in zip(self.training_data, self.training_data):
            yield x, y

    def get_epoch_validation_data(self):
        for x, y in zip(self.validation_data, self.validation_data):
            yield x, y

    # def get_epoch_validation_data(self):
    #     for x, y in zip(self.validation_data, self.validation_data):
    #         yield torch.tensor(sliding_window(x.squeeze().numpy(), 2*[self.block_size], 2*[self.block_size], flatten=False), dtype=torch.float32, requires_grad=True), torch.tensor(sliding_window(y.squeeze().numpy(), 2*[self.block_size], 2*[self.block_size], flatten=False), dtype=torch.float32, requires_grad=True)
        