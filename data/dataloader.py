import pickle
from typing import Optional
import torch 
from torchvision.transforms import v2, ToTensor, Grayscale
import os
from PIL import Image
import tqdm

from .kernel_space_generator import get_m_samples_with_n_kernels
import threading
import queue


def initialize_transforms(img_size: int = 512):
    transforms = v2.Compose([
        ToTensor(),
        Grayscale(),
        v2.RandomCrop(size=(img_size, img_size)),
        v2.RandomResizedCrop(size=(img_size, img_size), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        v2.ToDtype(torch.float32, scale=False),
    ])

    return transforms

class DataLoader:
    _dataloader_path = os.path.realpath(__file__).split("dataloader.py")[0]

    def __init__(self, data_path: str, img_size: int = 512, block_size: int = 16):
        self.block_size = block_size
        self.transforms = initialize_transforms(img_size)
        self.training_data = []
        self.validation_data = []
        self.training_data_path = os.path.join(os.path.realpath(__file__).split("dataloader.py")[0], data_path, "train")
        self.validation_data_path = os.path.join(os.path.realpath(__file__).split("dataloader.py")[0], data_path, "valid")
        self.initialized = False

    @property
    def training(self):
        return self.get("train", None, 3)

    def get_valid_pic(self):
        return self.transforms(Image.open(os.path.join(self.validation_data_path, os.pardir, "sample_comparisson_photo.png")).convert("RGB"))

    @staticmethod
    def generate_samples(q: queue.Queue, n_batches, n_blocks, n_kernels, block_size, include_zero=False, negative_experts=False):
        for _ in range(n_batches):
            print("Generating samples")
            samples = get_m_samples_with_n_kernels(n_blocks, n_kernels, block_size, include_zero=include_zero, negative_experts=negative_experts)
            print("Generated samples")
            q.put(samples)

    def generate_random_blocks(self, n_batches: int, n_blocks: int, n_kernels: int, block_size: int, include_zero: bool = False, negative_experts: bool = False):
        q = queue.Queue()
        thread = threading.Thread(target=self.generate_samples, args=(q, n_batches, n_blocks, n_kernels, block_size, include_zero, negative_experts), daemon=True)
        thread.start()
        return q

    def get(self, data: str = "train", limit_to: int = None, batch_size: int = 1):
        if data == "train":
            data = torch.cat(self.training_data[:limit_to], dim=0)
        elif data == "valid":
            data = torch.cat(self.validation_data[:limit_to], dim=0)
        else:
            raise ValueError("train or valid expected")
        perm = torch.randperm(len(data))
        data = data[perm]
        if batch_size == 0:
            batch_size = len(data)
        elif batch_size < 0:
            batch_size = max(1, int(len(data)/(-batch_size)))
        for i in range(0, len(data), batch_size):
            yield data[i:i+batch_size], data[i:i+batch_size]
        yield data[i+batch_size:], data[i+batch_size:]

    def initialize(self, n_repeats: int = 3, force_reinitialize: bool = False) -> None:
        train_pkl_not_found = not os.path.exists(f"{self.training_data_path}/train.pkl")
        valid_pkl_not_found = not os.path.exists(f"{self.validation_data_path}/valid.pkl")
        if force_reinitialize:
            self.fill_training_data(use_saved=False, n_repeats=n_repeats)
            self.fill_validation_data(use_saved=False, n_repeats=n_repeats)
            self.initialized = True
            return
        
        if train_pkl_not_found:
            self.fill_training_data(use_saved=False, n_repeats=n_repeats)
        else:
            self.fill_training_data(n_repeats=n_repeats)
        if valid_pkl_not_found:
            self.fill_validation_data(use_saved=False, n_repeats=n_repeats)
        else:
            self.fill_validation_data(n_repeats=n_repeats)
        self.initialized = True
        

    def fill_training_data(self, use_saved: bool = True, n_repeats: int = 3) -> None:
        if use_saved:
            with open(f"{self.training_data_path}/train.pkl", "rb") as f:
                self.training_data = pickle.load(f)
                return
            
        for image_path in tqdm.tqdm(os.listdir(self.training_data_path), "Filling Training Set: "):
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
        for image_path in tqdm.tqdm(os.listdir(self.validation_data_path), "Filling Test Set: "):
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
        