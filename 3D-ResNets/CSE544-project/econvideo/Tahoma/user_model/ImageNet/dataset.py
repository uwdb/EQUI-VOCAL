"""
Video Dataset
    Class wrapper for interfacing with the dataset of video frames
"""
import os
import cv2
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from utils import config


def resize(X):
    """
    Resizes the data partition X to the size specified in the config file.
    Uses bicubic interpolation for resizing.

    Returns:
        the resized images as a numpy array.
    """
    image_dim = config('image_dim')
    resized = np.zeros((X.shape[0], image_dim, image_dim, 3))

    for i in range(0, X.shape[0]):
        resized[i] = cv2.resize(X[i], (image_dim, image_dim), interpolation=cv2.INTER_CUBIC)

    return resized


class ImageStandardizer(object):
    """
    Channel-wise standardization for batch of images to mean 0 and variance 1.
    The mean and standard deviation parameters are computed in `fit(X)` and
    applied using `transform(X)`.

    X has shape (N, image_height, image_width, color_channel)
    """
    def __init__(self):
        super().__init__()
        self.image_mean = None
        self.image_std = None

    def fit(self, X):
        color_channel = X.shape[3]

        self.image_mean = np.zeros(color_channel)
        self.image_std = np.zeros(color_channel)

        for i in range(0, color_channel):
            layer = X[:, :, :, i]
            self.image_mean[i] = layer.mean()
            self.image_std[i] = layer.std()

        print("image_mean:", self.image_mean, "image_std:", self.image_std)

    def transform(self, X):
        # for i in range(0, 3):
        #     X[:, :, :, i] = (X[:, :, :, i] - self.image_mean[i])/self.image_std[i]
        # print(X)
        normalized_X = (X - self.image_mean) / self.image_std
        #print(normalized_X)

        return normalized_X
    
    def save_info(self):
        with open(config("info_json"), 'w') as f:
            json.dump({"mean": self.image_mean.tolist(), "std": self.image_std.tolist(), "image_dim": config("image_dim")}, f)


def get_train_val_loaders(data_path, num_classes):
    tr, va, _ = get_train_val_dataset(data_path, num_classes=num_classes)
    
    batch_size = config('batch_size')
    tr_loader = DataLoader(tr, batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(va, batch_size=batch_size, shuffle=False)
    
    return tr_loader, va_loader


def get_train_val_dataset(data_path, num_classes=2):
    tr = ImageNetDataset('train', data_path, num_classes)
    va = ImageNetDataset('val', data_path, num_classes)

    # Resize
    tr.X = resize(tr.X)
    va.X = resize(va.X)
    
    # Standardize
    standardizer = ImageStandardizer()
    standardizer.fit(tr.X)
    standardizer.save_info()
    tr.X = standardizer.transform(tr.X)
    va.X = standardizer.transform(va.X)
    
    # Transpose the dimensions from (N,H,W,C) to (N,C,H,W)
    tr.X = tr.X.transpose(0,3,1,2)
    va.X = va.X.transpose(0,3,1,2)
    
    return tr, va, standardizer


class ImageNetDataset(Dataset):
    def __init__(self, partition, data_path, num_classes=2):
        """
        Reads in the necessary data from disk.
        """
        super().__init__()
        
        if partition not in ['train', 'val']:
            raise ValueError('Partition {} does not exist'.format(partition))
        
        np.random.seed(0)
        self.partition = partition
        self.num_classes = num_classes
        self.img_path = data_path
        
        # Load in all the data we need from disk
        # self.metadata = pd.read_csv(config('image_season_label_csv_file'))
        self.X, self.y = self._load_data()

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]).float(), torch.tensor(self.y[idx], dtype=torch.int64)
    
    def _load_data(self):
        """
        Loads a single data partition from file.
        """
        print(f"Loading {self.partition} data ...")
        
        X, y = [], []
        true_count, false_count = 0, 0

        pos_img_path = os.path.join(self.img_path, self.partition, "pos")
        neg_img_path = os.path.join(self.img_path, self.partition, "neg")

        for img_name in os.listdir(pos_img_path):
            if img_name.endswith("JPEG"):
                img_file_path = os.path.join(pos_img_path, img_name)
                image = cv2.imread(img_file_path)
                X.append(image)
                y.append(1)

                true_count += 1
                print(f"{self.partition} true: {true_count}")
  
        for img_name in os.listdir(neg_img_path):
            if img_name.endswith("JPEG"):
                img_file_path = os.path.join(neg_img_path, img_name)
                image = cv2.imread(img_file_path)
                X.append(image)
                y.append(0)

                false_count += 1
                print(f"{self.partition} false: {false_count}")
        
        return np.array(X), np.array(y)