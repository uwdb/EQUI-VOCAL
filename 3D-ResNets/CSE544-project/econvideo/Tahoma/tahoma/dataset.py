"""
Video Dataset
    Class wrapper for interfacing with the dataset of video frames
"""
import json
import os

import cv2
# from imageio import imread
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split, TensorDataset
import skimage.transform
from utils import config

# def resize(X, image_dim):
#     """
#     Resizes the data partition X to the size specified in the config file.
#     Uses bicubic interpolation for resizing.

#     Returns:X
#         the resized images as a numpy array.
#     """
#     #print(X.shape)
#     resized = np.zeros((X.shape[0], image_dim, image_dim, 1)) if X[0].shape[2] == 1 else np.zeros((X.shape[0], image_dim, image_dim, 3))
#     for i in range(0, X.shape[0]):
#         # imresize will be deprecated
#         #resized[i] = imresize(arr = X[i], size = (image_dim, image_dim, 3), interp = 'bicubic')
#         if X[i].shape[2] == 4:
#             resized[i] = np.array(Image.fromarray(X[i]).convert('RGB').resize(size=(image_dim, image_dim), resample=Image.BICUBIC))
#         elif X[i].shape[2] == 3:
#             resized[i] = cv2.resize(X[i], (image_dim, image_dim), interpolation=cv2.INTER_CUBIC)
#             # resized[i] = np.array(Image.fromarray(X[i]).resize(size=(image_dim, image_dim), resample=Image.BICUBIC))
#         else: # shape[2] == 1 
#             resized[i] = skimage.transform.resize(X[i], (image_dim, image_dim), anti_aliasing=True, mode='constant', preserve_range=True).astype(np.uint8).astype(np.float64)
#     return resized

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

    # def save_info(self):
    #     with open(config("info_json"), 'w') as f:
    #         json.dump({"mean": self.image_mean.tolist(), "std": self.image_std.tolist(), "image_dim": config("image_dim")}, f)

    def fit(self, X):
        color_channel = X.shape[3]

        self.image_mean = np.zeros(color_channel)
        self.image_std = np.zeros(color_channel)

        for i in range(0, color_channel):
            layer = X[:, :, :, i]
            self.image_mean[i] = layer.mean()
            self.image_std[i] = layer.std()
        
        # print("image_mean:", self.image_mean, "image_std:", self.image_std)

    def transform(self, X):
        # for i in range(0, X.shape[3]):
        #     arr = X[:, :, :, i]
        #     new_arr = (arr - self.image_mean[i]) / self.image_std[i]
        #     X[:, :, :, i] = new_arr
        normalized_X = (X - self.image_mean) / self.image_std
        # print(X)
        return normalized_X
    
    def detransform(self, X):
        for i in range(0, X.shape[3]):
            arr = X[:, :, :, i]
            arr = arr * self.image_std[i] + self.image_mean[i]
            X[:, :, :, i] = arr
        return X.astype(np.uint8)


def get_train_val_loaders(num_classes, data_dir, color_mode, image_dim, batch_size):
    #X_train_original, X_val_original, X_train, X_val, y_train, y_val, _ = get_train_val_dataset(num_classes=num_classes)
    tr, va, cascade, standardizer = get_train_val_dataset(data_dir, color_mode, image_dim, num_classes=num_classes)
    tr_loader = DataLoader(tr, batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(va, batch_size=batch_size, shuffle=False)
    cascade_loader = DataLoader(cascade, batch_size=batch_size, shuffle=False)

    return tr_loader, va_loader, cascade_loader, standardizer

def get_train_val_dataset(data_dir, color_mode, image_dim, num_classes=2):
    
    image_size_folder = str(image_dim[0])
    tr_data_dir = os.path.join(data_dir, image_size_folder, 'train')
    va_data_dir = os.path.join(data_dir, image_size_folder, 'validation')
    cascade_data_dir = os.path.join(data_dir, image_size_folder, 'cascade')
    
    tr = VideoDataset(tr_data_dir, color_mode, num_classes)
    va = VideoDataset(va_data_dir, color_mode, num_classes)
    cascade = VideoDataset(cascade_data_dir, color_mode, num_classes)
    
    # Crop images as specified
    if image_dim[0] != image_dim[1]:
        # 0, 1, 2, 3 represent the left, top, right, bottom portion respectively
        smaller_dim = image_dim[0] // 2
        if image_dim[1] == 0:
            tr.X = tr.X[:, :, 0:smaller_dim, :]
            va.X = va.X[:, :, 0:smaller_dim, :]
            cascade.X = cascade.X[:, :, 0:smaller_dim, :]
        elif image_dim[1] == 1:
            tr.X = tr.X[:, 0:smaller_dim, :, :]
            va.X = va.X[:, 0:smaller_dim, :, :]
            cascade.X = cascade.X[:, 0:smaller_dim, :, :]
        elif image_dim[1] == 2:
            tr.X = tr.X[:, :, smaller_dim:, :]
            va.X = va.X[:, :, smaller_dim:, :]
            cascade.X = cascade.X[:, :, smaller_dim:, :]
        else:
            tr.X = tr.X[:, smaller_dim:, :, :]
            va.X = va.X[:, smaller_dim:, :, :]
            cascade.X = cascade.X[:, smaller_dim:, :, :]

    # Standardize
    standardizer = ImageStandardizer()
    #standardizer.fit(tr.X.astype(np.float))
    standardizer.fit(tr.X)
    # standardizer.save_info()
    # tr.X = standardizer.transform(tr.X.astype(np.float))
    # va.X = standardizer.transform(va.X.astype(np.float))
    # cascade.X = standardizer.transform(cascade.X.astype(np.float))
    tr.X = standardizer.transform(tr.X)
    va.X = standardizer.transform(va.X)
    cascade.X = standardizer.transform(cascade.X)

    # Transpose the dimensions from (N,H,W,C) to (N,C,H,W)
    tr.X = tr.X.transpose(0,3,1,2)
    va.X = va.X.transpose(0,3,1,2)
    cascade.X = cascade.X.transpose(0,3,1,2)

    return tr, va, cascade, standardizer

class VideoDataset(Dataset):
    def __init__(self, data_dir, color_mode, num_classes=2):
        """
        Reads in the necessary data from disk.
        """
        super().__init__()

        np.random.seed(0)
        
        self.data_dir = data_dir
        self.color_mode = color_mode
        self.num_classes = num_classes

        # Load in all the data we need from disk
        self.X, self.y = self._load_data()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # return torch.from_numpy(self.X[idx]).float(), torch.tensor(self.y[idx], dtype=torch.float)
        return torch.from_numpy(self.X[idx]).float(), torch.tensor(self.y[idx], dtype=torch.int64)
        
    def _load_data(self):
        """
        Loads a single data partition from file. (either training or validating partition)
        """
        pos_data_dir = os.path.join(self.data_dir, 'pos')
        neg_data_dir = os.path.join(self.data_dir, 'neg')
        
        X, y = [], []

        if self.color_mode == 'BWImage':
            for file in os.listdir(pos_data_dir):
                image = cv2.imread(os.path.join(pos_data_dir, file), 0)
                image = np.expand_dims(image, axis=2)
                X.append(image)
                y.append(1)

            for file in os.listdir(neg_data_dir):
                image = cv2.imread(os.path.join(neg_data_dir, file), 0)
                image = np.expand_dims(image, axis=2)
                X.append(image)
                y.append(0)
        elif self.color_mode == "ColorImage":
            for file in os.listdir(pos_data_dir):
                image = cv2.imread(os.path.join(pos_data_dir, file))
                X.append(image)
                y.append(1)

            for file in os.listdir(neg_data_dir):
                image = cv2.imread(os.path.join(neg_data_dir, file))
                X.append(image)
                y.append(0)      
        else:
            channel_idx = -1
            if self.color_mode == 'BlueChannel':
                channel_idx = 0
            elif self.color_mode == 'GreenChannel':
                channel_idx = 1
            else:
                channel_idx = 2

            for file in os.listdir(pos_data_dir):
                image = cv2.imread(os.path.join(pos_data_dir, file))
                sliced_image = image[:, :, channel_idx]
                sliced_image = np.expand_dims(sliced_image, axis=2)
                X.append(sliced_image)
                y.append(1)

            for file in os.listdir(neg_data_dir):
                image = cv2.imread(os.path.join(neg_data_dir, file))
                sliced_image = image[:, :, channel_idx]
                sliced_image = np.expand_dims(sliced_image, axis=2)
                X.append(sliced_image)
                y.append(0)

        return np.array(X), np.array(y)

if __name__ == '__main__':
    # set the number of digits of precision to 3 for printing output
    np.set_printoptions(precision=3)
    data_dir = '/z/analytics/VideoDB/is_fall/FAST1/labeled_images/'
    color_modes = ['ColorImage', 'RedChannel', 'GreenChannel', 'BlueChannel', 'BWImage'] 
    image_dim = 120
    tr, va, cascade, standardizer = get_train_val_dataset(data_dir, color_modes[0], image_dim)
    print("Train:\t", len(tr.X))
    print("Val:\t", len(va.X))
    print("Mean:", standardizer.image_mean)
    print("Std: ", standardizer.image_std)