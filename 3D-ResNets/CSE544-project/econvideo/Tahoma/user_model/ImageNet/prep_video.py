import os
import random

import cv2

classes = ['n01818515', 'n02123045', 'n02981792', 'n03100240', 'n03594945', 'n03642806', 'n03769881', 'n03770679', 'n03791053', 'n03792782', 'n03977966', 'n03868863', 'n03902125', 'n03930630', 'n04399382', 'n04456115', 'n06794110', 'n06874185', 'n07583066', 'n07614500', 'n07753592']

img_folder = "/z/wenjiah/ImageNet/train/"
imgs = []

for img_dir in os.listdir(img_folder):
    if img_dir in classes:
        dir_path = os.path.join(img_folder, img_dir)
        for img_name in os.listdir(dir_path):
            if img_name.endswith(".JPEG"):
                img_path = os.path.join(dir_path, img_name)
                imgs.append(img_path)

random.shuffle(imgs)
print(f"Number of validation images: {len(imgs)}")

for i in range(100): 
    video_name = f"videos/imagenet_train_{i}.avi"
    start_idx = 273 * i
    end_idx = 273 * (i + 1)
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    shape = (480, 360)

    video = cv2.VideoWriter(video_name, fourcc, 1, shape)

    for j in range(start_idx, end_idx):
        img_path = imgs[j]
        image = cv2.imread(img_path)
        resized = cv2.resize(image, shape)
        video.write(resized)

    video.release()