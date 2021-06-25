from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from torchvision.utils import save_image
import pandas as pd
import cv2 
import csv
from PIL import Image


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def crop(image):
    return transforms.functional.crop(image, 350, 0, 190, 820)

tsfm_car = transforms.Compose([
    transforms.Lambda(crop),
    transforms.Resize((256,256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

tsfm_person = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Temporal predicates
def before(intrvl1, intrvl2, min_dist=0, max_dist="infty"):
    return intrvl1[0] + min_dist <= intrvl2[0] and (max_dist == "infty" or intrvl1[0] + max_dist >= intrvl2[0])


def is_overlapped(intrvl1, intrvl2):
    return max(intrvl1[0], intrvl2[0]) <= min(intrvl1[1], intrvl2[1])


def coalesce(out_segments):
    if (len(out_segments) == 0):
        return out_segments

    coalesced_segments = []

    out_segments = sorted(out_segments, key=lambda intrvl: (intrvl[0], intrvl[0]))
    
    merged_interval = out_segments[0]

    for intrvl in out_segments:
        if is_overlapped(merged_interval, intrvl):
            merged_interval = [min(merged_interval[0], intrvl[0]), max(merged_interval[1], intrvl[1])]
        else:
            coalesced_segments.append(merged_interval)
            merged_interval = intrvl

    return coalesced_segments


def pattern_matching_before_within_10s(person_segments, car_segments):
    out_segments = []
    # Person, followed by car, within 10 seconds
    for intrvl1 in person_segments:
        for intrvl2 in car_segments:
            if before(intrvl1, intrvl2, max_dist=10 * 25):
                out_segments.append([min(intrvl1[0], intrvl2[0]), max(intrvl1[1], intrvl2[1])])
    return out_segments


def query_atomic(model_person_edge_corner, model_car_turning_right, video_idx):

    # video = cv2.VideoCapture("/home/ubuntu/CSE544-project/data/visual_road/traffic-4k-002.mp4")
    video = cv2.VideoCapture("/home/ubuntu/CSE544-project/data/visual_road/traffic-" + str(video_idx) + ".mp4")
    
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    
    person_edge_corner_results = [0] * num_frames
    car_turning_right_results = [0] * num_frames

    frame_id = 0

    if not video.isOpened():
        print("Error opening video stream or file: ", file)
    else:
        while video.isOpened():
            # print("frame_id: ", frame_id)
            success, frame_raw = video.read()
            if not success: 
                break
            color_coverted = cv2.cvtColor(frame_raw, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(color_coverted)
            transformed_frame = tsfm_car(pil_image)
            frame = torch.unsqueeze(transformed_frame, 0)
            frame = frame.to(device)

            transformed_frame_person = tsfm_person(pil_image)
            frame_person = torch.unsqueeze(transformed_frame_person, 0)
            frame_person = frame_person.to(device)
            
            # Run Model: person edge corner
            output = model_person_edge_corner(frame_person)
            _, pred = torch.max(output, 1)
            predicted_label = pred[0].item()
            person_edge_corner_results[frame_id] = predicted_label

            # Run Model: car turning right
            output = model_car_turning_right(frame)
            _, pred = torch.max(output, 1)
            predicted_label = pred[0].item()
            car_turning_right_results[frame_id] = predicted_label

            frame_id += 1 

    print("number of [car turning right] frames: ", sum(car_turning_right_results))
    print("number of [person edge corner] frames: ", sum(person_edge_corner_results))
    
    pd.DataFrame({'frame id': list(range(num_frames)), 'prediction[car turning right]': car_turning_right_results, 'prediction[person edge corner]':person_edge_corner_results}).to_csv('query_atomic_test/traffic' + str(video_idx) + '.csv', index=False)

def construct_video_clips(video_idx):
    with open('query_atomic_test/traffic' + str(video_idx) + '.csv', newline='') as f:
        reader = csv.reader(f)
        data = list(reader)

    data_array = np.asarray(data[1:], dtype=int)

    car_turning_right_results = data_array[:, 1]
    person_edge_corner_results = data_array[:, 2]

    car_segments = []
    person_segments = []
    i = 0
    while i < len(car_turning_right_results):
        # detected the start of the car event
        if car_turning_right_results[i] == 1:
            start_frame = i
            duration = 1
            while i + duration < len(car_turning_right_results) and car_turning_right_results[i + duration] == 1:
                duration += 1
            end_frame = i + duration - 1
            # filter out segments that are too short
            if duration >= 25:
                car_segments.append([start_frame, end_frame])
            i += duration
        i += 1

    i = 0
    while i < len(person_edge_corner_results):
        # detected the start of the car event
        if person_edge_corner_results[i] == 1:
            start_frame = i
            duration = 1
            while i + duration < len(person_edge_corner_results) and person_edge_corner_results[i + duration] == 1:
                duration += 1
            end_frame = i + duration - 1
            # filter out segments that are too short
            if duration >= 25:
                person_segments.append([start_frame, end_frame])
            i += duration
        i += 1


    out_segments = pattern_matching_before_within_10s(person_segments, car_segments)

    # Coalesce results
    out_segments = coalesce(out_segments)

    # Writing video clips
    # video = cv2.VideoCapture("/home/ubuntu/CSE544-project/data/visual_road/traffic-4k-002.mp4")
    video = cv2.VideoCapture("/home/ubuntu/CSE544-project/data/visual_road/traffic-" + str(video_idx) + ".mp4")

    # Get video configurations
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_id = 0

    output_dir = 'video_clips_test2/traffic' + str(video_idx)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for idx, intrvl in enumerate(out_segments):
        # print(idx)
        start_frame = intrvl[0]
        end_frame = intrvl[1]
        
        video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Define the codec and create VideoWriter object. The output is stored in 'outpy.avi' file.
        out = cv2.VideoWriter(os.path.join(output_dir, 'traffic' + str(video_idx) + '_' + str(idx) + '_' + str(start_frame) + '_' + str(end_frame) + '.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width,frame_height))
        
        for frame_count in range(end_frame - start_frame + 1):
            ret, frame = video.read()
            out.write(frame)
        out.release()
            

if __name__ == '__main__':
    # Load Model: person edge corner
    model_person_edge_corner = models.resnet18(pretrained=False)
    num_ftrs = model_person_edge_corner.fc.in_features
    model_person_edge_corner.fc = nn.Linear(num_ftrs, 2)
    model_person_edge_corner.load_state_dict(torch.load('/home/ubuntu/CSE544-project/rekall/tutorials/person_edge_corner_test/state_dict_model.pt'))
    model_person_edge_corner.eval()
    model_person_edge_corner.to(device)
    
    # Load Model: car turning right
    model_car_turning_right = models.resnet18(pretrained=False)
    num_ftrs = model_car_turning_right.fc.in_features
    model_car_turning_right.fc = nn.Linear(num_ftrs, 2)
    model_car_turning_right.load_state_dict(torch.load('/home/ubuntu/CSE544-project/rekall/tutorials/car_turning_right_test/state_dict_model_no_car_and_passing_car.pt'))
    model_car_turning_right.eval()
    model_car_turning_right.to(device)
    
    # construct_video_clips(2)
    # traffic-1, traffic-2, ..., traffic-20
    for i in range(1, 21):
        print("processing video: traffic-" + str(i))
        query_atomic(model_person_edge_corner, model_car_turning_right, i)
        construct_video_clips(i)