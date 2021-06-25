# import some common libraries
import numpy as np
import os, json, cv2, random, time, csv
from tqdm import tqdm
import pickle
from PIL import Image
from torchvision import transforms as T


input_video_dir = "/home/ubuntu/CSE544-project/data/visual_road/"
output_dir = "/home/ubuntu/CSE544-project/data/visual_road_snippets"

for i in range(1, 21):
    file = "traffic-{}.mp4".format(i)

    video = cv2.VideoCapture(os.path.join(input_video_dir, file))
    # Get video configurations
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    keep_reading = True
    idx = 0
    while video.isOpened() and keep_reading:
        idx += 1
        # Define the codec and create VideoWriter object. The output is stored in 'outpy.avi' file.
        out = cv2.VideoWriter(os.path.join(output_dir, 'traffic' + str(i) + '_' + str(idx) + '.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width,frame_height))

        for _ in range(int(10 * fps)):
            ret, frame = video.read()
            if ret:
                out.write(frame)
            else:
                keep_reading = False
                break
        out.release()