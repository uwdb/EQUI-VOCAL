import argparse
import os

import cv2
import swag
import tqdm
import numpy as np

# from blazeit.data.video_data import get_video_data
from blazeit.data.generate_fnames import get_csv_fname, get_video_fname

def frame_from_video(video):
    while video.isOpened():
        success, frame = video.read()
        if success:
            yield frame
        else:
            break

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='/home/ubuntu/CSE544-project/blazeit/data')
    parser.add_argument('--out_dir', default='/home/ubuntu/CSE544-project/blazeit/data/resol-65/')
    parser.add_argument('--base_name', required=True)
    parser.add_argument('--date', required=True)
    parser.add_argument('--max_frames', required=False, default=None, type=int)
    
    args = parser.parse_args()

    DATA_PATH = args.data_path
    base_name = args.base_name
    date = args.date
    RESOL = 65
    OUT_DIR = args.out_dir
    OUT_DIR = os.path.join(OUT_DIR, base_name)
    os.makedirs(OUT_DIR, exist_ok=True)
    OUT_FNAME = os.path.join(OUT_DIR, base_name + '-' + date + '.npy')

    # vd = get_video_data(base_name)
    file_base = '%s_video_list_%s' % (base_name, date)
    video_list_fname = os.path.join("/home/ubuntu/CSE544-project/data/bdd100k/", '%s.txt' % file_base)
    
    with open(video_list_fname, 'r') as f:
         video_list = f.read().splitlines()

    input_video_dir = "/home/ubuntu/CSE544-project/data/bdd100k/videos/test"
    nb_frames = 0
    for viedo_file in video_list:
        video = cv2.VideoCapture(os.path.join(input_video_dir, viedo_file))
        nb_frames += int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    if args.max_frames:
        nb_frames = min(nb_frames, args.max_frames)
    
    print(f"Reading {nb_frames} frames")
    
    data = np.zeros((nb_frames, 3, RESOL, RESOL), dtype=np.float32)

    idx = 0
    for viedo_file in video_list:
        video = cv2.VideoCapture(os.path.join(input_video_dir, viedo_file))
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        if not video.isOpened():
            print("Error opening video stream or file: ", viedo_file)
        else:
            frame_gen = frame_from_video(video)
            frame_list = []
            for frame in frame_gen:
                frame_list.append(frame)
            for frame in frame_list:
                frame = cv2.resize(frame, (RESOL, RESOL)).astype('float32')
                frame /= 255.
                frame[...,:] -= [0.485, 0.456, 0.406]
                frame[...,:] /= [0.229, 0.224, 0.225]
                frame = frame.transpose(2, 0, 1).copy()
                data[idx] = frame
                idx += 1

    np.save(OUT_FNAME, data)

if __name__ == '__main__':
    main()
