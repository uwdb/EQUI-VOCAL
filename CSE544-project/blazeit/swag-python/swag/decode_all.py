import argparse
import os

import cv2
import numpy as np

from video_capture import VideoCapture

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', required=True)
    parser.add_argument('--index_fname', required=True)
    parser.add_argument('--npy_fname', required=True)
    parser.add_argument('--resol', type=int, required=True)
    parser.add_argument('--base_name', default=None)
    args = parser.parse_args()

    vid_dir = args.video_dir
    index_fname = args.index_fname
    npy_fname = args.npy_fname
    resol = args.resol
    if args.base_name is None:
        crop_and_exclude = False
    else:
        crop_and_exclude = True
        from blazeit.data.video_data import get_video_data
        video_data = get_video_data(args.base_name)

    cap = VideoCapture(vid_dir, index_fname)
    data = np.zeros((cap.cum_frames[-1], resol, resol, 3), dtype=np.float32)
    for i in range(len(data)):
        if i % 1000 == 0:
            print('Processing frame', i)
        ret, frame = cap.read()
        if not ret:
            print('Something really bad happened')
            sys.exit(1)
        if crop_and_exclude:
            frame = video_data.process_frame(frame)
        data[i] = cv2.resize(frame, (resol, resol))

    data /= 255.
    data[...,:] -= [0.485, 0.456, 0.406]
    data[...,:] /= [0.229, 0.224, 0.225]

    np.save(npy_fname, data)

if __name__ == '__main__':
    main()
