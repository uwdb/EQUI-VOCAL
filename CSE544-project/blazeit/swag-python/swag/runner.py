import os

import cv2
import numpy as np

from video_capture import VideoCapture

def main():
    # dir_name = '/lfs/1/ddkang/blazeit/data/svideo/jackson-town-square/2017-12-14'
    # index_fname = '/lfs/1/ddkang/blazeit/data/svideo/jackson-town-square/2017-12-14.json'
    dir_name = '/lfs/1/ddkang/blazeit/data/svideo/jackson-town-square/short'
    index_fname = dir_name + '.json'
    cap = VideoCapture(dir_name, index_fname)

    '''cap.set(cv2.CAP_PROP_POS_FRAMES, 300)
    ret, frame = cap.read()
    cap2 = cv2.VideoCapture(os.path.join(dir_name, '3.mp4'))
    _, f2 = cap2.read()'''
    i = 0
    ret, frame = cap.read()
    while ret:
      i += 1
      ret, frame = cap.read()
    print(i)

    print(np.array_equal(frame, f2))

if __name__ == '__main__':
    main()
