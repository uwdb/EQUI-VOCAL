import os
import numpy as np
import json
import argparse
from pprint import pprint
import pycocotools._mask as _mask
import cv2
from torchvision.ops import masks_to_boxes
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--idx_video', type=int, default=12)
parser.add_argument('--idx_frame', type=int, default=95)
parser.add_argument('--read_src', default='/mmfs1/gscratch/balazinska/enhaoz/complex_event_video/data/clevrer/processed_proposals')

args = parser.parse_args()


def decode(rleObjs):
    if type(rleObjs) == list:
        return _mask.decode(rleObjs)
    else:
        return _mask.decode([rleObjs])[:,:,0]


with open(os.path.join(args.read_src, 'sim_%05d.json' % args.idx_video)) as f:
    data = json.load(f)

# pprint(data)

print(data['video_name'])
frame = data['frames'][args.idx_frame]
print('frame_name', frame['frame_filename'])
print('frame_index', frame['frame_index'])
objects = frame['objects']
print(len(objects))

cap = cv2.VideoCapture("/mmfs1/gscratch/balazinska/enhaoz/complex_event_video/data/clevrer/video_train/video_00000-01000/video_00012.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 95)
ret, image = cap.read()

for i in range(len(objects)):
    print(objects[i]['material'], objects[i]['color'], objects[i]['shape'])
    mask = decode(objects[i]['mask'])
    # O represents black, 1 represents white.
    # print(np.sum(mask))
    box = masks_to_boxes(torch.from_numpy(mask[np.newaxis, :]))
    box = np.squeeze(box.numpy(), axis=0)

    res = cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0,255,0), 1)
    # cv2.imwrite('mask_%d.png' % i, res)


