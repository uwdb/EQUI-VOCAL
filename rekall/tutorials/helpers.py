from rekall import Interval, IntervalSet, IntervalSetMapping, Bounds3D
from rekall.predicates import *
from vgrid import VGridSpec, VideoMetadata, VideoBlockFormat, FlatFormat, SpatialType_Bbox
from vgrid_jupyter import VGridWidget
import urllib3, requests, os, posixpath
import pickle
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import cv2 

# Hack to disable warnings about olimar's certificate
urllib3.disable_warnings()

# Construct metadata
input_video_dir = "/home/ubuntu/CSE544-project/rekall/data/input_videos"
metadata = []
idx = 0
for file in os.listdir(input_video_dir):
    if os.path.splitext(file)[1] != '.mp4':
        continue
    video = cv2.VideoCapture(os.path.join(input_video_dir, file))
    print(os.path.join(input_video_dir, file))
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)
    metadata.append({
                        "num_frames": str(num_frames), 
                        "height": height, 
                        "width": width, 
                        "fps": fps, 
                        "filename": file, 
                        "id": idx
    })
    idx += 1

print(metadata)

video_metadata_intel = [
    VideoMetadata(os.path.join(input_video_dir, v["filename"]), v["id"], v["fps"], int(v["num_frames"]), v["width"], v["height"])
    for v in metadata
]

# Retrieve bboxes
#
# maskrcnn_bboxes_intel: 4-dimensional 
# 1st: video instance. maskrcnn_bboxes_intel[0]: first video 
# 2nd: each frame. maskrcnn_bboxes_intel[0][0]: first frame of the first video.
# 3rd: all objects in that frame. maskrcnn_bboxes_intel[0][0][0]: first object in the first frame of the first video.
# 4th: [x1, x2, y1, y2, class, score, img_name]
maskrcnn_bboxes_intel = []
bbox_file_dir = "/home/ubuntu/CSE544-project/rekall/data/bbox_files/"
for file in os.listdir(bbox_file_dir):
    if os.path.splitext(file)[1] != '.pkl':
        continue
    with open(os.path.join(bbox_file_dir, file), 'rb') as f:
        maskrcnn_bboxes_intel.append(pickle.load(f))

def get_maskrcnn_bboxes():
    maskrcnn_bboxes_ism = IntervalSetMapping({
        vm.id: IntervalSet([
            Interval(
                Bounds3D(
                    t1 = frame_num / vm.fps,
                    t2 = (frame_num + 1) / vm.fps,
                    x1 = bbox[0] / vm.width,
                    x2 = bbox[2] / vm.width,
                    y1 = bbox[1] / vm.height,
                    y2 = bbox[3] / vm.height
                ),
                payload = {
                    'class': bbox[4],
                    'score': bbox[5],
                    'spatial_type': SpatialType_Bbox(text=bbox[4])
                }
            )
            for frame_num, bboxes_in_frame in enumerate(maskrcnn_frame_list)
            for bbox in bboxes_in_frame
        ])
        for vm, maskrcnn_frame_list in zip(video_metadata_intel, maskrcnn_bboxes_intel)
    })
    
    return maskrcnn_bboxes_ism

def visualize_helper(box_list):
    vgrid_spec = VGridSpec(
        video_meta = video_metadata_intel,
        vis_format = VideoBlockFormat(imaps = [
            (str(i), box)
            for i, box in enumerate(box_list)
        ])
    )
    return VGridWidget(vgrid_spec = vgrid_spec.to_json())
