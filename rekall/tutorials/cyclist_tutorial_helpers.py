from rekall import Interval, IntervalSet, IntervalSetMapping, Bounds3D
from rekall.predicates import *
from vgrid import VGridSpec, VideoMetadata, VideoBlockFormat, FlatFormat, SpatialType_Bbox
from vgrid_jupyter import VGridWidget
import urllib3, requests, os, posixpath
import pickle
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

# Hack to disable warnings about olimar's certificate
urllib3.disable_warnings()

# Intel data
VIDEO_COLLECTION_BASEURL_INTEL = "https://storage.googleapis.com/esper/dan_olimar/rekall_tutorials/cydet" 
VIDEO_METADATA_FILENAME_INTEL = "metadata.json"
req = requests.get(posixpath.join(VIDEO_COLLECTION_BASEURL_INTEL, VIDEO_METADATA_FILENAME_INTEL), verify=False)
video_collection_intel = sorted(req.json(), key=lambda vm: vm['filename'])

maskrcnn_bbox_files_intel = [ 'maskrcnn_bboxes_0001.pkl', 'maskrcnn_bboxes_0004.pkl' ]

# maskrcnn_bboxes_intel: 4-dimensional 
#
# 1st: video instance. maskrcnn_bboxes_intel[0]: first video 
# 2nd: each frame. maskrcnn_bboxes_intel[0][0]: first frame of the first video.
# 3rd: all objects in that frame. maskrcnn_bboxes_intel[0][0][0]: first object in the first frame of the first video.
# 4th: [x1, x2, y1, y2, class, score, img_name]
# maskrcnn_bboxes_intel = []
# for bbox_file in maskrcnn_bbox_files_intel:
#     req = requests.get(posixpath.join(VIDEO_COLLECTION_BASEURL_INTEL, bbox_file), verify=False)
#     maskrcnn_bboxes_intel.append(pickle.loads(req.content))
maskrcnn_bboxes_intel = []
bbox_file_dir = "/home/ubuntu/CSE544-project/rekall/data/bbox_files/"
for file in os.listdir(bbox_file_dir):
    if os.path.splitext(file)[1] != '.pkl':
        continue
    with open(os.path.join(bbox_file_dir, file), 'rb') as f:
        maskrcnn_bboxes_intel.append(pickle.load(f))
        
video_metadata_intel = [
    VideoMetadata(v["filename"], v["id"], v["fps"], int(v["num_frames"]), v["width"], v["height"])
    for v in video_collection_intel
]

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
        ]),
        video_endpoint = VIDEO_COLLECTION_BASEURL_INTEL
    )
    return VGridWidget(vgrid_spec = vgrid_spec.to_json_compressed())
