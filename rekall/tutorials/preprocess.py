# Some basic setup:
# Setup detectron2 logger
import torch, torchvision
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
from tqdm import tqdm
import pickle

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer


def frame_from_video(video):
    while video.isOpened():
        success, frame = video.read()
        if success:
            yield frame
        else:
            break

input_video_dir = "/home/ubuntu/CSE544-project/rekall/data/input_videos/"
bbox_file_dir = "/home/ubuntu/CSE544-project/rekall/data/bbox_files/"

with open("ms_coco_classnames.txt") as f:
    coco_names = f.read().splitlines() 

cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)

# im = cv2.imread("/home/ubuntu/CSE544-project/detectron2/demo/1.jpg")
# outputs = predictor(im)

# print(outputs["instances"].pred_classes)
# print(outputs["instances"].pred_boxes)
for file in os.listdir(input_video_dir):
    if os.path.splitext(file)[1] != '.mp4':
        continue
    video = cv2.VideoCapture(os.path.join(input_video_dir, file))
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    res_per_video = []
    frame_cnt = 0

    if not video.isOpened():
        print("Error opening video stream or file: ", file)
    else:
        frame_gen = frame_from_video(video)
        for frame in tqdm(frame_gen, total=num_frames):
            frame_cnt += 1 
            outputs = predictor(frame)
            instances = outputs["instances"]
            pred_boxes = instances.pred_boxes
            scores = instances.scores
            # pred_class is idx, not string; need coco_names[pred_class.item()] to access class name 
            pred_classes = instances.pred_classes
            res_per_frame = []
            for pred_box, score, pred_class in zip(pred_boxes, scores, pred_classes):
                res_per_frame.append([pred_box[0].item(), pred_box[1].item(), pred_box[2].item(), pred_box[3].item(), coco_names[pred_class.item()], score.item(), str(frame_cnt).zfill(10) + '.jpg'])
            res_per_video.append(res_per_frame)

    output_filename = os.path.splitext(file)[0] + '.pkl'

    with open(os.path.join(bbox_file_dir, output_filename), 'wb') as f:
        pickle.dump(res_per_video, f)