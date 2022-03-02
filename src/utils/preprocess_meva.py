# Some basic setup:
# Setup detectron2 logger
import torch, torchvision
from torchvision import datasets, transforms
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random, time
from tqdm import tqdm
import pickle
from glob import glob
from pathlib import Path

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
import sys
# setting path
sys.path.append('../')
from faster_r_cnn import FasterRCNN
from utils import frame_from_video
import tools
import argparse

class FasterRCNNPreprocess(FasterRCNN):
    def __init__(self):
        super().__init__()
        with open("/mmfs1/gscratch/balazinska/enhaoz/complex_event_video/src/ms_coco_classnames.txt") as f:
            self.coco_names = f.read().splitlines()
        self.bbox_info = {}

    def __call__(self, inputs, frame_ids):
        outputs = self.model(inputs)
        for output, frame_id in zip(outputs, frame_ids):
            res_per_frame = []
            instances = output["instances"]
            pred_boxes = instances.pred_boxes
            scores = instances.scores
            pred_classes = instances.pred_classes
            for pred_box, score, pred_class in zip(pred_boxes, scores, pred_classes):
                # List[(x1, y1, x2, y2, class_name, score)]
                res_per_frame.append([pred_box[0].item(), pred_box[1].item(), pred_box[2].item(), pred_box[3].item(), self.coco_names[pred_class.item()], score.item()])
            self.bbox_info[frame_id] = res_per_frame

    def get_bbox_info(self):
        return self.bbox_info

# Input: List[path_to_video_file]
@tools.tik_tok
def preprocess(model, video_dir):
    files = [y for x in os.walk(os.path.join("/mmfs1/gscratch/balazinska/enhaoz/complex_event_video/data/meva", video_dir)) for y in glob(os.path.join(x[0], '*.avi'))]
    # print("files", files)
    for file in files:
        print("file: ", file)
        if "school.G336.r13.avi" not in file:
        # if "school.G421.r13.avi" not in file:
            continue
        out_file = os.path.join(os.path.dirname(file), os.path.basename(file)[:-4] + ".json")
        if os.path.isfile(out_file):
            print("Video has already been detected. Skip.")
            continue
        # if file in ["/home/ubuntu/complex_event_video/data/meva/2018-03-15/16/2018-03-15.15-55-00.16-00-00.school.G421.r13.avi"]:
        #     continue
        video = cv2.VideoCapture(file)
        frame_id = 0
        inputs = []
        frame_ids = []
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            # Process frame
            img_transposed = np.transpose(frame, (2, 0, 1)) # From (H, W, C) to (C, H, W)
            img_tensor = torch.from_numpy(img_transposed)
            inputs.append({"image":img_tensor}) # inputs is ready
            frame_ids.append(frame_id)
            if len(inputs) < 4:
                frame_id += 1
                continue
            print("frame id:", frame_id)
            model(inputs, frame_ids)
            inputs = []
            frame_ids = []
            frame_id += 1
        video.release()
        cv2.destroyAllWindows() # destroy all opened windows

        # Write bbox information to file
        bbox_info = model.get_bbox_info()
        with open(out_file, 'w') as f:
            f.write(json.dumps(bbox_info))

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--video-dir', type=str)
    args = ap.parse_args()
    print(args.video_dir)
    model = FasterRCNNPreprocess()
    preprocess(model, args.video_dir)