# Some basic setup:
# Setup detectron2 logger
import torch, torchvision
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random, time 
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

def preprocess_faster_rcnn():
    display_video_list = ["cabc30fc-e7726578", "cabc30fc-eb673c5a", "cabc30fc-fd79926f", "cabc9045-1b8282ba", "cabc9045-5a50690f"]

    total_time_start = time.time()

    # input_video_dir = "/home/ubuntu/CSE544-project/rekall/data/input_videos/"
    input_video_dir = "/home/ubuntu/CSE544-project/data/bdd100k/videos/test/"
    # bbox_file_dir = "/home/ubuntu/CSE544-project/rekall/data/bbox_files/"
    bbox_file_dir = "/home/ubuntu/CSE544-project/data/bdd100k/bbox_files_skip5"

    with open("ms_coco_classnames.txt") as f:
        coco_names = f.read().splitlines() 

    tik = time.time()
    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)
    model_loading_time = time.time() - tik 

    # im = cv2.imread("/home/ubuntu/CSE544-project/detectron2/demo/1.jpg")
    # outputs = predictor(im)

    # print(outputs["instances"].pred_classes)
    # print(outputs["instances"].pred_boxes)

    model_execution_time = 0
    video_loading_time = 0 
    bbox_file_construct_time = 0 

    for file in os.listdir(input_video_dir):
        if os.path.splitext(file)[1] != '.mp4' and os.path.splitext(file)[1] != '.mov':
            continue
        if os.path.splitext(file)[0] not in display_video_list:
            continue
        video_tik = time.time()
        video = cv2.VideoCapture(os.path.join(input_video_dir, file))
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        if not video.isOpened():
            print("Error opening video stream or file: ", file)
        else:
            frame_gen = frame_from_video(video)
            video_loading_time += (time.time() - video_tik)
            res_per_video = [[] for _ in range(len(frame_gen))]
            frame_detection_list = [-1 for _ in range(len(frame_gen))] # -1: hasn't invoked model; 0: doesn't contain object of interest; 1: contains object of interest
            base_step = 30
            while base_step > 0:
                for i in range(0, len(frame_gen), base_step):
                    outputs = predictor(frame_gen[i])
                    instances = outputs["instances"]
                    pred_boxes = instances.pred_boxes
                    scores = instances.scores
                    pred_classes = instances.pred_classes
                    res_per_frame = []
                    has_truck = False
                    
                    for pred_class in pred_classes:
                        if coco_names[pred_class.item()] =  "truck":
                            has_truck = True
                            break
                    
                    if has_truck:
                        frame_detection_list[i] = 1
                        for pred_box, score, pred_class in zip(pred_boxes, scores, pred_classes):
                            res_per_frame.append([pred_box[0].item(), pred_box[1].item(), pred_box[2].item(), pred_box[3].item(), coco_names[pred_class.item()], score.item(), str(i + 1).zfill(10) + '.jpg'])
                        res_per_video[i] = res_per_frame
                    else:
                        frame_detection_list[i] = 0
                
                # If two frames within 30 frames apart don't contain object of interest, then we can mark any frames in between as 0 as well. 
                for i in range(len(frame_detection_list))
                base_step /= 2



            for frame in tqdm(frame_gen, total=num_frames):
                frame_cnt += 1 
                if frame_cnt % 5 == 0:
                    continue
                tik = time.time()
                outputs = predictor(frame)
                model_execution_time += (time.time() - tik)
                bbox_tik = time.time()
                instances = outputs["instances"]
                pred_boxes = instances.pred_boxes
                scores = instances.scores
                # pred_class is idx, not string; need coco_names[pred_class.item()] to access class name 
                pred_classes = instances.pred_classes
                res_per_frame = []
                for pred_box, score, pred_class in zip(pred_boxes, scores, pred_classes):
                    res_per_frame.append([pred_box[0].item(), pred_box[1].item(), pred_box[2].item(), pred_box[3].item(), coco_names[pred_class.item()], score.item(), str(frame_cnt).zfill(10) + '.jpg'])

                res_per_video.append(res_per_frame)
                bbox_file_construct_time += (time.time() - bbox_tik)

        bbox_tik = time.time()
        
        output_filename = os.path.splitext(file)[0] + '.pkl'

        with open(os.path.join(bbox_file_dir, output_filename), 'wb') as f:
            pickle.dump(res_per_video, f)

        bbox_file_construct_time += (time.time() - bbox_tik)

    print("model loading time: ", model_loading_time)
    print("model execution time: ", model_execution_time)
    print("video loading time: ", video_loading_time)
    print("bbox file construct time: ", bbox_file_construct_time)
    print("total time: ", time.time() - total_time_start)


def preprocess_binary_search():
    display_video_list = ["cabc30fc-e7726578", "cabc30fc-eb673c5a", "cabc30fc-fd79926f", "cabc9045-1b8282ba", "cabc9045-5a50690f"]

    total_time_start = time.time()

    # input_video_dir = "/home/ubuntu/CSE544-project/rekall/data/input_videos/"
    input_video_dir = "/home/ubuntu/CSE544-project/data/bdd100k/videos/test/"
    # bbox_file_dir = "/home/ubuntu/CSE544-project/rekall/data/bbox_files/"
    bbox_file_dir = "/home/ubuntu/CSE544-project/data/bdd100k/bbox_files_skip5"

    with open("ms_coco_classnames.txt") as f:
        coco_names = f.read().splitlines() 

    tik = time.time()
    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)
    model_loading_time = time.time() - tik 

    # im = cv2.imread("/home/ubuntu/CSE544-project/detectron2/demo/1.jpg")
    # outputs = predictor(im)

    # print(outputs["instances"].pred_classes)
    # print(outputs["instances"].pred_boxes)

    model_execution_time = 0
    video_loading_time = 0 
    bbox_file_construct_time = 0 

    for file in os.listdir(input_video_dir):
        if os.path.splitext(file)[1] != '.mp4' and os.path.splitext(file)[1] != '.mov':
            continue
        if os.path.splitext(file)[0] not in display_video_list:
            continue
        video_tik = time.time()
        video = cv2.VideoCapture(os.path.join(input_video_dir, file))
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        res_per_video = []
        frame_cnt = 0

        if not video.isOpened():
            print("Error opening video stream or file: ", file)
        else:
            frame_gen = frame_from_video(video)
            video_loading_time += (time.time() - video_tik)
            for frame in tqdm(frame_gen, total=num_frames):
                frame_cnt += 1 
                if frame_cnt % 5 == 0:
                    continue
                tik = time.time()
                outputs = predictor(frame)
                model_execution_time += (time.time() - tik)
                bbox_tik = time.time()
                instances = outputs["instances"]
                pred_boxes = instances.pred_boxes
                scores = instances.scores
                # pred_class is idx, not string; need coco_names[pred_class.item()] to access class name 
                pred_classes = instances.pred_classes
                res_per_frame = []
                for pred_box, score, pred_class in zip(pred_boxes, scores, pred_classes):
                    res_per_frame.append([pred_box[0].item(), pred_box[1].item(), pred_box[2].item(), pred_box[3].item(), coco_names[pred_class.item()], score.item(), str(frame_cnt).zfill(10) + '.jpg'])

                res_per_video.append(res_per_frame)
                bbox_file_construct_time += (time.time() - bbox_tik)

        bbox_tik = time.time()
        
        output_filename = os.path.splitext(file)[0] + '.pkl'

        with open(os.path.join(bbox_file_dir, output_filename), 'wb') as f:
            pickle.dump(res_per_video, f)

        bbox_file_construct_time += (time.time() - bbox_tik)

    print("model loading time: ", model_loading_time)
    print("model execution time: ", model_execution_time)
    print("video loading time: ", video_loading_time)
    print("bbox file construct time: ", bbox_file_construct_time)
    print("total time: ", time.time() - total_time_start)


if __name__ == '__main__':
    preprocess_faster_rcnn()
    

# Stats
"""
fps 15, 5 videos:
model loading time:  3.9729175567626953
model execution time:  364.991281747818
video loading time:  0.04999542236328125
bbox file construct time:  3.359412908554077
total time:  390.0788278579712

skip 1 frame for every 5 frames, 5 videos:
model loading time:  4.0205676555633545
model execution time:  585.3846228122711
video loading time:  0.05267810821533203
bbox file construct time:  5.278503894805908
total time:  614.2306888103485
"""
