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

def preprocess_binary_search_rare_first():
    display_video_list = ["cabc30fc-e7726578", "cabc30fc-eb673c5a", "cabc30fc-fd79926f", "cabc9045-1b8282ba", "cabc9045-5a50690f"]
    total_invokation = 0
    total_time_start = time.time()

    # input_video_dir = "/home/ubuntu/CSE544-project/rekall/data/input_videos/"
    input_video_dir = "/home/ubuntu/CSE544-project/data/bdd100k/videos/test/"
    # bbox_file_dir = "/home/ubuntu/CSE544-project/rekall/data/bbox_files/"
    bbox_file_dir = "/home/ubuntu/CSE544-project/data/bdd100k/bbox_files_binary_search"

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
            frame_list = []
            for frame in frame_gen:
                frame_list.append(frame)
            video_loading_time += (time.time() - video_tik)
            res_per_video = [[] for _ in range(len(frame_list))]
            frame_detection_list = [-1 for _ in range(len(frame_list))] # -1: hasn't invoked model; 0: doesn't contain object; 1: contains object of interest; 2: doesn't contain object of interest
            base_step = 30
            while base_step > 0:
                print("base_step: ", base_step)
                for i in range(0, len(frame_list), base_step):
                    if frame_detection_list[i] != -1:
                        continue
                    tik = time.time()
                    outputs = predictor(frame_list[i])
                    total_invokation += 1
                    model_execution_time += (time.time() - tik)
                    instances = outputs["instances"]
                    pred_boxes = instances.pred_boxes
                    scores = instances.scores
                    pred_classes = instances.pred_classes
                    res_per_frame = []
                    has_truck = False
                    
                    for pred_class in pred_classes:
                        if coco_names[pred_class.item()] == "truck":
                            has_truck = True
                            break
                    
                    if has_truck:
                        frame_detection_list[i] = 1
                    else:
                        frame_detection_list[i] = 0
                    
                    for pred_box, score, pred_class in zip(pred_boxes, scores, pred_classes):
                        res_per_frame.append([pred_box[0].item(), pred_box[1].item(), pred_box[2].item(), pred_box[3].item(), coco_names[pred_class.item()], score.item(), str(i + 1).zfill(10) + '.jpg'])
                    res_per_video[i] = res_per_frame
                
                # If two frames within 30 frames apart don't contain object of interest, then we can mark any frames in between as 0 as well. 
                for i in range(0, len(frame_detection_list) - base_step, base_step):
                    if frame_detection_list[i] % 2 == 0 and frame_detection_list[i+base_step] % 2 == 0 and frame_detection_list[i+1] % 2 != 0:
                        for j in range(i+1, i+base_step):
                            frame_detection_list[j] = 2
                base_step = int(base_step / 2)

            # Done for truck. Next, we still need to run model on frames that haven't been detected but may contain cars, but only those within the window constraint.
            # print(frame_detection_list)
            print("Phase 2...")
            for idx_car in range(len(frame_detection_list)):
            # while idx_car < len(frame_detection_list):
                if frame_detection_list[idx_car] == 1:
                    for j in range(idx_car+1, min(idx_car+301, len(frame_detection_list))): 
                        if frame_detection_list[j] == 2:
                            tik = time.time()
                            outputs = predictor(frame_list[j])
                            total_invokation += 1
                            frame_detection_list[j] = 0
                            model_execution_time += (time.time() - tik)
                            instances = outputs["instances"]
                            pred_boxes = instances.pred_boxes
                            scores = instances.scores
                            pred_classes = instances.pred_classes
                            res_per_frame = []
                            
                            for pred_box, score, pred_class in zip(pred_boxes, scores, pred_classes):
                                res_per_frame.append([pred_box[0].item(), pred_box[1].item(), pred_box[2].item(), pred_box[3].item(), coco_names[pred_class.item()], score.item(), str(j + 1).zfill(10) + '.jpg'])
                            res_per_video[j] = res_per_frame

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
    print("total_invokation: ", total_invokation)


def preprocess_binary_search_rare_last():
    display_video_list = ["cabc30fc-e7726578", "cabc30fc-eb673c5a", "cabc30fc-fd79926f", "cabc9045-1b8282ba", "cabc9045-5a50690f"]

    total_invokation = 0
    total_time_start = time.time()

    # input_video_dir = "/home/ubuntu/CSE544-project/rekall/data/input_videos/"
    input_video_dir = "/home/ubuntu/CSE544-project/data/bdd100k/videos/test/"
    # bbox_file_dir = "/home/ubuntu/CSE544-project/rekall/data/bbox_files/"
    bbox_file_dir = "/home/ubuntu/CSE544-project/data/bdd100k/bbox_files_binary_search_rare_last"

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
            frame_list = []
            for frame in frame_gen:
                frame_list.append(frame)
            video_loading_time += (time.time() - video_tik)
            res_per_video = [[] for _ in range(len(frame_list))]
            frame_detection_list = [-1 for _ in range(len(frame_list))] # -1: hasn't invoked model; 0: doesn't contain object; 1: contains object of interest; 2: doesn't contain object of interest
            base_step = 30
            while base_step > 0:
                print("base_step: ", base_step)
                for i in range(0, len(frame_list), base_step):
                    if frame_detection_list[i] != -1:
                        continue
                    tik = time.time()
                    outputs = predictor(frame_list[i])
                    total_invokation += 1
                    model_execution_time += (time.time() - tik)
                    instances = outputs["instances"]
                    pred_boxes = instances.pred_boxes
                    scores = instances.scores
                    pred_classes = instances.pred_classes
                    res_per_frame = []
                    has_car = False
                    
                    for pred_class in pred_classes:
                        if coco_names[pred_class.item()] == "car":
                            has_car = True
                            break
                    
                    if has_car:
                        frame_detection_list[i] = 1
                    else:
                        frame_detection_list[i] = 0
                    
                    for pred_box, score, pred_class in zip(pred_boxes, scores, pred_classes):
                        res_per_frame.append([pred_box[0].item(), pred_box[1].item(), pred_box[2].item(), pred_box[3].item(), coco_names[pred_class.item()], score.item(), str(i + 1).zfill(10) + '.jpg'])
                    res_per_video[i] = res_per_frame
                
                # If two frames within 30 frames apart don't contain object of interest, then we can mark any frames in between as 0 as well. 
                for i in range(0, len(frame_detection_list) - base_step, base_step):
                    if frame_detection_list[i] % 2 == 0 and frame_detection_list[i+base_step] % 2 == 0 and frame_detection_list[i+1] % 2 != 0:
                        for j in range(i+1, i+base_step):
                            frame_detection_list[j] = 2
                base_step = int(base_step / 2)

            # Done for truck. Next, we still need to run model on frames that haven't been detected but may contain cars, but only those within the window constraint.
            # print(frame_detection_list)
            print("Phase 2...")
            for idx_car in range(len(frame_detection_list)):
            # while idx_car < len(frame_detection_list):
                if frame_detection_list[idx_car] == 1:
                    # for j in range(idx_car+1, min(idx_car+301, len(frame_detection_list))): 
                    for j in range(max(0, idx_car-300), idx_car): 
                        if frame_detection_list[j] == 2:
                            tik = time.time()
                            outputs = predictor(frame_list[j])
                            total_invokation += 1
                            frame_detection_list[j] = 0
                            model_execution_time += (time.time() - tik)
                            instances = outputs["instances"]
                            pred_boxes = instances.pred_boxes
                            scores = instances.scores
                            pred_classes = instances.pred_classes
                            res_per_frame = []
                            
                            for pred_box, score, pred_class in zip(pred_boxes, scores, pred_classes):
                                res_per_frame.append([pred_box[0].item(), pred_box[1].item(), pred_box[2].item(), pred_box[3].item(), coco_names[pred_class.item()], score.item(), str(j + 1).zfill(10) + '.jpg'])
                            res_per_video[j] = res_per_frame

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
    print("total_invokation: ", total_invokation)


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
        res_per_video = []
        frame_cnt = 0

        if not video.isOpened():
            print("Error opening video stream or file: ", file)
        else:
            frame_gen = frame_from_video(video)
            video_loading_time += (time.time() - video_tik)
            for frame in tqdm(frame_gen, total=num_frames):
                frame_cnt += 1 
                if frame_cnt % 10 == 0:
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
    # preprocess_faster_rcnn()
    preprocess_binary_search_rare_first()
    preprocess_binary_search_rare_last()
    

# Stats
"""
skip 1 frame for every 2 frames, 5 videos:
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

skip 1 frame for every 10 frames, 5 videos:
model loading time:  3.9522593021392822
model execution time:  653.4706573486328
video loading time:  0.05283379554748535
bbox file construct time:  6.196653127670288
total time:  683.4617691040039

binary search + rare first, 5 videos:
model loading time:  3.9572229385375977
model execution time:  370.66692757606506
video loading time:  22.426469326019287
bbox file construct time:  0.014455318450927734
total time:  399.9164888858795
total_invokation:  3059

binary search + rare last, 5 videos:
model loading time:  3.990025043487549
model execution time:  724.6807322502136
video loading time:  17.228809356689453
bbox file construct time:  0.03544926643371582
total time:  752.4066162109375
total_invokation:  6027
"""
