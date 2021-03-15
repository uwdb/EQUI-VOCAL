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

def preprocess_binary_search_rare_first_q2():
    display_video_list = ["cabc30fc-e7726578", "cabc30fc-eb673c5a", "cabc30fc-fd79926f", "cabc9045-1b8282ba", "cabc9045-5a50690f"]
    total_invokation_stage1 = 0
    total_invokation_stage2 = 0
    total_time_start = time.time()

    # input_video_dir = "/home/ubuntu/CSE544-project/rekall/data/input_videos/"
    input_video_dir = "/home/ubuntu/CSE544-project/data/bdd100k/videos/test/"
    # bbox_file_dir = "/home/ubuntu/CSE544-project/rekall/data/bbox_files/"
    bbox_file_dir = "/home/ubuntu/CSE544-project/data/bdd100k/bbox_files_binary_search_gt20_first_q2"

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
            frame_detection_list = [-1 for _ in range(len(frame_list))] # -1: hasn't invoked model; 0: doesn't contain event cars_gt_20; 1: contains event cars_gt_20; 2: skip model execution 
            base_step = 30
            while base_step > 0:
                print("base_step: ", base_step)
                for i in range(0, len(frame_list), base_step):
                    if frame_detection_list[i] != -1:
                        continue
                    tik = time.time()
                    outputs = predictor(frame_list[i])
                    total_invokation_stage1 += 1
                    model_execution_time += (time.time() - tik)
                    instances = outputs["instances"]
                    pred_boxes = instances.pred_boxes
                    scores = instances.scores
                    pred_classes = instances.pred_classes
                    res_per_frame = []
                    car_count = 0
                    
                    for pred_class in pred_classes:
                        if coco_names[pred_class.item()] == "car":
                            car_count += 1
                    
                    if car_count > 6:
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

            # Done for gt_20. Next, we still need to run model on frames that haven't been detected but may contain cars, but only those within the window constraint.
            # print(frame_detection_list)
            print("Phase 2...")
            # for idx_car in range(len(frame_detection_list)):
            idx_car = 0
            while idx_car < len(frame_detection_list):
                if frame_detection_list[idx_car] == 1:
                    event_a_count = 1
                    event_a_start = idx_car 
                    while idx_car+event_a_count < len(frame_detection_list) and frame_detection_list[idx_car+event_a_count] == 1:
                        event_a_count += 1
                    if event_a_count >= 30:
                        for j in range(idx_car+event_a_count, min(int(idx_car+9.5*30+1), len(frame_detection_list))): 
                            if frame_detection_list[j] == 2:
                                tik = time.time()
                                outputs = predictor(frame_list[j])
                                total_invokation_stage2 += 1
                                frame_detection_list[j] = 0
                                model_execution_time += (time.time() - tik)
                                instances = outputs["instances"]
                                pred_boxes = instances.pred_boxes
                                scores = instances.scores
                                pred_classes = instances.pred_classes
                                res_per_frame = []
                                
                                for pred_box, score, pred_class in zip(pred_boxes, scores, pred_classes):
                                    res_per_frame.append([pred_box[0].item(), pred_box[1].item(), pred_box[2].item(), pred_box[3].item(), coco_names[pred_class.item()], score.item(), str(j + 1).zfill(10) + '.jpg'])
                                
                                if not res_per_frame:
                                    res_per_frame.append([0, 1, 0, 1, "car", 0.99, str(j + 1).zfill(10) + '.jpg'])
                                res_per_video[j] = res_per_frame

                        for j in range(max(0, int(idx_car-9.5*30+event_a_count)), idx_car): 
                            if frame_detection_list[j] == 2:
                                tik = time.time()
                                outputs = predictor(frame_list[j])
                                total_invokation_stage2 += 1
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

                                if not res_per_frame:
                                    res_per_frame.append([0, 1, 0, 1, "car", 0.99, str(j + 1).zfill(10) + '.jpg'])
                                res_per_video[j] = res_per_frame
                    idx_car += event_a_count
                
                idx_car += 1

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
    print("total_invokation: ", total_invokation_stage1, total_invokation_stage2, total_invokation_stage1+total_invokation_stage2)


def preprocess_binary_search_rare_last_q2():
    display_video_list = ["cabc30fc-e7726578", "cabc30fc-eb673c5a", "cabc30fc-fd79926f", "cabc9045-1b8282ba", "cabc9045-5a50690f"]
    total_invokation_stage1 = 0
    total_invokation_stage2 = 0
    total_time_start = time.time()

    # input_video_dir = "/home/ubuntu/CSE544-project/rekall/data/input_videos/"
    input_video_dir = "/home/ubuntu/CSE544-project/data/bdd100k/videos/test/"
    # bbox_file_dir = "/home/ubuntu/CSE544-project/rekall/data/bbox_files/"
    bbox_file_dir = "/home/ubuntu/CSE544-project/data/bdd100k/bbox_files_binary_search_gt20_last_q2"

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
            frame_detection_list = [-1 for _ in range(len(frame_list))] # -1: hasn't invoked model; 0: doesn't contain event cars_gt_20; 1: contains event cars_gt_20; 2: skip model execution 
            base_step = 15
            while base_step > 0:
                print("base_step: ", base_step)
                for i in range(0, len(frame_list), base_step):
                    if frame_detection_list[i] != -1:
                        continue
                    tik = time.time()
                    outputs = predictor(frame_list[i])
                    total_invokation_stage1 += 1
                    model_execution_time += (time.time() - tik)
                    instances = outputs["instances"]
                    pred_boxes = instances.pred_boxes
                    scores = instances.scores
                    pred_classes = instances.pred_classes
                    res_per_frame = []
                    car_count = 0
                    
                    for pred_class in pred_classes:
                        if coco_names[pred_class.item()] == "car":
                            car_count += 1
                    
                    if car_count < 5:
                        frame_detection_list[i] = 1
                    else:
                        frame_detection_list[i] = 0
                    
                    for pred_box, score, pred_class in zip(pred_boxes, scores, pred_classes):
                        res_per_frame.append([pred_box[0].item(), pred_box[1].item(), pred_box[2].item(), pred_box[3].item(), coco_names[pred_class.item()], score.item(), str(i + 1).zfill(10) + '.jpg'])

                    if not res_per_frame:
                                    res_per_frame.append([0, 1, 0, 1, "car", 0.99, str(j + 1).zfill(10) + '.jpg'])
                    res_per_video[i] = res_per_frame
                
                # If two frames within 30 frames apart don't contain object of interest, then we can mark any frames in between as 0 as well. 
                for i in range(0, len(frame_detection_list) - base_step, base_step):
                    if frame_detection_list[i] % 2 == 0 and frame_detection_list[i+base_step] % 2 == 0 and frame_detection_list[i+1] % 2 != 0:
                        for j in range(i+1, i+base_step):
                            frame_detection_list[j] = 2
                base_step = int(base_step / 2)

            # Done for gt_20. Next, we still need to run model on frames that haven't been detected but may contain cars, but only those within the window constraint.
            # print(frame_detection_list)
            print("Phase 2...")
            # for idx_car in range(len(frame_detection_list)):
            idx_car = 0
            while idx_car < len(frame_detection_list):
                if frame_detection_list[idx_car] == 1:
                    event_a_count = 1
                    event_a_start = idx_car 
                    while idx_car+event_a_count < len(frame_detection_list) and frame_detection_list[idx_car+event_a_count] == 1:
                        event_a_count += 1
                    if event_a_count >= 15:
                        gap = 0
                        event_c_count = 0
                        event_c_start = len(frame_detection_list)
                        for i in range(1, min(10*30 - event_a_count, len(frame_detection_list)-idx_car-event_a_count)):
                            if frame_detection_list[idx_car+event_a_count+i] == 1:
                                event_c_count = 1
                                event_c_start = idx_car+event_a_count+i
                                break
                        
                        while event_c_start+event_c_count < len(frame_detection_list) and frame_detection_list[event_c_start+event_c_count] == 1:
                            event_c_count += 1
                        if event_c_count >= 15 and event_c_start - event_a_start <= 300:
                            # fill out event b in between
                            for j in range(idx_car+event_a_count, event_c_start):
                                if frame_detection_list[j] == 2:
                                    tik = time.time()
                                    outputs = predictor(frame_list[j])
                                    total_invokation_stage2 += 1
                                    frame_detection_list[j] = 0
                                    model_execution_time += (time.time() - tik)
                                    instances = outputs["instances"]
                                    pred_boxes = instances.pred_boxes
                                    scores = instances.scores
                                    pred_classes = instances.pred_classes
                                    res_per_frame = []
                                    
                                    for pred_box, score, pred_class in zip(pred_boxes, scores, pred_classes):
                                        res_per_frame.append([pred_box[0].item(), pred_box[1].item(), pred_box[2].item(), pred_box[3].item(), coco_names[pred_class.item()], score.item(), str(j + 1).zfill(10) + '.jpg'])
                                    
                                    if not res_per_frame:
                                        res_per_frame.append([0, 1, 0, 1, "car", 0.99, str(j + 1).zfill(10) + '.jpg'])
                                    res_per_video[j] = res_per_frame

                    idx_car += event_a_count
                
                idx_car += 1

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
    print("total_invokation: ", total_invokation_stage1, total_invokation_stage2, total_invokation_stage1+total_invokation_stage2)


def preprocess_faster_rcnn():
    display_video_list = ["cabc30fc-e7726578", "cabc30fc-eb673c5a", "cabc30fc-fd79926f", "cabc9045-1b8282ba", "cabc9045-5a50690f"]
    car_count = 0
    truck_count = 0
    lt5_count = 0
    gt6_count = 0
    total_time_start = time.time()
    total_invokation = 0
    # input_video_dir = "/home/ubuntu/CSE544-project/rekall/data/input_videos/"
    input_video_dir = "/home/ubuntu/CSE544-project/data/bdd100k/videos/test/"
    # bbox_file_dir = "/home/ubuntu/CSE544-project/rekall/data/bbox_files/"
    bbox_file_dir = "/home/ubuntu/CSE544-project/data/bdd100k/bbox_files_original"

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
                # if frame_cnt % 10 == 0:
                #     continue
                tik = time.time()
                outputs = predictor(frame)
                model_execution_time += (time.time() - tik)
                total_invokation += 1
                bbox_tik = time.time()
                instances = outputs["instances"]
                pred_boxes = instances.pred_boxes
                scores = instances.scores
                # pred_class is idx, not string; need coco_names[pred_class.item()] to access class name 
                pred_classes = instances.pred_classes
                res_per_frame = []
                for pred_box, score, pred_class in zip(pred_boxes, scores, pred_classes):
                    res_per_frame.append([pred_box[0].item(), pred_box[1].item(), pred_box[2].item(), pred_box[3].item(), coco_names[pred_class.item()], score.item(), str(frame_cnt).zfill(10) + '.jpg'])

                local_count = 0
                if res_per_frame:
                    for pred_class in pred_classes:
                        if coco_names[pred_class.item()] == "car":
                            local_count += 1

                    if local_count > 0:
                        car_count += 1
                    if local_count > 6:
                        gt6_count += 1
                    if local_count < 5:
                        lt5_count += 1
                    
                    for pred_class in pred_classes:
                        if coco_names[pred_class.item()] == "truck":
                            truck_count += 1
                            break
                else:
                    # Nothing in this frame 
                    lt5_count += 1
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
    print("total_invokation: ", total_invokation)
    print(car_count, truck_count, lt5_count, gt6_count)


def preprocess_binary_search_rare_first_clean_version():
    global frame_detection_list, res_per_video, model_execution_time, total_invokation_stage1, total_invokation_stage2, frame_detection_list2
    total_invokation_stage1 = 0
    total_invokation_stage2 = 0
    model_execution_time = 0
    video_loading_time = 0 
    bbox_file_construct_time = 0 

    def model_a(frame):
        global model_execution_time, total_invokation_stage1
        tik = time.time()
        outputs = predictor(frame)
        model_execution_time += (time.time() - tik)
        total_invokation_stage1 += 1
        instances = outputs["instances"]
        pred_boxes = instances.pred_boxes
        scores = instances.scores
        pred_classes = instances.pred_classes            
        for pred_class in pred_classes:
            if coco_names[pred_class.item()] == "truck":
                return True, pred_boxes, scores, pred_classes
        return False, pred_boxes, scores, pred_classes

    def model_b(frame):
        global model_execution_time, total_invokation_stage2
        tik = time.time()
        outputs = predictor(frame)
        model_execution_time += (time.time() - tik)
        total_invokation_stage2 += 1
        instances = outputs["instances"]
        pred_boxes = instances.pred_boxes
        scores = instances.scores
        pred_classes = instances.pred_classes            
        for pred_class in pred_classes:
            if coco_names[pred_class.item()] == "car":
                return True, pred_boxes, scores, pred_classes
        return False, pred_boxes, scores, pred_classes

                
    def binary_search(start_frame, end_frame, ML_model, min_duration, event_a=True):
        global frame_detection_list, res_per_video, frame_detection_list2
        if min_duration == 0: 
            return 
        for i in range(start_frame, end_frame, min_duration):
            if (event_a and frame_detection_list[i][0] != -1) or (event_a==False and frame_detection_list2[i][0] != -1):
                continue
            prediction, pred_boxes, scores, pred_classes = ML_model(frame_list[i])
            
            # Update F
            if event_a:
                frame_detection_list[i] = [int(prediction == True), pred_classes]
            else:
                frame_detection_list2[i] = [int(prediction == True), pred_classes]
        
            # Write results to bboxes list 
            res_per_frame = []
            for pred_box, score, pred_class in zip(pred_boxes, scores, pred_classes):
                res_per_frame.append([pred_box[0].item(), pred_box[1].item(), pred_box[2].item(), pred_box[3].item(), coco_names[pred_class.item()], score.item(), str(i + 1).zfill(10) + '.jpg'])
            res_per_video[i] = res_per_frame
        
        # This pass is done.
        # Update F by examining if there are any frames that can be skipped.
        for i in range(start_frame, end_frame - min_duration, min_duration):
            if event_a:
                if frame_detection_list[i][0] % 2 == 0 and frame_detection_list[i+min_duration][0] % 2 == 0 and frame_detection_list[i+1][0] % 2 != 0:
                    for j in range(i+1, i+min_duration):
                        if frame_detection_list[j][0] == -1:
                            frame_detection_list[j] = [2, []]
            else:
                if frame_detection_list2[i][0] % 2 == 0 and frame_detection_list2[i+min_duration][0] % 2 == 0 and frame_detection_list2[i+1][0] % 2 != 0:
                    for j in range(i+1, i+min_duration):
                        if frame_detection_list2[j][0] == -1:
                            frame_detection_list2[j] = [2, []]
        # Go to the next level of recursion.
        binary_search(start_frame, end_frame, ML_model, int(min_duration / 2), event_a)
        
    display_video_list = ["cabc30fc-e7726578", "cabc30fc-eb673c5a", "cabc30fc-fd79926f", "cabc9045-1b8282ba", "cabc9045-5a50690f"]
    
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

    

    for file in os.listdir(input_video_dir):
        if os.path.splitext(file)[1] != '.mp4' and os.path.splitext(file)[1] != '.mov':
            continue
        if os.path.splitext(file)[0] not in display_video_list:
            continue
        video_tik = time.time()
        video = cv2.VideoCapture(os.path.join(input_video_dir, file))
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_gen = frame_from_video(video)
        frame_list = []
        for frame in frame_gen:
            frame_list.append(frame)
        N = len(frame_list)
        video_loading_time += (time.time() - video_tik)
        res_per_video = [[] for _ in range(N)]
        frame_detection_list = [[-1, []] for _ in range(N)] # -1: hasn't invoked model; 0: doesn't contain object; 1: contains object of interest; 2: doesn't contain object of interest
        frame_detection_list2 = [[-1, []] for _ in range(N)]
        base_step = 30

        print("Phase 1...")
        binary_search(0, N, model_a, base_step)
        for i in range(N):
            if frame_detection_list[i][0] != 2:
                has_car = False
                pred_classes = frame_detection_list[i][1]
                for pred_class in pred_classes:
                    if coco_names[pred_class.item()] == "car":
                        has_car = True
                        break
                frame_detection_list2[i] = [int(has_car == True), pred_classes]

        print("Phase 2...")
        idx_car = 0
        while idx_car < N:
            if frame_detection_list[idx_car][0] == 1:
                event_a_count = 1
                event_a_start = idx_car 
                while idx_car+event_a_count < N and frame_detection_list[idx_car+event_a_count][0] == 1:
                    event_a_count += 1
                if event_a_count >= 30:
                    binary_search(idx_car, min(idx_car+301, N), model_b, base_step, False)
                idx_car += event_a_count
            idx_car += 1
        
        # Finally, write bboxes file to the disk
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
    print("total_invokation: ", total_invokation_stage1, total_invokation_stage2, total_invokation_stage1+total_invokation_stage2)

def preprocess_binary_search_rare_last_clean_version():
    global frame_detection_list, res_per_video, model_execution_time, total_invokation_stage1, total_invokation_stage2, frame_detection_list2
    total_invokation_stage1 = 0
    total_invokation_stage2 = 0
    model_execution_time = 0
    video_loading_time = 0 
    bbox_file_construct_time = 0 

    def model_a(frame):
        global model_execution_time, total_invokation_stage1
        tik = time.time()
        outputs = predictor(frame)
        model_execution_time += (time.time() - tik)
        total_invokation_stage1 += 1
        instances = outputs["instances"]
        pred_boxes = instances.pred_boxes
        scores = instances.scores
        pred_classes = instances.pred_classes            
        for pred_class in pred_classes:
            if coco_names[pred_class.item()] == "car":
                return True, pred_boxes, scores, pred_classes
        return False, pred_boxes, scores, pred_classes

    def model_b(frame):
        global model_execution_time, total_invokation_stage2
        tik = time.time()
        outputs = predictor(frame)
        model_execution_time += (time.time() - tik)
        total_invokation_stage2 += 1
        instances = outputs["instances"]
        pred_boxes = instances.pred_boxes
        scores = instances.scores
        pred_classes = instances.pred_classes            
        for pred_class in pred_classes:
            if coco_names[pred_class.item()] == "truck":
                return True, pred_boxes, scores, pred_classes
        return False, pred_boxes, scores, pred_classes

                
    def binary_search(start_frame, end_frame, ML_model, min_duration, event_a=True):
        global frame_detection_list, res_per_video, frame_detection_list2
        if min_duration == 0: 
            return 
        for i in range(start_frame, end_frame, min_duration):
            if (event_a and frame_detection_list[i][0] != -1) or (event_a==False and frame_detection_list2[i][0] != -1):
                continue
            prediction, pred_boxes, scores, pred_classes = ML_model(frame_list[i])
            
            # Update F
            if event_a:
                frame_detection_list[i] = [int(prediction == True), pred_classes]
            else:
                frame_detection_list2[i] = [int(prediction == True), pred_classes]
        
            # Write results to bboxes list 
            res_per_frame = []
            for pred_box, score, pred_class in zip(pred_boxes, scores, pred_classes):
                res_per_frame.append([pred_box[0].item(), pred_box[1].item(), pred_box[2].item(), pred_box[3].item(), coco_names[pred_class.item()], score.item(), str(i + 1).zfill(10) + '.jpg'])
            res_per_video[i] = res_per_frame
        
        # This pass is done.
        # Update F by examining if there are any frames that can be skipped.
        for i in range(start_frame, end_frame - min_duration, min_duration):
            if event_a:
                if frame_detection_list[i][0] % 2 == 0 and frame_detection_list[i+min_duration][0] % 2 == 0 and frame_detection_list[i+1][0] % 2 != 0:
                    for j in range(i+1, i+min_duration):
                        if frame_detection_list[j][0] == -1:
                            frame_detection_list[j] = [2, []]
            else:
                if frame_detection_list2[i][0] % 2 == 0 and frame_detection_list2[i+min_duration][0] % 2 == 0 and frame_detection_list2[i+1][0] % 2 != 0:
                    for j in range(i+1, i+min_duration):
                        if frame_detection_list2[j][0] == -1:
                            frame_detection_list2[j] = [2, []]
        # Go to the next level of recursion.
        binary_search(start_frame, end_frame, ML_model, int(min_duration / 2), event_a)
        
    display_video_list = ["cabc30fc-e7726578", "cabc30fc-eb673c5a", "cabc30fc-fd79926f", "cabc9045-1b8282ba", "cabc9045-5a50690f"]
    
    total_time_start = time.time()

    # input_video_dir = "/home/ubuntu/CSE544-project/rekall/data/input_videos/"
    input_video_dir = "/home/ubuntu/CSE544-project/data/bdd100k/videos/test/"
    # bbox_file_dir = "/home/ubuntu/CSE544-project/rekall/data/bbox_files/"
    bbox_file_dir = "/home/ubuntu/CSE544-project/data/bdd100k/bbox_files_binary_search_rare_last/"

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

    

    for file in os.listdir(input_video_dir):
        if os.path.splitext(file)[1] != '.mp4' and os.path.splitext(file)[1] != '.mov':
            continue
        if os.path.splitext(file)[0] not in display_video_list:
            continue
        video_tik = time.time()
        video = cv2.VideoCapture(os.path.join(input_video_dir, file))
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_gen = frame_from_video(video)
        frame_list = []
        for frame in frame_gen:
            frame_list.append(frame)
        N = len(frame_list)
        video_loading_time += (time.time() - video_tik)
        res_per_video = [[] for _ in range(N)]
        frame_detection_list = [[-1, []] for _ in range(N)] # -1: hasn't invoked model; 0: doesn't contain object; 1: contains object of interest; 2: doesn't contain object of interest
        frame_detection_list2 = [[-1, []] for _ in range(N)]
        base_step = 30

        print("Phase 1...")
        binary_search(0, N, model_a, base_step)
        for i in range(N):
            if frame_detection_list[i][0] != 2:
                has_car = False
                pred_classes = frame_detection_list[i][1]
                for pred_class in pred_classes:
                    if coco_names[pred_class.item()] == "truck":
                        has_car = True
                        break
                frame_detection_list2[i] = [int(has_car == True), pred_classes]

        print("Phase 2...")
        idx_car = 0
        while idx_car < N:
            if frame_detection_list[idx_car][0] == 1:
                event_a_count = 1
                event_a_start = idx_car 
                while idx_car+event_a_count < N and frame_detection_list[idx_car+event_a_count][0] == 1:
                    event_a_count += 1
                if event_a_count >= 30:
                    binary_search(max(0, idx_car-300), idx_car, model_b, base_step, False)
                idx_car += event_a_count
            idx_car += 1
        
        # Finally, write bboxes file to the disk
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
    print("total_invokation: ", total_invokation_stage1, total_invokation_stage2, total_invokation_stage1+total_invokation_stage2)


if __name__ == '__main__':
    preprocess_faster_rcnn()
    # preprocess_binary_search_rare_first()
    # preprocess_binary_search_rare_last()
    # preprocess_binary_search_rare_first_q2()
    # preprocess_binary_search_rare_last_q2()
    # preprocess_binary_search_rare_first_clean_version()
    # preprocess_binary_search_rare_last_clean_version()
    

# Stats
"""
no sampling, 5 videos:
model loading time:  3.0977327823638916
model execution time:  736.931215763092
video loading time:  0.053437232971191406
bbox file construct time:  5.971518516540527
total time:  764.7963237762451
total_invokation:  6027

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
model loading time:  3.167116403579712
model execution time:  320.55116534233093
video loading time:  17.704741954803467
bbox file construct time:  0.014028310775756836
total time:  343.9729516506195
total_invokation:  1517 1096 2613

binary search + rare last, 5 videos:
model loading time:  3.990025043487549
model execution time:  724.6807322502136
video loading time:  17.228809356689453
bbox file construct time:  0.03544926643371582
total time:  752.4066162109375
total_invokation:  6013 5 6018

preprocess_binary_search_rare_first_q2, 5 videos:
model loading time:  3.116368293762207
model execution time:  488.75887298583984
video loading time:  17.40657329559326
bbox file construct time:  0.025256872177124023
total time:  514.1527323722839
total_invokation:  2915 1096 4011

preprocess_binary_search_rare_last_q2, 5 videos:
model loading time:  3.1183226108551025
model execution time:  260.8483397960663
video loading time:  17.188172578811646
bbox file construct time:  0.009523153305053711
total time:  283.1461465358734
total_invokation:  2082 54 2136
"""
