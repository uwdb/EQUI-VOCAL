# Some basic setup:
# Setup detectron2 logger
import torch, torchvision
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random, time, csv
from tqdm import tqdm
import pickle
from PIL import Image
from torchvision import transforms as T
import sys
import logging 

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog

from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer

import mysql.connector

from construct_input_streams import *
from pattern_matching import *

from visualizer import Visualizer



def load_image(im):
    transforms = T.Compose([
        T.Resize(size=(288, 144)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    src = Image.fromarray(im)
    src = transforms(src)
    src = src.unsqueeze(dim=0)
    # (1, channel, width, height)
    return src


def draw_bounding_box_on_image():
    frame_id = 9630
    video = cv2.VideoCapture("/home/ubuntu/CSE544-project/data/visual_road/traffic-4k-002.mp4")
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    
    ret, frame = video.read()

    cursor = connection.cursor()
    cursor.execute("SELECT e.start_time, e.end_time, e.event_type, v.x1, v.x2, v.y1, v.y2 FROM Event e, VisibleAt v WHERE e.event_id = v.event_id AND v.filename = 'traffic-4k-002.mp4' AND v.frame_id = %s", [frame_id])

    for row in cursor:
        start_time, end_time, event_type, x1, x2, y1, y2 = row
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
    
    cv2.imwrite("test.jpg", frame)

    cursor.close()


def watch_out_person_cross_road_when_car_turning(connection):
    print("start preprocessing")
    # First time 
    # preprocess_faster_rcnn()
    person_stream, car_stream = construct_input_streams_watch_out_person_cross_road_when_car_turn_left(connection)
    print("start pattern matching")
    out_stream = pattern_matching_before_within_5s(person_stream, car_stream)
    print("start visualizing")
    visualize_results_watch_out_person_cross_road_when_car_turning(out_stream)

def car_turning(connection, idx):
    event_name = "car_turning"
    print("start preprocessing")
    # First time 
    # preprocess_faster_rcnn()
    car_stream = construct_input_streams_car_turning(connection, idx)
    print("start visualizing")
    vis = Visualizer(event_name)
    vis.visualize_results(car_stream, idx, event_name)
    # visualize_results_car_turning_right(car_stream, idx)

def motorbike_crossing(connection, idx):
    print("start preprocessing")
    # First time 
    # preprocess_faster_rcnn()
    # motorbike_stream = construct_input_streams_motorbike_crossing(connection, idx)
    motorbike_stream = construct_input_streams_motorbike_crossing_neg(connection, idx)
    print("start visualizing")
    visualize_results_motorbike_crossing(motorbike_stream, idx)

class Executor:
    def __init__(self, event_name, train_video_fn_list, val_video_fn_list):
        self.connection = mysql.connector.connect(
            user='admin', 
            password='123456abcABC', 
            host='database-1.cld3cb8o2zkf.us-east-1.rds.amazonaws.com', database='complex_event'
        )
        self.event_name = event_name
        self.train_video_fn_list = train_video_fn_list
        self.val_video_fn_list = val_video_fn_list

    @staticmethod
    def frame_from_video(video):
        while video.isOpened():
            success, frame = video.read()
            if success:
                yield frame
            else:
                break
    
    @staticmethod
    def frame_id_to_time_interval(frame_id, fps):
        start_time = frame_id / fps
        end_time = (frame_id + 1) / fps
        return start_time, end_time

    @staticmethod
    def preprocess_faster_rcnn():
        img_id = 0
        # display_video_list = ["car-pov-2k-000-shortened", "car-pov-2k-001-shortened", "traffic-4k-000", "traffic-4k-000-ds2k", "traffic-4k-002"]
        display_video_list = ["traffic-1"]
        # for i in range(2, 21):
        #     display_video_list.append("traffic-" + str(i))
            
        # display_video_list = ["cabc30fc-e7726578"]
        # display_video_list = ["cabc30fc-e7726578", "cabc30fc-eb673c5a", "cabc30fc-fd79926f", "cabc9045-1b8282ba", "cabc9045-5a50690f"]
        # display_video_list = ["VIRAT_S_000201_00_000018_000380"] # 6 min video
        # display_video_list = ["VIRAT_S_000200_00_000100_000171"] # 1 min video
        # input_video_dir = "/home/ubuntu/CSE544-project/data/"
        input_video_dir = "/home/ubuntu/CSE544-project/data/visual_road/"
        # input_video_dir = "/home/ubuntu/CSE544-project/data/bdd100k/videos/test/"
        # bbox_file_dir = "/home/ubuntu/CSE544-project/rekall/data/bbox_files/"
        # bbox_file_dir = "/home/ubuntu/CSE544-project/data/bdd100k/bbox_files_original"

        with open("ms_coco_classnames.txt") as f:
            coco_names = f.read().splitlines() 

        cfg = get_cfg()
        # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6  # set threshold for this model
        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
        
        # predictor = DefaultPredictor(cfg)
        model = build_model(cfg) # returns a torch.nn.Module
        DetectionCheckpointer(model).load('model_final_280758.pkl') # must load weights this way, can't use cfg.MODEL.WEIGHTS = "..."
        model.train(False) # inference mode


        for file in os.listdir(input_video_dir):
            if os.path.splitext(file)[1] != '.mp4' and os.path.splitext(file)[1] != '.mov':
                continue
            if os.path.splitext(file)[0] not in display_video_list:
                continue

            video = cv2.VideoCapture(os.path.join(input_video_dir, file))
            num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = video.get(cv2.CAP_PROP_FPS)
            
            frame_id = 0
            inputs = []

            if not video.isOpened():
                print("Error opening video stream or file: ", file)
            else:
                frame_gen = self.frame_from_video(video)
                for frame in tqdm(frame_gen, total=num_frames):
                    if frame_id % 10 != 0:
                        frame_id += 1
                        continue
                    img_transposed = np.transpose(frame, (2, 0, 1)) # From (H, W, C) to (C, H, W)
                    img_tensor = torch.from_numpy(img_transposed)
                    inputs.append({"image":img_tensor}) # inputs is ready
                    if len(inputs) < 8:
                        frame_id += 1
                        continue
                    outputs = model(inputs)
                    inputs = []
                    for idx, output in enumerate(outputs):
                        # outputs = predictor(frame)
                        instances = output["instances"]
                        pred_boxes = instances.pred_boxes
                        scores = instances.scores
                        # pred_class is idx, not string; need coco_names[pred_class.item()] to access class name 
                        pred_classes = instances.pred_classes

                        start_time, end_time = self.frame_id_to_time_interval(frame_id - (7 - idx) * 10, fps)

                        for pred_box, score, pred_class in zip(pred_boxes, scores, pred_classes):
                            cursor = connection.cursor()
                            # Store object detection results 

                            # Populate Event table
                            cursor.execute("INSERT INTO Event (event_type, start_time, end_time) VALUES (%s, %s, %s)", [coco_names[pred_class.item()], start_time, end_time])
                            event_id = cursor.lastrowid
                            
                            # Populate VisibleAt table 
                            cursor.execute("INSERT INTO VisibleAt VALUES (%s, %s, %s, %s, %s, %s, %s)", [file, frame_id - (7 - idx) * 10, event_id, pred_box[0].item(), pred_box[2].item(), pred_box[1].item(), pred_box[3].item()])
                            
                            # Recognize person attribute 
                            if coco_names[pred_class.item()] == "person":
                                # cropped_frame = frame[int(pred_box[1].item()):int(pred_box[3].item()), int(pred_box[0].item()):int(pred_box[2].item())]
                                # src = load_image(cropped_frame)
                                
                                # out = attribute_model.forward(src)

                                # pred = torch.gt(out, torch.ones_like(out)/2 )  # threshold=0.5
                                # pred = pred.squeeze(dim=0)
                                
                                # # Populate Person table 
                                # cursor.execute("INSERT INTO Person VALUES (%s, %s, %s)", [event_id, pred[12].item(), pred[14].item()])
                                cursor.execute("INSERT INTO Person VALUES (%s, %s, %s)", [event_id, 0, 0])

                            elif coco_names[pred_class.item()] in ["car"]:
                                # image_pil = Image.fromarray(np.uint8(frame)).convert('RGB')
                                # image_temp = np.array(image_pil)
                                
                                # top, bottom, right, left = pred_box[1].item(), pred_box[3].item(), pred_box[2].item(), pred_box[0].item()

                                # detected_vehicle_image = image_temp[int(top):int(bottom), int(left):int(right)]
                                
                                # # predicted_direction, predicted_speed, is_vehicle_detected, update_csv = speed_prediction.predict_speed(top, bottom, right, left, current_frame_number, detected_vehicle_image, ROI_POSITION)
                                
                                # predicted_size = (bottom - top) * (right - left)

                                # predicted_color = color_recognition_api.color_recognition(detected_vehicle_image)

                                # Populate Car table 
                                cursor.execute("INSERT INTO Car VALUES (%s, %s, %s)", [event_id, 'a', -1])

                                # cv2.imwrite("test_results/test" + str(img_id) + "-" + predicted_color + ".jpg", detected_vehicle_image)
                                # print(img_id)
                                # img_id += 1
                            
                        # Commit and close connection 
                        connection.commit()
                        cursor.close()

                    frame_id += 1 
    
    def execute(self):
        logging.info("Target event: {}".format(self.event_name))
        vis = Visualizer(self.event_name)
        # Construct sample dataset: training data
        for video_fn in self.train_video_fn_list:
            logging.info("Executing video file: {}".format(video_fn))
            # positive training data 
            car_stream = self.execute_pos(video_fn)
            logging.info("Retrieved {} positive training outputs".format(len(car_stream)))
            vis.visualize_results(car_stream, video_fn, self.event_name, "train", "pos")

            # negative training data 
            car_stream = self.execute_neg(video_fn)
            logging.info("Retrieved {} negative training outputs".format(len(car_stream)))
            vis.visualize_results(car_stream, video_fn, self.event_name, "train", "neg")

        # Construct sample dataset: validation data
        for video_fn in self.val_video_fn_list:
            logging.info("Executing video file: {}".format(video_fn))
            # positive validation data 
            car_stream = self.execute_pos(video_fn)
            logging.info("Retrieved {} positive validation outputs".format(len(car_stream)))
            vis.visualize_results(car_stream, video_fn, self.event_name, "val", "pos")

            # negative validation data 
            car_stream = self.execute_neg(video_fn)
            logging.info("Retrieved {} negative validation outputs".format(len(car_stream)))
            vis.visualize_results(car_stream, video_fn, self.event_name, "val", "neg")

    def execute_pos(self, video_fn):
        return construct_input_streams_car_turning(self.connection, video_fn)

    def execute_neg(self, video_fn):
        return construct_input_streams_car_turning_neg(self.connection, video_fn)

    def close_connection(self):
        self.connection.close()


if __name__ == '__main__':
    # preprocess_faster_rcnn()
    # watch_out_person_cross_road_when_car_turning(connection)
    # for i in range(1, 16):
        # avg_cars(connection, i)
        # car_turning(connection, i)
    # person_edge_corner(connection)
    # same_car_reappears(connection)
    logging.basicConfig(level=logging.INFO)
    train_video_fn_list = ["traffic-{}.mp4".format(i) for i in range(1, 3)]
    val_video_fn_list = ["traffic-{}.mp4".format(i) for i in range(16, 18)]

    executor = Executor("car_turning", train_video_fn_list, val_video_fn_list)
    executor.execute()
    executor.close_connection()