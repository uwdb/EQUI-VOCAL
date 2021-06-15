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
sys.path.append('/home/ubuntu/CSE544-project/person_attribute')
from net import get_model

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer

# color recognition module - import
sys.path.append('/home/ubuntu/CSE544-project/vehicle_counting_tensorflow')
from utils.color_recognition_module import color_recognition_api

import mysql.connector

from construct_input_streams import *
from pattern_matching import *

def frame_from_video(video):
    while video.isOpened():
        success, frame = video.read()
        if success:
            yield frame
        else:
            break

def frame_id_to_time_interval(frame_id, fps):
    start_time = frame_id / fps
    end_time = (frame_id + 1) / fps
    return start_time, end_time

class predict_decoder(object):
    def __init__(self, dataset):
        with open('/home/ubuntu/CSE544-project/person_attribute/doc/label.json', 'r') as f:
            self.label_list = json.load(f)[dataset]
        with open('/home/ubuntu/CSE544-project/person_attribute/doc/attribute.json', 'r') as f:
            self.attribute_dict = json.load(f)[dataset]
        self.dataset = dataset
        self.num_label = len(self.label_list)

    def decode(self, pred):
        pred = pred.squeeze(dim=0)
        for idx in range(self.num_label):
            name, chooce = self.attribute_dict[self.label_list[idx]]
            if chooce[pred[idx]]:
                print('{}: {}'.format(name, chooce[pred[idx]]))

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
    
def load_person_attribute_model():
    model_name = 'resnet50_nfc'
    num_label = 30
    num_id = 751

    def load_network(network):
        save_path = os.path.join('/home/ubuntu/CSE544-project/person_attribute/checkpoints', "market", model_name, 'net_last.pth')
        network.load_state_dict(torch.load(save_path))
        print('Resume model from {}'.format(save_path))
        return network

    model = get_model(model_name, num_label, num_id=num_id)
    model = load_network(model)
    model.eval()
    return model 


def visualize_results_motorbike_crossing(out_stream, idx):
    video = cv2.VideoCapture("/home/ubuntu/CSE544-project/data/visual_road/traffic-{0}.mp4".format(idx))
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    
    # Visualize bboxes
    for i, motorbike_frame_id in enumerate(out_stream):

        video.set(cv2.CAP_PROP_POS_FRAMES, motorbike_frame_id)
    
        ret, motorbike_frame = video.read()

        # cv2.rectangle(car_frame, (int(car_x1), int(car_y1)), (int(car_x2), int(car_y2)), (0, 255, 0), 3)
        cv2.imwrite("motorbike_crossing/val/neg/traffic{0}-{1}.jpg".format(idx, i), motorbike_frame)


def visualize_results_car_turning_right(out_stream, idx):
    # Write start_time, end_time to csv file
    # with open('naive_results_car_turning_right.csv', 'w') as f:
    #     writer = csv.writer(f)
    #     csv_line = "start_time, end_time"
    #     writer.writerows([csv_line.split(',')])
        
    #     for row in out_stream:
    #         writer.writerow([row[0], row[1]])
    

    video = cv2.VideoCapture("/home/ubuntu/CSE544-project/data/visual_road/traffic-{0}.mp4".format(idx))
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    
    # Visualize bboxes
    for i, car_frame_id in enumerate(out_stream):

        video.set(cv2.CAP_PROP_POS_FRAMES, car_frame_id)
    
        ret, car_frame = video.read()

        # cv2.rectangle(car_frame, (int(car_x1), int(car_y1)), (int(car_x2), int(car_y2)), (0, 255, 0), 3)
        cv2.imwrite("car_turning_right_test/val/neg/traffic{0}-{1}.jpg".format(idx, i), car_frame)


def visualize_results_person_edge_corner(out_stream):
    # Write start_time, end_time to csv file
    with open('naive_results_person_edge_corner.csv', 'w') as f:
        writer = csv.writer(f)
        csv_line = "start_time, end_time"
        writer.writerows([csv_line.split(',')])
        
        for row in out_stream:
            writer.writerow([row[0], row[1]])
    

    # video = cv2.VideoCapture("/home/ubuntu/CSE544-project/data/visual_road/traffic-4k-002.mp4")
    video = cv2.VideoCapture("/home/ubuntu/CSE544-project/data/visual_road/traffic-20.mp4")
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    
    # Visualize bboxes
    for i, row in enumerate(out_stream):
        start_time, end_time, x1, y1, x2, y2, frame_id = row 

        video.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    
        ret, frame = video.read()

        # cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
        # if i % 30 == 0:
        cv2.imwrite("person_edge_corner_test/train/traffic20-" + str(i) + ".jpg", frame)


def visualize_results_watch_out_person_cross_road_when_car_turning(out_stream):
    # Write start_time, end_time to csv file
    with open('naive_results_watch_out_person_cross_road_when_car_turn_left.csv', 'w') as f:
        writer = csv.writer(f)
        csv_line = "start_time, end_time"
        writer.writerows([csv_line.split(',')])
        
        for row in out_stream:
            writer.writerow([row[0], row[1]])

    video = cv2.VideoCapture("/home/ubuntu/CSE544-project/data/visual_road/traffic-4k-002.mp4")
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    
    # Visualize bboxes
    for i, row in enumerate(out_stream):
        start_time, end_time, person_x1, person_y1, person_x2, person_y2, person_frame_id, car_x1, car_y1, car_x2, car_y2, car_frame_id = row 
        
        video.set(cv2.CAP_PROP_POS_FRAMES, person_frame_id)
    
        ret, person_frame = video.read()
    
        cv2.rectangle(person_frame, (int(person_x1), int(person_y1)), (int(person_x2), int(person_y2)), (0, 255, 0), 3)
        # cropped_person = person_frame[int(person_box[1].item()):int(person_box[3].item()), int(person_box[0].item()):int(person_box[2].item())]
        cv2.imwrite("test_results_watch_out_person_cross_road_when_car_turn_left/" + str(i) + "-" + str(start_time) + "-" + str(end_time) + "-person.jpg", person_frame)

        video.set(cv2.CAP_PROP_POS_FRAMES, car_frame_id)
    
        ret, car_frame = video.read()

        cv2.rectangle(car_frame, (int(car_x1), int(car_y1)), (int(car_x2), int(car_y2)), (0, 255, 0), 3)
        # cropped_car = car_frame[int(car_box[1].item()):int(car_box[3].item()), int(car_box[0].item()):int(car_box[2].item())]
        cv2.imwrite("test_results_watch_out_person_cross_road_when_car_turn_left/" + str(i) + "-" + str(start_time) + "-" + str(end_time) + "-car.jpg", car_frame)
    

def visualize_results_three_motorbikes_in_a_row(out_stream):
    # Write start_time, end_time to csv file
    with open('naive_results_three_people_overlap.csv', 'w') as f:
        writer = csv.writer(f)
        csv_line = "start_time, end_time"
        writer.writerows([csv_line.split(',')])
        
        for row in out_stream:
            writer.writerow([row[0], row[1]])

    # Visualize bboxes
    for i, row in enumerate(out_stream):
        start_time, end_time, e1_x1, e1_y1, e1_x2, e1_y2, e2_x1, e2_y1, e2_x2, e2_y2, e3_x1, e3_y1, e3_x2, e3_y2, frame_id = row 


        video = cv2.VideoCapture("/home/ubuntu/CSE544-project/data/visual_road/traffic-4k-002.mp4")
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video.get(cv2.CAP_PROP_FPS)
        
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        
        ret, frame = video.read()
        
        cv2.rectangle(frame, (int(e1_x1), int(e1_y1)), (int(e1_x2), int(e1_y2)), (0, 255, 0), 3)

        cv2.rectangle(frame, (int(e2_x1), int(e2_y1)), (int(e2_x2), int(e2_y2)), (0, 255, 0), 3)

        cv2.rectangle(frame, (int(e3_x1), int(e3_y1)), (int(e3_x2), int(e3_y2)), (0, 255, 0), 3)
        
        cv2.imwrite("test_results_three_people_overlap/" + str(i) + "-" + str(start_time) + "-" + str(end_time) + ".jpg", frame)

def visualize_results_same_car_reappears(out_stream):
    # Write start_time, end_time to csv file
    with open('naive_results_same_car_reappears_orange.csv', 'w') as f:
        writer = csv.writer(f)
        csv_line = "start_time, end_time"
        writer.writerows([csv_line.split(',')])
        
        for row in out_stream:
            writer.writerow([row[0], row[1]])

    video = cv2.VideoCapture("/home/ubuntu/CSE544-project/data/visual_road/traffic-4k-002.mp4")
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    
    # Visualize bboxes
    for i, row in enumerate(out_stream):
        start_time, end_time, e1_x1, e1_y1, e1_x2, e1_y2, e1_frame_id, e2_x1, e2_y1, e2_x2, e2_y2, e2_frame_id = row 
        
        video.set(cv2.CAP_PROP_POS_FRAMES, e1_frame_id)
    
        ret, e1_frame = video.read()
    
        cv2.rectangle(e1_frame, (int(e1_x1), int(e1_y1)), (int(e1_x2), int(e1_y2)), (0, 255, 0), 3)
        # cropped_person = person_frame[int(person_box[1].item()):int(person_box[3].item()), int(person_box[0].item()):int(person_box[2].item())]
        cv2.imwrite("test_results_same_car_reappears_orange/" + str(i) + "-" + str(start_time) + "-" + str(end_time) + "-car1.jpg", e1_frame)

        video.set(cv2.CAP_PROP_POS_FRAMES, e2_frame_id)
    
        ret, e2_frame = video.read()

        cv2.rectangle(e2_frame, (int(e2_x1), int(e2_y1)), (int(e2_x2), int(e2_y2)), (0, 255, 0), 3)
        # cropped_car = car_frame[int(car_box[1].item()):int(car_box[3].item()), int(car_box[0].item()):int(car_box[2].item())]
        cv2.imwrite("test_results_same_car_reappears_orange/" + str(i) + "-" + str(start_time) + "-" + str(end_time) + "-car2.jpg", e2_frame)


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

    # Person Attribute Model 
    attribute_model = load_person_attribute_model()

    # predict_decoder
    Dec = predict_decoder("market")

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
            frame_gen = frame_from_video(video)
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

                    start_time, end_time = frame_id_to_time_interval(frame_id - (7 - idx) * 10, fps)

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


def same_car_reappears(connection):
    print("start preprocessing")
    car_stream = construct_input_streams_same_car_reappears(connection)
    print("start pattern matching")
    out_stream = pattern_matching_reappear_within_20s(car_stream)
    print("start visualizing")
    visualize_results_same_car_reappears(out_stream)


def watch_out_person_cross_road_when_car_turning(connection):
    print("start preprocessing")
    # First time 
    # preprocess_faster_rcnn()
    person_stream, car_stream = construct_input_streams_watch_out_person_cross_road_when_car_turn_left(connection)
    print("start pattern matching")
    out_stream = pattern_matching_before_within_5s(person_stream, car_stream)
    print("start visualizing")
    visualize_results_watch_out_person_cross_road_when_car_turning(out_stream)

def car_turning_right(connection, idx):
    print("start preprocessing")
    # First time 
    # preprocess_faster_rcnn()
    car_stream = construct_input_streams_car_turning_right(connection, idx)
    print("start visualizing")
    visualize_results_car_turning_right(car_stream, idx)

def motorbike_crossing(connection, idx):
    print("start preprocessing")
    # First time 
    # preprocess_faster_rcnn()
    # motorbike_stream = construct_input_streams_motorbike_crossing(connection, idx)
    motorbike_stream = construct_input_streams_motorbike_crossing_neg(connection, idx)
    print("start visualizing")
    visualize_results_motorbike_crossing(motorbike_stream, idx)

def person_edge_corner(connection):
    print("start preprocessing")
    # First time 
    # preprocess_faster_rcnn()
    person_stream = construct_input_streams_person_edge_corner(connection)
    print("start visualizing")
    visualize_results_person_edge_corner(person_stream)


if __name__ == '__main__':
    connection = mysql.connector.connect(user='admin', password='123456abcABC',
                              host='database-1.cld3cb8o2zkf.us-east-1.rds.amazonaws.com',
                              database='complex_event')


    # preprocess_faster_rcnn()
    # watch_out_person_cross_road_when_car_turning(connection)
    for i in range(16, 21):
        motorbike_crossing(connection, i)
    # person_edge_corner(connection)
    # same_car_reappears(connection)

    connection.close()