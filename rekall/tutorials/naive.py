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

# Temporal predicates
def before(intrvl1, intrvl2, min_dist=0, max_dist="infty"):
    return intrvl1[0] + min_dist <= intrvl2[0] and (intrvl1[0] + max_dist >= intrvl2[0] or max_dist == "infty")


def pattern_matching(person_stream, car_stream):
    out_stream = []
    # Person, followed by car, within 5 seconds
    for intrvl1 in person_stream:
        for intrvl2 in car_stream:
            if before(intrvl1, intrvl2, max_dist=5):
                out_stream.append((intrvl1[0], intrvl2[1], intrvl1[2], intrvl1[3], intrvl2[2], intrvl2[3]))
    return out_stream


def visualize_results(out_stream):
    # Write start_time, end_time to csv file
    with open('naive_results.csv', 'w') as f:
        writer = csv.writer(f)
        csv_line = "start_time, end_time"
        writer.writerows([csv_line.split(',')])
        
        for row in out_stream:
            writer.writerow([row[0], row[1]])

    # Visualize bboxes
    for i, row in enumerate(out_stream):
        start_time, end_time, person_box, person_frame, car_box, car_frame = row 
        cropped_person = person_frame[int(person_box[1].item()):int(person_box[3].item()), int(person_box[0].item()):int(person_box[2].item())]
        cv2.imwrite("test_results/" + str(i) + "-" + str(start_time) + "-" + str(end_time) + "-person.jpg", cropped_person)

        cropped_car = car_frame[int(car_box[1].item()):int(car_box[3].item()), int(car_box[0].item()):int(car_box[2].item())]
        cv2.imwrite("test_results/" + str(i) + "-" + str(start_time) + "-" + str(end_time) + "-car.jpg", cropped_car)
    

    
def preprocess_faster_rcnn():
    img_id = 0
    display_video_list = ["cabc30fc-e7726578"]
    # display_video_list = ["cabc30fc-e7726578", "cabc30fc-eb673c5a", "cabc30fc-fd79926f", "cabc9045-1b8282ba", "cabc9045-5a50690f"]
    # display_video_list = ["VIRAT_S_000201_00_000018_000380"] # 6 min video
    # display_video_list = ["VIRAT_S_000200_00_000100_000171"] # 1 min video
    # input_video_dir = "/home/ubuntu/CSE544-project/data/"
    input_video_dir = "/home/ubuntu/CSE544-project/data/bdd100k/videos/test/"
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
    predictor = DefaultPredictor(cfg)

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
        person_stream = []
        car_stream = []

        if not video.isOpened():
            print("Error opening video stream or file: ", file)
        else:
            frame_gen = frame_from_video(video)
            for frame in tqdm(frame_gen, total=num_frames):
                outputs = predictor(frame)
                instances = outputs["instances"]
                pred_boxes = instances.pred_boxes
                scores = instances.scores
                # pred_class is idx, not string; need coco_names[pred_class.item()] to access class name 
                pred_classes = instances.pred_classes
                for pred_box, score, pred_class in zip(pred_boxes, scores, pred_classes):
                    # Recognize person attribute 
                    if coco_names[pred_class.item()] == "person":
                        cropped_frame = frame[int(pred_box[1].item()):int(pred_box[3].item()), int(pred_box[0].item()):int(pred_box[2].item())]
                        src = load_image(cropped_frame)
                        
                        out = attribute_model.forward(src)

                        pred = torch.gt(out, torch.ones_like(out)/2 )  # threshold=0.5
                        pred = pred.squeeze(dim=0)
                        # female and upwhite
                        if pred[12] and pred[14]:
                            start_time, end_time = frame_id_to_time_interval(frame_id, fps)
                            person_stream.append((start_time, end_time, pred_box, frame))
                            # cv2.imwrite("test_results/test" + str(img_id) + ".jpg", cropped_frame)
                            # print(img_id)
                            # img_id += 1
                    elif coco_names[pred_class.item()] in ["car", "truck"]:
                        image_pil = Image.fromarray(np.uint8(frame)).convert('RGB')
                        image_temp = np.array(image_pil)
                        
                        top, bottom, right, left = pred_box[1].item(), pred_box[3].item(), pred_box[2].item(), pred_box[0].item()

                        detected_vehicle_image = image_temp[int(top):int(bottom), int(left):int(right)]
                        
                        # predicted_direction, predicted_speed, is_vehicle_detected, update_csv = speed_prediction.predict_speed(top, bottom, right, left, current_frame_number, detected_vehicle_image, ROI_POSITION)
                        
                        predicted_size = (bottom - top) * (right - left)

                        predicted_color = color_recognition_api.color_recognition(detected_vehicle_image)

                        if predicted_color == "white" and predicted_size >= 30000:
                            start_time, end_time = frame_id_to_time_interval(frame_id, fps)
                            car_stream.append((start_time, end_time, pred_box, frame))

                        # cv2.imwrite("test_results/test" + str(img_id) + "-" + predicted_color + ".jpg", detected_vehicle_image)
                        # print(img_id)
                        # img_id += 1

                frame_id += 1 
    return person_stream, car_stream

if __name__ == '__main__':
    load_person_attribute_model()
    print("start preprocessing")
    person_stream, car_stream = preprocess_faster_rcnn()
    print("start pattern matching")
    out_stream = pattern_matching(person_stream, car_stream)
    print("start visualizing")
    visualize_results(out_stream)
