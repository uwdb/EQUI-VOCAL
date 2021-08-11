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

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from faster_r_cnn import FasterRCNN
from utils import frame_from_video
import tools

class FasterRCNNPreprocess(FasterRCNN):
    def __init__(self):
        super().__init__()
        with open("/home/ubuntu/complex_event_video/src/ms_coco_classnames.txt") as f:
            self.coco_names = f.read().splitlines()
        self.bbox_info = {}

    def __call__(self, x, file_names):
        inputs = [{"image": img_tensor} for img_tensor in x]
        outputs = self.model(inputs)
        for i, output in enumerate(outputs):
            res_per_frame = []
            instances = output["instances"]
            pred_boxes = instances.pred_boxes
            scores = instances.scores
            pred_classes = instances.pred_classes
            for pred_box, score, pred_class in zip(pred_boxes, scores, pred_classes):
                # List[(x1, y1, x2, y2, class_name, score)]
                res_per_frame.append([pred_box[0].item(), pred_box[1].item(), pred_box[2].item(), pred_box[3].item(), self.coco_names[pred_class.item()], score.item()])
            self.bbox_info[file_names[i]] = res_per_frame

    def get_bbox_info(self):
        return self.bbox_info




data_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.0]*3, [1/255.0] * 3)])

inv_transforms = transforms.Normalize(mean = [0.] * 3, std = [255.0] * 3)

data_dir = '/home/ubuntu/complex_event_video/data/car_turning_traffic2'

image_dataset = datasets.ImageFolder(data_dir, data_transform)

dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=8, shuffle=False, num_workers=4)

dataset_size = len(image_dataset)
class_names = image_dataset.classes
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Input: List[path_to_video_file]
@tools.tik_tok
def preprocess(model):
    # Iterate over data.
    for i, (inputs, labels) in enumerate(tqdm(dataloader)):
        inputs = inputs.to(device)
        labels = labels.to(device)
        file_names = [os.path.split(dataloader.dataset.samples[8 * i + j][0])[1] for j in range(inputs.size()[0])]
        # forward
        model(inputs, file_names)

    bbox_info = model.get_bbox_info()

    with open("/home/ubuntu/complex_event_video/data/car_turning_traffic2/bbox.json", 'w') as f:
        f.write(json.dumps(bbox_info))


if __name__ == '__main__':
    model = FasterRCNNPreprocess()
    preprocess(model)