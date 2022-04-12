import json
import os
from glob import glob
import numpy as np
from torchvision.ops import masks_to_boxes
import pycocotools._mask as _mask
import torch

def decode(rleObjs):
    if type(rleObjs) == list:
        return _mask.decode(rleObjs)
    else:
        return _mask.decode([rleObjs])[:,:,0]


class IngestBboxMixin:
    def ingest_bbox_info(self):
        # Read in bbox info
        with open("/gscratch/balazinska/enhaoz/complex_event_video/data/car_turning_traffic2/bbox.json", 'r') as f:
            self.maskrcnn_bboxes = json.loads(f.read())
        self.n_frames = len(self.maskrcnn_bboxes)

    def ingest_bbox_info_meva(self):
        if self.query in ["meva_person_embraces_person", "meva_person_stands_up"]:
            video_camera = "school.G421"
        elif self.query in ["meva_person_enters_vehicle"]:
            video_camera = "school.G336"
        files = [y for x in os.walk("/gscratch/balazinska/enhaoz/complex_event_video/data/meva") for y in glob(os.path.join(x[0], '*.json'))]
        gt_annotations = [os.path.basename(y).replace(".activities.yml", "") for x in os.walk("/gscratch/balazinska/enhaoz/complex_event_video/data/meva/meva-data-repo/annotation/DIVA-phase-2/MEVA/kitware") for y in glob(os.path.join(x[0], '*.yml'))]
        self.maskrcnn_bboxes = {}
        self.video_list = [] # (video_basename, frame_offset, n_frames)
        for file in files:
            # if "school.G421.r13.json" not in file:
            if "{}.r13.json".format(video_camera) not in file:
                continue
            # If the video doesn't have annotations, skip.
            video_basename = os.path.basename(file).replace(".r13.json", "")
            if video_basename not in gt_annotations:
                continue
            frame_offset = len(self.maskrcnn_bboxes)
            # Read in bbox info
            with open(file, 'r') as f:
                bbox_dict = json.loads(f.read())
                for local_frame_id, v in bbox_dict.items():
                    self.maskrcnn_bboxes[video_basename + "_" + local_frame_id] = v
            self.video_list.append((video_basename, frame_offset, len(bbox_dict)))
        self.n_frames = len(self.maskrcnn_bboxes)
        print("# all frames: ", self.n_frames, "; # video files: ", len(self.video_list))

    def ingest_bbox_info_clevrer(self):
        # check if file exists
        if os.path.isfile("/gscratch/balazinska/enhaoz/complex_event_video/data/clevrer/processed_bboxes_1000.json") and os.path.isfile("/gscratch/balazinska/enhaoz/complex_event_video/data/clevrer/processed_video_list_1000.json"):
            with open("/gscratch/balazinska/enhaoz/complex_event_video/data/clevrer/processed_bboxes_1000.json", 'r') as f:
                self.maskrcnn_bboxes = json.loads(f.read())
            with open("/gscratch/balazinska/enhaoz/complex_event_video/data/clevrer/processed_video_list_1000.json", 'r') as f:
                self.video_list = json.loads(f.read())
            self.n_frames = len(self.maskrcnn_bboxes)
            print("# all frames: ", self.n_frames, "; # video files: ", len(self.video_list))
        else:
            self.maskrcnn_bboxes = {} # {video_id_frame_id: [x1, y1, x2, y2, material, color, shape]}
            self.video_list = [] # (video_basename, frame_offset, n_frames)
            # iterate all files in the folder
            for file in glob("/gscratch/balazinska/enhaoz/complex_event_video/data/clevrer/processed_proposals/*.json"):
                video_basename = os.path.basename(file).replace(".json", "").replace("sim_", "")
                if int(video_basename) >= 1000:
                    continue
                # read in the json file
                with open(file, 'r') as f:
                    data = json.loads(f.read())
                frame_offset = len(self.maskrcnn_bboxes)
                # iterate all videos in the json file
                current_fid = 0
                for frame in data['frames']:
                    local_frame_id = frame['frame_index']
                    while local_frame_id > current_fid:
                        self.maskrcnn_bboxes[video_basename + "_" + str(current_fid)] = []
                        current_fid += 1
                    objects = frame['objects']
                    box_list = []
                    for i in range(len(objects)):
                        # print(objects[i]['material'], objects[i]['color'], objects[i]['shape'])
                        mask = decode(objects[i]['mask'])
                        # O represents black, 1 represents white.
                        box = masks_to_boxes(torch.from_numpy(mask[np.newaxis, :]))
                        box = np.squeeze(box.numpy(), axis=0).tolist()
                        box.extend([objects[i]['material'], objects[i]['color'], objects[i]['shape']])
                        box_list.append(box)
                    self.maskrcnn_bboxes[video_basename + "_" + str(local_frame_id)] = box_list
                    current_fid += 1
                while current_fid < 128:
                    self.maskrcnn_bboxes_evaluation[video_basename + "_" + str(current_fid)] = []
                    current_fid += 1
                self.video_list.append((video_basename, frame_offset, 128))
            self.n_frames = len(self.maskrcnn_bboxes)
            print("# all frames: ", self.n_frames, "; # video files: ", len(self.video_list))
            # write dict to file
            with open("/gscratch/balazinska/enhaoz/complex_event_video/data/clevrer/processed_bboxes_1000.json", 'w') as f:
                f.write(json.dumps(self.maskrcnn_bboxes))
            # write video_list to file
            with open("/gscratch/balazinska/enhaoz/complex_event_video/data/clevrer/processed_video_list_1000.json", 'w') as f:
                f.write(json.dumps(self.video_list))