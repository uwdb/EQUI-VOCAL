"""
Convert bbox.pkl to datalog ingestible file obj.facts
List[List[List[x1, y1, x2, y2, class, score, img_name]]
Table schema:
Obj(oid: number, vid: number, fid: number, cid: number, x1: float, y1: float, x2: float, y2: float)
"""

import pickle
import os
import json

coco_names = {}
with open("/home/ubuntu/complex_event_video/src/ms_coco_classnames.txt") as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        coco_names[line.strip()] = i

oid = 0

dir_name = "/home/ubuntu/complex_event_video/data/bdd100k/bbox_files"
filename_json = {}
with open("/home/ubuntu/complex_event_video/src/datalog/data/facts/bdd_obj.facts", 'w') as f:
    for vid, file_name in enumerate(os.listdir(dir_name)):
        filename_json[vid] = file_name
        with open(os.path.join(dir_name, file_name), 'rb') as fr:
            maskrcnn_bboxes = pickle.load(fr)
        for frame_id in range(len(maskrcnn_bboxes)):
            res_per_frame = maskrcnn_bboxes[frame_id]
            for x1, y1, x2, y2, class_name, _, _ in res_per_frame:
                f.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(oid, vid, frame_id, coco_names[class_name], x1, y1, x2, y2))
                oid += 1

with open("/home/ubuntu/complex_event_video/src/datalog/data/bdd_filenames.json", 'w') as f:
    json.dump(filename_json, f)
