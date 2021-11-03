"""
Convert bbox.json to datalog ingestible file obj.facts
Table schema: 
Obj(oid: number, fid: number, cid: number, x1: float, y1: float, x2: float, y2: float)
"""

import json 

bbox_file = "/home/ubuntu/complex_event_video/data/car_turning_traffic2/bbox.json"
with open(bbox_file, 'r') as f:
    maskrcnn_bboxes = json.loads(f.read())

coco_names = {}
with open("/home/ubuntu/complex_event_video/src/ms_coco_classnames.txt") as f:
    lines = f.readlines()
    for i, line in enumerate(lines): 
        coco_names[line.strip()] = i

oid = 0
with open("obj.facts", 'w') as f:
    for frame_id in range(len(maskrcnn_bboxes)):
        res_per_frame = maskrcnn_bboxes["frame_{}.jpg".format(frame_id)]
        for x1, y1, x2, y2, class_name, _ in res_per_frame:
            f.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(oid, frame_id, coco_names[class_name], x1, y1, x2, y2))
            oid += 1
