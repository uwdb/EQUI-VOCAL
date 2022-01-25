# Read in bbox info
import json


bbox_file = "/home/ubuntu/complex_event_video/data/car_turning_traffic2/bbox.json"
with open(bbox_file, 'r') as f:
    maskrcnn_bboxes = json.loads(f.read())

car_count = 0
person_count = 0
frame_count = 0
for key, val in maskrcnn_bboxes.items():
    frame_count += 1
    has_car = False
    has_person = False
    for obj in val:
        if obj[4] == "bicycle":
            has_car = True
        if obj[4] == "motorbike":
            has_person = True
    if has_car:
        car_count += 1
    if has_person:
        person_count += 1
print(frame_count, car_count, person_count)

