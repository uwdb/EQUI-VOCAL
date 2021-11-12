import csv

from pandas.core import frame 
from utils.utils import isInsideIntersection, isOverlapping

def turning_car_and_pedestrain_at_intersection():
    pos_frames = []
    pos_frames_per_instance = {}
    with open("/home/ubuntu/complex_event_video/data/annotation.csv", "r") as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)
        for i, row in enumerate(csvreader):
            start_frame, end_frame = int(row[0]), int(row[1])
            pos_frames += list(range(start_frame, end_frame+1))
            pos_frames_per_instance[i] = (start_frame, end_frame+1, 0) # The third value is a flag: 0 represents no detections have been found; 1 represents detection with only one match
    return pos_frames, pos_frames_per_instance

def test_a(maskrcnn_bboxes):
    """
    Event definition:
    - Car and pedestrain at the intersection
    - spatial configuration of car: ratio r > 2
    """
    edge_corner_bbox = (367, 345, 540, 418)
    pos_frames = []
    pos_frames_per_instance = {}
    n_frames = len(maskrcnn_bboxes)

    for frame_id in range(n_frames):
        res_per_frame = maskrcnn_bboxes["frame_{}.jpg".format(frame_id)]
        has_car = 0
        has_pedestrian = 0
        for x1, y1, x2, y2, class_name, _ in res_per_frame:
            if (class_name == "person" and isOverlapping(edge_corner_bbox, (x1, y1, x2, y2))):
                has_pedestrian = 1
            elif (class_name in ["car", "truck"] and isInsideIntersection((x1, y1, x2, y2))):
                if 1.0 * (x2 - x1) / (y2 - y1) > 2: # If ratio r > 2
                    car_x1, car_y1, car_x2, car_y2 = x1, y1, x2, y2
                    has_car = 1
            if has_car and has_pedestrian:
                if frame_id == 203:
                    print(car_x1, car_y1, car_x2, car_y2)
                    exit(1)
                pos_frames.append(frame_id)
                break
    instance_id = 0
    current_frame = pos_frames[0] - 1
    start_frame = pos_frames[0]
    for frame_id in pos_frames:
        if frame_id == current_frame + 1:
            current_frame += 1
        else: 
            pos_frames_per_instance[instance_id] = (start_frame, current_frame+1, 0) # The third value is a flag: 0 represents no detections have been found; 1 represents detection with only one match
            instance_id += 1
            start_frame = frame_id
            current_frame = frame_id
    pos_frames_per_instance[instance_id] = (start_frame, current_frame+1, 0)
    return pos_frames, pos_frames_per_instance