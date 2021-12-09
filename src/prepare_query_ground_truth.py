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
    predicate = lambda x1, y1, x2, y2 : 1.0 * (x2 - x1) / (y2 - y1) > 2
    return template_one_object(maskrcnn_bboxes, predicate)

def test_b(maskrcnn_bboxes):
    """
    Event definition:
    - Car and pedestrain at the intersection
    - spatial configuration of car: r < 2 and x > 250
    """
    def predicate(x1, y1, x2, y2):
        x = (x1 + x2) / 2
        y = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        r = w / h
        return (r < 2 and x > 250)
    return template_one_object(maskrcnn_bboxes, predicate)

def test_c(maskrcnn_bboxes):
    """
    Event definition:
    - Car and pedestrain at the intersection
    - spatial configuration of car:
    """
    def predicate(x1, y1, x2, y2):
        x = (x1 + x2) / 2
        y = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        r = w / h
        return (r < 2 and x > 250) or (w > 100 and h > 50)
    return template_one_object(maskrcnn_bboxes, predicate)

def test_d(maskrcnn_bboxes):
    def predicate1(x1, y1, x2, y2):
        x = (x1 + x2) / 2
        y = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        r = w / h
        return (r < 2 and x > 250)
    def predicate2(x1, y1, x2, y2):
        x = (x1 + x2) / 2
        y = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        r = w / h
        # return (r > 1.5 and x > 500 and w > 100)
        return (x < 520 and r > 0.3)
    return template_two_objects_b(maskrcnn_bboxes, [predicate1, predicate2])


def test_e(maskrcnn_bboxes):
    """
    Event definition:
    - Car and pedestrain at the intersection
    - Predicate: spatial relationship between car and person
    """
    def predicate(car_box, person_box):
        x, y, x2, y2 = car_box
        xp, yp, x4, y4 = person_box
        w = x2 - x
        h = y2 - y
        wp = x4 - xp
        hp = y4 - yp
        s1 = (y - yp) / h # Relative distance between y1 and y1'
        s2 = (x - xp) / w # Relative distance between x1 and x1'
        s3 = (y + h - yp - hp) / h # Relative distance between y2 and y2'
        s4 = (x + w - xp - wp) / w # Relative distance between x2 and x2'
        s5 = hp / h # Height fraction
        s6 = wp / w # Width fraction
        s7 = (wp * hp) / (w * h) # Area fraction
        s8 = (wp + hp) / (w + h) # Perimeter fraction
        return (s1 > 0.2 and s1 < 0.9 and s2 > 0.2 and s2 < 0.9)
    return template_spatial_relationship(maskrcnn_bboxes, predicate)

def template_one_object(maskrcnn_bboxes, predicate):
    """
    Event definition:
    - Car and pedestrain at the intersection
    - Predicate: spatial configuration of car
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
                if predicate(x1, y1, x2, y2):
                    car_x1, car_y1, car_x2, car_y2 = x1, y1, x2, y2
                    has_car = 1
            if has_car and has_pedestrian:
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

def template_two_objects(maskrcnn_bboxes, predicates):
    """
    Event definition:
    - Two cars in the frame
    - Predicate: spatial configuration of two cars
    """
    pos_frames = []
    pos_frames_per_instance = {}
    n_frames = len(maskrcnn_bboxes)

    for frame_id in range(n_frames):
        res_per_frame = maskrcnn_bboxes["frame_{}.jpg".format(frame_id)]
        has_car_1 = 0
        has_car_2 = 0
        for x1, y1, x2, y2, class_name, _ in res_per_frame:
            if class_name in ["car", "truck"]:
                if predicates[0](x1, y1, x2, y2) and not has_car_1:
                    has_car_1 = 1
                elif predicates[1](x1, y1, x2, y2) and not has_car_2:
                    has_car_2 = 1
            if has_car_1 and has_car_2:
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


def template_two_objects_b(maskrcnn_bboxes, predicates):
    """
    Event definition:
    - Car and pedestrain at the intersection
    - Predicate: spatial configuration of car
    """
    edge_corner_bbox = (367, 345, 540, 418)
    pos_frames = []
    pos_frames_per_instance = {}
    n_frames = len(maskrcnn_bboxes)

    for frame_id in range(n_frames):
        res_per_frame = maskrcnn_bboxes["frame_{}.jpg".format(frame_id)]
        has_car = -1
        has_pedestrian = -1
        for x1, y1, x2, y2, class_name, _ in res_per_frame:
        # for x1, y1, x2, y2, class_name, tid in res_per_frame:
            if (class_name == "person" and isOverlapping(edge_corner_bbox, (x1, y1, x2, y2))):
                if predicates[1](x1, y1, x2, y2):
                    has_pedestrian = 1
                    # has_pedestrian = tid
            elif (class_name in ["car", "truck"] and isInsideIntersection((x1, y1, x2, y2))):
                if predicates[0](x1, y1, x2, y2):
                    has_car = 1
                    # has_car = tid
            if has_car > -1 and has_pedestrian > -1:
                pos_frames.append(frame_id)
                # pos_frames.append([frame_id, has_car, has_pedestrian])
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
    # instance_id = 0
    # start_frame = pos_frames[0][0]
    # current_instance = (pos_frames[0][1], pos_frames[0][2])
    # for frame_id, tid1, tid2 in pos_frames:
    #     if (tid1, tid2) == current_instance:
    #         end_frame = frame_id
    #     else:
    #         pos_frames_per_instance[instance_id] = (start_frame, end_frame+1, 0) # The third value is a flag: 0 represents no detections have been found; 1 represents detection with only one match
    #         instance_id += 1
    #         start_frame = frame_id
    #         end_frame = frame_id
    #         current_instance = (tid1, tid2)
    # pos_frames_per_instance[instance_id] = (start_frame, end_frame+1, 0)
    # pos_frames = [elem[0] for elem in pos_frames]

    return pos_frames, pos_frames_per_instance

def template_spatial_relationship(maskrcnn_bboxes, predicate):
    """
    Event definition:
    - Car and pedestrain at the intersection
    - Predicate: spatial relationship between car and person
    """
    edge_corner_bbox = (367, 345, 540, 418)
    pos_frames = []
    pos_frames_per_instance = {}
    n_frames = len(maskrcnn_bboxes)

    for frame_id in range(n_frames):
        res_per_frame = maskrcnn_bboxes["frame_{}.jpg".format(frame_id)]
        car_boxes = []
        person_boxes = []
        is_positive = False
        for x1, y1, x2, y2, class_name, _ in res_per_frame:
            if (class_name == "person" and isOverlapping(edge_corner_bbox, (x1, y1, x2, y2))):
                person_boxes.append((x1, y1, x2, y2))
            elif (class_name in ["car", "truck"] and isInsideIntersection((x1, y1, x2, y2))):
                car_boxes.append((x1, y1, x2, y2))
        # Check whether contains the target spatial relationship
        for car_box in car_boxes:
            for person_box in person_boxes:
                if predicate(car_box, person_box):
                    pos_frames.append(frame_id)
                    is_positive = True
                    break
            if is_positive:
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