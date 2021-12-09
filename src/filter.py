from utils.utils import isInsideIntersection, isOverlapping
import random

def car_and_pedestrain_at_intersection(res_per_frame, frame_id):
    edge_corner_bbox = (367, 345, 540, 418)
    has_car = 0
    has_pedestrian = 0
    for x1, y1, x2, y2, class_name, score in res_per_frame:
        if (class_name == "person" and isOverlapping(edge_corner_bbox, (x1, y1, x2, y2))):
            has_pedestrian = 1
        elif (class_name in ["car", "truck"] and isInsideIntersection((x1, y1, x2, y2))):
            # Watch the video and identify the correct cars. Hardcode the correct car bbox to use.
            if frame_id >= 14043 and frame_id <= 14079 and (x1 < 500 or x1 > 800):
                continue
            if frame_id >= 15312 and frame_id <= 15365 and y2 < 450:
                continue
            if frame_id >= 15649 and frame_id <= 15722 and x1 < 200:
                continue
            if frame_id >= 16005 and frame_id <= 16044 and y2 < 430:
                continue
            if frame_id >= 16045 and frame_id <= 16072 and x1 < 250:
                continue
            if frame_id >= 16073 and frame_id <= 16090 and y2 < 450:
                continue
            if frame_id >= 16091 and frame_id <= 16122 and x1 < 245:
                continue
            if frame_id >= 16123 and frame_id <= 16153 and x1 > 500:
                continue
            if frame_id >= 22375 and frame_id <= 22430 and y2 < 500:
                continue
            has_car = 1
            car_x1, car_y1, car_x2, car_y2 = x1, y1, x2, y2
        if has_car and has_pedestrian:
            return True, (car_x1, car_y1, car_x2, car_y2)
    return False, None

def test_a(res_per_frame, frame_id):
    predicate = lambda x1, y1, x2, y2 : 1.0 * (x2 - x1) / (y2 - y1) > 2
    return template_one_object(res_per_frame, frame_id, predicate)

def test_b(res_per_frame, frame_id):
    def predicate(x1, y1, x2, y2):
        x = (x1 + x2) / 2
        y = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        r = w / h
        return (r < 2 and x > 250)
    return template_one_object(res_per_frame, frame_id, predicate)

def test_c(res_per_frame, frame_id):
    def predicate(x1, y1, x2, y2):
        x = (x1 + x2) / 2
        y = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        r = w / h
        return (r < 2 and x > 250) or (w > 100 and h > 50)
    return template_one_object(res_per_frame, frame_id, predicate)


def template_one_object(res_per_frame, frame_id, predicate):
    edge_corner_bbox = (367, 345, 540, 418)
    has_car_and_satisfies_predicate = 0
    has_car = 0
    has_pedestrian = 0
    for x1, y1, x2, y2, class_name, score in res_per_frame:
        if (class_name == "person" and isOverlapping(edge_corner_bbox, (x1, y1, x2, y2))):
            has_pedestrian = 1
        elif (class_name in ["car", "truck"] and isInsideIntersection((x1, y1, x2, y2))):
            car_x1, car_y1, car_x2, car_y2 = x1, y1, x2, y2
            has_car = 1
            if predicate(x1, y1, x2, y2):
                has_car_and_satisfies_predicate = 1
                acar_x1, acar_y1, acar_x2, acar_y2 = x1, y1, x2, y2
        if has_car_and_satisfies_predicate and has_pedestrian:
            return True, (acar_x1, acar_y1, acar_x2, acar_y2)
    if has_car and has_pedestrian:
        return True, (car_x1, car_y1, car_x2, car_y2)
    return False, None


def test_d(res_per_frame):
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
        return (x < 520 and r > 0.3)
    return template_two_objects_b(res_per_frame, [predicate1, predicate2])

def test_e(res_per_frame):
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
        s1 = (x - xp) / w
        s2 = (y - yp) / h
        s3 = (y + h - yp - hp) / h
        s4 = (x + w - xp - wp) / w
        s5 = hp / h
        s6 = wp / w
        s7 = (wp * hp) / (w * h)
        s8 = (wp + hp) / (w + h)
        return (s1 > 0.2 and s1 < 0.9 and s2 > 0.2 and s2 < 0.9)
    return template_spatial_relationship(res_per_frame, predicate)

def template_two_objects(res_per_frame, predicates):
    has_car_1 = 0
    has_car_2 = 0
    car_box = []
    for x1, y1, x2, y2, class_name, _ in res_per_frame:
        if class_name in ["car", "truck"]:
            car_box.append((x1, y1, x2, y2))
            if predicates[0](x1, y1, x2, y2) and not has_car_1:
                has_car_1 = 1
                car1_box = (x1, y1, x2, y2)
            elif predicates[1](x1, y1, x2, y2) and not has_car_2:
                car2_box = (x1, y1, x2, y2)
                has_car_2 = 1
        if has_car_1 and has_car_2:
            return True, car1_box, car2_box
    if len(car_box) >= 2:
        box1, box2 = random.sample(car_box, 2)
        return True, box1, box2
    return False, None, None


def template_two_objects_b(res_per_frame, predicates):
    edge_corner_bbox = (367, 345, 540, 418)
    has_car_and_satisfies_predicate = 0
    has_car = 0
    has_pedestrian_and_satisfies_predicate = 0
    has_pedestrian = 0
    for x1, y1, x2, y2, class_name, _ in res_per_frame:
        if class_name == "person" and isOverlapping(edge_corner_bbox, (x1, y1, x2, y2)):
            person_box_candidate = (x1, y1, x2, y2)
            has_pedestrian = 1
            if predicates[1](x1, y1, x2, y2):
                person_box_pos = (x1, y1, x2, y2)
                has_pedestrian_and_satisfies_predicate = 1
        elif class_name in ["car", "truck"] and isInsideIntersection((x1, y1, x2, y2)):
            car_box_candidate = (x1, y1, x2, y2)
            has_car = 1
            if predicates[0](x1, y1, x2, y2):
                car_box_pos = (x1, y1, x2, y2)
                has_car_and_satisfies_predicate = 1
        if has_car_and_satisfies_predicate and has_pedestrian_and_satisfies_predicate:
            return True, car_box_pos, person_box_pos
    if has_car and has_pedestrian:
        return True, car_box_candidate, person_box_candidate
    return False, None, None


def template_spatial_relationship(res_per_frame, predicate):
    edge_corner_bbox = (367, 345, 540, 418)
    car_boxes = []
    person_boxes = []
    for x1, y1, x2, y2, class_name, _ in res_per_frame:
        if (class_name == "person" and isOverlapping(edge_corner_bbox, (x1, y1, x2, y2))):
            person_boxes.append((x1, y1, x2, y2))
        elif (class_name in ["car", "truck"] and isInsideIntersection((x1, y1, x2, y2))):
            car_boxes.append((x1, y1, x2, y2))
    # Check whether contains the target spatial relationship
    for car_box in car_boxes:
        for person_box in person_boxes:
            if predicate(car_box, person_box):
                return True, car_box, person_box
    if car_boxes and person_boxes:
        return True, random.choice(car_boxes), random.choice(person_boxes)
    return False, None, None