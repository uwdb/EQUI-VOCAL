from utils.utils import isInsideIntersection, isOverlapping

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
    return base_test(res_per_frame, frame_id, predicate)

def test_b(res_per_frame, frame_id):
    def predicate(x1, y1, x2, y2):
        x = (x1 + x2) / 2
        y = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        r = w / h
        return (r < 2 and x > 250)
    return base_test(res_per_frame, frame_id, predicate)

def test_c(res_per_frame, frame_id):
    def predicate(x1, y1, x2, y2):
        x = (x1 + x2) / 2
        y = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        r = w / h
        return (r < 2 and x > 250) or (w > 100 and h > 50)
    return base_test(res_per_frame, frame_id, predicate)

def base_test(res_per_frame, frame_id, predicate):
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