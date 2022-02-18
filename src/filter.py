from utils.utils import isInsideIntersection, isOverlapping
import random
import os
from glob import glob
import yaml
import numpy as np

def construct_spatial_feature(bbox):
    x1, y1, x2, y2 = bbox
    centroid_x = (x1 + x2) / 2
    centroid_y = (y1 + y2) / 2
    width = x2 - x1
    height = y2 - y1
    wh_ratio = width / height
    return np.array([centroid_x, centroid_y, width, height, wh_ratio])

def construct_spatial_feature_two_objects(bbox1, bbox2):
    x11, y11, x21, y21 = bbox1
    centroid_x1 = (x11 + x21) / 2
    centroid_y1 = (y11 + y21) / 2
    width1 = x21 - x11
    height1 = y21 - y11
    wh_ratio1 = width1 / height1

    x12, y12, x22, y22 = bbox2
    centroid_x2 = (x12 + x22) / 2
    centroid_y2 = (y12 + y22) / 2
    width2 = x22 - x12
    height2 = y22 - y12
    wh_ratio2 = width2 / height2
    return np.array([centroid_x1, centroid_y1, width1, height1, wh_ratio1, centroid_x2, centroid_y2, width2, height2, wh_ratio2])

def construct_spatial_feature_spatial_relationship(car_box, person_box):
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
    return np.array([s1, s2, s3, s4, s5, s6, s7, s8])

def turning_car_and_pedestrain_at_intersection(maskrcnn_bboxes):
    n_frames =len(maskrcnn_bboxes)
    spatial_feature_dim = 5
    feature_names = ["x", "y", "w", "h", "r"]
    edge_corner_bbox = (367, 345, 540, 418)
    spatial_features = np.zeros((n_frames, spatial_feature_dim), dtype=np.float64)
    # Filtering stage
    candidates = np.full(n_frames, True, dtype=np.bool)
    for frame_id in range(n_frames):
        res_per_frame = maskrcnn_bboxes["frame_{}.jpg".format(frame_id)]
        has_car = 0
        has_pedestrian = 0
        for x1, y1, x2, y2, class_name, _ in res_per_frame:
            if class_name == "person" and isOverlapping(edge_corner_bbox, (x1, y1, x2, y2)):
                has_pedestrian = 1
            elif class_name in ["car", "truck"] and isInsideIntersection((x1, y1, x2, y2)):
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
            spatial_features[frame_id] = construct_spatial_feature((car_x1, car_y1, car_x2, car_y2))
        else:
            candidates[frame_id] = False
    return feature_names, spatial_feature_dim, spatial_features, candidates

def meva_person_stands_up(maskrcnn_bboxes, video_list, pos_frames, cached=False):
    n_frames = len(maskrcnn_bboxes)
    spatial_feature_dim = 5
    feature_names = ["x", "y", "w", "h", "r"]
    if cached:
        return feature_names, spatial_feature_dim
    spatial_features = np.zeros((n_frames, spatial_feature_dim), dtype=np.float64)
    # Filtering stage
    candidates = np.full(n_frames, True, dtype=np.bool)
    for video_basename, frame_offset, n_local_frames in video_list:
        files = [y for x in os.walk("../data/meva/meva-data-repo/annotation/DIVA-phase-2/MEVA/kitware") for y in glob(os.path.join(x[0], '*.yml'))]
        matching = [f for f in files if video_basename + ".activities" in f]
        assert(len(matching) == 1)
        activities_file = matching[0]
        geom_file = activities_file.replace(".activities", ".geom")
        print("file: ", activities_file)
        with open(activities_file, 'r') as f:
            activities_annotation = yaml.load(f, Loader=yaml.CLoader)
        print("loaded activities file")
        with open(geom_file, 'r') as f:
            geom_annotation = yaml.load(f, Loader=yaml.CLoader)
        print("loaded geom file")
        for local_frame_id in range(n_local_frames):
            res_per_frame = maskrcnn_bboxes[video_basename + "_" + str(local_frame_id)]
            frame_id = frame_offset + local_frame_id
            if frame_id not in pos_frames:
                person_boxes = []
                for x1, y1, x2, y2, class_name, _ in res_per_frame:
                    if class_name == "person":
                        person_boxes.append((x1, y1, x2, y2))
                if person_boxes:
                    spatial_features[frame_id] = construct_spatial_feature(random.choice(person_boxes))
                else:
                    candidates[frame_id] = False
            else:
                for row in activities_annotation:
                    if "act" in row and "person_stands_up" in row["act"]["act2"]:
                        start_frame, end_frame = row["act"]["timespan"][0]["tsr0"]
                        if local_frame_id >= start_frame and local_frame_id <= end_frame:
                            actor_id = row["act"]["actors"][0]["id1"]
                            break
                # - {'geom': {'g0': '170 396 524 1001', 'id0': 1, 'id1': 185, 'keyframe': True, 'ts0': 4513}}
                for row in geom_annotation:
                    if "geom" in row and row["geom"]["id1"] == actor_id and row["geom"]["ts0"] == local_frame_id:
                        spatial_features[frame_id] = construct_spatial_feature(list(map(int, row["geom"]["g0"].split())))
                        break
    return feature_names, spatial_feature_dim, spatial_features, candidates


def meva_person_embraces_person(maskrcnn_bboxes, video_list, pos_frames, cached=False):
    return meva_base_spatial_relationship_same_object(maskrcnn_bboxes, video_list, pos_frames, "person_embraces_person", cached=cached)

def meva_person_enters_vehicle(maskrcnn_bboxes, video_list, pos_frames, cached=False):
    return meva_base_spatial_relationship_different_objects(maskrcnn_bboxes, video_list, pos_frames, "person_enters_vehicle", cached=cached)

def meva_base_spatial_relationship_different_objects(maskrcnn_bboxes, video_list, pos_frames, activity_name, cached=False):
    """Utilize spatial relationship
    """
    n_frames = len(maskrcnn_bboxes)
    spatial_feature_dim = 8
    feature_names = ["delta_x1", "delta_y1", "delta_y2", "delta_x2", "height_ratio", "width_ratio", "area_ratio", "perimeter_ratio"]
    if cached:
        return feature_names, spatial_feature_dim
    spatial_features = np.zeros((n_frames, spatial_feature_dim), dtype=np.float64)
    candidates = np.full(n_frames, True, dtype=np.bool)
    for video_basename, frame_offset, n_local_frames in video_list:
        files = [y for x in os.walk("../data/meva/meva-data-repo/annotation/DIVA-phase-2/MEVA/kitware") for y in glob(os.path.join(x[0], '*.yml'))]
        matching = [f for f in files if video_basename + ".activities" in f]
        assert(len(matching) == 1)
        activities_file = matching[0]
        geom_file = activities_file.replace(".activities", ".geom")
        types_file = activities_file.replace(".activities", ".types")
        print("file: ", activities_file)
        with open(activities_file, 'r') as f:
            activities_annotation = yaml.load(f, Loader=yaml.CLoader)
        print("loaded activities file")
        with open(geom_file, 'r') as f:
            geom_annotation = yaml.load(f, Loader=yaml.CLoader)
        print("loaded geom file")
        with open(types_file, 'r') as f:
            types_annotation = yaml.load(f, Loader=yaml.CLoader)
        print("loaded types file")
        for local_frame_id in range(n_local_frames):
            res_per_frame = maskrcnn_bboxes[video_basename + "_" + str(local_frame_id)]
            frame_id = frame_offset + local_frame_id
            if frame_id not in pos_frames:
                person_boxes = []
                car_boxes = []
                for x1, y1, x2, y2, class_name, _ in res_per_frame:
                    if class_name == "person":
                        person_boxes.append((x1, y1, x2, y2))
                    elif class_name in ["car", "bus", "truck"]:
                        car_boxes.append((x1, y1, x2, y2))
                if person_boxes and car_boxes:
                    # Candidate, negative frame
                    person_box = random.choice(person_boxes)
                    car_box = random.choice(car_boxes)
                    spatial_features[frame_id] = construct_spatial_feature_spatial_relationship(person_box, car_box)
                else:
                    candidates[frame_id] = False
            else:
                for row in activities_annotation:
                    if "act" in row and activity_name in row["act"]["act2"]:
                        # start_frame, end_frame = row["act"]["timespan"][0]["tsr0"]
                        start_frame = max(row["act"]["actors"][0]["timespan"][0]["tsr0"][0], row["act"]["actors"][1]["timespan"][0]["tsr0"][0])
                        end_frame = min(row["act"]["actors"][0]["timespan"][0]["tsr0"][1], row["act"]["actors"][1]["timespan"][0]["tsr0"][1])
                        if local_frame_id >= start_frame and local_frame_id <= end_frame:
                            # [{'id1': 82, 'timespan': [{'tsr0': [5167, 5568]}]}, {'id1': 83, 'timespan': [{'tsr0': [5167, 5568]}]}]
                            actor_id1 = row["act"]["actors"][0]["id1"]
                            actor_id2 = row["act"]["actors"][1]["id1"]
                            break
                # - {'geom': {'g0': '170 396 524 1001', 'id0': 1, 'id1': 185, 'keyframe': True, 'ts0': 4513}}
                for row in types_annotation:
                    if "types" in row and row["types"]["id1"] == actor_id1:
                        actor_type1 = list(row["types"]["cset3"].keys())[0]
                    elif "types" in row and row["types"]["id1"] == actor_id2:
                        actor_type2 = list(row["types"]["cset3"].keys())[0]
                if actor_type1 == "vehicle":
                    assert(actor_type2 == "person")
                    temp_id = actor_id1
                    actor_id1 = actor_id2
                    actor_id2 = temp_id
                actor1_box, actor2_box = None, None
                for row in geom_annotation:
                    if "geom" in row and row["geom"]["id1"] == actor_id1 and row["geom"]["ts0"] == local_frame_id:
                        actor1_box = list(map(int, row["geom"]["g0"].split()))
                    elif "geom" in row and row["geom"]["id1"] == actor_id2 and row["geom"]["ts0"] == local_frame_id:
                        actor2_box = list(map(int, row["geom"]["g0"].split()))
                    if actor1_box and actor2_box:
                        spatial_features[frame_id] = construct_spatial_feature_spatial_relationship(actor1_box, actor2_box)
                        break
    return feature_names, spatial_feature_dim, spatial_features, candidates


def meva_base_spatial_relationship_same_object(maskrcnn_bboxes, video_list, pos_frames, activity_name, cached=False):
    """Utilize spatial relationship
    """
    n_frames = len(maskrcnn_bboxes)
    spatial_feature_dim = 8
    feature_names = ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"]
    if cached:
        return feature_names, spatial_feature_dim
    spatial_features = np.zeros((n_frames, spatial_feature_dim), dtype=np.float64)
    candidates = np.full(n_frames, True, dtype=np.bool)
    for video_basename, frame_offset, n_local_frames in video_list:
        files = [y for x in os.walk("../data/meva/meva-data-repo/annotation/DIVA-phase-2/MEVA/kitware") for y in glob(os.path.join(x[0], '*.yml'))]
        matching = [f for f in files if video_basename + ".activities" in f]
        assert(len(matching) == 1)
        activities_file = matching[0]
        geom_file = activities_file.replace(".activities", ".geom")
        print("file: ", activities_file)
        with open(activities_file, 'r') as f:
            activities_annotation = yaml.load(f, Loader=yaml.CLoader)
        print("loaded activities file")
        with open(geom_file, 'r') as f:
            geom_annotation = yaml.load(f, Loader=yaml.CLoader)
        print("loaded geom file")
        for local_frame_id in range(n_local_frames):
            res_per_frame = maskrcnn_bboxes[video_basename + "_" + str(local_frame_id)]
            frame_id = frame_offset + local_frame_id
            if frame_id not in pos_frames:
                person_boxes = []
                for x1, y1, x2, y2, class_name, _ in res_per_frame:
                    if class_name == "person":
                        person_boxes.append((x1, y1, x2, y2))
                if len(person_boxes) >= 2:
                    # Candidate, negative frame
                    person_box1, person_box2 = random.sample(person_boxes, 2)
                    spatial_features[frame_id] = construct_spatial_feature_spatial_relationship(person_box1, person_box2)
                else:
                    candidates[frame_id] = False
            else:
                for row in activities_annotation:
                    if "act" in row and activity_name in row["act"]["act2"]:
                        start_frame, end_frame = row["act"]["timespan"][0]["tsr0"]
                        if local_frame_id >= start_frame and local_frame_id <= end_frame:
                            # [{'id1': 82, 'timespan': [{'tsr0': [5167, 5568]}]}, {'id1': 83, 'timespan': [{'tsr0': [5167, 5568]}]}]
                            actor_id1, actor_id2 = random.sample(row["act"]["actors"], 2)
                            actor_id1 = actor_id1["id1"]
                            actor_id2 = actor_id2["id1"]
                            break
                # - {'geom': {'g0': '170 396 524 1001', 'id0': 1, 'id1': 185, 'keyframe': True, 'ts0': 4513}}
                actor1_box, actor2_box = None, None
                for row in geom_annotation:
                    if "geom" in row and row["geom"]["id1"] == actor_id1 and row["geom"]["ts0"] == local_frame_id:
                        actor1_box = list(map(int, row["geom"]["g0"].split()))
                    elif "geom" in row and row["geom"]["id1"] == actor_id2 and row["geom"]["ts0"] == local_frame_id:
                        actor2_box = list(map(int, row["geom"]["g0"].split()))
                    if actor1_box and actor2_box:
                        spatial_features[frame_id] = construct_spatial_feature_spatial_relationship(actor1_box, actor2_box)
                        break
    return feature_names, spatial_feature_dim, spatial_features, candidates


def test_a(maskrcnn_bboxes):
    predicate = lambda x1, y1, x2, y2 : 1.0 * (x2 - x1) / (y2 - y1) > 2
    return template_one_object(maskrcnn_bboxes, predicate)

def test_b(maskrcnn_bboxes):
    def predicate(x1, y1, x2, y2):
        x = (x1 + x2) / 2
        y = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        r = w / h
        return (r < 2 and x > 250)
    return template_one_object(maskrcnn_bboxes, predicate)

def test_c(maskrcnn_bboxes):
    def predicate(x1, y1, x2, y2):
        x = (x1 + x2) / 2
        y = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        r = w / h
        return (r < 2 and x > 250) or (w > 100 and h > 50)
    return template_one_object(maskrcnn_bboxes, predicate)

def template_one_object(maskrcnn_bboxes, predicate):
    n_frames =len(maskrcnn_bboxes)
    spatial_feature_dim = 5
    feature_names = ["x", "y", "w", "h", "r"]
    edge_corner_bbox = (367, 345, 540, 418)
    spatial_features = np.zeros((n_frames, spatial_feature_dim), dtype=np.float64)
    # Filtering stage
    candidates = np.full(n_frames, True, dtype=np.bool)
    for frame_id in range(n_frames):
        res_per_frame = maskrcnn_bboxes["frame_{}.jpg".format(frame_id)]
        car_boxes = []
        person_boxes = []
        car_satisfies_predicate_boxes = []
        for x1, y1, x2, y2, class_name, _ in res_per_frame:
            if class_name == "person" and isOverlapping(edge_corner_bbox, (x1, y1, x2, y2)):
                person_boxes.append((x1, y1, x2, y2))
            elif class_name in ["car", "truck"] and isInsideIntersection((x1, y1, x2, y2)):
                car_boxes.append((x1, y1, x2, y2))
                if predicate(x1, y1, x2, y2):
                    car_satisfies_predicate_boxes.append((x1, y1, x2, y2))
        if car_boxes and person_boxes:
            if car_satisfies_predicate_boxes:
                spatial_features[frame_id] = construct_spatial_feature(random.choice(car_satisfies_predicate_boxes))
            else:
                spatial_features[frame_id] = construct_spatial_feature(random.choice(car_boxes))
        else:
            candidates[frame_id] = False
    return feature_names, spatial_feature_dim, spatial_features, candidates


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
        s1 = (x - xp) / w
        s2 = (y - yp) / h
        s3 = (y + h - yp - hp) / h
        s4 = (x + w - xp - wp) / w
        s5 = hp / h
        s6 = wp / w
        s7 = (wp * hp) / (w * h)
        s8 = (wp + hp) / (w + h)
        return (s1 > 0.2 and s1 < 0.9 and s2 > 0.2 and s2 < 0.9)
    return template_spatial_relationship(maskrcnn_bboxes, predicate)

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


def template_two_objects_b(maskrcnn_bboxes, predicates):
    n_frames =len(maskrcnn_bboxes)
    spatial_feature_dim = 10
    feature_names = ["x1", "y1", "w1", "h1", "r1", "x2", "y2", "w2", "h2", "r2"]
    edge_corner_bbox = (367, 345, 540, 418)
    spatial_features = np.zeros((n_frames, spatial_feature_dim), dtype=np.float64)
    # Filtering stage
    candidates = np.full(n_frames, True, dtype=np.bool)
    for frame_id in range(n_frames):
        res_per_frame = maskrcnn_bboxes["frame_{}.jpg".format(frame_id)]
        car_boxes = []
        person_boxes = []
        car_satisfies_predicate_boxes = []
        person_satisfies_predicate_boxes = []
        for x1, y1, x2, y2, class_name, _ in res_per_frame:
            if class_name == "person" and isOverlapping(edge_corner_bbox, (x1, y1, x2, y2)):
                person_boxes.append((x1, y1, x2, y2))
                if predicates[1](x1, y1, x2, y2):
                    person_satisfies_predicate_boxes.append((x1, y1, x2, y2))
            elif class_name in ["car", "truck"] and isInsideIntersection((x1, y1, x2, y2)):
                car_boxes.append((x1, y1, x2, y2))
                if predicates[0](x1, y1, x2, y2):
                    car_satisfies_predicate_boxes.append((x1, y1, x2, y2))
        if car_satisfies_predicate_boxes and person_satisfies_predicate_boxes:
            spatial_features[frame_id] = construct_spatial_feature_two_objects(random.choice(car_satisfies_predicate_boxes), random.choice(person_satisfies_predicate_boxes))
        else:
            if car_boxes and person_boxes:
                spatial_features[frame_id] = construct_spatial_feature_two_objects(random.choice(car_boxes), random.choice(person_boxes))
            else:
                candidates[frame_id] = False
    return feature_names, spatial_feature_dim, spatial_features, candidates


def template_spatial_relationship(maskrcnn_bboxes, predicate):
    n_frames =len(maskrcnn_bboxes)
    spatial_feature_dim = 8
    feature_names = ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"]
    edge_corner_bbox = (367, 345, 540, 418)
    spatial_features = np.zeros((n_frames, spatial_feature_dim), dtype=np.float64)
    # Filtering stage
    candidates = np.full(n_frames, True, dtype=np.bool)
    for frame_id in range(n_frames):
        res_per_frame = maskrcnn_bboxes["frame_{}.jpg".format(frame_id)]
        car_boxes = []
        person_boxes = []
        relationship_boxes = []
        for x1, y1, x2, y2, class_name, _ in res_per_frame:
            if class_name == "person" and isOverlapping(edge_corner_bbox, (x1, y1, x2, y2)):
                person_boxes.append((x1, y1, x2, y2))
            elif class_name in ["car", "truck"] and isInsideIntersection((x1, y1, x2, y2)):
                car_boxes.append((x1, y1, x2, y2))
        if car_boxes and person_boxes:
            # Check whether contains the target spatial relationship
            for car_box in car_boxes:
                for person_box in person_boxes:
                    if predicate(car_box, person_box):
                        relationship_boxes.append((car_box, person_box))
            if relationship_boxes:
                car_box, person_box = random.choice(relationship_boxes)
                spatial_features[frame_id] = construct_spatial_feature_spatial_relationship(car_box, person_box)
            else:
                spatial_features[frame_id] = construct_spatial_feature_spatial_relationship(random.choice(car_boxes), random.choice(person_boxes))
        else:
            candidates[frame_id] = False
    return feature_names, spatial_feature_dim, spatial_features, candidates