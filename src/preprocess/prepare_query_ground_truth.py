import csv
from pandas.core import frame
from utils.utils import isInsideIntersection, isOverlapping
import os
from glob import glob
import yaml
import math
import json
import numpy as np
import itertools
import random
from filter import construct_spatial_feature_spatial_relationship

class PrepareGroundTruthMixin:
    def turning_car_and_pedestrain_at_intersection(self):
        pos_frames = []
        pos_frames_per_instance = {}
        with open("/gscratch/balazinska/enhaoz/complex_event_video/data/annotation.csv", "r") as csvfile:
            csvreader = csv.reader(csvfile)
            next(csvreader)
            for i, row in enumerate(csvreader):
                start_frame, end_frame = int(row[0]), int(row[1])
                pos_frames += list(range(start_frame, end_frame+1))
                pos_frames_per_instance[i] = (start_frame, end_frame+1, 0) # The third value is a flag: 0 represents no detections have been found; 1 represents detection with only one match
        return pos_frames, pos_frames_per_instance

    def meva_person_stands_up(self):
        return self.meva_base("person_stands_up")

    def meva_person_embraces_person(self):
        return self.meva_base("person_embraces_person")

    def meva_person_enters_vehicle(self):
        return self.meva_base_b("person_enters_vehicle")

    def meva_base(self, activity_name):
        pos_frames = set()
        pos_frames_per_instance = {}
        files = [y for x in os.walk("/gscratch/balazinska/enhaoz/complex_event_video/data/meva/meva-data-repo/annotation/DIVA-phase-2/MEVA/kitware") for y in glob(os.path.join(x[0], '*.yml'))]
        num_instance = 0
        for video_basename, frame_offset, _ in self.video_list:
            matching = [f for f in files if video_basename + ".activities" in f]
            assert(len(matching) == 1)
            file = matching[0]
            # Read in bbox info
            with open(file, 'r') as f:
                annotation = yaml.safe_load(f)
                for row in annotation:
                    if "act" in row and activity_name in row["act"]["act2"]:
                        start_frame, end_frame = row["act"]["timespan"][0]["tsr0"]
                        for i in range(start_frame, end_frame+1):
                            pos_frames.add(frame_offset + i)
                        pos_frames_per_instance[num_instance] = (frame_offset + start_frame, frame_offset + end_frame + 1, 0)
                        num_instance += 1
        return sorted(pos_frames), pos_frames_per_instance


    def meva_base_b(self, activity_name):
        pos_frames = set()
        pos_frames_per_instance = {}
        files = [y for x in os.walk("/gscratch/balazinska/enhaoz/complex_event_video/data/meva/meva-data-repo/annotation/DIVA-phase-2/MEVA/kitware") for y in glob(os.path.join(x[0], '*.yml'))]
        num_instance = 0
        for video_basename, frame_offset, _ in self.video_list:
            matching = [f for f in files if video_basename + ".activities" in f]
            assert(len(matching) == 1)
            file = matching[0]
            # Read in bbox info
            with open(file, 'r') as f:
                annotation = yaml.safe_load(f)
                for row in annotation:
                    if "act" in row and activity_name in row["act"]["act2"]:
                        start_frame = max(row["act"]["actors"][0]["timespan"][0]["tsr0"][0], row["act"]["actors"][1]["timespan"][0]["tsr0"][0])
                        end_frame = min(row["act"]["actors"][0]["timespan"][0]["tsr0"][1], row["act"]["actors"][1]["timespan"][0]["tsr0"][1])
                        for i in range(start_frame, end_frame+1):
                            pos_frames.add(frame_offset + i)
                        pos_frames_per_instance[num_instance] = (frame_offset + start_frame, frame_offset + end_frame + 1, 0)
                        num_instance += 1
        return sorted(pos_frames), pos_frames_per_instance


    def test_a(self):
        """
        Event definition:
        - Car and pedestrain at the intersection
        - spatial configuration of car: ratio r > 2
        """
        predicate = lambda x1, y1, x2, y2 : 1.0 * (x2 - x1) / (y2 - y1) > 2
        return self.template_one_object(predicate)

    def test_b(self):
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
        return self.template_one_object(predicate)

    def test_c(self):
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
        return self.template_one_object(predicate)

    def test_d(self):
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
        return self.template_two_objects_b([predicate1, predicate2])


    def test_e(self):
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
            s1 = (y - yp) / h # delta y1
            s2 = (x - xp) / w # delta x1
            s3 = (y + h - yp - hp) / h # delta y2
            s4 = (x + w - xp - wp) / w # delta x2
            s5 = hp / h # Height ratio
            s6 = wp / w # Width ratio
            s7 = (wp * hp) / (w * h) # Area ratio
            s8 = (wp + hp) / (w + h) # Perimeter ratio
            return (s1 > 0.2 and s1 < 0.9 and s2 > 0.2 and s2 < 0.9)
        return self.template_spatial_relationship(predicate)

    def template_one_object(self, predicate):
        """
        Event definition:
        - Car and pedestrain at the intersection
        - Predicate: spatial configuration of car
        """
        edge_corner_bbox = (367, 345, 540, 418)
        pos_frames = []
        pos_frames_per_instance = {}
        n_frames = len(self.maskrcnn_bboxes)

        for frame_id in range(n_frames):
            res_per_frame = self.maskrcnn_bboxes["frame_{}.jpg".format(frame_id)]
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

    def template_two_objects(self, predicates):
        """
        Event definition:
        - Two cars in the frame
        - Predicate: spatial configuration of two cars
        """
        pos_frames = []
        pos_frames_per_instance = {}
        n_frames = len(self.maskrcnn_bboxes)

        for frame_id in range(n_frames):
            res_per_frame = self.maskrcnn_bboxes["frame_{}.jpg".format(frame_id)]
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


    def template_two_objects_b(self, predicates):
        """
        Event definition:
        - Car and pedestrain at the intersection
        - Predicate: spatial configuration of car
        """
        edge_corner_bbox = (367, 345, 540, 418)
        pos_frames = []
        pos_frames_per_instance = {}
        n_frames = len(self.maskrcnn_bboxes)

        for frame_id in range(n_frames):
            res_per_frame = self.maskrcnn_bboxes["frame_{}.jpg".format(frame_id)]
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

    def template_spatial_relationship(self, predicate):
        """
        Event definition:
        - Car and pedestrain at the intersection
        - Predicate: spatial relationship between car and person
        """
        edge_corner_bbox = (367, 345, 540, 418)
        pos_frames = []
        pos_frames_per_instance = {}
        n_frames = len(self.maskrcnn_bboxes)

        for frame_id in range(n_frames):
            res_per_frame = self.maskrcnn_bboxes["frame_{}.jpg".format(frame_id)]
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

    def clevrer_collision(self):
        pos_frames = set()
        pos_frames_per_instance = {}
        num_instance = 0
        for video_basename, frame_offset, _ in self.video_list:
            file = "/gscratch/balazinska/enhaoz/complex_event_video/data/clevrer/processed_proposals/sim_{}.json".format(video_basename)
            # Read in bbox info
            with open(file, 'r') as f:
                data = json.load(f)
            collisions = data["ground_truth"]["collisions"]
            for collision in collisions:
                # Option 1: collision could span multiple frames
                # start_frame = max(collision["frame"] - 3, 0)
                # end_frame = min(collision["frame"] + 3, 127)
                # for i in range(start_frame, end_frame+1):
                #     pos_frames.add(frame_offset + i)
                # pos_frames_per_instance[num_instance] = (frame_offset + start_frame, frame_offset + end_frame + 1, 0)

                # Option 2: collision is 1 frame
                pos_frames.add(frame_offset + collision["frame"])
                pos_frames_per_instance[num_instance] = (frame_offset + collision["frame"], frame_offset + collision["frame"] + 1, 0) # The third value is a flag: 0 represents no detections have been found; 1 represents detection with only one match
                num_instance += 1
        return sorted(pos_frames), pos_frames_per_instance

    def clevrer_collision_evaluation(self):
        n_frames_evaluation = len(self.maskrcnn_bboxes_evaluation)
        Y_evaluation = np.zeros(n_frames_evaluation, dtype=int)
        # spatial_features = np.zeros((n_frames_evaluation, spatial_feature_dim), dtype=np.float64)
        spatial_features = [] # List of lists. Row count: the total number of pairwaise relationships across all videos. Column count: dimension of spatial features (8)
        Y_pair_level_evaluation = []
        raw_data_pair_level_evaluation = []
        feature_index = [] # A video can also have no pairwise relationships. In this case, the feature_index for that vid is empty.
        for video_basename, frame_offset, _ in self.video_list_evaluation:
            file = "/gscratch/balazinska/enhaoz/complex_event_video/data/clevrer/processed_proposals/sim_{}.json".format(video_basename)
            # Read in bbox info
            with open(file, 'r') as f:
                data = json.load(f)
            collisions = data["ground_truth"]["collisions"]
            objects = data["ground_truth"]["objects"]

            for frame_id in range(128):
                res_per_frame = self.maskrcnn_bboxes_evaluation["{}_{}".format(video_basename, frame_id)]
                if len(res_per_frame) < 2:
                    continue

                # find all positive pairs. In most cases it should be only one
                positive_pairs = []
                for collision in collisions:
                    obj1 = None
                    obj2 = None
                    # if frame_id <= collision["frame"] + 3 and frame_id >= collision["frame"] - 3:
                    if frame_id == collision["frame"]:
                        pos_obj_id1 = collision["object"][0]
                        pos_obj_id2 = collision["object"][1]
                        for obj in objects:
                            if obj["id"] == pos_obj_id1:
                                pos_obj1 = obj
                            elif obj["id"] == pos_obj_id2:
                                pos_obj2 = obj
                        # obj: (x1, y1, x2, y2, material, color, shape)
                        for obj in res_per_frame:
                            if obj[4] == pos_obj1["material"] and obj[5] == pos_obj1["color"] and obj[6] == pos_obj1["shape"]:
                                obj1 = obj
                            elif obj[4] == pos_obj2["material"] and obj[5] == pos_obj2["color"] and obj[6] == pos_obj2["shape"]:
                                obj2 = obj
                        if obj1 and obj2:
                            Y_evaluation[frame_offset + frame_id] = 1
                            positive_pairs.append([obj1, obj2])
                            positive_pairs.append([obj2, obj1])
                for obj1, obj2 in itertools.combinations(res_per_frame, 2):
                    spatial_features.append(construct_spatial_feature_spatial_relationship(obj1[:4], obj2[:4]))
                    feature_index.append(frame_offset + frame_id)
                    # Construct Y_pair_level_evaluation
                    Y_pair_level_evaluation.append(0)
                    raw_data_pair_level_evaluation.append([video_basename, frame_id, obj1, obj2])
                    if Y_evaluation[frame_offset + frame_id] == 1:
                        for pos_obj1, pos_obj2 in positive_pairs:
                            if obj1[4] == pos_obj1[4] and obj1[5] == pos_obj1[5] and obj1[6] == pos_obj1[6] and obj2[4] == pos_obj2[4] and obj2[5] == pos_obj2[5] and obj2[6] == pos_obj2[6]:
                                Y_pair_level_evaluation[-1] = 1
                                break
        spatial_features = np.stack(spatial_features, axis=0)
        Y_pair_level_evaluation = np.asarray(Y_pair_level_evaluation)
        feature_index = np.asarray(feature_index)
        print("length of spatial_features: {}; Y_pair_level_evaluation : {}; feature_index: {}".format(len(spatial_features), len(Y_pair_level_evaluation), len(feature_index)))
        return spatial_features, Y_evaluation, Y_pair_level_evaluation, feature_index, raw_data_pair_level_evaluation

    def clevrer_far(self, cached=False):
        pos_frames = []
        pos_frames_per_instance = {}
        num_instance = 0
        n_frames = len(self.maskrcnn_bboxes)
        spatial_feature_dim = 8
        feature_names = ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"]
        if cached:
            return feature_names, spatial_feature_dim
        spatial_features = np.zeros((n_frames, spatial_feature_dim), dtype=np.float64)
        Y = np.zeros(n_frames, dtype=int)
        # Filtering stage
        candidates = np.full(n_frames, True, dtype=np.bool)

        for video_basename, frame_offset, _ in self.video_list:
            file = "/gscratch/balazinska/enhaoz/complex_event_video/data/clevrer/processed_proposals/sim_{}.json".format(video_basename)
            # Read in bbox info
            with open(file, 'r') as f:
                data = json.load(f)

            objects = data["ground_truth"]["objects"]

            for frame_id in range(128):
                res_per_frame = self.maskrcnn_bboxes["{}_{}".format(video_basename, frame_id)]
                if len(res_per_frame) < 2:
                    continue
                # find all positive pairs. In most cases it should be only one
                positive_pairs = []

                for obj1, obj2 in itertools.combinations(res_per_frame, 2):
                    if self.obj_distance(obj1[:4], obj2[:4]) <= self.thresh:
                        positive_pairs.append([obj1, obj2])
                if len(positive_pairs) > 0:
                    obj1, obj2 = random.choice(positive_pairs)
                    spatial_features[frame_offset + frame_id] = construct_spatial_feature_spatial_relationship(obj1[:4], obj2[:4])
                    Y[frame_offset + frame_id] = 1
                    pos_frames.append(frame_offset + frame_id)
                    pos_frames_per_instance[num_instance] = (frame_offset + frame_id, frame_offset + frame_id + 1, 0) # The third value is a flag: 0 represents no detections have been found; 1 represents detection with only one match
                    num_instance += 1
                else:
                    obj1, obj2 = random.sample(res_per_frame, 2)
                    spatial_features[frame_offset + frame_id] = construct_spatial_feature_spatial_relationship(obj1[:4], obj2[:4])

        return feature_names, spatial_feature_dim, spatial_features, candidates, Y, pos_frames, pos_frames_per_instance


    def clevrer_far_evaluation(self):
        n_frames_evaluation = len(self.maskrcnn_bboxes_evaluation)
        Y_evaluation = np.zeros(n_frames_evaluation, dtype=int)
        spatial_features = [] # List of lists. Row count: the total number of pairwaise relationships across all videos. Column count: dimension of spatial features (8)
        Y_pair_level_evaluation = []
        raw_data_pair_level_evaluation = []
        feature_index = [] # A video can also have no pairwise relationships. In this case, the feature_index for that vid is empty.
        for video_basename, frame_offset, _ in self.video_list_evaluation:
            file = "/gscratch/balazinska/enhaoz/complex_event_video/data/clevrer/processed_proposals/sim_{}.json".format(video_basename)
            # Read in bbox info
            with open(file, 'r') as f:
                data = json.load(f)
            objects = data["ground_truth"]["objects"]

            for frame_id in range(128):
                res_per_frame = self.maskrcnn_bboxes_evaluation["{}_{}".format(video_basename, frame_id)]
                if len(res_per_frame) < 2:
                    continue

                # find all positive pairs. In most cases it should be only one
                positive_pairs = []
                for obj1, obj2 in itertools.combinations(res_per_frame, 2):
                    if self.obj_distance(obj1[:4], obj2[:4]) <= self.thresh:
                        Y_evaluation[frame_offset + frame_id] = 1
                        positive_pairs.append([obj1, obj2])
                for obj1, obj2 in itertools.combinations(res_per_frame, 2):
                    spatial_features.append(construct_spatial_feature_spatial_relationship(obj1[:4], obj2[:4]))
                    feature_index.append(frame_offset + frame_id)
                    # Construct Y_pair_level_evaluation
                    Y_pair_level_evaluation.append(0)
                    raw_data_pair_level_evaluation.append([video_basename, frame_id, obj1, obj2])
                    if Y_evaluation[frame_offset + frame_id] == 1:
                        for pos_obj1, pos_obj2 in positive_pairs:
                            if obj1[4] == pos_obj1[4] and obj1[5] == pos_obj1[5] and obj1[6] == pos_obj1[6] and obj2[4] == pos_obj2[4] and obj2[5] == pos_obj2[5] and obj2[6] == pos_obj2[6]:
                                Y_pair_level_evaluation[-1] = 1
                                break
        spatial_features = np.stack(spatial_features, axis=0)
        Y_pair_level_evaluation = np.asarray(Y_pair_level_evaluation)
        feature_index = np.asarray(feature_index)
        print("length of spatial_features: {}; Y_pair_level_evaluation : {}; feature_index: {}".format(len(spatial_features), len(Y_pair_level_evaluation), len(feature_index)))
        return spatial_features, Y_evaluation, Y_pair_level_evaluation, feature_index, raw_data_pair_level_evaluation

    @staticmethod
    def obj_distance(bbox1, bbox2):
        x1, y1, x2, y2 = bbox1
        x3, y3, x4, y4 = bbox2
        cx1 = (x1 + x2) / 2
        cy1 = (y1 + y2) / 2
        cx2 = (x3 + x4) / 2
        cy2 = (y3 + y4) / 2
        return math.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) / ((x2 - x1 + y2 - y1 + x4 - x3 + y4 - y3) / 4)