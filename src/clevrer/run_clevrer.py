import ujson as json
import os
from glob import glob
from itertools import groupby
import cv2
from collections import defaultdict
import time
import pandas as pd
import numpy as np
from utils import tools
from proxy_model import *


"""
There are three types of models/predicates: perfect model, noisy model and real model.

Specifically for the Clevrer dataset:
    The shape, color and material uniquely identify an object in a video.
    Thus, data_oid = "video_id-shape-color-material".

Query region graph:
    g1 = [Objs, Rels]
    Objs = {oid : {'shape': shape, 'color': color, 'material': material}, ...}
    Rels = {rid : {'sub_id': sub_id, 'obj_id': obj_id, 'rel_name': rel_name}, ...}

Outputs:
    outputs = {'video_id': [[frame_id, [data_region_graph, ...]], ...], ...}
    where data_region_graph = [data_Objs, data_Rels],
          data_Objs = [data_oid, ...],
          data_Rels = {rid : {'sub_id': sub_id, 'obj_id': obj_id, 'rel_name': rel_name}, ...}
"""

def is_subdictionary(A, B):
   return set(A.items()).issubset(B.items())

def print_stats(outputs):
    # num_matching_frames = 0
    # num_matching_instances = 0
    # for v_lst in outputs.values():
    #     num_matching_frames += len(v_lst)
    #     for f_lst in v_lst:
    #         num_matching_instances += len(f_lst[1])
    # print("# matching videos: {}, # matching video frames: {}, # matching instances: {}".format(len(outputs), num_matching_frames, num_matching_instances))
    print("# matching videos: {}, # matching video frames: {}".format(len(outputs), sum(len(v_lst) for v_lst in outputs.values())))

@tools.tik_tok
def ingest_videos():
    """
    Return a dictionary of list, where each key-value pair stores a video_id along with a list of frame_ids of that video.

    Returns
    -------
    video_input: dictionary
        a dictionary of list, where each key-value pair stores a video_id along with a list of frame_ids of that video.
    """
    video_input = defaultdict(list)
    annotation_files = [y for x in os.walk("/gscratch/balazinska/enhaoz/complex_event_video/data/clevrer/annotation_train") for y in glob(os.path.join(x[0], '*.json'))]
    for annotation_file in annotation_files:
        # print(annotation_file)
        with open(annotation_file, 'r') as f:
            annotation_dict = json.loads(f.read())
        vid = annotation_dict["scene_index"]
        for motion_trajectory_dict in annotation_dict["motion_trajectory"]:
            fid = motion_trajectory_dict["frame_id"]
            # video_input[vid].append([fid, []])
            video_input[vid].append(fid)

    return video_input

@tools.tik_tok
def ingest_videos_fast():
    video_input = {}
    for i in range(8000, 9000):
        # video_input[i] = [ {i: []} for i in range(0, 128) ]
        video_input[i] = list(range(0, 128))
    print_stats(video_input)
    return video_input

@tools.tik_tok
def add_predicate_object(video_input, g1, object_list):
    """
    Return xxx

    Parameters
    ----------
    video_input: dictionary
        video frames to work on
    object_list: list[dict]
        each dict specifies an object of interest.
        Example: [{"oid": 0, "shape": "cube"}, {"oid": 1, "shape": "sphere"}]

    Returns
    -------
    outputs: dictionary
        matching video frames
    """
    for dct in object_list:
        g1[0][dct["oid"]]["p_shape"] = dct["shape"]
    print("region graph:", g1)
    outputs = defaultdict(list)
    query_object_class_count = {"cube": 0, "sphere": 0, "cylinder": 0}
    for dct in object_list:
        query_object_class_count[dct["shape"]] += 1
    vids = list(video_input.keys())
    annotation_files = [y for x in os.walk("/gscratch/balazinska/enhaoz/complex_event_video/data/clevrer/annotation_train") for y in glob(os.path.join(x[0], '*.json'))]
    for annotation_file in annotation_files:
        vid = int(os.path.split(annotation_file)[1][11:16])
        if vid in vids:
            video_object_class_count = {"cube": 0, "sphere": 0, "cylinder": 0}
            with open(annotation_file, 'r') as f:
                annotation_dict = json.loads(f.read())
            for obj_dict in annotation_dict["object_property"]:
                video_object_class_count[obj_dict["shape"]] += 1
            if video_object_class_count["cube"] >= query_object_class_count["cube"] and video_object_class_count["sphere"] >= query_object_class_count["sphere"] and video_object_class_count["cylinder"] >= query_object_class_count["cylinder"]:
                for motion_trajectory_dict in annotation_dict["motion_trajectory"]:
                    fid = motion_trajectory_dict["frame_id"]
                    frame_object_class_count = {"cube": 0, "sphere": 0, "cylinder": 0}
                    for obj_dict in motion_trajectory_dict["objects"]:
                        if obj_dict["inside_camera_view"]:
                            shape = annotation_dict["object_property"][obj_dict["object_id"]]["shape"]
                            frame_object_class_count[shape] += 1
                    if frame_object_class_count["cube"] >= query_object_class_count["cube"] and frame_object_class_count["sphere"] >= query_object_class_count["sphere"] and frame_object_class_count["cylinder"] >= query_object_class_count["cylinder"]:
                        outputs[vid].append(fid)

    print_stats(outputs)
    return outputs, g1

@tools.tik_tok
def add_predicate_attribute(video_input, g1, attribute_list):
    """
    Parameters
    ----------
    video_input: dictionary
        video frames to work on
    attribute_list: list[tuple]
        each tuple (oid, k, v) specifies an attribute
        Example: [(0, "color", "red"), (0, "material", "metal"), (1, "color", "gray")]

    Returns
    -------
    outputs: dictionary
        matching video frames
    """
    outputs = defaultdict(list)
    for tup in attribute_list:
        g1[0][tup[0]]["p_" + tup[1]] = tup[2]
    print("g1:", g1)
    Objs = g1[0]
    for vid in video_input.keys():
        annotation_file = "/gscratch/balazinska/enhaoz/complex_event_video/data/clevrer/annotation_train/annotation_{}-{}/annotation_{}.json".format(str((vid // 1000) * 1000).zfill(5), str((vid // 1000 + 1) * 1000).zfill(5), str(vid).zfill(5))
        with open(annotation_file, 'r') as f:
            annotation_dict = json.loads(f.read())

        # Construct a matching matrix
        # m[i, j] = 1 if the i-th object in Objs matches the j-th object (object_id = j) in the video.
        video_object_matching_matrix = np.zeros((len(Objs), len(annotation_dict["object_property"])))
        for i, oid in enumerate(Objs):
            target_obj = Objs[oid]
            for j, obj_dict in enumerate(annotation_dict["object_property"]):
                if is_subdictionary(target_obj, obj_dict):
                    video_object_matching_matrix[i, j] = 1
        # Check if video vid contians all objects of interest
        if np.linalg.matrix_rank(video_object_matching_matrix) == len(Objs):
            # Video contains all objects
            for motion_trajectory_dict in annotation_dict["motion_trajectory"]:
                fid = motion_trajectory_dict["frame_id"]
                if fid in video_input[vid]:
                    frame_object_matching_matrix = np.zeros((len(Objs), len(annotation_dict["object_property"])))
                    for i, obj_dict in enumerate(motion_trajectory_dict["objects"]):
                        if obj_dict["inside_camera_view"]:
                            frame_object_matching_matrix[:, i] += video_object_matching_matrix[:, i]
                    if np.linalg.matrix_rank(frame_object_matching_matrix) == len(Objs):
                        outputs[vid].append(fid)

    print_stats(outputs)
    return outputs, g1


@tools.tik_tok
def add_predicate_attribute_proxy(video_input, g1, proxy_list):
    """
    Parameters
    ----------
    video_input: dictionary
        video frames to work on
    proxy_list: list[tuple]
        each tuple (oid, proxy_name, proxy_object) specifies an attribute, where proxy_name is a function name and proxy_object is the corresponding function object predicting whether oid has the target attribute.
        Example: [(0, "r_color_red", color_red), (0, "r_material_metal", material_metal), (1, "r_color_gray", color_gray)]

    Returns
    -------
    outputs: dictionary
        matching video frames
    """
    outputs = defaultdict(list)
    for tup in proxy_list:
        g1[0][tup[0]][tup[1]] = tup[2]
    print("g1:", g1)
    Objs = g1[0]
    for vid in video_input.keys():
        annotation_file = "/gscratch/balazinska/enhaoz/complex_event_video/data/clevrer/annotation_train/annotation_{}-{}/annotation_{}.json".format(str((vid // 1000) * 1000).zfill(5), str((vid // 1000 + 1) * 1000).zfill(5), str(vid).zfill(5))
        with open(annotation_file, 'r') as f:
            annotation_dict = json.loads(f.read())

        # Construct a matching matrix
        # m[i, j] = 1 if the i-th object in Objs matches the j-th object (object_id = j) in the video.
        video_object_matching_matrix = np.zeros((len(Objs), len(annotation_dict["object_property"])))
        for i, oid in enumerate(Objs):
            target_obj = Objs[oid]
            for j, obj_dict in enumerate(annotation_dict["object_property"]):
                pred = True
                for proxy_name, proxy_object in target_obj.items():
                    if proxy_name[0] == 'p':
                        pred *= is_subdictionary({proxy_name[2:]: proxy_object}, obj_dict)
                    elif proxy_name[0] == 'n':
                        pred *= proxy_object(obj_dict, 1000, b=0.9)
                if pred:
                    video_object_matching_matrix[i, j] = 1
        # Check if video vid contians all objects of interest
        if np.linalg.matrix_rank(video_object_matching_matrix) == len(Objs):
            # Video contains all objects
            for motion_trajectory_dict in annotation_dict["motion_trajectory"]:
                fid = motion_trajectory_dict["frame_id"]
                if fid in video_input[vid]:
                    frame_object_matching_matrix = np.zeros((len(Objs), len(annotation_dict["object_property"])))
                    for i, obj_dict in enumerate(motion_trajectory_dict["objects"]):
                        if obj_dict["inside_camera_view"]:
                            frame_object_matching_matrix[:, i] += video_object_matching_matrix[:, i]
                    if np.linalg.matrix_rank(frame_object_matching_matrix) == len(Objs):
                        outputs[vid].append(fid)

    print_stats(outputs)
    return outputs, g1


@tools.tik_tok
def add_predicate_relationship(video_input, g1, rel_dict):
    """
    Parameters
    ----------
    video_input: dictionary
        video frames to work on
    rel_dict: dict[dict]
        each dict specifies a relationship
        Example: (0, {"sub_id": 0, "obj_id": 1, "rel_name": "collision"})

    Returns
    -------
    outputs: dictionary
        matching video frames
    """
    outputs = defaultdict(list)
    g1[1][rel_dict[0]] = rel_dict[1]
    print("g1:", g1)
    Objs, Rels = g1
    sub_id = rel_dict[1]["sub_id"]
    obj_id = rel_dict[1]["obj_id"]
    rel_name = rel_dict[1]["rel_name"]
    for vid in video_input.keys():
        annotation_file = "/gscratch/balazinska/enhaoz/complex_event_video/data/clevrer/annotation_train/annotation_{}-{}/annotation_{}.json".format(str((vid // 1000) * 1000).zfill(5), str((vid // 1000 + 1) * 1000).zfill(5), str(vid).zfill(5))
        with open(annotation_file, 'r') as f:
            annotation_dict = json.loads(f.read())

        # Construct a matching matrix
        # m[i, j] = 1 if the i-th object in Objs matches the j-th object (object_id = j) in the video.
        video_object_matching_matrix = np.zeros((2, len(annotation_dict["object_property"])))
        for i, oid in enumerate([sub_id, obj_id]):
            target_obj = Objs[oid]
            for j, obj_dict in enumerate(annotation_dict["object_property"]):
                if is_subdictionary(target_obj, obj_dict):
                    video_object_matching_matrix[i, j] = 1
        # Check if video vid contians all objects of interest
        if np.linalg.matrix_rank(video_object_matching_matrix) == 2:
            # Video contains all objects
            for collision_dict in annotation_dict["collision"]:
                if np.linalg.matrix_rank(video_object_matching_matrix[:, collision_dict["object_ids"]]) == 2:
                    outputs[vid].append(collision_dict["frame_id"])

    print_stats(outputs)
    return outputs, g1


@tools.tik_tok
def add_predicate_relationship_proxy(video_input, g1, proxy_list):
    """
    Parameters
    ----------
    video_input: dictionary
        video frames to work on
    proxy_list: tuple(int, dict)
        each dict specifies a relationship
        Example: (0, {"sub_id": 0, "obj_id": 1, "rel_name": "p_collision"})
                 (0, {"sub_id": 0, "obj_id": 1, "rel_name": "n_collision", "rel_object": collision})

    Returns
    -------
    outputs: dictionary
        matching video frames
    """
    outputs = defaultdict(list)
    g1[1][proxy_list[0]] = proxy_list[1]
    print("g1:", g1)
    Objs, Rels = g1
    sub_id = proxy_list[1]["sub_id"]
    obj_id = proxy_list[1]["obj_id"]
    rel_name = proxy_list[1]["rel_name"]
    if rel_name[:2] == "n_":
        rel_object = proxy_list[1]["rel_object"]

    for vid in video_input.keys():
        annotation_file = "/gscratch/balazinska/enhaoz/complex_event_video/data/clevrer/annotation_train/annotation_{}-{}/annotation_{}.json".format(str((vid // 1000) * 1000).zfill(5), str((vid // 1000 + 1) * 1000).zfill(5), str(vid).zfill(5))
        with open(annotation_file, 'r') as f:
            annotation_dict = json.loads(f.read())

        # Construct a matching matrix
        # m[i, j] = 1 if the i-th object in Objs matches the j-th object (object_id = j) in the video.
        video_object_matching_matrix = np.zeros((2, len(annotation_dict["object_property"])))
        for i, oid in enumerate([sub_id, obj_id]):
            target_obj = Objs[oid]
            for j, obj_dict in enumerate(annotation_dict["object_property"]):
                pred = True
                for proxy_name, proxy_object in target_obj.items():
                    if proxy_name[:2] == 'p_':
                        pred *= is_subdictionary({proxy_name[2:]: proxy_object}, obj_dict)
                    elif proxy_name[:2] == 'n_':
                        pred *= proxy_object(obj_dict, 1000)
                if pred:
                    video_object_matching_matrix[i, j] = 1

        # Check if video vid contians all objects of interest
        if np.linalg.matrix_rank(video_object_matching_matrix) == 2:
            if rel_name == "p_collision":
                # Video contains all objects
                for collision_dict in annotation_dict["collision"]:
                    if np.linalg.matrix_rank(video_object_matching_matrix[:, collision_dict["object_ids"]]) == 2 and collision_dict["frame_id"] in video_input[vid]:
                        outputs[vid].append(collision_dict["frame_id"])
            elif rel_name == "n_collision":
                for fid in video_input[vid]:
                    res = False
                    for i in range(len(annotation_dict["object_property"]) - 1):
                        if res:
                            continue
                        for j in range(i + 1, len(annotation_dict["object_property"])):
                            if res:
                                continue
                            if np.linalg.matrix_rank(video_object_matching_matrix[:, [i, j]]) == 2:
                                sub_object_id = i
                                obj_object_id = j
                                if video_object_matching_matrix[0, i] == 0 or video_object_matching_matrix[1, j] == 0:
                                    sub_object_id = j
                                    obj_object_id = i
                                if rel_object(sub_object_id, obj_object_id, fid, annotation_dict["collision"], 100, b=0.99):
                                    outputs[vid].append(fid)
                                    res = True

    print_stats(outputs)
    return outputs, g1


@tools.tik_tok
def add_predicate_seq_of_region_graphs(inputs1, g1, inputs2, g2):
    outputs = defaultdict(list)
    seq_region_graphs = (g1, g2)
    print("seqence of region graphs:", seq_region_graphs)
    for vid in inputs1.keys():
        if vid in inputs2:
            fid_list1 = inputs1[vid]
            fid_list2 = inputs2[vid]
            for fid1 in fid_list1:
                for fid2 in fid_list2:
                    if fid1 < fid2:
                        outputs[vid].append([fid1, fid2])

    print_stats(outputs)
    return outputs, seq_region_graphs

@tools.tik_tok
def visualize_outputs(outputs, query_name):
    output_dir = "/gscratch/balazinska/enhaoz/complex_event_video/tmp/clevrer_{}".format(query_name)
    for vid in outputs.keys():
        cap = cv2.VideoCapture("/gscratch/balazinska/enhaoz/complex_event_video/data/clevrer/video_train/video_{}-{}/video_{}.mp4".format(str((vid // 1000) * 1000).zfill(5), str((vid // 1000 + 1) * 1000).zfill(5), str(vid).zfill(5)))
        for start_fid, end_fid in outputs[vid]:
            if not os.path.exists(os.path.join(output_dir, str(vid), "{}_{}".format(start_fid, end_fid))):
                os.makedirs(os.path.join(output_dir, str(vid), "{}_{}".format(start_fid, end_fid)))
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_fid)
            ret, frame = cap.read()
            cv2.imwrite(os.path.join(output_dir, str(vid), "{}_{}".format(start_fid, end_fid), '{}_{}.jpg'.format(vid, start_fid)), frame)
            cap.set(cv2.CAP_PROP_POS_FRAMES, end_fid)
            ret, frame = cap.read()
            cv2.imwrite(os.path.join(output_dir, str(vid), "{}_{}".format(start_fid, end_fid), '{}_{}.jpg'.format(vid, end_fid)), frame)
        cap.release()

def rd_mtl_cb_clld_gry_sph_thn_clld_cyl():
    video_input = ingest_videos_fast()
    Objs = defaultdict(dict)
    Rels = defaultdict(dict)
    g1 = [Objs, Rels]

    outputs, g1 = add_predicate_object(video_input, g1, [{"oid": 0, "shape": "cube"}, {"oid": 1, "shape": "sphere"}])
    outputs, g1 = add_predicate_attribute(outputs, g1, [(0, "color", "red"), (0, "material", "metal"), (1, "color", "gray")])
    outputs1, g1 = add_predicate_relationship(outputs, g1, (0, {"sub_id": 0, "obj_id": 1, "rel_name": "collision"}))

    Objs = defaultdict(dict)
    Rels = defaultdict(dict)
    g2 = [Objs, Rels]

    outputs, g2 = add_predicate_object(video_input, g2, [{"oid": 2, "shape": "cube"}, {"oid": 3, "shape": "cylinder"}])
    outputs, g2 = add_predicate_attribute(outputs, g2, [(2, "color", "red"), (2, "material", "metal")])
    outputs2, g2 = add_predicate_relationship(outputs, g2, (1, {"sub_id": 2, "obj_id": 3, "rel_name": "collision"}))
    outputs, seq_region_graphs = add_predicate_seq_of_region_graphs(outputs1, g1, outputs2, g2)
    print("outputs:", outputs)
    visualize_outputs(outputs, "rd-mtl-cb-clld-gry-sph-thn-clld-cyl")


def bbb():
    video_input = ingest_videos_fast()
    Objs = defaultdict(dict)
    Rels = defaultdict(dict)
    g1 = [Objs, Rels]

    outputs, g1 = add_predicate_object(video_input, g1, [{"oid": 0, "shape": "cube"}, {"oid": 1, "shape": "sphere"}, {"oid": 2, "shape": "cube"}])
    outputs, g1 = add_predicate_attribute(outputs, g1, [(0, "color", "red")])
    outputs1, g1 = add_predicate_relationship(outputs, g1, (0, {"sub_id": 0, "obj_id": 1, "rel_name": "collision"}))

    Objs = defaultdict(dict)
    Rels = defaultdict(dict)
    g2 = [Objs, Rels]

    outputs, g2 = add_predicate_object(video_input, g2, [{"oid": 3, "shape": "cube"}, {"oid": 4, "shape": "cylinder"}])
    outputs, g2 = add_predicate_attribute(outputs, g2, [(3, "color", "red"), (4, "material", "metal")])
    outputs2, g2 = add_predicate_relationship(outputs, g2, (1, {"sub_id": 3, "obj_id": 4, "rel_name": "collision"}))
    outputs, seq_region_graphs = add_predicate_seq_of_region_graphs(outputs1, g1, outputs2, g2)
    print("outputs:", outputs)
    visualize_outputs(outputs, "bbb")

def ccc():
    video_input = ingest_videos_fast()
    Objs = defaultdict(dict)
    Rels = defaultdict(dict)
    g1 = [Objs, Rels]

    outputs, g1 = add_predicate_object(video_input, g1, [{"oid": 0, "shape": "cylinder"}, {"oid": 1, "shape": "sphere"}])
    outputs, g1 = add_predicate_attribute(outputs, g1, [(0, "material", "rubber"), (1, "color", "purple"), (1, "material", "metal")])
    outputs1, g1 = add_predicate_relationship(outputs, g1, (0, {"sub_id": 0, "obj_id": 1, "rel_name": "collision"}))

    Objs = defaultdict(dict)
    Rels = defaultdict(dict)
    g2 = [Objs, Rels]

    outputs, g2 = add_predicate_object(video_input, g2, [{"oid": 2, "shape": "cylinder"}, {"oid": 3, "shape": "cube"}])
    outputs, g2 = add_predicate_attribute(outputs, g2, [(0, "material", "rubber"), (1, "color", "green"), (1, "material", "metal")])
    outputs2, g2 = add_predicate_relationship(outputs, g2, (1, {"sub_id": 2, "obj_id": 3, "rel_name": "collision"}))
    outputs, seq_region_graphs = add_predicate_seq_of_region_graphs(outputs1, g1, outputs2, g2)
    print("outputs:", outputs)
    visualize_outputs(outputs, "ccc")

def test_noisy_model():
    video_input = ingest_videos_fast()
    Objs = defaultdict(dict)
    Rels = defaultdict(dict)
    g1 = [Objs, Rels]

    outputs, g1 = add_predicate_object(video_input, g1, [{"oid": 0, "shape": "cube"}, {"oid": 1, "shape": "sphere"}])
    outputs, g1 = add_predicate_attribute_proxy(outputs, g1, [(0, "p_color", "red"), (0, "p_material", "metal"), (1, "p_color", "gray")])
    # outputs, g1 = add_predicate_attribute_proxy(outputs, g1, [(0, "n_color_red", color_red), (0, "n_material_metal", material_metal), (1, "n_color_gray", color_gray)])
    # outputs1, g1 = add_predicate_relationship_proxy(outputs, g1, (0, {"sub_id": 0, "obj_id": 1, "rel_name": "p_collision"}))
    outputs1, g1 = add_predicate_relationship_proxy(outputs, g1, (0, {"sub_id": 0, "obj_id": 1, "rel_name": "n_collision", "rel_object": collision}))

    Objs = defaultdict(dict)
    Rels = defaultdict(dict)
    g2 = [Objs, Rels]

    outputs, g2 = add_predicate_object(video_input, g2, [{"oid": 2, "shape": "cube"}, {"oid": 3, "shape": "cylinder"}])
    outputs, g2 = add_predicate_attribute_proxy(outputs, g2, [(2, "p_color", "red"), (2, "p_material", "metal")])
    # outputs, g2 = add_predicate_attribute_proxy(outputs, g2, [(2, "n_color_red", color_red), (2, "n_material_metal", material_metal)])
    # outputs2, g2 = add_predicate_relationship_proxy(outputs, g2, (1, {"sub_id": 2, "obj_id": 3, "rel_name": "p_collision"}))
    outputs2, g2 = add_predicate_relationship_proxy(outputs, g2, (1, {"sub_id": 2, "obj_id": 3, "rel_name": "n_collision", "rel_object": collision}))
    outputs, seq_region_graphs = add_predicate_seq_of_region_graphs(outputs1, g1, outputs2, g2)
    print("outputs:", outputs)

    visualize_outputs(outputs, "test_noisy_model")

if __name__ == '__main__':
    # rd_mtl_cb_clld_gry_sph_thn_clld_cyl()
    test_noisy_model()
    # ccc()

