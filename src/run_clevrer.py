import json
import os
from glob import glob
from itertools import groupby
import operator
import cv2

def filter_object_a_and_object_b(obj_a_color=None, obj_a_material=None, obj_a_shape=None, obj_b_color=None, obj_b_material=None, obj_b_shape=None):
    outputs = []
    annotation_files = [y for x in os.walk("/mmfs1/gscratch/balazinska/enhaoz/complex_event_video/data/clevrer/annotation_train") for y in glob(os.path.join(x[0], '*.json'))]
    for annotation_file in annotation_files:
        # print(annotation_file)
        with open(annotation_file, 'r') as f:
            annotation_dict = json.loads(f.read())
        vid = annotation_dict["scene_index"]
        object_a_id = -1
        object_b_id = -1
        for obj_dict in annotation_dict["object_property"]:
            if (obj_dict["color"] == obj_a_color or not obj_a_color) and (obj_dict["material"] == obj_a_material or not obj_a_material) and (obj_dict["shape"] == obj_a_shape or not obj_a_shape):
                object_a_id = obj_dict["object_id"]
            if (obj_dict["color"] == obj_b_color or not obj_b_color) and (obj_dict["material"] == obj_b_material or not obj_b_material) and (obj_dict["shape"] == obj_b_shape or not obj_b_shape):
                object_b_id = obj_dict["object_id"]
        if object_a_id != -1 and object_b_id != -1:
            # Video contains both objects
            for motion_trajectory_dict in annotation_dict["motion_trajectory"]:
                fid = motion_trajectory_dict["frame_id"]
                has_object_a = False
                has_object_b = False
                for obj_dict in motion_trajectory_dict["objects"]:
                    if obj_dict["object_id"] == object_a_id and obj_dict["inside_camera_view"]:
                        has_object_a = True
                    elif obj_dict["object_id"] == object_b_id and obj_dict["inside_camera_view"]:
                        has_object_b = True
                    if has_object_a and has_object_b:
                        outputs.append([vid, fid])
                        break
    print("# matching videos: {}, # matching video frames: {}".format(len(set([x[0] for x in outputs])), len(outputs)))
    return outputs


def object_a_collides_with_object_b(inputs, obj_a_color=None, obj_a_material=None, obj_a_shape=None, obj_b_color=None, obj_b_material=None, obj_b_shape=None):
    outputs = []
    vids = set([x[0] for x in inputs])
    annotation_files = [y for x in os.walk("/mmfs1/gscratch/balazinska/enhaoz/complex_event_video/data/clevrer/annotation_train") for y in glob(os.path.join(x[0], '*.json'))]
    for annotation_file in annotation_files:
        vid = int(os.path.split(annotation_file)[1][11:16])
        if vid in vids:
            # print(annotation_file)
            with open(annotation_file, 'r') as f:
                annotation_dict = json.loads(f.read())
            object_a_id = -1
            object_b_id = -1
            for obj_dict in annotation_dict["object_property"]:
                if (obj_dict["color"] == obj_a_color or not obj_a_color) and (obj_dict["material"] == obj_a_material or not obj_a_material) and (obj_dict["shape"] == obj_a_shape or not obj_a_shape):
                    object_a_id = obj_dict["object_id"]
                if (obj_dict["color"] == obj_b_color or not obj_b_color) and (obj_dict["material"] == obj_b_material or not obj_b_material) and (obj_dict["shape"] == obj_b_shape or not obj_b_shape):
                    object_b_id = obj_dict["object_id"]
            for collision_dict in annotation_dict["collision"]:
                if object_a_id in collision_dict["object_ids"] and object_b_id in collision_dict["object_ids"]:
                    outputs.append([vid, collision_dict["frame_id"]])
    print("# matching videos: {}, # matching video frames: {}".format(len(set([x[0] for x in outputs])), len(outputs)))
    return outputs


def g1_after_g2(inputs_g1, inputs_g2):
    outputs = []
    inputs_g1 = sorted(inputs_g1, key=lambda x:x[0])
    inputs_g2 = sorted(inputs_g2, key=lambda x:x[0])
    for k1, g1 in groupby(inputs_g1, key=lambda x:x[0]):
        for k2, g2 in groupby(inputs_g2, key=lambda x:x[0]):
            if k1 == k2:
                for r1 in g1:
                    for r2 in g2:
                        if r1[1] < r2[1]:
                            outputs.append([k1, r1[1], r2[1]])
                break
    print("# matching videos: {}, # matching video segments: {}".format(len(set([x[0] for x in outputs])), len(outputs)))
    return outputs


def visualize_outputs(outputs, query_name):
    output_dir = "/mmfs1/gscratch/balazinska/enhaoz/complex_event_video/tmp/clevrer_{}".format(query_name)
    for vid, start_fid, end_fid in outputs:
        if not os.path.exists(os.path.join(output_dir, str(vid), "{}_{}".format(start_fid, end_fid))):
            os.makedirs(os.path.join(output_dir, str(vid), "{}_{}".format(start_fid, end_fid)))
        cap = cv2.VideoCapture("/mmfs1/gscratch/balazinska/enhaoz/complex_event_video/data/clevrer/video_train/video_{}-{}/video_{}.mp4".format(str((vid // 1000) * 1000).zfill(5), str((vid // 1000 + 1) * 1000).zfill(5), str(vid).zfill(5)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_fid)
        ret, frame = cap.read()
        cv2.imwrite(os.path.join(output_dir, str(vid), "{}_{}".format(start_fid, end_fid), '{}_{}.jpg'.format(vid, start_fid)), frame)
        cap.set(cv2.CAP_PROP_POS_FRAMES, end_fid)
        ret, frame = cap.read()
        cv2.imwrite(os.path.join(output_dir, str(vid), "{}_{}".format(start_fid, end_fid), '{}_{}.jpg'.format(vid, end_fid)), frame)
        cap.release()


if __name__ == '__main__':
    # filter: red metal cube and gray sphere
    outputs = filter_object_a_and_object_b(obj_a_color="red", obj_a_material="metal", obj_a_shape="cube", obj_b_color="gray", obj_b_shape="sphere")
    outputs_g1 = object_a_collides_with_object_b(outputs, obj_a_color="red", obj_a_material="metal", obj_a_shape="cube", obj_b_color="gray", obj_b_shape="sphere")
    # filter: red metal cube and cylinder
    outputs = filter_object_a_and_object_b(obj_a_color="red", obj_a_material="metal", obj_a_shape="cube", obj_b_shape="cylinder")
    outputs_g2 = object_a_collides_with_object_b(outputs, obj_a_color="red", obj_a_material="metal", obj_a_shape="cube", obj_b_shape="cylinder")
    outputs = g1_after_g2(outputs_g1, outputs_g2)
    visualize_outputs(outputs, "rd-mtl-cb-clld-gry-sph-thn-clld-cyl")
