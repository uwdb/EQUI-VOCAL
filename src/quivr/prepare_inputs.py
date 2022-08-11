"""
This script is used to generate ground-truth labels for a given target query, which are used to test the proposed algorithm on (arbitrarily) more complex queries.
"""

import json
import itertools
import random
import numpy as np
import os
from quivr.utils import print_program, str_to_program

segment_length = 128
n_chunks = int(128 / segment_length)
random.seed(1234)
np.random.seed(10)


def prepare_trajectory_pairs():
    with open("/gscratch/balazinska/enhaoz/complex_event_video/data/clevrer/processed_bboxes_evaluation_1000.json", 'r') as f:
        maskrcnn_bboxes_evaluation = json.loads(f.read())
    with open("/gscratch/balazinska/enhaoz/complex_event_video/data/clevrer/processed_video_list_evaluation_1000.json", 'r') as f:
        video_list_evaluation = json.loads(f.read())

    sample_inputs = []

    for video_i, (video_basename, _, _) in enumerate(video_list_evaluation):
        print(video_i, video_basename)
        # Construct object list
        obj_set = set()
        for frame_id in range(128):
            res_per_frame = maskrcnn_bboxes_evaluation["{}_{}".format(video_basename, frame_id)]
            # [81.0, 109.0, 128.0, 164.0, "metal", "blue", "cylinder"]
            for obj in res_per_frame:
                obj_set.add("{}_{}_{}".format(obj[4], obj[5], obj[6]))

        file = "/gscratch/balazinska/enhaoz/complex_event_video/data/clevrer/processed_proposals/sim_{}.json".format(video_basename)
        # Read in bbox info
        with open(file, 'r') as f:
            data = json.load(f)
        objects = data["ground_truth"]["objects"]

        # Start querying
        for obj1_str_id, obj2_str_id in itertools.combinations(obj_set, 2):
            obj1_id = obj1_str_id.split("_")
            obj2_id = obj2_str_id.split("_")

            # Check if both objects are registered in the video (i.e. not a misclassification)
            obj1 = None
            obj2 = None
            for obj in objects:
                if obj["color"] == obj1_id[1] and obj["material"] == obj1_id[0] and obj["shape"] == obj1_id[2]:
                    obj1 = obj
                if obj["color"] == obj2_id[1] and obj["material"] == obj2_id[0] and obj["shape"] == obj2_id[2]:
                    obj2 = obj

            if obj1 and obj2:
                sample_input = [[], []]

                for frame_id in range(segment_length):
                    obj1 = None
                    obj2 = None
                    res_per_frame = maskrcnn_bboxes_evaluation["{}_{}".format(video_basename, frame_id)]
                    for obj in res_per_frame:
                        if obj[4] == obj1_id[0] and obj[5] == obj1_id[1] and obj[6] == obj1_id[2]:
                            obj1 = obj
                        if obj[4] == obj2_id[0] and obj[5] == obj2_id[1] and obj[6] == obj2_id[2]:
                            obj2 = obj
                    # If both objects are present in the frame, then check for collision
                    if obj1 and obj2:
                        sample_input[0].append(obj1[:4])
                        sample_input[1].append(obj2[:4])

                sample_inputs.append(sample_input)
    print("Generated {} input pairs".format(len(sample_inputs)))

    with open(os.path.join("inputs/trajectory_pairs.json"), 'w') as f:
        f.write(json.dumps(sample_inputs))

def prepare_trajectory_pairs_given_target_query(program_str, n_pos=None, n_neg=None):
    """
    Given the target query (in string format), generate inputs.json and labels.json files containing
    inputs.json:
    """
    program = str_to_program(program_str)
    print(print_program(program))
    # read in trajectory pairs from file
    with open(os.path.join("inputs/trajectory_pairs.json"), 'r') as f:
        sample_inputs = json.loads(f.read())
    # sample_inputs = np.asarray(sample_inputs, dtype=object)

    positive_inputs = []
    negative_inputs = []

    label = -1 # dummy label
    for sample_input in sample_inputs:
        result, _ = program.execute(sample_input, label, {})
        if result[0, len(sample_input[0])] > 0:
            positive_inputs.append(sample_input)
            print("So far: {} positive inputs and {} negative inputs".format(len(positive_inputs), len(negative_inputs)))
        else:
            negative_inputs.append(sample_input)
        if n_pos and n_neg and len(positive_inputs) >= n_pos and len(negative_inputs) >= n_neg:
            break
    print("Generated {} positive inputs and {} negative inputs".format(len(positive_inputs), len(negative_inputs)))

    labels = [1] * len(positive_inputs) + [0] * len(negative_inputs)
    # write positive and negative inputs to one file
    with open("inputs/{}_inputs.json".format(program_str), 'w') as f:
        positive_inputs.extend(negative_inputs)
        f.write(json.dumps(positive_inputs))
    with open("inputs/{}_labels.json".format(program_str), 'w') as f:
        f.write(json.dumps(labels))

if __name__ == '__main__':
    # prepare_trajectory_pairs()
    # prepare_trajectory_pairs_given_target_query("Sequencing(Sequencing(Sequencing(Sequencing(Sequencing(Sequencing(True*, Conjunction(Right, Front)), True*), Duration(Far_0.9, 2)), True*), Conjunction(Back, Left)), True*)")
    prepare_trajectory_pairs_given_target_query("Sequencing(Sequencing(Sequencing(Sequencing(Sequencing(Sequencing(True*, Conjunction(Front, Left)), True*), Duration(Left, 2)), True*), Conjunction(Far_0.9, Left)), True*)")