"""
This script is used to compute the score of answer queries (/outputs/vocal/xxx.log) over all video fragments (e.g., 50 positive and 250 negative fragments).
"""

from utils_bu import print_program, str_to_program
import random
import json
import numpy as np
from sklearn.metrics import f1_score
import math
import os

random.seed(10)

# read from json file
with open("/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/collision_inputs.json", 'r') as f:
    inputs = json.load(f)
with open("/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/collision_labels.json", 'r') as f:
    labels = json.load(f)
inputs = np.asarray(inputs, dtype=object)
labels = np.asarray(labels, dtype=object)

n_pos = sum(labels)
sampled_labeled_index = random.sample(list(range(n_pos)), 50) + random.sample(list(range(n_pos, len(labels))), 250)
inputs = inputs[sampled_labeled_index]
labels = labels[sampled_labeled_index]
memoize_all_inputs = [{} for _ in range(len(inputs))]

def compute_query_score(current_query):
    y_pred = []
    # for i, (input, label) in enumerate(zip(self.inputs, self.labels)):
    for i in range(len(inputs)):
        input = inputs[i]
        label = labels[i]
        memoize = memoize_all_inputs[i]
        result, memoize = current_query.execute(input, label, memoize)
        y_pred.append(int(result[0, len(input[0])] > 0))
        memoize_all_inputs[i] = memoize
        dist_list = []
        for obj_a, obj_b in zip(input[0], input[1]):
            dist = obj_distance(obj_a, obj_b)
            dist_list.append(dist)
        # print("dist: ", dist_list)
        # print("y_pred: ", y_pred[-1])
        # print("label: ", label)
        # print("")
    score = f1_score(list(labels), y_pred)
    return score


def obj_distance(bbox1, bbox2):
    x1, y1, x2, y2 = bbox1
    x3, y3, x4, y4 = bbox2
    cx1 = (x1 + x2) / 2
    cy1 = (y1 + y2) / 2
    cx2 = (x3 + x4) / 2
    cy2 = (y3 + y4) / 2
    return math.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) / ((x2 - x1 + x4 - x3) / 2)


def parse(filename):
    results = []
    # Read in file. For each line in file, if it starts with "answer", then get the second string.
    with open("outputs/vocal/" + filename) as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("answer"):
                program_str = ' '.join(line.split()[1:-1])
                print(program_str)
                program = str_to_program(program_str)
                print_program(program)
                # compute score
                score = compute_query_score(program)
                print("score: ", score)
                results.append((program_str, score))

    # write results to log file
    with open("outputs/vocal/score_over_all_fragments/" + filename, 'w') as f:
        for result in results:
            f.write(result[0] + " " + str(result[1]) + "\n")

if __name__ == "__main__":
    # list files in directory. For each file, call parse.
    for filename in os.listdir("outputs/vocal"):
        if filename.endswith(".log"):
            print(filename)
            parse(filename)
    # parse("vocal-npos_50-nneg_250-npred_5-depth_3-k_16-model_picker_include_answers.log")