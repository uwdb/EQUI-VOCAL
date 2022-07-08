from utils import print_program, str_to_program
from quivr import QUIVR
from vocal import VOCAL
from quivr_soft import QUIVRSoft
import json
import random
import math
import numpy as np
from sklearn.metrics import f1_score
import argparse

random.seed(10)

def obj_distance(bbox1, bbox2):
    x1, y1, x2, y2 = bbox1
    x3, y3, x4, y4 = bbox2
    cx1 = (x1 + x2) / 2
    cy1 = (y1 + y2) / 2
    cx2 = (x3 + x4) / 2
    cy2 = (y3 + y4) / 2
    return math.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) / ((x2 - x1 + x4 - x3) / 2)

def test_quivr_exact(n_labeled_pos, n_labeled_neg):
    # read from json file
    with open("/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/collision_inputs.json", 'r') as f:
        inputs = json.load(f)
    with open("/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/collision_labels.json", 'r') as f:
        labels = json.load(f)
    inputs = np.asarray(inputs, dtype=object)
    labels = np.asarray(labels, dtype=object)

    # n_pos = sum(labels)
    # sampled_labeled_index = random.sample(list(range(n_pos)), 50) + random.sample(list(range(n_pos, len(labels))), 250)
    # inputs = inputs[sampled_labeled_index]
    # labels = labels[sampled_labeled_index]

    sampled_pos_index = [2340, 133, 1756, 1976, 2367, 60, 844, 1894, 2012, 1136, 656, 140, 2132, 2007, 1342, 311, 1023, 1479, 182, 1721, 568, 1454, 1563, 1725, 1161, 1074, 1871, 715, 1241, 1485, 544, 980, 1800, 1536, 181, 2387, 16, 965, 548, 798, 1240, 2196, 1499, 983, 1287, 2248, 1845, 1785, 1925, 266]
    sampled_neg_index = [13155, 12034, 7776, 10680, 5019, 6131, 9221, 6362, 8912, 5132, 10593, 7391, 12396, 13235, 3637, 11197, 3783, 4909, 8755, 11750, 8587, 12308, 4307, 4039, 9691, 5182, 5585, 8169, 9555, 9241, 9757, 6478, 13611, 6957, 4808, 12570, 11007, 5380, 4414, 6831, 9923, 7414, 5159, 13277, 13085, 5312, 5342, 10323, 8151, 6542]
    inputs = inputs[sampled_pos_index[:n_labeled_pos]].tolist() + inputs[sampled_neg_index[:n_labeled_neg]].tolist()
    labels = labels[sampled_pos_index[:n_labeled_pos]].tolist() + labels[sampled_neg_index[:n_labeled_neg]].tolist()
    # for input, label in zip(inputs, labels):
    #     print(label)
    #     for i in range(len(input[0])):
    #         print(obj_distance(input[0][i], input[1][i]))
    # exit(1)
    # inputs = [
    #     [
    #         [[0, 0, 50, 50], [2, 2, 52, 52], [5, 5, 55, 55]],
    #         [[100, 100, 150, 150], [100, 100, 150, 150], [100, 100, 150, 150]]
    #     ],
    #     [
    #         [[0, 0, 50, 50], [2, 2, 52, 52], [5, 5, 55, 55]],
    #         [[0, 0, 50, 50], [2, 2, 52, 52], [5, 5, 55, 55]]
    #     ],
    #     [
    #         [[0, 0, 50, 50], [2, 2, 52, 52], [5, 5, 55, 55]],
    #         [[0.2, 0.2, 50.2, 50.2], [100, 100, 150, 150], [5.2, 5.2, 55.2, 55.2]]
    #     ]
    # ]
    # labels = [0, 1, 0]
    algorithm = QUIVR(max_depth=3)
    answer = algorithm.run(inputs, labels)
    print("answer", len(answer))
    for q in answer:
        print(print_program(q))

def test_quivr_soft(n_labeled_pos, n_labeled_neg):
    # read from json file
    with open("/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/collision_inputs.json", 'r') as f:
        inputs = json.load(f)
    with open("/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/collision_labels.json", 'r') as f:
        labels = json.load(f)
    inputs = np.asarray(inputs, dtype=object)
    labels = np.asarray(labels, dtype=object)

    # sampled_pos_index = [2340, 133, 1756, 1976, 2367, 60, 844, 1894, 2012, 1136, 656, 140, 2132, 2007, 1342, 311, 1023, 1479, 182, 1721, 568, 1454, 1563, 1725, 1161, 1074, 1871, 715, 1241, 1485, 544, 980, 1800, 1536, 181, 2387, 16, 965, 548, 798, 1240, 2196, 1499, 983, 1287, 2248, 1845, 1785, 1925, 266]
    # sampled_neg_index = [13155, 12034, 7776, 10680, 5019, 6131, 9221, 6362, 8912, 5132, 10593, 7391, 12396, 13235, 3637, 11197, 3783, 4909, 8755, 11750, 8587, 12308, 4307, 4039, 9691, 5182, 5585, 8169, 9555, 9241, 9757, 6478, 13611, 6957, 4808, 12570, 11007, 5380, 4414, 6831, 9923, 7414, 5159, 13277, 13085, 5312, 5342, 10323, 8151, 6542]
    # inputs = inputs[sampled_pos_index[:n_labeled_pos]].tolist() + inputs[sampled_neg_index[:n_labeled_neg]].tolist()
    # labels = labels[sampled_pos_index[:n_labeled_pos]].tolist() + labels[sampled_neg_index[:n_labeled_neg]].tolist()
    n_pos = sum(labels)
    sampled_labeled_index = random.sample(list(range(n_pos)), n_labeled_pos) + random.sample(list(range(n_pos, len(labels))), n_labeled_neg)
    inputs = inputs[sampled_labeled_index]
    labels = labels[sampled_labeled_index]

    algorithm = QUIVRSoft(max_depth=3)
    answer = algorithm.run(inputs, labels)
    print("answer", len(answer))
    for q, s in answer:
        print(print_program(q), s)

def test_vocal():
    # read from json file
    with open("/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/collision_inputs.json", 'r') as f:
        inputs = json.load(f)
    with open("/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/collision_labels.json", 'r') as f:
        labels = json.load(f)

    inputs = np.asarray(inputs, dtype=object)
    labels = np.asarray(labels, dtype=object)

    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    sampled_labeled_index = random.sample(list(range(n_pos)), 50) + random.sample(list(range(n_pos, len(labels))), 250)
    inputs = inputs[sampled_labeled_index]
    labels = labels[sampled_labeled_index]
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    init_labeled_index = random.sample(list(range(n_pos)), 2) + random.sample(list(range(n_pos, len(labels))), 2)
    algorithm = VOCAL(inputs, labels, max_num_atomic_predicates=10, max_depth=5, k1=32, k2=64, budget=50, thresh=0.5)
    answer = algorithm.run(init_labeled_index)
    for q in answer:
        print(print_program(q))

def compute_query_score(current_query, inputs, labels):
    y_pred = []
    # for i, (input, label) in enumerate(zip(self.inputs, self.labels)):
    for i in range(len(inputs)):
        input = inputs[i]
        label = labels[i]
        memoize = {}
        result, memoize = current_query.execute(input, label, memoize)
        y_pred.append(int(result[0, len(input[0])] > 0))
        # self.memoize_all_inputs[i] = memoize
    # print(self.labels[self.labeled_index], y_pred)
    score = f1_score(list(labels), y_pred)
    return score

def test_program(program_str):
    with open("/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/collision_inputs.json", 'r') as f:
        inputs = json.load(f)
    with open("/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/collision_labels.json", 'r') as f:
        labels = json.load(f)

    inputs = np.asarray(inputs, dtype=object)
    labels = np.asarray(labels, dtype=object)

    program = str_to_program(program_str)

    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    sampled_labeled_index = random.sample(list(range(n_pos)), 50) + random.sample(list(range(n_pos, len(labels))), 250)
    inputs = inputs[sampled_labeled_index]
    labels = labels[sampled_labeled_index]
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    init_labeled_index = random.sample(list(range(n_pos)), 2) + random.sample(list(range(n_pos, len(labels))), 2)
    score = compute_query_score(program, inputs, labels)
    print(score)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    # ap.add_argument('--num_train', type=int)
    # ap.add_argument('--duration', type=int)
    ap.add_argument('--method', type=str)
    ap.add_argument('--n_labeled_pos', type=int)
    ap.add_argument('--n_labeled_neg', type=int)
    args = ap.parse_args()
    method_str = args.method
    n_labeled_pos = args.n_labeled_pos
    n_labeled_neg = args.n_labeled_neg
    print("method:", method_str, "n_labeled_pos:", n_labeled_pos, "n_labeled_neg:", n_labeled_neg)
    if method_str == 'quivr':
        test_quivr_exact(n_labeled_pos, n_labeled_neg)
    elif method_str == 'vocal':
        test_vocal()
    elif method_str == 'quivr_soft':
        test_quivr_soft(n_labeled_pos, n_labeled_neg)
    # test_program("Start(Sequencing(Sequencing(True*, Conjunction(Kleene(Near_1.05), MinLength_1.0)), True*))") # 0.8130