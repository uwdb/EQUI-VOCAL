from quivr.utils import print_program, str_to_program
import json
import random
import math
import numpy as np
from sklearn.metrics import f1_score
import time
from lru import LRU

random.seed(10)

def compute_query_score(current_query, inputs, labels, memoize_all_inputs):
    y_pred = []
    # for i, (input, label) in enumerate(zip(self.inputs, self.labels)):
    for i in range(len(inputs)):
        input = inputs[i]
        label = labels[i]
        memoize = memoize_all_inputs[i]
        result, new_memoize = current_query.execute(input, label, memoize, {})
        y_pred.append(int(result[0, len(input[0])] > 0))
        memoize_all_inputs[i].update(new_memoize)
        dist_list = []
        for obj_a, obj_b in zip(input[0], input[1]):
            dist = obj_distance(obj_a, obj_b)
            dist_list.append(dist)
        print("dist: ", dist_list)
        print("y_pred: ", y_pred[-1])
        print("label: ", label)
        print("")
    score = f1_score(list(labels), y_pred)
    return score, memoize_all_inputs


def obj_distance(bbox1, bbox2):
    x1, y1, x2, y2 = bbox1
    x3, y3, x4, y4 = bbox2
    cx1 = (x1 + x2) / 2
    cy1 = (y1 + y2) / 2
    cx2 = (x3 + x4) / 2
    cy2 = (y3 + y4) / 2
    return math.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) / ((x2 - x1 + x4 - x3) / 2)


def test_vocal(n_labeled_pos, n_labeled_neg, program_str):
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
    memoize_all_inputs = [{} for _ in range(len(inputs))]

    program = str_to_program(program_str)
    print(print_program(program))
    # compute score
    score, memoize_all_inputs = compute_query_score(program, inputs, labels, memoize_all_inputs)
    print("score: ", score)


def test_query_equivalent(n_labeled_pos, n_labeled_neg, program_str1, program_str2):
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
    memoize_all_inputs = [{} for _ in range(len(inputs))]

    # compute score
    _start = time.time()
    program1 = str_to_program(program_str1)
    print(print_program(program1))
    score1, memoize_all_inputs = compute_query_score(program1, inputs, labels, memoize_all_inputs)
    print("program1 score: ", score1)
    print("program1 time: ", time.time() - _start)

    _start = time.time()
    program2 = str_to_program(program_str2)
    print(print_program(program2))
    score2, memoize_all_inputs = compute_query_score(program2, inputs, labels, memoize_all_inputs)
    print("program2 score: ", score2)
    print("program2 time: ", time.time() - _start)

# Test quivr (topdown, exact)
def test_quivr_answer_correctness(n_labeled_pos, n_labeled_neg):
    # read from json file
    with open("/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/collision_inputs.json", 'r') as f:
        inputs = json.load(f)
    with open("/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/collision_labels.json", 'r') as f:
        labels = json.load(f)
    inputs = np.asarray(inputs, dtype=object)
    labels = np.asarray(labels, dtype=object)

    sampled_pos_index = [2340, 133, 1756, 1976, 2367, 60, 844, 1894, 2012, 1136, 656, 140, 2132, 2007, 1342, 311, 1023, 1479, 182, 1721, 568, 1454, 1563, 1725, 1161, 1074, 1871, 715, 1241, 1485, 544, 980, 1800, 1536, 181, 2387, 16, 965, 548, 798, 1240, 2196, 1499, 983, 1287, 2248, 1845, 1785, 1925, 266]
    sampled_neg_index = [13155, 12034, 7776, 10680, 5019, 6131, 9221, 6362, 8912, 5132, 10593, 7391, 12396, 13235, 3637, 11197, 3783, 4909, 8755, 11750, 8587, 12308, 4307, 4039, 9691, 5182, 5585, 8169, 9555, 9241, 9757, 6478, 13611, 6957, 4808, 12570, 11007, 5380, 4414, 6831, 9923, 7414, 5159, 13277, 13085, 5312, 5342, 10323, 8151, 6542]
    inputs = inputs[sampled_pos_index[:n_labeled_pos]].tolist() + inputs[sampled_neg_index[:n_labeled_neg]].tolist()
    labels = labels[sampled_pos_index[:n_labeled_pos]].tolist() + labels[sampled_neg_index[:n_labeled_neg]].tolist()

    memoize_all_inputs = [{} for _ in range(len(inputs))]
    count = 0
    print(inputs, labels)
    with open("/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/quivr-npos_{}-nneg_{}.log".format(n_labeled_pos, n_labeled_neg), 'r') as f:
        is_answer = False
        for line in f:
            line = line.rstrip()
            # check if line starts with "answer"
            if line.startswith("answer"):
                is_answer = True
                continue
            if is_answer:
                # convert program string to program object
                program = str_to_program(line)
                print(line)
                # compute score
                score, memoize_all_inputs = compute_query_score(program, inputs, labels, memoize_all_inputs)
                assert(score == 1)
                count += 1
                print("pass {}".format(count))
    print("OK")

# Test quivr_soft (topdown)
def test_quivr_soft(n_labeled_pos, n_labeled_neg, program_str):
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

    memoize_all_inputs = [{} for _ in range(len(inputs))]

    program = str_to_program(program_str)
    print(print_program(program))
    # compute score
    score, memoize_all_inputs = compute_query_score(program, inputs, labels, memoize_all_inputs)
    print("score: ", score)

def test_query(dataset_name, target_query, test_query):
    _start = time.time()
    with open("inputs/{}/train/{}_inputs.json".format(dataset_name, target_query), 'r') as f:
        inputs = json.load(f)
    # with open("inputs/{}/train/{}_labels.json".format(dataset_name, target_query), 'r') as f:
    #     labels = json.load(f)
    inputs = np.asarray(inputs, dtype=object)
    # labels = np.asarray(labels, dtype=object)

    # Top-10 queries, majority vote
    y_pred = []
    for i in range(len(inputs)):
        input = inputs[i]
        memoize = LRU(10000)
        program = str_to_program(test_query)
        result, new_memoize = program.execute(input, -1, memoize, {})
        y_pred.append(int(result[0, len(input[0])] > 0))
        memoize.update(new_memoize)

    print("y_pred: {}".format(y_pred))
    print("time: {}".format(time.time() - _start))

if __name__ == '__main__':
    # test_query_equivalent(100, 100, "Sequencing(Sequencing(Sequencing(Sequencing(Sequencing(Sequencing(True*, Conjunction(Kleene(Far_1.1), MinLength_2)), True*), Near_1.05), True*), Conjunction(Kleene(Conjunction(Near_1.05, Far_0.9)), MinLength_10)), True*)", "Sequencing(Sequencing(Sequencing(Sequencing(Sequencing(Sequencing(True*, Duration(Far_1.1, 2)), True*), Near_1.05), True*), Duration(Conjunction(Near_1.05, Far_0.9), 10)), True*)")
    # test_vocal(1, 5, "Sequencing(Sequencing(Sequencing(Sequencing(Sequencing(Sequencing(True*, Near_1.05), True*), Conjunction(LeftOf, BackOf)), True*), Duration(Conjunction(TopQuadrant, Far_0.9), 5)), True*)")
    test_query("synthetic_rare", "Sequencing(Sequencing(Sequencing(Sequencing(Sequencing(Sequencing(True*, Near_1.05), True*), Conjunction(LeftOf, BackOf)), True*), Duration(Conjunction(TopQuadrant, Far_0.9), 5)), True*)", "Sequencing(Sequencing(Sequencing(Sequencing(Sequencing(Sequencing(True*, Near_1.05), True*), Conjunction(LeftOf, BackOf)), True*), Duration(Conjunction(TopQuadrant, Far_0.9), 5)), True*)")


    # test_quivr_answer_correctness(10, 10)
    # test_quivr_soft(50, 50, "Start(Sequencing(True*, Conjunction(Sequencing(Near_1.05, True*), MinLength_2.0)))") # 0.951
    # test_quivr_soft(10, 50, "Start(Sequencing(True*, Conjunction(Sequencing(Near_1.05, True*), MinLength_2.0)))") # 0.783 "near and last frame not near"
    # test_quivr_soft(10, 50, "Start(Sequencing(True*, Sequencing(Far_1.1, Sequencing(True*, Sequencing(Near_1.05, True*)))))") # 0.75 “far, near"
    # test_quivr_soft(10, 50, "Start(Sequencing(Sequencing(True*, Near_1.05), True*))")