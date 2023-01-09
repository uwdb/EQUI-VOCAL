import os
import time
from utils import rewrite_program_postgres, postgres_execute_cache_sequence, str_to_program_postgres, complexity_cost
from sklearn.metrics import f1_score
from itertools import chain, combinations
import argparse
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import json

if __name__ == '__main__':
    def compute_f1_score(test_program):
        test_query_str = rewrite_program_postgres(test_program)
        outputs, new_memoize_scene_graph, new_memoize_sequence = postgres_execute_cache_sequence(dsn, test_program, memoize_scene_graph, memoize_sequence, "Obj_clevrer", input_vids, is_trajectory=False, sampling_rate=None)

        if lock:
            lock.acquire()
        for i, memo_dict in enumerate(new_memoize_scene_graph):
            for k, v in memo_dict.items():
                memoize_scene_graph[i][k] = v
        for i, memo_dict in enumerate(new_memoize_sequence):
            for k, v in memo_dict.items():
                memoize_sequence[i][k] = v
        if lock:
            lock.release()
        preds = []
        for input in input_vids:
            if input in outputs:
                preds.append(1)
            else:
                preds.append(0)
        score = f1_score(labels, preds)
        print(test_query_str)
        print(score)
        return score

    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5432)
    ap.add_argument("--multithread", type=int, default=4)
    ap.add_argument("--query_str", type=str)
    ap.add_argument("--dataset_name", type=str)
    args = ap.parse_args()
    port = args.port
    multithread = args.multithread
    query_str = args.query_str
    dataset_name = args.dataset_name

    if multithread > 1:
        executor = ThreadPoolExecutor(max_workers=multithread)
        m = multiprocessing.Manager()
        lock = m.Lock()
    elif multithread == 1:
        lock = None

    memoize_scene_graph = [{} for _ in range(10000)]
    memoize_sequence = [{} for _ in range(10000)]

    # Read the test data
    test_dir = "/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/inputs/{}/test".format(dataset_name)
    inputs_filename = query_str + "_inputs.json"
    labels_filename = query_str + "_labels.json"
    # n_examples = int(filename[11:-12])
    with open(os.path.join(test_dir, inputs_filename), 'r') as f:
        input_vids = json.load(f)
    with open(os.path.join(test_dir, labels_filename), 'r') as f:
        labels = json.load(f)


    # Generate all ancestors of the target query
    query_program_list = []
    target_query = str_to_program_postgres(query_str)
    # target_query = [{'scene_graph': [{'predicate': 'Far', 'parameter': 3.0, 'variables': ['o0', 'o1']}, {'predicate': 'RightQuadrant', 'parameter': None, 'variables': ['o1']}], 'duration_constraint': 15}, {'scene_graph': [{'predicate': 'FrontOf', 'parameter': None, 'variables': ['o0', 'o1']}, {'predicate': 'Near', 'parameter': 1.0, 'variables': ['o0', 'o1']}], 'duration_constraint': 1}, {'scene_graph': [{'predicate': 'Far', 'parameter': 3.0, 'variables': ['o0', 'o1']}, {'predicate': 'RightQuadrant', 'parameter': None, 'variables': ['o1']}], 'duration_constraint': 5}]
    print("target query str:", query_str)
    print("target query program:", target_query)
    g0_powerset = list(chain.from_iterable(combinations(target_query[0]['scene_graph'], r) for r in range(len(target_query[0]['scene_graph'])+1)))
    g1_powerset = list(chain.from_iterable(combinations(target_query[1]['scene_graph'], r) for r in range(len(target_query[1]['scene_graph'])+1)))
    g2_powerset = list(chain.from_iterable(combinations(target_query[2]['scene_graph'], r) for r in range(len(target_query[2]['scene_graph'])+1)))
    duration_unit = 5
    d0_list = [1]
    for i in range(1, target_query[0]["duration_constraint"] // duration_unit + 1):
        d0_list.append(i * duration_unit)
    d1_list = [1]
    for i in range(1, target_query[1]["duration_constraint"] // duration_unit + 1):
        d1_list.append(i * duration_unit)
    d2_list = [1]
    for i in range(1, target_query[2]["duration_constraint"] // duration_unit + 1):
        d2_list.append(i * duration_unit)
    for d0 in d0_list:
        for d1 in d1_list:
            for d2 in d2_list:
                for g0 in g0_powerset:
                    for g1 in g1_powerset:
                        for g2 in g2_powerset:
                            region_graph_sequence = []
                            if len(g0) > 0:
                                region_graph_sequence.append({'scene_graph': list(g0), 'duration_constraint': d0})
                            if len(g1) > 0:
                                region_graph_sequence.append({'scene_graph': list(g1), 'duration_constraint': d1})
                            if len(g2) > 0:
                                region_graph_sequence.append({'scene_graph': list(g2), 'duration_constraint': d2})
                            if len(region_graph_sequence) > 0:
                                query_program_list.append(region_graph_sequence)
    # Remove duplicates
    query_program_list_removing_duplicates = []
    signatures = set()
    for program in query_program_list:
        signature = rewrite_program_postgres(program)
        if signature not in signatures:
            query_program_list_removing_duplicates.append(str_to_program_postgres(signature))
            signatures.add(signature)
    query_program_list = query_program_list_removing_duplicates
    print("number of queries:", len(query_program_list))
    for query_program in query_program_list:
        print(query_program)
        print(rewrite_program_postgres(query_program))

    # compute f1 score
    dsn = "dbname=myinner_db user=enhaoz host=localhost port={}".format(port)
    f1_scores = []
    if multithread > 1:
        for result in executor.map(compute_f1_score, query_program_list):
            f1_scores.append(result)
    else:
        for test_query in query_program_list:
            score = compute_f1_score(test_query)
            f1_scores.append(score)

    results = []
    for score, query_program in zip(f1_scores, query_program_list):
        iter_depth = complexity_cost(query_program, a1=1, a2=1, a3=0)
        results.append((iter_depth, score, query_program))
    results = sorted(results, key=lambda x: (x[0], x[1]), reverse=True)

    # write to file
    log_dirname = os.path.join("outputs", "{}_ancestors".format(dataset_name))
    log_filename = "{}.txt".format(query_str)
    # if dir not exist, create it
    if not os.path.exists(log_dirname):
        os.makedirs(log_dirname, exist_ok=True)
    with open(os.path.join(log_dirname, log_filename), 'w') as f:
        for iter_depth, score, query_program in results:
            f.write("{}\n".format(iter_depth))
            f.write("{}\n".format(rewrite_program_postgres(query_program)))
            f.write("{}\n".format(score))