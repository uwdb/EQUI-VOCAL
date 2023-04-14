import csv
import json
import os
import random
import numpy as np
import src.dsl as dsl
import psycopg2 as psycopg
import copy
from lru import LRU
import time
import pandas as pd
import itertools
import math
import uuid
from sklearn.model_selection import train_test_split
from io import StringIO
from collections import deque
import resource
from sklearn.metrics import f1_score

def using(point=""):
    usage=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return '''%s: mem=%s MB'''%(point, usage/1024.0 )

def print_program(program, as_dict_key=False):
    # if isinstance(program, dsl.Predicate) or isinstance(program, dsl.Hole):
    if issubclass(type(program), dsl.Predicate):
        if program.has_theta:
            if program.with_hole:
                return program.name + "_withHole"
            else:
                return program.name + "_" + str(abs(program.theta))
        else:
            return program.name
    if issubclass(type(program), dsl.Hole):
        return program.name
    if issubclass(type(program), dsl.DurationOperator):
        if as_dict_key:
            return program.name + "(" + print_program(program.submodules["duration"]) + ")"
        else:
            return program.name + "(" + print_program(program.submodules["duration"]) + ", " + str(program.theta) + ")"
    else:
        collected_names = []
        for submodule, functionclass in program.submodules.items():
            collected_names.append(print_program(functionclass))
        joined_names = ', '.join(collected_names)
        return program.name + "(" + joined_names + ")"

# convert a str back to an executable program; this is an inverse of print_program()
def str_to_program(program_str):
    if program_str.startswith("PredicateHole"):
        return dsl.PredicateHole()
    elif program_str.startswith("ParameterHole"):
        functionclass = program_str.split("*")[1]
        predicate = getattr(dsl, functionclass)
        return dsl.ParameterHole(predicate())
    if program_str.startswith("Near"):
        return dsl.Near(theta=-float(program_str.split("_")[1]))
    elif program_str.startswith("Far"):
        return dsl.Far(theta=float(program_str.split("_")[1]))
    elif program_str.startswith("MinLength"):
        return dsl.MinLength(theta=float(program_str.split("_")[1]))
    elif program_str.startswith("True*"):
        return dsl.TrueStar()
    elif program_str.startswith("RightOf"):
        return dsl.RightOf()
    elif program_str.startswith("LeftOf"):
        return dsl.LeftOf()
    elif program_str.startswith("FrontOf"):
        return dsl.FrontOf()
    elif program_str.startswith("Behind"):
        return dsl.Behind()
    elif program_str.startswith("TopQuadrant"):
        return dsl.TopQuadrant()
    elif program_str.startswith("BottomQuadrant"):
        return dsl.BottomQuadrant()
    elif program_str.startswith("LeftQuadrant"):
        return dsl.LeftQuadrant()
    elif program_str.startswith("RightQuadrant"):
        return dsl.RightQuadrant()
    ###  dsl.AEastward4: None, dsl.AEastward3: None, dsl.AEastward2: None, dsl.AWestward2: None, dsl.ASouthward1Upper: None, dsl.AStopped: [2], dsl.AHighAccel: [2], dsl.BEastward4: None, dsl.BEastward3: None, dsl.BEastward2: None, dsl.BWestward2: None, dsl.BSouthward1Upper: None, dsl.BStopped: [2], dsl.BHighAccel: [2], dsl.DistanceSmall: [100], dsl.Faster: [1.5]
    elif program_str.startswith("AEastward4"):
        return dsl.AEastward4()
    elif program_str.startswith("AEastward3"):
        return dsl.AEastward3()
    elif program_str.startswith("AEastward2"):
        return dsl.AEastward2()
    elif program_str.startswith("AWestward2"):
        return dsl.AWestward2()
    elif program_str.startswith("ASouthward1Upper"):
        return dsl.ASouthward1Upper()
    elif program_str.startswith("AStopped"):
        return dsl.AStopped(theta=-float(program_str.split("_")[1]))
    elif program_str.startswith("AHighAccel"):
        return dsl.AHighAccel(theta=float(program_str.split("_")[1]))
    elif program_str.startswith("BEastward4"):
        return dsl.BEastward4()
    elif program_str.startswith("BEastward3"):
        return dsl.BEastward3()
    elif program_str.startswith("BEastward2"):
        return dsl.BEastward2()
    elif program_str.startswith("BWestward2"):
        return dsl.BWestward2()
    elif program_str.startswith("BSouthward1Upper"):
        return dsl.BSouthward1Upper()
    elif program_str.startswith("BStopped"):
        return dsl.BStopped(theta=-float(program_str.split("_")[1]))
    elif program_str.startswith("BHighAccel"):
        return dsl.BHighAccel(theta=float(program_str.split("_")[1]))
    elif program_str.startswith("DistanceSmall"):
        return dsl.DistanceSmall(theta=-float(program_str.split("_")[1]))
    elif program_str.startswith("Faster"):
        return dsl.Faster(theta=float(program_str.split("_")[1]))
    else:
        idx = program_str.find("(")
        idx_r = program_str.rfind(")")
        # True*, Sequencing(Near_1.0, MinLength_10.0)
        functionclass = program_str[:idx]
        submodules = program_str[idx+1:idx_r]
        counter = 0
        submodule_list = []
        submodule_start = 0
        for i, char in enumerate(submodules):
            if char == "," and counter == 0:
                submodule_list.append(submodules[submodule_start:i])
                submodule_start = i+2
            elif char == "(":
                counter += 1
            elif char == ")":
                counter -= 1
        submodule_list.append(submodules[submodule_start:])
        if functionclass == "Start":
            program_init = getattr(dsl, "StartOperator")
        if functionclass == "Sequencing":
            program_init = getattr(dsl, "SequencingOperator")
        elif functionclass == "Conjunction":
            program_init = getattr(dsl, "ConjunctionOperator")
        elif functionclass == "Kleene":
            program_init = getattr(dsl, "KleeneOperator")
        elif functionclass == "Duration":
            program_init = getattr(dsl, "DurationOperator")
            return program_init(str_to_program(submodule_list[0]), int(submodule_list[1]))
        submodule_list = [str_to_program(submodule) for submodule in submodule_list]
        return program_init(*submodule_list)

def get_depth_and_npred(program):
    """
    Return: depth, num_nontrivial_predicates, num_trivial_predicates
    """
    main_program = program.submodules["program"]
    return _get_depth_and_npred_helper(main_program)

def _get_depth_and_npred_helper(program):
    if isinstance(program, dsl.ParameterHole):
        return 0, 1, 0
    elif issubclass(type(program), dsl.Predicate):
        if isinstance(program, dsl.TrueStar):
            return 0, 0, 1
        else:
            return 0, 1, 0
    elif isinstance(program, dsl.PredicateHole):
        return 0, 1, 0 # assume predicate holes are non-trivial predicates
    elif isinstance(program, dsl.ConjunctionOperator):
        left = program.submodules["function1"]
        right = program.submodules["function2"]
        depth_left, n_nontrivial_left, n_trivial_left = _get_depth_and_npred_helper(left)
        depth_right, n_nontrivial_right, n_trivial_right = _get_depth_and_npred_helper(right)
        if isinstance(left, dsl.ConjunctionOperator) or isinstance(right, dsl.ConjunctionOperator):
            return max(depth_left, depth_right), n_nontrivial_left + n_nontrivial_right, n_trivial_left + n_trivial_right
        else:
            return max(depth_left, depth_right) + 1, n_nontrivial_left + n_nontrivial_right, n_trivial_left + n_trivial_right
    elif isinstance(program, dsl.SequencingOperator):
        left = program.submodules["function1"]
        right = program.submodules["function2"]
        depth_left, n_nontrivial_left, n_trivial_left = _get_depth_and_npred_helper(left)
        depth_right, n_nontrivial_right, n_trivial_right = _get_depth_and_npred_helper(right)
        if isinstance(left, dsl.SequencingOperator) or isinstance(right, dsl.SequencingOperator):
            return max(depth_left, depth_right), n_nontrivial_left + n_nontrivial_right, n_trivial_left + n_trivial_right
        else:
            return max(depth_left, depth_right) + 1, n_nontrivial_left + n_nontrivial_right, n_trivial_left + n_trivial_right
    elif isinstance(program, dsl.KleeneOperator):
        kleene = program.submodules["kleene"]
        depth_kleene, n_nontrivial_kleene, n_trivial_kleene = _get_depth_and_npred_helper(kleene)
        # return depth_kleene, n_nontrivial_kleene, n_trivial_kleene
        # NOTE: Quivr just included kleene in some of the predicates (e.g. <True>* is one of our predicates) and omitted it from search.
        if isinstance(kleene, dsl.KleeneOperator):
            return depth_kleene, n_nontrivial_kleene, n_trivial_kleene
        else:
            return depth_kleene + 1, n_nontrivial_kleene, n_trivial_kleene
    else:
        raise ValueError("Unknown program type:", type(program))

def complexity_cost(program, a1=1, a2=1, a3=0.1):
    duration_unit = 5
    cost_npred = sum([len(dict["scene_graph"]) * a1 for dict in program])
    cost_duration = sum([(dict["duration_constraint"] // duration_unit) * (a2 + a3 * len(dict["scene_graph"])) for dict in program])
    # return cost_npred + cost_depth * 0.5 + cost_duration
    return cost_npred + cost_duration

def construct_train_test(dir_name, query_str, n_labeled_pos=None, n_labeled_neg=None, n_train=None):
    inputs_filename = query_str + "_inputs"
    labels_filename = query_str + "_labels"

    # read from json file
    with open(os.path.join(dir_name, "{}.json".format(inputs_filename)), 'r') as f:
        inputs = json.load(f)
    with open(os.path.join(dir_name, "{}.json".format(labels_filename)), 'r') as f:
        labels = json.load(f)

    inputs = np.asarray(inputs, dtype=object)
    labels = np.asarray(labels, dtype=object)

    if n_train:
        inputs_train, inputs_test, labels_train, labels_test = train_test_split(inputs, labels, train_size=n_train, random_state=42, stratify=labels)
    if n_labeled_pos and n_labeled_neg:
        n_pos = sum(labels)
        sampled_labeled_index = random.sample(list(range(n_pos)), n_labeled_pos) + random.sample(list(range(n_pos, len(labels))), n_labeled_neg)
        sampled_labeled_index = np.asarray(sampled_labeled_index)
        print("before sort", sampled_labeled_index)
        # sort sampled_labeled_index in ascending order
        sampled_labeled_index = sampled_labeled_index[np.argsort(sampled_labeled_index)]
        print("after sort", sampled_labeled_index)
        inputs_train = inputs[sampled_labeled_index]
        labels_train = labels[sampled_labeled_index]

        remaining_index = np.delete(np.arange(len(labels)), sampled_labeled_index)

        inputs_test = inputs[remaining_index]
        labels_test = labels[remaining_index]
        print("labels_test", labels_test)

        n_pos = sum(labels_test)
        n_neg = len(labels_test) - n_pos
        print("n_pos", n_pos, "n_neg", n_neg)
        if n_pos * 5 < n_neg:
            inputs_test = inputs_test[:n_pos * 6]
            labels_test = labels_test[:n_pos * 6]
        else:
            inputs_test = inputs_test[:int(n_neg/5)] + inputs_test[-int(n_neg/5)*5:]
            labels_test = labels_test[:int(n_neg/5)] + labels_test[-int(n_neg/5)*5:]

    # if folder doesn't exist, create it
    if not os.path.exists(os.path.join(dir_name, "train/")):
        os.makedirs(os.path.join(dir_name, "train/"))
    if not os.path.exists(os.path.join(dir_name, "test/")):
        os.makedirs(os.path.join(dir_name, "test/"))

    with open(os.path.join(dir_name, "train/{}.json".format(inputs_filename)), 'w') as f:
        json.dump(inputs_train.tolist(), f)
    with open(os.path.join(dir_name, "train/{}.json".format(labels_filename)), 'w') as f:
        json.dump(labels_train.tolist(), f)
    with open(os.path.join(dir_name, "test/{}.json".format(inputs_filename)), 'w') as f:
        json.dump(inputs_test.tolist(), f)
    with open(os.path.join(dir_name, "test/{}.json".format(labels_filename)), 'w') as f:
        json.dump(labels_test.tolist(), f)

    print("inputs_train", len(inputs_train))
    print("labels_train", len(labels_train), sum(labels_train))
    print("inputs_test", len(inputs_test))
    print("labels_test", len(labels_test), sum(labels_test))


def get_query_str_from_filename(dir_name):
    query_list = []
    # for each file in the folder
    for filename in os.listdir(dir_name):
        if filename.endswith("_labels.json"):
            query_str = filename[:-12]
            with open(os.path.join(dir_name, "{}_labels.json".format(query_str)), 'r') as f:
                labels = json.load(f)
                query_list.append([query_str, sum(labels), len(labels) - sum(labels), sum(labels) / (len(labels) - sum(labels))])
            # len(positive_inputs), len(negative_inputs), len(positive_inputs) / len(negative_inputs)
            if not os.path.exists(os.path.join(dir_name, "test/{}_labels.json".format(query_str))):
                construct_train_test(dir_name, query_str, n_train=300)
    # write query_list to file
    with open(os.path.join(dir_name, "queries.csv"), "w") as csvfile:
        writer = csv.writer(csvfile)
        # write header
        writer.writerow(["query", "npos", "nneg", "ratio"])
        writer.writerows(query_list)

def correct_filename(dataset_name):
    """
    correct the filename to ensure the query string is ordered properly.
    """
    # for each file in the folder
    for filename in os.listdir(os.path.join("inputs", dataset_name)):
        if filename.endswith("_labels.json"):
            query_str = filename[:-12]
            new_query_str = query_str.replace("Back", "Behind")
            new_query_str = new_query_str.replace("Front", "FrontOf")
            new_query_str = new_query_str.replace("Left", "LeftOf")
            new_query_str = new_query_str.replace("Right", "RightOf")
            # change filename
            os.rename("inputs/{}/{}_inputs.json".format(dataset_name, query_str), "inputs/{}/{}_inputs.json".format(dataset_name, new_query_str))
            os.rename("inputs/{}/{}_labels.json".format(dataset_name, query_str), "inputs/{}/{}_labels.json".format(dataset_name, new_query_str))
            os.rename("inputs/{}/test/{}_inputs.json".format(dataset_name, query_str), "inputs/{}/test/{}_inputs.json".format(dataset_name, new_query_str))
            os.rename("inputs/{}/test/{}_labels.json".format(dataset_name, query_str), "inputs/{}/test/{}_labels.json".format(dataset_name, new_query_str))
            os.rename("inputs/{}/train/{}_inputs.json".format(dataset_name, query_str), "inputs/{}/train/{}_inputs.json".format(dataset_name, new_query_str))
            os.rename("inputs/{}/train/{}_labels.json".format(dataset_name, query_str), "inputs/{}/train/{}_labels.json".format(dataset_name, new_query_str))

def rewrite_program(program):
    """
    rewrite the query string to ensure the query string is ordered properly.
    """
    if issubclass(type(program), dsl.ConjunctionOperator):
        predicate_list = rewrite_program_helper(program)
        predicate_list.sort(key=lambda x: x, reverse=True)
        return print_scene_graph(predicate_list)
    if issubclass(type(program), dsl.Predicate):
        if program.has_theta:
            if program.with_hole:
                return program.name + "_withHole"
            else:
                return program.name + "_" + str(abs(program.theta))
        else:
            return program.name
    if issubclass(type(program), dsl.Hole):
        return program.name
    if issubclass(type(program), dsl.DurationOperator):
        return program.name + "(" + rewrite_program(program.submodules["duration"]) + ", " + str(program.theta) + ")"
    else:
        collected_names = []
        for submodule, functionclass in program.submodules.items():
            collected_names.append(rewrite_program(functionclass))
        joined_names = ', '.join(collected_names)
        return program.name + "(" + joined_names + ")"

def print_scene_graph(predicate_list):
    # Conj(Conj(p13, p12), p11)
    if len(predicate_list) == 1:
        return predicate_list[0]
    else:
        return "Conjunction({}, {})".format(print_scene_graph(predicate_list[:-1]), predicate_list[-1])

def rewrite_program_helper(program):
    if issubclass(type(program), dsl.Predicate):
        if program.has_theta:
            if program.with_hole:
                return [program.name + "_withHole"]
            else:
                return [program.name + "_" + str(abs(program.theta))]
        else:
            return [program.name]
    elif issubclass(type(program), dsl.ConjunctionOperator):
        predicate_list = []
        for submodule, functionclass in program.submodules.items():
            predicate_list.extend(rewrite_program_helper(functionclass))
        return predicate_list

def postgres_execute(dsn, current_query, memoize_scene_graph, memoize_sequence, inputs_table_name, input_vids, is_trajectory=True, sampling_rate=None):
    """
    [Deprecated]
    Caching the result of every sub-query of current_query (not used in evaluation)
    input_vids: list of video segment ids
    Example query:
        Sequencing(Sequencing(Sequencing(Sequencing(Sequencing(Sequencing(True*, Near_1.05), True*), Conjunction(LeftOf, Behind)), True*), Duration(Conjunction(TopQuadrant, Far_0.9), 5)), True*)
    two types of caches:
        1. scene graph cache (without duration constraints): cache[graph] = vid, fid, oids (where fid is every frame that satisfies the graph)
        2. sequence cache: cache[sequence] = vid, fid, oids (where fid is the minimum frame that satisfies the sequence)
        Example: g1, (g1, d1), g2, (g1, d1); (g2, d2), g3, (g1, d1); (g2, d2); (g3, d3)
    Output:
    new_memoize: new cached results from this query, which will be added to the global cache (for multi-threading)
    """
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            """
            Caching implementation:
            cached_df_deque = []
            cached_vids_deque = []
            remaining_vids = input_vids
            for each query (from the most top-level to the most bottom-level):
                cached_df_per_query = []
                cached_vids_per_query = []
                next_remaining_vids = []
                for each video segment in remaining_vids:
                    if the result is cached:
                        add the result to cached_df_per_query
                        add the video segment id to cached_vids_per_query
                    else:
                        add the video segment id to next_remaining_vids
                push cached_df_per_query to cached_df_deque
                push cached_vids_per_query to cached_vids_deque
                remaining_vids = next_remaining_vids
            """
            new_memoize_scene_graph = [{} for _ in range(len(memoize_scene_graph))]
            new_memoize_sequence = [{} for _ in range(len(memoize_sequence))]
            # select input videos
            _start = time.time()
            if isinstance(input_vids, int):
                if sampling_rate:
                    cur.execute("CREATE TEMPORARY TABLE Obj_filtered AS SELECT oid, vid, fid / {} as fid, shape, color, material, x1, y1, x2, y2 FROM {} WHERE vid < {} AND fid % {} = 0;".format(sampling_rate, inputs_table_name, input_vids, sampling_rate))
                else:
                    cur.execute("CREATE TEMPORARY TABLE Obj_filtered AS SELECT * FROM {} WHERE vid < {};".format(inputs_table_name, input_vids))
                input_vids = list(range(input_vids))
            else:
                if sampling_rate:
                    cur.execute("CREATE TEMPORARY TABLE Obj_filtered AS SELECT oid, vid, fid / {} as fid, shape, color, material, x1, y1, x2, y2 FROM {} WHERE vid = ANY(%s) AND fid %% {} = 0;".format(sampling_rate, inputs_table_name, sampling_rate), [input_vids])
                else:
                    cur.execute("CREATE TEMPORARY TABLE Obj_filtered AS SELECT * FROM {} WHERE vid = ANY(%s);".format(inputs_table_name), [input_vids])
            cur.execute("CREATE INDEX IF NOT EXISTS idx_obj_filtered ON Obj_filtered (vid, fid, oid);")
            # print("select input videos: ", time.time() - _start)

            # Prepare cache result
            cached_df_sg_deque = deque()
            cached_df_seq_deque = deque()
            cached_vids_deque = deque()
            remaining_vids = set(input_vids)
            signatures_and_vars_mapping = deque()
            for i in range(len(current_query)):
                cached_df_sg_per_query = [pd.DataFrame()]
                cached_df_seq_per_query = [pd.DataFrame()]

                # sequence cache
                seq_signature = rewrite_program_postgres(current_query[:len(current_query)-i], not is_trajectory)
                cached_vids_per_query = set()
                next_remaining_vids = set()
                for vid in remaining_vids:
                    if seq_signature in memoize_sequence[vid]:
                        cached_df_seq_per_query.append(memoize_sequence[vid][seq_signature])
                        cached_vids_per_query.add(vid)
                    else:
                        next_remaining_vids.add(vid)
                cached_df_seq_per_query = pd.concat(cached_df_seq_per_query, ignore_index=True)
                cached_df_seq_deque.append(cached_df_seq_per_query)
                if len(cached_vids_deque) > 0:
                    next_sg_cached_vids = cached_vids_deque.pop() # FIXME: for last sequence (aka the entire query)
                    # cached_vids_but_not_sg_per_query = cached_vids_per_query - next_sg_cached_vids
                    cached_vids_and_sg_per_query = cached_vids_per_query & next_sg_cached_vids
                    cached_vids_deque.append(cached_vids_and_sg_per_query)
                    cached_vids_deque.append(cached_vids_per_query)
                else:
                    cached_vids_deque.append(cached_vids_per_query)

                remaining_vids = next_remaining_vids

                # Scene graph cache
                sg_signature, vars_mapping = rewrite_vars_name_for_scene_graph({"scene_graph": current_query[len(current_query)-i-1]["scene_graph"], "duration_constraint": 1})
                cached_vids_sg_per_query = set()
                # next_remaining_vids = []
                for vid in remaining_vids:
                    if sg_signature in memoize_scene_graph[vid]:
                        cached_df_sg_per_query.append(memoize_scene_graph[vid][sg_signature])
                        cached_vids_sg_per_query.add(vid)
                    # else:
                    #     next_remaining_vids.append(vid)
                cached_df_sg_per_query = pd.concat(cached_df_sg_per_query, ignore_index=True)
                cached_df_sg_deque.append(cached_df_sg_per_query)
                cached_vids_deque.append(cached_vids_sg_per_query)
                # remaining_vids = next_remaining_vids
                signatures_and_vars_mapping.append([sg_signature, vars_mapping, seq_signature])
            cached_vids_deque.append(remaining_vids)

            encountered_variables_prev_graphs = []
            encountered_variables_current_graph = []
            delta_input_vids = []
            for graph_idx, dict in enumerate(current_query):
                _start = time.time()
                # Generate scene graph:
                scene_graph = dict["scene_graph"]
                duration_constraint = dict["duration_constraint"]
                for p in scene_graph:
                    for v in p["variables"]:
                        if v not in encountered_variables_current_graph:
                            encountered_variables_current_graph.append(v)
                # Read cached results
                sg_signature, vars_mapping, seq_signature = signatures_and_vars_mapping.pop()
                # signature, vars_mapping = rewrite_vars_name_for_scene_graph({"scene_graph": scene_graph, "duration_constraint": 1})
                cached_results = cached_df_sg_deque.pop()
                delta_input_vids.extend(cached_vids_deque.pop())
                # Execute for unseen videos
                _start_execute = time.time()
                encountered_variables_current_graph = sorted(encountered_variables_current_graph, key=lambda x: int(x[1:]))
                tables = ", ".join(["Obj_filtered as {}".format(v) for v in encountered_variables_current_graph])
                where_clauses = []
                where_clauses.append("{}.vid = ANY(%s)".format(encountered_variables_current_graph[0]))
                for i in range(len(encountered_variables_current_graph)-1):
                    where_clauses.append("{v1}.vid = {v2}.vid and {v1}.fid = {v2}.fid".format(v1=encountered_variables_current_graph[i], v2=encountered_variables_current_graph[i+1])) # join variables
                for p in scene_graph:
                    predicate = p["predicate"]
                    parameter = p["parameter"]
                    variables = p["variables"]
                    args = []
                    for v in variables:
                        args.append("{v}.shape, {v}.color, {v}.material, {v}.x1, {v}.y1, {v}.x2, {v}.y2".format(v=v))
                    args = ", ".join(args)
                    if parameter:
                        if isinstance(parameter, str):
                            args = "'{}', {}".format(parameter, args)
                        else:
                            args = "{}, {}".format(parameter, args)
                    where_clauses.append("{}({}) = true".format(predicate, args))
                if is_trajectory:
                    # only for trajectory example
                    for v in encountered_variables_current_graph:
                        where_clauses.append("{}.oid = {}".format(v, v[1:]))
                else:
                    # For general case
                    for var_pair in itertools.combinations(encountered_variables_current_graph, 2):
                        where_clauses.append("{}.oid <> {}.oid".format(var_pair[0], var_pair[1]))
                where_clauses = " and ".join(where_clauses)
                fields = "{v}.vid as vid, {v}.fid as fid, ".format(v=encountered_variables_current_graph[0])
                fields += ", ".join(["{v}.oid as {v}_oid".format(v=v) for v in encountered_variables_current_graph])
                oid_list = ["{}_oid".format(v) for v in encountered_variables_current_graph]
                oids = ", ".join(oid_list)
                sql_sring = """
                CREATE TEMPORARY TABLE g{} AS
                SELECT {}
                FROM {}
                WHERE {};
                """.format(graph_idx, fields, tables, where_clauses)
                # print(sql_sring)
                cur.execute(sql_sring, [delta_input_vids])
                # cur.execute("CREATE INDEX IF NOT EXISTS idx_g{} ON g{} (vid);".format(graph_idx, graph_idx))
                # print("execute for unseen videos: ", time.time() - _start_execute)
                # Store new cached results
                for input_vid in delta_input_vids:
                    new_memoize_scene_graph[input_vid][sg_signature] = pd.DataFrame()
                _start_execute = time.time()
                cur.execute("SELECT * FROM g{}".format(graph_idx))
                df = pd.DataFrame(cur.fetchall())
                # print("[store cache]: fetchall ", time.time() - _start_execute)
                _start_store = time.time()
                if df.shape[0]: # if results not empty
                    df.columns = ["vid", "fid"] + [vars_mapping[oid] for oid in oid_list]
                    for vid, group in df.groupby("vid"):
                        cached_df = group.reset_index(drop=True)
                        new_memoize_scene_graph[vid][sg_signature] = cached_df
                # print("[store cache]: store ", time.time() - _start_store)
                # Appending cached results of seen videos:
                _start_append_cache = time.time()
                if cached_results.shape[0]:
                    tem_table_insert_data = cached_results.copy()
                    tem_table_insert_data.columns = ["vid", "fid"] + oid_list
                    for k, v in vars_mapping.items():
                        tem_table_insert_data[k] = cached_results[v]
                    buffer = StringIO()
                    tem_table_insert_data.to_csv(buffer, header=False, index = False)
                    buffer.seek(0)
                    cur.copy_from(buffer, "g{}".format(graph_idx), sep=",")
                # print("append cache: ", time.time() - _start_append_cache)
                # print("Time for graph {}: {}".format(graph_idx, time.time() - _start))

                # Read cached results
                # signature = rewrite_program_postgres(current_query[:graph_idx+1], not is_trajectory)
                cached_results = cached_df_seq_deque.pop()
                delta_input_vids.extend(cached_vids_deque.pop())

                _start_filtered = time.time()
                if graph_idx > 0:
                    obj_union = copy.deepcopy(encountered_variables_prev_graphs)
                    obj_union_fields = []
                    obj_intersection_fields = []
                    for v in encountered_variables_prev_graphs:
                        obj_union_fields.append("t0.{}_oid".format(v))
                    for v in encountered_variables_current_graph:
                        if v in encountered_variables_prev_graphs:
                            obj_intersection_fields.append("t0.{v}_oid = t1.{v}_oid".format(v=v))
                        else:
                            for u in encountered_variables_prev_graphs:
                                obj_intersection_fields.append("t0.{u}_oid <> t1.{v}_oid".format(u=u, v=v))
                            obj_union.append(v)
                            obj_union_fields.append("t1.{}_oid".format(v))
                    obj_union_fields = ", ".join(obj_union_fields)
                    obj_intersection_fields = " and ".join(obj_intersection_fields)
                    # where_clauses = "t0.vid = ANY(%s)"
                    # if current_seq == "g0_seq_view":
                    #     where_clauses += " and t0.vid = t1.vid and t0.fid2 < t1.fid1"
                    # else:
                    #     where_clauses += " and t0.vid = t1.vid and t0.fid < t1.fid1"
                    sql_string = """
                    CREATE TEMPORARY TABLE g{graph_idx}_filtered AS (
                        SELECT t0.vid, t1.fid, {obj_union_fields}
                        FROM g{graph_idx_prev}_contiguous t0, g{graph_idx} t1
                        WHERE t0.vid = t1.vid AND {obj_intersection_fields} AND t0.fid < t1.fid
                    );
                    """.format(graph_idx=graph_idx, graph_idx_prev=graph_idx-1, obj_union_fields=obj_union_fields, obj_intersection_fields=obj_intersection_fields)
                    # print(sql_string)
                    cur.execute(sql_string)
                else:
                    # sql_string = """
                    # CREATE TEMPORARY TABLE g{graph_idx}_filtered AS (
                    #     SELECT *
                    #     FROM g{graph_idx}
                    # );
                    # """.format(graph_idx=graph_idx)
                    # cur.execute(sql_string)
                    obj_union = encountered_variables_current_graph
                # print("filtered: ", time.time() - _start_filtered)

                # Generate scene graph sequence:
                _start_windowed = time.time()
                table_name = "g{}_filtered".format(graph_idx) if graph_idx > 0 else "g{}".format(graph_idx)
                obj_union_fields = ", ".join(["{}_oid".format(v) for v in obj_union])
                _start = time.time()
                sql_string = """
                    CREATE TEMPORARY TABLE g{graph_idx}_windowed AS (
                    SELECT vid, fid, {obj_union_fields},
                    lead(fid, {duration_constraint} - 1, 0) OVER (PARTITION BY vid, {obj_union_fields} ORDER BY fid) as fid_offset
                    FROM {table_name}
                );
                """.format(graph_idx=graph_idx, duration_constraint=duration_constraint, obj_union_fields=obj_union_fields, table_name=table_name)
                # print(sql_string)
                cur.execute(sql_string)
                # print("windowed: ", time.time() - _start_windowed)

                _start_contiguous = time.time()
                sql_string = """
                    CREATE TEMPORARY TABLE g{graph_idx}_contiguous AS (
                    SELECT vid, {obj_union_fields}, min(fid_offset) AS fid
                    FROM g{graph_idx}_windowed
                    WHERE fid_offset = fid + ({duration_constraint} - 1)
                    GROUP BY vid, {obj_union_fields}
                );
                """.format(graph_idx=graph_idx, obj_union_fields=obj_union_fields, duration_constraint=duration_constraint)
                # print(sql_string)
                cur.execute(sql_string)
                # print("contiguous: ", time.time() - _start_contiguous)
                # Store new cached results
                _start_store_cache = time.time()
                for input_vid in delta_input_vids:
                    new_memoize_sequence[input_vid][seq_signature] = pd.DataFrame()
                _start_execute = time.time()
                cur.execute("SELECT * FROM g{}_contiguous".format(graph_idx))
                df = pd.DataFrame(cur.fetchall())
                # print("[store cache]: fetchall", time.time() - _start_execute)
                _start_store = time.time()
                if df.shape[0]: # if results not empty
                    df.columns = [x.name for x in cur.description]
                    for vid, group in df.groupby("vid"):
                        cached_df = group.reset_index(drop=True)
                        new_memoize_sequence[vid][seq_signature] = cached_df
                # print("[store cache]: store", time.time() - _start_store)
                # print("store cache: ", time.time() - _start_store_cache)
                # Appending cached results of seen videos:
                _start_append = time.time()
                if cached_results.shape[0]:
                    # save dataframe to an in memory buffer
                    buffer = StringIO()
                    cached_results.to_csv(buffer, header=False, index = False)
                    buffer.seek(0)
                    cur.copy_from(buffer, "g{}_contiguous".format(graph_idx), sep=",")
                # print("append: ", time.time() - _start_append)
                encountered_variables_prev_graphs = obj_union
                encountered_variables_current_graph = []

            cur.execute("SELECT DISTINCT vid FROM g{}_contiguous".format(len(current_query) - 1))
            # print("SELECT DISTINCT vid FROM g{}_contiguous".format(len(current_query) - 1))
            output_vids = cur.fetchall()
            output_vids = [row[0] for row in output_vids]
    return output_vids, new_memoize_scene_graph, new_memoize_sequence

def postgres_execute_cache_sequence_using_temp_tables(conn, current_query, memoize_scene_graph, memoize_sequence, inputs_table_name, input_vids, is_trajectory=True, sampling_rate=None):
    """
    [Deprecated]
    input_vids: list of video segment ids
    Example query:
        Sequencing(Sequencing(Sequencing(Sequencing(Sequencing(Sequencing(True*, Near_1.05), True*), Conjunction(LeftOf, Behind)), True*), Duration(Conjunction(TopQuadrant, Far_0.9), 5)), True*)
    two types of caches:
        1. scene graph cache (without duration constraints): cache[graph] = vid, fid, oids (where fid is every frame that satisfies the graph)
        2. sequence cache: cache[sequence] = vid, fid, oids (where fid is the minimum frame that satisfies the sequence)
        Example: "g1", "(g1, d1)", "g2", "(g1, d1); (g2, d2)", "g3", "(g1, d1); (g2, d2); (g3, d3)"
    Output:
    new_memoize: new cached results from this query, which will be added to the global cache (for multi-threading)
    """

    if inputs_table_name.startswith("Obj_warsaw") or inputs_table_name.startswith("Obj_shibuya"):
        is_traffic = True
    else:
        is_traffic = False
    connect_time = 0
    prepare_cache_time = 0
    create_temp_table_time = 0
    execute_time = 0
    filtered_time = 0
    windowed_time = 0
    contiguous_time = 0
    store_cache_time = 0
    drop_temp_table_time = 0
    commit_time = 0
    other_time = 0
    _start_connect = time.time()
    temp_views = []

    with conn.cursor() as cur:
        connect_time = time.time() - _start_connect
        """
        Caching implementation:
        cached_df_deque = []
        cached_vids_deque = []
        remaining_vids = input_vids
        for each query (from the most top-level to the most bottom-level):
            cached_df_per_query = []
            cached_vids_per_query = []
            next_remaining_vids = []
            for each video segment in remaining_vids:
                if the result is cached:
                    add the result to cached_df_per_query
                    add the video segment id to cached_vids_per_query
                else:
                    add the video segment id to next_remaining_vids
            push cached_df_per_query to cached_df_deque
            push cached_vids_per_query to cached_vids_deque
            remaining_vids = next_remaining_vids
        """
        _start_prepare_cache = time.time()
        new_memoize_scene_graph = [{} for _ in range(len(memoize_scene_graph))]
        new_memoize_sequence = [{} for _ in range(len(memoize_sequence))]

        # Prepare cache result
        filtered_vids = []
        cached_df_seq_deque = deque()
        cached_vids_deque = deque()
        if isinstance(input_vids, int):
            remaining_vids = set(range(input_vids))
        else:
            remaining_vids = set(input_vids)
        signatures = deque()
        for i in range(len(current_query)):
            cached_df_seq_per_query = [pd.DataFrame()]

            # sequence cache
            seq_signature = rewrite_program_postgres(current_query[:len(current_query)-i], not is_trajectory)
            cached_vids_per_query = set()
            next_remaining_vids = set()
            for vid in remaining_vids:
                if seq_signature in memoize_sequence[vid]:
                    cached_df_seq_per_query.append(memoize_sequence[vid][seq_signature])
                    cached_vids_per_query.add(vid)
                else:
                    next_remaining_vids.add(vid)
            cached_df_seq_per_query = pd.concat(cached_df_seq_per_query, ignore_index=True)
            cached_df_seq_deque.append(cached_df_seq_per_query)
            cached_vids_deque.append(cached_vids_per_query)
            if i == 0:
                filtered_vids = list(next_remaining_vids)
            remaining_vids = next_remaining_vids

            signatures.append(seq_signature)
        cached_vids_deque.append(remaining_vids)
        # print("filtered_vids", filtered_vids)
        prepare_cache_time = time.time() - _start_prepare_cache
        # select input videos
        _start_create_temp_table = time.time()
        if isinstance(input_vids, int):
            if sampling_rate:
                cur.execute("CREATE TEMPORARY TABLE Obj_filtered ON COMMIT DROP AS SELECT oid, vid, fid / {} as fid, {} FROM {} WHERE vid < {} AND fid % {} = 0;".format(sampling_rate, "x1, y1, x2, y2, vx, vy, ax, ay" if is_traffic else "x1, y1, x2, y2, shape, color, material", inputs_table_name, input_vids, sampling_rate))
            else:
                cur.execute("CREATE TEMPORARY TABLE Obj_filtered ON COMMIT DROP AS SELECT * FROM {} WHERE vid < {};".format(inputs_table_name, input_vids))
            input_vids = list(range(input_vids))
        else:
            if sampling_rate:
                cur.execute("CREATE TEMPORARY TABLE Obj_filtered ON COMMIT DROP AS SELECT oid, vid, fid / {} as fid, {} FROM {} WHERE vid = ANY(%s) AND fid %% {} = 0;".format(sampling_rate, "x1, y1, x2, y2, vx, vy, ax, ay" if is_traffic else "x1, y1, x2, y2, shape, color, material", inputs_table_name, sampling_rate), [filtered_vids])
            else:
                cur.execute("CREATE TEMPORARY TABLE Obj_filtered ON COMMIT DROP AS SELECT * FROM {} WHERE vid = ANY(%s);".format(inputs_table_name), [filtered_vids])
        cur.execute("CREATE INDEX IF NOT EXISTS idx_obj_filtered ON Obj_filtered (vid, fid, oid);")
        # print("select input videos: ", time.time() - _start)
        encountered_variables_prev_graphs = []
        encountered_variables_current_graph = []
        delta_input_vids = []
        create_temp_table_time = time.time() - _start_create_temp_table
        for graph_idx, dict in enumerate(current_query):
            _start_other = time.time()
            # Generate scene graph:
            scene_graph = dict["scene_graph"]
            duration_constraint = dict["duration_constraint"]
            for p in scene_graph:
                for v in p["variables"]:
                    if v not in encountered_variables_current_graph:
                        encountered_variables_current_graph.append(v)

            delta_input_vids.extend(cached_vids_deque.pop())
            other_time += time.time() - _start_other
            # Execute for unseen videos
            _start_execute = time.time()
            encountered_variables_current_graph = sorted(encountered_variables_current_graph, key=lambda x: int(x[1:]))
            tables = ", ".join(["Obj_filtered as {}".format(v) for v in encountered_variables_current_graph])
            where_clauses = []
            where_clauses.append("{}.vid = ANY(%s)".format(encountered_variables_current_graph[0]))
            for i in range(len(encountered_variables_current_graph)-1):
                where_clauses.append("{v1}.vid = {v2}.vid and {v1}.fid = {v2}.fid".format(v1=encountered_variables_current_graph[i], v2=encountered_variables_current_graph[i+1])) # join variables
            for p in scene_graph:
                predicate = p["predicate"]
                parameter = p["parameter"]
                variables = p["variables"]
                args = []
                for v in variables:
                    if is_traffic:
                        args.append("{v}.x1, {v}.y1, {v}.x2, {v}.y2, {v}.vx, {v}.vy, {v}.ax, {v}.ay".format(v=v))
                    else:
                        args.append("{v}.shape, {v}.color, {v}.material, {v}.x1, {v}.y1, {v}.x2, {v}.y2".format(v=v))
                args = ", ".join(args)
                if parameter:
                    if isinstance(parameter, str):
                        args = "'{}', {}".format(parameter, args)
                    else:
                        args = "{}, {}".format(parameter, args)
                where_clauses.append("{}({}) = true".format(predicate, args))
            if is_trajectory:
                # only for trajectory example
                for v in encountered_variables_current_graph:
                    where_clauses.append("{}.oid = {}".format(v, v[1:]))
            else:
                # For general case
                for var_pair in itertools.combinations(encountered_variables_current_graph, 2):
                    where_clauses.append("{}.oid <> {}.oid".format(var_pair[0], var_pair[1]))
            where_clauses = " and ".join(where_clauses)
            fields = "{v}.vid as vid, {v}.fid as fid, ".format(v=encountered_variables_current_graph[0])
            fields += ", ".join(["{v}.oid as {v}_oid".format(v=v) for v in encountered_variables_current_graph])
            oid_list = ["{}_oid".format(v) for v in encountered_variables_current_graph]
            oids = ", ".join(oid_list)
            sql_sring = """
            CREATE TEMPORARY VIEW g{} AS
            SELECT {}
            FROM {}
            WHERE {};
            """.format(graph_idx, fields, tables, where_clauses)
            # print(sql_sring)
            cur.execute(sql_sring, [delta_input_vids])
            temp_views.append("g{}".format(graph_idx))
            # cur.execute("CREATE INDEX IF NOT EXISTS idx_g{} ON g{} (vid);".format(graph_idx, graph_idx))
            # print("execute for unseen videos: ", time.time() - _start_execute)
            execute_time += time.time() - _start_execute
            # print("Time for graph {}: {}".format(graph_idx, time.time() - _start))

            _start_other = time.time()
            # Read cached results
            seq_signature = signatures.pop()
            cached_results = cached_df_seq_deque.pop()
            other_time += time.time() - _start_other

            _start_filtered = time.time()
            if graph_idx > 0:
                obj_union = copy.deepcopy(encountered_variables_prev_graphs)
                obj_union_fields = []
                obj_intersection_fields = []
                for v in encountered_variables_prev_graphs:
                    obj_union_fields.append("t0.{}_oid".format(v))
                for v in encountered_variables_current_graph:
                    if v in encountered_variables_prev_graphs:
                        obj_intersection_fields.append("t0.{v}_oid = t1.{v}_oid".format(v=v))
                    else:
                        for u in encountered_variables_prev_graphs:
                            obj_intersection_fields.append("t0.{u}_oid <> t1.{v}_oid".format(u=u, v=v))
                        obj_union.append(v)
                        obj_union_fields.append("t1.{}_oid".format(v))
                obj_union_fields = ", ".join(obj_union_fields)
                obj_intersection_fields = " and ".join(obj_intersection_fields)
                # where_clauses = "t0.vid = ANY(%s)"
                # if current_seq == "g0_seq_view":
                #     where_clauses += " and t0.vid = t1.vid and t0.fid2 < t1.fid1"
                # else:
                #     where_clauses += " and t0.vid = t1.vid and t0.fid < t1.fid1"
                sql_string = """
                CREATE TEMPORARY VIEW g{graph_idx}_filtered AS (
                    SELECT t0.vid, t1.fid, {obj_union_fields}
                    FROM g{graph_idx_prev}_contiguous t0, g{graph_idx} t1
                    WHERE t0.vid = t1.vid AND {obj_intersection_fields} AND t0.fid < t1.fid
                );
                """.format(graph_idx=graph_idx, graph_idx_prev=graph_idx-1, obj_union_fields=obj_union_fields, obj_intersection_fields=obj_intersection_fields)
                # print(sql_string)
                cur.execute(sql_string)
                temp_views.append("g{}_filtered".format(graph_idx))
            else:
                obj_union = encountered_variables_current_graph
            # print("filtered: ", time.time() - _start_filtered)
            filtered_time += time.time() - _start_filtered

            # Generate scene graph sequence:
            _start_windowed = time.time()
            table_name = "g{}_filtered".format(graph_idx) if graph_idx > 0 else "g{}".format(graph_idx)
            obj_union_fields = ", ".join(["{}_oid".format(v) for v in obj_union])
            sql_string = """
                CREATE TEMPORARY VIEW g{graph_idx}_windowed AS (
                SELECT vid, fid, {obj_union_fields},
                lead(fid, {duration_constraint} - 1, 0) OVER (PARTITION BY vid, {obj_union_fields} ORDER BY fid) as fid_offset
                FROM {table_name}
            );
            """.format(graph_idx=graph_idx, duration_constraint=duration_constraint, obj_union_fields=obj_union_fields, table_name=table_name)
            # print(sql_string)
            cur.execute(sql_string)
            temp_views.append("g{}_windowed".format(graph_idx))
            # print("windowed: ", time.time() - _start_windowed)
            windowed_time += time.time() - _start_windowed
            _start_contiguous = time.time()
            sql_string = """
                CREATE TEMPORARY TABLE g{graph_idx}_contiguous ON COMMIT DROP AS (
                SELECT vid, {obj_union_fields}, min(fid_offset) AS fid
                FROM g{graph_idx}_windowed
                WHERE fid_offset = fid + ({duration_constraint} - 1)
                GROUP BY vid, {obj_union_fields}
            );
            """.format(graph_idx=graph_idx, obj_union_fields=obj_union_fields, duration_constraint=duration_constraint)
            # print(sql_string)
            cur.execute(sql_string)
            # print("contiguous: ", time.time() - _start_contiguous)
            contiguous_time += time.time() - _start_contiguous
            _start_store = time.time()
            # Store new cached results
            for input_vid in delta_input_vids:
                new_memoize_sequence[input_vid][seq_signature] = pd.DataFrame()
            cur.execute("SELECT * FROM g{}_contiguous".format(graph_idx))
            df = pd.DataFrame(cur.fetchall())
            # print("[store cache]: fetchall", time.time() - _start_execute)
            # _start_store = time.time()
            if df.shape[0]: # if results not empty
                df.columns = [x.name for x in cur.description]
                for vid, group in df.groupby("vid"):
                    cached_df = group.reset_index(drop=True)
                    new_memoize_sequence[vid][seq_signature] = cached_df
            # print("[store cache]: store", time.time() - _start_store)
            # Appending cached results of seen videos:
            # _start_append = time.time()
            if cached_results.shape[0]:
                # print("cached_results", cached_results.head())
                # save dataframe to an in memory buffer
                buffer = StringIO()
                cached_results.to_csv(buffer, header=False, index = False)
                buffer.seek(0)
                cur.copy_from(buffer, "g{}_contiguous".format(graph_idx), sep=",")
            # print("append: ", time.time() - _start_append)
            encountered_variables_prev_graphs = obj_union
            encountered_variables_current_graph = []
            store_cache_time += time.time() - _start_store
        _start_other = time.time()
        cur.execute("SELECT DISTINCT vid FROM g{}_contiguous".format(len(current_query) - 1))
        # print("SELECT DISTINCT vid FROM g{}_contiguous".format(len(current_query) - 1))
        output_vids = cur.fetchall()
        output_vids = [row[0] for row in output_vids]
        other_time += time.time() - _start_other
        # Drop views
        _start_drop = time.time()
        cur.execute("DROP VIEW {}".format(", ".join(temp_views)))
        drop_temp_table_time = time.time() - _start_drop
        _start_commit = time.time()
        conn.commit()
        commit_time = time.time() - _start_commit
    return output_vids, new_memoize_scene_graph, new_memoize_sequence, connect_time, prepare_cache_time, create_temp_table_time, execute_time, filtered_time, windowed_time, contiguous_time, store_cache_time, drop_temp_table_time, commit_time, other_time


def postgres_execute_cache_sequence(conn, current_query, memo, inputs_table_name, input_vids, is_trajectory=True, sampling_rate=None):
    """
    This method uses temp views and only caches binary query predictions.
    input_vids: list of video segment ids
    Example query:
        Sequencing(Sequencing(Sequencing(Sequencing(Sequencing(Sequencing(True*, Near_1.05), True*), Conjunction(LeftOf, Behind)), True*), Duration(Conjunction(TopQuadrant, Far_0.9), 5)), True*)
    Only cache binary query prediction, rather than [vid, fid, oids]: cache[graph] = 1 (positive) or 0 (negative)
    Output:
    new_memoize: new cached results from this query, which will be added to the global cache (for multi-threading)
    """
    if inputs_table_name.startswith("Obj_warsaw") or inputs_table_name.startswith("Obj_shibuya"):
        is_traffic = True
    else:
        is_traffic = False

    store_cache_time = 0
    temp_views = []

    with conn.cursor() as cur:
        _start_prepare_cache = time.time()
        new_memo = [{} for _ in range(len(memo))]

        output_vids = []
        # Prepare cache result
        if isinstance(input_vids, int):
            remaining_vids = set(range(input_vids))
        else:
            remaining_vids = set(input_vids)

        signatures = []
        for i in range(len(current_query)):
            seq_signature = rewrite_program_postgres(current_query[:(i+1)], not is_trajectory)
            signatures.append(seq_signature)

        filtered_vids = []
        cached_output_vids = []
        for vid in remaining_vids:
            for i, seq_signature in enumerate(signatures):
                if seq_signature not in memo[vid]:
                    filtered_vids.append(vid)
                elif memo[vid][seq_signature] == 0:
                    break
                elif i == len(signatures) - 1: # The full query predicates it as positive
                    cached_output_vids.append(vid)

        prepare_cache_time = time.time() - _start_prepare_cache
        # select input videos
        _start_create_temp_table = time.time()
        if isinstance(input_vids, int):
            if sampling_rate:
                cur.execute("CREATE TEMPORARY TABLE Obj_filtered ON COMMIT DROP AS SELECT oid, vid, fid / {} as fid, {} FROM {} WHERE vid < {} AND fid % {} = 0;".format(sampling_rate, "x1, y1, x2, y2, vx, vy, ax, ay" if is_traffic else "x1, y1, x2, y2, shape, color, material", inputs_table_name, input_vids, sampling_rate))
            else:
                cur.execute("CREATE TEMPORARY TABLE Obj_filtered ON COMMIT DROP AS SELECT * FROM {} WHERE vid < {};".format(inputs_table_name, input_vids))
            filtered_vids = list(range(input_vids))
        else:
            if sampling_rate:
                cur.execute("CREATE TEMPORARY TABLE Obj_filtered ON COMMIT DROP AS SELECT oid, vid, fid / {} as fid, {} FROM {} WHERE vid = ANY(%s) AND fid %% {} = 0;".format(sampling_rate, "x1, y1, x2, y2, vx, vy, ax, ay" if is_traffic else "x1, y1, x2, y2, shape, color, material", inputs_table_name, sampling_rate), [filtered_vids])
            else:
                cur.execute("CREATE TEMPORARY TABLE Obj_filtered ON COMMIT DROP AS SELECT * FROM {} WHERE vid = ANY(%s);".format(inputs_table_name), [filtered_vids])
        cur.execute("CREATE INDEX IF NOT EXISTS idx_obj_filtered ON Obj_filtered (vid, fid);")
        # print("select input videos: ", time.time() - _start)
        encountered_variables_prev_graphs = []
        encountered_variables_current_graph = []
        for graph_idx, dict in enumerate(current_query):
            _start = time.time()
            # Generate scene graph:
            scene_graph = dict["scene_graph"]
            duration_constraint = dict["duration_constraint"]
            for p in scene_graph:
                for v in p["variables"]:
                    if v not in encountered_variables_current_graph:
                        encountered_variables_current_graph.append(v)

            # Execute for unseen videos
            _start_execute = time.time()
            encountered_variables_current_graph = sorted(encountered_variables_current_graph, key=lambda x: int(x[1:]))
            tables = ", ".join(["Obj_filtered as {}".format(v) for v in encountered_variables_current_graph])
            where_clauses = []
            for i in range(len(encountered_variables_current_graph)-1):
                where_clauses.append("{v1}.vid = {v2}.vid and {v1}.fid = {v2}.fid".format(v1=encountered_variables_current_graph[i], v2=encountered_variables_current_graph[i+1])) # join variables
            for p in scene_graph:
                predicate = p["predicate"]
                parameter = p["parameter"]
                variables = p["variables"]
                args = []
                for v in variables:
                    args.append("{v}.shape, {v}.color, {v}.material, {v}.x1, {v}.y1, {v}.x2, {v}.y2".format(v=v))
                args = ", ".join(args)
                if parameter:
                    if isinstance(parameter, str):
                        args = "'{}', {}".format(parameter, args)
                    else:
                        args = "{}, {}".format(parameter, args)
                where_clauses.append("{}({}) = true".format(predicate, args))
            if is_trajectory:
                # only for trajectory example
                for v in encountered_variables_current_graph:
                    where_clauses.append("{}.oid = {}".format(v, v[1:]))
            else:
                # For general case
                for var_pair in itertools.combinations(encountered_variables_current_graph, 2):
                    where_clauses.append("{}.oid <> {}.oid".format(var_pair[0], var_pair[1]))
            where_clauses = " and ".join(where_clauses)
            fields = "{v}.vid as vid, {v}.fid as fid, ".format(v=encountered_variables_current_graph[0])
            fields += ", ".join(["{v}.oid as {v}_oid".format(v=v) for v in encountered_variables_current_graph])
            oid_list = ["{}_oid".format(v) for v in encountered_variables_current_graph]
            oids = ", ".join(oid_list)
            sql_sring = """CREATE TEMPORARY VIEW g{} AS SELECT {} FROM {} WHERE {};""".format(graph_idx, fields, tables, where_clauses)
            # print(sql_sring)
            cur.execute(sql_sring)
            temp_views.append("g{}".format(graph_idx))
            # print("execute for unseen videos: ", time.time() - _start_execute)
            # print("Time for graph {}: {}".format(graph_idx, time.time() - _start))

            _start_filtered = time.time()
            if graph_idx > 0:
                obj_union = copy.deepcopy(encountered_variables_prev_graphs)
                obj_union_fields = []
                obj_intersection_fields = []
                for v in encountered_variables_prev_graphs:
                    obj_union_fields.append("t0.{}_oid".format(v))
                for v in encountered_variables_current_graph:
                    if v in encountered_variables_prev_graphs:
                        obj_intersection_fields.append("t0.{v}_oid = t1.{v}_oid".format(v=v))
                    else:
                        for u in encountered_variables_prev_graphs:
                            obj_intersection_fields.append("t0.{u}_oid <> t1.{v}_oid".format(u=u, v=v))
                        obj_union.append(v)
                        obj_union_fields.append("t1.{}_oid".format(v))
                obj_union_fields = ", ".join(obj_union_fields)
                obj_intersection_fields = " and ".join(obj_intersection_fields)
                # where_clauses = "t0.vid = ANY(%s)"
                # if current_seq == "g0_seq_view":
                #     where_clauses += " and t0.vid = t1.vid and t0.fid2 < t1.fid1"
                # else:
                #     where_clauses += " and t0.vid = t1.vid and t0.fid < t1.fid1"
                sql_string = """
                CREATE TEMPORARY VIEW g{graph_idx}_filtered AS (
                    SELECT t0.vid, t1.fid, {obj_union_fields}
                    FROM g{graph_idx_prev}_contiguous t0, g{graph_idx} t1
                    WHERE t0.vid = t1.vid AND {obj_intersection_fields} AND t0.fid < t1.fid
                );
                """.format(graph_idx=graph_idx, graph_idx_prev=graph_idx-1, obj_union_fields=obj_union_fields, obj_intersection_fields=obj_intersection_fields)
                # print(sql_string)
                cur.execute(sql_string)
                temp_views.append("g{}_filtered".format(graph_idx))
            else:
                obj_union = encountered_variables_current_graph
            # print("filtered: ", time.time() - _start_filtered)

            # Generate scene graph sequence:
            _start_windowed = time.time()
            table_name = "g{}_filtered".format(graph_idx) if graph_idx > 0 else "g{}".format(graph_idx)
            obj_union_fields = ", ".join(["{}_oid".format(v) for v in obj_union])
            _start = time.time()
            sql_string = """
                CREATE TEMPORARY VIEW g{graph_idx}_windowed AS (
                SELECT vid, fid, {obj_union_fields},
                lead(fid, {duration_constraint} - 1, 0) OVER (PARTITION BY vid, {obj_union_fields} ORDER BY fid) as fid_offset
                FROM {table_name}
            );
            """.format(graph_idx=graph_idx, duration_constraint=duration_constraint, obj_union_fields=obj_union_fields, table_name=table_name)
            # print(sql_string)
            cur.execute(sql_string)
            temp_views.append("g{}_windowed".format(graph_idx))
            # print("windowed: ", time.time() - _start_windowed)

            _start_contiguous = time.time()
            sql_string = """
                CREATE TEMPORARY VIEW g{graph_idx}_contiguous AS (
                SELECT vid, {obj_union_fields}, min(fid_offset) AS fid
                FROM g{graph_idx}_windowed
                WHERE fid_offset = fid + ({duration_constraint} - 1)
                GROUP BY vid, {obj_union_fields}
            );
            """.format(graph_idx=graph_idx, obj_union_fields=obj_union_fields, duration_constraint=duration_constraint)
            # print(sql_string)
            cur.execute(sql_string)
            temp_views.append("g{}_contiguous".format(graph_idx))
            cur.execute("SELECT DISTINCT vid FROM g{}_contiguous".format(graph_idx))
            res = cur.fetchall()
            output_vids = [row[0] for row in res]

            _start_store = time.time()
            # Store new cached results
            for input_vid in filtered_vids:
                if input_vid in output_vids:
                    new_memo[input_vid][signatures[graph_idx]] = 1
                else:
                    new_memo[input_vid][signatures[graph_idx]] = 0
            store_cache_time += time.time() - _start_store

            # print("contiguous: ", time.time() - _start_contiguous)
            encountered_variables_prev_graphs = obj_union
            encountered_variables_current_graph = []
        # cur.execute("SELECT DISTINCT vid FROM g{}_contiguous".format(len(current_query) - 1))
        # # print("SELECT DISTINCT vid FROM g{}_contiguous".format(len(current_query) - 1))
        # res = cur.fetchall()
        # output_vids.extend([row[0] for row in res])
        output_vids.extend(cached_output_vids)

        # Drop views
        cur.execute("DROP VIEW {}".format(", ".join(temp_views)))
        # Commit
        conn.commit()
    return output_vids, new_memo


def postgres_execute_no_caching(dsn, current_query, memo, inputs_table_name, input_vids, is_trajectory=True, sampling_rate=None):
    """
    input_vids: list of video segment ids
    Example query:
        Sequencing(Sequencing(Sequencing(Sequencing(Sequencing(Sequencing(True*, Near_1.05), True*), Conjunction(LeftOf, Behind)), True*), Duration(Conjunction(TopQuadrant, Far_0.9), 5)), True*)
    two types of caches:
        1. scene graph cache (without duration constraints): cache[graph] = vid, fid, oids
        2. sequence cache: cache[sequence] = vid, fid, oids
        Example: g1, (g1, d1), g2, (g1, d1); (g2, d2), g3, (g1, d1); (g2, d2); (g3, d3)
    Output:
    new_memoize: new cached results from this query, which will be added to the global cache (for multi-threading)
    """
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            new_memo = [{} for _ in range(len(memo))]
            # select input videos
            _start = time.time()
            if isinstance(input_vids, int):
                if sampling_rate:
                    cur.execute("CREATE TEMPORARY TABLE Obj_filtered AS SELECT oid, vid, fid / {} as fid, shape, color, material, x1, y1, x2, y2 FROM {} WHERE vid < {} AND fid % {} = 0;".format(sampling_rate, inputs_table_name, input_vids, sampling_rate))
                else:
                    cur.execute("CREATE TEMPORARY TABLE Obj_filtered AS SELECT * FROM {} WHERE vid < {};".format(inputs_table_name, input_vids))
                input_vids = list(range(input_vids))
            else:
                if sampling_rate:
                    cur.execute("CREATE TEMPORARY TABLE Obj_filtered AS SELECT oid, vid, fid / {} as fid, shape, color, material, x1, y1, x2, y2 FROM {} WHERE vid = ANY(%s) AND fid %% {} = 0;".format(sampling_rate, inputs_table_name, sampling_rate), [input_vids])
                else:
                    cur.execute("CREATE TEMPORARY TABLE Obj_filtered AS SELECT * FROM {} WHERE vid = ANY(%s);".format(inputs_table_name), [input_vids])
            cur.execute("CREATE INDEX IF NOT EXISTS idx_obj_filtered ON Obj_filtered (vid, fid);")
            # print("select input videos: ", time.time() - _start)
            encountered_variables_prev_graphs = []
            encountered_variables_current_graph = []
            for graph_idx, dict in enumerate(current_query):
                _start = time.time()
                # Generate scene graph:
                scene_graph = dict["scene_graph"]
                duration_constraint = dict["duration_constraint"]
                for p in scene_graph:
                    for v in p["variables"]:
                        if v not in encountered_variables_current_graph:
                            encountered_variables_current_graph.append(v)

                # Execute for unseen videos
                _start_execute = time.time()
                encountered_variables_current_graph = sorted(encountered_variables_current_graph, key=lambda x: int(x[1:]))
                tables = ", ".join(["Obj_filtered as {}".format(v) for v in encountered_variables_current_graph])
                where_clauses = []
                for i in range(len(encountered_variables_current_graph)-1):
                    where_clauses.append("{v1}.vid = {v2}.vid and {v1}.fid = {v2}.fid".format(v1=encountered_variables_current_graph[i], v2=encountered_variables_current_graph[i+1])) # join variables
                for p in scene_graph:
                    predicate = p["predicate"]
                    parameter = p["parameter"]
                    variables = p["variables"]
                    args = []
                    for v in variables:
                        args.append("{v}.shape, {v}.color, {v}.material, {v}.x1, {v}.y1, {v}.x2, {v}.y2".format(v=v))
                    args = ", ".join(args)
                    if parameter:
                        if isinstance(parameter, str):
                            args = "'{}', {}".format(parameter, args)
                        else:
                            args = "{}, {}".format(parameter, args)
                    where_clauses.append("{}({}) = true".format(predicate, args))
                if is_trajectory:
                    # only for trajectory example
                    for v in encountered_variables_current_graph:
                        where_clauses.append("{}.oid = {}".format(v, v[1:]))
                else:
                    # For general case
                    for var_pair in itertools.combinations(encountered_variables_current_graph, 2):
                        where_clauses.append("{}.oid <> {}.oid".format(var_pair[0], var_pair[1]))
                where_clauses = " and ".join(where_clauses)
                fields = "{v}.vid as vid, {v}.fid as fid, ".format(v=encountered_variables_current_graph[0])
                fields += ", ".join(["{v}.oid as {v}_oid".format(v=v) for v in encountered_variables_current_graph])
                oid_list = ["{}_oid".format(v) for v in encountered_variables_current_graph]
                oids = ", ".join(oid_list)
                sql_sring = """CREATE TEMPORARY VIEW g{} AS SELECT {} FROM {} WHERE {};""".format(graph_idx, fields, tables, where_clauses)
                # print(sql_sring)
                cur.execute(sql_sring)
                # print("execute for unseen videos: ", time.time() - _start_execute)
                # print("Time for graph {}: {}".format(graph_idx, time.time() - _start))

                _start_filtered = time.time()
                if graph_idx > 0:
                    obj_union = copy.deepcopy(encountered_variables_prev_graphs)
                    obj_union_fields = []
                    obj_intersection_fields = []
                    for v in encountered_variables_prev_graphs:
                        obj_union_fields.append("t0.{}_oid".format(v))
                    for v in encountered_variables_current_graph:
                        if v in encountered_variables_prev_graphs:
                            obj_intersection_fields.append("t0.{v}_oid = t1.{v}_oid".format(v=v))
                        else:
                            for u in encountered_variables_prev_graphs:
                                obj_intersection_fields.append("t0.{u}_oid <> t1.{v}_oid".format(u=u, v=v))
                            obj_union.append(v)
                            obj_union_fields.append("t1.{}_oid".format(v))
                    obj_union_fields = ", ".join(obj_union_fields)
                    obj_intersection_fields = " and ".join(obj_intersection_fields)
                    # where_clauses = "t0.vid = ANY(%s)"
                    # if current_seq == "g0_seq_view":
                    #     where_clauses += " and t0.vid = t1.vid and t0.fid2 < t1.fid1"
                    # else:
                    #     where_clauses += " and t0.vid = t1.vid and t0.fid < t1.fid1"
                    sql_string = """
                    CREATE TEMPORARY VIEW g{graph_idx}_filtered AS (
                        SELECT t0.vid, t1.fid, {obj_union_fields}
                        FROM g{graph_idx_prev}_contiguous t0, g{graph_idx} t1
                        WHERE t0.vid = t1.vid AND {obj_intersection_fields} AND t0.fid < t1.fid
                    );
                    """.format(graph_idx=graph_idx, graph_idx_prev=graph_idx-1, obj_union_fields=obj_union_fields, obj_intersection_fields=obj_intersection_fields)
                    # print(sql_string)
                    cur.execute(sql_string)
                else:
                    obj_union = encountered_variables_current_graph
                # print("filtered: ", time.time() - _start_filtered)

                # Generate scene graph sequence:
                _start_windowed = time.time()
                table_name = "g{}_filtered".format(graph_idx) if graph_idx > 0 else "g{}".format(graph_idx)
                obj_union_fields = ", ".join(["{}_oid".format(v) for v in obj_union])
                _start = time.time()
                sql_string = """
                    CREATE TEMPORARY VIEW g{graph_idx}_windowed AS (
                    SELECT vid, fid, {obj_union_fields},
                    lead(fid, {duration_constraint} - 1, 0) OVER (PARTITION BY vid, {obj_union_fields} ORDER BY fid) as fid_offset
                    FROM {table_name}
                );
                """.format(graph_idx=graph_idx, duration_constraint=duration_constraint, obj_union_fields=obj_union_fields, table_name=table_name)
                # print(sql_string)
                cur.execute(sql_string)
                # print("windowed: ", time.time() - _start_windowed)

                _start_contiguous = time.time()
                sql_string = """
                    CREATE TEMPORARY VIEW g{graph_idx}_contiguous AS (
                    SELECT vid, {obj_union_fields}, min(fid_offset) AS fid
                    FROM g{graph_idx}_windowed
                    WHERE fid_offset = fid + ({duration_constraint} - 1)
                    GROUP BY vid, {obj_union_fields}
                );
                """.format(graph_idx=graph_idx, obj_union_fields=obj_union_fields, duration_constraint=duration_constraint)
                # print(sql_string)
                cur.execute(sql_string)
                # print("contiguous: ", time.time() - _start_contiguous)
                encountered_variables_prev_graphs = obj_union
                encountered_variables_current_graph = []

            cur.execute("SELECT DISTINCT vid FROM g{}_contiguous".format(len(current_query) - 1))
            # print("SELECT DISTINCT vid FROM g{}_contiguous".format(len(current_query) - 1))
            output_vids = cur.fetchall()
            output_vids = [row[0] for row in output_vids]
            conn.commit()
    return output_vids, new_memo


def rewrite_program_postgres(orig_program, rewrite_variables=True):
    """
    Input:
    program: query in the dictionary format
    Output: query in string format, which is ordered properly (uniquely).
    NOTE: For trajectories, we don't rewrite the variables, since we expect the query to indicate which objects the predicate is referring to, as assumed in the Quivr paper.
    """
    def print_scene_graph(predicate_list):
        if len(predicate_list) == 1:
            if predicate_list[0]["parameter"]:
                predicate_name = "{}_{}".format(predicate_list[0]["predicate"], predicate_list[0]["parameter"])
            else:
                predicate_name = predicate_list[0]["predicate"]
            predicate_variables = ", ".join(predicate_list[0]["variables"])
            return "{}({})".format(predicate_name, predicate_variables)
        else:
            if predicate_list[-1]["parameter"]:
                predicate_name = "{}_{}".format(predicate_list[-1]["predicate"], predicate_list[-1]["parameter"])
            else:
                predicate_name = predicate_list[-1]["predicate"]
            predicate_variables = ", ".join(predicate_list[-1]["variables"])
            return "Conjunction({}, {}({}))".format(print_scene_graph(predicate_list[:-1]), predicate_name, predicate_variables)

    def print_query(scene_graphs):
        if len(scene_graphs) == 1:
            return scene_graphs[0]
        else:
            return "{}; {}".format(print_query(scene_graphs[:-1]), scene_graphs[-1])

    program = copy.deepcopy(orig_program)
    # Rewrite the program
    encountered_variables = []
    for dict in program:
        scene_graph = dict["scene_graph"]
        scene_graph = sorted(scene_graph, key=lambda x: x["predicate"] + " ".join(x["variables"]))
        if rewrite_variables:
            # Rewrite variables
            for i, p in enumerate(scene_graph):
                rewritten_variables = []
                for v in p["variables"]:
                    if v not in encountered_variables:
                        encountered_variables.append(v)
                        rewritten_variables.append("o" + str(len(encountered_variables) - 1))
                    else:
                        rewritten_variables.append("o" + str(encountered_variables.index(v)))
                # Sort rewritten variables
                # NOTE: Why do we want to sort?
                # We assume that the order of variables in a predicate does not matter
                rewritten_variables = sorted(rewritten_variables)
                scene_graph[i]["variables"] = rewritten_variables
        dict["scene_graph"] = scene_graph

    scene_graphs = []
    for dict in program:
        scene_graph = dict["scene_graph"]
        duration_constraint = int(dict["duration_constraint"])
        scene_graph_str = print_scene_graph(scene_graph)
        if duration_constraint > 1:
            scene_graph_str = "Duration({}, {})".format(scene_graph_str, duration_constraint)
        scene_graphs.append(scene_graph_str)

    query = print_query(scene_graphs)
    return query


def str_to_program_postgres(program_str):
    def parse_submodules(scene_graph_str):
        idx = scene_graph_str.find("(")
        idx_r = scene_graph_str.rfind(")")
        submodules = scene_graph_str[idx+1:idx_r]
        counter = 0
        submodule_list = []
        submodule_start = 0
        for i, char in enumerate(submodules):
            if char == "," and counter == 0:
                submodule_list.append(submodules[submodule_start:i])
                submodule_start = i+2
            elif char == "(":
                counter += 1
            elif char == ")":
                counter -= 1
        submodule_list.append(submodules[submodule_start:])
        return submodule_list

    def parse_conjunction(scene_graph_str):
        if scene_graph_str.startswith("Conjunction"):
            submodule_list = parse_submodules(scene_graph_str)
            return [*parse_conjunction(submodule_list[0]), *parse_conjunction(submodule_list[1])]
        else:
            return [parse_predicate(scene_graph_str)]

    def parse_predicate(predicate_str):
        dict = {}
        # Near_0.95(o0, o1)
        idx = predicate_str.find("(")
        idx_r = predicate_str.rfind(")")
        predicate_name = predicate_str[:idx].split("_")
        dict["predicate"] = predicate_name[0]
        if len(predicate_name) > 1:
            try:
                dict["parameter"] = float(predicate_name[1])
            except:
                dict["parameter"] = predicate_name[1]
        else:
            dict["parameter"] = None
        # dict["parameter"] = float(predicate_name[1]) if len(predicate_name) > 1 else None
        predicate_variables = predicate_str[idx+1:idx_r]
        dict["variables"] = predicate_variables.split(", ")
        return dict

    scene_graphs_str = program_str.split("; ")
    program = []
    for scene_graph_str in scene_graphs_str:
        duration_constraint = 1
        if scene_graph_str.startswith("Duration"):
            submodule_list = parse_submodules(scene_graph_str)
            duration_constraint = int(submodule_list[-1])
            scene_graph_str = submodule_list[0]
        scene_graph = {"scene_graph": parse_conjunction(scene_graph_str), "duration_constraint": duration_constraint}
        program.append(scene_graph)
    return program


def rewrite_vars_name_for_scene_graph(orig_dict):
    """
    Input:
    program: scene graph in the dictionary format. Predicates in the scene graph are sorted. The only rewrite is to rename variables.
    {'scene_graph': [{'predicate': 'LeftQuadrant', 'parameter': None, 'variables': ['o2']}, {'predicate': 'RightOf', 'parameter': None, 'variables': ['o0', 'o2']}], 'duration_constraint': 1}
    Output: query in string format, which is ordered properly (uniquely), and a dictionary maintaining the mapping between the original variable names and the rewritten variable names
    """
    def print_scene_graph(predicate_list):
        if len(predicate_list) == 1:
            if predicate_list[0]["parameter"]:
                predicate_name = predicate_list[0]["predicate"] + "_" + str(predicate_list[0]["parameter"])
            else:
                predicate_name = predicate_list[0]["predicate"]
            predicate_variables = ", ".join(predicate_list[0]["variables"])
            return "{}({})".format(predicate_name, predicate_variables)
        else:
            if predicate_list[-1]["parameter"]:
                predicate_name = predicate_list[-1]["predicate"] + "_" + str(predicate_list[-1]["parameter"])
            else:
                predicate_name = predicate_list[-1]["predicate"]
            predicate_variables = ", ".join(predicate_list[-1]["variables"])
            return "Conjunction({}, {}({}))".format(print_scene_graph(predicate_list[:-1]), predicate_name, predicate_variables)

    rewritten_dict = copy.deepcopy(orig_dict)
    vars_mapping = {}
    vars_inverted_mapping = {}
    # Rewrite the program
    encountered_variables = []
    scene_graph = rewritten_dict["scene_graph"]
    for i, p in enumerate(scene_graph):
        rewritten_variables = []
        for v in p["variables"]:
            if v not in encountered_variables:
                encountered_variables.append(v)
                rewritten_variables.append("o" + str(len(encountered_variables) - 1))
                vars_mapping["{}_oid".format(v)] = "o" + str(len(encountered_variables) - 1) + "_oid"
                vars_inverted_mapping["o" + str(len(encountered_variables) - 1) + "_oid"] = "{}_oid".format(v)
            else:
                rewritten_variables.append("o" + str(encountered_variables.index(v)))
        # Sort rewritten variables
        rewritten_variables = sorted(rewritten_variables)
        scene_graph[i]["variables"] = rewritten_variables
    rewritten_dict["scene_graph"] = scene_graph

    scene_graph = rewritten_dict["scene_graph"]
    duration_constraint = int(rewritten_dict["duration_constraint"])
    scene_graph_str = print_scene_graph(scene_graph)
    if duration_constraint > 1:
        scene_graph_str = "Duration({}, {})".format(scene_graph_str, duration_constraint)

    return scene_graph_str, vars_mapping


def quivr_str_to_postgres_program(quivr_str):
    # Start(Conjunction(Conjunction(Conjunction(Kleene(Behind), Kleene(Behind)), Kleene(Conjunction(Kleene(LeftQuadrant), Kleene(Near_1)))), Behind))
    if quivr_str.startswith("Near"):
        predicate = {
                        "predicate": "Near",
                        "parameter": float(quivr_str.split("(")[0].split("_")[1]),
                        "variables": ["o0", "o1"]
                    }
        scene_graph = {
                "scene_graph": [predicate],
                "duration_constraint": 1

            }
        return [scene_graph]
    elif quivr_str.startswith("Far"):
        predicate = {
                        "predicate": "Far",
                        "parameter": float(quivr_str.split("(")[0].split("_")[1]),
                        "variables": ["o0", "o1"]
                    }
        scene_graph = {
                "scene_graph": [predicate],
                "duration_constraint": 1

            }
        return [scene_graph]
    elif quivr_str.startswith("True*"):
        return None
    elif quivr_str.startswith("LeftOf"):
        predicate = {
                        "predicate": "LeftOf",
                        "parameter": None,
                        "variables": ["o0", "o1"]
                    }
        scene_graph = {
                "scene_graph": [predicate],
                "duration_constraint": 1

            }
        return [scene_graph]
    elif quivr_str.startswith("Behind"):
        predicate = {
                        "predicate": "Behind",
                        "parameter": None,
                        "variables": ["o0", "o1"]
                    }
        scene_graph = {
                "scene_graph": [predicate],
                "duration_constraint": 1

            }
        return [scene_graph]
    elif quivr_str.startswith("TopQuadrant"):
        predicate = {
                        "predicate": "TopQuadrant",
                        "parameter": None,
                        "variables": ["o0"]
                    }
        scene_graph = {
                "scene_graph": [predicate],
                "duration_constraint": 1

            }
        return [scene_graph]
    elif quivr_str.startswith("RightQuadrant"):
        predicate = {
                        "predicate": "RightQuadrant",
                        "parameter": None,
                        "variables": ["o0"]
                    }
        scene_graph = {
                "scene_graph": [predicate],
                "duration_constraint": 1

            }
        return [scene_graph]
    else:
        idx = quivr_str.find("(")
        idx_r = quivr_str.rfind(")")
        # True*, Sequencing(Near_1.0, MinLength_10.0)
        functionclass = quivr_str[:idx]
        submodules = quivr_str[idx+1:idx_r]
        counter = 0
        submodule_list = []
        submodule_start = 0
        for i, char in enumerate(submodules):
            if char == "," and counter == 0:
                submodule_list.append(submodules[submodule_start:i])
                submodule_start = i+2
            elif char == "(":
                counter += 1
            elif char == ")":
                counter -= 1
        submodule_list.append(submodules[submodule_start:])
        if functionclass == "Sequencing":
            g1 = quivr_str_to_postgres_program(submodule_list[0])
            g2 = quivr_str_to_postgres_program(submodule_list[1])
            if not g1:
                return g2
            if not g2:
                return g1
            else:
                return [*g1, *g2]
        elif functionclass == "Conjunction":
            scene_graph = quivr_str_to_postgres_program(submodule_list[0])
            scene_graph[0]["scene_graph"].append(quivr_str_to_postgres_program(submodule_list[1])[0]["scene_graph"][0])
            return scene_graph
        elif functionclass == "Duration":
            scene_graph = quivr_str_to_postgres_program(submodule_list[0])
            scene_graph[0]["duration_constraint"] = int(submodule_list[1])
            return scene_graph

if __name__ == '__main__':
    dsn = "dbname=myinner_db user=enhaoz host=localhost"

    memoize_scene_graph = [{} for _ in range(10000)]
    memoize_sequence = [{} for _ in range(10000)]
    inputs_table_name = "Obj_clevrer"
    # inputs_table_name = "Obj_collision"
    _start = time.time()
    # input_vids = [8267, 4627, 6380, 9615, 8146, 1499, 9097, 6414, 429, 6454, 3137, 7861, 6102, 5668, 4736, 3424, 5841, 1298, 2509, 9081, 4692, 7756, 1136, 5463, 8802, 3539, 5246, 9162, 6859, 561, 2346, 1747, 5813, 5008, 9585, 9595, 7353, 6632, 6191, 2866, 215, 8144, 3412, 5208, 3389, 8351, 3601, 3020, 6172, 7979, 9682, 2091, 9389, 7727, 3525, 7347, 5383, 1769, 8293, 4712, 1269, 1079, 6256, 7376, 4794, 3963, 6464, 3018, 2199, 4052, 8696, 499, 3055, 6272, 6881, 7013, 8990, 7782, 3489, 3992, 1859, 3348, 1140, 258, 9061, 1330, 6533, 3087, 2560, 2658, 2624, 835, 7570, 9851, 243, 5249, 6000, 6250, 7935, 8053, 7237, 6917, 4751, 377, 1482, 5562, 181, 7117, 3166, 3541, 6159, 1546, 1757, 891, 458, 6931, 8462, 7955, 8985, 6499, 2395, 8932, 4220, 2630, 6525, 9886, 8295, 5877, 4460, 9020, 4847, 1534, 5859, 8551, 9582, 2733, 4506, 3365, 5728, 7755, 1491, 2746, 3978, 328, 9395, 3070, 7415, 3393, 9697, 7898, 9890, 7584, 6284, 7787, 602, 400, 9067, 4210, 4797, 7222, 2609, 4379, 3856, 6539, 9457, 9345, 3114, 9975, 1892, 9036, 3343, 7625, 4258, 4260, 1209, 3051, 8801, 9013, 3709, 7837, 1694, 9738, 8879, 3350, 4733, 7663, 2122, 6426, 3581, 4317, 6944, 5665, 9712, 8171, 6507, 4117, 5833, 9085, 5330, 8475, 3826, 1258, 8318, 5891, 977, 6034, 8979, 9968, 6216, 3733, 4027, 6872, 1308, 5336, 5285, 8442, 6811, 738, 1366, 7340, 4878, 9290, 8104, 3684, 9059, 7076, 2867, 279, 9846, 1445, 5247, 9989, 331, 7198, 2311, 6486, 4604, 745, 705, 6044, 1741, 3038, 3632, 3579, 326, 1933, 9644, 17, 7341, 9788, 6445, 9565, 6862, 3566, 4456, 6879, 1525, 2277, 2937, 1144, 1044, 4966, 1582, 6393, 8033, 3458, 7497, 4853, 4459, 1485, 7030, 8084, 9602, 8130, 8757, 941, 5007, 2645, 3235, 4187, 7478, 677, 9791, 2066, 3724, 2448, 8324, 5695, 6492, 7454, 3551, 2836, 5483, 2807, 9279, 9709, 3112, 7269, 1199, 6787, 8120, 6738, 9876, 5820, 4857, 7004, 3718, 2074, 2479, 8772, 3468, 6490, 2852, 3497, 4025, 9659, 1909, 8997, 5438, 1556, 542, 5169, 5432, 1178, 572, 2706, 8950, 563, 317, 8115, 2383, 2297, 7332, 957, 6974, 4883, 7536, 7980, 6231, 9959, 7843, 2981, 5558, 4517, 9060, 888, 7645, 9993, 5795, 3614, 2648, 8035, 1536, 3290, 519, 8275, 2524, 7261, 9373, 958, 7370, 3334, 8683, 7443, 4154, 3445, 4619, 3934, 5277, 2553, 6473, 5219, 1508, 3369, 4881, 8611, 9910, 2212, 6453, 4059, 9380, 6513, 6603, 8492, 264, 9776, 7212, 5596, 274, 7842, 8968, 4205, 1573, 4643, 9528, 9626, 8586, 9728, 1237, 8647, 4896, 7425, 8523, 8976, 85, 2145, 6802, 9875, 1197, 7385, 7034, 3582, 6934, 2764, 9619, 5799, 3183, 7020, 1428, 3201, 7590, 1569, 2787, 5147, 8122, 1544, 6888, 3320, 6521, 1528, 3024, 5164, 5727, 6973, 8913, 1532, 5639, 4854, 8898, 878, 1344, 6077, 2629, 5275, 5440, 8070, 6543, 7798, 9663, 2038, 1215, 3016, 7044, 3238, 357, 7922, 9530, 2235, 6948, 9145, 9991, 1117, 6101, 6605, 6278, 1805, 7604, 7730, 9330, 21, 4126, 9952, 4065, 5409, 5361, 8571, 8396, 1681, 9763, 8030, 9477, 1861, 8356, 2315, 6085, 4920, 4271, 6571, 8688, 8960, 9444, 3414, 9295, 5115, 4801, 6384, 9296, 6756, 1516, 5374, 3395, 5822, 5070, 795, 9783, 8235, 5551, 7401, 6057, 839, 3030, 5375, 5575, 529, 6218, 3880, 5973, 9553, 9549, 4816, 1686, 6722, 9071, 5818, 1553, 6388, 2777, 7507, 2220, 6053, 6179, 9073, 5800, 5995, 4140, 9984, 190, 2663, 8326, 8415, 2310, 4203, 244, 2983, 762, 748, 5242, 2400, 463, 7042, 4158, 936, 3803, 598, 8767, 3292, 7984, 9231, 6096, 7412, 3366, 3850, 9809, 3267, 3544, 4413, 5725, 4954, 5500, 4434, 7824, 8073, 293, 3925, 437, 7664, 8059, 3976, 2820, 2697, 8323, 1124, 8764, 4000, 8426, 4754, 8597, 9121, 8565, 6667, 9234, 487, 6377, 8239, 8895, 7754, 505, 5583, 8959, 9654, 1663, 9483, 9157, 1664, 8684, 1023, 9572, 4216, 7207, 9171, 4680, 9417, 7951, 4239, 7422, 7230, 2141, 5940, 7213, 9372, 6877, 4784, 3161, 8629, 1794, 7823, 7301, 6594, 58, 8209, 5043, 9002, 5055, 7543, 7384, 6364, 4946, 3984, 7583, 7807, 3929, 6583, 8265, 2800, 2174, 9492, 1013, 1476, 4948, 6662, 3250, 1066, 3143, 1198, 7323, 2813, 9949, 7902, 3455, 9386, 3953, 1481, 7759, 9842, 1552, 2855, 2923, 4406, 3176, 5108, 30, 1149, 6534, 6848, 1728, 5585, 2043, 7243, 2399, 8472, 2979, 5369, 5506, 12, 3616, 2042, 492, 5099, 9282, 1375, 8705, 420, 3454, 6167, 955, 2338, 4294, 1419, 7858, 8349, 5855, 5127, 1588, 7693, 5232, 5885, 8925, 5510, 9982, 3164, 2100, 4898, 1858, 3973, 4093, 6797, 3202, 8389, 5265, 5198, 604, 2204, 4143, 7864, 8827, 5811, 7676, 7247, 4195, 297, 7674, 4940, 9526, 7854, 8041, 5290, 6850, 786, 7438, 8467, 8933, 1059, 8849, 6292, 4484, 8796, 1566, 3032, 5, 6285, 2687, 1186, 1541, 3450, 7375, 546, 4122, 3397, 628, 6916, 4485, 6541, 6701, 8044, 3986, 8532, 3198, 7636, 5385, 3585, 6427, 2052, 267, 6213, 361, 5653, 5731, 2885, 1981, 2353, 4298, 5555, 1820, 2751, 8786, 9507, 2252, 9070, 6404, 767, 2814, 2738, 7551, 1889, 7374, 1731, 5803, 792, 6276, 7268, 1157, 2960, 7280, 277, 1522, 8874, 5662, 2466, 9150, 7349, 4613, 2587, 315, 9696, 4686, 2933, 6681, 1430, 7190, 5571, 4734, 778, 3133, 4618, 5630, 1333, 8736, 1132, 5414, 5472, 1868, 6847, 1404, 5347, 7264, 1916, 8738, 7159, 5032, 7316, 3377, 7550, 1341, 8618, 8674, 8107, 3437, 3842, 3039, 9786, 4181, 3538, 9021, 5882, 1653, 5245, 872, 883, 7894, 7598, 5612, 6730, 3635, 8693, 3643, 2554, 2376, 2464, 7821, 7869, 1086, 2631, 6148, 2989, 6823, 6822, 3743, 8773, 2718, 8896, 734, 8466, 1165, 3077, 9432, 550, 7931, 7905, 6563, 1156, 7472, 1811, 1997, 4537, 6120, 5857, 6336, 4678, 2130, 7614, 8993, 6344, 4039, 203, 2525, 1863, 966, 3229, 2262, 2505, 1414, 4886, 1060, 739, 7468, 2947, 148, 4580, 2135, 391, 8303, 9301, 2819, 1604, 5720, 8481, 7889, 1236, 5441, 5143, 4933, 4486, 9625, 9760, 6695, 4471, 8799, 8402, 3418, 3950, 8732, 4576, 8029, 8256, 7308, 4302, 4715, 3731, 40, 3852, 5982, 5128, 4787, 7631, 4055, 4579, 4060, 9517, 6765, 6581, 7724, 8057, 419, 7781, 7397, 4474, 5224, 8255, 6843, 6417, 5284, 9165, 7129, 6246, 4611, 5479, 8430, 6194, 7137, 7097, 2305, 1765, 9344, 4725, 9684, 10, 7560, 5991, 3404, 4964, 2476, 9948, 4190, 3848, 447, 1321, 8564, 821, 8319, 566, 5777, 6584, 683, 7364, 8234, 8260, 7820, 3726, 6966, 9209, 2799, 1129, 8355, 9560, 7486, 8643, 3787, 2494, 7439, 7713, 2057, 396, 6456, 6821, 1995, 3716, 4002, 9608, 5400, 3003, 6742, 5497, 8386, 1202, 6707, 6909, 9274, 3499, 7411, 2752, 2071, 6081, 7063, 4008, 4097, 4162, 6735, 56, 624, 8052, 3798, 3642, 4336, 7113, 138, 3293, 3859, 4625, 7524, 5177, 235, 1779, 1242, 3426, 8298, 5069, 3975, 9855, 3628, 3316, 1693, 7494, 650, 4445, 2236, 679, 7286, 8998, 9822, 8542, 6501, 2891, 8158, 6405, 4925, 8320, 4646, 8906, 6119, 4592, 7424, 9239, 9409, 8346, 8956, 5786, 8497, 8131, 5905, 1252, 5397, 6838, 8547, 7144, 7708, 9269, 8145, 2523, 7271, 2853, 8994, 4938, 6018, 1662, 6268, 1125, 6255, 7254, 1943, 9700, 605, 9053, 2702, 8066, 3725, 8677, 2963, 3252, 5966, 6578, 1239, 7959, 7704, 7309, 8819, 2856, 1915, 7224, 9799, 6598, 2358, 8263, 4453, 9662, 5689, 1224, 3417, 7697, 7526, 2531, 5868, 4806, 6794, 6234, 4175, 9913, 5927, 204, 9001, 232, 246, 8461, 5930, 5124, 1172, 5512, 1976, 5223, 6110, 758, 4107, 9114, 5427, 2801, 2997, 2978, 72, 6801, 7949, 9573, 9928, 3633, 8880, 7069, 8491, 8479, 3536, 7632, 3175, 2365, 4350, 9323, 5306, 5755, 1251, 6226, 4793, 7829, 8820, 3047, 7899, 6429, 5957, 5522, 4061, 6566, 1123, 6479, 6827, 4331, 4851, 6836, 5871, 9430, 4997, 5436, 4261, 2606, 2046, 6497, 8741, 8480, 1802, 6199, 4667, 496, 160, 5732, 3243, 8395, 5627, 840, 226, 1235, 9789, 398, 3111, 7095, 9940, 5288, 9131, 9675, 3220, 1090, 9495, 1590, 8992, 3501, 465, 5867, 1077, 8409, 3782, 4279, 7210, 4134, 7011, 9287, 2893, 8804, 1674, 6585, 1118, 9158, 1540, 8813, 4157, 4425, 8357, 2369, 6590, 1145, 8987, 5587, 3378, 6700, 9772, 7408, 421, 684, 5474, 8078, 5717, 1840, 8458, 9306, 8695, 9208, 8176, 5184, 4799, 1471, 5161, 3524, 4872, 9262, 3141, 436, 8154, 6472, 3938, 9438, 3515, 4930, 9091, 3678, 4212, 6759, 6222, 8471, 959, 1791, 7314, 2623, 7723, 2324, 7502, 5403, 9309, 5299, 6642, 7407, 2603, 9827, 6750, 8754, 3720, 5988, 6055, 3531, 3634, 1061, 8541, 1677, 1204, 3500, 9489, 44, 2500, 183, 2549, 2668, 1723, 5276, 37, 6048, 8929, 9599, 8875, 5371, 655, 737, 2322, 1661, 7749, 9174, 8001, 9562, 4651, 943, 1982, 3144, 4775, 5735, 9610, 4573, 4759, 3600, 5389, 1658, 8872, 5606, 9643, 406, 7771, 3385, 7270, 8590, 210, 5804, 4085, 9402, 3816, 3351, 9756, 4477, 5022, 8549, 6088, 654, 6788, 6702, 6762, 5207, 1726, 6544, 5282, 7640, 5797, 8007, 2397, 6063, 8983, 4410, 167, 8909, 6895, 3213, 6220, 8981, 8644, 1807, 9830, 8524, 7313, 2437, 3471, 6993, 6540, 4292, 1975, 2008, 7593, 7553, 6493, 3756, 7580, 4590, 6087, 6356, 2731, 2649, 9775, 9027, 3332, 4609, 1505, 1116, 2274, 6419, 3212, 3165, 1073, 633, 8281, 1041, 1659, 8699, 5125, 7501, 3402, 7070, 1745, 3954, 5167, 8622, 6938, 376, 4149, 600, 6956, 1027, 424, 2737, 4963, 4704, 6589, 6902, 7925, 7418, 4828, 3908, 658, 3972, 9695, 4153, 161, 259, 6296, 3857, 4417, 9634, 3747, 527, 5296, 4762, 1621, 8940, 455, 2514, 5936, 2521, 3572, 3610, 2300, 2903, 9921, 1914, 7453, 2348, 6407, 5978, 1754, 2540, 4009, 4014, 6436, 5342, 9436, 3422, 3495, 5178, 4222, 578, 1150, 143, 7404, 503, 1047, 5392, 7856, 4211, 213, 1876, 7696, 1143, 3005, 4913, 8223, 1974, 254, 2189, 5201, 1133, 9349, 8617, 1766, 9217, 2229, 6197, 3861, 4068, 3258, 3002, 9841, 6023, 552, 5131, 134, 4761, 4030, 9127, 7747, 3337, 7460, 1138, 9426, 1062, 7087, 9184, 5580, 3960, 7872, 3732, 6293, 2157, 5303, 7721, 6679, 2905, 1039, 2968, 2492, 7715, 8955, 6612, 8637, 9434, 9360, 9285, 5620, 3493, 6721, 2932, 8954, 1854, 2041, 2149, 8886, 4081, 8127, 3587, 1925, 5863, 4548, 8313, 2158, 9981, 86, 3354, 8760, 4601, 2430, 2463, 7835, 1352, 9268, 186, 6261, 4673, 754, 5012, 6184, 8339, 2940, 3097, 7926, 8845, 3403, 1385, 2592, 1938, 475, 9134, 5130, 5518, 1592, 2719, 3522, 5158, 6958, 3855, 570, 7164, 2154, 5058, 7882, 1281, 8645, 2593, 8249, 3555, 743, 9735, 8832, 238, 1535, 4110, 402, 4353, 4094, 7893, 3882, 7790, 3239, 7793, 3205, 9310, 9973, 8614, 5040, 8359, 6016, 7834, 5324, 3677, 8939, 5325, 1019, 2513, 26, 9508, 5435, 8170, 1559, 69, 2194, 6491, 6355, 7505, 5508, 5295, 3940, 4026, 7325, 5654, 7027, 2580, 9951, 1798, 4074, 8878, 6458, 8423, 9187, 8620, 4396, 6025, 697, 4101, 603, 2292, 8708, 6712, 2374, 5218, 9506, 6275, 3735, 1179, 8005, 4637, 6203, 6395, 1721, 1979, 7358, 5607, 5006, 1233, 6714, 5492, 7324, 4864, 8332, 7449, 7147, 9623, 1821, 3163, 1642, 6591, 6302, 2726, 5588, 2485, 4096, 9302, 9857, 607, 7707, 6547, 169, 4360, 8328, 5498, 4986, 3339, 8529, 1652, 287, 6028, 7249, 3744, 1894, 7386, 7122, 394, 3460, 6294, 935, 3946, 3194, 1071, 2774, 5370, 1955, 5525, 3496, 6975, 7082, 1812, 5311, 2791, 4987, 3044, 7717, 1685, 5134, 1398, 338, 1299, 5217, 1777, 5989, 3188, 7797, 296, 6134, 5669, 5244, 1068, 7630, 1001, 4315, 3173, 9927, 9933, 9852, 9491, 5713, 4075, 379, 7292, 2063, 3998, 7295, 7008, 4855, 4401, 9771, 8394, 7098, 3714, 8852, 2148, 8625, 1102, 9095, 2599, 7567, 693, 5332, 7783, 2612, 8268, 1474, 4021, 6747, 6227, 2116, 2571, 7103, 4374, 5100, 9205, 6611, 4577, 6854, 6979, 3739, 6882, 8011, 4771, 4385, 1710, 9859, 7350, 2816, 2093, 7000, 1261, 543, 2266, 858, 4616, 3876, 5552, 7303, 2684, 9535, 6752, 5541, 7112, 9281, 6865, 5166, 7927, 6266, 7633, 8697, 5550, 5567, 3959, 9168, 9028, 9934, 3835, 1830, 827, 5368, 2015, 2326, 8882, 885, 7555, 5944, 318, 7778, 2428, 6019, 3939, 5395, 5544, 843, 4404, 8237, 4090, 3514, 8831, 6634, 6656, 8076, 4338, 9102, 3279, 9529, 3717, 7712, 5960, 2762, 8377, 1992, 2936, 9971, 2302, 9901, 9080, 7886, 1465, 198, 7644, 8038, 8798, 5431, 6455, 2689, 8405, 102, 6768, 2833, 1063, 8727, 3679, 1162, 3342, 6318, 5197, 7709, 1789, 7977, 9443, 7686, 5109, 4257, 6551, 1836, 8682, 4681, 6163, 6423, 5466, 8455, 6723, 4241, 4766, 7360, 5697, 2681, 5889, 3429, 4327, 3981, 5536, 6980, 3381, 1907, 1650, 8296, 8594, 9918, 9406, 7606, 2226, 8587, 942, 2030, 179, 8945, 239, 3278, 3965, 6150, 6204, 417, 1810, 5430, 9869, 27, 2588, 7983, 9397, 6518, 5505, 6030, 3333, 5794, 7620, 9843, 3135, 1219, 9297, 6460, 3751, 9464, 5450, 4636, 6831, 4907, 4123, 592, 5602, 760, 2278, 6280, 6281, 6791, 5802, 8765, 6795, 7659, 2162, 9818, 8036, 8651, 4049, 4300, 8701, 8619, 7766, 5009, 3606, 5764, 7736, 448, 5749, 9084, 775, 2242, 6654, 1103, 9862, 6641, 5861, 2090, 3771, 9099, 8936, 7558, 8510, 3786, 8915, 2761, 9377, 2822, 1911, 8536, 783, 6573, 6659, 6482, 3035, 4672, 8215, 9744, 4044, 7596, 2874, 8744, 6538, 7428, 1690, 9146, 1380, 3652, 7764, 8478, 9066, 874, 5300, 2068, 9745, 868, 9705, 5538, 118, 7154, 1510, 8885, 2327, 3729, 1120, 6026, 3317, 4347, 912, 9196, 5686, 7379, 93, 3795, 192, 7679, 9839, 3722, 6252, 5116, 8818, 5715, 1208, 4242, 805, 262, 49, 6782, 7688, 4237, 9648, 7377, 3181, 7928, 3436, 1787, 2534, 4182, 9551, 493, 6036, 1441, 5186, 8370, 6936, 8558, 2240, 4659, 8810, 8636, 4475, 5044, 4781, 7508, 3532, 1567, 1092, 6505, 7359, 3474, 1788, 6876, 3093, 4991, 2250, 2473, 7605, 647, 3646, 1312, 6282, 5002, 5866, 3673, 3905, 373, 6367, 4319, 6557, 7706, 2931, 3085, 2817, 3200, 456, 2329, 9396, 1780, 6223, 8187, 5976, 8054, 8638, 9437, 9264, 1640, 5259, 3822, 9678, 8775, 5281, 8737, 8797, 9211, 9681, 3068, 4824, 6346, 1153, 5348, 6009, 3699, 7993, 3775, 2955, 218, 8734, 5926, 5046, 967, 2529, 6826, 7488, 2094, 2610, 255, 4167, 8790, 9628, 782, 483, 5531, 2582, 2566, 6627, 635, 3502, 3824, 5809, 6918, 6174, 9331, 1695, 986, 3897, 7127, 5278, 6635, 7336, 2231, 5787, 7081, 8504, 1991, 4362, 5292, 7320, 6017, 9930, 2483, 2272, 220, 5434, 9351, 9692, 7236, 4936, 5511, 926, 6998, 443, 928, 3246, 8724, 5895, 3086, 869, 7458, 1187, 8809, 5018, 2576, 537, 7698, 2786, 2341, 314, 9770, 2672, 89, 1045, 4951, 5421, 4048, 4587, 1500, 1494, 9710, 2543, 9004, 1443, 8900, 8656, 3910, 6708, 7252, 3777, 9861, 1025, 905, 3737, 4867, 9415, 8220, 4839, 1770, 6967, 449, 841, 7870, 1206, 7267, 4974, 242, 2078, 3589, 6411, 2120, 2178, 6757, 7832, 3225, 7265, 4307, 859, 8821, 3027, 5487, 3431, 6267, 7918, 2851, 3352, 5111, 4833, 4982, 444, 6907, 3283, 5782, 1954, 9324, 7582, 5272, 9227, 4308, 4796, 4414, 1021, 8527, 8264, 8897, 9347, 6447, 3218, 9342, 7318, 1838, 9792, 8540, 5699, 5805, 5801, 6669, 5553, 9115, 4677, 4197, 2241, 4262, 2307, 2792, 4507, 3384, 8930, 9807, 9093, 2465, 214, 9733, 9375, 2922, 4387, 9312, 5366, 8292, 8911, 4730, 6869, 8856, 5135, 5485, 5035, 9265, 514, 7484, 3498, 3704, 5159, 9965, 6705, 8567, 4419, 5471, 2959, 1890, 8860, 1783, 5519, 8008, 1191, 9497, 4440, 6402, 7273, 1002, 3021, 8065, 3449, 340, 4724, 9343, 1024, 5682, 2809, 9251, 1809, 6645, 810, 4011, 6365, 252, 2812, 2103, 9164, 46, 1457, 5476, 6470, 2298, 2053, 9547, 9904, 4251, 2824, 9953, 4551, 7506, 4837, 9706, 5092, 4365, 1347, 1323, 3375, 7355, 1737, 5887, 3033, 6548, 8020, 5495, 1444, 6703, 7017, 9896, 574, 6682, 692, 5935, 8793, 5642, 7521, 8563, 6151, 2222, 8274, 324, 9012, 5024, 2115, 3075, 3071, 5229, 582, 9198, 7514, 2422, 2670, 4575, 7757, 1923, 2882, 3042, 9224, 9352, 4057, 6052, 599, 5448, 2969, 7891, 7805, 6430, 8570, 3330, 8499, 2879, 8090, 413, 8238, 3870, 3662, 412, 1016, 9350, 1356, 1898, 608, 674, 7155, 8191, 1291, 129, 2124, 9638, 1627, 8603, 2054, 272, 3315, 548, 2526, 2126, 291, 4674, 4135, 7322, 2698, 2420, 1619, 6994, 9798, 8748, 2892, 7139, 3629, 8539, 7476, 5719, 863, 6905, 4469, 2139, 2495, 8907, 5256, 7132, 292, 8407, 3745, 6382, 9950, 9484, 6353, 9236, 3590, 1377, 2938, 2291, 3591, 375, 5729, 9636, 2567, 5851, 1371, 524, 8317, 8304, 2225, 9647, 2643, 1776, 4947, 5647, 209, 6981, 3364, 9311, 4653, 2377, 2050, 1279, 8132, 8949, 441, 5560, 2009, 5770, 3763, 1223, 5273, 9008, 1940, 1326, 1842, 1903, 6804, 3569, 807, 6305, 2601, 2125, 8142, 3177, 7142, 9120, 1194, 8868, 9161, 9587, 3694, 1729, 9427, 3483, 9658, 8002, 815, 2127, 9288, 4179, 3754, 3121, 6485, 5503, 462, 9307, 6208, 2375, 2655, 4171, 569, 5979, 8014, 8214, 486, 8153, 4130, 5362, 5830, 6977, 1051, 5908, 979, 6803, 2691, 4151, 8240, 2834, 7503, 3207, 1551, 5010, 8689, 5692, 1367, 5137, 7388, 3523, 1989, 5569, 2766, 4318, 656, 8756, 9132, 3892, 7981, 6145, 9858, 769, 8494, 2747, 5648, 3013, 2136, 7094, 8624, 9983, 7944, 3812, 6230, 110, 6487, 8573, 2957, 2779, 3083, 5656, 5027, 8427, 547, 6290, 5020, 6245, 2408, 3259, 5152, 2536, 5205, 7180, 5334, 4945, 3096, 1560, 4274, 8605, 8676, 8, 5014, 4904, 7181, 5751, 3602, 8511, 4235, 2616, 7019, 8789, 3004, 1253, 7080, 6154, 9721, 2111, 3613, 8576, 2037, 4707, 4876, 3427, 5103, 7779, 8866, 6330, 1899, 7016, 2808, 6623, 60, 6343, 612, 3073, 545, 4172, 4265, 5089, 9155, 780, 1294, 1577, 988, 5279, 9439, 1134, 4037, 7635, 8226, 9299, 7232, 5230, 3845, 8190, 8899, 6372, 5775, 3328, 4355, 6992, 660, 2017, 6314, 1880, 2317, 5379, 9201, 6362, 2373, 6103, 3187, 4275, 4255, 3638, 1200, 9742, 9276, 662, 763, 8291, 980, 4225, 7700, 482, 2966, 3260, 8663, 2072, 4141, 6678, 4928, 7746, 727, 9329, 2060, 4243, 8971, 9670, 5428, 8207, 4004, 9589, 4981, 4325, 6666, 5896, 7602, 2638, 4063, 1201, 8338, 1424, 8628, 6132, 1185, 5304, 794, 1800, 1098, 5260, 5141, 8344, 9906, 7378, 3441, 9277, 3423, 1727, 2518, 4281, 4282, 5084, 2190, 3655, 2946, 8193, 1376, 5631, 2613, 6726, 8871, 9256, 4518, 3715, 9215, 2634, 751, 1771, 5468, 5532, 7326, 7001, 7162, 6291, 4295, 1610, 8731, 7607, 7929, 9583, 1576, 3100, 9181, 8666, 5200, 1715, 1865, 1987, 7991, 644, 3688, 227, 584, 276, 9092, 4720, 1402, 3409, 5862, 2699, 1711, 3253, 8112, 1315, 7682, 4017, 5945, 5469, 3023, 3222, 149, 1272, 3568, 8374, 2909, 5236, 2349, 3241, 3067, 9977, 5083, 613, 7588, 4267, 9795, 6483, 59, 6523, 4502, 1732, 814, 7532, 1, 268, 1831, 3274, 4818, 1159, 8725, 8782, 7239, 6406, 9800, 3962, 7710, 1412, 4669, 7312, 575, 6829, 6706, 1168, 471, 111, 4541, 3072, 5854, 5003, 7739, 9390, 7680, 1878, 9249, 2021, 3151, 2312, 666, 5338, 4545, 7671, 2927, 3089, 5456, 4860, 8600, 5640, 8408, 4430, 1463, 4512, 1946, 9954, 3983, 4871, 1874, 3804, 2012, 6963, 7850, 8098, 6152, 3487, 2171, 721, 9768, 8352, 4999, 4760, 8179, 2016, 9160, 1107, 2656, 3060, 8329, 4427, 7545, 7504, 2758, 2906, 9966, 9147, 1121, 6390, 9068, 5211, 7911, 90, 6162, 1297, 66, 5262, 2088, 9030, 7948, 1470, 6820, 5894, 9229, 5233, 816, 5412, 567, 7317, 7913, 7059, 3631, 2254, 1699, 4664, 4729, 7025, 6361, 1926, 2248, 2980, 8199, 8788, 2811, 7833, 3930, 1680, 3392, 3215, 6363, 7277, 6121, 5754, 1718, 2058, 9333, 8720, 4337, 5507, 3653, 4429, 451, 1896, 7838, 6892, 2263, 6699, 1307, 6401, 6084, 6631, 2070, 9533, 8227, 144, 236, 9924, 1163, 446, 9693, 9823, 9238, 8777, 2954, 2001, 8616, 3736, 3742, 205, 1114, 6209, 2092, 923, 2284, 8989, 4859, 5316, 7194, 6637, 6927, 9762, 5660, 2435, 6149, 8965, 2876, 8421, 6810, 7617, 6789, 8753, 427, 4516, 2917, 2363, 9408, 6233, 8136, 5328, 224, 5943, 1360, 1629, 7768, 5923, 9069, 5974, 3844, 3280, 8206, 1740, 5659, 2703, 7456, 2673, 6878, 7887, 2439, 8917, 4022, 944, 1542, 1670, 2219, 3780, 3432, 6937, 2503, 9086, 8947, 3046, 9600, 9247, 280, 4492, 555, 8999, 9152, 6265, 4524, 7839, 6833, 4248, 6079, 3459, 4719, 735, 6740, 7809, 8075, 7906, 2562, 7738, 4865, 7609, 4173, 1275, 4465, 8671, 1829, 8335, 7982, 4984, 6660, 5310, 3505, 3446, 8669, 5076, 1174, 6495, 5845, 2080, 2547, 3554, 1450, 9687, 6219, 8316, 9548, 2131, 8665, 3839, 8847, 3609, 621, 8203, 5189, 9289, 9278, 5203, 4095, 8591, 2849, 8986, 3521, 4666, 2255, 4357, 2789, 8091, 2026, 5821, 8580, 1768, 5380, 2559, 7090, 2965, 7608, 3372, 4656, 1285, 6999, 4849, 7796, 7728, 7481, 8698, 7447, 685, 2179, 2823, 3276, 9384, 9079, 8745, 3311, 804, 1014, 9945, 4340, 7363, 2095, 1749, 2458, 4563, 4731, 8037, 893, 7490, 2555, 583, 9185, 9332, 922, 6688, 1846, 8398, 6064, 766, 8865, 9672, 4088, 6410, 6415, 5225, 9985, 4443, 6351, 4219, 3915, 4596, 7685, 9193, 916, 3746, 7115, 385, 3126, 1490, 9118, 1108, 362, 3049, 8711, 9220, 920, 1555, 8022, 7, 5424, 1614, 1927, 5175, 6366, 3902, 2827, 9729, 9365, 9452, 8679, 8322, 717, 5677, 2846, 7794, 4888, 2436, 2459, 8345, 2715, 1368, 6433, 2200, 3158, 4264, 5759, 8186, 7762, 8931, 6307, 8126, 3935, 8063, 8493, 749, 5418, 7958, 9849, 5710, 3979, 6558, 9793, 7575, 1078, 4178, 6976, 9580, 4405, 4542, 5980, 4432, 5502, 6864, 7668, 8700, 3982, 659, 4660, 6648, 2516, 4723, 4029, 4268, 8087, 4689, 7189, 8404, 4384, 4844, 6720, 9246, 9967, 1692, 8459, 6626, 4939, 8384, 8094, 2611, 4927, 2356, 1738, 642, 5318, 6991, 5913, 3649, 7226, 2773, 4326, 6396, 4786, 8935, 7051, 4584, 8061, 4312, 120, 3527, 3052, 8483, 8657, 2522, 9864, 8103, 3415, 7479, 3057, 1762, 6125, 7752, 5609, 7233, 8530, 886, 2499, 855, 1832, 9113, 2904, 6704, 9024, 1105, 270, 24, 6749, 8046, 9746, 2573, 7520, 2398, 6301, 9880, 8195, 8414, 8560, 2336, 2507, 3545, 216, 6904, 2480, 5274, 2282, 8884, 5193, 4064, 2039, 1538, 3809, 3550, 8673, 1408, 2795, 3006, 6674, 9130, 2538, 4809, 3767, 7702, 5772, 8569, 2106, 6190, 6921, 6564, 591, 9566, 4045, 8129, 4694, 4391, 8200, 7121, 5718, 7156, 3889, 2362, 1193, 7145, 6996, 5234, 3186, 6033, 3820, 8830, 7461, 3025, 6496, 1080, 8297, 1284, 5672, 8373, 4206, 8837, 9727, 6816, 6158, 5352, 1397, 6755, 9126, 2675, 9673, 2748, 9556, 6602, 6834, 1824, 700, 1736, 3791, 6041, 6672, 925, 1296, 4431, 8019, 2270, 2288, 5454, 5661, 832, 5326, 2707, 7495, 1069, 3608, 1127, 8118, 2739, 1358, 731, 5646, 9487, 7765, 5766, 8707, 5611, 9837, 597, 1839, 7792, 9754, 5624, 7085, 5650, 5377, 5584, 8450, 7496, 7240, 5910, 968, 3949, 3549, 7221, 9364, 1469, 614, 9116, 9031, 5621, 9357, 6080, 9064, 5072, 1611, 9520, 2295, 4259, 4593, 7091, 3028, 9462, 7851, 2108, 1389, 6940, 2279, 8217, 9474, 233, 1173, 1514, 3996, 5582, 9995, 6970, 1719, 4520, 7068, 9563, 3651, 8612, 2372, 4290, 9691, 5363, 1919, 7152, 6734, 9831, 3989, 9554, 173, 513, 8189, 2234, 5566, 1523, 4570, 5881, 588, 7788, 4829, 8259, 7660, 1651, 9339, 2605, 8972, 6815, 1248, 3174, 4795, 416, 136, 5015, 2109, 9478, 303, 8974, 661, 8258, 5091, 3344, 2575, 9197, 5829, 5736, 5950, 8717, 94, 5542, 849, 1245, 1518, 8602, 8277, 7405, 5261, 5390, 108, 9346, 4792, 8486, 3190, 915, 3099, 2872, 8454, 5790, 1226, 4647, 3644, 3400, 7945, 5984, 9882, 435, 829, 8157, 6873, 7600, 2859, 969, 2744, 4675, 4322, 3888, 7561, 5228, 5445, 240, 2924, 3288, 4377, 6358, 1064, 2831, 8288, 2182, 4168, 9087, 1309, 4246, 5831, 3788, 4985, 4109, 9627, 1956, 7387, 8525, 1717, 3115, 6932, 4726, 4509, 4641, 6990, 4467, 8350, 7199, 8632, 474, 7921, 7119, 9903, 7581, 5085, 2941, 1407, 3647, 643, 3712, 5340, 5563, 8401, 1823, 1011, 2970, 2948, 5634, 9003, 8507, 3142, 364, 38, 5785, 234, 2596, 6953, 9169, 8246, 5354, 2296, 9258, 2404, 7010, 8301, 2664, 6082, 2185, 6900, 9942, 4957, 4105, 1860, 9651, 831, 121, 3765, 6633, 1493, 9469, 7705, 5351, 723, 812, 6095, 9180, 864, 1521, 894, 2769, 4016, 3346, 8375, 8598, 2469, 3977, 7969, 298, 906, 638, 639, 5723, 6200, 9925, 6189, 6883, 9334, 7586, 7623, 3000, 549, 6207, 753, 2901, 8169, 5402, 7626, 5675, 1570, 5372, 2561, 7075, 6658, 1327, 1329, 9089, 5378, 7776, 8151, 9191, 6545, 9794, 3697, 5080, 341, 4191, 3341, 6737, 3560, 2967, 2382, 8894, 6049, 8252, 5651, 1513, 4046, 828, 4645, 2671, 3480, 302, 9216, 853, 6891, 5215, 6549, 1325, 4372, 7300, 9422, 6817, 9555, 5578, 6139, 6244, 5019, 7533, 4363, 520, 7499, 2676, 1724, 9404, 9739, 7537, 1020, 6517, 8662, 9336, 9946, 6884, 3196, 6778, 8692, 4087, 2921, 9943, 5317, 9294, 8397, 2996, 3800, 785, 2581, 716, 1447, 3952, 686, 45, 5590, 6577, 2620, 4309, 7811, 3576, 1141, 6906, 3261, 776, 1773, 9511, 9804, 1169, 2129, 7726, 8517, 1562, 4083, 9355, 68, 8406, 4291, 4738, 1871, 9769, 6987, 6777, 3376, 3827, 9455, 5082, 2142, 4944, 3639, 4956, 8221, 4769, 9544, 9039, 9052, 889, 9998, 8382, 7299, 7367, 1462, 7896, 5016, 3388, 8664, 4743, 6643, 81, 6849, 6644, 7919, 3961, 1782, 3894, 3475, 8004, 6123, 590, 6137, 2206, 4112, 3886, 8543, 5678, 8163, 1099, 2147, 6952, 8447, 3465, 4316, 6841, 2331, 9907, 2022, 6013, 4906, 7974, 1937, 7061, 7278, 5747, 675, 4344, 6715, 5308, 4728, 1608, 2451, 9112, 3836, 3773, 8460, 8881, 5608, 9393, 3937, 4184, 2002, 4323, 9714, 3776, 7932, 9660, 8713, 636, 6684, 5528, 1755, 8015, 2146, 5883, 6675, 6357, 3228, 1648, 8251, 8379, 6273, 9510, 8168, 2840, 2881, 9702, 1148, 8650, 5142, 1758, 3993, 4200, 4628, 9871, 8378, 557, 9606, 9076, 9433, 3575, 5638, 8123, 5873, 8952, 2710, 3286, 9698, 7005, 5540, 2290, 8364, 1238, 497, 165, 4498, 9094, 7195, 7179, 746, 6650, 2918, 1654, 146, 6887, 854, 1691, 1158, 4798, 8416, 7253, 7333, 8250, 1055, 7150, 4120, 6986, 1502, 8469, 6065, 8188, 5416, 736, 453, 9140, 2064, 6175, 4595, 6599, 4718, 9490, 29, 6845, 5961, 7675, 3805, 1792, 6901, 1170, 2440, 6005, 2198, 3371, 3622, 914, 384, 3048, 9156, 7241, 3466, 347, 4924, 2340, 5213, 6340, 3162, 1716, 84, 2325, 6439, 5237, 9514, 1084, 826, 9908, 4273, 1029, 7395, 4403, 1010, 2782, 5026, 634, 6225, 1888, 3533, 57, 3477, 2765, 1583, 5847, 5616, 9335, 6319, 9363, 3338, 6972, 6232, 6595, 5255, 1703, 8816, 9272, 4335, 8116, 4186, 2098, 6378, 5708, 6806, 6960, 8887, 949, 8505, 2110, 2360, 9671, 8918, 450, 2005, 7445, 5953, 2826, 8006, 1558, 5848, 8128, 6824, 1434, 7933, 8841, 7475, 6673, 6462, 8833, 3947, 7211, 5344, 899, 1216, 1101, 6312, 3248, 7930, 8836, 9429, 6885, 1458, 3015, 8742, 1841, 8093, 9183, 2528, 1104, 4740, 3396, 1579, 856, 401, 223, 2621, 5875, 3299, 5865, 393, 3224, 8584, 5398, 7389, 2438, 8286, 8627, 9041, 3592, 3329, 4170, 5346, 7279, 8843, 6785, 866, 8024, 320, 2877, 3184, 3681, 1561, 8247, 8140, 3574, 1342, 9757, 9633, 3136, 1467, 4980, 7067, 4892, 3530, 4240, 3705, 8751, 3150, 9576, 729, 1594, 4342, 8658, 6489, 7569, 842, 615, 4700, 2558, 8219, 2024, 9796, 6133, 3119, 9655, 3710, 7816, 6376, 3247, 6743, 4114, 7968, 3830, 2992, 3262, 1637, 9685, 4194, 5139, 3146, 4538, 9428, 4224, 9107, 4612, 9063, 3059, 5897, 386, 5179, 2366, 2004, 8702, 5745, 9538, 6206, 8245, 9494, 2047, 1644, 2246, 9034, 4846, 4386, 3345, 8343, 4124, 5835, 2211, 1856, 8285, 1043, 7392, 5294, 8197, 6969, 790, 4513, 2107, 1278, 9784, 1537, 713, 2163, 1410, 9009, 4780, 4287, 7800, 480, 1697, 9019, 96, 7868, 9022, 3421, 2303, 2939, 8327, 9447, 5595, 8058, 5105, 918, 7244, 4473, 2510, 4388, 7687, 4076, 8783, 7812, 802, 9889, 8910, 5739, 5345, 1957, 6304, 9518, 5666, 9011, 9243, 4072, 8817, 9899, 564, 2690, 2683, 3307, 1343, 7492, 8385, 5752, 5564, 7366, 1475, 7642, 7743, 1166, 8721, 6531, 8779, 4556, 11, 3507, 4840, 8230, 8300, 7801, 8588, 6576, 4163, 6945, 5331, 5521, 283, 6122, 3577, 2838, 6141, 756, 6860, 8942, 2011, 5104, 7173, 3781, 2186, 2351, 5322, 8668, 7200, 9378, 4399, 4013, 788, 7639, 6853, 2118, 9811, 3413, 3904, 3043, 568, 3078, 9522, 7100, 6561, 6105, 151, 6240, 7202, 7804, 3862, 2584, 5181, 1527, 9761, 6628, 383, 3645, 6574, 3145, 131, 8101, 9035, 7351, 2361, 9593, 4574, 3349, 2745, 4691, 7346, 5664, 6868, 4284, 5350, 5784, 7290, 7996, 4774, 1074, 1382, 7819, 9337, 7072, 9459, 2695, 4145, 2987, 589, 6, 3671, 8716, 1231, 6649, 2380, 4314, 1387, 6861, 2489, 2394, 358, 8243, 4690, 4324, 4056, 6502, 1547, 5013, 9206, 6431, 6477, 180, 6596, 170, 2446, 1497, 2930, 6398, 7556, 7939, 3147, 4620, 6971, 5663, 6727, 7006, 4519, 5473, 3318, 4921, 1643, 4160, 8780, 5501, 5337, 8204, 5122, 4354, 6333, 7216, 7737, 6352, 6959, 3877, 6532, 6027, 4128, 2735, 6615, 7151, 1189, 7272, 5687, 8333, 6965, 4092, 2794, 2732, 9666, 5810, 300, 4977, 9230, 1147, 1303, 9232, 9571, 8182, 9480, 7923, 7836, 4990, 9200, 7767, 9440, 4698, 9704, 5903, 7527, 9468, 3240, 651, 3095, 4922, 1440, 9592, 6625, 1340, 8728, 8229, 4976, 4747, 9845, 6983, 7848, 9974, 1862, 4969, 8312, 4919, 1317, 137, 7083, 485, 8064, 2138, 3182, 3410, 2973, 6325, 7662, 2964, 4034, 6592, 193, 4482, 793, 8261, 2563, 3294, 3304, 7146, 2496, 8496, 787, 1126, 9139, 3719, 4899, 1391, 9032, 6248, 8943, 2416, 1181, 773, 9190, 1501, 947, 5333, 5227, 8149, 764, 2711, 5846, 2210, 5618, 8248, 8609, 2952, 2405, 7423, 6744, 3179, 175, 6135, 325, 1361, 3091, 4789, 4931, 9558, 6664, 8372, 7666, 5054, 6345, 7511, 4624, 7874, 9632, 123, 4753, 725, 3657, 8452, 126, 565, 7064, 1932, 2597, 1928, 6807, 7093, 3511, 6692, 5239, 2956, 2520, 3159, 7105, 289, 9603, 7695, 9751, 9328, 3314, 998, 6766, 7999, 8218, 1875, 6035, 408, 2660, 1221, 9524, 2830, 3411, 9819, 8526, 8863, 8889, 1814, 4035, 8419, 4748, 4129, 5086, 6099, 5783, 9391, 3630, 701, 7880, 6498, 1442, 9456, 5967, 8201, 3306, 1113, 7573, 1446, 8025, 7753, 3757, 4146, 5335, 6258, 3012, 7489, 498, 8607, 9137, 3674, 4983, 3680, 9767, 4631, 2364, 415, 7250, 9885, 1439, 9539, 2641, 9828, 9778, 1131, 8574, 5117, 4481, 1900, 7138, 7033, 1082, 1451, 5253, 8763, 2832, 1050, 4529, 4373, 7166, 9707, 3390, 4288, 6444, 9785, 9461, 1887, 4779, 2238, 8202, 5460, 9569, 4802, 472, 3510, 335, 8310, 1843, 587, 4861, 2928, 6185, 1155, 4756, 6786, 1857, 4418, 301, 6818, 5488, 6300, 2704, 4333, 9642, 2591, 8581, 342, 7859, 9475, 7599, 6165, 7654, 4066, 5097, 1195, 9922, 47, 197, 6586, 7041, 8017, 2470, 7493, 1586, 5557, 8314, 3991, 8476, 1668, 8919, 4249, 3789, 2842, 3394, 4115, 5693, 5112, 7875, 8646, 8012, 3457, 1464, 5515, 1572, 9361, 7281, 7220, 5600, 1429, 4202, 9388, 676, 8519, 2020, 6328, 1314, 7637, 1468, 7917, 536, 8468, 1122, 3708, 9825, 2196, 3792, 4441, 8577, 2453, 6212, 3701, 808, 6677, 7826, 1646, 7742, 671, 9537, 3721, 5033, 7910, 8559, 5151, 8729, 7964, 195, 6259, 5433, 9471, 2991, 1967, 8596, 5744, 8340, 7356, 4047, 5051, 5415, 2150, 8410, 4863, 4204, 5153, 4882, 7562, 4407, 2, 4466, 4594, 718, 2289, 4361, 8417, 9609, 9892, 3817, 4511, 4375, 8631, 1767, 1106, 7862, 7149, 2644, 8138, 9153, 9816, 3558, 39, 7780, 4367, 9509, 5129, 5721, 374, 9990, 681, 8307, 640, 2899, 7750, 1431, 8284, 6298, 428, 2857, 1906, 7203, 7174, 5667, 1725, 6277, 902, 1633, 2155, 8272, 2639, 2482, 8537, 9410, 2541, 9065, 6180, 1963, 2447, 5615, 9887, 5537, 4079, 8231, 5838, 7245, 9527, 3490, 158, 7857, 5679, 4812, 1128, 8125, 4496, 295, 8648, 9082, 6014, 1466, 9252, 9096, 8984, 5674, 1423, 999, 3854, 8085, 3463, 2491, 3139, 309, 3621, 316, 744, 8709, 7619, 1256, 7416, 2686, 1965, 3885, 6161, 4462, 4452, 359, 5349, 887, 9978, 5484, 4621, 4525, 1675, 6935, 7624, 9840, 4351, 2478, 6524, 5005, 4100, 4426, 132, 9362, 8175, 9014, 5767, 6409, 5586, 5704, 7789, 2337, 6136, 371, 7302, 140, 4504, 366, 7878, 372, 1929, 3506, 7844, 7814, 3358, 3660, 4583, 7512, 3494, 6962, 397, 1912, 2168, 3783, 4356, 9917, 2728, 9442, 7998, 2544, 6946, 2724, 2391, 7371, 8321, 4591, 1384, 4086, 672, 6559, 3578, 1334, 965, 5320, 2285, 3336, 2902, 452, 5589, 352, 4567, 8137, 898, 1270, 5898, 4776, 3481, 6108, 9955, 2497, 2023, 313, 1952, 2568, 6113, 8858, 5601, 9481, 6805, 3031, 7135, 4979, 6024, 8096, 1511, 4138, 5404, 1575, 7118, 5120, 5405, 601, 113, 4826, 4024, 470, 5073, 4657, 5061, 8003, 8495, 7343, 5173, 1018, 7647, 3806, 9570, 3464, 768, 5449, 6693, 6957, 9853, 3906, 6908, 1641, 3899, 917, 164, 4104, 7022, 8758, 8815, 1942, 9077, 6494, 5157, 1706, 7321, 1263, 5455, 8846, 1818, 5053, 994, 5852, 8353, 3058, 3428, 7611, 4772, 8473, 1374, 8762, 1348, 540, 1363, 9598, 9300, 7544, 8034, 7191, 5034, 3831, 5425, 2688, 571, 6413, 1970, 6006, 1339, 5146, 8124, 6856, 4996, 6066, 6941, 8010, 4166, 8520, 504, 7050, 345, 2985, 8023, 8117, 5619, 48, 7946, 3890, 9078, 4912, 1996, 2137, 6792, 3974, 248, 2778, 221, 511, 3997, 3287, 365, 1688, 5676, 8937, 3656, 1283, 9327, 5842, 9179, 3878, 5017, 4768, 5671, 3851, 1260, 6169, 7255, 3534, 1280, 1585, 9284, 4177, 5919, 4464, 6313, 3561, 5041, 5516, 433, 479, 3192, 9451, 6761, 8991, 1600, 392, 8633, 490, 7339, 67, 6697, 3916, 1872, 9226, 1167, 154, 8077, 7419, 6153, 8604, 9605, 3268, 2330, 2519, 7615, 6176, 837, 6331, 3942, 539, 8418, 7107, 7153, 871, 5287, 7559, 7218, 6636, 7634, 6452, 5314, 2468, 848, 3807, 6168, 8490, 3664, 4476, 2754, 5254, 4412, 6550, 3571, 8970, 5753, 6897, 2896, 5126, 1288, 8749, 4832, 3583, 3636, 4534, 1665, 7530, 1096, 8546, 1994, 8309, 5565, 5220, 1827, 5869, 6527, 4803, 4041, 9777, 1998, 5444, 1707, 8538, 8412, 9616, 9856, 1825, 4415, 4755, 9701, 1418, 1708, 5437, 6819, 3451, 1666, 2589, 6480, 8513, 4213, 8528, 9366, 860, 7840, 311, 1774, 1390, 7399, 6503, 7785, 8086, 4328, 8211, 1951, 4785, 1048, 5998, 378, 5110, 5490, 8982, 3668, 5475, 953, 8290, 41, 3957, 9054, 260, 77, 2662, 5994, 8805, 8562, 698, 7903, 1799, 3801, 3686, 1415, 9781, 2392, 8441, 2032, 2357, 5407, 201, 981, 5087, 755, 6166, 4805, 7828, 7601, 2760, 9545, 3204, 9614, 6930, 6474, 9033, 7691, 2977, 6140, 7052, 1882, 9803, 3707, 1985, 3868, 2433, 1893, 6146, 3217, 4392, 5916, 4687, 9732, 23, 8305, 9505, 3749, 4188, 6397, 8080, 3654, 2889, 5534, 712, 6772, 4901, 351, 3543, 3102, 1425, 3435, 7518, 6536, 7263, 6919, 896, 2293, 250, 7467, 9125, 3682, 7847, 1886, 7722, 8921, 4972, 1405, 4113, 3310, 3955, 6769, 963, 5832, 8518, 9496, 6306, 323, 5986, 670, 2661, 6237, 1833, 5758, 3130, 4973, 4442, 7192, 4461, 3659, 1370, 2415, 4320, 1949, 106, 6253, 9586, 995, 3469, 8276, 1962, 387, 4578, 2175, 4012, 1135, 2393, 2445, 306, 4435, 418, 8944, 4283, 2757, 6691, 8439, 7079, 5136, 5850, 5068, 5023, 4310, 6029, 6047, 2319, 6736, 3063, 6653, 938, 7334, 7009, 6961, 3834, 8445, 4827, 4558, 3823, 770, 3509, 1655, 2907, 50, 1960, 6078, 390, 7943, 9815, 911, 18, 6630, 1966, 7975, 7714, 7900, 9159, 4383, 5808, 7327, 3941, 9271, 9630, 7990, 1456, 1399, 9994, 2998, 211, 5156, 6350, 3832, 9167, 6001, 7257, 7183, 9748, 820, 7914, 2506, 2413, 4585, 8670, 3675, 8047, 7073, 619, 7786, 9018, 6205, 1625, 5499, 2156, 3076, 321, 973, 8743, 8216, 9601, 8685, 5597, 2201, 1934, 7997, 1337, 4783, 82, 8173, 7427, 4589, 6739, 4165, 9920, 2501, 7007, 5827, 2249, 6394, 7092, 2632, 9106, 8585, 5965, 1801, 157, 5992, 1222, 8568, 1936, 2209, 3327, 711, 7621, 876, 6335, 1250, 9055, 8451, 348, 63, 319, 8951, 8592, 695, 1081, 2647, 7677, 6617, 230, 9304, 7338, 4369, 2790, 5593, 1217, 3447, 7936, 9275, 9248, 5458, 473, 4701, 3895, 2780, 2565, 4436, 4530, 7670, 4856, 7065, 3084, 5168, 809, 9316, 707, 4758, 2143, 4069, 3557, 7209, 4180, 5706, 6984, 1196, 3764, 3029, 8923, 9143, 7256, 2618, 5613, 4421, 5738, 369, 2243, 9237, 5886, 9848, 8747, 1212, 7037, 9045, 5774, 4603, 9048, 551, 2172, 212, 4654, 6835, 3650, 3088, 442, 4522, 5327, 1365, 7354, 5439, 2572, 7563, 6420, 3504, 332, 9110, 6985, 288, 500, 4965, 7773, 312, 3191, 8072, 1563, 9407, 8639, 2803, 5741, 2423, 489, 9133, 1422, 6646, 2218, 9617, 8139, 2320, 1837, 3833, 3700, 8433, 8100, 4447, 9734, 2102, 5906, 4875, 617, 9100, 6893, 4941, 7178, 3567, 99, 2805, 1730, 4395, 3665, 6600, 2678, 4493, 1520, 8694, 6535, 9690, 8996, 4382, 508, 2532, 2865, 5258, 7305, 8027, 5807, 5123, 3819, 9382, 6326, 1761, 5559, 9596, 2443, 3387, 7440, 6263, 8453, 2763, 798, 8903, 1028, 257, 7446, 7865, 1545, 256, 5095, 6478, 4254, 6663, 2000, 7274, 2657, 2579, 2268, 2920, 1669, 1873, 6142, 6796, 4233, 8904, 3116, 5657, 710, 7474, 6725, 145, 1364, 5065, 3540, 177, 7638, 3512, 6003, 5313, 946, 8557, 9431, 6863, 861, 5257, 2665, 9970, 8465, 9806, 6202, 4649, 3406, 4989, 1075, 9891, 8888, 5576, 8400, 7576, 2177, 6799, 9611, 8946, 8026, 5176, 5779, 6287, 6753, 194, 1550, 9568, 9512, 7665, 4515, 8766, 7175, 4608, 2434, 2191, 7074, 7160, 5145, 5493, 4470, 6392, 6111, 2378, 5212, 528, 1803, 2878, 8253, 2590, 6400, 5526, 6463, 4071, 1351, 3479, 5789, 5297, 4547, 7294, 2594, 1301, 1146, 9559, 9649, 8850, 4174, 9280, 2880, 3094, 8678, 1529, 4749, 2900, 2101, 1913, 1797, 1834, 4108, 1160, 5684, 8978, 1310, 2694, 1203, 3869, 2871, 6942, 7912, 4111, 4155, 6058, 8165, 6842, 2818, 2637, 2797, 7406, 3214, 4869, 6334, 4192, 882, 8768, 1089, 875, 7134, 2914, 42, 645, 7125, 3309, 9850, 467, 8302, 3103, 7699, 7372, 1993, 3123, 7242, 3818, 8184, 7822, 7825, 9240, 6928, 3529, 3476, 2837, 114, 9182, 2755, 1908, 182, 6408, 285, 3770, 2339, 8653, 7509, 7658, 9425, 9960, 3208, 673, 4533, 5088, 5633, 3794, 9369, 4133, 8369, 2213, 880, 5477, 7535, 7167, 8533, 931, 806, 1453, 4521, 7769, 9072, 4697, 7003, 9394, 2082, 1354, 6228, 7430, 2772, 8806, 5690, 2257, 5343, 6242, 247, 2890, 191, 7552, 5222, 4841, 1917, 9467, 3995, 9017, 1760, 5443, 664, 370, 1427, 2839, 6116, 2548, 1772, 1042, 3010, 8926, 4565, 1228, 7791, 4566, 2696, 4193, 5489, 9235, 3491, 5067, 991, 4019, 2460, 5819, 6748, 3874, 596, 9341, 207, 1227, 3152, 4708, 71, 8228, 3296, 3255, 6050, 8769, 3926, 6093, 1935, 1290, 4716, 5899, 1835, 8279, 740, 6593, 8198, 2019, 3872, 2350, 9119, 7628, 4634, 7444, 4089, 719, 6780, 4508, 830, 2321, 5837, 2432, 1953, 8391, 2318, 263, 4599, 8105, 7136, 4811, 4227, 7437, 8710, 9005, 9266, 1230, 3367, 7330, 4929, 4132, 3814, 5798, 8503, 7258, 9814, 3785, 6074, 5685, 1554, 4131, 273, 4147, 5396, 862, 8109, 1606, 1753, 8336, 8306, 2743, 2264, 1100, 6235, 7421, 8715, 2151, 5844, 8659, 8509, 5993, 9532, 8556, 9387, 2014, 199, 2974, 1345, 7104, 4543, 8092, 1007, 8800, 7655, 7513, 7510, 8610, 5948, 7978, 877, 6021, 4668, 7459, 4116, 2627, 2850, 3045, 9725, 2806, 1484, 4568, 3713, 3922, 8770, 5360, 1152, 1631, 2410, 6784, 3007, 2729, 6070, 4176, 9385, 1973, 7287, 9552, 6339, 1984, 6567, 9900, 4137, 5761, 9597, 2056, 2119, 4868, 3323, 7831, 4582, 921, 4329, 5271, 7096, 6655, 3793, 3101, 3453, 797, 2354, 4127, 3866, 9142, 8746, 3203, 5912, 538, 2214, 5760, 4992, 3026, 5301, 6579, 3586, 2775, 834, 8435, 8583, 1094, 5494, 4252, 9575, 4297, 6516, 5382, 879, 2988, 6511, 9743, 3847, 5417, 3547, 1920, 7592, 3074, 8362, 1959, 185, 8399, 7480, 4339, 8667, 78, 7143, 6614, 9766, 5756, 5878, 5985, 2825, 2467, 9038, 3223, 8966, 993, 5357, 6241, 8655, 2355, 648, 7920, 2717, 2075, 6379, 7133, 2539, 1701, 3552, 6011, 4136, 8561, 1182, 5062, 3667, 8811, 3969, 6481, 4800, 7126, 5826, 3125, 6428, 5577, 6526, 5030, 3227, 7186, 3593, 1622, 632, 3584, 8515, 5981, 6528, 8434, 2868, 1660, 9446, 9826, 3740, 1381, 7086, 3050, 8358, 6968, 3611, 4433, 8236, 1683, 6949, 9103, 5815, 269, 1593, 4891, 411, 122, 9403, 6173, 6509, 507, 1362, 1612, 8088, 5709, 5094, 1565, 4958, 2990, 4119, 6572, 5956, 9453, 7337, 7661, 8824, 7763, 6710, 7967, 1295, 1945, 6852, 6607, 4606, 2384, 5972, 5149, 9724, 3964, 1436, 2417, 2045, 7028, 7741, 7718, 3265, 2854, 1778, 2076, 1819, 178, 5140, 3594, 3244, 9186, 3438, 7815, 2323, 4463, 6467, 8082, 3170, 1977, 229, 2898, 1986, 3553, 6711, 6260, 6899, 5645, 237, 9348, 3537, 4823, 7775, 4630, 9326, 4091, 8891, 6751, 9026, 3535, 2237, 431, 8892, 6338, 7879, 1292, 6764, 9652, 1400, 4156, 9098, 5904, 5959, 9043, 6640, 7595, 9833, 8854, 4745, 2666, 7394, 1905, 7597, 3405, 6466, 1969, 2607, 8470, 616, 8853, 2984, 9498, 5915, 7701, 904, 9909, 1720, 4562, 4652, 4305, 5734, 6955, 1350, 1702, 9835, 9782, 4897, 7813, 5060, 940, 3210, 9502, 1628, 4062, 4679, 7275, 3605, 562, 3256, 7089, 5726, 9645, 3379, 6045, 1700, 1388, 9641, 3284, 4810, 8680, 847, 7282, 5530, 4070, 7248, 5548, 9317, 5001, 2176, 2636, 2919, 1884, 4655, 3105, 4491, 4272, 404, 8883, 4209, 9214, 3625, 4487, 9737, 3727, 8489, 7950, 4159, 7871, 1517, 16, 6670, 1922, 6007, 6370, 9947, 1506, 2188, 7808, 2971, 2112, 159, 8242, 2105, 6929, 2873, 1378, 9176, 2749, 9779, 7554, 6127, 6776, 1065, 5381, 9381, 5480, 6839, 2615, 174, 1247, 8975, 1645, 8681, 2685, 1091, 9359, 6783, 6129, 2768, 9988, 4804, 6115, 7572, 4501, 2275, 1432, 7818, 6846, 5570, 3219, 5535, 7529, 107, 4455, 405, 7719, 8334, 8718, 9534, 3370, 202, 6060, 5574, 7519, 1844, 1460, 1947, 4866, 3335, 8392, 8368, 3444, 6685, 8803, 9189, 9124, 2370, 3482, 5491, 9838, 9956, 6320, 2271, 1177, 2230, 5573, 4358, 8083, 1183, 8282, 8848, 5235, 4253, 6440, 554, 6247, 2031, 2720, 430, 6418, 747, 3893, 8774, 525, 368, 975, 982, 2912, 8672, 5771, 2845, 8071, 7168, 5298, 4389, 380, 9273, 8484, 1609, 4495, 9996, 4993, 7205, 3923, 2635, 5714, 6286, 7744, 1180, 726, 2180, 1058, 355, 7231, 4626, 2402, 7414, 8016, 8973, 2692, 5610, 4359, 2953, 8009, 6911, 9122, 5520, 3901, 800, 6097, 176, 5462, 2884, 7185, 3355, 595, 2035, 5195, 7703, 8723, 2725, 2908, 4285, 5071, 4817, 7165, 5113, 6914, 824, 518, 1049, 2049, 8280, 6042, 6964, 5291, 1826, 1739, 426, 2273, 4960, 76, 128, 9223, 3873, 5983, 2829, 1607, 682, 8102, 5447, 2344, 2626, 2741, 6072, 3623, 9379, 576, 1904, 6416, 1918, 4489, 2653, 4526, 9376, 4067, 5393, 8792, 4277, 8706, 730, 4080, 6073, 9867, 4005, 900, 850, 1696, 8174, 8778, 4483, 9676, 9340, 696, 2490, 9708, 3828, 9177, 1657, 9, 8448, 9546, 9557, 6443, 8048, 7106, 432, 8068, 8958, 924, 5356, 5626, 8967, 6399, 8121, 3980, 1624, 19, 5042, 5655, 8873, 1076, 5252, 9631, 3325, 2734, 2858, 2828, 3233, 502, 3913, 4830, 8166, 2166, 1804, 8341, 1449, 9062, 7830, 6814, 8365, 8278, 5387, 2736, 6652, 2427, 4161, 6187, 4808, 1328, 1944, 460, 6421, 7259, 3178, 9574, 5160, 8464, 559, 978, 3321, 6385, 703, 7909, 759, 9392, 9358, 4003, 53, 7953, 389, 4371, 2450, 4306, 8172, 13, 1507, 9194, 5781, 9441, 6295, 7053, 2723, 8056, 4788, 8325, 4169, 1111, 6924, 92, 1093, 382, 1930, 6647, 5172, 9488, 6201, 8640, 8957, 1359, 6144, 702, 4150, 5191, 4231, 4778, 8740, 8838, 2679, 8178, 3022, 741, 4084, 714, 3565, 2228, 1635, 3702, 3472, 1411, 4376, 2602, 1636, 7140, 1435, 4598, 5423, 3867, 2117, 3815, 970, 9759, 8254, 2083, 7539, 8606, 3840, 2223, 7434, 5319, 7021, 3113, 1744, 5358, 4139, 2861, 4822, 7057, 5545, 7517, 2617, 3485, 7653, 6997, 2205, 7491, 3439, 631, 5459, 119, 626, 724, 8106, 6046, 9542, 6332, 9286, 9449, 6083, 9674, 2551, 7306, 2574, 3924, 5969, 6587, 6825, 4676, 2556, 8726, 2114, 4001, 9567, 4349, 9999, 8051, 7229, 4051, 457, 7860, 7193, 3772, 2508, 6147, 1512, 2633, 4018, 7452, 5834, 5021, 585, 4942, 4737, 7188, 9029, 4451, 4614, 8213, 8552, 9898, 5429, 2403, 7817, 5339, 7897, 9872, 9202, 8661, 708, 6037, 3658, 6732, 4198, 1616, 1589, 6360, 9932, 526, 8089, 8851, 962, 5163, 9178, 3107, 7048, 1316, 2132, 200, 8330, 7455, 9931, 4554, 7045, 4732, 6601, 7365, 6696, 950, 4280, 2256, 811, 281, 8482, 772, 2007, 3696, 7867, 8675, 2487, 3956, 6560, 8031, 8641, 3373, 8069, 6808, 4490, 4028, 9726, 5733, 4549, 4020, 6264, 6182, 9905, 7393, 2722, 2314, 1531, 3090, 5280, 8354, 6432, 4703, 4764, 2426, 9879, 1282, 4693, 7283, 6553, 2133, 310, 7610, 7433, 9716, 1454, 8508, 4345, 5958, 6638, 3056, 1246, 5858, 9401, 5921, 9016, 4370, 7466, 1682, 8730, 7669, 6349, 8183, 15, 1386, 1486, 8861, 33, 9997, 6989, 5968, 6091, 9083, 8308, 9504, 1712, 8512, 9128, 6438, 637, 6570, 5293, 4909, 8045, 217, 6469, 3331, 3104, 779, 5814, 9577, 6375, 1161, 6529, 3900, 4915, 3883, 9607, 1417, 6323, 9166, 7088, 7450, 1097, 7585, 509, 7732, 5031, 1581, 3080, 2173, 9466, 7761, 8545, 7023, 3564, 5359, 8232, 9536, 3347, 2316, 7845, 8167, 1704, 1689, 6683, 3909, 512, 4437, 5680, 9722, 4560, 7564, 8311, 3486, 4952, 1277, 2227, 6341, 8739, 5312, 8196, 8825, 7992, 7855, 6508, 5769, 4500, 9136, 5946, 845, 1483, 6774, 4709, 9421, 6446, 1756, 9454, 5048, 1595, 4341, 5823, 4343, 3703, 1630, 7331, 989, 8908, 9503, 2192, 7204, 5399, 7062, 4023, 7071, 3037, 5000, 1598, 7772, 6156, 1885, 9902, 9219, 2472, 4706, 6450, 5605, 5388, 8839, 680, 6434, 9318, 3706, 5268, 2875, 7988, 4949, 586, 3009, 28, 5603, 6866, 6954, 3408, 3881, 8432, 2993, 6690, 8514, 1613, 7651, 253, 9151, 687, 8347, 3237, 996, 6010, 308, 2301, 7157, 3661, 7131, 6923, 5192, 1678, 9844, 9111, 2716, 1268, 3430, 6178, 307, 4368, 8855, 8914, 8114, 2025, 3932, 927, 5121, 733, 6651, 2994, 6461, 5776, 4438, 4968, 8403, 8152, 8283, 6514, 208, 1448, 5791, 4015, 2570, 890, 2267, 6039, 6676, 6950, 2652, 7806, 1053, 8781, 1489, 9267, 581, 706, 9405, 8147, 1259, 8018, 4874, 8812, 7084, 8449, 6459, 5598, 5712, 2951, 3155, 3456, 2712, 2233, 1851, 5199, 5302, 9637, 1498, 9894, 1649, 3945, 8506, 7656, 9604, 4696, 4503, 3762, 4450, 8013, 7525, 4552, 6812, 7046, 6269, 9370, 6837, 6321, 1087, 9058, 6668, 7770, 8180, 6289, 9040, 1698, 7169, 2034, 8390, 5457, 7130, 7342, 6106, 609, 363, 2195, 5740, 2771, 9482, 7217, 3193, 2160, 5174, 1815, 657, 4561, 1218, 774, 6840, 6309, 9108, 1009, 1786, 1850, 4082, 1881, 1939, 9962, 9897, 8544, 8593, 2550, 105, 7956, 1286, 3127, 3966, 6071, 7368, 3062, 3131, 9519, 3197, 5724, 4472, 2040, 8750, 3245, 1421, 7566, 1620, 6624, 6741, 5901, 3698, 2029, 7402, 5102, 6689, 9715, 1033, 5703, 9212, 9298, 3760, 5955, 4010, 5461, 5707, 3098, 9172, 5226, 231, 2396, 2181, 2911, 7344, 6813, 3517, 7464, 8342, 2925, 3994, 8457, 8687, 5004, 5673, 6922, 3275, 8840, 5730, 8162, 6773, 4536, 7184, 2604, 2502, 5093, 6978, 9753, 5702, 4819, 7522, 5206, 8177, 1524, 6117, 461, 7483, 8733, 3172, 3921, 8857, 3167, 5238, 7565, 4908, 7957, 2847, 9228, 1748, 5996, 1220, 2863, 1543, 4767, 2387, 594, 4449, 6894, 2860, 4380, 948, 3519, 5394, 3391, 9584, 9399, 4600, 3273, 9188, 2406, 8111, 2498, 3843, 9677, 4961, 3301, 8425, 7577, 757, 884, 6100, 1596, 3448, 1515, 3618, 5059, 5911, 2512, 7276, 6236, 8437, 9860, 6449, 2418, 974, 8834, 3759, 3199, 7014, 2368, 7720, 2287, 3140, 6484, 8714, 3017, 9218, 7989, 4905, 622, 7352, 3615, 6542, 7863, 8601, 9354, 5963, 3753, 6371, 4831, 2371, 6299, 5860, 3690, 414, 1533, 2390, 464, 5481, 7171, 5384, 6118, 2897, 7802, 2128, 4870, 6054, 2457, 6303, 8995, 3933, 6069, 8262, 8049, 3254, 7420, 459, 6913, 1171, 4444, 9780, 5594, 9222, 5892, 6171, 9044, 1244, 9105, 5812, 9305, 9374, 1948, 2036, 3603, 337, 1452, 3271, 9622, 9877, 1184, 2269, 9203, 6008, 5321, 1883, 3442, 4398, 3462, 4142, 2542, 5264, 913, 7056, 5879, 5451, 7078, 903, 6620, 4236, 6092, 3912, 80, 2089, 1487, 7799, 7994, 8582, 5165, 2134, 7040, 2265, 2333, 6565, 2412, 3117, 491, 9254, 278, 7641, 5305, 103, 2756, 6112, 3232, 9664, 6713, 3619, 3295, 5828, 4862, 4713, 7591, 997, 8150, 168, 3383, 1687, 1455, 7579, 6369, 1845, 8595, 222, 961, 9747, 7731, 8953, 9242, 8962, 4918, 4228, 3596, 8294, 5209, 1806, 9847, 9470, 5077, 8361, 9935, 6107, 9657, 343, 8759, 8876, 9225, 6471, 9612, 152, 8691, 2995, 8225, 8456, 7795, 3766, 8578, 1895, 8522, 4531, 6995, 438, 6221, 9472, 8934, 5649, 3689, 8348, 4043, 2841, 8703, 2332, 8271, 6283, 3160, 9257, 5107, 4695, 4900, 4879, 7170, 8097, 9199, 6359, 6196, 6709, 5364, 3863, 2084, 7873, 4632, 1176, 4185, 8299, 3891, 8055, 407, 3, 3774, 4834, 8961, 1673, 9790, 7613, 1722, 4884, 7319, 8555, 6424, 1999, 1394, 5401, 141, 5171, 6217, 5975, 5504, 1336, 2051, 3266, 7689, 3825, 4622, 2367, 4845, 5216, 3263, 593, 3641, 399, 8160, 5907, 7971, 124, 2065, 6342, 7124, 5592, 5513, 9088, 9292, 6830, 1617, 7972, 7182, 983, 34, 9057, 2750, 1070, 3270, 6451, 9500, 3528, 327, 1557, 3180, 5971, 6800, 1034, 4975, 73, 477, 481, 7952, 7487, 9741, 2184, 6386, 2622, 6515, 6746, 7853, 9540, 8062, 6308, 4201, 5556, 5870, 5909, 2651, 9865, 7307, 3738, 9579, 353, 5529, 3473, 4814, 8787, 7916, 6002, 3741, 1941, 4688, 2886, 678, 7937, 5763, 6130, 9915, 9893, 2767, 6038, 8535, 3297, 5367, 6403, 1304, 7442, 9149, 1437, 3124, 9594, 6475, 6315, 3014, 5465, 6348, 8920, 5307, 3269, 7015, 6874, 1623, 8222, 2533, 7039, 6639, 8608, 522, 5888, 8443, 1305, 8436, 294, 867, 7451, 2484, 5997, 5386, 4144, 3797, 6076, 468, 930, 501, 956, 2159, 189, 9263, 5202, 162, 649, 7177, 7381, 818, 9123, 9836, 7578, 5681, 4409, 9713, 2714, 4394, 1972, 6731, 8690, 9141, 6619, 8367, 1869, 3508, 3242, 7694, 3691, 2261, 8040, 2462, 8119, 4528, 2486, 8134, 6337, 8315, 7215, 6519, 1372, 5231, 5049, 6243, 1426, 3065, 5742, 3695, 560, 88, 3865, 1478, 1571, 3728, 8257, 611, 9521, 2527, 6915, 2682, 9656, 5604, 3784, 6512, 6851, 6089, 1109, 5064, 5170, 7557, 5637, 4757, 9090, 3298, 2477, 1784, 2081, 1603, 3559, 1714, 9964, 1891, 2121, 9109, 5523, 7335, 1331, 6609, 937, 1601, 6510, 2942, 1008, 3692, 4962, 4226, 6457, 5917, 7745, 857, 7417, 7547, 7594, 3927, 8927, 4214, 4717, 2944, 2407, 9074, 646, 3001, 6155, 4932, 6368, 7400, 454, 9624, 4663, 466, 4006, 4269, 2208, 1289, 8704, 3588, 5788, 9764, 9283, 6068, 1393, 4208, 1004, 7549, 6437, 7024, 5900, 3470, 2456, 8429, 4263, 3302, 3221, 3433, 1088, 6262, 1311, 5289, 2202, 9911, 2003, 7711, 3626, 9987, 5700, 9969, 9578, 8735, 5572, 1241, 629, 4682, 299, 9424, 2027, 3149, 2258, 2359, 7587, 3120, 8164, 1017, 4714, 5478, 9400, 1406, 9963, 3122, 3110, 2217, 4588, 663, 8752, 9056, 990, 4842, 1322, 196, 4597, 9736, 7099, 8081, 1026, 1574, 2677, 3368, 4054, 6193, 2535, 3595, 3849, 9050, 2785, 2087, 5931, 5144, 7329, 6779, 8826, 4408, 1355, 9259, 1332, 7462, 971, 6022, 8864, 2578, 3597, 6488, 5309, 7729, 897, 1983, 9192, 4527, 2759, 7915, 4777, 4478, 9755, 1504, 9895, 4366, 8988, 1255, 8938, 2740, 9356, 5937, 7111, 2006, 2334, 8755, 5119, 8205, 4098, 1273, 5240, 4448, 2431, 4661, 7311, 6933, 8440, 9719, 1257, 5743, 2170, 7066, 8043, 3884, 1438, 577, 1901, 4843, 8630, 1083, 1240, 3672, 6211, 9703, 7571, 3357, 653, 469, 7827, 6733, 7110, 4234, 5036, 188, 5353, 5824, 2086, 3079, 5947, 6857, 2244, 6855, 9640, 952, 6903, 5286, 7285, 125, 9037, 4807, 8110, 5872, 5185, 6198, 7469, 4739, 5579, 2044, 4514, 9957, 1264, 3277, 6425, 6109, 4148, 70, 439, 7031, 1763, 3520, 3683, 6608, 135, 2048, 9824, 6575, 350, 6665, 7740, 3967, 4705, 2386, 5269, 2224, 9173, 1346, 8428, 1416, 1205, 153, 6613, 6771, 7473, 1759, 8376, 4439, 9923, 5954, 4934, 851, 5081, 9476, 9854, 8287, 6698, 3853, 8531, 2895, 1849, 6465, 440, 7235, 9319, 3685, 3082, 3599, 3748, 5836, 3750, 184, 4218, 2894, 7646, 6090, 7348, 225, 822, 4744, 3864, 7410, 7960, 2999, 6939, 4424, 5924, 694, 1950, 5190, 5074, 9195, 7603, 1338, 6412, 5482, 3943, 4650, 8474, 3985, 7531, 4422, 228, 9730, 690, 558, 6760, 964, 9101, 7986, 4223, 3730, 5816, 6183, 6606, 7734, 3779, 3548, 4479, 495, 172, 9912, 1605, 4102, 6170, 3034, 9711, 544, 5154, 7965, 6288, 4978, 8411, 6530, 771, 2207, 4848, 5025, 1119, 573, 5188, 3503, 1602, 5162, 3988, 1379, 1225, 8488, 2283, 7884, 5568, 5856, 6745, 5762, 2545, 4540, 8877, 5817, 1568, 4950, 7369, 3054, 7683, 909, 8548, 5641, 881, 5952, 3440, 2259, 7362, 7963, 4247, 4256, 2598, 2061, 5746, 9314, 7032, 9550, 5737, 7435, 9958, 3604, 62, 2585, 4742, 2069, 7904, 5922, 8794, 5422, 8388, 1968, 8273, 334, 7114, 6568, 2197, 1243, 8269, 7528, 7970, 1110, 5376, 3134, 7542, 5101, 9689, 8660, 9941, 844, 2586, 9485, 6562, 3612, 1790, 4499, 2708, 5696, 410, 8589, 5194, 7463, 5524, 6215, 7885, 7246, 5614, 5792, 9148, 5514, 9813, 8635, 2079, 6871, 3948, 5757, 8795, 3920, 4152, 5670, 4880, 7616, 2962, 3879, 6556, 8141, 2976, 4215, 7716, 5138, 8212, 1022, 4765, 1743, 907, 2958, 7498, 305, 329, 1632, 4644, 833, 1072, 6500, 9679, 2798, 7681, 4428, 7538, 1796, 1461, 4773, 1012, 5420, 6051, 3443, 3914, 3419, 4569, 4299, 7618, 3970, 4770, 9448, 5599, 6195, 4042, 4036, 3513, 2557, 6610, 4572, 2693, 494, 7733, 3230, 4229, 6157, 6580, 5632, 4815, 2343, 2028, 852, 4910, 9564, 3420, 2169, 1030, 2379, 1639, 4638, 4741, 9976, 1353, 1971, 7962, 1052, 3011, 7500, 3898, 2455, 9986, 7774, 5999, 5918, 6422, 4393, 3598, 6210, 8807, 1413, 5066, 9023, 166, 9646, 5951, 3019, 9515, 9919, 1828, 742, 5928, 4040, 1054, 3216, 5045, 6910, 5196, 1056, 9944, 3257, 4293, 1488, 6889, 2511, 3171, 2642, 4586, 7234, 8621, 7995, 9221, 1403, 8487, 8808, 4289, 2193, 2583, 5509, 7760, 6040, 1031, 5452, 7574, 7961, 98, 3936, 219, 8500, 9717, 4250, 9413, 3518, 8686, 5694, 2935, 5180, 720, 7457, 2788, 3987, 7477, 2161, 9878, 1924, 3810, 7942, 7648, 3990, 7471, 4953, 6754, 9618, 5920, 9383, 1130, 3148, 4539, 7101, 7413, 6104, 7751, 1274, 516, 6767, 8331, 3340, 7373, 2770, 5549, 2077, 7161, 9104, 3755, 4480, 2073, 1961, 100, 7470, 5075, 8776, 3251, 4557, 8393, 2454, 541, 9338, 1057, 3452, 4550, 8869, 8867, 5517, 1705, 5210, 791, 752, 3944, 2530, 8371, 6128, 9420, 6177, 7627, 1112, 3841, 6391, 1750, 251, 8722, 1037, 5243, 1964, 349, 1615, 75, 8928, 9773, 4658, 8785, 7516, 531, 4735, 7102, 7803, 7523, 6251, 3467, 395, 2870, 2286, 6552, 6828, 892, 3407, 9731, 2299, 6224, 9010, 4642, 7398, 6311, 9051, 8159, 2950, 5970, 4821, 2183, 7482, 187, 9866, 425, 9961, 510, 3206, 5182, 5929, 9423, 4710, 8579, 7380, 6324, 275, 2669, 5410, 6890, 7018, 2784, 8922, 7866, 8771, 9418, 4670, 4648, 5628, 6062, 1313, 9015, 2104, 3811, 4304, 1369, 1000, 266, 6729, 4671, 8028, 3796, 8194, 1781, 2493, 4303, 1591, 9523, 7938, 9668, 2504, 6015, 7987, 3640, 6181, 2335, 2600, 976, 476, 5148, 109, 3132, 6329, 8964, 7432, 9561, 1921, 5876, 3813, 3808, 7361, 2328, 4763, 2309, 6724, 7692, 3195, 9797, 1990, 8554, 7163, 8185, 4454, 4330, 9458, 4889, 5683, 2929, 2730, 5617, 4894, 1496, 7012, 5962, 8363, 728, 346, 8095, 5644, 6775, 9175, 6270, 9543, 1038, 8079, 6257, 7784, 9720, 4914, 3837, 1139, 1822, 8893, 8463, 7357, 3968, 7383, 150, 3617, 5949, 5533, 8424, 1626, 901, 4121, 8244, 6844, 8381, 8823, 9261, 939, 1015, 54, 9812, 2153, 8719, 2934, 4662, 2099, 2810, 241, 3734, 3189, 4311, 9883, 5283, 2700, 556, 873, 9980, 2425, 7266, 5047, 8143, 4699, 4820, 1775, 9916, 9006, 4535, 9820, 1318, 4684, 4207, 7426, 5114, 9937, 9322, 3313, 933, 2546, 1549, 3769, 580, 668, 7214, 3573, 2910, 2429, 2913, 3226, 688, 932, 2619, 9479, 8135, 3918, 3778, 960, 5864, 2659, 1300, 9303, 689, 9929, 7546, 5029, 5765, 4890, 7877, 9688, 8626, 1320, 3461, 2614, 9665, 6279, 7852, 4033, 4245, 3858, 290, 6686, 8842, 5132, 3326, 43, 3153, 3562, 9829, 5748, 6880, 1526, 8241, 3917, 2537, 5893, 9253, 7673, 4926, 9046, 8566, 5543, 7128, 5547, 7123, 7810, 9260, 9473, 691, 6387, 2113, 1492, 4523, 9723, 8270, 3838, 9435, 5796, 4722, 2961, 4602, 1006, 3666, 5750, 4885, 5688, 4887, 7293, 7725, 7568, 985, 1349, 1067, 9368, 7109, 6061, 91, 1265, 5063, 984, 6988, 4103, 5453, 836, 5263, 1902, 3648, 1459, 5365, 4199, 4505, 5938, 9367, 987, 1931, 5251, 2018, 9808, 5711, 6687, 5355, 2187, 7901, 5625, 2474, 4617, 9007, 3106, 3129, 9802, 2916, 6770, 1852, 2167, 4238, 3871, 1958, 3958, 9465, 64, 4073, 8969, 2221, 5964, 5315, 9210, 6554, 2577, 2727, 5825, 7678, 1335, 6520, 1813, 6661, 1306, 3758, 3580, 9817, 6657, 5267, 9541, 7541, 7892, 4916, 1287, 6522, 2654, 2345, 4232, 6274, 865, 5987, 1392, 4488, 4721, 8050, 4446, 9513, 5270, 7148, 7238, 8553, 112, 9419, 1420, 2915, 6728, 2304, 9979, 4397, 9270, 972, 777, 2864, 1548, 2352, 3399, 6032, 2165, 409, 5773, 6793, 5373, 4544, 7667, 7841, 2926, 699, 1634, 20, 7060, 5635, 2804, 3624, 5011, 3374, 2869, 9667, 147, 506, 1978, 9591, 4615, 7390, 2781, 9718, 344, 7260, 4555, 4970, 4058, 9821, 6798, 6322, 3363, 9653, 3040, 5155, 6716, 8099, 1980, 8829, 7954, 3896, 4266, 156, 951, 2449, 704, 8654, 3903, 5780, 2515, 9992, 2667, 7612, 667, 3209, 6719, 2251, 8761, 6671, 5554, 1477, 7941, 8550, 3570, 5539, 2680, 8980, 6925, 3620, 6546, 7548, 9750, 422, 6468, 6629, 9740, 4364, 65, 7304, 2401, 8266, 9650, 8835, 3821, 4902, 5701, 2793, 610, 7973, 4038, 2650, 97, 796, 3907, 6569, 523, 7934, 2067, 9293, 5214, 6717, 6763, 8032, 5527, 6131, 7298, 22, 4967, 6249, 2164, 8862, 5446, 9493, 3319, 117, 784, 3154, 9460, 3108, 7976, 35, 7409, 4852, 8485, 7172, 4296, 3041, 8155, 2949, 6297, 2409, 7485, 330, 271, 9868, 2253, 6094, 5486, 171, 6718, 2294, 7054, 139, 4750, 0, 7291, 3627, 7288, 4321, 4276, 9412, 4836, 1324, 6160, 5546, 7038, 3802, 3546, 9758, 155, 9315, 3231, 2276, 1509, 8859, 4752, 4286, 3036, 8948, 2972, 2862, 1530, 9680, 8870, 3386, 3264, 4125, 7284, 1319, 3687, 2945, 534, 1618, 1817, 3971, 7197, 8905, 8615, 4032, 8924, 7026, 8941, 1373, 25, 2986, 8060, 4007, 5118, 7890, 8498, 1479, 4727, 7206, 8501, 4605, 2308, 6442, 9291, 4607, 3723, 7077, 3911, 7777, 813, 5323, 7758, 4313, 3484, 9144, 9926, 2843, 4416, 4334, 1401, 3887, 2342, 6020, 2388, 206, 5056, 908, 4050, 6758, 1847, 9863, 4955, 9774, 7429, 5806, 9686, 4053, 4746, 4937, 910, 7657, 133, 8784, 722, 4683, 1211, 9635, 4381, 627, 7436, 5079, 5914, 9914, 9204, 3931, 1302, 5426, 2096, 5050, 8444, 1164, 8108, 6373, 9075, 6192, 7448, 286, 5925, 9049, 7650, 5464, 1793, 1519, 7176, 8912, 7672, 2140, 7187, 5391, 2452, 5187, 7055, 1473, 781, 9313, 249, 2844, 6618, 1142, 4838, 2646, 9752, 3846, 4301, 3416, 5623, 1735, 5150, 356, 8366, 2232, 2247, 7883, 641, 4895, 1679, 1409, 652, 1866, 9000, 2215, 8042, 4457, 3305, 8067, 61, 163, 4553, 3401, 7223, 5843, 530, 9620, 7297, 8916, 9117, 1564, 9047, 6504, 4332, 2471, 14, 1480, 2815, 9499, 6383, 5090, 7049, 5419, 5037, 5052, 4835, 1638, 620, 6164, 2640, 423, 2381, 1584, 9516, 4858, 3752, 5591, 3829, 5698, 1036, 8208, 1234, 8572, 9445, 5643, 9870, 1742, 2776, 1293, 5768, 1667, 7391, 6947, 2424, 895, 3663, 5057, 5793, 6310, 4633, 2245, 1190, 1192, 3488, 4078, 4685, 9325, 4217, 3761, 1877, 6435, 9621, 533, 4564, 1733, 5442, 1254, 2848, 4635, 8446, 4581, 322, 2313, 5874, 954, 9501, 1855, 618, 1207, 1816, 9525, 9590, 5098, 7315, 7208, 1578, 8977, 1870, 8360, 4468, 6448, 709, 7849, 245, 7924, 4923, 5839, 9411, 7296, 6886, 3556, 732, 7116, 6809, 9398, 9765, 5716, 3693, 4813, 9320, 823, 1676, 3128, 945, 3282, 532, 2097, 6043, 7141, 32, 5941, 74, 7403, 1032, 1472, 2887, 5133, 3069, 6238, 3563, 8822, 5933, 2608, 5329, 5942, 9416, 5884, 4623, 9874, 2552, 8337, 5990, 2888, 750, 8575, 6186, 623, 127, 4873, 360, 1599, 6604, 6506, 3999, 7985, 846, 9463, 4031, 7431, 3236, 3425, 7589, 9972, 484, 2414, 4935, 3300, 79, 4640, 1988, 4893, 9639, 838, 1879, 6926, 1213, 2010, 1433, 2753, 870, 1266, 6327, 7201, 825, 7228, 7652, 9163]
    # labels = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    query_strs = ["Duration(Conjunction(Far_3(o0, o1), RightQuadrant(o1)), 15); Conjunction(FrontOf(o0, o1), Near_1(o0, o1)); Duration(Conjunction(Far_3(o0, o1), RightQuadrant(o1)), 5)", "Conjunction(Conjunction(Color_cyan(o0), Far_3(o1, o2)), Shape_sphere(o1)); Conjunction(FrontOf(o1, o2), Near_1(o1, o2)); Conjunction(Far_3(o1, o2), TopQuadrant(o2))", "Duration(Conjunction(LeftOf(o0, o1), Shape_sphere(o1)), 15); Conjunction(FrontOf(o0, o2), Near_1(o0, o2)); Duration(Conjunction(Behind(o0, o2), Far_3(o0, o2)), 5)"]
    for query_str in query_strs:
        print(query_str)
        current_query = str_to_program_postgres(query_str)
        # query_str = rewrite_program_postgres(current_query)
        # print(current_query)
        outputs, new_memoize_scene_graph, new_memoize_sequence = postgres_execute_no_caching(dsn, current_query, memoize_scene_graph, memoize_sequence, inputs_table_name, input_vids=500, is_trajectory=False, sampling_rate=None)
        # print(len(outputs))
        # print(outputs)
        print("time", time.time() - _start)

        # preds = []
        # for input in input_vids:
        #     if input in outputs:
        #         preds.append(1)
        #     else:
        #         preds.append(0)
        # score = f1_score(labels, preds)
        # print(score)