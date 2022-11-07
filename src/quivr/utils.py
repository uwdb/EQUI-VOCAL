import csv
import json
import os
import random
import numpy as np
import quivr.dsl as dsl
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

def complexity_cost(program):
    cost_depth = len(program)
    cost_npred = sum([len(dict["scene_graph"]) for dict in program])
    cost_duration = sum([(dict["duration_constraint"] - 1) * (1 + 0.1 * len(dict["scene_graph"])) for dict in program])
    return cost_npred + cost_depth * 0.5 + cost_duration

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
                    cur.execute("CREATE TEMPORARY TABLE Obj_filtered AS SELECT oid, vid, fid / 4 as fid, shape, color, material, x1, y1, x2, y2 FROM {} WHERE vid < {} AND fid % {} = 0;".format(inputs_table_name, input_vids, sampling_rate))
                else:
                    cur.execute("CREATE TEMPORARY TABLE Obj_filtered AS SELECT * FROM {} WHERE vid < {};".format(inputs_table_name, input_vids))
                input_vids = list(range(input_vids))
            else:
                if sampling_rate:
                    cur.execute("CREATE TEMPORARY TABLE Obj_filtered AS SELECT oid, vid, fid / 4 as fid, shape, color, material, x1, y1, x2, y2 FROM {} WHERE vid = ANY(%s) AND fid %% {} = 0;".format(inputs_table_name, sampling_rate), [input_vids])
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
                seq_signature = rewrite_program_postgres(current_query[:len(current_query)-i])
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
                # signature = rewrite_program_postgres(current_query[:graph_idx+1])
                cached_results = cached_df_seq_deque.pop()
                delta_input_vids.extend(cached_vids_deque.pop())

                _start_filtered = time.time()
                if graph_idx > 0:
                    obj_union = copy.deepcopy(encountered_variables_prev_graphs)
                    obj_intersection = []
                    obj_union_fields = []
                    obj_intersection_fields = []
                    for v in encountered_variables_prev_graphs:
                        obj_union_fields.append("t0.{}_oid".format(v))
                    for v in encountered_variables_current_graph:
                        if v in encountered_variables_prev_graphs:
                            obj_intersection.append(v)
                            obj_intersection_fields.append("t0.{v}_oid = t1.{v}_oid".format(v=v))
                        else:
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

def postgres_execute_cache_sequence(dsn, current_query, memoize_scene_graph, memoize_sequence, inputs_table_name, input_vids, is_trajectory=True, sampling_rate=None):
    """
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
                seq_signature = rewrite_program_postgres(current_query[:len(current_query)-i])
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
            # select input videos
            _start = time.time()
            if isinstance(input_vids, int):
                if sampling_rate:
                    cur.execute("CREATE TEMPORARY TABLE Obj_filtered AS SELECT oid, vid, fid / 4 as fid, shape, color, material, x1, y1, x2, y2 FROM {} WHERE vid < {} AND fid % {} = 0;".format(inputs_table_name, input_vids, sampling_rate))
                else:
                    cur.execute("CREATE TEMPORARY TABLE Obj_filtered AS SELECT * FROM {} WHERE vid < {};".format(inputs_table_name, input_vids))
                input_vids = list(range(input_vids))
            else:
                if sampling_rate:
                    cur.execute("CREATE TEMPORARY TABLE Obj_filtered AS SELECT oid, vid, fid / 4 as fid, shape, color, material, x1, y1, x2, y2 FROM {} WHERE vid = ANY(%s) AND fid %% {} = 0;".format(inputs_table_name, sampling_rate), [filtered_vids])
                else:
                    cur.execute("CREATE TEMPORARY TABLE Obj_filtered AS SELECT * FROM {} WHERE vid = ANY(%s);".format(inputs_table_name), [filtered_vids])
            cur.execute("CREATE INDEX IF NOT EXISTS idx_obj_filtered ON Obj_filtered (vid, fid, oid);")
            # print("select input videos: ", time.time() - _start)

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
                # print("Time for graph {}: {}".format(graph_idx, time.time() - _start))

                # Read cached results
                seq_signature = signatures.pop()
                cached_results = cached_df_seq_deque.pop()

                _start_filtered = time.time()
                if graph_idx > 0:
                    obj_union = copy.deepcopy(encountered_variables_prev_graphs)
                    obj_intersection = []
                    obj_union_fields = []
                    obj_intersection_fields = []
                    for v in encountered_variables_prev_graphs:
                        obj_union_fields.append("t0.{}_oid".format(v))
                    for v in encountered_variables_current_graph:
                        if v in encountered_variables_prev_graphs:
                            obj_intersection.append(v)
                            obj_intersection_fields.append("t0.{v}_oid = t1.{v}_oid".format(v=v))
                        else:
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
                    obj_union = encountered_variables_current_graph
                # print("filtered: ", time.time() - _start_filtered)

                # Generate scene graph sequence:
                _start_windowed = time.time()
                table_name = "g{}_filtered".format(graph_idx) if graph_idx > 0 else "g{}".format(graph_idx)
                obj_union_fields = ", ".join(["{}_oid".format(v) for v in obj_union])
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


def postgres_execute_no_caching(dsn, current_query, memoize_scene_graph, memoize_sequence, inputs_table_name, input_vids, is_trajectory=True, sampling_rate=None):
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
            new_memoize_scene_graph = [{} for _ in range(len(memoize_scene_graph))]
            new_memoize_sequence = [{} for _ in range(len(memoize_sequence))]
            # select input videos
            _start = time.time()
            if isinstance(input_vids, int):
                if sampling_rate:
                    cur.execute("CREATE TEMPORARY TABLE Obj_filtered AS SELECT oid, vid, fid / 4 as fid, shape, color, material, x1, y1, x2, y2 FROM {} WHERE vid < {} AND fid % {} = 0;".format(inputs_table_name, input_vids, sampling_rate))
                else:
                    cur.execute("CREATE TEMPORARY TABLE Obj_filtered AS SELECT * FROM {} WHERE vid < {};".format(inputs_table_name, input_vids))
                input_vids = list(range(input_vids))
            else:
                if sampling_rate:
                    cur.execute("CREATE TEMPORARY TABLE Obj_filtered AS SELECT oid, vid, fid / 4 as fid, shape, color, material, x1, y1, x2, y2 FROM {} WHERE vid = ANY(%s) AND fid %% {} = 0;".format(inputs_table_name, sampling_rate), [input_vids])
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
                sql_sring = """CREATE TEMPORARY TABLE g{} AS SELECT {} FROM {} WHERE {};""".format(graph_idx, fields, tables, where_clauses)
                # print(sql_sring)
                cur.execute(sql_sring)
                # print("execute for unseen videos: ", time.time() - _start_execute)
                # print("Time for graph {}: {}".format(graph_idx, time.time() - _start))

                _start_filtered = time.time()
                if graph_idx > 0:
                    obj_union = copy.deepcopy(encountered_variables_prev_graphs)
                    obj_intersection = []
                    obj_union_fields = []
                    obj_intersection_fields = []
                    for v in encountered_variables_prev_graphs:
                        obj_union_fields.append("t0.{}_oid".format(v))
                    for v in encountered_variables_current_graph:
                        if v in encountered_variables_prev_graphs:
                            obj_intersection.append(v)
                            obj_intersection_fields.append("t0.{v}_oid = t1.{v}_oid".format(v=v))
                        else:
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
                encountered_variables_prev_graphs = obj_union
                encountered_variables_current_graph = []

            cur.execute("SELECT DISTINCT vid FROM g{}_contiguous".format(len(current_query) - 1))
            # print("SELECT DISTINCT vid FROM g{}_contiguous".format(len(current_query) - 1))
            output_vids = cur.fetchall()
            output_vids = [row[0] for row in output_vids]
            conn.commit()
    return output_vids, new_memoize_scene_graph, new_memoize_sequence


def rewrite_program_postgres(orig_program):
    """
    Input:
    program: query in the dictionary format
    Output: query in string format, which is ordered properly (uniquely).
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
        scene_graph = sorted(scene_graph, key=lambda x: x["predicate"])
        for i, p in enumerate(scene_graph):
            rewritten_variables = []
            for v in p["variables"]:
                if v not in encountered_variables:
                    encountered_variables.append(v)
                    rewritten_variables.append("o" + str(len(encountered_variables) - 1))
                else:
                    rewritten_variables.append("o" + str(encountered_variables.index(v)))
            # Sort rewritten variables
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
    # correct_filename("synthetic-fn_error_rate_0.3-fp_error_rate_0.075")
    # get_query_str_from_filename("inputs/synthetic_rare",)
    # construct_train_test("Sequencing(Sequencing(Sequencing(Sequencing(Sequencing(Sequencing(True*, Back), True*), Left), True*), Conjunction(Conjunction(Back, Left), Far_0.9)), True*)", n_train=300)

    memoize_scene_graph = [{} for _ in range(12747)]
    memoize_sequence = [{} for _ in range(12747)]
    # inputs_table_name = "Obj_clevrer"
    inputs_table_name = "Obj_collision"
    _start = time.time()
    input_vids = [11942, 11182, 6948, 5482, 7188, 10228, 1567, 823, 750, 47, 1808, 11473, 7632, 6041, 3278, 8306, 313, 12143, 4217, 1795, 6545, 8136, 12175, 5955, 10872, 2053, 11360, 8189, 10697, 6711, 11393, 4303, 1964, 5481, 12742, 7915, 4359, 12324, 2469, 10259, 10720, 1134, 6220, 4387, 12086, 6979, 508, 10843, 5684, 1860, 2324, 5288, 1133, 1415, 6365, 4700, 5501, 8891, 11742, 4529, 6252, 1015, 7118, 5891, 2850, 7453, 11330, 12394, 6444, 9865, 7810, 8252, 6103, 447, 12595, 7450, 5770, 11504, 11898, 4497, 6378, 11329, 9346, 3772, 10210, 11039, 8790, 10116, 10306, 12323, 9924, 6344, 9735, 2495, 7762, 9708, 6851, 9664, 11779, 8918, 4437, 10818, 11509, 2757, 8240, 8050, 5008, 8480, 6900, 544, 5843, 4455, 3406, 3675, 6855, 12094, 1035, 7970, 6510, 4515, 1759, 9659, 6426, 10954, 10593, 1060, 9788, 7226, 4157, 6165, 6116, 1135, 1127, 4236, 9311, 2130, 3499, 10095, 1888, 1870, 6353, 4312, 7394, 12336, 8086, 1414, 1868, 9123, 8715, 11146, 9370, 7640, 12330, 4649, 11010, 11243, 5805, 9993, 548, 8768, 6606, 11249, 5375, 1111, 6432, 7726, 2217, 3152, 7916, 8986, 4740, 4306, 11781, 8618, 5026, 6824, 850, 9159, 435, 539, 587, 9217, 353, 3935, 6418, 4822, 9869, 526, 7727, 4188, 4125, 11802, 8817, 4851, 11045, 4645, 7923, 6113, 7218, 5822, 487, 11223, 1041, 742, 5511, 7539, 1506, 10361, 2394, 5218, 7300, 10686, 411, 1468, 6328, 10953, 6625, 4887, 11524, 8058, 4260, 286, 4259, 4654, 11469, 2770, 3448, 7375, 2256, 12578, 12513, 6448, 10204, 10473, 2634, 8693, 7913, 5395, 12413, 4082, 5597, 11833, 628, 11835, 1986, 3864, 7769, 2896, 5156, 10196, 9429, 1667, 1098, 1814, 4490, 7802, 5975, 5273, 11566, 3858, 1926, 5445, 138, 10188, 7997, 6336, 2849, 12333, 7423, 7584, 2267, 4042, 11860, 6347, 8644, 8234, 11014, 7939, 9025, 7753, 12435, 168, 1259, 5942, 6745, 2866, 251, 2506, 6441, 10949, 4543, 8820, 8281, 2168, 5558, 10082, 8575, 4663, 5515, 622, 96, 8465, 8992, 6821, 9386, 7856, 1592, 7142, 6991, 8831, 8777, 7343, 10897, 5954, 11900, 3590, 6992, 8038, 7634, 6012, 4802, 2360, 4993, 7801, 1583, 11552, 11739, 0, 3624, 3969, 12680, 386, 11325, 3091, 4166, 5253, 7531, 5043, 10710, 20, 9913, 9896, 867, 11766, 11053, 5127, 9525, 1157, 5860, 4250, 6145, 11095, 2597, 12142, 7380, 182, 10241, 5700, 11944, 1570, 1293, 4355, 4666, 2617, 8356, 2304, 1588, 4307, 9494, 12311, 10734, 5912, 790, 8903, 11345, 7964, 1230, 1682, 1597, 886, 10286, 5560, 5894, 3000, 10371, 171, 12166, 7298, 4009, 3240, 5261, 4404, 6283, 10192, 10465, 105, 2660, 7221, 8214, 11617, 743, 3148, 8608, 3493, 9085, 10271, 6837, 6397, 4023, 989, 5141, 3885, 8529, 10841, 10603, 5652, 9033, 12164, 210, 9609, 4033, 5806, 2858, 1953, 2807, 5444, 10254, 7881, 3467, 3413, 5687, 2546, 5295, 7072, 151, 4324, 754, 4863, 11843, 12363, 7699, 6509, 12533, 1646, 6532, 12008, 12275, 3459, 8090, 6298, 4705, 2607, 1287, 1147, 11998, 1854, 525, 11485, 3813, 7491, 937, 10451, 7093, 3907, 9447, 9764, 9356, 10121, 1164, 6675, 12741, 4180, 4423, 4224, 5341, 6021, 4875, 8842, 6724, 10001, 1384, 4492, 821, 6471, 10501, 7525, 6648, 9915, 4464, 8911, 695, 7874, 11677, 6636, 11857, 4821, 1489, 4273, 4796, 4963, 3411, 12539, 5212, 10869, 6078, 12335, 3428, 11301, 4981, 2475, 5627, 5526, 10025, 5474, 423, 8401, 2272, 2861, 2340, 5123, 4159, 7237, 10656, 9553, 9686, 11593, 2373, 7617, 11384, 7330, 1092, 1262, 3296, 8581, 6223, 6933, 8440, 4598, 1620, 6234, 3911, 1466, 6572, 2747, 6070, 9523, 12381, 5057, 9034, 5518, 4291, 3945, 243, 4838, 2707, 8660, 1523, 4428, 4342, 1710, 1605, 5918, 535, 9670, 10735, 2583, 3971, 8219, 1434, 2833, 2916, 4642, 12616, 4335, 11806, 3981, 1151, 4441, 6322, 291, 8872, 7867, 3430, 2455, 12518, 8521, 8974, 8742, 7353, 12640, 9393, 5346, 3634, 5978, 2374, 3639, 4947, 9461, 8507, 7608, 7886, 4652, 7703, 281, 4098, 9114, 3967, 10039, 5514, 1026, 3795, 5017, 7553, 12045, 9985, 12725, 9808, 5432, 755, 7204, 9264, 10581, 12288, 5050, 11531, 10274, 3787, 1354, 10157, 1179, 2117, 7259, 402, 10903, 9032, 9931, 5797, 1688, 2543, 8352, 7609, 10670, 6579, 287, 8536, 11331, 4269, 11892, 7508, 9811, 10172, 6536, 4126, 1612, 6507, 1051, 11365, 5923, 4002, 10491, 5106, 892, 7503, 1435, 8459, 6581, 6776, 9385, 5899, 4334, 10577, 7796, 2642, 255, 2673, 7993, 8975, 9117, 2658, 7774, 12672, 2701, 7768, 6790, 2464, 4787, 2715, 311, 6212, 6743, 12456, 11029, 9588, 1530, 12476, 12459, 6812, 12570, 296, 5807, 2787, 10987, 4187, 11723, 9277, 10480, 11284, 6187, 7316, 10307, 11015, 8961, 2083, 1683, 10328, 2626, 6340, 5177, 12383, 12643, 1881, 1055, 6809, 4569, 5934, 5529, 10235, 7504, 1657, 3014, 651, 2780, 7944, 11948, 6057, 8794, 1873, 10470, 5608, 8796, 6303, 574, 3404, 2406, 2236, 6167, 3844, 11708, 2278, 4208, 1824, 9911, 5965, 6067, 6769, 3336, 2391, 641, 3926, 7569, 5137, 6475, 5335, 12681, 5209, 10090, 1501, 2604, 2358, 921, 10534, 7145, 10002, 11107, 8027, 4069, 1780, 4512, 2742, 4185, 12638, 4118, 8737, 9340, 4088, 6683, 9969, 7686, 9462, 12220, 7857, 2645, 11929, 4605, 10114, 2592, 7597, 4527, 9041, 12046, 5830, 1838, 1980, 418, 6394, 1386, 233, 8009, 1758, 9460, 9813, 6923, 8576, 2095, 11172, 1563, 1136, 1817, 5650, 1834, 511, 1928, 7593, 5827, 7742, 1074, 12053, 6002, 11984, 6830, 1955, 12492, 2847, 5500, 11285, 2987, 6525, 12138, 8175, 6274, 1188, 5952, 588, 6034, 7870, 5297, 1513, 9415, 2385, 7341, 4169, 10574, 8276, 5165, 8205, 1312, 6356, 1057, 10636, 8861, 9774, 1783, 8770, 10524, 10376, 10824, 12583, 6291, 9487, 854, 10858, 7961, 6616, 8203, 6038, 11760, 2173, 6604, 7515, 6700, 6951, 581, 4669, 11298, 7098, 11465, 7705, 8167, 10969, 2438, 12237, 4401, 2914, 6294, 7455, 1426, 1737, 2154, 9016, 4371, 864, 12044, 8807, 6513, 11357, 8233, 946, 4890, 5271, 1094, 10006, 9948, 5076, 11106, 4684, 12450, 1492, 11931, 12373, 11868, 2555, 4341, 11131, 8826, 11237, 8889, 2183, 3228, 9778, 6331, 11499, 3244, 10297, 10181, 9495, 4521, 2462, 530, 8724, 1155, 2518, 983, 9742, 1613, 11402, 3301, 235, 7645, 6360, 4278, 10126, 11327, 11838, 8125, 2899, 12319, 1300, 8122, 6468, 73, 10290, 8508, 12608, 7782, 10988, 4123, 10007, 834, 5107, 11597, 3204, 9403, 6632, 2084, 12377, 3110, 7128, 6682, 5384, 11973, 3275, 6789, 12126, 12393, 1922, 8523, 267, 7170, 2630, 11801, 10508, 2997, 5673, 3095, 7039, 1537, 1171, 9408, 246, 5232, 8387, 5284, 1574, 10409, 10564, 11344, 10472, 4618, 5328, 2375, 5223, 12594, 11211, 1826, 1884, 1786, 6090, 7751, 10021, 5071, 6261, 9641, 2416, 9126, 759, 11612, 12485, 7401, 2842, 12442, 4743, 910, 12320, 3796, 9904, 147, 3239, 12380, 7470, 7577, 9672, 12628, 9794, 752, 6863, 2526, 9516, 576, 11385, 10248, 2915, 12106, 4114, 224, 917, 6407, 7037, 8814, 7826, 4456, 252, 2298, 4228, 12648, 5047, 7578, 4611, 6679, 8548, 12579, 8056, 12326, 2714, 6375, 9543, 6169, 2063, 4803, 8490, 7186, 12192, 6516, 11118, 2361, 12692, 3366, 4332, 3062, 7311, 6232, 3122, 368, 9839, 7942, 1202, 8863, 5160, 10045, 11735, 11133, 2057, 12671, 6267, 10227, 1449, 1105, 4623, 507, 11981, 664, 1932, 5858, 3401, 11281, 10166, 10324, 4672, 11954, 11152, 8003, 973, 1069, 653, 8653, 7307, 11451, 334, 10905, 9899, 3567, 6671, 11094, 6036, 4765, 3022, 7189, 5577, 2773, 8953, 8519, 7638, 10803, 7349, 11352, 12230, 1798, 580, 9389, 1973, 2606, 1183, 5629, 5839, 551, 11255, 7325, 9453, 4050, 8750, 6377, 8678, 1266, 9269, 5407, 220, 11304, 7399, 170, 8468, 3471, 6387, 12359, 9152, 1856, 7084, 5679, 4855, 2004, 10943, 9413, 9391, 11869, 8609, 6115, 11535, 2956, 11264, 6056, 6767, 7614, 1771, 8896, 11060, 4186, 6221, 11132, 3616, 9124, 6594, 9190, 10449, 9621, 10978, 1994, 11962, 8821, 5148, 3956, 11691, 5527, 2740, 4762, 7935, 10927, 3827, 11932, 8015, 1791, 4221, 1455, 14, 12424, 8603, 5752, 3899, 11000, 2661, 3023, 12542, 10673, 2138, 9038, 8544, 7721, 5574, 12543, 8115, 6911, 11272, 8210, 649, 7843, 4903, 6230, 6121, 7083, 9011, 11188, 2224, 3145, 4135, 2570, 479, 7244, 6136, 10398, 11210, 10952, 178, 12508, 7020, 7265, 3913, 10624, 2722, 771, 4621, 2901, 5167, 4956, 4952, 12710, 4912, 976, 119, 10343, 3703, 7528, 1645, 9312, 5084, 6889, 8668, 12223, 5259, 3678, 11128, 6561, 12170, 9466, 730, 8950, 10393, 521, 4600, 5510, 9040, 8366, 6383, 4235, 11055, 3603, 8755, 796, 2711, 6423, 8971, 6247, 2463, 460, 3267, 7160, 760, 8798, 5436, 9524, 4469, 6244, 6598, 3565, 603, 11221, 10777, 5584, 658, 573, 1864, 11518, 9950, 9301, 6697, 5396, 2728, 8810, 227, 10358, 201, 5030, 6715, 3547, 5392, 3124, 7757, 11736, 9486, 11972, 3099, 2152, 7033, 3820, 10272, 919, 6844, 12258, 1614, 4517, 5012, 12076, 2501, 12236, 5731, 5685, 9727, 6503, 2312, 3431, 980, 10163, 6005, 4542, 6243, 6939, 10298, 9369, 1140, 337, 1405, 10388, 9349, 4004, 1753, 4977, 10389, 1036, 3342, 1475, 5675, 9879, 7460, 5964, 7581, 10767, 11533, 12209, 1085, 4865, 6217, 1203, 3863, 3583, 11401, 1012, 6566, 2935, 956, 9184, 9101, 4535, 1963, 3943, 11315, 9339, 1130, 7363, 3535, 10482, 6524, 10075, 10844, 3222, 11917, 8221, 4433, 9245, 3586, 3873, 4346, 11716, 7937, 7513, 10203, 9936, 10351, 1950, 10104, 4737, 11945, 7959, 8291, 9754, 3501, 6124, 6467, 282, 6301, 7955, 2162, 4957, 6593, 5380, 4538, 10962, 2960, 523, 10031, 1555, 12000, 7206, 11899, 2232, 10490, 307, 5495, 28, 2428, 5959, 4682, 734, 7506, 8819, 4502, 9195, 11024, 3677, 2751, 9452, 8965, 6052, 11542, 5278, 12480, 7158, 10558, 5775, 9206, 7422, 2142, 3983, 3528, 1187, 1471, 271, 3897, 8140, 9564, 11318, 1152, 4866, 12228, 3818, 2983, 10842, 8749, 8686, 12745, 4052, 9165, 8812, 10046, 11695, 9832, 4989, 9396, 11491, 9220, 3257, 11313, 2246, 3163, 5470, 9502, 192, 10756, 211, 1587, 1585, 7002, 9310, 4878, 8278, 1028, 12569, 6957, 11910, 4816, 1792, 11577, 9997, 8959, 6279, 8008, 8195, 1930, 2113, 5906, 6216, 7157, 11346, 3625, 10792, 11157, 2672, 4610, 993, 11379, 1410, 3573, 2872, 8605, 9887, 6013, 12352, 12452, 928, 6964, 4444, 7278, 5853, 6674, 3177, 1602, 7056, 4853, 6033, 1095, 11655, 3097, 12286, 7940, 4588, 12191, 12724, 5420, 5872, 3745, 2911, 848, 5319, 8398, 10857, 3433, 12122, 7591, 5980, 5331, 9291, 12099, 7507, 9721, 2037, 2383, 6128, 4176, 9761, 566, 3672, 4976, 2223, 11966, 376, 11438, 8105, 5888, 798, 3001, 10594, 11648, 2295, 8274, 3377, 7465, 10309, 10231, 1446, 877, 1848, 9319, 305, 6249, 12096, 1680, 4163, 10890, 4850, 7790, 6235, 3292, 12557, 7053, 605, 8927, 9784, 5066, 3855, 7547, 10959, 4222, 5730, 11380, 1081, 6527, 6708, 3846, 4053, 1709, 10773, 7374, 9725, 10947, 5051, 10920, 8877, 562, 7069, 3809, 10717, 12641, 309, 8505, 5974, 10568, 7731, 614, 5653, 701, 6186, 10676, 4749, 11213, 5070, 10347, 3298, 1316, 7448, 11034, 568, 2468, 4168, 3415, 3728, 5190, 8665, 10659, 9853, 3828, 5035, 10733, 2888, 5580, 10469, 1330, 361, 1162, 3141, 10113, 2477, 2572, 7498, 6185, 298, 10567, 950, 8201, 4746, 11620, 2379, 6332, 2822, 7319, 4064, 10322, 3211, 3853, 9024, 2242, 6819, 9841, 5065, 12390, 5594, 11042, 4295, 833, 8082, 12102, 9986, 4357, 4065, 5758, 10528, 11368, 9907, 292, 11943, 7521, 9248, 10909, 8687, 5450, 10812, 9229, 9892, 5566, 5210, 11018, 7342, 9180, 2652, 9925, 11669, 8361, 1985, 8500, 12033, 951, 10131, 1102, 5522, 8583, 7876, 11424, 12235, 859, 8147, 5405, 3, 8892, 2479, 10178, 10752, 200, 9286, 6871, 7579, 7588, 9463, 12219, 9728, 1359, 4714, 2378, 7, 2248, 1875, 303, 6849, 3238, 9544, 10008, 6003, 145, 8197, 11758, 7416, 422, 9769, 5710, 11986, 259, 5430, 11088, 7711, 6201, 8145, 9857, 8432, 12403, 8997, 8253, 1685, 7287, 6936, 12028, 1445, 8273, 6639, 7430, 9171, 4612, 7367, 8566, 1332, 3557, 888, 774, 5413, 985, 12531, 11541, 3650, 2959, 11394, 8671, 9376, 7836, 638, 7832, 7501, 12667, 3867, 2010, 8163, 1484, 10432, 3053, 686, 7070, 12668, 1050, 8681, 7830, 11240, 11626, 5458, 818, 12688, 6227, 3434, 9579, 5391, 5866, 9086, 12272, 2760, 908, 2219, 10819, 3688, 7682, 9317, 3993, 3788, 5411, 6127, 6156, 9004, 5838, 4885, 10879, 8101, 6181, 7432, 6755, 11606, 4563, 12318, 2827, 7598, 7045, 1781, 9909, 12656, 405, 12239, 190, 3826, 2507, 9650, 7282, 26, 11215, 2192, 7284, 4844, 7211, 1390, 4211, 5454, 1917, 5969, 12482, 5117, 11343, 1581, 4415, 2013, 12365, 12306, 12254, 9185, 12541, 6309, 738, 4252, 6995, 5007, 10512, 11733, 7580, 408, 787, 1785, 8851, 8241, 2237, 7185, 2909, 3288, 8457, 6887, 5727, 5124, 2693, 5240, 10403, 5543, 1483, 5960, 1907, 8816, 2936, 5264, 3105, 7350, 6741, 2111, 4012, 6016, 6548, 583, 1472, 3924, 2890, 7868, 8447, 5435, 2222, 1526, 10981, 4142, 3778, 10531, 59, 1625, 9504, 5857, 2335, 10925, 10597, 2512, 7352, 5867, 10392, 2153, 6343, 979, 12590, 528, 2466, 12457, 9692, 10373, 11623, 11887, 8849, 6218, 3526, 3237, 8257, 12417, 10833, 9325, 424, 5154, 310, 10694, 11364, 11893, 4047, 1733, 3599, 10434, 3076, 9276, 1782, 10544, 11734, 8590, 885, 11778, 6242, 6139, 2674, 1314, 9192, 9548, 9666, 2009, 10532, 8526, 10730, 9989, 9804, 11403, 5041, 10561, 6897, 8799, 6446, 1493, 9350, 740, 7429, 692, 3044, 7273, 8453, 3957, 2128, 2025, 1627, 3970, 7220, 3203, 6556, 6260, 10707, 4971, 4980, 769, 7107, 5151, 81, 12051, 952, 11277, 3437, 11017, 571, 2951, 9953, 9878, 10850, 5682, 2258, 3419, 2241, 6670, 916, 8016, 6562, 3378, 510, 7076, 9290, 7549, 12427, 5820, 328, 8847, 7615, 1779, 1338, 591, 4190, 297, 12062, 2483, 3593, 1833, 2410, 5647, 1010, 6240, 6677, 5195, 10719, 8301, 277, 4708, 4544, 9027, 3175, 7641, 12732, 2903, 10387, 9612, 10522, 3026, 6804, 7621, 11965, 5568, 12160, 7474, 8738, 9178, 7360, 6019, 270, 1608, 6958, 12473, 8662, 10582, 11852, 2539, 6039, 4381, 2874, 4172, 9477, 8153, 3814, 11425, 3882, 4899, 1715, 11296, 2323, 2175, 8370, 11982, 4758, 2811, 8354, 10199, 3017, 9922, 5569, 4391, 8394, 9607, 6771, 4074, 3790, 11877, 8380, 6564, 9741, 1277, 1549, 713, 8989, 8317, 2344, 1535, 6104, 6158, 8514, 11021, 10296, 10779, 3041, 8923, 6341, 10375, 1149, 10963, 3123, 10681, 169, 6310, 11068, 9244, 9238, 3132, 4547, 4054, 10912, 11755, 8483, 4886, 8005, 7813, 12697, 6943, 6006, 9651, 5598, 12516, 12421, 10965, 4043, 9309, 8858, 2382, 2887, 856, 11634, 1142, 12247, 10551, 451, 1851, 8270, 945, 6168, 5578, 12519, 6883, 4620, 3052, 12314, 11561, 3931, 3952, 12118, 7500, 6308, 8637, 4750, 690, 58, 9720, 2006, 4677, 10881, 11537, 705, 6810, 4261, 10019, 7564, 541, 5946, 880, 3533, 12368, 6993, 11472, 7754, 9696, 2351, 6756, 954, 9963, 10127, 11543, 12295, 7837, 12498, 9632, 2795, 1082, 5680, 9046, 12479, 1286, 1684, 253, 214, 7601, 11417, 9940, 1850, 5176, 2207, 11581, 4440, 6568, 7478, 7967, 7079, 4225, 4370, 5802, 10380, 11307, 6137, 3842, 2123, 1388, 5443, 2905, 1432, 6752, 7253, 9457, 12174, 2955, 12662, 11759, 12280, 2884, 2419, 6653, 12205, 1956, 12367, 2805, 11686, 6559, 4518, 6928, 5755, 545, 9792, 12290, 7249, 2790, 5644, 8765, 1915, 5029, 7271, 4536, 783, 10976, 2534, 5372, 1306, 9499, 3357, 9302, 10065, 12218, 8368, 2781, 11822, 1508, 8502, 5509, 2229, 11613, 9471, 7155, 11202, 7684, 2376, 12717, 10222, 4389, 4726, 6740, 11427, 7014, 12465, 6450, 12693, 1624, 613, 9606, 1117, 10130, 12022, 643, 1862, 2931, 2812, 1539, 4412, 1302, 9158, 1381, 3402, 11582, 7631, 477, 11192, 2088, 2523, 4008, 5475, 7567, 8054, 1257, 12455, 4084, 8762, 4678, 988, 3068, 4799, 1313, 8595, 3162, 10782, 12555, 1829, 11744, 11358, 9928, 23, 8066, 8407, 4026, 4375, 5824, 10404, 5788, 11809, 4400, 1796, 12310, 10625, 1337, 8614, 10893, 1616, 558, 10934, 11511, 2554, 11611, 7489, 12599, 2798, 3112, 11619, 12305, 6829, 4730, 11601, 4729, 11855, 4024, 3064, 6469, 5748, 5817, 6633, 9801, 8260, 6894, 12376, 10776, 1546, 8436, 8535, 5712, 7179, 3368, 2275, 8328, 11050, 7885, 8338, 11480, 5118, 527, 9260, 3313, 11645, 5625, 10266, 363, 11560, 3319, 9637, 7302, 11637, 7998, 5992, 960, 2944, 2125, 4994, 12184, 9390, 4057, 11347, 12340, 1739, 7332, 5995, 1393, 9292, 9714, 7384, 4218, 5661, 9755, 4385, 5751, 6908, 5227, 3164, 4367, 7820, 6836, 5554, 7612, 10086, 7853, 5642, 430, 3339, 10951, 4795, 4205, 1724, 7267, 3640, 4905, 9106, 1934, 3937, 4349, 5767, 7297, 2415, 7385, 3656, 11040, 3739, 11476, 2440, 1408, 3933, 10619, 11035, 3754, 10957, 7035, 3988, 11827, 4465, 6631, 480, 11170, 11731, 2925, 3968, 12042, 11280, 4017, 7289, 12069, 2412, 5880, 4820, 7021, 111, 4814, 3191, 9766, 9418, 10429, 9876, 6413, 11177, 8165, 7292, 2310, 4571, 9895, 12430, 816, 10240, 811, 5238, 4852, 9926, 3954, 12285, 91, 3865, 5701, 1702, 2595, 3773, 11208, 5920, 2826, 9941, 9327, 10875, 2284, 5581, 1984, 8870, 10908, 9902, 2627, 706, 9520, 2488, 1548, 1431, 12582, 5716, 6269, 11680, 6591, 2897, 2398, 10774, 7958, 11008, 11596, 6796, 4733, 11752, 2434, 8596, 9328, 10420, 9231, 6860, 11423, 95, 9531, 9224, 11895, 1090, 10048, 4629, 10026, 9353, 7626, 2514, 12031, 2066, 9901, 4856, 6215, 12337, 1008, 9943, 9255, 8362, 1965, 8513, 8316, 10411, 5244, 10924, 10412, 10146, 486, 5479, 7463, 10462, 5936, 54, 9210, 3198, 6505, 1706, 2788, 3084, 4289, 11608, 5562, 8584, 1722, 934, 842, 10805, 1173, 7990, 2527, 7783, 2181, 4281, 5184, 4813, 10431, 2929, 4641, 2712, 10757, 6661, 5181, 5774, 7425, 9128, 4138, 8806, 12715, 10829, 7666, 718, 6807, 5477, 10703, 5781, 499, 12596, 12050, 7556, 4304, 5902, 9693, 5666, 2266, 8103, 8023, 9072, 11386, 1412, 11980, 10699, 10743, 1114, 3759, 12055, 11319, 10852, 8941, 913, 11292, 5353, 11928, 2961, 3500, 11175, 3352, 132, 6540, 2536, 8754, 2068, 11407, 6573, 9957, 2967, 1960, 3648, 3073, 4399, 3548, 2500, 11579, 11793, 9109, 4121, 6717, 2461, 8326, 2253, 9576, 8901, 3595, 10580, 11349, 1801, 2560, 806, 8479, 2681, 10851, 2569, 2868, 12704, 1167, 12325, 813, 7379, 4943, 2799, 9468, 6841, 5532, 3705, 3684, 7248, 9146, 5988, 8188, 7965, 3841, 2932, 9653, 330, 10, 8340, 7050, 6869, 11306, 3353, 2327, 4691, 11231, 8943, 2764, 10436, 5478, 5143, 9265, 12351, 1362, 7132, 9620, 11383, 4451, 12524, 6738, 569, 9009, 10769, 12212, 7707, 1849, 4592, 9095, 8811, 506, 5359, 7893, 1034, 11661, 12718, 2504, 2417, 2136, 12623, 12738, 5340, 9918, 6142, 2614, 10590, 3541, 5394, 9259, 1360, 4974, 3895, 10212, 4514, 12244, 650, 3330, 5109, 3588, 12592, 11880, 700, 7208, 6154, 5841, 11503, 12321, 5096, 261, 12520, 10118, 1727, 696, 9284, 120, 9833, 3297, 2637, 8550, 6203, 5191, 4715, 1021, 5018, 12098, 8795, 9585, 2134, 6030, 7760, 1158, 1052, 5365, 5157, 2402, 11104, 5309, 8037, 2169, 12495, 1697, 461, 12550, 862, 7146, 1275, 11957, 7596, 6110, 7016, 2952, 231, 420, 9921, 7536, 6264, 7671, 12217, 9980, 10804, 8021, 3702, 4712, 1800, 12014, 8791, 2973, 2198, 2695, 8868, 9584, 5085, 6739, 10525, 10421, 11154, 7882, 531, 8289, 4164, 5865, 10402, 1799, 4105, 7294, 11804, 3602, 1464, 9591, 4154, 7647, 4664, 11279, 11489, 12331, 1713, 12630, 8696, 12090, 10284, 10230, 10155, 10754, 6910, 10892, 8549, 1265, 1043, 3416, 11588, 5247, 10183, 10067, 10588, 10645, 7099, 5622, 6672, 10215, 11148, 6588, 5996, 11183, 9849, 7034, 5910, 1579, 11959, 11439, 593, 484, 1494, 5926, 8655, 5646, 4564, 2698, 11108, 3540, 12066, 4238, 10672, 11260, 4325, 2457, 6800, 3196, 6862, 10817, 5350, 634, 12586, 7176, 7582, 11662, 12120, 12494, 12266, 10550, 8604, 2934, 5403, 9331, 8390, 3560, 6246, 2737, 5360, 7777, 1411, 546, 5725, 3223, 1442, 9916, 9748, 12011, 6153, 785, 10849, 10793, 3286, 6531, 12510, 8290, 11178, 12472, 10076, 12642, 9315, 10280, 10755, 1660, 6080, 635, 5382, 791, 4870, 8995, 11493, 4073, 2612, 10712, 1132, 4940, 4352, 1686, 5441, 1933, 12461, 8355, 8028, 538, 2748, 7275, 10250, 4690, 9617, 2632, 7562, 8001, 3463, 10314, 12408, 7977, 9820, 3444, 4251, 7387, 9274, 9412, 4864, 6382, 7225, 11955, 9144, 1886, 7213, 42, 8657, 10147, 12700, 559, 10702, 2563, 1897, 3208, 694, 7286, 4960, 4511, 11751, 8922, 4013, 11963, 4709, 586, 1639, 7138, 2854, 5164, 354, 4144, 3553, 9711, 9582, 2094, 5626, 10349, 1747, 6481, 1609, 10352, 6177, 11283, 6747, 4504, 11761, 1278, 2585, 532, 319, 808, 4861, 3059, 3972, 9663, 7554, 1558, 11484, 6537, 11636, 8443, 11505, 4902, 11122, 1379, 8778, 7976, 1298, 9039, 3412, 3808, 5059, 8988, 5944, 3716, 4327, 10854, 11689, 5455, 1518, 385, 624, 7095, 8462, 5061, 12194, 8176, 6196, 10663, 9409, 2366, 7917, 2218, 3229, 7359, 2195, 2261, 9594, 6010, 7780, 2624, 4917, 11372, 7131, 5053, 8615, 4893, 1044, 8434, 12049, 4434, 6321, 6994, 5429, 12606, 10607, 10245, 2533, 5339, 8261, 12554, 12379, 6474, 3115, 5589, 1744, 12439, 6130, 2726, 11664, 1761, 1674, 6195, 12002, 8310, 12549, 10479, 8040, 3542, 7190, 4954, 9858, 3893, 10855, 5287, 5262, 6200, 10184, 10780, 637, 9455, 5766, 3564, 7196, 2980, 9983, 10526, 12012, 820, 11665, 9870, 2160, 11052, 12546, 7052, 3226, 4362, 5024, 2453, 5720, 3629, 1161, 9730, 12684, 3405, 6500, 2545, 4229, 1632, 5234, 1025, 11840, 3213, 9538, 4683, 228, 11859, 5552, 188, 3051, 2816, 11776, 2516, 7851, 1215, 1208, 11445, 5299, 8098, 2073, 6511, 781, 4120, 7743, 11540, 11391, 9490, 3660, 2201, 3635, 2031, 1524, 12087, 10635, 12631, 3018, 7392, 2565, 478, 5074, 3318, 9900, 318, 4051, 11624, 6482, 7509, 10726, 161, 5676, 5714, 6172, 8746, 563, 6974, 8413, 7773, 3689, 9676, 247, 2619, 2804, 12537, 9223, 5804, 2235, 11426, 8074, 1977, 4900, 10150, 11116, 6296, 10334, 8437, 9558, 6135, 5393, 248, 3470, 8431, 8400, 7652, 99, 9893, 2054, 5904, 8878, 2421, 1852, 8059, 3199, 2490, 5786, 9757, 4262, 7532, 283, 8085, 7719, 5668, 5338, 4630, 12514, 5163, 8193, 10770, 12708, 5831, 4741, 12600, 9437, 4591, 729, 2640, 11938, 4020, 1125, 10827, 94, 11418, 10718, 7989, 4798, 3249, 7575, 5098, 4599, 5220, 1947, 4697, 9599, 12150, 5572, 5153, 3720, 5402, 1358, 4970, 12227, 11756, 7119, 4906, 778, 11683, 12664, 8200, 5228, 144, 5523, 19, 9890, 12618, 1363, 687, 7834, 1618, 10942, 4728, 9601, 6276, 9322, 4413, 7643, 12005, 11780, 12262, 7495, 7276, 12679, 2920, 4336, 9384, 6390, 8160, 11366, 9110, 6917, 815, 7115, 1742, 2730, 7610, 9826, 10825, 9874, 2210, 416, 12423, 7408, 1708, 2745, 2749, 4045, 4173, 4867, 11550, 7337, 65, 4944, 11144, 1952, 5048, 7485, 8663, 4832, 8006, 618, 7824, 10219, 10330, 2404, 8327, 8597, 8520, 7281, 5001, 5042, 1119, 1545, 6122, 7611, 3611, 12729, 1457, 12398, 8850, 4356, 5691, 2689, 10233, 10880, 5442, 12441, 3462, 10742, 8239, 1991, 7996, 4704, 7283, 1732, 9017, 9546, 12677, 3008, 4393, 577, 7741, 9992, 10929, 80, 9660, 3065, 6882, 4366, 11268, 11842, 9005, 2763, 12265, 5031, 11259, 6384, 8601, 12116, 11740, 4668, 7042, 6656, 8409, 2143, 1148, 6587, 11897, 11823, 400, 8669, 3168, 1288, 8642, 366, 9445, 2524, 12332, 3946, 6440, 2226, 9436, 1794, 3921, 8720, 11022, 8843, 7396, 11854, 9442, 2990, 4531, 3877, 9051, 242, 3801, 4578, 321, 9819, 4493, 7910, 987, 3265, 4302, 9421, 11404, 9962, 6749, 5967, 12117, 5296, 1235, 6358, 237, 222, 2537, 1018, 12105, 5387, 10560, 11777, 3647, 5697, 9906, 4354, 1827, 3472, 8222, 1163, 6219, 5604, 6402, 3483, 3312, 241, 1371, 7377, 9589, 560, 12686, 747, 12249, 11123, 8506, 4129, 12362, 4508, 949, 9739, 12437, 2001, 4293, 279, 2174, 10958, 7388, 11746, 10122, 6750, 6528, 6645, 9846, 3078, 4326, 8546, 7027, 6949, 6828, 3561, 4343, 8067, 3420, 7103, 10010, 7159, 346, 9894, 10216, 12277, 4661, 4311, 3977, 12291, 9083, 4698, 2408, 8152, 10599, 10992, 6159, 3250, 5816, 9407, 8516, 1420, 5094, 9419, 7393, 11461, 8565, 4936, 1765, 9451, 10317, 4736, 9688, 2852, 6123, 3744, 4147, 10972, 552, 12216, 6494, 3009, 7026, 972, 11074, 8424, 10475, 3464, 7543, 4237, 4634, 2046, 254, 11112, 4282, 7187, 4204, 8537, 7931, 11273, 7288, 5989, 2322, 5958, 493, 9580, 8320, 9960, 8100, 12018, 2487, 1846, 529, 7571, 9771, 10492, 3571, 12665, 1084, 7494, 6746, 5379, 1644, 2600, 8395, 11818, 11950, 7168, 6392, 1007, 7558, 9815, 11800, 2216, 10696, 1913, 1560, 1375, 6205, 1068, 5823, 2364, 7061, 2768, 9975, 7776, 12629, 3193, 1748, 10715, 10596, 4315, 6015, 8530, 10634, 8539, 6305, 5882, 4747, 12463, 4112, 3730, 1720, 3385, 6570, 6206, 12003, 606, 5058, 1046, 6557, 12125, 5897, 4835, 9616, 10435, 3180, 7678, 10865, 4198, 8844, 7842, 12553, 9863, 4199, 4922, 1982, 9073, 2433, 3574, 10423, 9966, 8164, 5103, 4806, 6924, 8349, 7024, 2281, 4474, 4317, 4982, 8740, 688, 9717, 1305, 5416, 7184, 533, 5193, 3200, 5198, 11253, 12245, 3194, 4779, 11961, 5539, 5119, 6245, 11501, 4392, 7693, 9338, 8280, 6762, 8717, 3479, 5174, 679, 266, 10516, 12447, 8636, 2862, 10725, 4617, 10029, 4879, 2497, 7066, 11370, 7841, 741, 6151, 5983, 1556, 11467, 12577, 4587, 1038, 4739, 10990, 3016, 5890, 1380, 7987, 8024, 9611, 779, 2313, 7607, 9956, 10705, 9596, 12369, 7969, 1496, 1365, 3734, 9775, 10165, 3445, 9084, 9575, 3279, 304, 4284, 6284, 4093, 3838, 746, 8283, 8823, 3558, 3751, 9037, 9103, 12370, 2194, 4243, 1017, 196, 5719, 684, 3497, 11873, 10226, 11293, 10289, 8481, 1728, 2666, 11362, 2880, 9304, 397, 5342, 3383, 7888, 1481, 7879, 4572, 1270, 3955, 11764, 8867, 10527, 9107, 2892, 3481, 8397, 2159, 10115, 8230, 3418, 8156, 8463, 7081, 167, 12449, 7574, 5559, 6256, 3158, 5125, 4019, 8531, 3671, 1927, 11696, 895, 4895, 7109, 8052, 6114, 10496, 8829, 10154, 7105, 3661, 12189, 11711, 4271, 3538, 6166, 6986, 12649, 6178, 9722, 8412, 172, 3966, 3575, 1714, 11722, 1531, 4275, 6763, 4773, 8445, 8743, 901, 130, 11455, 10051, 3802, 9563, 1340, 4925, 5236, 8761, 722, 1307, 11828, 6945, 11020, 8308, 6149, 7655, 8091, 5067, 5609, 10083, 1459, 10759, 4670, 8815, 8379, 884, 7793, 12558, 3391, 12361, 9003, 8946, 1396, 5945, 727, 5901, 6007, 2365, 3600, 2636, 10498, 11989, 8875, 2953, 3927, 11135, 4701, 11378, 4583, 7202, 10627, 8839, 8680, 10778, 9236, 370, 2950, 11356, 10069, 757, 8781, 7404, 12634, 2800, 10341, 11673, 1908, 265, 8759, 11479, 7402, 4031, 12300, 9699, 6497, 1263, 766, 4533, 5619, 12299, 7736, 11069, 10466, 6000, 3825, 11100, 8135, 9133, 2011, 944, 82, 11454, 10495, 10592, 8802, 2146, 11361, 10521, 3789, 5285, 736, 12282, 11896, 8249, 9768, 629, 8375, 7177, 1527, 6847, 12731, 11117, 10414, 7262, 11141, 11299, 12019, 2041, 11220, 5009, 11706, 8774, 11305, 6853, 2221, 11003, 3047, 3398, 9955, 10845, 9275, 835, 5982, 786, 5011, 9116, 9230, 1661, 12581, 3906, 12067, 4997, 10939, 10179, 5871, 9449, 11323, 12169, 3909, 6388, 11573, 3370, 12547, 12374, 1649, 5984, 8938, 8242, 5485, 3489, 8945, 2315, 4063, 3896, 6208, 12639, 1430, 4557, 7165, 2080, 6001, 11204, 8482, 2188, 9615, 1504, 9176, 3309, 3919, 599, 5302, 6306, 10304, 2547, 12712, 6465, 293, 12696, 11258, 5199, 7616, 5736, 5306, 3015, 12278, 6179, 3514, 406, 8315, 7960, 12659, 3794, 380, 12132, 7361, 7403, 8226, 7487, 2535, 4003, 10000, 2814, 4552, 10390, 2628, 4406, 3359, 9446, 5607, 10319, 10548, 1040, 11490, 5178, 7897, 3642, 9047, 10135, 10287, 826, 12119, 3461, 8617, 8957, 6133, 10217, 4659, 10190, 5779, 570, 3653, 10806, 4639, 5358, 11457, 10932, 12385, 8196, 5355, 943, 5170, 3546, 4948, 7936, 12124, 8430, 10848, 10242, 3245, 11850, 3442, 5553, 4476, 8960, 4101, 5768, 1517, 10132, 9541, 11246, 12391, 1655, 7644, 9137, 8143, 11234, 10208, 6716, 11153, 10060, 12366, 10109, 10565, 4461, 3089, 10545, 2359, 1651, 1244, 8107, 11874, 11678, 5364, 11659, 2105, 11738, 8118, 4148, 12037, 1512, 958, 12511, 8178, 9397, 4338, 840, 10948, 2482, 10996, 8111, 3839, 12141, 11590, 11190, 10367, 1321, 1528, 10980, 1197, 4470, 626, 789, 10700, 10747, 12538, 3830, 2205, 12730, 10292, 1903, 452, 7175, 7595, 9381, 10097, 561, 5590, 7823, 10621, 2393, 11674, 9380, 5846, 2211, 10264, 8914, 3491, 2680, 6171, 636, 6934, 5308, 177, 11995, 5021, 10563, 43, 5215, 1006, 6140, 6712, 2561, 2831, 11974, 4368, 5898, 4653, 4793, 6629, 7348, 8932, 8120, 11698, 12574, 6017, 3007, 3669, 4528, 7566, 9634, 11452, 1730, 12146, 5749, 896, 10476, 5196, 10232, 3975, 8887, 7017, 5575, 1509, 11016, 10182, 2869, 1619, 4882, 2898, 7656, 1641, 9889, 8080, 10070, 2927, 10005, 2551, 11478, 6544, 4431, 922, 12740, 10385, 11005, 8577, 10967, 6399, 3487, 9648, 7635, 3282, 6665, 4360, 9235, 12605, 8673, 3304, 7676, 8970, 7497, 3883, 10998, 11826, 6268, 5976, 6930, 5203, 12259, 8070, 4819, 11276, 9842, 2962, 6947, 3393, 8611, 1373, 7718, 6641, 12145, 9282, 5986, 6965, 12276, 9249, 4467, 5438, 3329, 11405, 966, 3592, 5939, 7071, 3738, 572, 9867, 4975, 4419, 9252, 3726, 3088, 6027, 5997, 4825, 5169, 10012, 10379, 8885, 11670, 10079, 1487, 6794, 115, 3217, 7811, 10595, 6688, 4751, 5305, 11007, 10968, 9363, 5303, 1866, 5938, 765, 10103, 2719, 7746, 6554, 2559, 5112, 3922, 5588, 2436, 2671, 6962, 11115, 2027, 3246, 5197, 799, 8170, 11676, 517, 2599, 8460, 3577, 6772, 12147, 6054, 10206, 11278, 10265, 1972, 268, 3028, 2844, 1734, 3834, 7771, 6980, 1355, 11449, 6350, 1030, 12293, 9759, 11466, 2091, 4130, 1698, 6366, 10746, 883, 2576, 3792, 6977, 9530, 3539, 11166, 4274, 2251, 5916, 8718, 5111, 12357, 7338, 374, 4079, 2120, 5993, 7679, 2779, 7440, 11785, 5293, 3087, 3963, 6085, 7805, 5486, 155, 1451, 9507, 11845, 12507, 9862, 6596, 5632, 9719, 10831, 5887, 1271, 4361, 1912, 10928, 4174, 7122, 4986, 6170, 7279, 9001, 5732, 11513, 3407, 2870, 8133, 10867, 494, 10847, 4848, 6634, 6826, 12382, 4040, 8709, 2492, 6780, 6082, 8392, 12064, 6621, 8756, 7744, 9344, 3707, 9706, 3632, 7201, 7236, 5307, 6662, 9823, 11482, 1407, 6454, 11001, 9359, 9491, 4984, 3268, 4160, 12523, 2485, 1341, 1333, 11271, 3090, 9783, 2171, 5337, 7312, 449, 1469, 9120, 11789, 1521, 10369, 10933, 8651, 3182, 2431, 1654, 4857, 33, 9478, 8616, 10668, 2975, 3644, 2065, 9608, 716, 7060, 9008, 9745, 4382, 4194, 8307, 5536, 3890, 10809, 1463, 10519, 5833, 5376, 782, 5987, 3230, 8324, 3563, 2918, 647, 86, 11139, 1910, 112, 8294, 6765, 5567, 11056, 2520, 7785, 2206, 11456, 6094, 6106, 10821, 8217, 4276, 10467, 8244, 6463, 11902, 8991, 1120, 9149, 2034, 11584, 1342, 10698, 3035, 6997, 7019, 2735, 2665, 12283, 5086, 5907, 9560, 2699, 3736, 10301, 6143, 2106, 7517, 9080, 2035, 11824, 9562, 12257, 5214, 11625, 7943, 2148, 3116, 10895, 1918, 2839, 964, 24, 2368, 10395, 7318, 10502, 62, 2370, 9537, 10860, 7112, 7844, 12739, 9108, 1477, 11061, 7437, 3263, 3836, 11525, 3612, 10044, 11252, 5861, 9932, 11851, 8213, 12573, 4448, 9156, 4626, 3627, 11787, 6988, 11216, 8408, 9691, 8499, 8168, 7799, 6241, 10117, 1832, 7589, 6811, 1893, 4478, 11958, 5985, 9362, 4516, 652, 8708, 4525, 6109, 9493, 585, 6117, 7816, 4316, 7335, 2276, 6214, 12528, 163, 11574, 4115, 11782, 3621, 379, 3749, 2521, 10437, 9212, 11621, 7924, 4384, 12009, 4810, 1488, 3156, 10169, 3556, 2129, 52, 8332, 1199, 6412, 1088, 4104, 604, 3219, 11921, 9121, 5152, 9029, 5256, 6695, 10846, 312, 8979, 325, 2590, 554, 12409, 4760, 10050, 11510, 860, 2493, 4329, 6552, 12737, 3085, 11725, 1607, 7624, 7559, 8568, 7861, 7065, 5929, 10864, 7815, 11099, 11762, 10986, 8474, 6751, 5738, 9207, 1482, 6878, 8127, 912, 6150, 5440, 6761, 5139, 7933, 10744, 1622, 1303, 10708, 7317, 3663, 10453, 7905, 10830, 8573, 2801, 1696, 2945, 1954, 9263, 12091, 4156, 5810, 11558, 12269, 10684, 11164, 1949, 5799, 8426, 4696, 3628, 8181, 4646, 11705, 3345, 3712, 41, 153, 6704, 7552, 5036, 3987, 8994, 8639, 3126, 10541, 3631, 12650, 3425, 2731, 9802, 5585, 495, 5914, 6603, 8493, 2942, 7205, 3326, 3167, 1969, 8444, 240, 6480, 6119, 1143, 6907, 7878, 10995, 4178, 1234, 2957, 11544, 2809, 4717, 6615, 10384, 5688, 762, 8731, 11975, 2277, 2923, 3576, 6783, 1746, 1876, 1946, 12448, 1813, 2078, 3582, 4407, 8228, 4410, 7137, 6642, 9570, 2803, 7784, 3113, 9528, 4907, 5254, 8475, 11916, 1788, 3713, 10062, 12466, 10808, 9459, 10556, 8962, 3422, 7633, 8187, 10066, 3273, 8818, 142, 1091, 5998, 12084, 1540, 12196, 3651, 11475, 8312, 11089, 3290, 6155, 3149, 15, 986, 10894, 11990, 7431, 1872, 4267, 11309, 7895, 1999, 5794, 7586, 9377, 7351, 8906, 4141, 9168, 2531, 3214, 2697, 9667, 9668, 2544, 10923, 3727, 2522, 11496, 2616, 4826, 2367, 3905, 7242, 9649, 106, 6953, 1416, 3264, 4839, 6495, 7238, 10499, 676, 3373, 5963, 9991, 6408, 206, 10543, 7664, 432, 11205, 11066, 6560, 10009, 8417, 5032, 4774, 3294, 6820, 4631, 64, 1138, 3135, 4718, 10456, 11396, 7178, 5698, 10530, 6659, 11572, 5905, 7523, 10028, 1233, 10455, 1491, 10022, 11447, 11162, 2155, 10911, 11571, 7673, 1217, 5348, 2794, 5724, 8268, 3456, 12620, 4680, 5211, 10716, 8406, 12566, 11262, 4007, 3516, 2282, 1240, 158, 4, 8899, 2414, 6266, 1334, 3908, 4383, 2387, 9519, 4636, 3150, 5235, 11563, 7166, 314, 7999, 6485, 792, 3392, 8358, 2691, 4660, 11667, 11064, 8150, 3843, 3737, 3235, 4648, 10180, 3189, 1840, 11911, 11126, 2643, 8372, 7847, 8620, 11290, 3579, 10608, 2283, 5924, 3928, 8093, 8728, 935, 5940, 890, 1766, 5538, 9090, 8232, 4959, 18, 10318, 9474, 9566, 9595, 175, 12388, 9590, 4545, 10868, 89, 5734, 7657, 9702, 4725, 7127, 1770, 57, 8079, 4769, 8773, 2022, 7956, 12083, 8130, 4568, 6395, 8477, 9100, 6401, 8469, 10786, 12572, 8039, 630, 5078, 6998, 5829, 6046, 8297, 7551, 12438, 10689, 6449, 4595, 12041, 4699, 4001, 12197, 8764, 2992, 2625, 8560, 4193, 2841, 10214, 7054, 6654, 1690, 1767, 12564, 7738, 7044, 7285, 9153, 1252, 11471, 5769, 12658, 8719, 7228, 1979, 4206, 11340, 12722, 4175, 6666, 2019, 3543, 9458, 7355, 11093, 1831, 12345, 11495, 4103, 4719, 1031, 689, 9852, 8425, 2810, 10142, 10333, 1743, 4609, 9645, 4753, 4607, 4244, 726, 4140, 8869, 7789, 2499, 7280, 1394, 8403, 8832, 2289, 4711, 8830, 9873, 4048, 12493, 3435, 10601, 772, 4930, 409, 11168, 9218, 7466, 4443, 11247, 8420, 1580, 3698, 4818, 8019, 830, 5638, 5873, 6935, 7689, 8083, 5332, 2552, 3831, 1072, 1557, 7859, 2907, 1704, 7951, 2692, 10576, 331, 8421, 3667, 6307, 8855, 3811, 1653, 5233, 10616, 6649, 5016, 2622, 11314, 947, 9423, 7443, 8018, 5953, 11527, 5216, 6850, 9354, 11607, 8077, 394, 3959, 3371, 9204, 12683, 5033, 4834, 417, 2991, 3323, 12176, 7708, 1129, 8876, 6919, 3652, 10979, 10887, 5546, 2922, 6989, 3270, 12462, 4967, 11441, 12074, 6574, 8592, 5815, 5561, 2096, 455, 7835, 2564, 4432, 9186, 4526, 5620, 8494, 490, 3674, 9375, 6457, 12705, 74, 1261, 12327, 6681, 9791, 6369, 2670, 4790, 6607, 11671, 10084, 8551, 2843, 141, 5930, 11767, 5020, 8527, 995, 1013, 3409, 3916, 6107, 609, 10910, 8266, 7675, 5311, 11516, 8321, 3708, 5677, 11251, 2754, 10940, 1738, 5492, 8339, 9015, 5121, 2878, 11229, 6265, 6719, 11320, 135, 9273, 8828, 7945, 4094, 5968, 9059, 6472, 6275, 6491, 6435, 3454, 3308, 10944, 8000, 11971, 2859, 10753, 4627, 11145, 466, 3003, 5717, 7555, 6278, 8365, 3050, 9088, 1659, 11181, 810, 11934, 9079, 8528, 1586, 12487, 2067, 11876, 9439, 9715, 872, 627, 3277, 10921, 9680, 8954, 11763, 11149, 11628, 777, 8732, 8385, 5101, 6637, 5927, 10511, 4215, 12464, 2821, 11598, 6861, 2532, 3941, 5892, 6337, 8218, 12483, 12313, 3280, 4344, 8589, 7767, 3856, 9561, 3157, 2471, 1422, 1456, 805, 7291, 1182, 4992, 4210, 11639, 801, 4058, 3549, 2778, 7803, 11976, 7920, 256, 3821, 4011, 1967, 11914, 5491, 11044, 8598, 134, 7546, 9785, 8882, 11134, 2696, 5672, 941, 10970, 6430, 9228, 4133, 5795, 923, 9373, 12121, 5034, 7215, 6421, 3159, 11217, 10985, 3083, 5077, 7557, 1218, 12115, 7568, 12451, 4021, 3512, 5743, 4378, 3332, 180, 4871, 2989, 1064, 10041, 9179, 1575, 3805, 9535, 11891, 10938, 5506, 11901, 9294, 2426, 2584, 2988, 3529, 3356, 2131, 1810, 1329, 8258, 10646, 4489, 924, 2668, 4756, 6622, 11889, 898, 6660, 10164, 2762, 10750, 12392, 4985, 9456, 10504, 3610, 2178, 9837, 7481, 10428, 6664, 7475, 10450, 8007, 2399, 1224, 8852, 10589, 1254, 9812, 1145, 703, 1473, 2542, 2678, 2090, 8825, 7745, 1014, 11460, 5044, 10936, 6946, 817, 10837, 3414, 6846, 3429, 7086, 10839, 2182, 11294, 10783, 4628, 814, 3299, 3325, 3142, 4305, 7199, 10866, 721, 6112, 10896, 8382, 6814, 3961, 2293, 12560, 1735, 1441, 2846, 2834, 10609, 4076, 3348, 12728, 12404, 3870, 9552, 11956, 4953, 4483, 2875, 6970, 9786, 1437, 8022, 7480, 8345, 9880, 9628, 9201, 6706, 9203, 11991, 1892, 10110, 5800, 465, 3495, 4845, 3452, 9435, 10810, 3509, 7415, 7073, 10610, 3072, 9208, 11502, 4880, 1992, 1189, 3096, 8714, 5260, 351, 9654, 6228, 10329, 8360, 3478, 11927, 10224, 2043, 2769, 7720, 2993, 3384, 2667, 11715, 36, 10922, 12187, 9438, 9605, 1474, 10014, 11354, 10859, 10119, 1668, 8208, 12063, 1712, 12703, 12024, 5801, 7057, 575, 10120, 3859, 2877, 5499, 5726, 8123, 3328, 1729, 12107, 5605, 10570, 7808, 12339, 4314, 5108, 11169, 8793, 12104, 9866, 11462, 7981, 8087, 8212, 2921, 2126, 1962, 4771, 8698, 5694, 9187, 8206, 1264, 2857, 5279, 8112, 6575, 10510, 4731, 4328, 6628, 118, 6680, 7681, 6605, 589, 10247, 5447, 5870, 9939, 12060, 9334, 3769, 1629, 5175, 3800, 4233, 11685, 11398, 12221, 9387, 7063, 11250, 4365, 8410, 9378, 4889, 9749, 4505, 8060, 6198, 4874, 9492, 3045, 7978, 11924, 11638, 6549, 12694, 12144, 8679, 10011, 8594, 1996, 5762, 2314, 4463, 1993, 6597, 2225, 9196, 4018, 6520, 4800, 5742, 12521, 8895, 8441, 3139, 4888, 4522, 11875, 800, 166, 1170, 6961, 1895, 9267, 2686, 9551, 12372, 8333, 6224, 2265, 7015, 3302, 9912, 2618, 1402, 1397, 8973, 2702, 6764, 7698, 6876, 8917, 4650, 8907, 5217, 12082, 9597, 11184, 702, 6049, 9877, 12109, 8179, 8467, 1097, 4581, 7195, 12346, 12092, 6822, 3798, 10732, 8343, 12443, 3183, 2815, 12687, 1764, 3938, 5576, 12565, 3860, 768, 12149, 7126, 3104, 670, 10537, 6929, 8627, 6044, 5962, 8121, 11101, 3128, 3255, 10088, 3024, 11920, 7875, 5226, 1433, 5015, 2836, 10701, 5006, 9917, 5979, 9111, 6777, 12226, 11724, 2397, 9253, 1522, 8758, 7573, 861, 1294, 4230, 6888, 911, 555, 3725, 1438, 6909, 6300, 7140, 1628, 9485, 3447, 6627, 1637, 11514, 6530, 6676, 9022, 5796, 4622, 3426, 10442, 7382, 1891, 136, 12614, 7427, 7156, 1841, 6286, 1269, 10089, 8650, 9678, 5064, 471, 7324, 5628, 9443, 7499, 5090, 9077, 12397, 75, 9287, 12633, 7461, 2263, 12317, 11925, 2369, 4692, 6295, 3709, 4565, 4482, 9082, 10724, 2974, 10356, 3455, 1204, 4836, 10053, 8250, 2239, 8963, 9127, 2307, 4292, 11939, 6431, 7725, 2690, 1936, 8670, 9824, 8161, 4550, 3192, 3699, 3080, 1079, 1777, 11124, 7649, 4226, 9521, 3327, 3717, 4179, 185, 3151, 10158, 12279, 5583, 1378, 5525, 1762, 5764, 4248, 7974, 10056, 3143, 6781, 11435, 10054, 12081, 2257, 7088, 8998, 10937, 10037, 4586, 5776, 8956, 9398, 11567, 1070, 4657, 7667, 7862, 8586, 1009, 7900, 7550, 7049, 10935, 2050, 6257, 8933, 10820, 780, 7613, 125, 11082, 5325, 7848, 5269, 7576, 11603, 4594, 10515, 6973, 10557, 640, 7660, 9268, 11195, 3137, 3723, 6736, 10989, 1959, 6032, 11336, 8296, 6466, 7812, 11269, 7232, 12338, 5221, 9522, 8433, 715, 10444, 11497, 3784, 9795, 8215, 7315, 2255, 6835, 10004, 4127, 9410, 2662, 1274, 1476, 8061, 1695, 10410, 2362, 121, 5425, 5943, 6071, 10826, 4219, 12415, 11600, 5290, 8470, 9036, 8838, 5893, 7714, 4454, 7328, 1638, 4876, 4824, 4784, 871, 8183, 7370, 4373, 505, 8654, 4339, 4778, 1896, 8031, 9500, 204, 8767, 7371, 68, 7866, 10061, 11604, 12020, 8199, 11741, 1718, 6312, 8114, 10425, 12154, 6827, 6, 4150, 160, 9045, 8983, 502, 11554, 7899, 8134, 3109, 9057, 1705, 7102, 12611, 657, 6565, 3847, 1939, 5219, 5655, 6512, 5654, 10167, 378, 343, 4067, 6320, 6386, 1804, 344, 7880, 5635, 5409, 2556, 2296, 2793, 5850, 10294, 8944, 9469, 1511, 8964, 3904, 3102, 8119, 592, 8771, 8207, 6188, 9830, 8096, 2045, 11803, 6638, 1658, 1763, 11720, 10823, 4809, 1877, 6226, 5367, 1156, 3803, 9233, 6609, 9695, 6877, 11890, 8949, 5083, 9999, 9669, 7765, 11721, 1550, 4286, 1166, 11140, 6138, 9840, 7025, 467, 2494, 4647, 4941, 1749, 11786, 3082, 11226, 5310, 9333, 5875, 8800, 5386, 1447, 2853, 5745, 10112, 9448, 2285, 5095, 12085, 4608, 1258, 9577, 5842, 9567, 10176, 463, 8180, 11992, 11308, 1538, 11979, 11783, 4029, 3666, 10801, 10529, 7323, 3686, 6975, 10174, 8034, 3947, 8182, 9172, 2044, 10225, 11459, 12604, 1369, 9341, 4376, 669, 1324, 7263, 12567, 2775, 3134, 10748, 9361, 1059, 11996, 10200, 2353, 7362, 12471, 9365, 6692, 1976, 10246, 1049, 8824, 8449, 6193, 1460, 11291, 5937, 2675, 11143, 87, 5533, 5715, 3850, 5649, 3423, 4466, 4100, 2470, 6233, 5541, 10267, 61, 5530, 1847, 2517, 2271, 1255, 9125, 8416, 10391, 2767, 4860, 6526, 7407, 3996, 9627, 9374, 6838, 4580, 11200, 6406, 2196, 6455, 4247, 7234, 8784, 44, 8476, 6238, 5460, 6427, 807, 4446, 3055, 10091, 5330, 3767, 472, 12699, 9289, 3768, 3641, 11907, 9977, 8004, 117, 4330, 6535, 3597, 3031, 11926, 8999, 5489, 2785, 7921, 5508, 10536, 12173, 7091, 6567, 704, 7058, 3985, 9683, 12157, 3427, 4255, 8935, 3531, 11408, 2705, 9716, 187, 4840, 7306, 5777, 8139, 9665, 1880, 6285, 11569, 8645, 2074, 9189, 2203, 1693, 3029, 5243, 11092, 9000, 8859, 3750, 12499, 7526, 11353, 468, 10941, 2651, 11481, 9326, 7934, 5596, 5911, 11224, 3374, 3746, 7441, 1968, 10862, 9643, 3762, 401, 11589, 9647, 4182, 3833, 11470, 6941, 6770, 5013, 2746, 8335, 215, 5884, 7605, 5648, 11964, 3694, 4966, 3978, 5245, 9738, 427, 10899, 5811, 6584, 10331, 3300, 8373, 697, 9549, 8925, 11712, 9183, 9526, 8458, 7858, 9935, 9891, 11442, 9640, 6787, 1839, 9799, 11468, 9559, 7838, 10394, 1906, 9145, 2092, 5548, 2115, 8702, 6693, 12355, 7697, 9028, 6546, 12204, 12636, 1621, 9182, 5322, 6898, 8048, 11191, 1100, 5570, 12580, 3545, 3251, 8602, 3129, 5129, 10315, 10213, 3397, 1828, 598, 12129, 7454, 11969, 10123, 8969, 10382, 5473, 12419, 1346, 3799, 9624, 3071, 8854, 7013, 4427, 12657, 12073, 12136, 5046, 4551, 7930, 10771, 10728, 9007, 10220, 12733, 5868, 7661, 8534, 4000, 5915, 10651, 11862, 6499, 12075, 173, 12673, 1498, 7067, 12071, 4892, 5186, 5281, 10278, 12484, 475, 749, 5255, 6486, 11538, 8045, 84, 2147, 5437, 9401, 9064, 1353, 4253, 10244, 7111, 616, 492, 2654, 72, 4656, 894, 11230, 7529, 1818, 6857, 10433, 7709, 645, 1398, 7250, 12402, 7794, 4006, 1811, 7855, 8512, 1584, 2881, 12652, 9337, 2761, 273, 5397, 3891, 8570, 4881, 1048, 295, 8607, 6650, 7938, 11448, 9043, 8064, 4321, 751, 10381, 12515, 1582, 3054, 7683, 11568, 7544, 30, 11059, 3807, 8628, 8647, 971, 4480, 4757, 12489, 4116, 5321, 2596, 3532, 10555, 6248, 9295, 9243, 4209, 8699, 5814, 10035, 11701, 6813, 8769, 1423, 6182, 7339, 6290, 4398, 1865, 841, 9202, 12661, 5616, 776, 1470, 8288, 10631, 7143, 10615, 12500, 8948, 3307, 11635, 174, 10064, 11536, 10960, 6976, 3048, 10891, 4110, 564, 12260, 10633, 10983, 11109, 2970, 5390, 6197, 10652, 12744, 2491, 454, 7892, 9690, 398, 11903, 3584, 3190, 1071, 12676, 3403, 1297, 5809, 9174, 8350, 404, 6617, 1599, 4884, 3732, 4301, 12232, 6065, 4675, 9619, 1503, 9329, 11923, 7321, 2411, 3984, 10692, 7691, 60, 3607, 2797, 5347, 974, 6398, 10863, 1485, 2481, 12506, 10977, 4429, 2648, 2334, 2930, 2190, 9467, 10661, 7397, 1042, 7326, 12101, 10378, 2199, 7864, 5848, 3060, 6942, 4111, 11222, 10904, 7972, 3537, 4420, 6614, 6984, 5645, 5412, 8137, 1087, 5144, 3797, 6069, 4964, 11342, 2474, 5357, 358, 7136, 8987, 5573, 12108, 3119, 12709, 1569, 7992, 12348, 849, 1345, 4655, 9261, 6011, 8532, 6355, 5544, 7696, 7032, 2985, 6710, 3706, 4472, 12405, 5990, 4403, 4767, 8780, 1711, 8155, 5951, 9513, 2688, 7251, 10161, 3233, 6987, 3004, 4347, 4369, 355, 11399, 1787, 4632, 4265, 1634, 12128, 3386, 198, 4520, 12273, 7904, 5114, 6937, 10796, 11421, 11797, 1045, 3305, 10727, 7378, 1478, 4462, 3372, 1196, 2659, 3879, 3453, 9161, 3866, 9075, 4949, 932, 209, 3780, 1209, 7124, 9677, 3365, 12714, 3032, 10027, 11004, 4554, 8455, 8329, 2087, 6720, 8051, 10507, 7700, 11709, 3804, 1885, 2782, 7428, 4108, 5465, 6699, 3468, 7424, 8138, 9313, 5949, 1578, 2729, 9358, 10262, 10630, 8542, 333, 6858, 3424, 699, 1080, 5601, 9712, 1898, 4553, 10059, 3006, 5549, 1741, 83, 9427, 7723, 11412, 929, 6022, 6429, 2167, 7756, 7389, 9937, 3034, 6501, 10982, 6333, 9982, 3476, 7254, 10489, 3260, 6231, 501, 8688, 732, 3668, 4353, 12095, 5707, 2184, 11254, 6058, 3999, 9425, 1249, 2684, 6126, 11130, 4923, 2270, 1861, 2036, 5385, 8629, 2392, 6476, 9343, 1444, 11727, 5192, 8587, 6277, 9489, 1160, 101, 4345, 2301, 2193, 8555, 8721, 6782, 9689, 6868, 3234, 3569, 199, 5718, 6577, 2508, 164, 11999, 5641, 5670, 2048, 6792, 4777, 9444, 7807, 1066, 1029, 12526, 9112, 11813, 10640, 1256, 5168, 3695, 4485, 2377, 4998, 7716, 110, 9411, 11732, 8256, 7688, 12478, 9061, 205, 10101, 2250, 7327, 5600, 12040, 10285, 2033, 981, 5603, 4761, 4601, 1719, 10553, 2300, 11422, 6175, 7919, 8524, 2615, 4288, 6314, 7650, 3515, 9777, 764, 1665, 1243, 5633, 12613, 1809, 9379, 6362, 8243, 2042, 4589, 7994, 12188, 9089, 335, 7358, 5618, 597, 7198, 9013, 6589, 3665, 846, 6093, 7592, 12039, 6160, 6657, 6879, 4727, 2723, 6892, 12364, 2444, 7787, 6766, 10535, 1272, 5903, 11575, 6723, 7139, 8171, 11866, 6053, 5563, 10081, 4078, 8186, 290, 5327, 12202, 5571, 9227, 6420, 6460, 2018, 4770, 4496, 9881, 3503, 7730, 11697, 2964, 4181, 8676, 6686, 5994, 9793, 8285, 744, 10547, 443, 12540, 2342, 10018, 332, 3446, 5531, 9305, 3432, 10263, 2649, 4207, 139, 7180, 3153, 2601, 2867, 2407, 3823, 7602, 11242, 1703, 7209, 3835, 1280, 7458, 7090, 11257, 9078, 11199, 10159, 10277, 6684, 8097, 7413, 6211, 2150, 11324, 1812, 7154, 8063, 7877, 8797, 3387, 2594, 10886, 8572, 1285, 1241, 7171, 433, 8169, 7975, 9402, 10424, 387, 9246, 1159, 336, 5472, 12536, 11836, 1328, 4689, 8734, 5487, 8411, 982, 3781, 274, 9947, 5852, 10302, 6543, 9483, 4113, 10108, 7925, 847, 4374, 9020, 9838, 2003, 4340, 11032, 6263, 5623, 8559, 6668, 869, 660, 9479, 7469, 4933, 6400, 7334, 7873, 7046, 9781, 644, 5706, 4703, 6184, 1919, 9475, 3349, 619, 3221, 3522, 6439, 578, 691, 4424, 1689, 6808, 1448, 7548, 3942, 4099, 5025, 9105, 7152, 1707, 11710, 2086, 3395, 11458, 2823, 4263, 191, 10517, 1126, 8942, 671, 5713, 11814, 9565, 3439, 11311, 11885, 7231, 4928, 1943, 5483, 6183, 11339, 4679, 1443, 8393, 5252, 7449, 8707, 2771, 8569, 7022, 2703, 1566, 2871, 2388, 2965, 9250, 6324, 8364, 8473, 4914, 9303, 6079, 9933, 11488, 12287, 893, 8351, 7729, 4005, 10772, 809, 7085, 7839, 11831, 2830, 4134, 4124, 3687, 9068, 1131, 12010, 6280, 7788, 9814, 11316, 5524, 4167, 10671, 1515, 5277, 4453, 8078, 10396, 7651, 3822, 3711, 5664, 4945, 3093, 226, 3013, 8710, 10653, 12488, 6655, 9704, 6098, 8675, 9162, 288, 7243, 756, 2550, 7410, 3609, 8299, 9173, 9626, 4842, 6422, 1461, 9809, 2969, 10094, 822, 6514, 6100, 11193, 2055, 9115, 889, 4782, 156, 6451, 9102, 10642, 11120, 7333, 8486, 4331, 6325, 11261, 4894, 9701, 7511, 6737, 11267, 3216, 407, 11583, 9351, 663, 5787, 284, 2963, 10554, 672, 3980, 9707, 8880, 5547, 5639, 2882, 301, 2580, 2076, 4107, 6966, 9623, 3011, 5356, 1677, 5378, 410, 12406, 2840, 3049, 9321, 5699, 6687, 8745, 7411, 4507, 4560, 2306, 8271, 1901, 4414, 1424, 4161, 9527, 9671, 2602, 278, 4460, 12241, 2838, 6714, 9547, 504, 2758, 7492, 4602, 3774, 5760, 9383, 9122, 10308, 4638, 6102, 7357, 7197, 2446, 5045, 12210, 5812, 9951, 6864, 1988, 4662, 4417, 3077, 438, 2613, 2243, 12347, 7516, 7369, 7264, 419, 12350, 2308, 306, 5304, 675, 5068, 10538, 9622, 7009, 2101, 5772, 10340, 1510, 9540, 9355, 1802, 10282, 10799, 7005, 3912, 3958, 12328, 8920, 12111, 12678, 7446, 7227, 5783, 11328, 2620, 2, 98, 6773, 3111, 2792, 9395, 1440, 12231, 11953, 7295, 1116, 8600, 4931, 11239, 12234, 17, 8391, 12211, 2954, 8013, 5895, 3504, 11532, 1062, 4961, 5756, 2039, 8864, 4083, 1971, 10766, 4447, 4898, 7854, 5667, 3335, 1935, 4548, 1940, 9532, 1590, 10749, 7963, 3824, 9242, 53, 1175, 9067, 11367, 10704, 324, 7256, 3704, 415, 4932, 6996, 3786, 8117, 12647, 7828, 4216, 8313, 2682, 8229, 8897, 2328, 3806, 6262, 8866, 8848, 9473, 133, 8404, 12619, 3012, 8685, 6956, 7217, 611, 6199, 56, 9097, 9630, 1692, 8263, 11717, 11909, 3170, 7270, 2409, 3037, 6464, 3354, 7541, 9517, 7947, 10193, 3322, 11627, 11137, 442, 10354, 11073, 2738, 8314, 520, 8047, 549, 7120, 3107, 8035, 10443, 685, 11834, 6959, 11041, 7600, 2396, 3752, 250, 8652, 10446, 5274, 1372, 13, 8752, 1193, 5128, 10874, 9098, 3566, 853, 897, 6722, 1106, 9297, 3715, 5737, 7642, 7365, 6348, 2865, 2548, 8634, 9810, 9751, 12548, 3173, 2303, 6590, 12182, 1648, 8515, 9888, 3212, 3005, 4831, 6831, 4086, 4901, 3291, 4377, 11025, 1672, 1533, 11382, 491, 668, 2355, 9954, 642, 8129, 1347, 4153, 5014, 6832, 10198, 10099, 3146, 10072, 5928, 2103, 3534, 9995, 9420, 9614, 5315, 4232, 1731, 12047, 10800, 5534, 6157, 708, 3317, 9332, 7685, 2299, 12645, 8295, 9166, 5818, 9836, 2047, 11415, 11629, 4973, 2641, 8976, 10573, 11871, 6148, 9636, 3676, 12612, 9049, 2215, 5122, 9927, 10020, 3287, 10461, 11209, 11111, 8142, 4546, 4979, 5023, 3918, 1054, 9154, 10523, 6411, 12156, 8563, 2230, 1404, 11046, 8947, 5859, 6088, 9828, 6748, 11214, 1882, 11156, 4685, 11142, 9687, 8888, 4027, 10136, 8071, 2390, 10731, 8146, 12215, 8727, 10687, 10876, 208, 8010, 998, 9822, 3872, 542, 2005, 5686, 4170, 216, 3994, 10795, 9205, 2456, 2623, 9357, 4742, 8788, 12407, 5851, 1368, 464, 10760, 6445, 10377, 9944, 2454, 1987, 40, 6620, 12552, 6569, 10518, 8429, 4619, 11125, 7991, 9470, 9216, 3172, 7272, 441, 6651, 341, 7420, 2976, 9136, 3331, 2977, 3782, 9744, 10353, 6050, 7966, 1387, 9604, 4358, 717, 10474, 11348, 7775, 12637, 8192, 4435, 10452, 12158, 6490, 2635, 8808, 7717, 10721, 5075, 6805, 1945, 7304, 8538, 3951, 3862, 11844, 12334, 10339, 9501, 2578, 12177, 2579, 8558, 6534, 8554, 5329, 8376, 10788, 2734, 12389, 7216, 7345, 9163, 10418, 11805, 997, 1938, 11687, 1673, 10509, 10883, 6051, 2240, 8977, 11633, 8384, 2427, 2605, 12131, 6461, 906, 2589, 1931, 6335, 8929, 9638, 6063, 4908, 12068, 10838, 5126, 2663, 1775, 7331, 3070, 2336, 3915, 2144, 11067, 4991, 12088, 11440, 3513, 4667, 8934, 1911, 12527, 1894, 2999, 8211, 513, 8002, 10337, 11564, 12517, 5428, 12486, 3613, 8787, 10657, 4576, 10741, 4624, 4486, 159, 1600, 3953, 8552, 6125, 2505, 12078, 8980, 12436, 5966, 6029, 11081, 735, 8330, 5773, 10612, 1349, 7347, 10566, 8981, 1309, 11186, 10281, 4603, 11595, 3892, 8610, 4183, 10209, 12490, 10626, 3195, 8331, 10195, 6938, 803, 3973, 12261, 7336, 11799, 11693, 5896, 3622, 737, 2540, 269, 7303, 8277, 3791, 2449, 5704, 3103, 11337, 10975, 4030, 4823, 12396, 10239, 5723, 4644, 12497, 5709, 3375, 8822, 8879, 4499, 4962, 11420, 3316, 4780, 1843, 2894, 5326, 2739, 11788, 10542, 2610, 6529, 5744, 4299, 10822, 11610, 10549, 275, 5251, 11036, 8915, 12070, 11410, 11147, 918, 2708, 10907, 7849, 936, 11967, 11058, 10111, 2611, 1593, 793, 5925, 1228, 12469, 7648, 509, 11580, 51, 10270, 384, 11719, 927, 4106, 3475, 3261, 9703, 9018, 12691, 5587, 67, 11526, 12534, 12371, 6685, 5080, 991, 12181, 6229, 7451, 1192, 3283, 3776, 610, 851, 1736, 3188, 5878, 5595, 9266, 582, 8703, 9848, 1002, 534, 11173, 12491, 12682, 2750, 11185, 6854, 10049, 11236, 5545, 2297, 8723, 11225, 124, 12444, 2503, 8862, 3765, 262, 2430, 3845, 557, 232, 7436, 1039, 12653, 11647, 9232, 7400, 5333, 2349, 6329, 12734, 31, 11728, 9903, 8931, 2024, 3154, 11303, 667, 3869, 12030, 9898, 5844, 1502, 9536, 103, 6733, 6972, 10629, 12399, 2981, 10579, 584, 11159, 9298, 12130, 3477, 9506, 540, 108, 8913, 3816, 7004, 4473, 9996, 9740, 10711, 4220, 3242, 3125, 10386, 4272, 5246, 4968, 5270, 5521, 8300, 4872, 2064, 2912, 3334, 8264, 5092, 11821, 10173, 1631, 3810, 8184, 5886, 7418, 11198, 1857, 9404, 79, 11389, 3554, 5826, 2339, 4524, 9324, 10758, 162, 11072, 11830, 802, 763, 5004, 9674, 5097, 5138, 8492, 632, 5849, 8666, 8984, 12183, 3187, 70, 4203, 8760, 7224, 631, 4924, 11570, 34, 11905, 8893, 2759, 1005, 1357, 11338, 7354, 10520, 3965, 10669, 2718, 7147, 4426, 5728, 5056, 4395, 6640, 11657, 329, 9330, 2435, 11630, 5133, 3581, 5739, 10105, 10291, 11618, 4707, 1916, 8553, 7519, 5921, 326, 5741, 1547, 615, 1185, 2060, 5505, 711, 6287, 2982, 1519, 10691, 3074, 6592, 5651, 7604, 7031, 1146, 4939, 10311, 10583, 1238, 35, 6760, 8747, 8916, 2209, 8388, 6793, 4562, 4266, 9850, 11256, 10856, 11063, 4297, 431, 6921, 71, 6504, 3934, 12418, 12307, 11446, 1784, 10335, 7786, 1093, 3655, 10478, 11227, 2496, 2234, 3038, 10784, 920, 2694, 3848, 600, 4781, 5602, 5400, 6778, 9113, 5099, 10170, 8319, 8630, 6424, 3658, 39, 11941, 2472, 10283, 9035, 9976, 7658, 11915, 1772, 9821, 5027, 8782, 2683, 7627, 3355, 4318, 3236, 8309, 3321, 154, 3254, 5695, 1351, 5323, 2081, 2040, 7758, 8267, 1003, 3887, 3232, 2984, 4090, 2598, 12654, 1835, 1205, 7151, 8591, 8846, 2919, 10665, 12193, 8809, 4394, 9586, 1773, 7062, 7518, 9860, 9406, 5205, 6415, 1096, 5591, 1957, 1374, 8638, 299, 4541, 10143, 6351, 5582, 6081, 11545, 3215, 10739, 10372, 12663, 11507, 11940, 7560, 12378, 11009, 902, 10175, 1109, 1700, 6074, 1942, 8937, 7247, 2777, 4596, 16, 6983, 9508, 362, 5750, 11884, 4409, 1687, 8921, 4942, 6409, 10003, 4792, 3400, 2331, 8510, 2008, 8110, 1617, 6795, 7852, 2343, 11087, 3673, 12058, 9388, 9054, 8841, 9424, 2026, 11297, 10493, 753, 10427, 4055, 7163, 3606, 10261, 5298, 109, 1186, 12434, 10871, 5150, 9675, 3274, 4323, 550, 7075, 2014, 9971, 2573, 4256, 448, 1726, 6076, 4491, 3551, 8174, 2102, 11652, 9293, 2511, 7117, 11820, 11692, 10638, 6775, 12401, 3763, 4131, 8694, 5283, 9433, 2030, 9347, 4396, 6817, 2820, 7514, 10693, 12195, 2633, 3771, 10279, 4503, 488, 2789, 931, 12674, 446, 3604, 4579, 9710, 6694, 8452, 3019, 4523, 9254, 4386, 6209, 8488, 11219, 12670, 5448, 3974, 1902, 11414, 1904, 5885, 9861, 12093, 12038, 8951, 12349, 7356, 12702, 10364, 3438, 8017, 194, 3201, 4290, 1308, 12207, 3036, 4786, 11861, 8396, 5520, 6191, 1180, 4808, 11832, 6727, 5970, 2933, 7055, 2452, 7266, 12707, 6239, 8422, 8996, 12015, 3505, 97, 3992, 7308, 10145, 12185, 10152, 7831, 12148, 3989, 4721, 3165, 12057, 3247, 4972, 11351, 12341, 4192, 1198, 3367, 6643, 4102, 1529, 8337, 11028, 8909, 9066, 3341, 2926, 5136, 10238, 3570, 914, 2928, 6502, 5542, 11026, 2172, 2197, 2480, 8487, 10789, 3507, 665, 8640, 11951, 9934, 6089, 1525, 10362, 7646, 3382, 7036, 10973, 10047, 12584, 3136, 9618, 10745, 6580, 9164, 2166, 1591, 1565, 3271, 4791, 5069, 12432, 10666, 1776, 5746, 9806, 1559, 11180, 7007, 1409, 4122, 6960, 5692, 4694, 12201, 4916, 1319, 11865, 6202, 4287, 5000, 3664, 7884, 8223, 1640, 5837, 12503, 2450, 9658, 10406, 4510, 1279, 7889, 11212, 7313, 4195, 828, 1061, 12420, 10503, 875, 7654, 11241, 1016, 4801, 137, 11672, 4165, 678, 373, 6180, 9221, 7572, 8318, 78, 7381, 10194, 9868, 9514, 9897, 3874, 8224, 500, 1231, 6608, 5140, 8751, 5735, 249, 5231, 959, 7390, 4909, 3266, 10497, 4671, 8325, 10853, 1327, 7486, 10618, 3925, 8835, 5366, 4539, 10416, 9129, 12304, 11443, 9505, 3740, 11530, 12477, 2558, 6338, 9240, 7909, 12635, 11726, 7735, 1805, 11047, 8284, 11483, 10562, 2720, 3061, 8900, 6282, 2214, 8292, 6872, 7941, 9170, 8275, 8525, 3572, 12268, 12545, 5349, 8293, 10234, 2425, 7662, 6364, 9258, 12563, 9681, 6176, 3100, 8744, 3324, 10611, 3361, 6768, 9094, 11517, 3718, 1552, 7108, 12651, 6595, 8564, 11387, 4071, 9555, 5421, 9023, 4449, 10751, 6818, 12706, 1419, 6508, 1242, 12023, 725, 8334, 4869, 7001, 5147, 9871, 12601, 2110, 7068, 11819, 10797, 5657, 11653, 1401, 6600, 10129, 4574, 10312, 10445, 3441, 5063, 516, 10024, 10073, 10310, 7737, 381, 10156, 213, 10345, 3697, 1941, 7779, 9844, 392, 2669, 4096, 2029, 2180, 6586, 547, 6896, 6004, 9682, 2380, 9780, 12162, 1000, 2114, 458, 66, 7023, 5131, 1642, 6037, 3991, 1542, 8094, 413, 4706, 655, 9646, 3108, 2116, 9724, 1284, 5819, 11815, 10775, 9042, 7537, 1248, 8414, 3303, 9281, 1110, 6414, 195, 9026, 8109, 6786, 6759, 1210, 7092, 6371, 9767, 1995, 7929, 7048, 6613, 863, 8985, 5171, 11699, 4254, 1890, 2423, 1304, 11302, 12292, 855, 7791, 2519, 9756, 2249, 899, 4951, 10092, 4811, 7434, 4935, 865, 6906, 2386, 3523, 7414, 12114, 6120, 8092, 12609, 7949, 9938, 12113, 1121, 6757, 9847, 5791, 1630, 1905, 9213, 2119, 1751, 3902, 6968, 1951, 707, 4983, 6192, 11289, 2860, 2947, 9856, 316, 3633, 7003, 1981, 7863, 76, 3440, 602, 7398, 12180, 11837, 8124, 85, 1909, 4337, 9787, 2824, 10737, 6367, 909, 11684, 9949, 7210, 1323, 165, 3179, 4422, 7766, 3436, 673, 11077, 4364, 5935, 11282, 2318, 1675, 8803, 11867, 5856, 3683, 3585, 12103, 4025, 3998, 5456, 3376, 5120, 5971, 3258, 10236, 9698, 1343, 10606, 11745, 3350, 1500, 6459, 4616, 3449, 4028, 6731, 7636, 11090, 8202, 11757, 5540, 5621, 2571, 11110, 536, 5453, 8, 2145, 965, 4484, 2158, 7902, 7840, 2736, 984, 12625, 8464, 2032, 1206, 1989, 25, 5981, 257, 10613, 5424, 12505, 7733, 11978, 8438, 7530, 1137, 5972, 8990, 4348, 9709, 3417, 7462, 633, 12474, 7583, 1458, 1797, 12206, 3063, 1869, 3184, 4109, 7984, 3601, 8623, 1669, 9118, 7903, 10974, 4151, 1023, 1237, 1596, 915, 4807, 6161, 7059, 6784, 8905, 2562, 5614, 10572, 3950, 90, 356, 4913, 515, 2685, 8785, 6258, 1604, 11512, 12186, 12561, 3276, 7533, 11829, 9587, 11918, 10419, 4561, 9994, 9130, 518, 1027, 812, 12559, 2509, 8194, 6271, 2876, 4748, 3315, 8726, 5778, 4837, 4258, 1647, 7827, 6281, 6539, 3490, 6162, 10295, 1499, 7233, 7988, 1283, 3046, 3819, 8044, 709, 8733, 2891, 4744, 7113, 450, 770, 10171, 9987, 12446, 3733, 4577, 4408, 5808, 1975, 12481, 2638, 6709, 389, 3206, 2245, 6815, 3901, 1299, 11632, 12077, 8672, 4720, 12056, 8389, 7116, 5180, 7603, 12029, 11937, 1201, 6108, 9476, 7421, 8282, 710, 10677, 12743, 10168, 5419, 11609, 1113, 3997, 1225, 1153, 11817, 2883, 5516, 5351, 6663, 6498, 2320, 4016, 4268, 5610, 7948, 1652, 8036, 5790, 7456, 7074, 9905, 3120, 10914, 4379, 5142, 3058, 3344, 3311, 11038, 3714, 9919, 567, 11769, 12698, 3742, 6902, 5172, 5678, 6726, 2653, 3347, 7191, 4239, 1128, 5263, 4637, 11057, 6576, 4309, 2608, 12303, 10917, 8622, 6456, 143, 7891, 10585, 6689, 3169, 6061, 4184, 6802, 5132, 8517, 9752, 10714, 8682, 9803, 390, 5747, 11585, 3521, 8062, 4087, 6288, 3598, 9225, 6840, 7106, 7665, 3066, 712, 5415, 7946, 8057, 7161, 11054, 11012, 403, 12512, 3101, 5361, 11987, 8341, 2016, 5535, 5182, 12315, 9012, 12646, 10763, 7077, 8113, 10882, 3043, 5149, 8711, 10682, 2337, 5780, 12440, 7545, 3346, 10160, 7169, 2292, 4896, 3837, 4854, 2346, 6690, 4322, 4405, 10736, 10729, 4128, 3492, 4212, 9450, 9434, 8919, 11265, 340, 217, 8706, 9598, 9942, 7447, 12587, 9673, 2817, 11079, 5663, 12135, 11492, 4468, 9030, 414, 11614, 11913, 10068, 5702, 11853, 123, 1176, 2753, 11882, 7954, 5900, 1998, 2071, 6141, 12080, 12711, 10360, 8763, 3857, 2756, 2260, 3930, 11765, 4498, 4843, 7637, 7538, 6190, 2776, 10077, 9143, 2910, 11935, 10323, 3854, 10177, 9364, 9270, 733, 9776, 1281, 8904, 5173, 7772, 12344, 8853, 9482, 6926, 11749, 2118, 10359, 8632, 12532, 9571, 2268, 4783, 8716, 10107, 5721, 5617, 12602, 2371, 364, 9662, 3889, 9062, 3269, 900, 11549, 11539, 7821, 8190, 5343, 11663, 5869, 4487, 4695, 12453, 1, 1966, 11207, 6517, 6833, 2156, 5832, 674, 1676, 9610, 579, 5249, 3086, 2280, 5146, 8042, 6806, 5062, 5155, 1889, 2557, 3777, 7781, 9988, 2603, 9060, 3207, 8741, 2422, 4459, 8251, 9631, 8633, 6075, 5334, 2458, 6873, 878, 357, 10644, 5643, 5230, 10078, 470, 8856, 7869, 399, 7724, 6095, 2347, 3224, 5881, 9930, 6515, 8624, 7433, 11946, 590, 5519, 11048, 10260, 3517, 4805, 2650, 4969, 9139, 11179, 7985, 10918, 3861, 4158, 496, 6963, 9827, 8503, 11713, 10023, 9160, 12624, 4201, 7926, 838, 5513, 11658, 7043, 10440, 1352, 6893, 2774, 9014, 11642, 10997, 844, 6326, 4764, 7918, 11773, 10709, 1150, 891, 3849, 6542, 473, 12255, 6875, 10459, 11795, 7534, 2744, 3178, 10186, 7759, 481, 3410, 5834, 7694, 8371, 10667, 2176, 2655, 10878, 4059, 731, 12021, 3594, 3185, 1366, 9198, 5498, 10320, 9694, 469, 3458, 4285, 7542, 775, 3248, 3002, 5055, 9657, 10032, 4234, 6969, 7819, 953, 1948, 2994, 10604, 6785, 12308, 3133, 5624, 683, 5022, 1929, 5272, 3685, 9, 8643, 9961, 8415, 7493, 12248, 565, 12593, 10074, 5845, 9096, 6553, 6713, 992, 104, 8451, 11070, 2317, 8238, 5159, 1803, 11129, 10575, 11023, 1562, 11288, 10552, 1571, 5476, 9705, 10685, 625, 1001, 4046, 2784, 9973, 9805, 7628, 9272, 8069, 10632, 10811, 5, 3960, 5565, 7064, 3884, 393, 5874, 8450, 10834, 1227, 8626, 2424, 7172, 2772, 12048, 6927, 8580, 11675, 2316, 4245, 4136, 7412, 3520, 1958, 7911, 7293, 9296, 6213, 9845, 9642, 2023, 6473, 1296, 2122, 8454, 2363, 2679, 8664, 8225, 12375, 12079, 5410, 9964, 4990, 7028, 3081, 8993, 10813, 879, 7222, 1944, 7130, 10405, 11373, 7983, 8126, 10955, 6635, 11702, 11592, 3380, 5222, 3626, 3657, 10471, 7883, 7457, 2279, 9773, 5784, 10723, 9496, 4768, 7907, 7296, 10316, 11947, 9394, 3701, 2100, 129, 2121, 7417, 11037, 10762, 9278, 6087, 4068, 12431, 7018, 1670, 7928, 8890, 12727, 2170, 3502, 2208, 9063, 8177, 7606, 12233, 10993, 7510, 5636, 317, 7314, 10189, 930, 6880, 8030, 11317, 4416, 5484, 4804, 4277, 6891, 7329, 436, 4308, 11453, 5300, 6618, 7677, 7687, 2828, 1679, 3480, 7599, 5038, 7986, 6881, 9155, 4036, 12525, 1844, 9194, 9430, 2465, 5941, 11043, 5162, 4918, 12617, 8254, 11559, 845, 6678, 832, 2588, 6585, 7087, 9652, 3067, 4987, 5229, 4713, 4197, 4929, 10602, 3990, 4477, 11968, 1925, 7973, 10738, 9736, 12316, 698, 5248, 5102, 7932, 2213, 6905, 2069, 10038, 11872, 1115, 5640, 2107, 11615, 8484, 4830, 315, 12100, 7471, 804, 6707, 2252, 6644, 10408, 2273, 11002, 1077, 5049, 2851, 3524, 4320, 238, 4519, 9639, 5087, 3753, 9981, 3741, 12246, 5883, 12585, 4296, 10207, 12735, 3197, 350, 10148, 8151, 4673, 10033, 6483, 10680, 7121, 11355, 4584, 2765, 6563, 623, 8972, 10546, 4625, 10413, 4500, 4077, 12386, 7669, 3253, 11071, 1678, 8191, 8132, 5370, 9797, 7006, 10417, 6376, 10877, 10768, 6967, 3731, 7144, 2664, 1633, 10919, 8363, 4604, 4759, 7908, 5265, 3871, 6493, 968, 6479, 5314, 6848, 11031, 8128, 3979, 8367, 11187, 8748, 2007, 5977, 7261, 1662, 4080, 11431, 9512, 12054, 1520, 881, 6834, 11113, 5488, 437, 12167, 4202, 6207, 5312, 10678, 10321, 3875, 11666, 6436, 3281, 2791, 868, 7468, 1610, 6189, 10034, 11591, 10087, 11660, 537, 1568, 6728, 7590, 9835, 3580, 12626, 7134, 3617, 5449, 5301, 870, 3138, 2879, 1823, 1118, 8966, 102, 9065, 7123, 4196, 10674, 3948, 1024, 9058, 5313, 963, 6096, 10971, 7540, 2902, 7663, 12589, 8837, 11750, 8968, 7829, 9884, 11326, 8011, 1089, 1141, 8491, 7125, 3381, 7320, 12199, 12061, 2420, 1755, 2946, 1247, 5658, 2700, 7512, 12529, 11430, 8571, 6852, 11985, 11011, 9417, 7164, 12433, 7752, 6174, 2510, 1486, 9816, 3396, 382, 6194, 5093, 693, 276, 9746, 10539, 12603, 2943, 8939, 7502, 1019, 7739, 7980, 1983, 7129, 6702, 5208, 11477, 5504, 10085, 4732, 9019, 827, 5369, 11334, 9798, 2725, 5005, 388, 5452, 8471, 7912, 8072, 11369, 9432, 5947, 8418, 11768, 4034, 543, 9539, 4877, 6097, 745, 9048, 8323, 2341, 11883, 10571, 7467, 11194, 12134, 1701, 7625, 10052, 4085, 6250, 6865, 1978, 2885, 6487, 372, 4828, 10915, 5497, 12264, 3840, 3451, 10798, 8448, 8075, 5081, 11904, 6410, 4846, 7623, 9770, 2259, 9006, 7167, 11, 12454, 12252, 395, 8691, 11322, 12263, 3829, 512, 3092, 9790, 1650, 996, 8496, 6086, 6101, 9052, 7257, 5681, 10900, 4590, 4242, 7659, 1260, 9405, 9131, 4862, 1778, 10438, 1124, 7914, 8689, 8735, 887, 2848, 3729, 11886, 4772, 9602, 9053, 7310, 2478, 10441, 12576, 6434, 230, 4763, 2061, 661, 11894, 4868, 219, 4119, 607, 3696, 10578, 9285, 10273, 5189, 8635, 7639, 7101, 1011, 6578, 2200, 4532, 1376, 9734, 1752, 11151, 10093, 4177, 5592, 9750, 1315, 8386, 11743, 8144, 2108, 4351, 2405, 10649, 9308, 7706, 3917, 9010, 6571, 10513, 1383, 12746, 10488, 503, 8547, 9188, 8883, 5194, 7239, 5362, 6068, 9488, 6442, 8088, 4815, 6099, 6492, 11332, 9557, 7141, 11096, 4171, 654, 9050, 9104, 2713, 4132, 8068, 8012, 1356, 3231, 10660, 8427, 6042, 7482, 5462, 2677, 9416, 12294, 519, 7982, 11062, 714, 3025, 4614, 4333, 1184, 1663, 3536, 12250, 2515, 857, 12243, 2056, 5733, 12660, 11602, 11114, 8439, 7922, 9440, 1505, 10015, 2015, 11656, 12214, 4214, 882, 6020, 4032, 3056, 11796, 3421, 7979, 6583, 6372, 6874, 3615, 955, 10144, 9554, 9081, 1295, 2049, 426, 6419, 10256, 11605, 12360, 2137, 5368, 10600, 11522, 6040, 9603, 3362, 1194, 2028, 5836, 10313, 5908, 6478, 1974, 10350, 9972, 7953, 12669, 1561, 2716, 7150, 1853, 5889, 4999, 1723, 9300, 4776, 5145, 3121, 2644, 2227, 5371, 5503, 2220, 4279, 4537, 6496, 12190, 11770, 4072, 8029, 6132, 4585, 5493, 1615, 4380, 9854, 3399, 428, 7845, 1391, 12695, 482, 2873, 148, 2354, 6425, 5089, 9800, 4687, 948, 193, 739, 11747, 12001, 4958, 8641, 10124, 7865, 2141, 11428, 12163, 6523, 3679, 10106, 8501, 1656, 2732, 12726, 4827, 11163, 9581, 5753, 3881, 11286, 11274, 8497, 8886, 1385, 10043, 2766, 4421, 940, 5183, 5019, 8099, 3555, 439, 12097, 4092, 10679, 10252, 596, 5317, 10761, 1768, 9283, 7795, 27, 4540, 7817, 3868, 4788, 10407, 6730, 7804, 1576, 2447, 10861, 4841, 1428, 5754, 6816, 55, 9119, 2900, 6626, 10464, 10463, 8874, 3473, 524, 258, 11295, 10439, 11523, 8043, 1883, 10253, 2254, 11810, 1207, 3722, 6884, 3337, 7505, 3166, 4549, 11371, 10258, 1452, 10399, 9656, 10080, 5100, 6541, 2796, 12387, 621, 11528, 5961, 6718, 8766, 1320, 7871, 4022, 5877, 6404, 4010, 758, 11714, 1165, 5110, 1301, 157, 1403, 5003, 8881, 2978, 7096, 2202, 12172, 2294, 12551, 2157, 1623, 682, 4833, 7496, 2986, 6134, 8789, 7770, 9076, 46, 1076, 10598, 5200, 723, 11232, 377, 10383, 7950, 4996, 3220, 359, 6236, 9093, 6146, 5502, 3272, 11864, 5324, 3020, 7763, 8095, 2808, 11983, 9600, 474, 9074, 9371, 2262, 2413, 4488, 5088, 6023, 2305, 2186, 2072, 6982, 3682, 7968, 2966, 7749, 5266, 1289, 9134, 10945, 8423, 6129, 12298, 6612, 7561, 6754, 4873, 8065, 2309, 6691, 1429, 8704, 4676, 7906, 9431, 10688, 3932, 2783, 10458, 12716, 5207, 11679, 7524, 21, 4593, 6899, 9635, 6164, 1099, 11878, 12655, 9367, 10346, 6932, 1900, 11790, 4313, 1112, 3106, 2274, 9882, 440, 3117, 1103, 1454, 8446, 724, 9920, 3637, 9279, 2979, 4702, 5037, 456, 10017, 5054, 7764, 7305, 5792, 8279, 12329, 11792, 5879, 11245, 975, 5862, 11688, 2161, 4797, 3748, 6518, 12072, 12354, 9515, 444, 1221, 4847, 4858, 4075, 3295, 6317, 7012, 2269, 8860, 10355, 1032, 2135, 4752, 6083, 5414, 10237, 7901, 6922, 48, 5771, 7080, 8845, 5703, 6701, 8631, 990, 371, 10229, 10828, 4785, 7778, 7713, 1144, 1268, 12591, 9251, 4439, 2541, 9138, 6667, 9910, 2577, 1220, 6484, 8084, 3174, 2104, 9796, 2127, 391, 6443, 4062, 6357, 10950, 11555, 3241, 3623, 11335, 6458, 6886, 11849, 11521, 4041, 186, 2621, 9883, 9967, 6405, 5373, 1664, 4223, 8612, 4915, 285, 4556, 4573, 476, 11771, 7629, 2319, 12032, 7476, 3494, 7670, 8730, 967, 8247, 970, 8173, 1122, 2972, 342, 9151, 7444, 6009, 212, 8166, 2332, 9843, 11594, 4567, 6417, 7235, 5550, 761, 2451, 5950, 595, 2109, 10961, 3202, 92, 3394, 10991, 2165, 434, 8567, 7570, 9142, 5606, 6433, 2302, 5919, 7527, 5166, 11988, 12426, 4310, 9348, 4479, 4038, 10628, 8894, 10300, 2566, 9465, 11848, 9886, 8055, 1721, 6669, 12685, 11515, 9959, 6931, 9414, 5825, 7734, 7850, 11730, 10257, 3466, 8908, 6373, 11085, 8162, 1822, 10946, 5289, 9335, 9923, 45, 6363, 1825, 6111, 4575, 9685, 11643, 3608, 4475, 2233, 6742, 10486, 5434, 7246, 514, 3876, 5798, 11651, 7309, 6916, 3662, 3587, 2886, 3940, 2059, 10898, 11300, 3596, 999, 8489, 9055, 9753, 11474, 239, 1514, 1859, 11097, 9503, 5659, 11863, 1725, 11551, 8435, 9336, 9908, 6843, 360, 1878, 8108, 10816, 12358, 2895, 6045, 9392, 6381, 5105, 11436, 11825, 6885, 11753, 12610, 10690, 5631, 9141, 7622, 5225, 10401, 3692, 11411, 1370, 5363, 3443, 3039, 2706, 2204, 9147, 6359, 12429, 12509, 2587, 7219, 7391, 9700, 2356, 11409, 3225, 10622, 1793, 12159, 8466, 4921, 6952, 8757, 4883, 11703, 12343, 3285, 4145, 6391, 12356, 7479, 11519, 6318, 4240, 8272, 9952, 3390, 1821, 3210, 10211, 12302, 11433, 4722, 12502, 244, 5696, 11506, 10125, 10269, 2097, 12267, 497, 7183, 3369, 114, 1418, 4606, 1595, 100, 3027, 5374, 3144, 836, 3939, 9021, 11534, 1063, 11161, 8958, 4735, 8485, 1251, 2924, 5399, 8924, 12562, 2372, 3920, 2442, 3530, 11808, 9484, 1239, 3160, 6043, 2345, 3118, 7618, 6856, 8884, 6025, 12007, 3186, 6292, 6901, 5135, 7203, 2743, 10102, 10448, 7252, 2855, 12467, 9534, 12675, 2460, 8073, 9568, 2825, 9545, 6944, 352, 4300, 6047, 9747, 2832, 12598, 8804, 5039, 4530, 8303, 11103, 1536, 1232, 1830, 9087, 3735, 9454, 12, 12412, 11754, 8827, 280, 824, 1769, 9574, 794, 6925, 12139, 594, 7585, 140, 11794, 11807, 10591, 10400, 126, 6940, 8522, 8131, 12544, 5471, 5757, 8562, 11841, 8353, 3506, 5729, 10650, 8649, 1589, 3259, 10134, 5052, 308, 10913, 1169, 10055, 904, 12281, 5377, 6173, 5931, 6028, 7620, 9518, 12627, 11105, 9132, 11413, 1757, 5507, 6519, 2443, 5922, 1335, 2733, 3069, 8701, 8677, 7368, 221, 10293, 12458, 50, 3363, 383, 11160, 3986, 7364, 6380, 8495, 2418, 12297, 12416, 1282, 375, 3793, 4411, 9069, 9825, 1318, 5275, 5793, 2448, 4978, 9529, 4117, 8198, 7260, 7223, 3450, 1229, 6732, 10514, 348, 4988, 4911, 4910, 6744, 962, 9306, 12043, 3888, 1507, 2530, 3618, 3510, 10664, 2325, 3691, 11919, 11076, 4937, 8574, 7473, 11429, 1108, 4430, 10484, 1223, 3755, 11176, 4509, 2264, 7748, 3379, 10533, 9875, 4555, 11078, 11546, 8692, 9965, 7927, 6904, 338, 2212, 453, 183, 7442, 8149, 3636, 11994, 11774, 7995, 10483, 3680, 6696, 9613, 12568, 6255, 7182, 10197, 11150, 245, 11171, 10790, 1750, 2489, 10344, 3568, 874, 4162, 2290, 11881, 1699, 485, 10276, 7761, 6551, 9256, 11508, 1104, 6339, 5557, 12445, 1436, 2904, 5480, 4143, 12400, 5876, 9070, 9968, 7747, 6798, 11997, 11576, 612, 12342, 11102, 11486, 8336, 10994, 11174, 8561, 5564, 10457, 11993, 6825, 2581, 3645, 7010, 2112, 6915, 236, 1417, 12720, 8952, 7104, 8148, 5316, 4246, 11690, 1611, 12701, 9262, 3747, 7089, 11888, 11700, 11922, 3508, 3147, 7194, 6210, 3646, 7896, 8346, 8722, 10884, 9426, 6489, 720, 12238, 6323, 1058, 8661, 2958, 2133, 9731, 12161, 3910, 4249, 9318, 12296, 11487, 9998, 9990, 6866, 29, 11013, 1053, 2140, 4559, 938, 7395, 7000, 4155, 4789, 9271, 2051, 11395, 2818, 8556, 3758, 6330, 8912, 2149, 6870, 4723, 6014, 6315, 1214, 9732, 8245, 8578, 1273, 8305, 1997, 9237, 146, 12607, 8442, 11138, 4149, 3923, 3670, 2075, 9288, 1479, 12422, 7702, 11933, 6091, 6698, 3643, 8543, 1635, 2802, 223, 12165, 3710, 3550, 1213, 10888, 5613, 5286, 9573, 6368, 7630, 2484, 6272, 6971, 12468, 8377, 8265, 639, 10481, 4495, 10326, 6346, 7522, 10617, 5204, 1389, 7792, 7245, 9193, 9763, 347, 10201, 11846, 6623, 7162, 11084, 2085, 107, 5318, 63, 3944, 10494, 7435, 3181, 9219, 1245, 4613, 5282, 8833, 4738, 1226, 5344, 10487, 10706, 8076, 7240, 10870, 2381, 3578, 8033, 6914, 2937, 12395, 5115, 6890, 7269, 11388, 5556, 3886, 3693, 10901, 12208, 784, 3457, 264, 9498, 7484, 873, 7464, 5847, 553, 11080, 2591, 8898, 9817, 7800, 5188, 11375, 8621, 10370, 4615, 10873, 2656, 8405, 7082, 608, 6031, 12168, 9872, 4452, 9737, 7594, 4640, 1253, 10926, 7814, 6131, 8836, 5763, 5113, 1754, 9644, 3469, 10605, 8159, 6349, 3757, 2940, 9323, 9181, 9593, 8772, 925, 8936, 6673, 9733, 2549, 9958, 5389, 5294, 11287, 1740, 6950, 6077, 1920, 1325, 6550, 7110, 819, 11341, 8216, 3340, 11553, 5082, 907, 6297, 7212, 659, 12274, 4070, 323, 3649, 7078, 11030, 969, 6703, 2389, 5593, 8684, 10098, 11912, 7200, 3900, 203, 7372, 8347, 9135, 9199, 8619, 7406, 1246, 1361, 3589, 10299, 12571, 12460, 7728, 1923, 5855, 12501, 2179, 1168, 9779, 11203, 6753, 5010, 4390, 8478, 9583, 12140, 10836, 2819, 6705, 3243, 11694, 6374, 10275, 2646, 2020, 8658, 2813, 1819, 11350, 11233, 10342, 6954, 5354, 2995, 8381, 2582, 9056, 8940, 11682, 4039, 1439, 2321, 7894, 4457, 767, 4849, 3460, 8236, 556, 6999, 11839, 12428, 179, 12179, 8262, 6316, 6912, 5840, 1961, 6619, 994, 5674, 367, 5991, 8402, 322, 12301, 8873, 1671, 2191, 4955, 5765, 263, 225, 11650, 8910, 2228, 4812, 12013, 1924, 9345, 9342, 9215, 6319, 3218, 4294, 8683, 483, 10140, 4060, 7255, 9929, 12530, 11263, 1331, 6253, 1073, 795, 8209, 4458, 5669, 6062, 10889, 88, 1056, 4745, 1717, 6259, 412, 3525, 3209, 5237, 5957, 10477, 1212, 4227, 2052, 5835, 12034, 7030, 12127, 9578, 1532, 7100, 8304, 1154, 2941, 7040, 4965, 3010, 1694, 942, 876, 9360, 5002, 11494, 8736, 7690, 5426, 2917, 7041, 9569, 2586, 4044, 9831, 4425, 1427, 4938, 10332, 8053, 2330, 11847, 10999, 10013, 3817, 3351, 7192, 3293, 197, 8227, 8648, 3880, 11165, 3075, 8141, 10695, 6918, 11816, 7619, 12713, 2089, 5490, 4450, 5459, 1594, 5201, 202, 11737, 12723, 10468, 8185, 12253, 7798, 8865, 1806, 2231, 11400, 8231, 2244, 10623, 1937, 719, 5821, 2021, 3620, 4724, 2058, 3700, 1047, 10781, 2151, 2609, 4350, 3314, 6624, 10620, 7133, 11155, 10722, 4506, 4904, 8646, 4095, 2631, 10057, 10357, 10243, 1970, 9222, 1083, 2459, 429, 6823, 1495, 8046, 5116, 12353, 7797, 9368, 1344, 4049, 10202, 1037, 2647, 7299, 9864, 681, 9970, 9280, 3654, 9758, 9140, 127, 10338, 2473, 11248, 5187, 8928, 8383, 9307, 8541, 4418, 8311, 122, 7755, 3094, 4734, 12224, 4686, 9372, 327, 3760, 797, 11586, 10683, 2189, 12203, 11377, 5091, 12017, 7952, 2835, 3252, 5528, 3057, 5258, 12271, 8456, 10133, 7301, 11622, 10454, 1350, 4372, 7971, 1756, 1348, 1845, 9945, 11775, 8049, 3485, 3962, 2429, 5202, 11363, 5671, 1067, 7957, 6547, 149, 9684, 8805, 8705, 1107, 3764, 12198, 6582, 1178, 5224, 10584, 12229, 8081, 7483, 6105, 11033, 3343, 2017, 8667, 8540, 3140, 1572, 5134, 11167, 1573, 11529, 837, 2357, 3815, 1858, 6237, 4081, 10569, 11432, 4674, 10336, 2593, 6024, 9200, 9226, 9257, 11616, 2432, 5439, 12414, 7409, 12112, 8248, 10096, 6144, 9859, 4665, 12496, 7439, 9002, 4388, 2476, 9150, 10251, 10447, 852, 2000, 1311, 2287, 8357, 1453, 6453, 1691, 620, 6477, 7153, 4643, 9978, 10956, 12110, 1022, 5813, 5250, 2538, 1267, 9497, 320, 11091, 4319, 5104, 6734, 6035, 1020, 4688, 4298, 1879, 7322, 11464, 2329, 3130, 4097, 6222, 8659, 6779, 5060, 8246, 1534, 9175, 2247, 12322, 7386, 3719, 1836, 5446, 7038, 4775, 9542, 10764, 1480, 3310, 1219, 11654, 116, 5433, 8020, 5782, 11587, 5404, 6018, 12152, 7376, 1606, 11437, 4471, 2806, 7094, 12065, 11812, 3389, 10368, 9428, 7459, 6462, 12644, 8857, 1544, 7653, 1291, 6985, 5345, 6920, 5408, 3552, 1326, 8700, 5206, 12222, 8269, 3591, 3098, 11086, 11228, 10058, 11075, 1276, 2079, 4926, 3619, 3030, 3131, 2553, 11556, 1490, 5909, 8783, 77, 12133, 8026, 5913, 3360, 3486, 10366, 9211, 3544, 7565, 5352, 6725, 3559, 5427, 8674, 2829, 10641, 5158, 260, 5072, 5932, 1816, 5789, 11748, 6797, 2687, 6903, 272, 11049, 11856, 4438, 2704, 6758, 2528, 4191, 9629, 1564, 829, 9148, 3724, 10191, 3976, 2498, 2326, 7011, 5917, 12632, 9726, 11244, 1554, 11196, 6273, 7701, 7373, 2629, 207, 1191, 1004, 12504, 1177, 5267, 8545, 1541, 7822, 10460, 1774, 3770, 7419, 12151, 11562, 6646, 2338, 218, 8041, 2864, 11977, 11952, 11960, 7135, 8157, 8204, 6558, 11019, 9191, 6163, 926, 12284, 6352, 839, 4280, 2288, 5028, 9655, 4015, 4946, 5257, 2384, 9914, 4481, 9234, 2837, 2913, 4755, 11065, 10832, 10422, 1807, 11949, 12016, 3155, 9974, 6073, 11312, 5073, 9352, 7674, 11649, 5599, 4146, 9472, 4582, 3033, 5973, 9625, 1400, 5803, 3227, 6729, 7193, 11333, 7097, 7149, 2291, 11646, 8344, 12025, 9807, 11784, 8399, 10654, 6610, 6895, 5040, 7898, 3284, 8840, 1871, 9167, 113, 11557, 9247, 1914, 5185, 9789, 843, 9399, 866, 8557, 8697, 10255, 6601, 1339, 2352, 12289, 396, 5517, 3681, 4264, 5948, 8298, 4200, 6389, 3785, 6147, 7490, 8599, 1815, 10785, 6251, 3832, 5388, 8739, 8955, 8287, 2185, 9366, 7344, 1292, 9441, 10303, 6048, 648, 1887, 7740, 6393, 189, 1399, 6488, 10505, 7426, 5630, 11083, 11197, 6801, 8259, 7241, 7207, 1139, 9762, 5665, 9177, 656, 6396, 8472, 11811, 6437, 11858, 11908, 3743, 1601, 5161, 10916, 10218, 2077, 11238, 7277, 6066, 11189, 3638, 12242, 11406, 1421, 5179, 2350, 677, 9044, 9239, 12036, 10655, 5662, 11681, 1842, 11444, 9829, 1789, 7174, 7472, 5242, 457, 4897, 5586, 2038, 421, 8625, 5383, 10639, 8930, 4570, 8776, 1626, 10765, 601, 12213, 1874, 5320, 825, 11206, 8359, 1867, 8779, 5381, 4397, 10614, 8419, 9723, 3388, 12689, 2755, 4402, 12425, 2093, 8172, 5292, 5241, 7872, 8712, 3851, 1462, 10740, 1395, 69, 5689, 1790, 4035, 6647, 10363, 4934, 131, 2567, 10063, 11668, 12059, 9592, 10662, 5933, 11936, 11359, 234, 9197, 8753, 10815, 9760, 5611, 1577, 3040, 3903, 4927, 8978, 3021, 6055, 4139, 10036, 11098, 6270, 11027, 6345, 8106, 9099, 10205, 12556, 5512, 6803, 1377, 9092, 5656, 10835, 1666, 1497, 2786, 8374, 617, 11392, 7274, 12588, 2099, 7806, 5683, 9633, 4534, 3465, 3498, 9718, 9314, 12622, 9169, 3205, 6981, 7405, 7438, 2908, 9556, 4091, 8089, 8158, 6470, 1065, 10931, 2163, 8725, 5693, 10485, 9782, 1310, 4635, 3127, 10675, 2311, 9157, 5463, 5711, 8025, 6438, 8713, 10559, 10138, 7818, 7833, 1250, 5276, 3783, 6447, 9533, 5468, 9855, 8237, 6913, 3995, 7229, 11321, 12153, 11201, 5555, 1425, 2568, 2467, 7535, 2164, 10223, 8428, 9299, 6791, 10807, 9885, 5291, 2238, 2639, 12621, 7181, 459, 6060, 4817, 6118, 10840, 8032, 11158, 2403, 11500, 9772, 6059, 7445, 11729, 5708, 1681, 8286, 4566, 11970, 6084, 5579, 10802, 10415, 3779, 8369, 5785, 4501, 8613, 150, 300, 10249, 11381, 12256, 6302, 7860, 11397, 302, 2574, 5615, 2098, 1406, 6839, 3898, 1516, 2187, 8982, 7346, 6522, 11547, 8690, 646, 7715, 666, 2177, 12027, 10149, 5634, 12052, 8342, 7230, 10648, 9697, 445, 3721, 1211, 2529, 12411, 6299, 3176, 11565, 7750, 12225, 10153, 2676, 1598, 5761, 6354, 8871, 10100, 9316, 1382, 6313, 3161, 8509, 3114, 3408, 3042, 1603, 5457, 12251, 11450, 1543, 3511, 5496, 12200, 3761, 5469, 10791, 9071]
    labels = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    # outputs, new_memoize = postgres_execute(dsn, current_query, memoize, inputs_table_name, list(range(300)), is_trajectory=False)
    query_str = "Far_1.1(o0, o1); Near_1.05(o0, o1)"
    current_query = str_to_program_postgres(query_str)
    outputs, new_memoize_scene_graph, new_memoize_sequence = postgres_execute_cache_sequence(dsn, current_query, memoize_scene_graph, memoize_sequence, inputs_table_name, input_vids, is_trajectory=True, sampling_rate=4)
    print(len(outputs))
    print(outputs)
    print("time", time.time() - _start)

    preds = []
    for input in input_vids:
        if input in outputs:
            preds.append(1)
        else:
            preds.append(0)
    score = f1_score(labels, preds)
    print(score)
    # for i, memo_dict in enumerate(new_memoize_scene_graph):
    #     for k, v in memo_dict.items():
    #         memoize_scene_graph[i][k] = v
    # for i, memo_dict in enumerate(new_memoize_sequence):
    #     for k, v in memo_dict.items():
    #         memoize_sequence[i][k] = v

    # query_str = "Far_3(o0, o1); Conjunction(Conjunction(Near_1(o0, o1), LeftQuadrant(o0)), Behind(o0, o1))"
    # current_query = str_to_program_postgres(query_str)
    # outputs, new_memoize_scene_graph, new_memoize_sequence = postgres_execute_cache_sequence(dsn, current_query, memoize_scene_graph, memoize_sequence, inputs_table_name, input_vids, is_trajectory=True, sampling_rate=4)
    # print(len(outputs))
    # print(outputs)
    # print("time", time.time() - _start)

    # preds = []
    # for input in input_vids:
    #     if input in outputs:
    #         preds.append(1)
    #     else:
    #         preds.append(0)
    # # score = f1_score(labels, preds)
    # # print(score)
    # for i, memo_dict in enumerate(new_memoize_scene_graph):
    #     for k, v in memo_dict.items():
    #         memoize_scene_graph[i][k] = v
    # for i, memo_dict in enumerate(new_memoize_sequence):
    #     for k, v in memo_dict.items():
    #         memoize_sequence[i][k] = v

    # query_str = "Far_3(o0, o1); Conjunction(Conjunction(Near_1(o0, o1), LeftQuadrant(o0)), Behind(o0, o1))"
    # current_query = str_to_program_postgres(query_str)
    # outputs, new_memoize_scene_graph, new_memoize_sequence = postgres_execute_cache_sequence(dsn, current_query, memoize_scene_graph, memoize_sequence, inputs_table_name, input_vids, is_trajectory=True, sampling_rate=4)
    # print(len(outputs))
    # print(outputs)
    # print("time", time.time() - _start)

    # preds = []
    # for input in input_vids:
    #     if input in outputs:
    #         preds.append(1)
    #     else:
    #         preds.append(0)
    # score = f1_score(labels, preds)
    # print(score)

    # current_query = [{'scene_graph': [{'predicate': 'LeftOf', 'parameter': None, 'variables': ['o0', 'o1']}], 'duration_constraint': 1}, {'scene_graph': [{'predicate': 'RightQuadrant', 'parameter': None, 'variables': ['o0']}], 'duration_constraint': 1}, {'scene_graph': [{'predicate': 'LeftQuadrant', 'parameter': None, 'variables': ['o2']}, {'predicate': 'RightOf', 'parameter': None, 'variables': ['o0', 'o1']}], 'duration_constraint': 1}]
    # query_str = "Duration(LeftOf(o0, o1), 5); Conjunction(Conjunction(Conjunction(Conjunction(Behind(o0, o2), Cyan(o2)), FrontOf(o0, o1)), RightQuadrant(o2)), Sphere(o2)); Duration(RightQuadrant(o2), 3)"
    # current_query = str_to_program_postgres(query_str)
    # memoize = [{} for _ in range(10000)]
    # inputs_table_name = "Obj_clevrer"
    # _start = time.time()
    # outputs, new_memoize = postgres_execute(dsn, current_query, memoize, inputs_table_name, 301, is_trajectory=False)
    # print("time", time.time() - _start)


    # _start = time.time()
    # outputs, new_memoize = postgres_execute(dsn, current_query, memoize, inputs_table_name, 300, is_trajectory=False)
    # print("time", time.time() - _start)
    # print(len(outputs))
    # print(outputs)
    # print(using("profile"))

    # for i, memo_dict in enumerate(new_memoize):
    #         for k, v in memo_dict.items():
    #             memoize[i][k] = v
    # _start = time.time()
    # outputs, new_memoize = postgres_execute(dsn, current_query, memoize, inputs_table_name, 301, is_trajectory=False)
    # print("time", time.time() - _start)
    # print(len(outputs))
    # print(outputs)
    # print(using("profile"))

    # # # TODO: benchmark: 1. reevaluate a query over 5 more new videos. 2. evaluate a query whose subqueries are already evaluated.