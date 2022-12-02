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
    duration_unit = 5
    cost_depth = len(program)
    cost_npred = sum([len(dict["scene_graph"]) for dict in program])
    cost_duration = sum([(dict["duration_constraint"] // duration_unit) * (1 + 0.1 * len(dict["scene_graph"])) for dict in program])
    # return cost_npred + cost_depth * 0.5 + cost_duration
    return cost_npred + cost_depth * 0 + cost_duration

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
                    cur.execute("CREATE TEMPORARY TABLE Obj_filtered AS SELECT oid, vid, fid / {} as fid, shape, color, material, x1, y1, x2, y2 FROM {} WHERE vid < {} AND fid % {} = 0;".format(sampling_rate, inputs_table_name, input_vids, sampling_rate))
                else:
                    cur.execute("CREATE TEMPORARY TABLE Obj_filtered AS SELECT * FROM {} WHERE vid < {};".format(inputs_table_name, input_vids))
                input_vids = list(range(input_vids))
            else:
                if sampling_rate:
                    cur.execute("CREATE TEMPORARY TABLE Obj_filtered AS SELECT oid, vid, fid / {} as fid, shape, color, material, x1, y1, x2, y2 FROM {} WHERE vid = ANY(%s) AND fid %% {} = 0;".format(sampling_rate, inputs_table_name, sampling_rate), [filtered_vids])
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
                sql_sring = """CREATE TEMPORARY TABLE g{} AS SELECT {} FROM {} WHERE {};""".format(graph_idx, fields, tables, where_clauses)
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

    memoize_scene_graph = [{} for _ in range(10000)]
    memoize_sequence = [{} for _ in range(10000)]
    inputs_table_name = "Obj_clevrer"
    # inputs_table_name = "Obj_collision"
    _start = time.time()
    input_vids = [8388, 4753, 1039, 1711, 1657, 8019, 8284, 2520, 5327, 9150, 2752, 2499, 2333, 5523, 6545, 1873, 6010, 1712, 7982, 8917, 232, 3861, 8065, 6326, 5501, 9604, 4362, 4447, 2398, 3486, 6572, 535, 539, 7784, 5092, 3053, 8504, 6820, 2765, 6752, 4029, 3155, 3399, 4220, 8746, 6178, 7217, 8440, 2596, 6529, 9051, 9765, 5199, 4314, 7406, 8737, 9658, 2693, 135, 6381, 4526, 8946, 4373, 2610, 9902, 56, 1234, 684, 9175, 8046, 8243, 1337, 1567, 7803, 1169, 907, 6244, 8407, 5655, 6850, 8819, 8316, 4967, 3812, 94, 3403, 2667, 9084, 1126, 7971, 259, 8573, 1628, 3522, 8115, 4527, 2191, 5674, 3090, 6461, 9379, 3824, 7899, 5801, 6561, 6453, 5771, 3075, 8144, 5010, 1732, 1018, 8666, 8014, 6774, 1342, 796, 4346, 4652, 144, 3519, 7401, 8380, 8587, 5461, 7508, 1191, 8272, 8634, 1976, 8203, 7572, 1922, 4499, 1405, 8220, 2652, 5218, 3340, 1885, 8916, 1434, 2841, 5615, 5312, 1214, 9907, 6608, 2450, 1680, 1896, 2032, 3634, 4184, 6434, 6227, 2683, 6963, 3164, 9989, 6715, 5460, 2780, 5175, 5877, 2971, 8663, 2304, 9495, 6120, 3802, 9486, 156, 2026, 5055, 1832, 2804, 7609, 3153, 1245, 9140, 9491, 2870, 5714, 4620, 7044, 8231, 5856, 2314, 6578, 9610, 6594, 8056, 545, 4065, 6729, 3138, 6257, 3044, 2349, 2040, 5658, 8521, 1226, 2849, 7718, 8077, 9258, 4651, 554, 3526, 6014, 4641, 752, 3424, 1790, 5251, 6251, 4909, 3951, 1684, 416, 2151, 4932, 6408, 2469, 3481, 6621, 7601, 7563, 8697, 1850, 402, 2671, 8566, 3088, 2027, 7690, 4012, 4399, 2982, 6436, 5617, 915, 5593, 6939, 1520, 9817, 6423, 3941, 6241, 7662, 3007, 5597, 7014, 9346, 362, 2500, 4660, 7761, 3179, 1328, 8153, 6318, 6470, 9015, 2510, 7991, 5693, 8605, 1032, 3814, 2805, 2298, 7631, 2570, 7054, 5219, 1637, 8790, 1908, 3449, 1298, 4280, 8066, 5290, 9012, 5866, 619, 6890, 6323, 3840, 53, 472, 4359, 8209, 3134, 1727, 1344, 441, 9213, 4626, 6377, 1650, 7589, 1007, 1563, 3282, 5721, 989, 2455, 8099, 8955, 8551, 9069, 9946, 1798, 6635, 978, 7140, 5511, 2660, 987, 6893, 9173, 9246, 4166, 2854, 8408, 8688, 7311, 9575, 6274, 4074, 9447, 4891, 8646, 8594, 8834, 4727, 4544, 4368, 2678, 9542, 6595, 3919, 2509, 9785, 9860, 5422, 8345, 8178, 1299, 913, 4264, 5036, 2830, 468, 2519, 2418, 7846, 7388, 7831, 3289, 6269, 7270, 2936, 7476, 4045, 3414, 3794, 4632, 1010, 4559, 551, 5081, 692, 7998, 2052, 8441, 1781, 9690, 8201, 9640, 7727, 3103, 904, 3483, 1467, 8840, 2697, 2167, 7442, 8367, 8947, 3354, 5383, 2672, 4787, 8602, 7325, 2981, 2261, 7837, 9443, 2952, 8745, 2977, 9392, 2091, 2589, 7716, 1695, 122, 4913, 645, 3238, 5124, 6350, 1152, 4150, 7521, 7125, 614, 2501, 8322, 9676, 412, 7172, 1323, 2867, 2600, 3970, 2864, 7296, 9768, 8422, 1725, 596, 196, 8818, 4363, 3326, 1440, 6940, 7278, 7266, 4682, 8914, 9916, 4185, 8870, 5228, 761, 644, 4782, 9894, 6507, 5646, 3336, 5551, 8221, 5602, 4623, 7425, 2833, 9232, 7613, 8876, 4560, 4995, 7397, 1135, 7932, 509, 5751, 564, 6961, 9534, 6843, 6728, 4639, 8932, 7747, 7412, 506, 9694, 7470, 9776, 8583, 9497, 2252, 2306, 8673, 641, 7630, 4607, 5215, 7657, 2327, 695, 2294, 200, 9341, 1636, 6662, 9756, 7479, 6930, 4041, 3580, 2944, 9511, 2232, 8945, 2770, 9087, 6688, 7853, 1274, 3378, 3380, 2254, 7679, 6487, 3817, 516, 3669, 9878, 7382, 4075, 1016, 3028, 7802, 1361, 6272, 1788, 6667, 5761, 7650, 3809, 1613, 2319, 2348, 369, 2669, 9170, 5516, 3241, 4558, 4887, 9702, 3280, 1124, 8789, 1624, 1688, 4615, 3780, 7733, 5458, 9013, 4070, 5515, 4348, 4376, 1541, 2352, 8353, 7707, 2756, 8608, 7966, 5948, 6290, 3984, 1941, 6603, 3463, 6334, 6921, 9783, 7081, 1978, 3866, 2106, 1821, 9770, 8637, 6935, 7714, 3361, 8000, 2003, 4649, 708, 7724, 865, 962, 3557, 4576, 9915, 1379, 5431, 7596, 9081, 7634, 3798, 1648, 8510, 3244, 2185, 5533, 6605, 2169, 6335, 6637, 6773, 3163, 9624, 8741, 5882, 3978, 7396, 8926, 2807, 4261, 9963, 5772, 3453, 3727, 4296, 7407, 7029, 5647, 7755, 4785, 2245, 8865, 3655, 9578, 2709, 8145, 2893, 33, 329, 2275, 4945, 133, 72, 6742, 2405, 115, 3291, 4060, 6271, 1584, 4866, 8983, 7214, 9478, 1247, 8082, 7788, 1161, 9651, 5099, 7159, 8590, 5982, 9074, 788, 7694, 941, 8073, 9264, 7395, 1795, 7523, 8426, 4634, 520, 4673, 7874, 5692, 5755, 3312, 5626, 277, 9352, 2712, 4271, 1115, 1041, 853, 5101, 4203, 8544, 5645, 4528, 5733, 160, 5091, 1867, 6994, 6397, 2336, 9689, 2917, 6422, 3234, 6388, 9959, 4555, 7639, 3595, 4795, 4353, 4420, 1528, 6382, 7522, 1055, 8885, 8406, 4519, 920, 6975, 8679, 9820, 4665, 5718, 5509, 5644, 1980, 5494, 1355, 79, 9072, 7440, 6211, 4400, 5842, 7182, 5652, 789, 665, 5673, 6395, 3610, 248, 5904, 7424, 5173, 5112, 9589, 6456, 5878, 5975, 5477, 1414, 4419, 7347, 6433, 7986, 5067, 4722, 6086, 8585, 1545, 5830, 9142, 4766, 6950, 8064, 2913, 651, 8237, 301, 4003, 6807, 3668, 3615, 8044, 7004, 4434, 2173, 1529, 1891, 6573, 384, 1111, 1951, 5379, 3577, 8904, 6974, 5915, 124, 3799, 9972, 4023, 7384, 8093, 7108, 7011, 7034, 6988, 2188, 9237, 3324, 8432, 9348, 7745, 8629, 5745, 6990, 3905, 8461, 5640, 263, 505, 4024, 5415, 9982, 6588, 6549, 6062, 6596, 2590, 1829, 3314, 7453, 8188, 2011, 5028, 8866, 3452, 4638, 2914, 9257, 772, 8370, 1570, 9880, 9725, 5765, 5224, 2038, 2471, 4636, 8507, 5956, 5406, 5062, 8260, 2466, 922, 7438, 3296, 2062, 9490, 1263, 5160, 960, 9523, 5520, 5424, 3747, 3741, 5052, 1424, 2136, 5957, 2439, 6647, 6779, 6125, 7076, 8911, 634, 9251, 8239, 5293, 3228, 7310, 2342, 8570, 8022, 5329, 3395, 4442, 1898, 6902, 6791, 7366, 1954, 182, 8820, 4409, 9025, 5167, 6191, 7355, 8564, 7715, 4237, 912, 6079, 8172, 3837, 4483, 9719, 6685, 4541, 6873, 6936, 1149, 7301, 3584, 3186, 9304, 5319, 7416, 7142, 1579, 6213, 9007, 7431, 414, 2226, 3442, 5797, 624, 4479, 4183, 9146, 8319, 5282, 1849, 3725, 6192, 2615, 6518, 4962, 7219, 8502, 1438, 6428, 1164, 9121, 3345, 1201, 9897, 696, 8257, 6256, 4836, 3445, 8483, 3343, 6514, 6513, 227, 640, 849, 4781, 9787, 210, 5453, 3957, 3204, 9891, 1501, 5187, 9969, 8803, 2158, 261, 4738, 6217, 6519, 2934, 7898, 6073, 4401, 1987, 171, 7705, 1217, 2748, 9449, 2329, 6042, 2511, 476, 6310, 9901, 6919, 8671, 6187, 6708, 1118, 15, 4713, 1834, 4952, 6900, 2096, 1472, 502, 3640, 3908, 3598, 8379, 7938, 4481, 443, 8852, 808, 783, 3920, 3477, 4130, 2210, 7990, 2276, 4470, 8307, 4054, 5783, 273, 726, 2891, 7530, 3832, 3189, 7791, 3932, 2414, 9808, 4176, 3983, 6312, 8684, 9600, 5133, 7008, 1155, 7235, 5283, 4082, 7935, 3835, 2131, 7465, 281, 8137, 6308, 7685, 6672, 8786, 671, 6942, 7638, 5505, 6730, 2865, 5360, 9564, 8301, 2070, 2058, 3218, 4278, 4595, 7212, 4427, 2616, 2766, 4893, 7074, 2657, 843, 4971, 9711, 9394, 5730, 1703, 1871, 5985, 2516, 5131, 5007, 8499, 6415, 5343, 1159, 2633, 4157, 9874, 3781, 6971, 9555, 738, 1311, 6401, 3560, 5867, 7342, 9510, 6023, 3889, 8884, 6771, 7750, 9996, 1797, 4611, 851, 3637, 6837, 3287, 395, 749, 5793, 7771, 552, 3107, 3248, 8545, 230, 2122, 2638, 5284, 1720, 7010, 4394, 9784, 3370, 1401, 3525, 8166, 4287, 1458, 2377, 7792, 2782, 370, 9225, 185, 6844, 3158, 5316, 4231, 3761, 8286, 73, 3478, 4872, 6020, 3247, 1142, 8058, 5321, 5534, 6526, 9356, 1293, 4448, 4919, 2395, 6754, 7687, 4217, 9162, 5562, 6530, 9858, 4013, 6221, 7152, 43, 492, 4266, 1345, 3726, 6887, 9691, 944, 2688, 5291, 1753, 1451, 4182, 3909, 3501, 3914, 6671, 2016, 9607, 1722, 6477, 8100, 5870, 2300, 1638, 8645, 9408, 3394, 4113, 2724, 8864, 8329, 5323, 6860, 5061, 8134, 3859, 1212, 6222, 121, 6601, 1203, 6922, 2465, 7243, 1824, 9113, 8336, 9753, 6973, 2513, 952, 8657, 2463, 318, 6172, 3633, 9745, 4778, 909, 4126, 1815, 1875, 3193, 7532, 9885, 8961, 4904, 8383, 142, 8835, 7948, 9543, 3438, 1617, 4873, 5045, 2843, 5767, 5150, 8556, 8901, 3077, 1374, 4329, 450, 3451, 8781, 839, 1024, 1914, 2285, 4972, 5440, 1479, 4282, 6849, 700, 8397, 805, 6665, 5021, 5401, 8777, 6630, 478, 3856, 3455, 7389, 3398, 6160, 4223, 7804, 9492, 5815, 8001, 6699, 1105, 5413, 6992, 7209, 7409, 3542, 5577, 3663, 1967, 7576, 3497, 6279, 4462, 22, 9499, 7901, 5069, 332, 7475, 5192, 5419, 5492, 7597, 1237, 8011, 5552, 4179, 7279, 3569, 3489, 3426, 1942, 5607, 9241, 8238, 3013, 2926, 1430, 7676, 1125, 3076, 1215, 5356, 2028, 8843, 2368, 4720, 8186, 4487, 4583, 5361, 5576, 5084, 3827, 5196, 1004, 6078, 4473, 3142, 9826, 6698, 5337, 1813, 5006, 9, 2472, 2714, 5365, 5628, 4001, 2617, 7404, 1108, 5039, 1810, 1990, 566, 1422, 5313, 3320, 6853, 7681, 4635, 7263, 2768, 7602, 5660, 5041, 7481, 3949, 3981, 2677, 7329, 25, 2009, 5110, 5336, 7413, 394, 3245, 4943, 8482, 5182, 6103, 8668, 4088, 4843, 8859, 1113, 1585, 430, 5953, 1819, 784, 2908, 4283, 9159, 4888, 3300, 1101, 8910, 6267, 9199, 7587, 6420, 253, 8302, 2066, 2727, 7240, 9344, 9570, 2924, 2591, 8004, 605, 4994, 4631, 9956, 2899, 9981, 3004, 6731, 2628, 7558, 8365, 5857, 937, 9484, 633, 262, 8200, 1410, 7489, 845, 6783, 9906, 6949, 4663, 765, 5950, 892, 1982, 3955, 2916, 544, 6985, 7430, 7665, 2737, 5153, 4250, 9393, 372, 7670, 6486, 7095, 9751, 9671, 8987, 8980, 3368, 2960, 7250, 4206, 1198, 5342, 5486, 5065, 2130, 3654, 3763, 2112, 177, 730, 5075, 3929, 4277, 7824, 953, 6607, 5202, 1669, 7480, 6645, 9338, 1148, 7908, 8360, 9675, 137, 9664, 4417, 20, 4233, 7560, 2420, 8154, 5122, 1956, 9459, 734, 9174, 8754, 5754, 2675, 1098, 1400, 979, 6660, 870, 3092, 1889, 7752, 6085, 8763, 6391, 2527, 5103, 7700, 5666, 9990, 2184, 9296, 5814, 7381, 1830, 2546, 9541, 2401, 5053, 8943, 6827, 4488, 439, 2647, 5778, 9475, 3124, 2562, 7775, 6527, 4897, 7890, 9839, 2581, 4774, 2989, 3214, 6165, 908, 2315, 6846, 4658, 3554, 3108, 5589, 859, 1686, 2088, 1679, 8807, 278, 1486, 7721, 4215, 5583, 5303, 1697, 3420, 6253, 6242, 9876, 327, 966, 6029, 3386, 727, 4490, 7813, 3645, 2216, 2140, 4159, 1420, 5967, 1393, 3739, 3770, 9766, 4093, 7443, 6410, 3792, 5180, 8400, 1364, 4944, 3883, 4832, 8457, 1911, 4758, 7364, 7856, 6692, 3118, 3258, 9371, 6899, 2565, 8340, 9253, 2825, 3511, 8453, 9053, 4902, 1843, 4643, 7538, 9970, 1383, 8222, 5705, 3099, 7306, 4142, 4163, 3821, 6205, 7790, 7709, 5498, 7114, 856, 9323, 4540, 7123, 7608, 4274, 8728, 3574, 9228, 5702, 4032, 802, 6223, 3776, 4281, 4421, 8536, 9105, 6878, 4546, 1080, 5543, 6224, 1140, 7028, 184, 2515, 9596, 6479, 482, 6769, 8863, 9123, 7031, 4265, 97, 2786, 4930, 1314, 3871, 2135, 7056, 9155, 6081, 2995, 5544, 4047, 307, 1412, 1147, 4520, 9467, 8280, 1745, 9616, 530, 4863, 9556, 5787, 4468, 2129, 1252, 9557, 7628, 3609, 4262, 4964, 2815, 4316, 9859, 6240, 88, 1074, 3437, 406, 5585, 9134, 4751, 8891, 6734, 6212, 9769, 8595, 816, 431, 4946, 8887, 5435, 4201, 8116, 9268, 2370, 8627, 2488, 8168, 8382, 9014, 234, 947, 5266, 8123, 2080, 3085, 7302, 5095, 6898, 6439, 6203, 7703, 8470, 4936, 918, 5822, 7584, 6937, 8920, 8098, 419, 280, 1002, 317, 7110, 1134, 388, 2295, 9529, 8076, 3971, 5478, 724, 3512, 6229, 6039, 5996, 5944, 9423, 4868, 5664, 1981, 7815, 2845, 6539, 3317, 4227, 6743, 8460, 8224, 4119, 3491, 1307, 5141, 1011, 5786, 9767, 4951, 5559, 4052, 4445, 6780, 3502, 5845, 6374, 489, 8816, 5860, 8680, 4492, 5038, 1326, 2108, 8091, 946, 9993, 5137, 1123, 151, 5391, 7915, 4886, 4456, 2291, 2010, 6785, 8263, 2434, 8087, 9961, 1484, 682, 1289, 3348, 3177, 9076, 9432, 2852, 8250, 2445, 4105, 7203, 2037, 537, 9362, 9618, 9458, 7994, 3484, 9977, 6430, 8641, 3106, 178, 9124, 2909, 6634, 7261, 4716, 2986, 2686, 4798, 9779, 6792, 31, 5109, 4644, 9056, 7215, 381, 5248, 3599, 174, 5377, 933, 8775, 7806, 4081, 9682, 1812, 7625, 9617, 9585, 1001, 4839, 6929, 1088, 7052, 3868, 2614, 4459, 2858, 2474, 9317, 247, 70, 4254, 2689, 1273, 9986, 5475, 2060, 3307, 7248, 5514, 5113, 9165, 984, 9943, 4565, 2411, 3260, 2723, 3988, 1444, 7659, 8418, 3032, 2311, 4584, 1280, 8971, 917, 9270, 5118, 6336, 4255, 5700, 894, 5322, 8631, 4827, 4937, 2887, 931, 5684, 5472, 1120, 9886, 4007, 3096, 8180, 3678, 267, 3467, 7331, 8245, 1427, 2997, 9829, 995, 2918, 3878, 4912, 638, 5400, 7089, 2973, 4209, 2278, 6835, 7098, 5696, 5902, 852, 3660, 4511, 87, 9439, 7090, 6813, 8774, 6127, 8606, 9704, 3333, 8303, 3934, 7877, 228, 8642, 3443, 6802, 6830, 4752, 4532, 4370, 1271, 7776, 5662, 2785, 7376, 2109, 5773, 9602, 9856, 4935, 5448, 4026, 8471, 4867, 8312, 7053, 2577, 4365, 3756, 1036, 4610, 7680, 7756, 6543, 3629, 7345, 6767, 2264, 8640, 706, 9149, 9096, 1932, 6801, 3996, 5819, 5436, 5561, 4585, 8748, 188, 6353, 787, 7314, 6938, 2961, 7284, 7970, 1672, 493, 9397, 1286, 6293, 3818, 5855, 9870, 1468, 4147, 2332, 3401, 3482, 9531, 2432, 2817, 5308, 4452, 9415, 2597, 613, 8309, 2802, 6585, 69, 4294, 5779, 7120, 8954, 3005, 8732, 5582, 3313, 4898, 1432, 6536, 8363, 7622, 4108, 4791, 2927, 9435, 7916, 3428, 5708, 6457, 5722, 4724, 2152, 7073, 1417, 3285, 4592, 9101, 9574, 9538, 2940, 1763, 9303, 6402, 2572, 8849, 4084, 1435, 2180, 126, 8085, 5740, 3848, 8912, 343, 4158, 7434, 9866, 2985, 9343, 9504, 5177, 3136, 4455, 2162, 7797, 2739, 9441, 867, 2970, 2798, 2489, 1925, 2850, 7294, 1301, 3267, 366, 2806, 8253, 1221, 7418, 4507, 9017, 3772, 9581, 6865, 1612, 2449, 7096, 1287, 7067, 577, 5579, 5603, 6571, 4297, 7910, 8193, 731, 1687, 8458, 3071, 6817, 4162, 8487, 4966, 6577, 6234, 1242, 7566, 5978, 3450, 3029, 7356, 1377, 4384, 5654, 6788, 3374, 9342, 7379, 2061, 9905, 3194, 5490, 3882, 7513, 9483, 3716, 7051, 484, 9425, 7667, 611, 1510, 2958, 2788, 9871, 9983, 6169, 8944, 5375, 6681, 2571, 667, 763, 8439, 2627, 1300, 4506, 7307, 3676, 7267, 9789, 2508, 9351, 9410, 9366, 187, 8466, 9833, 221, 4397, 1308, 2492, 4813, 4324, 3627, 3212, 621, 4180, 6385, 5863, 304, 3249, 8411, 8413, 985, 2206, 9434, 4513, 5209, 3901, 9888, 1270, 4572, 9633, 6653, 3215, 4578, 9697, 8922, 7168, 8833, 7170, 6881, 4037, 155, 2972, 8121, 9274, 7183, 7361, 1519, 9663, 1764, 4129, 9772, 9223, 662, 3732, 5015, 2708, 2391, 8249, 3576, 9978, 426, 8950, 5032, 3281, 2991, 2929, 3009, 7174, 8698, 4606, 3566, 7562, 32, 8271, 5976, 7947, 6554, 3068, 7964, 6260, 2832, 9934, 2269, 9506, 8574, 663, 4870, 7632, 2422, 8592, 1249, 1966, 4871, 6999, 6496, 5600, 4711, 7160, 3480, 8691, 1571, 6035, 8822, 7422, 4489, 9252, 3551, 2118, 2998, 7992, 6811, 5115, 9188, 1747, 2935, 9666, 7652, 5210, 3785, 9777, 6944, 496, 257, 4042, 3396, 6687, 2397, 4194, 8874, 2796, 1139, 385, 5905, 8869, 6738, 6261, 3073, 2345, 9045, 8601, 4709, 7319, 8315, 9456, 9469, 4168, 5633, 815, 1086, 2681, 833, 6145, 120, 3953, 3991, 598, 9971, 311, 874, 7369, 3980, 8756, 8897, 3969, 9020, 1943, 7773, 215, 6740, 1048, 6000, 3815, 6119, 1265, 9791, 2256, 4132, 593, 7830, 1054, 3892, 3616, 8614, 9595, 7943, 6173, 9322, 5497, 47, 6597, 5557, 9255, 6076, 7405, 1678, 6258, 3219, 9632, 3371, 7195, 2322, 4110, 7385, 8196, 423, 6007, 8175, 750, 288, 3973, 2569, 296, 6954, 4755, 5247, 8140, 4048, 1740, 5326, 9191, 5429, 6909, 1621, 9421, 4358, 9280, 3950, 8939, 6455, 6516, 292, 2965, 6819, 1386, 6497, 1719, 2755, 6064, 7629, 1333, 4771, 6508, 8617, 9921, 3664, 2170, 1103, 9369, 4811, 6196, 9571, 8206, 7188, 5883, 6306, 6389, 7975, 9724, 3690, 1544, 7833, 5817, 3976, 9010, 7995, 3688, 5818, 24, 693, 9446, 5712, 7876, 5750, 6541, 5827, 5221, 9514, 8785, 3538, 9806, 1879, 923, 521, 3939, 7411, 5405, 2711, 5206, 2548, 3604, 3233, 3684, 6442, 6194, 3033, 4311, 3797, 2695, 2820, 2436, 8015, 3360, 3270, 3392, 881, 1526, 473, 4689, 2239, 7981, 4743, 1771, 5051, 4800, 8372, 224, 4554, 9367, 2270, 1000, 5314, 5506, 5439, 6686, 741, 3154, 6049, 8979, 9178, 113, 2231, 8398, 735, 963, 6041, 9308, 1989, 8202, 5639, 3411, 1554, 2613, 6967, 2620, 4876, 3010, 1480, 4433, 8179, 474, 3850, 6262, 549, 2430, 3444, 2992, 5756, 1901, 2903, 1170, 4838, 7206, 4764, 9471, 7358, 6705, 3683, 2187, 3917, 3704, 3923, 7147, 2208, 3304, 5479, 4679, 1907, 5325, 7427, 2113, 3115, 8081, 6250, 7586, 2861, 8279, 1558, 7963, 7156, 5711, 6520, 7002, 7220, 4131, 5123, 5848, 4847, 1248, 1677, 9436, 2502, 6804, 9107, 1627, 4882, 7313, 5085, 6694, 3192, 8157, 9634, 7016, 8611, 7460, 8474, 7099, 7795, 8722, 4705, 9649, 1057, 8856, 5663, 4286, 2124, 2964, 4517, 742, 7692, 5249, 5872, 4491, 3621, 6379, 6673, 3000, 7595, 1905, 2236, 5921, 6610, 8324, 7951, 2575, 3416, 3344, 2477, 4055, 3902, 3846, 3775, 1367, 5488, 7458, 4762, 9473, 5016, 6467, 8039, 6091, 7577, 5370, 6099, 291, 2859, 3933, 3286, 6378, 9117, 1762, 1321, 2186, 382, 9788, 9786, 585, 8187, 3779, 6476, 6875, 8586, 2043, 5804, 8603, 8703, 5299, 9457, 8204, 1071, 8385, 3594, 8273, 3762, 2822, 7372, 2394, 9028, 9780, 8708, 8751, 6569, 7835, 2378, 3529, 4195, 677, 8399, 1630, 17, 8072, 6089, 775, 7977, 7731, 4564, 1456, 3903, 255, 4173, 2125, 7297, 6285, 1683, 9739, 9329, 2618, 6903, 5980, 8528, 4568, 4412, 7646, 4263, 4880, 4822, 2966, 3608, 869, 6032, 6983, 2053, 6, 3631, 1938, 7660, 9942, 9760, 1842, 387, 8793, 6759, 3677, 9881, 7285, 8412, 4589, 8718, 6447, 6003, 4612, 52, 639, 721, 7499, 6669, 2128, 2679, 8893, 5445, 4175, 4038, 7223, 7647, 6550, 6747, 9339, 2035, 1428, 4925, 9764, 513, 927, 4219, 407, 8219, 168, 8780, 9854, 3440, 4033, 2824, 429, 5554, 6617, 7517, 7368, 8711, 1514, 7720, 3332, 8051, 9505, 5107, 7048, 5049, 9748, 2713, 1050, 3642, 9650, 4350, 4984, 6102, 8215, 885, 5838, 8075, 8127, 681, 9204, 8071, 1847, 1522, 6998, 9479, 4245, 2912, 988, 5087, 1327, 1851, 1663, 1632, 7973, 2963, 3992, 6584, 9613, 6069, 8259, 8581, 4039, 5111, 3127, 9377, 3097, 685, 6177, 7463, 5096, 6443, 4780, 3672, 2829, 5914, 3658, 2323, 6609, 9775, 9033, 9391, 3744, 6296, 5695, 9709, 7383, 6547, 5054, 8969, 4596, 9202, 1107, 6058, 1766, 8149, 6136, 1594, 5411, 5446, 8632, 906, 3588, 7193, 9154, 5675, 8255, 9454, 7653, 5812, 7525, 2793, 6517, 9665, 5469, 7177, 9561, 6065, 3023, 8334, 8588, 5701, 4957, 1652, 6655, 9422, 9273, 792, 2609, 9554, 3896, 7520, 8838, 5339, 7371, 123, 8053, 725, 4144, 8653, 195, 2145, 3555, 1378, 8600, 4927, 2229, 4767, 8496, 5205, 9964, 6458, 3738, 2673, 2931, 3319, 5572, 6882, 7739, 2580, 4987, 380, 965, 6510, 7231, 1730, 428, 4772, 1803, 6128, 2468, 6084, 9080, 1448, 1166, 728, 2214, 7506, 2624, 8655, 1116, 147, 3100, 5093, 9401, 4539, 8347, 1272, 3069, 1276, 2863, 2836, 3967, 1216, 4702, 1944, 323, 1027, 5910, 873, 2005, 2549, 1119, 3940, 5560, 7961, 1527, 4285, 7759, 3387, 470, 9494, 5747, 4963, 5140, 3759, 5484, 4312, 5737, 3223, 8162, 3562, 5474, 9522, 1070, 8689, 1409, 5686, 2483, 3446, 8258, 6121, 5004, 6857, 3895, 3169, 3243, 5077, 3733, 2219, 4476, 4310, 8404, 65, 1317, 5573, 970, 8847, 8003, 4091, 9718, 7789, 7514, 5009, 8896, 9799, 3938, 4725, 6359, 2429, 8707, 5047, 4450, 7610, 2691, 7439, 3037, 5014, 9896, 9186, 5609, 5297, 2470, 5699, 1924, 8702, 2392, 7800, 4501, 8373, 3540, 7065, 3436, 9884, 4697, 6920, 3309, 1488, 4629, 8696, 8700, 3018, 7116, 8436, 972, 6278, 8975, 2923, 7672, 929, 3987, 4034, 7162, 1776, 2337, 6502, 4820, 948, 3213, 594, 2433, 6435, 7585, 7079, 6028, 6185, 9524, 2197, 2063, 9164, 7466, 9218, 4650, 9299, 3702, 4327, 9848, 924, 9418, 4825, 5545, 7822, 3935, 5142, 4823, 1165, 8727, 2194, 3740, 7860, 3133, 6463, 2978, 1359, 8333, 2872, 7959, 7055, 6744, 2428, 9525, 6540, 1744, 7006, 7924, 360, 3149, 4000, 7184, 9818, 8758, 3315, 1079, 8212, 8832, 1095, 4099, 751, 658, 5758, 5981, 154, 4160, 5525, 1494, 451, 245, 6321, 7952, 4885, 6199, 2915, 687, 3944, 4234, 7143, 1959, 1948, 8344, 9363, 550, 4477, 235, 7247, 3509, 3537, 7738, 8991, 5803, 5414, 4850, 1675, 7322, 9683, 6826, 102, 1498, 5354, 4063, 8094, 6396, 4728, 1352, 7419, 5310, 2556, 7957, 4505, 1939, 5859, 9738, 9419, 2490, 5217, 2659, 9102, 6599, 3302, 3705, 4028, 422, 8674, 3310, 4571, 3730, 2922, 4068, 4948, 2068, 512, 8190, 4268, 7729, 413, 655, 3543, 2290, 4073, 5452, 9503, 7543, 2201, 5906, 5330, 7423, 8476, 8630, 7575, 8289, 2380, 7836, 432, 826, 7179, 5238, 7333, 576, 447, 1682, 5387, 8620, 6789, 8787, 1822, 5670, 1786, 9728, 1073, 9021, 8690, 4735, 959, 1649, 8113, 4575, 1610, 1565, 8389, 7258, 6140, 9256, 166, 9601, 2150, 6333, 4809, 2146, 9698, 1804, 2543, 2583, 5592, 4341, 6151, 6247, 7167, 1536, 5309, 8027, 9535, 3421, 340, 5371, 5807, 1717, 6724, 2312, 518, 9630, 3094, 3922, 7059, 1350, 7556, 6488, 5724, 8356, 4574, 646, 4624, 4598, 4674, 6675, 420, 4929, 6130, 2967, 9684, 275, 2988, 9384, 1865, 4392, 4669, 1705, 7695, 5916, 2435, 2884, 7398, 5042, 4980, 3643, 565, 5986, 4252, 4308, 2731, 7633, 1443, 8772, 2604, 7805, 3120, 5768, 3057, 595, 3131, 1801, 5841, 1035, 5631, 6313, 2237, 5522, 9126, 1957, 7062, 2099, 511, 7163, 6583, 5809, 9489, 4300, 3012, 7077, 4145, 6799, 9914, 8092, 2778, 2317, 6231, 3235, 1223, 220, 4956, 2980, 5252, 1483, 1232, 3749, 9852, 209, 4332, 7669, 9807, 3534, 8449, 1357, 3700, 6452, 7070, 5527, 8218, 2771, 1066, 872, 8403, 6097, 9722, 1870, 8766, 1937, 1253, 4734, 7501, 8335, 1741, 7287, 1173, 777, 8029, 21, 5048, 7754, 8040, 4518, 9381, 3527, 4664, 5710, 5558, 3523, 5508, 4642, 5444, 3662, 1560, 6181, 6188, 3495, 7289, 8158, 5040, 9659, 3880, 4103, 9184, 8813, 8131, 4701, 4102, 9354, 5734, 2297, 5302, 747, 6122, 9472, 9976, 2351, 9790, 4165, 5442, 8539, 334, 994, 6854, 1143, 7561, 236, 5731, 7151, 8008, 2456, 5539, 6766, 240, 8503, 1702, 1368, 4857, 3294, 457, 4812, 6576, 5154, 2339, 8265, 9209, 743, 9426, 9861, 3717, 5256, 7760, 8048, 794, 7020, 3904, 7642, 4258, 9873, 2281, 1381, 2808, 9732, 7753, 3635, 9620, 64, 7840, 709, 4425, 8486, 8478, 6953, 2599, 4418, 7934, 3157, 5638, 5974, 6057, 4550, 864, 2049, 3017, 9388, 2888, 7022, 6398, 401, 8002, 8462, 5208, 8907, 9250, 2636, 8542, 1222, 4136, 480, 773, 558, 8924, 7615, 1940, 2921, 9376, 1961, 1920, 527, 4426, 3712, 2920, 1174, 5694, 5318, 5271, 4146, 9939, 1876, 1699, 4799, 8350, 2521, 8232, 2625, 9382, 7493, 3851, 6030, 408, 4680, 5876, 9631, 3337, 4903, 6166, 9773, 297, 9648, 5189, 9974, 3493, 2172, 7746, 4757, 2366, 2498, 8623, 1757, 9577, 5614, 3295, 6454, 9657, 2213, 2745, 1736, 4414, 7814, 6979, 7432, 8394, 352, 322, 100, 8079, 8644, 4403, 1665, 309, 5667, 5100, 7764, 2946, 3366, 9307, 8191, 2075, 6763, 1433, 2425, 5888, 5862, 2008, 5507, 9215, 9295, 6315, 8393, 2198, 6674, 1172, 8369, 4236, 152, 4579, 1864, 9944, 4736, 373, 1026, 8223, 8982, 6506, 3692, 5850, 2732, 879, 4361, 3325, 1668, 7816, 2530, 6663, 3305, 6982, 8495, 8794, 4677, 6417, 3830, 7849, 98, 6209, 2020, 6340, 6303, 8017, 3078, 7617, 7496, 5491, 8057, 8941, 548, 2097, 6749, 3067, 7039, 9276, 9402, 5409, 8857, 9893, 2582, 4990, 7417, 8519, 5566, 5483, 6642, 8936, 4046, 8892, 1562, 2301, 568, 4398, 2133, 9065, 8548, 2827, 3647, 649, 9364, 8246, 8446, 1550, 6357, 7867, 5104, 7483, 9451, 3583, 7202, 8074, 7387, 5068, 9011, 5211, 6297, 5288, 5005, 4808, 2525, 2626, 6460, 2361, 4683, 7534, 7158, 1030, 1532, 9285, 9405, 2938, 1877, 7854, 8067, 6531, 6679, 8796, 3356, 5237, 6869, 5078, 3891, 9476, 4928, 2373, 7829, 8721, 8712, 8633, 6168, 3147, 2313, 12, 5060, 5230, 35, 4199, 4380, 1186, 4855, 204, 609, 2892, 2481, 9653, 2898, 354, 9412, 4290, 1117, 4988, 7706, 8672, 7905, 2023, 4372, 6011, 2341, 4881, 3731, 1581, 3605, 1831, 4408, 6657, 3928, 2668, 3541, 2310, 1392, 8806, 8628, 5470, 5820, 4336, 4826, 6146, 146, 483, 9865, 9985, 7315, 1840, 2630, 1894, 6883, 7420, 1042, 9836, 3479, 1909, 8949, 1459, 7180, 2684, 9995, 6170, 8815, 9318, 4293, 2740, 620, 290, 9233, 1524, 5151, 8402, 8425, 4837, 3040, 2177, 9265, 6619, 4153, 8030, 8827, 4959, 1017, 4465, 7471, 6877, 8853, 4523, 9396, 3388, 2467, 8080, 1709, 8625, 1069, 5197, 7305, 8826, 9953, 5201, 252, 9559, 3065, 1077, 6139, 932, 5134, 7892, 3801, 465, 6235, 3168, 1857, 2054, 6946, 7320, 2886, 4894, 3800, 1542, 4796, 1573, 6108, 6017, 4288, 1151, 1089, 3473, 3172, 3423, 2794, 6282, 2497, 8650, 8024, 7186, 6100, 9288, 1787, 3579, 9721, 6226, 8128, 2179, 8814, 6468, 2930, 9563, 9024, 6087, 5575, 2522, 3425, 5226, 2763, 59, 6970, 7272, 9407, 347, 7507, 8549, 2559, 9453, 8724, 7426, 7993, 5738, 2379, 3961, 1266, 4789, 1106, 6060, 2446, 4148, 4953, 6432, 2353, 8391, 3367, 1290, 5861, 2704, 5519, 7702, 2116, 8541, 3224, 5810, 7751, 9235, 6053, 6206, 2994, 5620, 9680, 1892, 1658, 5232, 7743, 4019, 49, 8996, 716, 5193, 2881, 7135, 1340, 48, 1681, 1626, 5473, 2000, 167, 9156, 5162, 4112, 2760, 2102, 2875, 1478, 5798, 7786, 2074, 1127, 1511, 3810, 9266, 7105, 4301, 2134, 9428, 4135, 477, 7606, 6923, 3552, 4120, 8463, 583, 8514, 3916, 3293, 762, 4218, 358, 1045, 6161, 9980, 7091, 8182, 5865, 3713, 8526, 4374, 6239, 2360, 584, 5080, 9792, 7713, 9591, 5181, 7001, 2495, 3137, 5570, 1502, 1296, 7618, 5285, 6246, 5531, 9417, 7682, 2014, 4593, 2358, 282, 1661, 2227, 3734, 205, 3433, 1006, 8308, 6927, 9917, 1910, 9009, 8669, 9619, 2072, 8228, 7535, 8993, 8384, 302, 9637, 7275, 4807, 3897, 3556, 7699, 4668, 4444, 6620, 2814, 6894, 5335, 5390, 9158, 405, 4211, 3786, 2141, 1183, 8770, 3782, 6322, 8368, 4295, 877, 6858, 2030, 9429, 4786, 6916, 6245, 8562, 75, 7130, 9599, 3795, 9979, 80, 3182, 5829, 1964, 5627, 850, 848, 4547, 1662, 427, 1838, 3250, 1710, 580, 1756, 1853, 6056, 3469, 6845, 8515, 7827, 785, 1178, 9882, 3864, 6360, 4438, 5027, 9325, 6363, 3925, 2324, 8477, 5669, 4388, 936, 175, 1841, 1466, 5135, 2821, 363, 7510, 2811, 6812, 2334, 5171, 3126, 1979, 9030, 9211, 3329, 2454, 7946, 9932, 8520, 5417, 5943, 3384, 5643, 7809, 737, 9562, 9948, 5691, 8555, 9975, 6925, 6394, 3767, 8114, 6158, 1694, 5399, 5912, 9136, 4848, 8291, 3573, 7273, 8359, 5301, 27, 5530, 6932, 8593, 557, 4533, 6891, 5146, 6968, 9512, 9731, 9203, 7774, 1866, 1331, 2937, 1496, 8543, 2753, 5874, 165, 8749, 9800, 2639, 8141, 4529, 3435, 9261, 6112, 7392, 9293, 7900, 3457, 3351, 9245, 2789, 2828, 1282, 6314, 3298, 5726, 391, 954, 8894, 2123, 5464, 2523, 4896, 1460, 5689, 2799, 3813, 9001, 842, 2987, 2475, 8465, 2259, 7619, 6521, 5723, 2073, 590, 8681, 2690, 2217, 1439, 8951, 9309, 8225, 418, 2750, 2640, 1926, 55, 1193, 5043, 1291, 5125, 6856, 6941, 7737, 1583, 6709, 2335, 801, 6380, 9263, 4221, 1906, 6806, 914, 7349, 8267, 9311, 7497, 3546, 6782, 1415, 5476, 4841, 4200, 6842, 393, 9580, 6505, 3911, 4655, 8765, 2267, 8306, 7603, 5930, 7838, 9605, 2894, 6483, 9082, 9639, 2407, 1993, 8985, 6851, 9877, 3881, 4319, 4731, 1365, 1607, 9055, 4616, 9061, 3254, 7529, 980, 9290, 672, 6466, 9242, 4917, 4969, 1471, 6947, 6027, 1549, 6735, 295, 9160, 5493, 9224, 1138, 3582, 365, 5190, 1461, 5969, 45, 6832, 4309, 8262, 1918, 8129, 5098, 4879, 6480, 2632, 9104, 310, 6625, 153, 9238, 8069, 1975, 2910, 6220, 1371, 2662, 5843, 8321, 2738, 717, 176, 8953, 889, 3283, 9539, 3459, 9869, 3695, 7337, 4548, 2343, 1096, 6176, 4496, 9761, 4775, 5945, 2176, 1640, 1561, 3964, 3711, 9558, 8464, 830, 1255, 4338, 3651, 5764, 7171, 7249, 1593, 6928, 449, 2107, 940, 2250, 5911, 8851, 7360, 3225, 2560, 9286, 9332, 6254, 6440, 4694, 827, 9306, 5094, 8491, 4905, 3054, 5408, 7801, 5590, 9109, 2919, 4580, 6355, 1349, 460, 3038, 3093, 7175, 9029, 6150, 9742, 2512, 9465, 4460, 89, 9925, 4008, 8111, 2289, 6094, 7429, 8695, 1423, 6538, 9315, 1160, 8195, 8802, 3117, 3843, 8327, 5263, 4381, 2692, 2326, 4918, 1397, 1078, 4877, 491, 5352, 1060, 9918, 8207, 503, 2253, 7551, 3876, 1407, 9485, 9947, 1014, 3470, 2557, 9909, 4750, 1651, 1094, 7299, 81, 7857, 974, 3376, 9960, 6628, 7972, 7855, 4077, 4947, 831, 8923, 6793, 3862, 9641, 272, 8052, 8550, 2853, 4050, 9416, 9551, 574, 9319, 6002, 6407, 1693, 7447, 325, 4171, 1348, 4241, 326, 9950, 5858, 718, 1603, 5244, 8800, 8120, 3803, 8995, 8837, 458, 7234, 732, 324, 5681, 8908, 2878, 6977, 2174, 1408, 7161, 6702, 7958, 2100, 510, 5752, 2517, 1548, 4760, 7903, 4895, 4975, 9294, 666, 5698, 1646, 7955, 4315, 8694, 9863, 1660, 6364, 2680, 7304, 2685, 679, 284, 5563, 4095, 2241, 5993, 9331, 7623, 1084, 4196, 6367, 6464, 7441, 1915, 1855, 567, 190, 8130, 2403, 4006, 9002, 6111, 5267, 2460, 336, 1019, 7370, 5029, 733, 1387, 1805, 4229, 9086, 7467, 7225, 5457, 3152, 8938, 3826, 8278, 4198, 9546, 2007, 4890, 3266, 9035, 8160, 7711, 504, 3406, 3572, 720, 1220, 4732, 9038, 2781, 7817, 3888, 7568, 2725, 2452, 294, 1789, 8505, 7545, 5207, 9685, 5279, 1921, 7571, 5972, 3975, 847, 2196, 7491, 8455, 8351, 9935, 7237, 2482, 8682, 3806, 7592, 1784, 5240, 1100, 2955, 7987, 8143, 8580, 4536, 581, 2801, 1609, 8423, 7025, 3198, 4222, 5179, 9169, 9757, 6066, 8734, 6219, 4391, 8598, 2741, 7227, 1390, 7037, 3952, 1235, 193, 8508, 211, 8456, 1752, 4681, 3221, 3879, 8266, 9743, 9867, 2163, 7978, 8729, 268, 4049, 3427, 389, 8992, 4242, 6871, 834, 6500, 2880, 4824, 8784, 2904, 5685, 8045, 6299, 440, 5031, 6300, 1555, 6193, 1465, 3589, 1988, 6676, 3062, 9390, 5428, 8007, 6499, 5989, 7502, 2350, 7494, 3999, 3024, 8084, 3328, 4123, 5185, 8878, 5220, 2171, 2729, 4154, 3306, 7141, 8540, 2705, 1755, 3528, 424, 4044, 3603, 3139, 6503, 4151, 9629, 8841, 5987, 4010, 2856, 41, 3722, 8304, 7046, 2816, 6951, 5025, 6772, 7875, 9414, 1986, 333, 9120, 5846, 1190, 111, 1676, 2065, 8960, 6280, 4922, 8343, 592, 9108, 1950, 8934, 5234, 5064, 4187, 2424, 6870, 9208, 6478, 5704, 1463, 4156, 6602, 8738, 2376, 3102, 1236, 3587, 1947, 1538, 6392, 5512, 1009, 981, 8341, 1388, 400, 9438, 9645, 9118, 3064, 1533, 2142, 575, 5997, 6174, 1043, 9923, 5656, 3638, 6764, 8977, 8937, 8918, 8236, 9955, 6956, 938, 5825, 8136, 8314, 2021, 226, 9759, 7112, 2033, 455, 9370, 6997, 9131, 8169, 6147, 139, 8488, 1991, 1525, 1224, 6862, 2036, 9656, 3383, 8649, 2369, 5555, 5465, 8366, 5549, 9172, 8761, 9067, 5250, 9222, 8731, 6237, 4656, 9221, 4939, 3419, 4715, 7863, 9752, 3022, 1476, 4430, 7194, 1935, 3016, 5717, 1404, 1718, 1963, 9517, 7644, 1591, 7798, 148, 2212, 8297, 2271, 8717, 7600, 814, 7027, 2093, 6748, 7150, 4835, 2316, 7748, 1449, 1065, 1353, 9750, 8931, 8135, 7393, 3558, 4098, 6418, 299, 5763, 4695, 4396, 63, 6993, 9192, 3460, 1792, 1904, 3619, 3811, 6987, 2564, 4941, 2161, 2051, 4923, 7339, 2558, 1447, 6302, 8665, 4267, 6371, 270, 9272, 9244, 6342, 5532, 4357, 6117, 9047, 6981, 7363, 7819, 7288, 5690, 2325, 1037, 8795, 6404, 4874, 6283, 3335, 9747, 8226, 2393, 7512, 835, 1482, 2356, 5050, 916, 6022, 4498, 6712, 3180, 837, 8714, 7332, 6640, 3151, 5672, 6962, 1785, 7604, 3259, 7956, 4892, 2218, 2622, 9334, 7260, 7033, 3724, 3641, 9668, 2552, 5977, 3910, 7839, 6689, 3035, 6945, 8325, 1425, 9892, 1572, 3404, 8616, 1158, 9487, 8256, 3626, 6265, 8235, 1185, 2719, 8031, 316, 7169, 4360, 1739, 5999, 1260, 7049, 2262, 6437, 6703, 2443, 8831, 2340, 9566, 4204, 11, 5260, 5844, 7461, 5236, 1325, 2568, 7949, 2303, 2941, 3997, 8829, 7871, 7428, 1602, 4191, 4804, 8354, 2803, 7550, 7336, 6459, 8599, 8358, 1635, 2138, 822, 8381, 5253, 546, 3674, 8287, 4717, 4531, 6810, 2228, 5642, 1305, 7740, 3845, 4345, 7654, 3462, 8854, 1985, 712, 5102, 4646, 9635, 6124, 3899, 3, 8561, 3644, 4569, 2328, 6490, 3656, 8997, 7698, 3208, 5574, 243, 2034, 374, 6616, 7636, 6462, 7192, 7531, 2442, 6059, 6424, 8485, 2476, 2223, 5528, 3269, 6786, 3858, 1005, 9079, 3820, 464, 3122, 7557, 9887, 7701, 8981, 8472, 6911, 8621, 6319, 8037, 5569, 4072, 5130, 7210, 8078, 132, 5868, 8227, 6018, 5537, 7374, 4627, 4078, 6562, 6800, 3758, 116, 5884, 1634, 7907, 6331, 8716, 5278, 9234, 96, 7826, 774, 4303, 6796, 8830, 3476, 4251, 1324, 6141, 7626, 212, 3262, 2386, 7128, 8963, 306, 8480, 5885, 5426, 7134, 2721, 722, 897, 4305, 4628, 1389, 4989, 2831, 5898, 3008, 6207, 4851, 8535, 2338, 4053, 3675, 206, 6798, 8639, 249, 3026, 622, 6451, 6753, 563, 8109, 8523, 7732, 637, 9688, 1616, 7276, 434, 3825, 5194, 3570, 128, 3890, 8484, 5805, 2029, 6162, 1859, 517, 1208, 1706, 328, 882, 8662, 9973, 46, 6880, 7075, 4216, 4503, 6469, 368, 8035, 5959, 6719, 5090, 8371, 8337, 3162, 3072, 8264, 4457, 3838, 1038, 6412, 9623, 202, 9052, 4292, 547, 9670, 3196, 5353, 7599, 4482, 9501, 4721, 3311, 1997, 5564, 4097, 4670, 3098, 4790, 1836, 3666, 2451, 5621, 8643, 8607, 8635, 1189, 999, 9603, 8747, 1516, 7068, 1109, 6349, 3498, 534, 3947, 3581, 4458, 2553, 7019, 1022, 1974, 2654, 3210, 3865, 2990, 7221, 9262, 4270, 8023, 3031, 1099, 7931, 6714, 5648, 5741, 1239, 8443, 7101, 702, 7118, 6836, 3885, 1380, 2045, 3145, 4189, 2595, 7252, 1162, 2189, 4961, 6109, 9420, 6317, 2362, 250, 1796, 4512, 8110, 6991, 9498, 6482, 1083, 4991, 4304, 1033, 748, 0, 4364, 2645, 2541, 9300, 6370, 1090, 6556, 198, 8687, 256, 5929, 5227, 7765, 2485, 7612, 4844, 2115, 7145, 6074, 9796, 4723, 8622, 699, 4502, 78, 9462, 2111, 736, 2885, 5273, 481, 5159, 7649, 4269, 1592, 6972, 9716, 9729, 1167, 9734, 5659, 2496, 5735, 7579, 9819, 7017, 4284, 3839, 3043, 5784, 238, 8270, 9941, 1426, 3796, 3506, 4950, 3143, 9403, 4325, 7593, 9066, 1531, 2149, 3461, 7492, 125, 8028, 9277, 9904, 9335, 5634, 7985, 5002, 8906, 6320, 8567, 8214, 3042, 8269, 1302, 2928, 26, 1995, 7554, 2734, 1441, 5788, 378, 5965, 9938, 9161, 8845, 6135, 2975, 6696, 4371, 4860, 4770, 6061, 8810, 6327, 5833, 9037, 5082, 4436, 2039, 6678, 9821, 3167, 9229, 1499, 969, 7683, 1391, 1977, 5281, 770, 5166, 5889, 582, 5412, 9736, 8317, 715, 3001, 9138, 4022, 6329, 9755, 533, 8006, 3041, 4633, 7205, 9368, 3518, 3547, 1228, 9284, 5623, 1835, 315, 7885, 6351, 4992, 9189, 7500, 396, 2458, 8138, 82, 7767, 9034, 9097, 4137, 6701, 7133, 5195, 5139, 9984, 6535, 4485, 3788, 6008, 6255, 7902, 3918, 7308, 4140, 7226, 1303, 9430, 1509, 7335, 2406, 8295, 9654, 8338, 2605, 1930, 6762, 4765, 3965, 6564, 5165, 3417, 3365, 3190, 2544, 2248, 7064, 6895, 2247, 821, 7598, 9058, 8445, 2273, 3768, 5035, 9041, 9847, 3358, 7126, 5286, 2951, 3948, 7663, 7410, 67, 371, 4367, 9924, 9824, 8935, 5306, 1068, 2694, 4690, 3632, 4768, 3391, 2664, 467, 4214, 5535, 3284, 887, 2839, 9843, 3912, 3454, 3113, 7211, 6501, 2710, 522, 1631, 3372, 6656, 2199, 6473, 4852, 7668, 140, 6534, 7455, 4793, 5147, 811, 8881, 7509, 6888, 7799, 4801, 9018, 1281, 2791, 7242, 9553, 8973, 976, 246, 9374, 1629, 9193, 7088, 4586, 3318, 4378, 5044, 4740, 905, 3110, 1046, 6943, 1726, 7300, 8661, 8192, 1754, 2696, 8578, 1903, 9857, 2078, 7348, 5513, 7519, 7678, 6068, 452, 2013, 1575, 5348, 5037, 5454, 719, 3613, 5613, 5619, 2255, 7469, 9831, 1195, 2372, 7153, 9148, 2090, 5766, 3510, 7691, 7106, 2087, 8042, 6965, 1999, 8177, 6750, 6525, 7723, 9548, 2081, 2956, 9593, 1569, 4318, 2999, 6376, 5995, 4079, 7542, 7015, 471, 9679, 2658, 9179, 528, 9598, 7154, 4017, 6913, 5610, 2676, 2947, 9803, 9057, 479, 1257, 6088, 3203, 4257, 579, 5114, 818, 5521, 9851, 812, 4769, 9431, 1261, 7178, 6465, 3565, 2209, 1614, 5071, 711, 3808, 8579, 7198, 36, 1246, 9540, 1318, 3559, 5410, 2818, 7181, 7421, 1597, 5924, 6309, 4637, 7944, 9125, 119, 4573, 8675, 9141, 1145, 8967, 6287, 8163, 9106, 2792, 1559, 4749, 6338, 2707, 6847, 901, 8152, 6986, 5934, 2381, 3507, 3373, 2637, 9042, 6225, 3874, 6311, 8352, 8025, 8875, 825, 1505, 6284, 6868, 6373, 8685, 4933, 3697, 9746, 4739, 4609, 6291, 5079, 6885, 3362, 5732, 4235, 6210, 5913, 7848, 7394, 2082, 2744, 4759, 5265, 9868, 4537, 7472, 1643, 9749, 5132, 4239, 6186, 7415, 9927, 5553, 108, 6009, 5547, 3687, 7359, 7148, 2365, 4685, 4125, 8921, 1047, 2473, 3422, 5923, 9774, 5487, 6522, 8525, 652, 9115, 9781, 9365, 860, 3239, 9297, 9226, 705, 2272, 7324, 1955, 5294, 5242, 7565, 7689, 1209, 2048, 5727, 1827, 3597, 884, 7253, 746, 107, 1807, 9004, 4521, 5715, 6693, 5008, 2415, 3549, 4707, 8554, 875, 8768, 6132, 776, 105, 6727, 2190, 8990, 5683, 7060, 6126, 1556, 7357, 766, 4339, 3227, 5598, 4742, 8524, 7544, 5665, 9673, 6840, 3059, 1171, 4508, 8759, 7810, 657, 3209, 2902, 5269, 7350, 3341, 626, 5785, 2706, 3111, 3191, 2056, 162, 4551, 4094, 8692, 2869, 3400, 5421, 1748, 8493, 8667, 2367, 2487, 7400, 2758, 9936, 4534, 338, 3308, 8638, 5157, 6295, 926, 8757, 4613, 8049, 6978, 967, 3611, 6016, 753, 5169, 5599, 4900, 1799, 1883, 6723, 7448, 9662, 8435, 2962, 8454, 5517, 9798, 4911, 1058, 4608, 588, 5106, 5918, 4080, 3990, 5966, 2159, 9032, 6838, 4128, 6557, 4243, 1168, 487, 9933, 3405, 8277, 9444, 7094, 6722, 7298, 7433, 1485, 7338, 4653, 9828, 2137, 670, 3229, 660, 2155, 7083, 2574, 5231, 8468, 2939, 1598, 6884, 239, 1625, 9919, 9834, 6721, 1852, 7013, 6964, 3698, 7591, 7547, 9210, 2665, 5257, 898, 6648, 7569, 2168, 3612, 841, 9269, 9592, 7286, 3520, 7581, 158, 8651, 8855, 3729, 3972, 4164, 9797, 5649, 3236, 4410, 3166, 6533, 9597, 3617, 5624, 7954, 8088, 5023, 8966, 9988, 2331, 8518, 2453, 2154, 6264, 7063, 8261, 7929, 1104, 9911, 7255, 6758, 4087, 7922, 3994, 8900, 5849, 9695, 5808, 4853, 2050, 1868, 2266, 8509, 2057, 9681, 4107, 2441, 5229, 1698, 5183, 5937, 8285, 7000, 7939, 5243, 5510, 2478, 8720, 7485, 390, 2950, 9360, 4515, 4273, 8185, 8862, 4347, 6704, 3985, 6369, 8743, 7334, 5066, 3753, 8010, 8447, 8836, 2576, 7960, 1197, 7078, 9054, 5245, 3735, 349, 5550, 4248, 6559, 8798, 8744, 1971, 8427, 6344, 5074, 1490, 8940, 3694, 1927, 1177, 4622, 973, 958, 4092, 866, 5083, 9958, 3181, 961, 4625, 6892, 5467, 6142, 1481, 1893, 1419, 1132, 9091, 7883, 379, 5983, 7893, 5120, 6409, 3819, 6524, 1413, 4783, 9707, 8652, 8613, 8903, 2222, 4549, 3299, 8062, 760, 5427, 7620, 4475, 9357, 661, 5941, 9794, 5407, 7807, 7928, 1396, 6143, 5891, 5601, 5246, 6167, 438, 8572, 2860, 2079, 4981, 8414, 8378, 4389, 5367, 5895, 6123, 5988, 6639, 3030, 4356, 353, 6425, 8282, 7894, 495, 9674, 5706, 114, 6305, 7032, 7841, 4461, 8247, 5105, 4756, 9508, 7012, 8778, 8571, 9219, 3339, 8161, 9313, 757, 886, 4272, 7207, 8659, 8706, 1897, 943, 6523, 7230, 7540, 3141, 8061, 992, 8155, 1639, 9111, 2260, 1946, 4232, 5739, 172, 6159, 1446, 1464, 5688, 4708, 6532, 5204, 300, 6106, 9922, 5086, 2996, 1259, 1382, 1312, 1008, 1315, 6026, 1793, 1112, 3116, 1495, 1608, 7136, 9898, 8386, 3255, 5901, 9358, 3256, 8147, 8283, 7166, 7246, 9385, 1363, 421, 9740, 8522, 4186, 445, 8821, 2661, 1205, 6001, 8005, 9240, 7343, 8133, 2493, 7677, 6268, 1453, 3954, 3628, 9153, 3441, 9710, 5871, 7936, 4605, 5596, 40, 6879, 4924, 320, 768, 5380, 3533, 8374, 508, 8860, 218, 169, 4066, 1455, 6096, 6118, 7859, 5338, 5315, 6606, 4431, 4404, 4700, 1913, 2287, 8767, 3945, 1945, 8801, 8434, 6113, 2563, 1800, 9468, 3347, 7058, 8811, 4859, 5990, 3545, 9730, 1750, 8043, 7069, 9267, 8170, 4659, 4983, 1489, 1518, 625, 7482, 3771, 9411, 4406, 6281, 1880, 5954, 2551, 7965, 8552, 1462, 1156, 2144, 8288, 8534, 7686, 1452, 1751, 3081, 2243, 7097, 7722, 3006, 1491, 9089, 2747, 9965, 6707, 6093, 469, 9383, 375, 3468, 1503, 2192, 4228, 9291, 9386, 8171, 7921, 3357, 2895, 3618, 5030, 3240, 8867, 7780, 3488, 3447, 5502, 8569, 2769, 7330, 7021, 694, 5129, 1129, 1543, 4621, 2238, 7708, 7201, 7926, 4432, 2968, 9519, 5949, 1539, 6133, 8537, 4106, 494, 203, 8298, 744, 1157, 2416, 4192, 2157, 7850, 5584, 4747, 9737, 955, 4335, 4413, 7414, 1816, 8332, 3625, 5839, 8229, 531, 7084, 6989, 713, 8604, 4040, 6092, 9048, 6107, 5148, 5725, 5909, 1696, 9409, 6215, 6248, 1196, 7233, 9895, 6650, 2529, 2984, 8183, 1153, 8719, 6474, 3508, 2484, 7624, 1568, 3226, 780, 3407, 134, 5350, 9239, 8038, 4817, 1809, 3264, 707, 8165, 9567, 6580, 4009, 680, 6489, 4784, 8516, 9518, 3926, 6725, 6537, 8346, 8705, 8420, 5423, 6757, 9137, 197, 1385, 701, 6484, 6105, 591, 2942, 5274, 8490, 3585, 4225, 2001, 7456, 1692, 9103, 459, 1200, 643, 6751, 4864, 3995, 9301, 5757, 3046, 4279, 9660, 3222, 9298, 5073, 1063, 9231, 4884, 8895, 4474, 7872, 8925, 9176, 5072, 2019, 555, 9701, 8958, 8268, 462, 3492, 519, 627, 9147, 3174, 1457, 4604, 4696, 9166, 4369, 3058, 5687, 4133, 9455, 3532, 6755, 6163, 8342, 9182, 4441, 4449, 4244, 3504, 3600, 2412, 9077, 1028, 4553, 7719, 4819, 525, 39, 1102, 6553, 9133, 4155, 7858, 686, 321, 2120, 1962, 1949, 9763, 977, 507, 4710, 3754, 6405, 3900, 7843, 2547, 7925, 8591, 597, 9717, 9744, 3471, 1283, 8208, 3847, 7444, 8773, 4955, 9951, 2862, 3531, 4829, 9810, 7717, 5176, 7870, 7449, 3448, 1775, 7734, 4121, 1373, 1362, 1878, 2534, 4878, 7845, 1595, 9075, 4538, 8557, 9845, 7648, 6286, 4954, 179, 514, 8362, 6897, 6809, 5143, 1445, 1356, 1574, 541, 2479, 3671, 2288, 5668, 3707, 3962, 6555, 4509, 2608, 8181, 6777, 2715, 8532, 8055, 5462, 2728, 9528, 6438, 1854, 6399, 1923, 4718, 1163, 8928, 4109, 1064, 5386, 6383, 9579, 4124, 303, 6012, 4379, 925, 951, 9482, 2084, 3252, 9448, 9830, 2400, 5203, 9404, 7516, 1402, 769, 1313, 2759, 7594, 1020, 855, 8517, 319, 9802, 1774, 4733, 9333, 2643, 7793, 8198, 524, 1347, 6563, 9302, 7989, 2494, 846, 910, 30, 110, 4666, 8205, 3751, 2653, 1487, 7567, 930, 6427, 4015, 42, 4463, 8726, 6952, 3834, 2751, 8105, 2265, 5369, 578, 3601, 9588, 7323, 1773, 6831, 1936, 5262, 6441, 7999, 8331, 3764, 3430, 8861, 3681, 4343, 1770, 3789, 5736, 9185, 1704, 824, 6613, 5933, 5971, 3105, 798, 6090, 3550, 3237, 653, 1670, 7736, 8036, 1411, 3082, 8032, 6491, 4761, 5152, 5225, 8167, 5571, 9569, 2207, 7354, 9324, 7873, 1304, 6581, 1023, 5881, 8512, 4152, 8296, 1916, 8159, 5776, 3665, 7487, 1319, 5357, 3921, 7896, 6075, 2282, 3183, 6390, 1267, 3176, 149, 7281, 871, 4729, 2071, 1081, 6861, 6908, 6214, 5020, 8880, 6063, 6046, 2674, 437, 556, 956, 4993, 781, 2385, 2787, 6152, 3375, 9466, 2067, 4377, 7222, 9350, 4763, 4802, 2320, 7878, 8531, 1769, 2579, 3748, 7316, 6013, 7984, 6393, 7968, 8740, 8762, 3701, 9793, 3200, 3884, 5636, 9094, 7351, 2025, 207, 648, 4676, 5742, 4188, 4566, 7730, 8479, 2754, 9669, 4298, 9230, 1406, 8349, 9840, 3907, 2202, 19, 9532, 5630, 8429, 3003, 2203, 7121, 9583, 4423, 698, 6095, 9008, 5418, 4291, 2699, 1811, 6815, 8034, 7109, 5024, 2505, 1450, 7781, 490, 409, 1742, 4116, 1329, 7477, 4833, 935, 9677, 1861, 6784, 7050, 4910, 1059, 1436, 2957, 1334, 1087, 136, 5058, 8012, 5635, 289, 6243, 4600, 1130, 6795, 2925, 7864, 8538, 2357, 854, 3207, 9714, 7264, 1912, 618, 3342, 4208, 9389, 189, 7866, 2900, 2094, 463, 8033, 3636, 8952, 3352, 1707, 5938, 4942, 8009, 4821, 7353, 6918, 1802, 8610, 5697, 181, 1818, 6611, 4603, 2427, 1370, 5854, 6834, 5637, 9194, 1934, 6805, 4275, 5940, 3514, 6472, 8713, 4238, 191, 3659, 7533, 2823, 4249, 7451, 341, 4471, 3760, 4018, 8752, 8733, 5774, 7614, 8915, 745, 7146, 8974, 3831, 7104, 7269, 2889, 7071, 3539, 5504, 1664, 9842, 5548, 8475, 7611, 138, 5355, 602, 1199, 6765, 6153, 3429, 6182, 9387, 8636, 6356, 2280, 2164, 5612, 2277, 664, 8742, 4440, 2720, 4535, 6912, 9375, 6632, 260, 6567, 5057, 7138, 6148, 9314, 5305, 7035, 350, 1493, 44, 799, 9460, 6352, 5671, 6055, 5716, 1778, 4556, 1097, 3544, 868, 5719, 2085, 1474, 444, 691, 5968, 628, 3927, 601, 2012, 6445, 5289, 5821, 6384, 7464, 9062, 8097, 7391, 1204, 4730, 659, 2147, 6343, 6565, 6328, 1737, 3855, 4996, 5003, 9547, 4570, 2268, 4630, 1557, 4648, 2641, 6717, 6298, 1863, 2233, 9879, 8216, 7103, 7173, 6233, 8998, 8148, 7851, 683, 7043, 4907, 861, 7564, 7710, 8083, 5919, 4117, 6368, 1075, 5896, 5984, 2800, 4467, 7950, 3773, 829, 903, 2974, 6566, 7041, 9437, 4602, 208, 2055, 2855, 1700, 6934, 9201, 6808, 7061, 8824, 1820, 7918, 3458, 8090, 1351, 4393, 9741, 8582, 5108, 2735, 3173, 3205, 2242, 5034, 8872, 832, 7218, 9536, 8230, 1336, 3959, 5347, 5499, 4703, 8444, 3915, 6179, 6403, 3474, 4411, 1738, 5869, 5381, 862, 2979, 8326, 560, 6995, 3074, 6855, 6144, 461, 8199, 4059, 2733, 398, 3989, 1749, 4901, 8739, 9433, 9200, 251, 9805, 5255, 8417, 8242, 5346, 6471, 5000, 9275, 8858, 1887, 6684, 5970, 8769, 6816, 357, 5116, 7072, 5796, 2383, 8122, 337, 7082, 7539, 4960, 6171, 225, 9464, 9440, 2423, 810, 6969, 6710, 1902, 6134, 7580, 1366, 5588, 6050, 7290, 4754, 4773, 1984, 2703, 9931, 9816, 9019, 466, 7825, 9727, 9260, 3823, 2933, 8089, 7891, 1900, 173, 7655, 8086, 1611, 9380, 84, 9606, 3536, 4115, 4446, 2119, 1144, 7503, 2182, 2086, 759, 3966, 9312, 7005, 1952, 2183, 9373, 7541, 8330, 8050, 6346, 5017, 7912, 9853, 7100, 4986, 1062, 9987, 4230, 6542, 5952, 756, 3723, 6713, 9889, 9399, 4614, 5300, 7983, 2234, 2586, 4861, 942, 1431, 4545, 1025, 9493, 2533, 1641, 8401, 1599, 6818, 3524, 5358, 7087, 5128, 9282, 6387, 1983, 3755, 7862, 501, 2722, 4170, 8424, 8106, 587, 2648, 6498, 9132, 163, 3673, 6623, 8293, 1091, 5606, 4333, 778, 9502, 5059, 6741, 6646, 8217, 1343, 6823, 1767, 2876, 4167, 6444, 7402, 1969, 4306, 5373, 3870, 6116, 293, 4111, 4422, 6155, 5287, 2110, 3620, 6332, 76, 276, 4737, 5304, 2389, 8527, 1716, 9152, 888, 8103, 1085, 740, 4662, 3056, 5212, 7086, 3084, 6031, 5595, 1182, 1225, 222, 1437, 7295, 1137, 8438, 3857, 8888, 3338, 3496, 1082, 3877, 383, 3986, 2117, 6038, 498, 2623, 9093, 3465, 7582, 2249, 7763, 7616, 6680, 5022, 150, 9890, 2004, 3052, 274, 1606, 2421, 1421, 2153, 8396, 4127, 2905, 9715, 9336, 4451, 3667, 1667, 5366, 8970, 66, 4472, 6697, 7328, 5769, 3202, 603, 9945, 5709, 1845, 4025, 9825, 9180, 4581, 8300, 8563, 1330, 1714, 5657, 6551, 6270, 8096, 2764, 4213, 9929, 4557, 7923, 6905, 5958, 5536, 5629, 5333, 2607, 2983, 3039, 229, 3349, 1553, 4415, 2619, 2528, 5852, 7036, 3150, 5163, 4524, 9355, 7559, 3852, 5979, 5268, 6604, 4543, 283, 5922, 3863, 5119, 1294, 5678, 2211, 9999, 6528, 1728, 3680, 4328, 1176, 6582, 3025, 9815, 3829, 4745, 6289, 8619, 9712, 5394, 2413, 3385, 1477, 2779, 344, 7621, 532, 5676, 1826, 4965, 1141, 5161, 5567, 9181, 68, 5748, 8251, 8213, 710, 2437, 9287, 9278, 3783, 8989, 703, 8559, 6814, 6273, 5307, 8416, 3686, 5345, 5777, 3842, 858, 6644, 3389, 8467, 5931, 7772, 6683, 5942, 9872, 9006, 949, 9130, 1181, 161, 9127, 8328, 7241, 6797, 192, 5932, 6414, 9463, 2022, 9395, 2139, 5416, 9587, 5451, 7165, 3275, 4698, 5853, 6726, 4797, 93, 3571, 9070, 4069, 8299, 2127, 112, 9804, 1933, 6431, 4699, 4, 6560, 7869, 607, 6690, 1713, 8241, 3670, 4693, 4323, 6288, 6316, 4453, 1862, 8442, 1833, 9713, 1244, 7446, 305, 2402, 1051, 7904, 2419, 891, 4831, 7326, 3624, 6077, 9838, 553, 4454, 678, 4970, 1825, 656, 4672, 8498, 1929, 9114, 8102, 4143, 7777, 2240, 4424, 3736, 610, 9823, 1604, 928, 536, 632, 5403, 4530, 9994, 2160, 7283, 6654, 3487, 3021, 9991, 7454, 8782, 4043, 5468, 9545, 529, 543, 8409, 3146, 145, 2813, 9614, 9530, 4577, 7132, 8533, 2064, 1689, 3119, 6362, 7244, 8063, 3886, 6200, 3277, 9699, 7204, 5363, 7282, 9022, 8560, 803, 8387, 2959, 3596, 5720, 7274, 9003, 5389, 5760, 6347, 29, 5831, 4748, 4004, 5899, 170, 6202, 5749, 8899, 446, 6307, 3836, 4714, 9513, 8361, 9196, 7823, 7879, 1783, 2826, 9090, 1403, 9488, 9754, 4998, 8469, 4067, 6236, 9544, 2371, 7749, 4849, 6770, 3757, 6768, 9243, 8275, 3844, 9967, 5794, 7124, 355, 5546, 8395, 1192, 3505, 6568, 9116, 6228, 4322, 4342, 2868, 3679, 4906, 6848, 6643, 214, 669, 786, 3822, 3561, 7, 9627, 2044, 9100, 6175, 5456, 820, 7953, 6015, 5033, 3860, 3048, 4002, 4854, 7271, 1928, 8060, 8323, 7251, 8670, 1701, 8791, 1655, 5489, 2834, 4865, 5524, 934, 5388, 883, 1690, 6874, 8013, 8919, 797, 3132, 7674, 3630, 3125, 6339, 5887, 5239, 5632, 9452, 7390, 2943, 8118, 6711, 4525, 4667, 9128, 1093, 6700, 2726, 3359, 1061, 7852, 7293, 9236, 6337, 1960, 5438, 1254, 4688, 9046, 8725, 9954, 5903, 6666, 3369, 8618, 2742, 7365, 1772, 8647, 1998, 8783, 500, 85, 1332, 5450, 6184, 8883, 410, 8709, 8965, 2844, 9043, 1072, 5174, 2017, 9509, 3355, 7129, 7656, 4344, 4141, 5191, 5780, 3937, 1375, 5824, 3710, 6776, 2717, 5292, 1053, 9248, 813, 6446, 2621, 7327, 5441, 9837, 9700, 9095, 448, 9864, 8281, 6866, 1723, 7042, 2224, 9758, 7199, 9957, 330, 1576, 5713, 8986, 9900, 9220, 6901, 6043, 6756, 8797, 3086, 758, 8686, 9289, 5973, 2235, 2363, 1416, 2246, 4478, 3165, 3363, 3742, 3413, 8976, 9340, 3867, 7528, 5926, 6587, 5826, 3381, 9406, 689, 2426, 6926, 7437, 4330, 5420, 5126, 8047, 3123, 9733, 7268, 4260, 3652, 361, 3646, 8879, 2730, 1508, 3129, 5076, 5362, 3466, 3750, 4862, 6183, 1673, 2606, 2293, 8054, 3014, 8704, 8104, 4205, 9576, 2588, 3047, 5233, 5198, 5892, 704, 5526, 1128, 7452, 8119, 3548, 5586, 5019, 3623, 8764, 2514, 7778, 9151, 9461, 1012, 7895, 4977, 739, 3872, 7341, 2307, 4386, 8117, 1279, 5920, 5158, 2772, 6593, 5188, 4428, 6413, 5241, 8959, 790, 2847, 1229, 4856, 4193, 6104, 1206, 3327, 6129, 3231, 9361, 3563, 4500, 6641, 3854, 4224, 6137, 9349, 6101, 2104, 9413, 8274, 9316, 7842, 675, 2077, 6406, 2655, 1295, 7979, 7265, 2387, 3322, 1150, 6960, 6036, 2286, 386, 8070, 911, 7478, 5806, 3080, 7462, 5591, 3873, 2095, 1642, 1953, 4407, 7018, 4005, 9723, 4777, 6138, 3140, 3693, 10, 3696, 4383, 2506, 9849, 2041, 2593, 8735, 571, 3622, 939, 9500, 5172, 4675, 9937, 4597, 1817, 18, 8107, 2491, 1992, 9801, 8846, 6778, 8112, 286, 308, 5992, 5703, 213, 6157, 141, 2089, 755, 180, 2777, 8494, 4375, 4259, 5235, 7256, 3160, 6668, 4979, 9321, 2783, 5056, 1049, 7726, 1874, 4149, 9212, 996, 7941, 4938, 4828, 6276, 1268, 697, 1492, 4671, 1288, 8693, 9187, 2890, 3261, 7664, 4926, 5378, 7573, 6411, 900, 3590, 6915, 7590, 5529, 3408, 1828, 4931, 5605, 1794, 8788, 2835, 8294, 8, 7549, 5272, 1566, 5213, 3272, 4247, 8174, 3769, 3472, 4464, 1972, 7216, 3379, 7725, 8233, 8530, 2736, 3015, 7375, 298, 2031, 4395, 8676, 5485, 5837, 7588, 7303, 6984, 5449, 7693, 5395, 9157, 5127, 5136, 7457, 6614, 3334, 7913, 7651, 9442, 8189, 7808, 5759, 3066, 5781, 6204, 2175, 7144, 7367, 8020, 1537, 8290, 8513, 9073, 8699, 2848, 6626, 3297, 7684, 4741, 7770, 3060, 3931, 6841, 3657, 2539, 2775, 9039, 4818, 1551, 6761, 1469, 1376, 8452, 8553, 5368, 2536, 2220, 1003, 3982, 8817, 1213, 5223, 6745, 3765, 2225, 8068, 7865, 7527, 2907, 6760, 313, 9326, 339, 6959, 6636, 8929, 9827, 7498, 5936, 9920, 2906, 7257, 6512, 8173, 3187, 58, 3553, 6575, 9119, 6024, 1588, 9059, 6019, 6552, 7490, 199, 7536, 9353, 2532, 997, 1056, 3743, 1031, 9271, 6579, 3377, 6958, 6824, 9582, 4100, 8597, 8779, 8886, 3728, 2263, 4057, 3156, 3721, 8390, 6416, 74, 2701, 5264, 2882, 795, 4999, 8211, 7157, 1219, 2059, 7346, 456, 219, 34, 9642, 4178, 9550, 6746, 7555, 8877, 4914, 8041, 9480, 6917, 7828, 1512, 3456, 3943, 3045, 6739, 3271, 3958, 4619, 5587, 604, 3432, 3321, 5259, 6189, 8410, 7712, 3849, 9327, 1615, 5622, 314, 2396, 3894, 3188, 6638, 1341, 7224, 183, 6649, 4815, 3602, 2046, 3171, 2749, 1587, 880, 5616, 8730, 3737, 4647, 3199, 5770, 3568, 7155, 2762, 4704, 6996, 3178, 6082, 9992, 404, 7229, 5927, 6304, 7280, 828, 5707, 6589, 9247, 2531, 4062, 5886, 8988, 9643, 6906, 857, 3288, 3027, 13, 7766, 9573, 6980, 4493, 1644, 3924, 8481, 2114, 3790, 6448, 1515, 7942, 3714, 3242, 7312, 957, 1586, 8156, 1, 6624, 2602, 92, 7408, 6481, 1044, 233, 16, 9400, 8589, 6828, 1269, 9207, 7583, 7844, 6787, 5540, 5835, 1837, 4858, 3397, 9345, 3804, 5840, 4480, 5186, 4030, 485, 9726, 6677, 7213, 9143, 650, 7176, 331, 9337, 3500, 7861, 8615, 8459, 3833, 876, 8898, 9040, 890, 6914, 5880, 3475, 629, 6365, 635, 2092, 403, 5320, 1600, 7785, 9049, 1122, 8984, 7920, 71, 9720, 9844, 4134, 8377, 5832, 5565, 4207, 8909, 636, 4402, 2200, 359, 9609, 7474, 4326, 3778, 2954, 9477, 8660, 9163, 6924, 8124, 617, 1782, 4011, 1262, 6659, 3499, 164, 2440, 6492, 3128, 7377, 62, 6475, 5782, 4921, 6682, 2292, 8677, 231, 3303, 9966, 8248, 3439, 3130, 8021, 5955, 7537, 2896, 863, 5578, 1931, 9145, 5496, 6216, 1322, 4354, 1808, 9646, 2550, 397, 3766, 8433, 7373, 840, 5928, 5156, 7380, 2461, 6067, 2702, 14, 6201, 7340, 6548, 5393, 7040, 7524, 2611, 5385, 5396, 3720, 5324, 1136, 8492, 3893, 7933, 2354, 2819, 4816, 3353, 7459, 4618, 7917, 4706, 779, 2542, 4494, 346, 3898, 5963, 1691, 1580, 8812, 1399, 5542, 9832, 376, 5800, 782, 2069, 7236, 9183, 8450, 3703, 2076, 9899, 4563, 573, 4640, 9112, 4246, 9279, 4240, 4657, 367, 5328, 1860, 9177, 6886, 7889, 3968, 101, 6054, 2761, 4834, 4973, 3805, 3535, 2649, 8489, 2457, 5270, 8957, 7189, 7782, 7378, 5960, 1791, 8310, 6706, 9586, 6633, 800, 6400, 1256, 1278, 612, 5744, 2873, 9997, 9195, 3393, 1504, 1346, 5900, 4197, 2670, 1372, 4974, 3091, 9031, 7911, 3745, 3936, 3049, 1238, 950, 2024, 7783, 7515, 8626, 2355, 4138, 1227, 2603, 1369, 1768, 4869, 7735, 9496, 8654, 1251, 8126, 8930, 8804, 3161, 3464, 8844, 9507, 8501, 2537, 3390, 9026, 9626, 991, 6493, 5097, 6372, 8648, 244, 1958, 159, 9098, 1506, 7635, 7228, 7546, 2410, 7386, 2587, 9811, 5680, 1194, 5608, 6419, 3232, 1823, 7658, 254, 8348, 7887, 9027, 4744, 6190, 623, 5295, 6071, 4416, 7881, 201, 4916, 7834, 8244, 6618, 95, 3485, 3691, 6574, 921, 8736, 2598, 6733, 4830, 6651, 3246, 8715, 7473, 819, 2774, 7832, 2195, 8357, 1899, 9328, 9378, 6933, 9227, 4405, 2042, 129, 1320, 7945, 8392, 1552, 9912, 9171, 6044, 5275, 4212, 2838, 7812, 1146, 6622, 2399, 9565, 1310, 5168, 8142, 3513, 7969, 674, 6277, 1623, 2897, 7627, 2932, 2002, 1306, 4027, 630, 5254, 990, 3276, 28, 7107, 1596, 1759, 6794, 8500, 4875, 6966, 3290, 9450, 6904, 2857, 1746, 5459, 2444, 586, 4510, 3564, 2388, 3095, 9612, 9968, 8956, 6110, 9372, 2879, 3020, 4351, 2776, 7758, 9129, 3274, 8942, 5280, 2047, 3185, 2953, 2103, 1546, 6345, 9521, 4591, 6230, 1577, 442, 2148, 8318, 6149, 5795, 647, 4542, 608, 6955, 7553, 2447, 7847, 8405, 7768, 3002, 4071, 1564, 425, 8964, 3887, 2098, 6033, 216, 7796, 8497, 475, 1131, 806, 4514, 9281, 8577, 8624, 8948, 9930, 1507, 6324, 8805, 6825, 7191, 223, 109, 2812, 6821, 2612, 5404, 9584, 4601, 7868, 2359, 3593, 998, 9568, 5998, 7047, 3816, 1856, 3719, 2840, 50, 1715, 8101, 6070, 4139, 515, 9167, 9068, 2871, 9910, 2540, 8683, 9005, 714, 3906, 5500, 8473, 878, 5650, 8320, 8575, 4317, 1666, 4349, 4086, 2790, 1968, 1521, 2156, 8873, 7045, 1354, 8658, 3974, 7976, 7880, 9445, 5682, 3586, 499, 3216, 5088, 9692, 9611, 2459, 1475, 6586, 1814, 5384, 3998, 5728, 9520, 8609, 562, 3418, 1582, 2364, 6197, 7484, 2274, 8890, 3653, 7818, 3036, 9135, 5063, 6025, 9206, 4958, 4035, 1284, 6341, 9064, 9071, 3606, 5170, 9560, 8240, 5145, 5651, 3253, 9655, 809, 2976, 5951, 3135, 3170, 4020, 3382, 217, 2687, 5568, 3011, 8978, 127, 9771, 2404, 8678, 4313, 8994, 9474, 2797, 2205, 3942, 7505, 1647, 9099, 2594, 4997, 6021, 3195, 971, 5939, 488, 1844, 6652, 570, 1121, 4719, 157, 4031, 7605, 7119, 6833, 526, 4101, 1656, 3869, 8529, 5397, 8902, 6629, 9259, 6664, 9672, 6421, 8146, 5581, 4382, 2486, 5164, 1622, 5026, 7930, 9139, 2743, 237, 8612, 2866, 2842, 2666, 4096, 7024, 6495, 194, 2651, 61, 1919, 9283, 8823, 7131, 8842, 5334, 2382, 5538, 4915, 9594, 2601, 2330, 5447, 9835, 9526, 8825, 9050, 5480, 9949, 982, 5873, 615, 8252, 7149, 8151, 8428, 1589, 265, 8972, 3913, 9678, 1654, 3828, 4299, 3661, 8194, 2578, 9855, 7570, 9998, 6208, 9687, 8016, 1973, 8176, 1240, 6045, 5792, 60, 5962, 9608, 9647, 5466, 2126, 1872, 9330, 5828, 9249, 3089, 9850, 5816, 6034, 4687, 4334, 3346, 7093, 5359, 5890, 4845, 2631, 1175, 7897, 7637, 1886, 7232, 131, 3175, 6361, 4190, 8419, 1758, 4976, 919, 3946, 9913, 688, 271, 3409, 1258, 8276, 1316, 6515, 1180, 3364, 2375, 4089, 1735, 4181, 2251, 1633, 3101, 7262, 5799, 8809, 7291, 4469, 4320, 7254, 9644, 6736, 4691, 764, 8364, 1530, 1540, 7450, 433, 5811, 2193, 3402, 2018, 6737, 2846, 3979, 1358, 6775, 3273, 1547, 6006, 6716, 3517, 5495, 5746, 1210, 3689, 5834, 5425, 5743, 6449, 5117, 7675, 2257, 3415, 4497, 4289, 1013, 6867, 5013, 4684, 7742, 4883, 1275, 6661, 1760, 5611, 5618, 2682, 599, 7436, 3682, 8313, 5214, 838, 7238, 2480, 1335, 9516, 4352, 968, 2545, 1297, 7321, 9254, 899, 5790, 2698, 5216, 5364, 9952, 6494, 729, 6600, 7787, 5018, 9527, 3649, 8448, 279, 5947, 668, 5518, 5138, 3055, 3211, 5222, 4331, 1218, 5925, 7696, 9537, 5277, 7967, 6732, 9092, 6354, 9016, 4582, 5875, 600, 3331, 2321, 5372, 7511, 5046, 5556, 83, 7057, 4253, 6252, 8431, 5894, 351, 4355, 7190, 6631, 5392, 9216, 4390, 4516, 6511, 1881, 6040, 3578, 7362, 542, 1884, 3109, 945, 2524, 2503, 895, 8656, 6052, 5946, 5851, 3591, 9310, 2773, 7779, 2716, 6718, 2431, 6195, 2663, 5258, 3061, 1015, 8701, 6720, 3159, 7185, 3350, 5907, 3648, 8018, 6115, 2166, 2308, 6670, 7038, 5580, 91, 9197, 4889, 4587, 3567, 9515, 2143, 7704, 5455, 3521, 3494, 7643, 2584, 7399, 6612, 6957, 6852, 9696, 8210, 2644, 1724, 8546, 1965, 7127, 4443, 5443, 8850, 2165, 1339, 4810, 8254, 2344, 2305, 3410, 4899, 3184, 1285, 5332, 7117, 7197, 3746, 5775, 9078, 9533, 4021, 51, 896, 5991, 3323, 5144, 2746, 1500, 9122, 5917, 258, 9190, 8311, 2006, 1743, 569, 5011, 8871, 5813, 2554, 3201, 4934, 1523, 2083, 241, 1882, 7488, 5121, 6781, 1618, 7728, 2874, 7962, 9217, 9088, 5341, 4366, 1309, 572, 4177, 2561, 1734, 7794, 993, 9703, 3515, 4599, 7661, 6072, 3051, 7906, 335, 4337, 8565, 817, 1590, 6839, 5893, 436, 7352, 4661, 754, 2299, 4908, 4940, 6627, 453, 8150, 2634, 4016, 2629, 2949, 1513, 7909, 5879, 9686, 9762, 2945, 1470, 7820, 6047, 1187, 964, 2566, 2877, 9214, 1292, 6829, 3114, 4083, 1454, 6691, 902, 6896, 9706, 4076, 6695, 1179, 1052, 2221, 8664, 6822, 1761, 2438, 8568, 7486, 7518, 1777, 7919, 2650, 3034, 9667, 3503, 2230, 6083, 4437, 9470, 9782, 6366, 285, 7640, 23, 4085, 6509, 2504, 6005, 9572, 8234, 4654, 3752, 3070, 3206, 5729, 2374, 4776, 2507, 7697, 7403, 6294, 1685, 5344, 4794, 3121, 435, 9661, 654, 8576, 2283, 1846, 9841, 8723, 5402, 7139, 2718, 7468, 3607, 4439, 7504, 5178, 1517, 6976, 3592, 2592, 3575, 7435, 8848, 3875, 2347, 8753, 5463, 6544, 9198, 9621, 1338, 5298, 4161, 538, 3793, 3148, 8933, 986, 2646, 9044, 8868, 7757, 7574, 7671, 7007, 9063, 1202, 7187, 4302, 9822, 7009, 5641, 7769, 1473, 6292, 2105, 2851, 6429, 4805, 1858, 791, 836, 559, 7914, 1888, 3960, 3050, 4114, 983, 2911, 9883, 1578, 5433, 2901, 9809, 5791, 77, 1780, 4172, 9625, 1188, 7137, 5908, 6859, 7292, 4321, 4036, 2993, 8596, 5935, 4256, 8913, 9846, 9652, 1021, 3841, 4466, 1653, 6259, 606, 3112, 7318, 2573, 377, 6114, 2535, 2181, 5376, 6358, 1671, 4806]
    labels = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # outputs, new_memoize = postgres_execute(dsn, current_query, memoize, inputs_table_name, list(range(300)), is_trajectory=False)
    query_str = "Conjunction(Behind(o0, o1), RightOf(o0, o1)); Conjunction(Conjunction(Color_cyan(o0), FrontOf(o0, o2)), Material_rubber(o0)); Conjunction(Behind(o1, o2), Color_yellow(o1))"
    current_query = str_to_program_postgres(query_str)
    outputs, new_memoize_scene_graph, new_memoize_sequence = postgres_execute_cache_sequence(dsn, current_query, memoize_scene_graph, memoize_sequence, inputs_table_name, input_vids, is_trajectory=False, sampling_rate=None)
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