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
def dsl_to_program_quivr(program_str):
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
            return program_init(dsl_to_program_quivr(submodule_list[0]), int(submodule_list[1]))
        submodule_list = [dsl_to_program_quivr(submodule) for submodule in submodule_list]
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
                seq_signature = program_to_dsl(current_query[:len(current_query)-i], not is_trajectory)
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
                # signature = program_to_dsl(current_query[:graph_idx+1], not is_trajectory)
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
            seq_signature = program_to_dsl(current_query[:len(current_query)-i], not is_trajectory)
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

    temp_views = []

    with conn.cursor() as cur:
        new_memo = [{} for _ in range(len(memo))]

        output_vids = []
        # Prepare cache result
        if isinstance(input_vids, int):
            remaining_vids = set(range(input_vids))
        else:
            remaining_vids = set(input_vids)

        signatures = []
        for i in range(len(current_query)):
            seq_signature = program_to_dsl(current_query[:(i+1)], not is_trajectory)
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

        # select input videos
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
            # Generate scene graph:
            scene_graph = dict["scene_graph"]
            duration_constraint = dict["duration_constraint"]
            for p in scene_graph:
                for v in p["variables"]:
                    if v not in encountered_variables_current_graph:
                        encountered_variables_current_graph.append(v)

            # Execute for unseen videos
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
            sql_sring = """CREATE TEMPORARY VIEW g{} AS SELECT {} FROM {} WHERE {};""".format(graph_idx, fields, tables, where_clauses)
            # print(sql_sring)
            cur.execute(sql_sring)
            temp_views.append("g{}".format(graph_idx))
            # print("execute for unseen videos: ", time.time() - _start_execute)
            # print("Time for graph {}: {}".format(graph_idx, time.time() - _start))

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

            # Generate scene graph sequence:
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
            # Store new cached results
            for input_vid in filtered_vids:
                if input_vid in output_vids:
                    new_memo[input_vid][signatures[graph_idx]] = 1
                else:
                    new_memo[input_vid][signatures[graph_idx]] = 0
            encountered_variables_prev_graphs = obj_union
            encountered_variables_current_graph = []
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


def program_to_dsl(orig_program, rewrite_variables=True):
    """
    Input:
    program: query in the dictionary format
    Output: query in dsl string format, which is ordered properly (uniquely).
    NOTE: For trajectories, we don't rewrite the variables, since we expect the query to indicate which objects the predicate is referring to, as assumed in the Quivr paper.
    """
    def print_scene_graph(predicate_list):
        if len(predicate_list) == 1:
            return print_scene_graph_helper(predicate_list)
        else:
            return "({})".format(print_scene_graph_helper(predicate_list))

    def print_scene_graph_helper(predicate_list):
        predicate = predicate_list[-1]
        predicate_name = predicate['predicate']
        # f"{predicate['predicate']}_{predicate.get('parameter')}" if predicate.get('parameter') else predicate['predicate']
        predicate_variables = ", ".join(predicate["variables"])
        if predicate.get("parameter"):
            if isinstance(predicate["parameter"], str):
                predicate_variables = "{}, '{}'".format(predicate_variables, predicate["parameter"])
            else:
                predicate_variables = "{}, {}".format(predicate_variables, predicate["parameter"])
        if len(predicate_list) == 1:
            return "{}({})".format(predicate_name, predicate_variables)
        else:
            return "{}, {}({})".format(print_scene_graph_helper(predicate_list[:-1]), predicate_name, predicate_variables)

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
                # We assume that the order of variables in a predicate does not matter:
                # 1) Near(o1, o2) == Near(o2, o1)
                # 2) Although LeftOf(o1, o2) != LeftOf(o2, o1), we have LeftOf(o1, o2) == RightOf(o2, o1)
                # However, this is not true in general.
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

# Function to check if the string is numeric (integer or float)
def is_numeric(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def dsl_to_program(dsl_str):
    """
    Converts a DSL string into a program.

    Args:
        dsl_str (str): The DSL string to be converted.

    Returns:
        list: A list of scene graphs, where each scene graph is a dictionary containing a list of predicates and a duration constraint.
    """

    def parse_duration(scene_graph_str):
        """
        Parses the duration of a scene graph.

        Args:
            scene_graph_str (str): The scene graph string to be parsed.

        Returns:
            list: A list of submodules.
        """
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
        """
        Parses the conjunction of a scene graph.

        Args:
            scene_graph_str (str): The scene graph string to be parsed.

        Returns:
            list: A list of predicates.
        """
        predicate_list = []
        idx = 0
        counter = 0
        for i, char in enumerate(scene_graph_str):
            if char == "(":
                counter += 1
            elif char == ")":
                counter -= 1
            elif char == "," and counter == 0:
                predicate_list.append(scene_graph_str[idx:i].strip())
                idx = i+1
        predicate_list.append(scene_graph_str[idx:].strip())
        return [parse_predicate(predicate) for predicate in predicate_list]

    def parse_predicate(predicate_str):
        """
        Parses a predicate. A predicate has the following format:
            predicate_name(variable1[, variable2][, parameter])
        parameter can be a string or a number.

        Args:
            predicate_str (str): The predicate string to be parsed.

        Returns:
            dict: A dictionary containing the predicate name, parameter (if any), and variables.
        """
        dict = {}
        # Near_0.95(o0, o1)
        # Color(o1, "purple")
        idx = predicate_str.find("(")
        idx_r = predicate_str.rfind(")")
        predicate_name = predicate_str[:idx]
        dict["predicate"] = predicate_name
        predicate_arguments = predicate_str[idx+1:idx_r].split(", ")
        # if the last argument is a string, then it is the parameter
        if predicate_arguments[-1][0] in ["'", '"'] and predicate_arguments[-1][-1] in ["'", '"']:
            dict["parameter"] = predicate_arguments[-1][1:-1]
            dict["variables"] = predicate_arguments[:-1]
        # if the last argument is numeric, it is also a parameter
        elif is_numeric(predicate_arguments[-1]):
            dict["parameter"] = float(predicate_arguments[-1])
            dict["variables"] = predicate_arguments[:-1]
        # otherwise, there is no parameter
        else:
            dict["parameter"] = None
            dict["variables"] = predicate_arguments
        return dict

    scene_graph_str_list = dsl_str.split("; ")
    program = []
    for scene_graph_str in scene_graph_str_list:
        duration_constraint = 1
        if scene_graph_str.startswith("Duration"):
            submodule_list = parse_duration(scene_graph_str)
            duration_constraint = int(submodule_list[-1])
            scene_graph_str = submodule_list[0]
        # remove the outermost parentheses if any
        if scene_graph_str.startswith("(") and scene_graph_str.endswith(")"):
            scene_graph_str = scene_graph_str[1:-1]
        scene_graph = {"scene_graph": parse_conjunction(scene_graph_str), "duration_constraint": duration_constraint}
        program.append(scene_graph)
    return program


def rewrite_vars_name_for_scene_graph(orig_dict):
    """
    [deprecated] This logic is integrated into the program_to_dsl method (when rewrite_variables=True)
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


def program_to_dsl_v2(orig_program, rewrite_variables=True):
    """
    [deprecated]
    Input:
    program: query in the dictionary format
    Output: query in dsl string format, which is ordered properly (uniquely).
    NOTE: For trajectories, we don't rewrite the variables, since we expect the query to indicate which objects the predicate is referring to, as assumed in the Quivr paper.
    """
    def print_scene_graph(predicate_list):
        if len(predicate_list) == 1:
            return print_scene_graph_helper(predicate_list)
        else:
            return "({})".format(print_scene_graph_helper(predicate_list))

    def print_scene_graph_helper(predicate_list):
        predicate = predicate_list[-1]
        predicate_name = f"{predicate['predicate']}_{predicate.get('parameter')}" if predicate.get('parameter') else predicate['predicate']
        predicate_variables = ", ".join(predicate["variables"])
        if len(predicate_list) == 1:
            return "{}({})".format(predicate_name, predicate_variables)
        else:
            return "{}, {}({})".format(print_scene_graph_helper(predicate_list[:-1]), predicate_name, predicate_variables)

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


def dsl_to_program_v2(dsl_str):
    """
    [deprecated]
    Converts a DSL string into a program.

    Args:
        dsl_str (str): The DSL string to be converted.

    Returns:
        list: A list of scene graphs, where each scene graph is a dictionary containing a list of predicates and a duration constraint.
    """

    def parse_duration(scene_graph_str):
        """
        Parses the duration of a scene graph.

        Args:
            scene_graph_str (str): The scene graph string to be parsed.

        Returns:
            list: A list of submodules.
        """
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
        """
        Parses the conjunction of a scene graph.

        Args:
            scene_graph_str (str): The scene graph string to be parsed.

        Returns:
            list: A list of predicates.
        """
        predicate_list = []
        idx = 0
        counter = 0
        for i, char in enumerate(scene_graph_str):
            if char == "(":
                counter += 1
            elif char == ")":
                counter -= 1
            elif char == "," and counter == 0:
                predicate_list.append(scene_graph_str[idx:i].strip())
                idx = i+1
        predicate_list.append(scene_graph_str[idx:].strip())
        return [parse_predicate(predicate) for predicate in predicate_list]

    def parse_predicate(predicate_str):
        """
        Parses a predicate.

        Args:
            predicate_str (str): The predicate string to be parsed.

        Returns:
            dict: A dictionary containing the predicate name, parameter (if any), and variables.
        """
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

    scene_graph_str_list = dsl_str.split("; ")
    program = []
    for scene_graph_str in scene_graph_str_list:
        duration_constraint = 1
        if scene_graph_str.startswith("Duration"):
            submodule_list = parse_duration(scene_graph_str)
            duration_constraint = int(submodule_list[-1])
            scene_graph_str = submodule_list[0]
        # remove the outermost parentheses if any
        if scene_graph_str.startswith("(") and scene_graph_str.endswith(")"):
            scene_graph_str = scene_graph_str[1:-1]
        scene_graph = {"scene_graph": parse_conjunction(scene_graph_str), "duration_constraint": duration_constraint}
        program.append(scene_graph)
    return program


def program_to_dsl_v1(orig_program, rewrite_variables=True):
    """
    [deprecated]
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
                # We assume that the order of variables in a predicate does not matter; LeftOf(o1, o0) == RightOf(o0, o1)
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


def dsl_to_program_v1(program_str):
    """
    [deprecated]
    """
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

def get_inputs_table_name_and_is_trajectory(dataset_name):
    if dataset_name.startswith("collision"):
        inputs_table_name = "Obj_collision"
        is_trajectory = True
    elif "scene_graph" in dataset_name:
        inputs_table_name = "Obj_clevrer"
        is_trajectory = False
    elif dataset_name == "warsaw":
        inputs_table_name = "Obj_warsaw"
        is_trajectory = True
    else:
        inputs_table_name = "Obj_trajectories"
        is_trajectory = True
    return inputs_table_name, is_trajectory


if __name__ == '__main__':
    target_query = "Duration((Color(o0, 'red'), Far(o0, o1, 3), Shape(o1, 'cylinder')), 25); (Near(o0, o1, 1), RightQuadrant(o2), TopQuadrant(o2))"
    target_program = dsl_to_program(target_query)
    print(target_program)
    print(program_to_dsl(target_program))
    exit()

    dsn = "dbname=myinner_db user=enhaoz host=localhost"
    conn = psycopg.connect(dsn)

    target_query = "Duration(Conjunction(Eastward2(o1), Eastward4(o0)), 5); Duration(Conjunction(Eastward2(o1), Eastward3(o0)), 5)"
    list_size = 72159
    memo = [{} for _ in range(list_size)]
    # Read the test data
    input_dir = "/mmfs1/gscratch/balazinska/enhaoz/complex_event_video/inputs"
    dataset_name = "warsaw"
    inputs_table_name = "Obj_warsaw"
    is_trajectory = True
    test_dir = os.path.join(input_dir, dataset_name, "test")
    inputs_filename = target_query + "_inputs.json"
    labels_filename = target_query + "_labels.json"
    with open(os.path.join(test_dir, inputs_filename), 'r') as f:
        input_vids = json.load(f)
    with open(os.path.join(test_dir, labels_filename), 'r') as f:
        labels = json.load(f)
    test_query = "Eastward4(o0); Eastward3(o0); Eastward2(o1)"
    test_program = dsl_to_program(test_query)
    print(test_program)
    outputs, new_memo = postgres_execute_cache_sequence(conn, test_program, memo, inputs_table_name, input_vids, is_trajectory, sampling_rate=None)
    preds = []
    print(outputs)
    for input in input_vids:
        if input in outputs:
            preds.append(1)
        else:
            preds.append(0)
    score = f1_score(labels, preds)
    print(score)