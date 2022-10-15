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
    # TODO: the kleene operator in the <true>* predicate should also be considered when computing the depth of a query.
    main_program = program.submodules["program"]
    return _get_depth_and_npred_helper(main_program)

def _get_depth_and_npred_helper(program):
    if isinstance(program, dsl.ParameterHole):
        return 0, 1
    elif issubclass(type(program), dsl.Predicate):
        return 0, 1
    elif isinstance(program, dsl.PredicateHole):
        return 0, 1
    elif isinstance(program, dsl.ConjunctionOperator):
        left = program.submodules["function1"]
        right = program.submodules["function2"]
        depth_left, npred_left = _get_depth_and_npred_helper(left)
        depth_right, npred_right = _get_depth_and_npred_helper(right)
        if isinstance(left, dsl.ConjunctionOperator) or isinstance(right, dsl.ConjunctionOperator):
            return max(depth_left, depth_right), npred_left + npred_right
        else:
            return max(depth_left, depth_right) + 1, npred_left + npred_right
    elif isinstance(program, dsl.SequencingOperator):
        left = program.submodules["function1"]
        right = program.submodules["function2"]
        depth_left, npred_left = _get_depth_and_npred_helper(left)
        depth_right, npred_right = _get_depth_and_npred_helper(right)
        if isinstance(left, dsl.SequencingOperator) or isinstance(right, dsl.SequencingOperator):
            return max(depth_left, depth_right), npred_left + npred_right
        else:
            return max(depth_left, depth_right) + 1, npred_left + npred_right
    elif isinstance(program, dsl.KleeneOperator):
        kleene = program.submodules["kleene"]
        depth_kleene, npred_kleene = _get_depth_and_npred_helper(kleene)
        if isinstance(kleene, dsl.KleeneOperator):
            return depth_kleene, npred_kleene
        else:
            return depth_kleene + 1, npred_kleene
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

def postgres_execute(dsn, current_query, memoize, inputs_table_name, input_vids, is_trajectory=True, sampling_rate=None):
    """
    input_vids: list of video segment ids
    Example query:
        Sequencing(Sequencing(Sequencing(Sequencing(Sequencing(Sequencing(True*, Near_1.05), True*), Conjunction(LeftOf, Behind)), True*), Duration(Conjunction(TopQuadrant, Far_0.9), 5)), True*)
    two types of caches:
        1. scene graph cache: cache[graph] = vid, fid1, fid2, oids
        2. sequence cache: cache[sequence] = vid, fid1, fid2, oids
    Output:
    new_memoize: new cached results from this query, which will be added to the global cache (for multi-threading)
    """
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            new_memoize = [LRU(10000) for _ in range(len(memoize))]
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
                # delta_input_vids = []
                # df_list = [pd.DataFrame()]
                # signature, vars_mapping = rewrite_vars_name_for_scene_graph(dict)
                # for input_vid in input_vids:
                #     if signature in memoize[input_vid]:
                #         df_list.append(memoize[input_vid][signature])
                #     else:
                #         delta_input_vids.append(input_vid)

                # cached_results = pd.concat(df_list, ignore_index=True)
                # Execute for unseen videos
                encountered_variables_current_graph = sorted(encountered_variables_current_graph, key=lambda x: int(x[1:]))
                tables = ", ".join(["Obj_filtered as {}".format(v) for v in encountered_variables_current_graph])
                where_clauses = []
                # where_clauses.append("{}.vid = ANY(%s)".format(encountered_variables_current_graph[0]))
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
                # print("Time for graph {}: {}".format(graph_idx, time.time() - _start))

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

                # Generate scene graph sequence:
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

                encountered_variables_prev_graphs = obj_union
                encountered_variables_current_graph = []
                # Store new cached results
                # _start = time.time()
                # cur.execute("SELECT * FROM g{}_seq_view".format(graph_idx))
                # df = pd.DataFrame(cur.fetchall())
                # if df.shape[0]: # if results not empty
                #     df.columns = ["vid", "fid1", "fid2"] + [vars_mapping[oid] for oid in oid_list]
                #     for vid, group in df.groupby("vid"):
                #         cached_df = group.reset_index(drop=True)
                #         new_memoize[vid][signature] = cached_df

                # # Appending cached results of seen videos:
                # if cached_results.shape[0]:
                #     tem_table_insert_data = cached_results.copy()
                #     tem_table_insert_data.columns = ["vid", "fid1", "fid2"] + oid_list
                #     for k, v in vars_mapping.items():
                #         tem_table_insert_data[k] = cached_results[v]
                #     buffer = StringIO()
                #     tem_table_insert_data.to_csv(buffer, header=False, index = False)
                #     buffer.seek(0)
                #     cur.copy_from(buffer, "g{}_seq_view".format(graph_idx), sep=",")



                # # Store new cached results
                # cur.execute("SELECT * FROM q{}".format(graph_idx))
                # df = pd.DataFrame(cur.fetchall())
                # if df.shape[0]: # if results not empty
                #     df.columns = [x.name for x in cur.description]
                #     for vid, group in df.groupby("vid"):
                #         cached_df = group.reset_index(drop=True)
                #         new_memoize[vid][signature] = cached_df

                # # Appending cached results of seen videos:
                # if cached_results.shape[0]:
                #     # save dataframe to an in memory buffer
                #     buffer = StringIO()
                #     cached_results.to_csv(buffer, header=False, index = False)
                #     buffer.seek(0)
                #     cur.copy_from(buffer, "q{}".format(graph_idx), sep=",")

            cur.execute("SELECT DISTINCT vid FROM g{}_contiguous".format(len(current_query) - 1))
            # print("SELECT DISTINCT vid FROM g{}_contiguous".format(len(current_query) - 1))
            output_vids = cur.fetchall()
            output_vids = [row[0] for row in output_vids]
            conn.commit()
    return output_vids, new_memoize

def rewrite_program_postgres(orig_program):
    """
    Input:
    program: query in the dictionary format
    Output: query in string format, which is ordered properly (uniquely).
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
    # correct_filename("synthetic-fn_error_rate_0.3-fp_error_rate_0.075")
    # get_query_str_from_filename("inputs/synthetic_rare",)
    # construct_train_test("Sequencing(Sequencing(Sequencing(Sequencing(Sequencing(Sequencing(True*, Back), True*), Left), True*), Conjunction(Conjunction(Back, Left), Far_0.9)), True*)", n_train=300)
    # current_query = [{'scene_graph': [{'predicate': 'LeftOf', 'parameter': None, 'variables': ['o0', 'o1']}], 'duration_constraint': 1}, {'scene_graph': [{'predicate': 'RightQuadrant', 'parameter': None, 'variables': ['o0']}], 'duration_constraint': 1}]
    # query_str = rewrite_program_postgres(current_query)
    # print(query_str)
    # query_str = "Near_1.05(o0, o1); RightQuadrant(o0)"
    # query_str = "Near_1.05(o0, o1); Conjunction(Near_1.05(o0, o1), RightQuadrant(o0))"
    query_str = "Near_1.05(o0, o1); Far_0.9(o0, o1); Near_1.05(o0, o1)"
    current_query = str_to_program_postgres(query_str)
    predicate_list = [{"name": "Near", "parameters": [1.05], "nargs": 2}, {"name": "Far", "parameters": [0.9], "nargs": 2}, {"name": "LeftOf", "parameters": None, "nargs": 2}, {"name": "Behind", "parameters": None, "nargs": 2}, {"name": "RightQuadrant", "parameters": None, "nargs": 1}, {"name": "TopQuadrant", "parameters": None, "nargs": 1}, {"name": "Gray", "parameters": None, "nargs": 1}, {"name": "Red", "parameters": None, "nargs": 1}, {"name": "Blue", "parameters": None, "nargs": 1}, {"name": "Green", "parameters": None, "nargs": 1}, {"name": "Brown", "parameters": None, "nargs": 1}, {"name": "Cyan", "parameters": None, "nargs": 1}, {"name": "Purple", "parameters": None, "nargs": 1}, {"name": "Yellow", "parameters": None, "nargs": 1}, {"name": "Cube", "parameters": None, "nargs": 1}, {"name": "Sphere", "parameters": None, "nargs": 1}, {"name": "Cylinder", "parameters": None, "nargs": 1}, {"name": "Metal", "parameters": None, "nargs": 1}, {"name": "Rubber", "parameters": None, "nargs": 1}, {"name": "Center", "parameters": 0.004, "nargs": 1}, {"name": "Edge", "parameters": 0.5, "nargs": 1}]
    dsn = "dbname=myinner_db user=enhaoz host=localhost"
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            # Create predicate functions (if not exists)
            for predicate in predicate_list:
                args = ", ".join(["text, text, text, double precision, double precision, double precision, double precision"] * predicate["nargs"])
                if predicate["parameters"]:
                    args = "double precision, " + args
                cur.execute("CREATE OR REPLACE FUNCTION {name}({args}) RETURNS boolean AS '/mmfs1/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/postgres/functors', '{name}' LANGUAGE C STRICT;".format(name=predicate["name"], args=args))
            conn.commit()

    memoize = [LRU(10000) for _ in range(10122)]
    # inputs_table_name = "Obj_clevrer"
    inputs_table_name = "Obj_trajectories"
    _start = time.time()
    input_vids = [4857, 1721, 6339, 101, 8647, 1322, 138, 5260, 8041, 4102, 5985, 5189, 8165, 1067, 9325, 680, 5866, 9811, 124, 2120, 6864, 603, 5268, 5290, 9406, 7193, 9875, 5086, 8839, 3897, 4540, 5572, 3990, 2008, 3259, 8255, 8831, 2242, 7228, 136, 841, 5283, 3182, 7018, 2266, 8859, 1716, 9134, 827, 8215, 3490, 3251, 1958, 5831, 6874, 6547, 10078, 3368, 8337, 6209, 4531, 7609, 5312, 9564, 5156, 9722, 6543, 9063, 4104, 9395, 6000, 5361, 4562, 9154, 273, 7902, 1762, 3592, 6496, 5135, 329, 2153, 8303, 1849, 3464, 10067, 3489, 4822, 7251, 9771, 9713, 6867, 1832, 3247, 5718, 10044, 3932, 6595, 1434, 9280, 5002, 7480, 9685, 7471, 2246, 663, 9753, 9551, 4990, 2383, 1879, 9023, 2271, 1401, 5196, 6784, 9733, 9738, 2590, 7267, 8599, 2369, 108, 5262, 8192, 1620, 356, 1599, 1345, 6674, 6648, 3450, 8661, 1572, 725, 9093, 1205, 6152, 10072, 1237, 6284, 3245, 5982, 9983, 6447, 3176, 9766, 8999, 4671, 5426, 6974, 9171, 8188, 3421, 2362, 6055, 1438, 2976, 2747, 7196, 2772, 161, 3813, 2207, 7713, 5443, 3021, 8372, 3686, 5173, 9918, 7963, 6990, 5871, 4907, 2542, 2721, 1813, 6690, 1955, 1922, 4363, 193, 6321, 4211, 749, 6796, 3287, 5224, 86, 4996, 2452, 3325, 9780, 6037, 8849, 7207, 1353, 1581, 1229, 8475, 6802, 1687, 6024, 4739, 6399, 5457, 6025, 1591, 3783, 8864, 3254, 6609, 7315, 1953, 1782, 8972, 1939, 3169, 846, 2767, 10043, 254, 92, 2503, 3278, 4295, 5029, 5983, 2685, 6332, 4945, 5678, 4059, 6555, 7231, 1824, 9315, 2608, 7113, 7244, 43, 4939, 3193, 1736, 5455, 7429, 7335, 664, 8481, 253, 9341, 7264, 7488, 8838, 4398, 951, 5792, 4169, 6480, 5917, 7830, 1192, 7122, 628, 2472, 6915, 2576, 5640, 1286, 8184, 3921, 4088, 2428, 594, 3385, 9858, 2291, 4150, 8530, 7109, 1893, 4924, 3258, 8729, 9647, 7249, 3536, 8881, 5399, 1968, 8444, 1959, 8515, 1311, 1218, 5067, 8333, 5120, 9338, 2180, 7773, 9588, 9956, 883, 1504, 6514, 8375, 7211, 9263, 2875, 9320, 4137, 2612, 9791, 8449, 880, 7138, 596, 3372, 3992, 5265, 9128, 7198, 5072, 4555, 5918, 8293, 8608, 7700, 6472, 5921, 6847, 291, 3178, 3662, 6088, 4957, 4565, 7088, 967, 3496, 3556, 2717, 4593, 5555, 5031, 7245, 9989, 9246, 1834, 2268, 8779, 1911, 9388, 9438, 1240, 8957, 3858, 3576, 955, 908, 7040, 652, 6116, 8425, 3614, 6165, 2521, 9987, 1313, 779, 2408, 7459, 6468, 1204, 4162, 4231, 3552, 7154, 6726, 5966, 8103, 5302, 9409, 6740, 9002, 7490, 6431, 8813, 9001, 5955, 9584, 5651, 7287, 6997, 2549, 5197, 9146, 7283, 1222, 1840, 404, 9595, 7715, 8133, 7334, 18, 8664, 3079, 309, 9345, 2305, 3414, 6263, 10075, 7172, 1569, 5403, 3266, 6482, 5137, 6689, 1809, 3408, 2972, 7469, 8113, 3519, 6231, 7706, 2809, 9489, 9649, 484, 5670, 721, 7835, 4506, 7094, 5867, 372, 6227, 1925, 9303, 8737, 119, 4892, 1884, 2730, 7391, 420, 7779, 7182, 9213, 2799, 4404, 2627, 6678, 3179, 6598, 8056, 1881, 7744, 684, 3929, 1695, 2779, 9109, 1103, 1478, 881, 4729, 7394, 8427, 6719, 8780, 8883, 4296, 770, 7566, 9917, 4768, 4884, 9579, 6637, 8848, 25, 8207, 2433, 690, 8326, 7431, 8508, 5430, 7554, 9856, 3405, 243, 111]
    # outputs, _ = postgres_execute(dsn, current_query, memoize, inputs_table_name, list(range(0, 300)), is_trajectory=False)
    outputs, new_memoize = postgres_execute(dsn, current_query, memoize, inputs_table_name, input_vids, is_trajectory=True, sampling_rate=4)
    print(len(outputs))
    print(outputs)
    print("time", time.time() - _start)

    # preds = []
    # for input in input_vids:
    #     if input in outputs:
    #         preds.append(1)
    #     else:
    #         preds.append(0)
    # score = f1_score(labels, preds)
    # print(score)
    # for i, memo_dict in enumerate(new_memoize):
    #     for k, v in memo_dict.items():
    #         memoize[i][k] = v

    # _start = time.time()
    # current_query = [{'scene_graph': [{'predicate': 'LeftOf', 'parameter': None, 'variables': ['o0', 'o1']}], 'duration_constraint': 1}, {'scene_graph': [{'predicate': 'RightQuadrant', 'parameter': None, 'variables': ['o0']}], 'duration_constraint': 1}, {'scene_graph': [{'predicate': 'LeftQuadrant', 'parameter': None, 'variables': ['o2']}, {'predicate': 'RightOf', 'parameter': None, 'variables': ['o0', 'o1']}], 'duration_constraint': 1}]
    # # outputs, _ = postgres_execute(dsn, current_query, memoize, inputs_table_name, list(range(0, 300)), is_trajectory=True)
    # outputs, new_memoize = postgres_execute(dsn, current_query, memoize, inputs_table_name, list(range(0, 305)), is_trajectory=False)
    # print(len(outputs))
    # print(outputs)
    # print("time", time.time() - _start)

    # # TODO: benchmark: 1. reevaluate a query over 5 more new videos. 2. evaluate a query whose subqueries are already evaluated.