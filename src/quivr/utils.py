import csv
import json
import os
import random
import numpy as np
from sklearn.model_selection import train_test_split
import quivr.dsl as dsl
import psycopg
import copy
from lru import LRU
import time
import pandas as pd
import itertools
import uuid

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
    elif program_str.startswith("BackOf"):
        return dsl.BackOf()
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
            new_query_str = query_str.replace("Back", "BackOf")
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

def postgres_execute(dsn, current_query, input_vids, memoize, inputs_table_name):
    """
    input_vids: list of video segment ids
    Example query:
        Sequencing(Sequencing(Sequencing(Sequencing(Sequencing(Sequencing(True*, Near_1.05), True*), Conjunction(LeftOf, BackOf)), True*), Duration(Conjunction(TopQuadrant, Far_0.9), 5)), True*)
    two types of caches:
        1. scene graph cache: cache[graph] = vid, fid1, fid2, oids
        2. sequence cache: cache[sequence] = vid, fid1, fid2, oids
    Output:
    new_memoize: new cached results from this query, which will be added to the global cache (for multi-threading)
    """
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            new_memoize = [LRU(10000) for _ in range(len(memoize))]
            index_names = []
            # select input videos
            cur.execute("CREATE TEMPORARY TABLE Obj_filtered AS SELECT * FROM {} WHERE vid = ANY(%s);".format(inputs_table_name), [input_vids])
            cur.execute("CREATE INDEX IF NOT EXISTS idx_obj_filtered ON Obj_filtered (vid, fid);")
            encountered_variables_all_graphs = []
            for graph_idx, dict in enumerate(current_query):
                # Generate scene graph:
                scene_graph = dict["scene_graph"]
                duration_constraint = dict["duration_constraint"]
                encountered_variables = set()
                for p in scene_graph:
                    for v in p["variables"]:
                        if v not in encountered_variables:
                            encountered_variables.add(v)
                encountered_variables_all_graphs.append(encountered_variables)
                delta_input_vids = []
                df_list = [pd.DataFrame()]
                signature = rewrite_program_postgres([dict])
                for input_vid in input_vids:
                    if signature in memoize[input_vid]:
                        df_list.append(memoize[input_vid][signature])
                    else:
                        delta_input_vids.append(input_vid)

                cached_results = pd.concat(df_list, ignore_index=True)

                # Execute for unseen videos
                tables = ", ".join(["Obj_filtered as {}".format(v) for v in encountered_variables])
                encountered_variables_list = list(encountered_variables)
                where_clauses = []
                where_clauses.append("{}.vid = ANY(%s)".format(encountered_variables_list[0]))
                for i in range(len(encountered_variables_list)-1):
                    where_clauses.append("{v1}.vid = {v2}.vid and {v1}.fid = {v2}.fid".format(v1=encountered_variables_list[i], v2=encountered_variables_list[i+1])) # join variables
                for p in scene_graph:
                    predicate = p["predicate"]
                    parameter = p["parameter"]
                    variables = p["variables"]
                    args = []
                    for v in variables:
                        args.append("{v}.x1, {v}.y1, {v}.x2, {v}.y2".format(v=v))
                    args = ", ".join(args)
                    if parameter:
                        args = "{}, {}".format(parameter, args)
                    where_clauses.append("{}({}) = true".format(predicate, args))
                # NOTE: only for trajectory example
                for v in encountered_variables_list:
                    where_clauses.append("{}.oid = {}".format(v, v[1:]))
                # For general case
                # for var_pair in itertools.combinations(encountered_variables_list, 2):
                #     where_clauses.append("{}.oid <> {}.oid".format(var_pair[0], var_pair[1]))
                where_clauses = " and ".join(where_clauses)
                fields = "{v}.vid as vid, {v}.fid as fid, ".format(v=encountered_variables_list[0])
                fields += ", ".join(["{v}.oid as {v}_oid".format(v=v) for v in encountered_variables_list])
                oid_list = ["{}_oid".format(v) for v in encountered_variables_list]
                oids = ", ".join(oid_list)
                index_fields = "vid, fid, " + oids
                sql_sring = """CREATE TEMPORARY TABLE g{} AS SELECT {} FROM {} WHERE {};""".format(graph_idx, fields, tables, where_clauses)
                cur.execute(sql_sring, [delta_input_vids])

                # Create index for g{i}:
                cur.execute("CREATE INDEX IF NOT EXISTS idx_g{i} ON g{i} ({fields});".format(i=graph_idx, fields=index_fields))
                index_names.append("idx_g{}".format(graph_idx))

                # Generate scene graph sequence:
                base_fields = "vid, fid, fid, {}".format(oids)
                step_fields = "s.vid, s.fid1, g.fid, {}".format(", ".join(["s.{}".format(oid) for oid in oid_list]))
                view_fields = "vid, fid1, fid2, {}".format(oids)
                where_clauses = "s.vid = g.vid and g.fid = s.fid2 + 1 and {} and g.fid - s.fid1 + 1 <= {}".format(" and ".join(["s.{oid} = g.{oid}".format(oid=oid) for oid in oid_list]), duration_constraint)
                sql_string = """
                    CREATE TEMPORARY TABLE g{graph_idx}_seq_view AS
                    WITH RECURSIVE g{graph_idx}_seq ({view_fields}) AS (
                        SELECT {base_fields} FROM g{graph_idx}
                            UNION ALL
                        SELECT {step_fields}
                        FROM g{graph_idx}_seq s, g{graph_idx} g
                        WHERE {where_clauses}
                    )
                    SELECT DISTINCT * FROM g{graph_idx}_seq WHERE fid2 - fid1 + 1 = {duration_constraint};
                    """.format(graph_idx=graph_idx, view_fields=view_fields, base_fields=base_fields, step_fields=step_fields, where_clauses=where_clauses, duration_constraint=duration_constraint)
                cur.execute(sql_string)

                # Store new cached results
                cur.execute("SELECT * FROM g{}_seq_view".format(graph_idx))
                df = pd.DataFrame(cur.fetchall())
                df.columns = [x.name for x in cur.description]
                for vid, group in df.groupby("vid"):
                    new_memoize[vid][signature] = group

                # Appending cached results of seen videos:
                if cached_results.shape[0]:
                    placeholder = '(' + ','.join(['%s' for i in range(3 + len(oid_list))]) + ')'
                    cur.executemany("INSERT INTO g{}_seq_view VALUES {};".format(graph_idx, placeholder), cached_results)

                # Create index for g{i}_seq_view:
                cur.execute("CREATE INDEX IF NOT EXISTS idx_g{i}_seq_view ON g{i}_seq_view ({fields});".format(i=graph_idx, fields=view_fields))
                index_names.append("idx_g{}_seq_view".format(graph_idx))

            # Sequencing
            current_seq = "g0_seq_view"
            current_encountered_variables = copy.deepcopy(encountered_variables_all_graphs[0])
            for graph_idx in range(len(current_query) - 1):
                # Read cached results
                delta_input_vids = []
                df_list = [pd.DataFrame()]
                signature = rewrite_program_postgres(current_query[:graph_idx+2])
                for input_vid in input_vids:
                    if signature in memoize[input_vid]:
                        df_list.append(memoize[input_vid][signature])
                    else:
                        delta_input_vids.append(input_vid)
                cached_results = pd.concat(df_list, ignore_index=True)

                # Execute for unseen videos
                obj_fields = []
                obj_groupby = []
                index_obj_fields = []
                for v in current_encountered_variables:
                    obj_fields.append("t0.{}_oid as {}_oid".format(v, v))
                    obj_groupby.append("t0.{}_oid".format(v))
                    index_obj_fields.append("{}_oid".format(v))
                obj_intersection = []
                for v in encountered_variables_all_graphs[graph_idx+1]:
                    if v in current_encountered_variables:
                        obj_intersection.append(v)
                    else:
                        current_encountered_variables.add(v)
                        obj_fields.append("t1.{v}_oid as {v}_oid".format(v=v))
                        obj_groupby.append("t1.{}_oid".format(v))
                        index_obj_fields.append("{}_oid".format(v))
                obj_fields = ", ".join(obj_fields)
                index_obj_fields = ", ".join(index_obj_fields)
                fields = "t0.vid as vid, min(t1.fid2) as fid, {}".format(obj_fields)
                where_clauses = "t0.vid = ANY(%s)"
                if current_seq == "g0_seq_view":
                    where_clauses += " and t0.vid = t1.vid and t0.fid2 < t1.fid1"
                else:
                    where_clauses += " and t0.vid = t1.vid and t0.fid < t1.fid1"
                groupby_clauses = "t0.vid"
                for v in obj_intersection:
                    where_clauses += " and t0.{v}_oid = t1.{v}_oid".format(v=v)
                groupby_clauses += ", {}".format(", ".join(obj_groupby))
                index_fields = "vid, fid, {}".format(index_obj_fields)
                sql_string = """
                CREATE TEMPORARY TABLE q{i} AS
                SELECT DISTINCT {fields}
                FROM {current_seq} as t0, g{j}_seq_view as t1
                WHERE {where_clauses}
                GROUP BY {groupby_clauses};
                """.format(i=graph_idx, j=graph_idx+1, fields=fields, where_clauses=where_clauses, current_seq=current_seq, groupby_clauses=groupby_clauses)
                # print(sql_string)
                cur.execute(sql_string, [delta_input_vids])

                # Store new cached results
                cur.execute("SELECT * FROM q{}".format(graph_idx))
                df = pd.DataFrame(cur.fetchall())
                df.columns = [x.name for x in cur.description]
                for vid, group in df.groupby("vid"):
                    new_memoize[vid][signature] = group

                # Appending cached results of seen videos:
                if cached_results.shape[0]:
                    placeholder = '(' + ','.join(['%s' for i in range(3 + len(oid_list))]) + ')'
                    cur.executemany("INSERT INTO q{} VALUES {};".format(graph_idx, placeholder), cached_results)

                # Create index
                cur.execute("CREATE INDEX IF NOT EXISTS idx_q{} ON q{} ({});".format(graph_idx, graph_idx, index_fields))
                index_names.append("idx_q{}".format(graph_idx))

                current_seq = "q{}".format(graph_idx)
            cur.execute("SELECT DISTINCT vid FROM {}".format(current_seq))
            output_vids = cur.fetchall()
            output_vids = [row[0] for row in output_vids]
            conn.commit()
    return output_vids, new_memoize

def rewrite_program_postgres(program):
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
        dict["parameter"] = float(predicate_name[1]) if len(predicate_name) > 1 else None
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


if __name__ == '__main__':
    # correct_filename("synthetic-fn_error_rate_0.3-fp_error_rate_0.075")
    # get_query_str_from_filename("inputs/synthetic_rare",)
    # construct_train_test("Sequencing(Sequencing(Sequencing(Sequencing(Sequencing(Sequencing(True*, Back), True*), Left), True*), Conjunction(Conjunction(Back, Left), Far_0.9)), True*)", n_train=300)
    current_query = [
        {
            "scene_graph": [
                {
                    "predicate": "Near",
                    "parameter": 1.05,
                    "variables": ["o0", "o1"]
                }
            ],
            "duration_constraint": 1
        },
        {
            "scene_graph": [
                {
                    "predicate": "LeftOf",
                    "parameter": None,
                    "variables": ["o0", "o1"]
                },
                {
                    "predicate": "Behind",
                    "parameter": None,
                    "variables": ["o0", "o1"]
                }
            ],
            "duration_constraint": 1
        },
        {
            "scene_graph": [
                {
                    "predicate": "TopQuadrant",
                    "parameter": None,
                    "variables": ["o0"]
                },
                {
                    "predicate": "Far",
                    "parameter": 0.9,
                    "variables": ["o0", "o1"]
                }
            ],
            "duration_constraint": 5
        }
    ]
    # query_str = rewrite_program_postgres(current_query)
    # print(query_str)
    # print(str_to_program_postgres(query_str))
    _start = time.time()
    current_query = str_to_program_postgres("LeftOf(o0, o1); Near_1.05(o0, o1)")
    predicate_list = [{"name": "Near", "parameters": [1.05], "nargs": 2}, {"name": "Far", "parameters": [0.9], "nargs": 2}, {"name": "LeftOf", "parameters": None, "nargs": 2}, {"name": "Behind", "parameters": None, "nargs": 2}, {"name": "RightQuadrant", "parameters": None, "nargs": 1}, {"name": "TopQuadrant", "parameters": None, "nargs": 1}]
    with open("/mmfs1/gscratch/balazinska/enhaoz/postgres_server.info", 'r') as f:
        host = f.readlines()[0].strip().split(" ")[0]
    dsn = "dbname=myinner_db user=enhaoz host={}".format(host)
    memoize = [LRU(10000) for _ in range(300)]
    with open("inputs/synthetic_rare/train/Sequencing(Sequencing(Sequencing(Sequencing(Sequencing(Sequencing(True*, Near_1.05), True*), Conjunction(LeftOf, BackOf)), True*), Duration(Conjunction(TopQuadrant, Far_0.9), 5)), True*)_inputs.json", 'r') as f:
        inputs = json.load(f)
    inputs = np.asarray(inputs, dtype=object)
    inputs_table_name = "Obj_trajectories_{}".format(uuid.uuid4().hex)
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            # Create temporary table for inputs
            cur.execute("""
            CREATE TABLE {} (
                oid INT,
                vid INT,
                fid INT,
                shape varchar,
                color varchar,
                material varchar,
                x1 float,
                y1 float,
                x2 float,
                y2 float
            );
            """.format(inputs_table_name))

            # Load inputs into temporary table
            csv_data = []
            for vid, pair in enumerate(inputs):
                t1 = pair[0]
                t2 = pair[1]
                assert(len(t1) == len(t2))
                for fid, (bbox1, bbox2) in enumerate(zip(t1, t2)):
                    csv_data.append((0, vid, fid, "cube", "red", "metal", bbox1[0], bbox1[1], bbox1[2], bbox1[3]))
                    csv_data.append((1, vid, fid, "cube", "red", "metal", bbox2[0], bbox2[1], bbox2[2], bbox2[3]))
            with cur.copy("COPY {} FROM STDIN".format(inputs_table_name)) as cur_copy:
                for row in csv_data:
                    cur_copy.write_row(row)
            conn.commit()
    print("prepare time", time.time() - _start)
    _start = time.time()
    outputs, _ = postgres_execute(dsn, current_query, list(range(0, 300)), memoize, inputs_table_name)
    print(len(outputs))
    print("time", time.time() - _start)
    _start = time.time()
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("DROP TABLE {};".format(inputs_table_name))
    print("clean up time", time.time() - _start)