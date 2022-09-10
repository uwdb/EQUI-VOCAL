import csv
import json
import os
import random
import numpy as np
from sklearn.model_selection import train_test_split
import quivr.dsl as dsl
import psycopg
import copy
import time


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

def postgres_execute(current_query, input_vids):
    """
    input_vids: list of video segment ids
    Example query:
        Sequencing(Sequencing(Sequencing(Sequencing(Sequencing(Sequencing(True*, Near_1.05), True*), Conjunction(LeftOf, BackOf)), True*), Duration(Conjunction(TopQuadrant, Far_0.9), 5)), True*)
    current_query format:
    [
        {
            "scene_graph": [
                {
                    "predicate": "Near",
                    "variables": ["o1", "o2"]
                }
            ],
            "duration_constraint": 1
        },
        {
            "scene_graph": [
                {
                    "predicate": "LeftOf",
                    "variables": ["o1", "o2"]
                },
                {
                    "predicate": "BackOf",
                    "variables": ["o1", "o2"]
                }
            ],
            "duration_constraint": 1
        },
        {
            "scene_graph": [
                {
                    "predicate": "TopQuadrant",
                    "variables": ["o1"]
                },
                {
                    "predicate": "Far",
                    "variables": ["o1", "o2"]
                }
            ],
            "duration_constraint": 5
        }
    ]
    """
    stmts = []
    index_names = []
    # select input videos
    stmts.append(["CREATE TEMPORARY TABLE Obj_filtered AS SELECT * FROM Obj_trajectories WHERE vid = ANY(%s);", [input_vids]])
    stmts.append(["CREATE INDEX idx_obj_filtered ON Obj_filtered (vid, fid);", None])
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
        tables = ", ".join(["Obj_filtered as {}".format(v) for v in encountered_variables])
        where_clauses = []
        encountered_variables_list = list(encountered_variables)
        for i in range(len(encountered_variables_list)-1):
            where_clauses.append("{v1}.vid = {v2}.vid and {v1}.fid = {v2}.fid".format(v1=encountered_variables_list[i], v2=encountered_variables_list[i+1])) # join conditions
        for p in scene_graph:
            predicate = p["predicate"]
            variables = p["variables"]
            args = []
            for v in variables:
                args.append("{}.x1, {}.y1, {}.x2, {}.y2".format(v, v, v, v))
            args = ", ".join(args)
            where_clauses.append("{}({}) = true".format(predicate, args))
        # NOTE: only for trajectory example
        for v in encountered_variables:
            where_clauses.append("{}.oid = {}".format(v, int(v[1:]) - 1)) # 0-indexed
        where_clauses = " and ".join(where_clauses)
        fields = ["{v}.vid as vid, {v}.fid as fid".format(v=encountered_variables_list[0])]
        oids = ["{}.oid as {}_oid".format(v, v) for v in encountered_variables]
        fields.extend(oids)
        fields = ", ".join(fields)
        index_fields = "vid, fid, " + ", ".join(["{}_oid".format(v) for v in encountered_variables])
        sql_sring = "CREATE TEMPORARY TABLE g{} AS SELECT {} FROM {} WHERE {};".format(graph_idx, fields, tables, where_clauses)
        stmts.append([sql_sring, None])
        # Create index for g{graph_idx}:
        stmts.append(["CREATE INDEX idx_g{} ON g{} ({});".format(graph_idx, graph_idx, index_fields), None])
        index_names.append("idx_g{}".format(graph_idx))
        # Generate scene graph sequence:
        oids = ["{}_oid".format(v, v) for v in encountered_variables]
        base_fields = "vid, fid, fid, {}".format(", ".join(oids))
        step_fields = "s.vid, s.fid1, g.fid, {}".format(", ".join(["s.{}".format(oid) for oid in oids]))
        view_fields = "vid, fid1, fid2, {}".format(", ".join(oids))
        where_clauses = "s.vid = g.vid and g.fid = s.fid2 + 1 and {}".format(" and ".join(["s.{} = g.{}".format(oid, oid) for oid in oids]))
        duration_clause = "fid2 - fid1 + 1 >= {}".format(duration_constraint)
        sql_string = """
            CREATE TEMPORARY TABLE g{graph_idx}_seq_view AS
            WITH RECURSIVE g{graph_idx}_seq (vid, fid1, fid2, {oids}) AS (
                -- base case
                SELECT {base_fields} FROM g{graph_idx}
                    UNION ALL
                -- step case
                SELECT {step_fields}
                FROM g{graph_idx}_seq s, g{graph_idx} g
                WHERE {where_clauses}
            )
            SELECT DISTINCT {view_fields} FROM g{graph_idx}_seq
            WHERE {duration_clause};
            """.format(graph_idx=graph_idx, oids=", ".join(oids), base_fields=base_fields, step_fields=step_fields, where_clauses=where_clauses, view_fields=view_fields, duration_clause=duration_clause)
        stmts.append([sql_string, None])
        # Create index for g{graph_idx}_seq_view:
        stmts.append(["CREATE INDEX idx_g{}_seq_view ON g{}_seq_view ({});".format(graph_idx, graph_idx, view_fields), None])
        index_names.append("idx_g{}_seq_view".format(graph_idx))
    # Sequencing
    current_seq = "g0_seq_view"
    current_encountered_variables = copy.deepcopy(encountered_variables_all_graphs[0])
    for graph_idx in range(len(current_query) - 1):
        where_clauses = "t1.vid = t2.vid and t1.fid2 < t2.fid1"
        obj_fields = []
        index_obj_fields = []
        for v in current_encountered_variables:
            obj_fields.append("t1.{}_oid as {}_oid".format(v, v))
            index_obj_fields.append("{}_oid".format(v))
        for v in encountered_variables_all_graphs[graph_idx+1]:
            if v in current_encountered_variables:
                where_clauses += " and t1.{v}_oid = t2.{v}_oid".format(v=v)
            else:
                current_encountered_variables.add(v)
                obj_fields.append("t2.{v}_oid as {v}_oid".format(v=v))
                index_obj_fields.append("{}_oid".format(v))
        obj_fields = ", ".join(obj_fields)
        index_obj_fields = ", ".join(index_obj_fields)
        fields = "t1.vid as vid, t1.fid1 as fid1, t2.fid2 as fid2, {}".format(obj_fields)
        index_fields = "vid, fid1, fid2, {}".format(index_obj_fields)
        sql_string = """
        CREATE TEMPORARY TABLE q{graph_idx} AS
        SELECT DISTINCT {fields}
        FROM {current_seq} as t1, g{graph_idx2}_seq_view as t2
        WHERE {where_clauses};
        """.format(graph_idx=graph_idx, graph_idx2=graph_idx+1, fields=fields, where_clauses=where_clauses, current_seq=current_seq)
        stmts.append([sql_string, None])
        current_seq = "q{}".format(graph_idx)
        # Create index
        stmts.append(["CREATE INDEX idx_q{} ON q{} ({});".format(graph_idx, graph_idx, index_fields), None])
        index_names.append("idx_q{}".format(graph_idx))
    stmts.append(["SELECT DISTINCT vid FROM {}".format(current_seq), None])
    print("stmts", stmts)
    with psycopg.connect("dbname=myinner_db user=enhaoz host=localhost") as conn:
        # Open a cursor to perform database operations
        with conn.cursor() as cur:
            # NOTE: temp
            cur.execute("DROP FUNCTION IF EXISTS Near, Far, LeftOf, BackOf, RightQuadrant, TopQuadrant;")
            cur.execute("DROP INDEX IF EXISTS idx_obj_filtered, idx_g1, idx_g2, idx_g3, idx_g1_seq_view, idx_g2_seq_view, idx_g3_seq_view, idx_q1;")
            cur.execute("""CREATE FUNCTION Near(double precision, double precision, double precision, double precision, double precision, double precision, double precision, double precision) RETURNS boolean
            AS '/mmfs1/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/postgres/functors', 'near'
            LANGUAGE C STRICT;""")
            cur.execute("""CREATE FUNCTION Far(double precision, double precision, double precision, double precision, double precision, double precision, double precision, double precision) RETURNS boolean
            AS '/mmfs1/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/postgres/functors', 'far'
            LANGUAGE C STRICT;""")
            cur.execute("""CREATE FUNCTION LeftOf(double precision, double precision, double precision, double precision, double precision, double precision, double precision, double precision) RETURNS boolean
            AS '/mmfs1/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/postgres/functors', 'left_of'
            LANGUAGE C STRICT;""")
            cur.execute("""CREATE FUNCTION BackOf(double precision, double precision, double precision, double precision, double precision, double precision, double precision, double precision) RETURNS boolean
            AS '/mmfs1/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/postgres/functors', 'behind'
            LANGUAGE C STRICT;""")
            cur.execute("""CREATE FUNCTION RightQuadrant(double precision, double precision, double precision, double precision) RETURNS boolean
            AS '/mmfs1/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/postgres/functors', 'right_quadrant'
            LANGUAGE C STRICT;""")
            cur.execute("""CREATE FUNCTION TopQuadrant(double precision, double precision, double precision, double precision) RETURNS boolean
            AS '/mmfs1/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/postgres/functors', 'top_quadrant'
            LANGUAGE C STRICT;""")
            for stmt, vars in stmts:
                cur.execute(stmt, vars)
            output_vids = cur.fetchall()
            output_vids = [row[0] for row in output_vids]
            cur.execute("DROP INDEX IF EXISTS {};".format(", ".join(index_names)))
            # Make the changes to the database persistent
            conn.commit()
    print("output_vids", output_vids)
    return output_vids

if __name__ == '__main__':
    # correct_filename("synthetic-fn_error_rate_0.3-fp_error_rate_0.075")
    # get_query_str_from_filename("inputs/synthetic_rare",)
    # construct_train_test("Sequencing(Sequencing(Sequencing(Sequencing(Sequencing(Sequencing(True*, Back), True*), Left), True*), Conjunction(Conjunction(Back, Left), Far_0.9)), True*)", n_train=300)
    current_query = [
        {
            "scene_graph": [
                {
                    "predicate": "Near",
                    "variables": ["o1", "o2"]
                }
            ],
            "duration_constraint": 1
        },
        {
            "scene_graph": [
                {
                    "predicate": "LeftOf",
                    "variables": ["o1", "o2"]
                },
                {
                    "predicate": "BackOf",
                    "variables": ["o1", "o2"]
                }
            ],
            "duration_constraint": 1
        },
        {
            "scene_graph": [
                {
                    "predicate": "TopQuadrant",
                    "variables": ["o1"]
                },
                {
                    "predicate": "Far",
                    "variables": ["o1", "o2"]
                }
            ],
            "duration_constraint": 5
        }
    ]
    _start = time.time()
    postgres_execute(current_query, list(range(0, 100)))
    print("time", time.time() - _start)