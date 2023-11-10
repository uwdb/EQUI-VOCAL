from utils import program_to_dsl, dsl_to_program, complexity_cost
import dsl
import copy
import time
import itertools
from concurrent.futures import ThreadPoolExecutor
import psycopg2 as psycopg
import argparse
import numpy as np
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

def postgres_execute_cache_sequence(job_id):
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
    _time = time.time()
    is_trajectory=False
    sampling_rate=None
    current_query= current_query = [{'scene_graph': [{'predicate': 'LeftOf', 'parameter': None, 'variables': ['o0', 'o1']}], 'duration_constraint': 1}, {'scene_graph': [{'predicate': 'RightQuadrant', 'parameter': None, 'variables': ['o0']}], 'duration_constraint': 1}, {'scene_graph': [{'predicate': 'LeftQuadrant', 'parameter': None, 'variables': ['o2']}, {'predicate': 'RightOf', 'parameter': None, 'variables': ['o0', 'o1']}], 'duration_constraint': 1}]
    memoize_scene_graph = [{} for _ in range(10000)]
    memoize_sequence = [{} for _ in range(10000)]
    inputs_table_name = "Obj_clevrer"
    input_vids = [8388, 4753, 1039, 1711, 1657, 8019, 8284, 2520, 5327, 9150, 2752, 2499, 2333, 5523, 6545, 1873, 6010, 1712, 7982, 8917, 232, 3861, 8065, 6326, 5501, 9604, 4362, 4447, 2398, 3486, 6572, 535, 539, 7784, 5092, 3053, 8504, 6820, 2765, 6752, 4029, 3155, 3399, 4220, 8746, 6178, 7217, 8440, 2596, 6529, 9051, 9765, 5199, 4314, 7406, 8737, 9658, 2693, 135, 6381, 4526, 8946, 4373, 2610, 9902, 56, 1234, 684, 9175, 8046, 8243, 1337, 1567, 7803, 1169, 907, 6244, 8407, 5655, 6850, 8819, 8316, 4967, 3812, 94, 3403, 2667, 9084, 1126, 7971, 259, 8573, 1628, 3522, 8115, 4527, 2191, 5674, 3090, 6461, 9379, 3824, 7899, 5801, 6561, 6453, 5771, 3075, 8144, 5010, 1732, 1018, 8666, 8014, 6774, 1342, 796, 4346, 4652, 144, 3519, 7401, 8380, 8587, 5461, 7508, 1191, 8272, 8634, 1976, 8203, 7572, 1922, 4499, 1405, 8220, 2652, 5218, 3340, 1885, 8916, 1434, 2841, 5615, 5312, 1214, 9907, 6608, 2450, 1680, 1896, 2032, 3634, 4184, 6434, 6227, 2683, 6963, 3164, 9989, 6715, 5460, 2780, 5175, 5877, 2971, 8663, 2304, 9495, 6120, 3802, 9486, 156, 2026, 5055, 1832, 2804, 7609, 3153, 1245, 9140, 9491, 2870, 5714, 4620, 7044, 8231, 5856, 2314, 6578, 9610, 6594, 8056, 545, 4065, 6729, 3138, 6257, 3044, 2349, 2040, 5658, 8521, 1226, 2849, 7718, 8077, 9258, 4651, 554, 3526, 6014, 4641, 752, 3424, 1790, 5251, 6251, 4909, 3951, 1684, 416, 2151, 4932, 6408, 2469, 3481, 6621, 7601, 7563, 8697, 1850, 402, 2671, 8566, 3088, 2027, 7690, 4012, 4399, 2982, 6436, 5617, 915, 5593, 6939, 1520, 9817, 6423, 3941, 6241, 7662, 3007, 5597, 7014, 9346, 362, 2500, 4660, 7761, 3179, 1328, 8153, 6318, 6470, 9015, 2510, 7991, 5693, 8605, 1032, 3814, 2805, 2298, 7631, 2570, 7054, 5219, 1637, 8790, 1908, 3449, 1298, 4280, 8066, 5290, 9012, 5866, 619, 6890, 6323, 3840, 53, 472, 4359, 8209, 3134, 1727, 1344, 441, 9213, 4626, 6377, 1650, 7589, 1007, 1563, 3282, 5721, 989, 2455, 8099, 8955, 8551, 9069, 9946, 1798, 6635, 978, 7140, 5511, 2660, 987, 6893, 9173, 9246, 4166, 2854, 8408, 8688, 7311, 9575, 6274, 4074, 9447, 4891, 8646, 8594, 8834, 4727, 4544, 4368, 2678, 9542, 6595, 3919, 2509, 9785, 9860, 5422, 8345, 8178, 1299, 913, 4264, 5036, 2830, 468, 2519, 2418, 7846, 7388, 7831, 3289, 6269, 7270, 2936, 7476, 4045, 3414, 3794, 4632, 1010, 4559, 551, 5081, 692, 7998, 2052, 8441, 1781, 9690, 8201, 9640, 7727, 3103, 904, 3483, 1467, 8840, 2697, 2167, 7442, 8367, 8947, 3354, 5383, 2672, 4787, 8602, 7325, 2981, 2261, 7837, 9443, 2952, 8745, 2977, 9392, 2091, 2589, 7716, 1695, 122, 4913, 645, 3238, 5124, 6350, 1152, 4150, 7521, 7125, 614, 2501, 8322, 9676, 412, 7172, 1323, 2867, 2600, 3970, 2864, 7296, 9768, 8422, 1725, 596, 196, 8818, 4363, 3326, 1440, 6940, 7278, 7266, 4682, 8914, 9916, 4185, 8870, 5228, 761, 644, 4782, 9894, 6507, 5646, 3336, 5551, 8221, 5602, 4623, 7425, 2833, 9232, 7613, 8876, 4560, 4995, 7397, 1135, 7932, 509, 5751, 564, 6961, 9534, 6843, 6728, 4639, 8932, 7747, 7412, 506, 9694, 7470, 9776, 8583, 9497, 2252, 2306, 8673, 641, 7630, 4607, 5215, 7657, 2327, 695, 2294, 200, 9341, 1636, 6662, 9756, 7479, 6930, 4041, 3580, 2944, 9511, 2232, 8945, 2770, 9087, 6688, 7853, 1274, 3378, 3380, 2254, 7679, 6487, 3817]
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
                seq_signature = program_to_dsl(current_query[:len(current_query)-i])
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
    return "job: {}, time: {}".format(job_id, time.time() - _time)


def postgres_execute_no_caching(job_id):
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
    _time = time.time()
    is_trajectory=False
    sampling_rate=None
    current_query= current_query = [{'scene_graph': [{'predicate': 'LeftOf', 'parameter': None, 'variables': ['o0', 'o1']}], 'duration_constraint': 1}, {'scene_graph': [{'predicate': 'RightQuadrant', 'parameter': None, 'variables': ['o0']}], 'duration_constraint': 1}, {'scene_graph': [{'predicate': 'LeftQuadrant', 'parameter': None, 'variables': ['o2']}, {'predicate': 'RightOf', 'parameter': None, 'variables': ['o0', 'o1']}], 'duration_constraint': 1}]
    memoize_scene_graph = [{} for _ in range(10000)]
    memoize_sequence = [{} for _ in range(10000)]
    inputs_table_name = "Obj_clevrer"
    input_vids = [8388, 4753, 1039, 1711, 1657, 8019, 8284, 2520, 5327, 9150, 2752, 2499, 2333, 5523, 6545, 1873, 6010, 1712, 7982, 8917, 232, 3861, 8065, 6326, 5501, 9604, 4362, 4447, 2398, 3486, 6572, 535, 539, 7784, 5092, 3053, 8504, 6820, 2765, 6752, 4029, 3155, 3399, 4220, 8746, 6178, 7217, 8440, 2596, 6529, 9051, 9765, 5199, 4314, 7406, 8737, 9658, 2693, 135, 6381, 4526, 8946, 4373, 2610, 9902, 56, 1234, 684, 9175, 8046, 8243, 1337, 1567, 7803, 1169, 907, 6244, 8407, 5655, 6850, 8819, 8316, 4967, 3812, 94, 3403, 2667, 9084, 1126, 7971, 259, 8573, 1628, 3522, 8115, 4527, 2191, 5674, 3090, 6461, 9379, 3824, 7899, 5801, 6561, 6453, 5771, 3075, 8144, 5010, 1732, 1018, 8666, 8014, 6774, 1342, 796, 4346, 4652, 144, 3519, 7401, 8380, 8587, 5461, 7508, 1191, 8272, 8634, 1976, 8203, 7572, 1922, 4499, 1405, 8220, 2652, 5218, 3340, 1885, 8916, 1434, 2841, 5615, 5312, 1214, 9907, 6608, 2450, 1680, 1896, 2032, 3634, 4184, 6434, 6227, 2683, 6963, 3164, 9989, 6715, 5460, 2780, 5175, 5877, 2971, 8663, 2304, 9495, 6120, 3802, 9486, 156, 2026, 5055, 1832, 2804, 7609, 3153, 1245, 9140, 9491, 2870, 5714, 4620, 7044, 8231, 5856, 2314, 6578, 9610, 6594, 8056, 545, 4065, 6729, 3138, 6257, 3044, 2349, 2040, 5658, 8521, 1226, 2849, 7718, 8077, 9258, 4651, 554, 3526, 6014, 4641, 752, 3424, 1790, 5251, 6251, 4909, 3951, 1684, 416, 2151, 4932, 6408, 2469, 3481, 6621, 7601, 7563, 8697, 1850, 402, 2671, 8566, 3088, 2027, 7690, 4012, 4399, 2982, 6436, 5617, 915, 5593, 6939, 1520, 9817, 6423, 3941, 6241, 7662, 3007, 5597, 7014, 9346, 362, 2500, 4660, 7761, 3179, 1328, 8153, 6318, 6470, 9015, 2510, 7991, 5693, 8605, 1032, 3814, 2805, 2298, 7631, 2570, 7054, 5219, 1637, 8790, 1908, 3449, 1298, 4280, 8066, 5290, 9012, 5866, 619, 6890, 6323, 3840, 53, 472, 4359, 8209, 3134, 1727, 1344, 441, 9213, 4626, 6377, 1650, 7589, 1007, 1563, 3282, 5721, 989, 2455, 8099, 8955, 8551, 9069, 9946, 1798, 6635, 978, 7140, 5511, 2660, 987, 6893, 9173, 9246, 4166, 2854, 8408, 8688, 7311, 9575, 6274, 4074, 9447, 4891, 8646, 8594, 8834, 4727, 4544, 4368, 2678, 9542, 6595, 3919, 2509, 9785, 9860, 5422, 8345, 8178, 1299, 913, 4264, 5036, 2830, 468, 2519, 2418, 7846, 7388, 7831, 3289, 6269, 7270, 2936, 7476, 4045, 3414, 3794, 4632, 1010, 4559, 551, 5081, 692, 7998, 2052, 8441, 1781, 9690, 8201, 9640, 7727, 3103, 904, 3483, 1467, 8840, 2697, 2167, 7442, 8367, 8947, 3354, 5383, 2672, 4787, 8602, 7325, 2981, 2261, 7837, 9443, 2952, 8745, 2977, 9392, 2091, 2589, 7716, 1695, 122, 4913, 645, 3238, 5124, 6350, 1152, 4150, 7521, 7125, 614, 2501, 8322, 9676, 412, 7172, 1323, 2867, 2600, 3970, 2864, 7296, 9768, 8422, 1725, 596, 196, 8818, 4363, 3326, 1440, 6940, 7278, 7266, 4682, 8914, 9916, 4185, 8870, 5228, 761, 644, 4782, 9894, 6507, 5646, 3336, 5551, 8221, 5602, 4623, 7425, 2833, 9232, 7613, 8876, 4560, 4995, 7397, 1135, 7932, 509, 5751, 564, 6961, 9534, 6843, 6728, 4639, 8932, 7747, 7412, 506, 9694, 7470, 9776, 8583, 9497, 2252, 2306, 8673, 641, 7630, 4607, 5215, 7657, 2327, 695, 2294, 200, 9341, 1636, 6662, 9756, 7479, 6930, 4041, 3580, 2944, 9511, 2232, 8945, 2770, 9087, 6688, 7853, 1274, 3378, 3380, 2254, 7679, 6487, 3817]
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
    return "job: {}, time: {}".format(job_id, time.time() - _time)

def get_query_score(job_id):
    _time = time.time()
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE TEMPORARY TABLE Obj_filtered AS SELECT oid, vid, fid, shape, color, material, x1, y1, x2, y2 FROM Obj_clevrer_temp WHERE vid < 500")
            cur.execute("CREATE TEMPORARY TABLE g0 AS SELECT o0.vid as vid, o0.fid as fid, o0.oid as o0_oid, o1.oid as o1_oid, o2.oid as o2_oid FROM Obj_filtered as o1, Obj_filtered as o0, Obj_filtered as o2 WHERE o0.vid = o1.vid and o0.fid = o1.fid and o1.vid = o2.vid and o1.fid = o2.fid and Near(1.05, o0.shape, o0.color, o0.material, o0.x1, o0.y1, o0.x2, o0.y2, o1.shape, o1.color, o1.material, o1.x1, o1.y1, o1.x2, o1.y2) = true and RightOf(o1.shape, o1.color, o1.material, o1.x1, o1.y1, o1.x2, o1.y2, o2.shape, o2.color, o2.material, o2.x1, o2.y1, o2.x2, o2.y2) = true and o0.oid <> o1.oid and o0.oid <> o2.oid and o1.oid <> o2.oid;")
            cur.execute("""
            CREATE TEMPORARY TABLE g0_windowed AS (
                SELECT vid, fid, o0_oid, o1_oid, o2_oid,
                lead(fid, 3 - 1, 0) OVER (PARTITION BY vid, o0_oid, o1_oid, o2_oid ORDER BY fid) as fid_offset
                FROM g0
            );
            """)
            cur.execute("""
                CREATE TEMPORARY TABLE g0_contiguous AS (
                SELECT t0.vid, t0.o0_oid, t0.o1_oid, t0.o2_oid, min(t0.fid_offset) AS fid
                FROM g0_windowed t0
                WHERE t0.fid_offset = t0.fid + (3 - 1)
                GROUP BY t0.vid, t0.o0_oid, t0.o1_oid, t0.o2_oid
            );
            """)
            cur.execute("""
                CREATE TEMPORARY TABLE g1 AS SELECT o0.vid as vid, o0.fid as fid, o0.oid as o0_oid, o1.oid as o1_oid FROM Obj_filtered as o1, Obj_filtered as o0 WHERE o0.vid = o1.vid and o0.fid = o1.fid and Cyan(o1.shape, o1.color, o1.material, o1.x1, o1.y1, o1.x2, o1.y2) = true and LeftQuadrant(o0.shape, o0.color, o0.material, o0.x1, o0.y1, o0.x2, o0.y2) = true and Metal(o0.shape, o0.color, o0.material, o0.x1, o0.y1, o0.x2, o0.y2) = true and o0.oid <> o1.oid;
            """)
            cur.execute("""
                CREATE TEMPORARY TABLE g1_filtered AS (
                SELECT t0.vid, t1.fid, t0.o0_oid, t0.o1_oid, t0.o2_oid
                FROM g0_contiguous t0, g1 t1
                WHERE t0.vid = t1.vid AND t0.o0_oid = t1.o0_oid AND t0.o1_oid = t1.o1_oid AND t0.fid < t1.fid
            );
            """)
            cur.execute("""
                CREATE TEMPORARY TABLE g1_windowed AS (
                    SELECT vid, fid, o0_oid, o1_oid, o2_oid,
                    lead(fid, 2 - 1, 0) OVER (PARTITION BY vid, o0_oid, o1_oid, o2_oid ORDER BY fid) as fid_offset
                    FROM g1_filtered
                );
            """)
            cur.execute("""
                CREATE TEMPORARY TABLE g1_contiguous AS (
                SELECT t0.vid, t0.o0_oid, t0.o1_oid, t0.o2_oid, min(t0.fid_offset) AS fid
                FROM g1_windowed t0
                WHERE t0.fid_offset = t0.fid + (2 - 1)
                GROUP BY t0.vid, t0.o0_oid, t0.o1_oid, t0.o2_oid
            );
            """)
            cur.execute("CREATE TEMPORARY TABLE g2 AS SELECT o0.vid as vid, o0.fid as fid, o0.oid as o0_oid, o1.oid as o1_oid, o2.oid as o2_oid FROM Obj_filtered as o1, Obj_filtered as o0, Obj_filtered as o2 WHERE o0.vid = o1.vid and o0.fid = o1.fid and o1.vid = o2.vid and o1.fid = o2.fid and LeftOf(o0.shape, o0.color, o0.material, o0.x1, o0.y1, o0.x2, o0.y2, o2.shape, o2.color, o2.material, o2.x1, o2.y1, o2.x2, o2.y2) = true and TopQuadrant(o1.shape, o1.color, o1.material, o1.x1, o1.y1, o1.x2, o1.y2) = true and o0.oid <> o1.oid and o0.oid <> o2.oid and o1.oid <> o2.oid")
            cur.execute("""
                CREATE TEMPORARY TABLE g2_filtered AS (
                SELECT t0.vid, t1.fid, t0.o0_oid, t0.o1_oid, t0.o2_oid
                FROM g1_contiguous t0, g2 t1
                WHERE t0.vid = t1.vid AND t0.o0_oid = t1.o0_oid AND t0.o1_oid = t1.o1_oid AND t0.o2_oid = t1.o2_oid AND t0.fid < t1.fid
            );
            """)
            cur.execute("""
                CREATE TEMPORARY TABLE g2_windowed AS (
                SELECT vid, fid, o0_oid, o1_oid, o2_oid,
                lead(fid, 4 - 1, 0) OVER (PARTITION BY vid, o0_oid, o1_oid, o2_oid ORDER BY fid) as fid_offset
                FROM g2_filtered
            );
            """)
            cur.execute("""
                    CREATE TEMPORARY TABLE g2_contiguous AS (
                    SELECT t0.vid, t0.o0_oid, t0.o1_oid, t0.o2_oid, min(t0.fid_offset) AS fid
                    FROM g2_windowed t0
                    WHERE t0.fid_offset = t0.fid + (4 - 1)
                    GROUP BY t0.vid, t0.o0_oid, t0.o1_oid, t0.o2_oid
                );
            """)
            cur.execute("SELECT DISTINCT vid FROM g2_contiguous")
            output_vids = cur.fetchall()
    return "job: {}, time: {}".format(job_id, time.time() - _time)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--multithread', type=int, default=1)
    ap.add_argument('--port', type=int, default=5432)

    args = ap.parse_args()
    multithread = args.multithread
    port = args.port

    _start = time.time()
    dsn = "dbname=myinner_db user=enhaoz host=localhost port={}".format(port)
    print("dsn", dsn)
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            inputs_table_name = "Obj_clevrer_temp"
            cur.execute("""
            CREATE TABLE {} AS
            SELECT *
            FROM Obj_clevrer;
            """.format(inputs_table_name))
            cur.execute("CREATE INDEX IF NOT EXISTS idx_{t} ON {t} (vid);".format(t=inputs_table_name))
    print("Time to create inputs table: {}".format(time.time() - _start))

    _start = time.time()
    with ThreadPoolExecutor(max_workers=multithread) as executor:
        for result in executor.map(postgres_execute_no_caching, list(range(20))):
            print(result)
    print("Time to run queries: {}".format(time.time() - _start))
