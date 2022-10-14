from quivr.methods.base_method import BaseMethod
from quivr.utils import rewrite_program_postgres, str_to_program_postgres
from quivr.query_graph import QueryGraph
import copy
import numpy as np
import time
from scipy import stats
import itertools
from lru import LRU
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import resource
import random
import quivr.dsl as dsl
from functools import cmp_to_key
import psycopg
import uuid

def using(point=""):
    usage=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return '''%s: mem=%s MB
           '''%(point, usage/1024.0 )

def compare_with_ties(x, y):
    if x < y:
        return -1
    elif x > y:
        return 1
    else:
        return random.randint(0, 1) * 2 - 1

# random.seed(10)
random.seed(time.time())
class VOCALPostgres(BaseMethod):

    def __init__(self, dataset_name, inputs, labels, predicate_list, max_npred, max_depth, max_duration, beam_width, pool_size, k, samples_per_iter, budget, multithread, strategy, max_vars, port):
        self.dataset_name = dataset_name
        self.inputs = inputs
        self.labels = labels
        self.predicate_list = predicate_dict
        self.max_npred = max_npred
        self.max_depth = max_depth
        self.beam_width = beam_width
        self.pool_size = pool_size
        self.k = k
        self.samples_per_iter = samples_per_iter
        self.budget = budget
        self.max_duration = max_duration
        self.multithread = multithread
        self.strategy = strategy
        self.max_vars = max_vars

        self.query_expansion_time = 0
        self.segment_selection_time = 0
        self.retain_top_k_queries_time = 0
        self.answers = []

        if self.multithread > 1:
            self.m = multiprocessing.Manager()
            self.lock = self.m.Lock()
        elif self.multithread == 1:
            self.lock = None
        else:
            raise ValueError("multithread must be 1 or greater")

        self.dsn = "dbname=myinner_db user=enhaoz host=localhost port={}".format(port)
        print("dsn", self.dsn)
        self.inputs_table_name = "Obj_trajectories_{}".format(uuid.uuid4().hex)
        with psycopg.connect(self.dsn) as conn:
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
                """.format(self.inputs_table_name))

                # Load inputs into temporary table
                csv_data = []
                for vid, pair in enumerate(self.inputs):
                    t1 = pair[0]
                    t2 = pair[1]
                    assert(len(t1) == len(t2))
                    for fid, (bbox1, bbox2) in enumerate(zip(t1, t2)):
                        csv_data.append((0, vid, fid, "cube", "red", "metal", bbox1[0], bbox1[1], bbox1[2], bbox1[3]))
                        csv_data.append((1, vid, fid, "cube", "red", "metal", bbox2[0], bbox2[1], bbox2[2], bbox2[3]))
                with cur.copy("COPY {} FROM STDIN".format(self.inputs_table_name)) as cur_copy:
                    for row in csv_data:
                        cur_copy.write_row(row)

                # Create predicate functions (if not exists)
                for predicate in self.predicate_list:
                    # TODO: update args to include all scene graph information (e.g., attributes)
                    args = ", ".join(["double precision, double precision, double precision, double precision"] * predicate["nargs"])
                    if predicate["parameters"]:
                        args = "double precision, " + args
                    cur.execute("CREATE OR REPLACE FUNCTION {name}({args}) RETURNS boolean AS '/mmfs1/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/postgres/functors', '{name}' LANGUAGE C STRICT;".format(name=predicate["name"], args=args))
                conn.commit()

    def run(self, init_labeled_index):
        _start_total_time = time.time()
        self.init_nlabels = len(init_labeled_index)
        self.labeled_index = init_labeled_index
        self.memoize_all_inputs = [LRU(10000) for _ in range(len(self.inputs))] # For each (input, label) pair, memoize the results of all sub-queries encountered during synthesis.

        # Initialize candidate queries: [query_graph, score]
        self.candidate_queries = []
        pred_instances = []
        for pred in self.predicate_list:
            if pred["parameters"]:
                for param in pred["parameters"]:
                    pred_instances.append({"name": pred["name"], "parameter": param, "nargs": pred["nargs"]})
            else:
                pred_instances.append({"name": pred["name"], "parameter": None, "nargs": pred["nargs"]})
        for pred_instance in pred_instances:
            nvars = pred_instance["nargs"]
            if nvars > self.max_vars:
                raise ValueError("The predicate has more variables than the number of variables in the query.")
            variables = ["o{}".format(i) for i in range(nvars)]
            query_graph = QueryGraph(self.max_npred, self.max_depth, self.max_vars)
            query_graph.program = [
                {
                    "scene_graph": [
                        {
                            "predicate": pred_instance["name"],
                            "parameter": pred_instance["parameter"],
                            "variables": list(variables)
                        }
                    ],
                    "duration_constraint": 1
                }
            ]
            query_graph.npred = 1
            query_graph.depth = 1
            score = self.compute_query_score_postgres(query_graph.program)
            print("initialization", rewrite_program_postgres(query_graph.program), score)
            self.candidate_queries.append([query_graph, score])
            self.answers.append([query_graph, score])

        _start_segmnet_selection_time = time.time()
        # video_segment_ids = self.pick_next_segment()
        video_segment_ids = self.pick_next_segment_model_picker_postgres()
        print("pick next segments", video_segment_ids)
        self.labeled_index += video_segment_ids
        print("# labeled segments", len(self.labeled_index))
        print("# positive: {}, # negative: {}".format(sum(self.labels[self.labeled_index]), len(self.labels[self.labeled_index]) - sum(self.labels[self.labeled_index])))
        # assert labeled_index does not contain duplicates
        assert(len(self.labeled_index) == len(set(self.labeled_index)))
        if self.multithread > 1:
            updated_scores = []
            with ThreadPoolExecutor(max_workers=self.multithread) as executor:
                for result in executor.map(self.compute_query_score_postgres, [query.program for query, _ in self.candidate_queries]):
                    updated_scores.append(result)
            for i in range(len(self.candidate_queries)):
                self.candidate_queries[i][1] = updated_scores[i]
        else:
            for i in range(len(self.candidate_queries)):
                self.candidate_queries[i][1] = self.compute_query_score_postgres(self.candidate_queries[i][0].program)
        self.segment_selection_time += time.time() - _start_segmnet_selection_time

        # Sample beam_width queries
        if self.strategy == "sampling":
            self.candidate_queries = np.asarray(self.candidate_queries, dtype=object)
            weight = self.candidate_queries[:, 1].astype(np.float)
            weight = weight / np.sum(weight)
            candidate_idx = np.random.choice(np.arange(self.candidate_queries.shape[0]), size=min(self.beam_width, self.candidate_queries.shape[0]), replace=False, p=weight)
            print("candidate_idx", candidate_idx)
            self.candidate_queries = self.candidate_queries[candidate_idx].tolist()
            print("beam_width queries", [(rewrite_program_postgres(query.program), score) for query, score in self.candidate_queries])
        elif self.strategy == "topk":
            # Sorted with randomized ties
            self.candidate_queries = sorted(self.candidate_queries, key=lambda x: cmp_to_key(compare_with_ties)(x[1]), reverse=True)[:self.beam_width]
            print("beam_width queries", [(rewrite_program_postgres(query.program), score) for query, score in self.candidate_queries])
        elif self.strategy == "topk_including_ties":
            self.candidate_queries = sorted(self.candidate_queries, key=lambda x: x[1], reverse=True)
            utility_bound = self.candidate_queries[:self.beam_width][-1][1]
            self.candidate_queries = [e for e in self.candidate_queries if e[1] >= utility_bound]
            print("beam_width {} queries".format(len(self.candidate_queries)), [(rewrite_program_postgres(query.program), score) for query, score in self.candidate_queries])

        while len(self.candidate_queries):
            # Generate more queries
            _start_query_expansion_time = time.time()
            new_candidate_queries = []

            if self.multithread > 1:
                with ThreadPoolExecutor(max_workers=self.multithread) as executor:
                    for result in executor.map(self.expand_query_and_compute_score, self.candidate_queries):
                        new_candidate_queries.extend(result)
            else:
                for candidate_query in self.candidate_queries:
                    new_candidate_queries.extend(self.expand_query_and_compute_score(candidate_query))
            self.answers.extend(new_candidate_queries)
            self.query_expansion_time += time.time() - _start_query_expansion_time

            # Remove duplicates for new_candidate_queries
            new_candidate_queries_removing_duplicates = []
            print("[new_candidate_queries] before removing duplicates:", len(new_candidate_queries))
            for query, score in new_candidate_queries:
                print(rewrite_program_postgres(query.program), score)
            signatures = set()
            for query, score in new_candidate_queries:
                signature = rewrite_program_postgres(query.program)
                if signature not in signatures:
                    query.program = str_to_program_postgres(signature)
                    new_candidate_queries_removing_duplicates.append([query, score])
                    signatures.add(signature)
            print("[new_candidate_queries] after removing duplicates:", len(new_candidate_queries_removing_duplicates))
            for query, score in new_candidate_queries_removing_duplicates:
                print(rewrite_program_postgres(query.program), score)
            self.candidate_queries = new_candidate_queries_removing_duplicates

            # Select new video segments to label
            if len(self.labeled_index) < self.budget and len(self.candidate_queries):
                _start_segmnet_selection_time = time.time()
                # video_segment_ids = self.pick_next_segment()
                video_segment_ids = self.pick_next_segment_model_picker_postgres()
                print("pick next segments", video_segment_ids)
                self.labeled_index += video_segment_ids
                print("# labeled segments", len(self.labeled_index))
                print("# positive: {}, # negative: {}".format(sum(self.labels[self.labeled_index]), len(self.labels[self.labeled_index]) - sum(self.labels[self.labeled_index])))
                # assert labeled_index does not contain duplicates
                assert(len(self.labeled_index) == len(set(self.labeled_index)))
                self.segment_selection_time += time.time() - _start_segmnet_selection_time

            _start_segmnet_selection_time = time.time()
            if self.multithread > 1:
                updated_scores = []
                with ThreadPoolExecutor(max_workers=self.multithread) as executor:
                    for result in executor.map(self.compute_query_score_postgres, [query.program for query, _ in self.candidate_queries]):
                        updated_scores.append(result)
                for i in range(len(self.candidate_queries)):
                    self.candidate_queries[i][1] = updated_scores[i]
            else:
                for i in range(len(self.candidate_queries)):
                    self.candidate_queries[i][1] = self.compute_query_score_postgres(self.candidate_queries[i][0].program)
            self.segment_selection_time += time.time() - _start_segmnet_selection_time

            # Sample beam_width queries
            if len(self.candidate_queries):
                if self.strategy == "sampling":
                    self.candidate_queries = np.asarray(self.candidate_queries, dtype=object)
                    weight = self.candidate_queries[:, 1].astype(np.float)
                    weight = weight / np.sum(weight)
                    candidate_idx = np.random.choice(np.arange(self.candidate_queries.shape[0]), size=min(self.beam_width, self.candidate_queries.shape[0]), replace=False, p=weight)
                    print("candidate_idx", candidate_idx)
                    self.candidate_queries = self.candidate_queries[candidate_idx].tolist()
                    print("beam_width queries", [(rewrite_program_postgres(query.program), score) for query, score in self.candidate_queries])
                elif self.strategy == "topk":
                    # Sorted with randomized ties
                    self.candidate_queries = sorted(self.candidate_queries, key=lambda x: cmp_to_key(compare_with_ties)(x[1]), reverse=True)[:self.beam_width]
                    print("beam_width queries", [(rewrite_program_postgres(query.program), score) for query, score in self.candidate_queries])
                elif self.strategy == "topk_including_ties":
                    self.candidate_queries = sorted(self.candidate_queries, key=lambda x: x[1], reverse=True)
                    utility_bound = self.candidate_queries[:self.beam_width][-1][1]
                    self.candidate_queries = [e for e in self.candidate_queries if e[1] >= utility_bound]
                    print("beam_width {} queries".format(len(self.candidate_queries)), [(rewrite_program_postgres(query.program), score) for query, score in self.candidate_queries])
            else:
                self.candidate_queries = []
            # Remove duplicates for self.answers
            answers_removing_duplicates = []
            print("[self.answers] before removing duplicates:", len(self.answers))
            for query, score in self.answers:
                print(rewrite_program_postgres(query.program), score)
            signatures = set()
            for query, score in self.answers:
                signature = rewrite_program_postgres(query.program)
                if signature not in signatures:
                    query.program = str_to_program_postgres(signature)
                    answers_removing_duplicates.append([query, score])
                    signatures.add(signature)
            print("[self.answers] after removing duplicates:", len(answers_removing_duplicates))
            for query, score in answers_removing_duplicates:
                print(rewrite_program_postgres(query.program), score)
            self.answers = answers_removing_duplicates

            # Retain the top k queries with the highest scores as answers
            _start_retain_top_k_queries_time = time.time()
            if self.multithread > 1:
                updated_scores = []
                with ThreadPoolExecutor(max_workers=self.multithread) as executor:
                    for result in executor.map(self.compute_query_score_postgres, [query.program for query, _ in self.answers]):
                        updated_scores.append(result)
                for i in range(len(self.answers)):
                    self.answers[i][1] = updated_scores[i]
            else:
                for i in range(len(self.answers)):
                    self.answers[i][1] = self.compute_query_score_postgres(self.answers[i][0].program)
            self.answers = sorted(self.answers, key=lambda x: x[1], reverse=True)
            self.answers = self.answers[:self.k]
            print("top k queries", [(rewrite_program_postgres(query.program), score) for query, score in self.answers])
            self.retain_top_k_queries_time += time.time() - _start_retain_top_k_queries_time

        # RETURN: the list.
        print("final_answers")
        for query_graph, score in self.answers:
            print("answer", rewrite_program_postgres(query_graph.program), score)

        total_time = time.time() - _start_total_time
        print("[Runtime] query expansion time: {}, segment selection time: {}, retain top k queries time: {}, total time: {}".format(self.query_expansion_time, self.segment_selection_time, self.retain_top_k_queries_time, total_time))
        print(using("profile"))

        # Drop input table
        with psycopg.connect(self.dsn) as conn:
            with conn.cursor() as cur:
                cur.execute("DROP TABLE {};".format(self.inputs_table_name))

        return self.answers, total_time

    def expand_query_and_compute_score(self, current_query):
        current_query_graph, _ = current_query
        current_query = current_query_graph.program
        print("expand search space", rewrite_program_postgres(current_query))
        # all_children = current_query_graph.get_all_children_bu(self.predicate_list, self.max_duration)
        all_children = current_query_graph.get_all_children_unrestricted_postgres(self.predicate_list, self.max_duration)

        new_candidate_queries = []

        # Compute F1 score of each candidate query
        for child in all_children:
            score = self.compute_query_score_postgres(child.program)
            if score > 0:
                new_candidate_queries.append([child, score])
                print(rewrite_program_postgres(child.program), score)
        return new_candidate_queries