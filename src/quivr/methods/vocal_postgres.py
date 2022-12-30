from quivr.methods.base_method import BaseMethod
from quivr.utils import rewrite_program_postgres, str_to_program_postgres, complexity_cost
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
import psycopg2 as psycopg
import uuid

def using(point=""):
    usage=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return '''%s: mem=%s MB'''%(point, usage/1024.0 )

class VOCALPostgres(BaseMethod):

    def __init__(self, dataset_name, inputs, labels, predicate_list, max_npred, max_depth, max_duration, beam_width, pool_size, k, budget, multithread, strategy, max_vars, port, sampling_rate, lru_capacity, reg_lambda, n_init_pos, n_init_neg):
        self.dataset_name = dataset_name
        self.inputs = inputs
        self.labels = labels
        self.predicate_list = predicate_list
        self.max_npred = max_npred
        self.max_nontrivial = None
        self.max_trivial = None
        self.max_depth = max_depth
        self.beam_width = beam_width
        self.pool_size = pool_size
        self.k = k
        self.budget = budget
        self.max_duration = max_duration
        self.multithread = multithread
        self.strategy = strategy
        self.max_vars = max_vars
        self.lru_capacity = lru_capacity
        self.n_sampled_videos = 100
        self.reg_lambda = reg_lambda
        self.active_learning = self.pick_next_segment_model_picker_postgres
        self.get_query_score = self.compute_query_score_postgres
        self.n_init_pos = n_init_pos
        self.n_init_neg = n_init_neg
        self.duration_unit = 5

        if self.duration_unit == 1:
            samples_per_iter = [0] * (self.max_npred + (self.max_duration - 1) * self.max_depth)
        else:
            samples_per_iter = [0] * (self.max_npred + (self.max_duration // self.duration_unit) * self.max_depth)
        for i in range((self.budget - self.n_init_pos - self.n_init_neg)):
            # samples_per_iter[len(samples_per_iter) - 1 - i % len(samples_per_iter)] += 1 # Lazy
            # samples_per_iter[i % len(samples_per_iter)] += 1 # Eager
            samples_per_iter[len(samples_per_iter)//2+((i% len(samples_per_iter)+1)//2)*(-1)**(i% len(samples_per_iter))] += 1 # Iterate from the middle
        self.samples_per_iter = samples_per_iter

        print(samples_per_iter)

        if "scene_graph" in self.dataset_name:
            self.is_trajectory = False
        else:
            self.is_trajectory = True
        print("is_trajectory", self.is_trajectory)

        self.sampling_rate = sampling_rate

        self.best_query_after_each_iter = []
        self.iteration = 0
        self.query_expansion_time = 0
        self.segment_selection_time = 0
        self.retain_top_k_queries_time = 0
        self.answers = []
        self.n_queries_explored = 0
        self.n_prediction_count = 0
        _start = time.time()
        if self.multithread > 1:
            self.executor = ThreadPoolExecutor(max_workers=self.multithread)
            self.m = multiprocessing.Manager()
            self.lock = self.m.Lock()
        elif self.multithread == 1:
            self.lock = None
        else:
            raise ValueError("multithread must be 1 or greater")
        print("process pool init time:", time.time() - _start)

        self.output_log = []

        _start = time.time()
        self.dsn = "dbname=myinner_db user=enhaoz host=localhost port={}".format(port)
        print("dsn", self.dsn)
        with psycopg.connect(self.dsn) as conn:
            with conn.cursor() as cur:
                if not self.is_trajectory: # Queries complying with the scene graph model
                    self.inputs_table_name = "Obj_clevrer_{}".format(uuid.uuid4().hex)
                    cur.execute("""
                    CREATE TABLE {} AS
                    SELECT *
                    FROM Obj_clevrer
                    WHERE vid = ANY(%s);
                    """.format(self.inputs_table_name), [inputs.tolist()])
                else: # Queries complying with the trajectory model
                    if dataset_name.startswith("collision"):
                        self.inputs_table_name = "Obj_collision_{}".format(uuid.uuid4().hex)
                        cur.execute("""
                        CREATE TABLE {} AS
                        SELECT *
                        FROM Obj_collision
                        WHERE vid = ANY(%s);
                        """.format(self.inputs_table_name), [inputs.tolist()])
                    else:
                        self.inputs_table_name = "Obj_trajectories_{}".format(uuid.uuid4().hex)
                        cur.execute("""
                        CREATE TABLE {} AS
                        SELECT *
                        FROM Obj_trajectories
                        WHERE vid = ANY(%s);
                        """.format(self.inputs_table_name), [inputs.tolist()])

                cur.execute("CREATE INDEX IF NOT EXISTS idx_{t} ON {t} (vid);".format(t=self.inputs_table_name))
                # Create predicate functions (if not exists)
                for predicate in self.predicate_list:
                    args = ", ".join(["text, text, text, double precision, double precision, double precision, double precision"] * predicate["nargs"])
                    if predicate["parameters"]:
                        if isinstance(predicate["parameters"][0], str):
                            args = "text, " + args
                        else:
                            args = "double precision, " + args
                    cur.execute("""
                    CREATE OR REPLACE FUNCTION {name}({args}) RETURNS boolean AS
                    '/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/postgres/functors', '{name}'
                    LANGUAGE C STRICT;
                    """.format(name=predicate["name"], args=args))
                conn.commit()
        print("Time to create inputs table: {}".format(time.time() - _start))

    def compare_with_ties(self, x, y):
        if x < y:
            return -1
        elif x > y:
            return 1
        else:
            return random.randint(0, 1) * 2 - 1

    def run(self, init_labeled_index):
        self._start_total_time = time.time()
        self.init_nlabels = len(init_labeled_index)
        self.labeled_index = init_labeled_index

        if self.is_trajectory:
            if self.dataset_name.startswith("collision"):
                list_size = 12747
            else:
                list_size = 10080
        else:
            list_size = 10000

        if self.lru_capacity:
            self.memoize_scene_graph_all_inputs = [LRU(self.lru_capacity) for _ in range(list_size)]
            self.memoize_sequence_all_inputs = [LRU(self.lru_capacity) for _ in range(list_size)]
        else:
            self.memoize_scene_graph_all_inputs = [{} for _ in range(list_size)]
            self.memoize_sequence_all_inputs = [{} for _ in range(list_size)]
        self.candidate_queries = []

        self.main()

        # RETURN: the list.
        print("final_answers")
        self.output_log.append("[Final answers]")
        for query_graph, score in self.answers:
            print("answer", rewrite_program_postgres(query_graph.program), score)
            self.output_log.append((rewrite_program_postgres(query_graph.program), score))
        total_time = time.time() - self._start_total_time
        print("[Runtime] query expansion time: {}, segment selection time: {}, retain top k queries time: {}, total time: {}".format(self.query_expansion_time, self.segment_selection_time, self.retain_top_k_queries_time, total_time))
        self.output_log.append("[Total runtime] query expansion time: {}, segment selection time: {}, retain top k queries time: {}, total time: {}".format(self.query_expansion_time, self.segment_selection_time, self.retain_top_k_queries_time, total_time))
        print("n_queries_explored", self.n_queries_explored)
        self.output_log.append("[# queries explored] {}".format(self.n_queries_explored))
        print("n_prediction_count", self.n_prediction_count)
        self.output_log.append("[# predictions] {}".format(self.n_prediction_count))
        print(using("profile"))
        self.output_log.append(using("profile"))

        # Drop input table
        with psycopg.connect(self.dsn) as conn:
            with conn.cursor() as cur:
                cur.execute("DROP TABLE {};".format(self.inputs_table_name))

        return self.output_log

    def main(self):
        self.first_stage()

    def first_stage(self):
        while len(self.candidate_queries) or self.iteration == 0:
            print("[Step {}]".format(self.iteration))
            self.output_log.append("[Step {}]".format(self.iteration))
            print("n_queries_explored", self.n_queries_explored)
            print("n_prediction_count", self.n_prediction_count)
            # Generate more queries
            _start_query_expansion_time = time.time()
            new_candidate_queries = []
            if self.iteration == 0:
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
                    query_graph = QueryGraph(self.max_npred, self.max_depth, self.max_nontrivial, self.max_trivial, self.max_duration, self.max_vars, self.predicate_list, is_trajectory=self.is_trajectory)
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
                    score = self.get_query_score(query_graph.program)
                    print("initialization", rewrite_program_postgres(query_graph.program), score)
                    new_candidate_queries.append([query_graph, score])
                    self.candidate_queries = sorted(new_candidate_queries, key=lambda x: cmp_to_key(self.compare_with_ties)(x[1]), reverse=True)
                    self.answers.append([query_graph, score])
                self.n_prediction_count += len(self.labeled_index) * len(pred_instances)
            else:
                # Expand queries
                for candidate_query in self.candidate_queries:
                    candidate_query_graph, _ = candidate_query
                    print("expand search space", rewrite_program_postgres(candidate_query_graph.program))
                    all_children = candidate_query_graph.get_all_children_unrestricted_postgres()
                    new_candidate_queries.extend([[child, -1] for child in all_children])

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
                self.n_queries_explored += len(self.candidate_queries)

                # Compute scores
                self.n_prediction_count += len(self.labeled_index) * len(self.candidate_queries)
                candidate_queries_greater_than_zero = []
                if self.multithread > 1:
                    updated_scores = []
                    for result in self.executor.map(self.get_query_score, [query.program for query, _ in self.candidate_queries]):
                        updated_scores.append(result)
                    for i in range(len(self.candidate_queries)):
                        if updated_scores[i] > 0:
                            candidate_queries_greater_than_zero.append([self.candidate_queries[i][0], updated_scores[i]])
                else:
                    # Compute F1 score of each candidate query
                    for i in range(len(self.candidate_queries)):
                        score = self.get_query_score(self.candidate_queries[i][0].program)
                        if score > 0:
                            candidate_queries_greater_than_zero.append([self.candidate_queries[i][0], score])
                self.candidate_queries = candidate_queries_greater_than_zero
                self.answers.extend(self.candidate_queries)
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
            # Keep the top-k queries
            self.answers = sorted(self.answers, key=lambda x: cmp_to_key(self.compare_with_ties)(x[1]), reverse=True)
            best_score = self.answers[0][1]
            utility_bound = self.answers[:self.k][-1][1]
            self.answers = [e for e in self.answers if e[1] >= utility_bound]
            self.query_expansion_time += time.time() - _start_query_expansion_time

            # Select new video segments to label
            if len(self.labeled_index) < self.budget and len(self.candidate_queries):
                _start_segment_selection_time = time.time()
                for _ in range(self.samples_per_iter[self.iteration]):
                    _start_segment_selection_time_per_iter = time.time()
                    self.candidate_queries = sorted(self.candidate_queries, key=lambda x: cmp_to_key(self.compare_with_ties)(x[1]), reverse=True)
                    new_labeled_index = self.active_learning()
                    if len(new_labeled_index) > self.budget - len(self.labeled_index):
                        new_labeled_index = new_labeled_index[:self.budget - len(self.labeled_index)]
                    print("pick next segments", new_labeled_index)
                    self.labeled_index += new_labeled_index
                    print("# labeled segments", len(self.labeled_index))
                    print("# positive: {}, # negative: {}".format(sum(self.labels[self.labeled_index]), len(self.labels[self.labeled_index]) - sum(self.labels[self.labeled_index])))
                    # assert labeled_index does not contain duplicates
                    # assert(len(self.labeled_index) == len(set(self.labeled_index)))
                    # Update scores
                    self.n_prediction_count += len(new_labeled_index) * len(self.candidate_queries)
                    if self.multithread > 1:
                        updated_scores = []
                        for result in self.executor.map(self.get_query_score, [query.program for query, _ in self.candidate_queries]):
                            updated_scores.append(result)
                        for i in range(len(self.candidate_queries)):
                            self.candidate_queries[i][1] = updated_scores[i]
                    else:
                        for i in range(len(self.candidate_queries)):
                            self.candidate_queries[i][1] = self.get_query_score(self.candidate_queries[i][0].program)
                    print("test segment_selection_time_per_iter time:", time.time() - _start_segment_selection_time_per_iter)
                self.segment_selection_time += time.time() - _start_segment_selection_time

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
                    self.candidate_queries = sorted(self.candidate_queries, key=lambda x: cmp_to_key(self.compare_with_ties)(x[1]), reverse=True)[:self.beam_width]
                    print("beam_width queries", [(rewrite_program_postgres(query.program), score) for query, score in self.candidate_queries])
                elif self.strategy == "topk_including_ties":
                    self.candidate_queries = sorted(self.candidate_queries, key=lambda x: x[1], reverse=True)
                    utility_bound = self.candidate_queries[:self.beam_width][-1][1]
                    self.candidate_queries = [e for e in self.candidate_queries if e[1] >= utility_bound]
                    print("beam_width {} queries".format(len(self.candidate_queries)), [(rewrite_program_postgres(query.program), score) for query, score in self.candidate_queries])
            else:
                self.candidate_queries = []

            # Retain the top k queries with the highest scores as answers
            _start_retain_top_k_queries_time = time.time()
            self.n_prediction_count += len(self.answers) * len(self.labeled_index)
            if self.multithread > 1:
                updated_scores = []
                for result in self.executor.map(self.get_query_score, [query.program for query, _ in self.answers]):
                    updated_scores.append(result)
                for i in range(len(self.answers)):
                    self.answers[i][1] = updated_scores[i]
            else:
                for i in range(len(self.answers)):
                    self.answers[i][1] = self.get_query_score(self.answers[i][0].program)
            self.answers = sorted(self.answers, key=lambda x: cmp_to_key(self.compare_with_ties)(x[1]), reverse=True)
            best_score = self.answers[0][1]
            utility_bound = self.answers[:self.k][-1][1]
            self.answers = [e for e in self.answers if e[1] >= utility_bound]
            # self.answers = sorted(self.answers, key=lambda x: x[1], reverse=True)
            # self.answers = self.answers[:self.k]
            print("top k queries", [(rewrite_program_postgres(query.program), score) for query, score in self.answers])
            self.retain_top_k_queries_time += time.time() - _start_retain_top_k_queries_time
            self.best_query_after_each_iter = [e for e in self.answers if e[1] >= best_score]
            print("best query after each iter", [(rewrite_program_postgres(query.program), score) for query, score in self.best_query_after_each_iter])
            print(using("profile"))
            for query, score in self.best_query_after_each_iter:
                self.output_log.append((rewrite_program_postgres(query.program), score))
            self.output_log.append("[Runtime so far] {}".format(time.time() - self._start_total_time))
            self.iteration += 1


    def expand_query_and_compute_score(self, current_query):
        current_query_graph, _ = current_query
        current_query = current_query_graph.program
        print("expand search space", rewrite_program_postgres(current_query))
        all_children = current_query_graph.get_all_children_unrestricted_postgres()

        new_candidate_queries = []

        # Compute F1 score of each candidate query
        for child in all_children:
            score = self.get_query_score(child.program)
            if score > 0:
                new_candidate_queries.append([child, score])
                print(rewrite_program_postgres(child.program), score)
        return new_candidate_queries