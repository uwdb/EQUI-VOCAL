from src.methods.base_method import BaseMethod
from src.utils import program_to_dsl, dsl_to_program
from src.query_graph import QueryGraph
import numpy as np
import time
from lru import LRU
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
import resource
import random
from functools import cmp_to_key
import psycopg2 as psycopg
from psycopg2 import pool
import uuid
import threading
import copy
import math

def using(point=""):
    usage=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return '''%s: mem=%s MB'''%(point, usage/1024.0 )

class VOCALPostgres(BaseMethod):
    #ARMADILLO
    # def __init__(self, dataset_name, inputs, labels, predicate_list, max_npred, max_depth, max_duration, beam_width, pool_size, n_sampled_videos, k, budget, multithread, strategy, max_vars, port, sampling_rate, lru_capacity, reg_lambda, n_init_pos, n_init_neg, test_inputs=None, test_labels=None, iteration = 0, candidate_queries = None):
    #     print("hi")
    #     self.__init__(self, dataset_name, inputs, labels, predicate_list, max_npred, max_depth, max_duration, beam_width, pool_size, n_sampled_videos, k, budget, multithread, strategy, max_vars, port, sampling_rate, lru_capacity, reg_lambda, n_init_pos, n_init_neg, test_inputs=None, test_labels=None)
    #     self.iteration = iteration 
    #     self.candidate_queries = candidate_queries

    def __init__(self, dataset_name, inputs, labels, predicate_list, max_npred, max_depth, max_duration, beam_width, pool_size, n_sampled_videos, k, budget, multithread, strategy, max_vars, port, sampling_rate, lru_capacity, reg_lambda, n_init_pos, n_init_neg, test_inputs=None, test_labels=None, iteration=0, seed_queries=None):
        self.dataset_name = dataset_name
        self.inputs = inputs
        self.labels = labels
        self.test_inputs = test_inputs
        self.test_labels = test_labels
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
        self.num_threads = multithread
        self.strategy = strategy
        self.max_vars = max_vars
        self.lru_capacity = lru_capacity
        self.n_sampled_videos = n_sampled_videos
        self.reg_lambda = reg_lambda
        self.active_learning = self.pick_next_segment_model_picker_postgres
        self.get_query_score = self.compute_query_score_postgres
        self.n_init_pos = n_init_pos
        self.n_init_neg = n_init_neg
        if "scene_graph" in self.dataset_name:
            self.is_trajectory = False
        else:
            self.is_trajectory = True
        print("is_trajectory", self.is_trajectory)

        #ARMADILLO
        self.candidate_queries = []
        self.is_seeded = False
        if(seed_queries is not None):
            self.is_seeded = True
            print("Seed queries passed")
            for sq in seed_queries:
                prog = sq[0]
                seed_query_graph = QueryGraph(self.dataset_name, self.max_npred, self.max_depth, self.max_nontrivial, self.max_trivial, self.max_duration, self.max_vars, self.predicate_list, depth = len(prog), npred =sum([len(prog[i]['scene_graph']) for i in range(len(prog))]) , is_trajectory=self.is_trajectory)
                seed_query_graph.program = prog             
                self.candidate_queries.append((seed_query_graph, sq[1]))
                print("candidate queries:", self.candidate_queries)
        if self.dataset_name.startswith("user_study"):
            self.duration_unit = 25
        else:
            self.duration_unit = 5



        self.label_count_per_iter = 0 # Used for live demo task: the number of labels collected in the current iteration so far
        # self.filtered_index = [] # Create an empty list to store the indices of the videos that have been filtered out during the active learning process
        
        # how many steps in 
        num_pred = 0
        num_duration = 0
        if seed_queries is not None:
            num_pred = math.inf
            for seed_query in seed_queries:
                q = seed_query[0] #dsl_to_program(seed_query[0])
                np = 0
                for scene_graph in q:
                    print(scene_graph["duration_constraint"])
                    num_duration += round(scene_graph["duration_constraint"])
                    np += len(scene_graph["scene_graph"])
                num_pred = min(np, num_pred)
        if self.duration_unit == 1:
            samples_per_iter = [0] * ((self.max_npred - int(num_pred) )+ (self.max_duration - 1) * self.max_depth)
        else:
            samples_per_iter = [0] * ((self.max_npred - int(num_pred) ) + (self.max_duration // self.duration_unit) * self.max_depth)
        for i in range((self.budget - self.n_init_pos - self.n_init_neg)):
            samples_per_iter[len(samples_per_iter)//2+((i% len(samples_per_iter)+1)//2)*(-1)**(i% len(samples_per_iter))] += 1 # Iterate from the middle
        self.samples_per_iter = samples_per_iter



        self.rewrite_variables = not self.is_trajectory

        self.sampling_rate = sampling_rate

        self.best_query_after_each_iter = []
        self.iteration = iteration #0
        self.query_expansion_time = 0
        self.segment_selection_time = 0
        self.retain_top_k_queries_time = 0
        self.answers = []
        self.n_queries_explored = 0
        self.n_prediction_count = 0
        _start = time.time()

        if self.num_threads > 1:
            self.m = multiprocessing.Manager()
            self.lock = self.m.Lock()
        elif self.num_threads == 1:
            self.lock = None
        else:
            raise ValueError("num_threads must be 1 or greater")
        # Initialize the pools
        self.dsn = "dbname=myinner_db user=mganti host=127.0.0.1 port={}".format(port)
        # TODO: set the isolation level to read committed?
        # self.connections = [psycopg.connect(self.dsn) for _ in range(self.num_threads)]
        self.connections = psycopg.pool.ThreadedConnectionPool(1, self.num_threads, self.dsn)
        self.executor = ThreadPoolExecutor(max_workers=self.num_threads)
        print("process pool init time:", time.time() - _start)

        self.output_log = []

        _start = time.time()

        print("dsn", self.dsn)
        conn = self.connections.getconn()
        with conn.cursor() as cur:
            filtered_vids = inputs.tolist()
            if test_inputs is not None:
                filtered_vids += test_inputs.tolist()
            if not self.is_trajectory: # Queries complying with the scene graph model
                self.inputs_table_name = "Obj_clevrer_{}".format(uuid.uuid4().hex)
                cur.execute("""
                CREATE TABLE {} AS
                SELECT *
                FROM Obj_clevrer
                WHERE vid = ANY(%s);
                """.format(self.inputs_table_name), [filtered_vids])
            else: # Queries complying with the trajectory model
                if dataset_name.startswith("collision"):
                    self.inputs_table_name = "Obj_collision_{}".format(uuid.uuid4().hex)
                    cur.execute("""
                    CREATE TABLE {} AS
                    SELECT *
                    FROM Obj_collision
                    WHERE vid = ANY(%s);
                    """.format(self.inputs_table_name), [filtered_vids])
                elif dataset_name == "warsaw":
                    self.inputs_table_name = "Obj_warsaw_{}".format(uuid.uuid4().hex)
                    cur.execute("""
                    CREATE TABLE {} AS
                    SELECT *
                    FROM Obj_warsaw
                    WHERE vid = ANY(%s);
                    """.format(self.inputs_table_name), [filtered_vids])
                else:
                    self.inputs_table_name = "Obj_trajectories_{}".format(uuid.uuid4().hex)
                    cur.execute("""
                    CREATE TABLE {} AS
                    SELECT *
                    FROM Obj_trajectories
                    WHERE vid = ANY(%s);
                    """.format(self.inputs_table_name), [filtered_vids])

            cur.execute("CREATE INDEX IF NOT EXISTS idx_{t} ON {t} (vid);".format(t=self.inputs_table_name))
            conn.commit()
        self.connections.putconn(conn)
        print("Time to create inputs table: {}".format(time.time() - _start))
        print(self.iteration)
        print("candidate queries: {}".format(self.candidate_queries))

    @staticmethod
    def chunk_list(lst, n):
        """Split a list into n equally-sized chunks"""
        k, m = divmod(len(lst), n)
        return [lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]

    def compare_with_ties(self, x, y):
        if x < y:
            return -1
        elif x > y:
            return 1
        else:
            return random.randint(0, 1) * 2 - 1

    def run_init(self, init_labeled_index, user_labels=None):
        self._start_total_time = time.time()
        self.init_nlabels = len(init_labeled_index)
        # Deep copy
        self.labeled_index = copy.deepcopy(init_labeled_index)

        if user_labels is not None:
            self.labels[self.labeled_index] = user_labels

        if self.is_trajectory:
            if self.dataset_name.startswith("collision"):
                list_size = 12747
            elif self.dataset_name == "warsaw":
                list_size = 72159
            else:
                list_size = 10080
        else:
            list_size = 10000

        if self.lru_capacity:
            self.memo = [LRU(self.lru_capacity) for _ in range(list_size)]
        else:
            self.memo = [{} for _ in range(list_size)]
        #self.candidate_queries = []

    def run(self, init_labeled_index):
        self.run_init(init_labeled_index)

        self.main()

        # RETURN: the list.
        print("[Final answers]")
        self.output_log.append("[Final answers]")
        for query_graph, score in self.answers:
            print("answer", program_to_dsl(query_graph.program, self.rewrite_variables), score)
            self.output_log.append((program_to_dsl(query_graph.program, self.rewrite_variables), score))
        total_time = time.time() - self._start_total_time
        print("[Runtime] query expansion time: {}, segment selection time: {}, retain top k queries time: {}, total time: {}".format(self.query_expansion_time, self.segment_selection_time, self.retain_top_k_queries_time, total_time))
        self.output_log.append("[Total runtime]")
        self.output_log.append("query expansion time: {}, segment selection time: {}, retain top k queries time: {}, total time: {}".format(self.query_expansion_time, self.segment_selection_time, self.retain_top_k_queries_time, total_time))
        print("n_queries_explored", self.n_queries_explored)
        self.output_log.append("[# queries explored] {}".format(self.n_queries_explored))
        print("n_prediction_count", self.n_prediction_count)
        self.output_log.append("[# predictions] {}".format(self.n_prediction_count))
        print(using("profile"))
        self.output_log.append(using("profile"))

        # Drop input table
        conn = self.connections.getconn()
        with conn.cursor() as cur:
            cur.execute("DROP TABLE {};".format(self.inputs_table_name))
            conn.commit()
        self.connections.putconn(conn)

        # Close the connections and the executor
        self.connections.closeall()
        self.executor.shutdown()

        return self.output_log

    def main(self):
        while len(self.candidate_queries) or self.iteration == 0:
            print("[Step {}]".format(self.iteration))
            self.output_log.append("[Step {}]".format(self.iteration))
            print("n_queries_explored", self.n_queries_explored)
            print("n_prediction_count", self.n_prediction_count)
            # Generate more queries
            _start_query_expansion_time = time.time()
            new_candidate_queries = []
            if self.iteration == 0:
                iteration0_time = time.time()
                #ARMADILLO
                if self.is_seeded:
                    scores = []
                    for result in self.executor.map(self.get_query_score, [query.program for query, _ in self.candidate_queries]):
                        scores.append(result)
                    print("Getting scores, time = ", time.time() - iteration0_time)
                    iteration0_time1 = time.time()
                    for i in range(len(self.candidate_queries)):
                        self.candidate_queries[i] = (self.candidate_queries[i][0], scores[i])
                    #self.candidate_queries = [q for q in self.candidate_queries if q[1] in sorted(scores)[-1*self.beam_width:]]
                    self.candidate_queries = sorted(self.candidate_queries, key=lambda x: cmp_to_key(self.compare_with_ties)(x[1]), reverse=True)[:self.beam_width]
                    #optionally, truncate to beam width
                    for i in range(len(self.candidate_queries)):
                        print("seed query filtering", program_to_dsl(self.candidate_queries[i][0].program, self.rewrite_variables), self.candidate_queries[i][1])

                    self.iteration += 1
                    print("filtering time = ", time.time() - iteration0_time1)
                    print("[INITIALIZING SEED QUERIES]: time = ", time.time() - iteration0_time, "; ", self.candidate_queries)
                    continue
                print("initializing regular queries")
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
                    # Special case: For warsaw dataset, we also consider the case where the single predicate is applied to the second object.
                    if nvars == 1 and self.dataset_name == "warsaw":
                        variable_lists = [["o0"], ["o1"]]
                    else:
                        variable_lists = [["o{}".format(i) for i in range(nvars)]]
                    for variables in variable_lists:
                        query_graph = QueryGraph(self.dataset_name, self.max_npred, self.max_depth, self.max_nontrivial, self.max_trivial, self.max_duration, self.max_vars, self.predicate_list, is_trajectory=self.is_trajectory)
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
                        new_candidate_queries.append([query_graph, -1])
                # Assign queries to the threads
                updated_scores = []
                for result in self.executor.map(self.get_query_score, [query.program for query, _ in new_candidate_queries]):
                    updated_scores.append(result)
                for i in range(len(new_candidate_queries)):
                    new_candidate_queries[i][1] = updated_scores[i]
                    print("initialization", program_to_dsl(new_candidate_queries[i][0].program, self.rewrite_variables), updated_scores[i])
                self.candidate_queries = new_candidate_queries
                self.answers.extend(self.candidate_queries)
                self.n_prediction_count += len(self.labeled_index) * len(self.candidate_queries)

                print("Total first iteration time: ", time.time() - iteration0_time)
            else:
                # Expand queries
                for candidate_query in self.candidate_queries:
                    candidate_query_graph, _ = candidate_query
                    print("expand search space", program_to_dsl(candidate_query_graph.program, self.rewrite_variables))
                    all_children = candidate_query_graph.get_all_children_unrestricted_postgres()
                    new_candidate_queries.extend([[child, -1] for child in all_children])

                # Remove duplicates for new_candidate_queries
                new_candidate_queries_removing_duplicates = []
                print("[new_candidate_queries] before removing duplicates:", len(new_candidate_queries))
                for query, score in new_candidate_queries:
                    print(program_to_dsl(query.program, self.rewrite_variables), score)
                signatures = set()
                for query, score in new_candidate_queries:
                    signature = program_to_dsl(query.program, self.rewrite_variables)
                    if signature not in signatures:
                        query.program = dsl_to_program(signature)
                        new_candidate_queries_removing_duplicates.append([query, score])
                        signatures.add(signature)
                print("[new_candidate_queries] after removing duplicates:", len(new_candidate_queries_removing_duplicates))
                for query, score in new_candidate_queries_removing_duplicates:
                    print(program_to_dsl(query.program, self.rewrite_variables), score)
                self.candidate_queries = new_candidate_queries_removing_duplicates
                self.n_queries_explored += len(self.candidate_queries)

                # Compute scores
                self.n_prediction_count += len(self.labeled_index) * len(self.candidate_queries)
                candidate_queries_greater_than_zero = []
                updated_scores = []
                for result in self.executor.map(self.get_query_score, [query.program for query, _ in self.candidate_queries]):
                    updated_scores.append(result)
                for i in range(len(self.candidate_queries)):
                    if updated_scores[i] > 0:
                        candidate_queries_greater_than_zero.append([self.candidate_queries[i][0], updated_scores[i]])
                self.candidate_queries = candidate_queries_greater_than_zero
                self.answers.extend(self.candidate_queries)
            # Remove duplicates for self.answers
            answers_removing_duplicates = []
            print("[self.answers] before removing duplicates:", len(self.answers))
            for query, score in self.answers:
                print(program_to_dsl(query.program, self.rewrite_variables), score)
            signatures = set()
            for query, score in self.answers:
                signature = program_to_dsl(query.program, self.rewrite_variables)
                if signature not in signatures:
                    query.program = dsl_to_program(signature)
                    answers_removing_duplicates.append([query, score])
                    signatures.add(signature)
            print("[self.answers] after removing duplicates:", len(answers_removing_duplicates))
            for query, score in answers_removing_duplicates:
                print(program_to_dsl(query.program, self.rewrite_variables), score)
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
                    updated_scores = []
                    for result in self.executor.map(self.get_query_score, [query.program for query, _ in self.candidate_queries]):
                        updated_scores.append(result)
                    for i in range(len(self.candidate_queries)):
                        self.candidate_queries[i][1] = updated_scores[i]
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
                    print("beam_width queries", [(program_to_dsl(query.program, self.rewrite_variables), score) for query, score in self.candidate_queries])
                elif self.strategy == "topk":
                    # Sorted with randomized ties
                    self.candidate_queries = sorted(self.candidate_queries, key=lambda x: cmp_to_key(self.compare_with_ties)(x[1]), reverse=True)[:self.beam_width]
                    print("beam_width queries", [(program_to_dsl(query.program, self.rewrite_variables), score) for query, score in self.candidate_queries])
                elif self.strategy == "topk_including_ties":
                    self.candidate_queries = sorted(self.candidate_queries, key=lambda x: x[1], reverse=True)
                    utility_bound = self.candidate_queries[:self.beam_width][-1][1]
                    self.candidate_queries = [e for e in self.candidate_queries if e[1] >= utility_bound]
                    print("beam_width {} queries".format(len(self.candidate_queries)), [(program_to_dsl(query.program, self.rewrite_variables), score) for query, score in self.candidate_queries])
            else:
                self.candidate_queries = []

            # Retain the top k queries with the highest scores as answers
            _start_retain_top_k_queries_time = time.time()
            self.n_prediction_count += len(self.answers) * len(self.labeled_index)
            updated_scores = []
            for result in self.executor.map(self.get_query_score, [query.program for query, _ in self.answers]):
                updated_scores.append(result)
            for i in range(len(self.answers)):
                self.answers[i][1] = updated_scores[i]
            self.answers = sorted(self.answers, key=lambda x: cmp_to_key(self.compare_with_ties)(x[1]), reverse=True)
            best_score = self.answers[0][1]
            utility_bound = self.answers[:self.k][-1][1]
            self.answers = [e for e in self.answers if e[1] >= utility_bound]
            # self.answers = sorted(self.answers, key=lambda x: x[1], reverse=True)
            # self.answers = self.answers[:self.k]
            print("top k queries", [(program_to_dsl(query.program, self.rewrite_variables), score) for query, score in self.answers])
            self.retain_top_k_queries_time += time.time() - _start_retain_top_k_queries_time
            self.best_query_after_each_iter = [e for e in self.answers if e[1] >= best_score]
            print("best query after each iter", [(program_to_dsl(query.program, self.rewrite_variables), score) for query, score in self.best_query_after_each_iter])
            print(using("profile"))
            for query, score in self.best_query_after_each_iter:
                self.output_log.append((program_to_dsl(query.program, self.rewrite_variables), score))
            self.output_log.append("[Runtime so far] {}".format(time.time() - self._start_total_time))
            self.iteration += 1

    def demo_main(self):
        # Precompute for the demo
        log = []
        while len(self.candidate_queries) or self.iteration == 0:
            log_dict = {}
            log_dict["state"] = "label_first"
            log_dict["iteration"] = self.iteration
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
                    # Special case: For warsaw dataset, we also consider the case where the single predicate is applied to the second object.
                    if nvars == 1 and self.dataset_name == "warsaw":
                        variable_lists = [["o0"], ["o1"]]
                    else:
                        variable_lists = [["o{}".format(i) for i in range(nvars)]]
                    for variables in variable_lists:
                        query_graph = QueryGraph(self.dataset_name, self.max_npred, self.max_depth, self.max_nontrivial, self.max_trivial, self.max_duration, self.max_vars, self.predicate_list, is_trajectory=self.is_trajectory)
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
                        new_candidate_queries.append([query_graph, -1])
                # Assign queries to the threads
                updated_scores = []
                for result in self.executor.map(self.get_query_score, [query.program for query, _ in new_candidate_queries]):
                    updated_scores.append(result)
                for i in range(len(new_candidate_queries)):
                    new_candidate_queries[i][1] = updated_scores[i]
                    print("initialization", program_to_dsl(new_candidate_queries[i][0].program, self.rewrite_variables), updated_scores[i])
                self.candidate_queries = new_candidate_queries
                self.answers.extend(self.candidate_queries)
                self.n_prediction_count += len(self.labeled_index) * len(self.candidate_queries)
            else:
                # Expand queries
                for candidate_query in self.candidate_queries:
                    candidate_query_graph, _ = candidate_query
                    print("expand search space", program_to_dsl(candidate_query_graph.program, self.rewrite_variables))
                    all_children = candidate_query_graph.get_all_children_unrestricted_postgres()
                    new_candidate_queries.extend([[child, -1] for child in all_children])

                # Remove duplicates for new_candidate_queries
                new_candidate_queries_removing_duplicates = []
                print("[new_candidate_queries] before removing duplicates:", len(new_candidate_queries))
                for query, score in new_candidate_queries:
                    print(program_to_dsl(query.program, self.rewrite_variables), score)
                signatures = set()
                for query, score in new_candidate_queries:
                    signature = program_to_dsl(query.program, self.rewrite_variables)
                    if signature not in signatures:
                        query.program = dsl_to_program(signature)
                        new_candidate_queries_removing_duplicates.append([query, score])
                        signatures.add(signature)
                print("[new_candidate_queries] after removing duplicates:", len(new_candidate_queries_removing_duplicates))
                for query, score in new_candidate_queries_removing_duplicates:
                    print(program_to_dsl(query.program, self.rewrite_variables), score)
                self.candidate_queries = new_candidate_queries_removing_duplicates
                self.n_queries_explored += len(self.candidate_queries)

                # Compute scores
                self.n_prediction_count += len(self.labeled_index) * len(self.candidate_queries)
                candidate_queries_greater_than_zero = []
                updated_scores = []
                for result in self.executor.map(self.get_query_score, [query.program for query, _ in self.candidate_queries]):
                    updated_scores.append(result)
                for i in range(len(self.candidate_queries)):
                    if updated_scores[i] > 0:
                        candidate_queries_greater_than_zero.append([self.candidate_queries[i][0], updated_scores[i]])
                self.candidate_queries = candidate_queries_greater_than_zero
                self.answers.extend(self.candidate_queries)
            if len(self.candidate_queries) == 0:
                # Terminating synthesis algorithm
                break
            # Remove duplicates for self.answers
            answers_removing_duplicates = []
            print("[self.answers] before removing duplicates:", len(self.answers))
            for query, score in self.answers:
                print(program_to_dsl(query.program, self.rewrite_variables), score)
            signatures = set()
            for query, score in self.answers:
                signature = program_to_dsl(query.program, self.rewrite_variables)
                if signature not in signatures:
                    query.program = dsl_to_program(signature)
                    answers_removing_duplicates.append([query, score])
                    signatures.add(signature)
            print("[self.answers] after removing duplicates:", len(answers_removing_duplicates))
            for query, score in answers_removing_duplicates:
                print(program_to_dsl(query.program, self.rewrite_variables), score)
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
                log_dict["selected_segments"] = []
                log_dict["selected_gt_labels"] = []
                for _ in range(self.samples_per_iter[self.iteration]):
                    _start_segment_selection_time_per_iter = time.time()
                    self.candidate_queries = sorted(self.candidate_queries, key=lambda x: cmp_to_key(self.compare_with_ties)(x[1]), reverse=True)
                    new_labeled_index = self.active_learning()
                    if len(new_labeled_index) > self.budget - len(self.labeled_index):
                        new_labeled_index = new_labeled_index[:self.budget - len(self.labeled_index)]
                    print("pick next segments", new_labeled_index)
                    log_dict["selected_segments"].extend(new_labeled_index)
                    log_dict["selected_gt_labels"].extend(self.labels[new_labeled_index].tolist())
                    self.labeled_index += new_labeled_index
                    print("# labeled segments", len(self.labeled_index))
                    print("# positive: {}, # negative: {}".format(sum(self.labels[self.labeled_index]), len(self.labels[self.labeled_index]) - sum(self.labels[self.labeled_index])))
                    log_dict["current_npos"] = sum(self.labels[self.labeled_index]).item()
                    log_dict["current_nneg"] = (len(self.labels[self.labeled_index]) - sum(self.labels[self.labeled_index])).item()
                    # assert labeled_index does not contain duplicates
                    # assert(len(self.labeled_index) == len(set(self.labeled_index)))
                    # Update scores
                    self.n_prediction_count += len(new_labeled_index) * len(self.candidate_queries)
                    updated_scores = []
                    for result in self.executor.map(self.get_query_score, [query.program for query, _ in self.candidate_queries]):
                        updated_scores.append(result)
                    for i in range(len(self.candidate_queries)):
                        self.candidate_queries[i][1] = updated_scores[i]
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
                    print("beam_width queries", [(program_to_dsl(query.program, self.rewrite_variables), score) for query, score in self.candidate_queries])
                elif self.strategy == "topk":
                    # Sorted with randomized ties
                    self.candidate_queries = sorted(self.candidate_queries, key=lambda x: cmp_to_key(self.compare_with_ties)(x[1]), reverse=True)[:self.beam_width]
                    print("beam_width queries", [(program_to_dsl(query.program, self.rewrite_variables), score) for query, score in self.candidate_queries])
                elif self.strategy == "topk_including_ties":
                    self.candidate_queries = sorted(self.candidate_queries, key=lambda x: x[1], reverse=True)
                    utility_bound = self.candidate_queries[:self.beam_width][-1][1]
                    self.candidate_queries = [e for e in self.candidate_queries if e[1] >= utility_bound]
                    print("beam_width {} queries".format(len(self.candidate_queries)), [(program_to_dsl(query.program, self.rewrite_variables), score) for query, score in self.candidate_queries])
            else:
                self.candidate_queries = []

            # Retain the top k queries with the highest scores as answers
            _start_retain_top_k_queries_time = time.time()
            self.n_prediction_count += len(self.answers) * len(self.labeled_index)
            updated_scores = []
            for result in self.executor.map(self.get_query_score, [query.program for query, _ in self.answers]):
                updated_scores.append(result)
            for i in range(len(self.answers)):
                self.answers[i][1] = updated_scores[i]
            self.answers = sorted(self.answers, key=lambda x: cmp_to_key(self.compare_with_ties)(x[1]), reverse=True)
            best_score = self.answers[0][1]
            utility_bound = self.answers[:self.k][-1][1]
            self.answers = [e for e in self.answers if e[1] >= utility_bound]
            # self.answers = sorted(self.answers, key=lambda x: x[1], reverse=True)
            # self.answers = self.answers[:self.k]
            top_k_queries_with_scores = [(program_to_dsl(query.program, self.rewrite_variables), score) for query, score in self.answers]
            print("top k queries", top_k_queries_with_scores)
            log_dict["top_k_queries_with_scores"] = top_k_queries_with_scores
            self.retain_top_k_queries_time += time.time() - _start_retain_top_k_queries_time
            self.best_query_after_each_iter = [e for e in self.answers if e[1] >= best_score]
            print("best query after each iter", [(program_to_dsl(query.program, self.rewrite_variables), score) for query, score in self.best_query_after_each_iter])
            print(using("profile"))
            for query, score in self.best_query_after_each_iter:
                self.output_log.append((program_to_dsl(query.program, self.rewrite_variables), score))
            self.output_log.append("[Runtime so far] {}".format(time.time() - self._start_total_time))
            self.iteration += 1
            log_dict["best_query"] = program_to_dsl(self.best_query_after_each_iter[0][0].program, self.rewrite_variables)
            log_dict["best_score"] = self.best_query_after_each_iter[0][1].item()
            # Prediction
            pred_per_query = self.execute_over_all_inputs_postgres(self.answers[0][0].program, is_test=True)
            print("predicted_labels_test", pred_per_query)
            log_dict["predicted_labels_test"] = pred_per_query
            log.append(log_dict)
        return log

    def interactive_live(self, user_labels=None):
        self.label_count_per_iter += len(user_labels) if user_labels else 0 # len(user_labels) should be 1
        log_dict = {}
        if len(self.candidate_queries): # iteration > 0 and not done yet
            print("# labeled segments", len(self.labeled_index)) # should be 1
            self.labels[self.new_labeled_index] = user_labels[0]
            print("# positive: {}, # negative: {}".format(sum(self.labels[self.labeled_index]), len(self.labels[self.labeled_index]) - sum(self.labels[self.labeled_index])))
            # Select new video segments to label
            if len(self.labeled_index) < self.budget and len(self.candidate_queries):
                _start_segment_selection_time = time.time()
                if self.label_count_per_iter < self.samples_per_iter[self.iteration]:
                    log_dict["state"] = "label_more"
                    log_dict["iteration"] = self.iteration
                    log_dict["sample_idx"] = self.label_count_per_iter
                    log_dict["selected_segments"] = []
                    log_dict["selected_gt_labels"] = []
                    _start_segment_selection_time_per_iter = time.time()
                    self.candidate_queries = sorted(self.candidate_queries, key=lambda x: cmp_to_key(self.compare_with_ties)(x[1]), reverse=True)
                    self.new_labeled_index = self.active_learning()
                    if len(self.new_labeled_index) > self.budget - len(self.labeled_index):
                        self.new_labeled_index = self.new_labeled_index[:self.budget - len(self.labeled_index)]
                    print("pick next segments", self.new_labeled_index)
                    log_dict["selected_segments"].extend(self.new_labeled_index)
                    log_dict["selected_gt_labels"].extend(self.labels[self.new_labeled_index].tolist()) # TODO: This is the ground truth label and should be used for experimental analysis only. We should not send this to the front-end.
                    self.labeled_index += self.new_labeled_index # The segment will be labeled after the user submits the label
                    # log_dict["selected_gt_labels"].extend(self.labels[new_labeled_index].tolist())
                    print("test segment_selection_time_per_iter time:", time.time() - _start_segment_selection_time_per_iter)
                    self.segment_selection_time += time.time() - _start_segment_selection_time
                    return log_dict
                else:
                    # Done with labeling for this iteration
                    self.label_count_per_iter = 0
            log_dict["current_npos"] = sum(self.labels[self.labeled_index]).item()
            log_dict["current_nneg"] = (len(self.labels[self.labeled_index]) - sum(self.labels[self.labeled_index])).item()
            # Update scores
            self.n_prediction_count += len(self.new_labeled_index) * len(self.candidate_queries)
            updated_scores = []
            for result in self.executor.map(self.get_query_score, [query.program for query, _ in self.candidate_queries]):
                updated_scores.append(result)
            for i in range(len(self.candidate_queries)):
                self.candidate_queries[i][1] = updated_scores[i]

            # Sample beam_width queries
            if len(self.candidate_queries):
                if self.strategy == "sampling":
                    self.candidate_queries = np.asarray(self.candidate_queries, dtype=object)
                    weight = self.candidate_queries[:, 1].astype(np.float)
                    weight = weight / np.sum(weight)
                    candidate_idx = np.random.choice(np.arange(self.candidate_queries.shape[0]), size=min(self.beam_width, self.candidate_queries.shape[0]), replace=False, p=weight)
                    print("candidate_idx", candidate_idx)
                    self.candidate_queries = self.candidate_queries[candidate_idx].tolist()
                    print("beam_width queries", [(program_to_dsl(query.program, self.rewrite_variables), score) for query, score in self.candidate_queries])
                elif self.strategy == "topk":
                    # Sorted with randomized ties
                    self.candidate_queries = sorted(self.candidate_queries, key=lambda x: cmp_to_key(self.compare_with_ties)(x[1]), reverse=True)[:self.beam_width]
                    print("beam_width queries", [(program_to_dsl(query.program, self.rewrite_variables), score) for query, score in self.candidate_queries])
                elif self.strategy == "topk_including_ties":
                    self.candidate_queries = sorted(self.candidate_queries, key=lambda x: x[1], reverse=True)
                    utility_bound = self.candidate_queries[:self.beam_width][-1][1]
                    self.candidate_queries = [e for e in self.candidate_queries if e[1] >= utility_bound]
                    print("beam_width {} queries".format(len(self.candidate_queries)), [(program_to_dsl(query.program, self.rewrite_variables), score) for query, score in self.candidate_queries])
            else:
                self.candidate_queries = []

            # Retain the top k queries with the highest scores as answers
            _start_retain_top_k_queries_time = time.time()
            self.n_prediction_count += len(self.answers) * len(self.labeled_index)
            updated_scores = []
            for result in self.executor.map(self.get_query_score, [query.program for query, _ in self.answers]):
                updated_scores.append(result)
            for i in range(len(self.answers)):
                self.answers[i][1] = updated_scores[i]
            self.answers = sorted(self.answers, key=lambda x: cmp_to_key(self.compare_with_ties)(x[1]), reverse=True)
            best_score = self.answers[0][1]
            utility_bound = self.answers[:self.k][-1][1]
            self.answers = [e for e in self.answers if e[1] >= utility_bound]
            # self.answers = sorted(self.answers, key=lambda x: x[1], reverse=True)
            # self.answers = self.answers[:self.k]
            top_k_queries_with_scores = [(program_to_dsl(query.program, self.rewrite_variables), score) for query, score in self.answers]
            print("top k queries", top_k_queries_with_scores)
            log_dict["top_k_queries_with_scores"] = top_k_queries_with_scores
            self.retain_top_k_queries_time += time.time() - _start_retain_top_k_queries_time
            self.best_query_after_each_iter = [e for e in self.answers if e[1] >= best_score]
            print("best query after each iter", [(program_to_dsl(query.program, self.rewrite_variables), score) for query, score in self.best_query_after_each_iter])
            print(using("profile"))
            for query, score in self.best_query_after_each_iter:
                self.output_log.append((program_to_dsl(query.program, self.rewrite_variables), score))
            self.output_log.append("[Runtime so far] {}".format(time.time() - self._start_total_time))
            self.iteration += 1
            log_dict["best_query_list"] = [program_to_dsl(query.program, self.rewrite_variables) for query, _ in self.best_query_after_each_iter]
            log_dict["best_score_list"] = [score.item() for _, score in self.best_query_after_each_iter]
            # Prediction
            pred_per_query = self.execute_over_all_inputs_postgres(self.best_query_after_each_iter[0][0].program, is_test=True)
            print("predicted_labels_test", pred_per_query)
            log_dict["predicted_labels_test"] = pred_per_query
        if len(self.candidate_queries) or self.iteration == 0:
            log_dict["state"] = "label_first"
            log_dict["iteration"] = self.iteration
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
                    # Special case: For warsaw dataset, we also consider the case where the single predicate is applied to the second object.
                    if nvars == 1 and self.dataset_name == "warsaw":
                        variable_lists = [["o0"], ["o1"]]
                    else:
                        variable_lists = [["o{}".format(i) for i in range(nvars)]]
                    for variables in variable_lists:
                        query_graph = QueryGraph(self.dataset_name, self.max_npred, self.max_depth, self.max_nontrivial, self.max_trivial, self.max_duration, self.max_vars, self.predicate_list, is_trajectory=self.is_trajectory)
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
                        new_candidate_queries.append([query_graph, -1])
                # Assign queries to the threads
                updated_scores = []
                for result in self.executor.map(self.get_query_score, [query.program for query, _ in new_candidate_queries]):
                    updated_scores.append(result)
                for i in range(len(new_candidate_queries)):
                    new_candidate_queries[i][1] = updated_scores[i]
                    print("initialization", program_to_dsl(new_candidate_queries[i][0].program, self.rewrite_variables), updated_scores[i])
                self.candidate_queries = new_candidate_queries
                self.answers.extend(self.candidate_queries)
                self.n_prediction_count += len(self.labeled_index) * len(self.candidate_queries)
            else:
                # Expand queries
                for candidate_query in self.candidate_queries:
                    candidate_query_graph, _ = candidate_query
                    print("expand search space", program_to_dsl(candidate_query_graph.program, self.rewrite_variables))
                    all_children = candidate_query_graph.get_all_children_unrestricted_postgres()
                    new_candidate_queries.extend([[child, -1] for child in all_children])

                # Remove duplicates for new_candidate_queries
                new_candidate_queries_removing_duplicates = []
                print("[new_candidate_queries] before removing duplicates:", len(new_candidate_queries))
                for query, score in new_candidate_queries:
                    print(program_to_dsl(query.program, self.rewrite_variables), score)
                signatures = set()
                for query, score in new_candidate_queries:
                    signature = program_to_dsl(query.program, self.rewrite_variables)
                    if signature not in signatures:
                        query.program = dsl_to_program(signature)
                        new_candidate_queries_removing_duplicates.append([query, score])
                        signatures.add(signature)
                print("[new_candidate_queries] after removing duplicates:", len(new_candidate_queries_removing_duplicates))
                for query, score in new_candidate_queries_removing_duplicates:
                    print(program_to_dsl(query.program, self.rewrite_variables), score)
                self.candidate_queries = new_candidate_queries_removing_duplicates
                self.n_queries_explored += len(self.candidate_queries)

                # Compute scores
                self.n_prediction_count += len(self.labeled_index) * len(self.candidate_queries)
                candidate_queries_greater_than_zero = []
                updated_scores = []
                for result in self.executor.map(self.get_query_score, [query.program for query, _ in self.candidate_queries]):
                    updated_scores.append(result)
                for i in range(len(self.candidate_queries)):
                    if updated_scores[i] > 0:
                        candidate_queries_greater_than_zero.append([self.candidate_queries[i][0], updated_scores[i]])
                self.candidate_queries = candidate_queries_greater_than_zero
                self.answers.extend(self.candidate_queries)
            if len(self.candidate_queries) == 0:
                # Terminating synthesis algorithm
                log_dict["state"] = "terminated"
                return log_dict
            # Remove duplicates for self.answers
            answers_removing_duplicates = []
            print("[self.answers] before removing duplicates:", len(self.answers))
            for query, score in self.answers:
                print(program_to_dsl(query.program, self.rewrite_variables), score)
            signatures = set()
            for query, score in self.answers:
                signature = program_to_dsl(query.program, self.rewrite_variables)
                if signature not in signatures:
                    query.program = dsl_to_program(signature)
                    answers_removing_duplicates.append([query, score])
                    signatures.add(signature)
            print("[self.answers] after removing duplicates:", len(answers_removing_duplicates))
            for query, score in answers_removing_duplicates:
                print(program_to_dsl(query.program, self.rewrite_variables), score)
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
                log_dict["selected_segments"] = []
                log_dict["selected_gt_labels"] = []
                log_dict["sample_idx"] = 0
                # if self.label_count_per_iter < self.samples_per_iter[self.iteration]:
                # for _ in range(self.samples_per_iter[self.iteration]):
                _start_segment_selection_time_per_iter = time.time()
                self.candidate_queries = sorted(self.candidate_queries, key=lambda x: cmp_to_key(self.compare_with_ties)(x[1]), reverse=True)
                self.new_labeled_index = self.active_learning()
                if len(self.new_labeled_index) > self.budget - len(self.labeled_index):
                    self.new_labeled_index = self.new_labeled_index[:self.budget - len(self.labeled_index)]
                print("pick next segments", self.new_labeled_index)
                log_dict["selected_segments"].extend(self.new_labeled_index)
                log_dict["selected_gt_labels"].extend(self.labels[self.new_labeled_index].tolist()) # TODO: This is the ground truth label and should be used for experimental analysis only. We should not send this to the front-end.
                self.labeled_index += self.new_labeled_index # The segment will be labeled after the user submits the label
                print("test segment_selection_time_per_iter time:", time.time() - _start_segment_selection_time_per_iter)
                self.segment_selection_time += time.time() - _start_segment_selection_time
            return log_dict
            # TODO: When we reach the labeling budget but the algorithm has not terminated, we should return zero video segments for labels. At the front-end, we should simply indicate that no more labels are needed and the user could continue to the next iteration.
        else:
            # Terminating synthesis algorithm
            log_dict["state"] = "terminated"
            return log_dict