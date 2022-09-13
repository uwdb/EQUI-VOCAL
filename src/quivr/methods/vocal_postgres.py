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

# TODO: get_all_children_unrestricted_postgres

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

    def __init__(self, inputs, labels, predicate_dict, max_npred, max_depth, max_duration, beam_width, k, samples_per_iter, budget, multithread, strategy):
        self.inputs = np.array(inputs, dtype=object)
        self.labels = np.array(labels, dtype=object)
        self.predicate_dict = predicate_dict
        self.max_npred = max_npred
        self.max_depth = max_depth
        self.beam_width = beam_width
        self.k = k
        self.samples_per_iter = samples_per_iter
        self.budget = budget
        self.max_duration = max_duration
        self.multithread = multithread
        self.strategy = strategy

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

        self.conn = psycopg.connect("dbname=myinner_db user=enhaoz host=localhost")
        # Open a cursor to perform database operations
        with self.conn.cursor() as cur:
            # Create predicates
            cur.execute("DROP FUNCTION IF EXISTS Near, Far, LeftOf, BackOf, RightQuadrant, TopQuadrant;")
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
            self.conn.commit()

    def run(self, init_labeled_index):
        _start_total_time = time.time()
        self.init_nlabels = len(init_labeled_index)
        self.labeled_index = init_labeled_index
        self.memoize_all_inputs = [LRU(10000) for _ in range(len(self.inputs))] # For each (input, label) pair, memoize the results of all sub-queries encountered during synthesis.

        # Initialize candidate queries: [query_graph, score]
        self.candidate_queries = []
        for pred in self.predicate_dict:
            pred_instances = []
            if self.predicate_dict[pred]:
                for param in self.predicate_dict[pred]:
                    pred_instances.append("{}_{}".format(pred, param))
            else:
                pred_instances.append(pred)
            for pred_instance in pred_instances:
                query_graph = QueryGraph(self.max_npred, self.max_depth)
                query_graph.program = [
                    {
                        "scene_graph": [
                            {
                                "predicate": pred_instance,
                                "variables": ["o0", "o1"]
                            }
                        ],
                        "duration_constraint": 1
                    }
                ]
                query_graph.npred = 1
                query_graph.depth = 1
                query_graph.nvars = 2
                score = self.compute_query_score_postgres(query_graph.program)
                print("initialization", rewrite_program_postgres(query_graph.program), score)
                self.candidate_queries.append([query_graph, score])
                self.answers.append([query_graph, score])

        _start_segmnet_selection_time = time.time()
        # video_segment_ids = self.pick_next_segment()
        video_segment_ids = self.pick_next_segment_model_picker()
        print("pick next segments", video_segment_ids)
        self.labeled_index += video_segment_ids
        print("# labeled segments", len(self.labeled_index))
        print("# positive: {}, # negative: {}".format(sum(self.labels[self.labeled_index]), len(self.labels[self.labeled_index]) - sum(self.labels[self.labeled_index])))
        # assert labeled_index does not contain duplicates
        assert(len(self.labeled_index) == len(set(self.labeled_index)))
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
                video_segment_ids = self.pick_next_segment_model_picker()
                print("pick next segments", video_segment_ids)
                self.labeled_index += video_segment_ids
                print("# labeled segments", len(self.labeled_index))
                print("# positive: {}, # negative: {}".format(sum(self.labels[self.labeled_index]), len(self.labels[self.labeled_index]) - sum(self.labels[self.labeled_index])))
                # assert labeled_index does not contain duplicates
                assert(len(self.labeled_index) == len(set(self.labeled_index)))
                self.segment_selection_time += time.time() - _start_segmnet_selection_time

            for i in range(len(self.candidate_queries)):
                self.candidate_queries[i][1] = self.compute_query_score_postgres(self.candidate_queries[i][0].program)

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
        return self.answers, total_time

    def expand_query_and_compute_score(self, current_query):
        current_query_graph, _ = current_query
        current_query = current_query_graph.program
        print("expand search space", rewrite_program_postgres(current_query))
        # all_children = current_query_graph.get_all_children_bu(self.predicate_dict, self.max_duration)
        all_children = current_query_graph.get_all_children_unrestricted_postgres(self.predicate_dict, self.max_duration)

        new_candidate_queries = []

        # Compute F1 score of each candidate query
        for child in all_children:
            score = self.compute_query_score_postgres(child.program)
            if score > 0:
                new_candidate_queries.append([child, score])
                print(rewrite_program_postgres(child.program), score)
        return new_candidate_queries