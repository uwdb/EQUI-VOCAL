from methods.vocal_postgres import VOCALPostgres
from methods.base_method import BaseMethod
from utils import rewrite_program_postgres, str_to_program_postgres, complexity_cost
from query_graph import QueryGraph
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
import dsl
from functools import cmp_to_key
import psycopg2 as psycopg
import uuid

def using(point=""):
    usage=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return '''%s: mem=%s MB'''%(point, usage/1024.0 )

class VOCALPostgresExplore(VOCALPostgres):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.explore_ratio = 0.3
        self.explore_beam_width = int(self.beam_width * self.explore_ratio)
        self.exploit_beam_width = self.beam_width - self.explore_beam_width
        self.explore_query_expansion_time = 0

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
                    with ThreadPoolExecutor(max_workers=self.multithread) as executor:
                        for result in executor.map(self.get_query_score, [query.program for query, _ in self.candidate_queries]):
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
                        with ThreadPoolExecutor(max_workers=self.multithread) as executor:
                            for result in executor.map(self.get_query_score, [query.program for query, _ in self.candidate_queries]):
                                updated_scores.append(result)
                        for i in range(len(self.candidate_queries)):
                            self.candidate_queries[i][1] = updated_scores[i]
                    else:
                        for i in range(len(self.candidate_queries)):
                            self.candidate_queries[i][1] = self.get_query_score(self.candidate_queries[i][0].program)
                    print("test segment_selection_time_per_iter time:", time.time() - _start_segment_selection_time_per_iter)
                self.segment_selection_time += time.time() - _start_segment_selection_time

            # Generate explore_beam_width queries from scratch
            _start_explore_query_expansion_time = time.time()
            explore_iteration = 0
            explore_candidate_queries = []
            while (explore_iteration < self.iteration + 1) and (len(explore_candidate_queries) or explore_iteration == 0):
                new_candidate_queries = []
                if explore_iteration == 0:
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
                        explore_candidate_queries = sorted(new_candidate_queries, key=lambda x: cmp_to_key(self.compare_with_ties)(x[1]), reverse=True)
                    self.n_prediction_count += len(self.labeled_index) * len(pred_instances)
                else:
                    # Expand queries
                    for candidate_query in explore_candidate_queries:
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
                    explore_candidate_queries = new_candidate_queries_removing_duplicates
                    self.n_queries_explored += len(explore_candidate_queries)

                    # Compute scores
                    self.n_prediction_count += len(self.labeled_index) * len(explore_candidate_queries)
                    candidate_queries_greater_than_zero = []
                    if self.multithread > 1:
                        updated_scores = []
                        with ThreadPoolExecutor(max_workers=self.multithread) as executor:
                            for result in executor.map(self.get_query_score, [query.program for query, _ in explore_candidate_queries]):
                                updated_scores.append(result)
                        for i in range(len(explore_candidate_queries)):
                            if updated_scores[i] > 0:
                                candidate_queries_greater_than_zero.append([explore_candidate_queries[i][0], updated_scores[i]])
                    else:
                        # Compute F1 score of each candidate query
                        for i in range(len(explore_candidate_queries)):
                            score = self.get_query_score(explore_candidate_queries[i][0].program)
                            if score > 0:
                                candidate_queries_greater_than_zero.append([explore_candidate_queries[i][0], score])
                    explore_candidate_queries = candidate_queries_greater_than_zero
                # Sample beam_width queries
                if len(explore_candidate_queries):
                    if self.strategy == "sampling":
                        explore_candidate_queries = np.asarray(explore_candidate_queries, dtype=object)
                        weight = explore_candidate_queries[:, 1].astype(np.float)
                        weight = weight / np.sum(weight)
                        candidate_idx = np.random.choice(np.arange(explore_candidate_queries.shape[0]), size=min(self.beam_width, explore_candidate_queries.shape[0]), replace=False, p=weight)
                        print("candidate_idx", candidate_idx)
                        explore_candidate_queries = explore_candidate_queries[candidate_idx].tolist()
                        print("beam_width queries", [(rewrite_program_postgres(query.program), score) for query, score in explore_candidate_queries])
                    elif self.strategy == "topk":
                        # Sorted with randomized ties
                        explore_candidate_queries = sorted(explore_candidate_queries, key=lambda x: cmp_to_key(self.compare_with_ties)(x[1]), reverse=True)[:self.beam_width]
                        print("beam_width queries", [(rewrite_program_postgres(query.program), score) for query, score in explore_candidate_queries])
                    elif self.strategy == "topk_including_ties":
                        explore_candidate_queries = sorted(explore_candidate_queries, key=lambda x: x[1], reverse=True)
                        utility_bound = explore_candidate_queries[:self.beam_width][-1][1]
                        explore_candidate_queries = [e for e in explore_candidate_queries if e[1] >= utility_bound]
                        print("beam_width {} queries".format(len(explore_candidate_queries)), [(rewrite_program_postgres(query.program), score) for query, score in explore_candidate_queries])
                else:
                    explore_candidate_queries = []
                explore_iteration += 1
            explore_candidate_queries = explore_candidate_queries[:self.explore_beam_width]
            self.explore_query_expansion_time += time.time() - _start_explore_query_expansion_time

            # Remove duplicates of explore_candidate_queries from candidate_queries
            candidate_queries_removing_duplicates = []
            signatures = set()
            for query, score in explore_candidate_queries:
                signature = rewrite_program_postgres(query.program)
                signatures.add(signature)
            for query, score in self.candidate_queries:
                signature = rewrite_program_postgres(query.program)
                if signature not in signatures:
                    query.program = str_to_program_postgres(signature)
                    candidate_queries_removing_duplicates.append([query, score])
                    signatures.add(signature)
            self.candidate_queries = candidate_queries_removing_duplicates

            # Sample exploit_beam_width queries
            if len(self.candidate_queries):
                if self.strategy == "sampling":
                    self.candidate_queries = np.asarray(self.candidate_queries, dtype=object)
                    weight = self.candidate_queries[:, 1].astype(np.float)
                    weight = weight / np.sum(weight)
                    candidate_idx = np.random.choice(np.arange(self.candidate_queries.shape[0]), size=min(self.exploit_beam_width, self.candidate_queries.shape[0]), replace=False, p=weight)
                    print("candidate_idx", candidate_idx)
                    self.candidate_queries = self.candidate_queries[candidate_idx].tolist()
                    print("exploit_beam_width queries", [(rewrite_program_postgres(query.program), score) for query, score in self.candidate_queries])
                elif self.strategy == "topk":
                    # Sorted with randomized ties
                    self.candidate_queries = sorted(self.candidate_queries, key=lambda x: cmp_to_key(self.compare_with_ties)(x[1]), reverse=True)[:self.exploit_beam_width]
                    print("exploit_beam_width queries", [(rewrite_program_postgres(query.program), score) for query, score in self.candidate_queries])
                elif self.strategy == "topk_including_ties":
                    self.candidate_queries = sorted(self.candidate_queries, key=lambda x: x[1], reverse=True)
                    utility_bound = self.candidate_queries[:self.exploit_beam_width][-1][1]
                    self.candidate_queries = [e for e in self.candidate_queries if e[1] >= utility_bound]
                    print("exploit_beam_width {} queries".format(len(self.candidate_queries)), [(rewrite_program_postgres(query.program), score) for query, score in self.candidate_queries])
            else:
                self.candidate_queries = []

            self.candidate_queries.extend(explore_candidate_queries)
            print("beam_width {} queries".format(len(self.candidate_queries)), [(rewrite_program_postgres(query.program), score) for query, score in self.candidate_queries])

            # Retain the top k queries with the highest scores as answers
            _start_retain_top_k_queries_time = time.time()
            self.n_prediction_count += len(self.answers) * len(self.labeled_index)
            if self.multithread > 1:
                updated_scores = []
                with ThreadPoolExecutor(max_workers=self.multithread) as executor:
                    for result in executor.map(self.get_query_score, [query.program for query, _ in self.answers]):
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