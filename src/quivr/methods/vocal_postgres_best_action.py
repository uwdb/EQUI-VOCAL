from quivr.methods.vocal_postgres import VOCALPostgres
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

# random.seed(10)
random.seed(time.time())

class VOCALPostgresBestAction(VOCALPostgres):
    def main(self):
        while len(self.candidate_queries) or self.iteration == 0:
            print("[Step {}]".format(self.iteration))
            self.output_log.append("[Step {}]".format(self.iteration))
            # Generate more queries
            _start_query_expansion_time = time.time()
            new_candidate_queries_a = []
            new_candidate_queries_b = []
            new_candidate_queries_c = []
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
                    score = self.compute_query_score_postgres(query_graph.program)
                    print("initialization", rewrite_program_postgres(query_graph.program), score)
                    new_candidate_queries_b.append([query_graph, score])
                    new_candidate_queries_b = sorted(new_candidate_queries_b, key=lambda x: cmp_to_key(self.compare_with_ties)(x[1]), reverse=True)
                    self.answers.append([query_graph, score])
            else:
                if self.multithread > 1:
                    with ThreadPoolExecutor(max_workers=self.multithread) as executor:
                        for result_a, result_b, result_c in executor.map(self.expand_query_and_compute_score, self.candidate_queries):
                            new_candidate_queries_a.extend(result_a)
                            new_candidate_queries_b.extend(result_b)
                            new_candidate_queries_c.extend(result_c)
                else:
                    for candidate_query in self.candidate_queries:
                        result_a, result_b, result_c = self.expand_query_and_compute_score(candidate_query)
                        new_candidate_queries_a.extend(result_a)
                        new_candidate_queries_b.extend(result_b)
                        new_candidate_queries_c.extend(result_c)
                self.answers.extend(new_candidate_queries_a)
                self.answers.extend(new_candidate_queries_b)
                self.answers.extend(new_candidate_queries_c)
            self.query_expansion_time += time.time() - _start_query_expansion_time

            # Remove duplicates for new_candidate_queries
            new_candidate_queries_removing_duplicates = []
            signatures = set()
            for query, score in new_candidate_queries_a:
                signature = rewrite_program_postgres(query.program)
                if signature not in signatures:
                    query.program = str_to_program_postgres(signature)
                    new_candidate_queries_removing_duplicates.append([query, score])
                    signatures.add(signature)
            new_candidate_queries_a = new_candidate_queries_removing_duplicates

            new_candidate_queries_removing_duplicates = []
            signatures = set()
            for query, score in new_candidate_queries_b:
                signature = rewrite_program_postgres(query.program)
                if signature not in signatures:
                    query.program = str_to_program_postgres(signature)
                    new_candidate_queries_removing_duplicates.append([query, score])
                    signatures.add(signature)
            new_candidate_queries_b = new_candidate_queries_removing_duplicates

            new_candidate_queries_removing_duplicates = []
            signatures = set()
            for query, score in new_candidate_queries_c:
                signature = rewrite_program_postgres(query.program)
                if signature not in signatures:
                    query.program = str_to_program_postgres(signature)
                    new_candidate_queries_removing_duplicates.append([query, score])
                    signatures.add(signature)
            new_candidate_queries_c = new_candidate_queries_removing_duplicates


            # Select new video segments to label
            if len(self.labeled_index) < self.budget and len(new_candidate_queries_a) + len(new_candidate_queries_b) + len(new_candidate_queries_c):
                _start_segment_selection_time = time.time()
                # for _ in range(self.samples_per_iter[self.iteration]):
                # _start_segment_selection_time_per_iter = time.time()
                # Compute disagreement score for each action
                new_candidate_queries_all_actions = [new_candidate_queries_a, new_candidate_queries_b, new_candidate_queries_c]
                disagreement_scores = []
                next_labels = []
                for new_candidate_queries in new_candidate_queries_all_actions:
                    if len(new_candidate_queries) == 0:
                        disagreement_scores.append(-1)
                        next_labels.append(-1)
                    else:
                        new_candidate_queries = sorted(new_candidate_queries, key=lambda x: cmp_to_key(self.compare_with_ties)(x[1]), reverse=True)
                        disagreement_score, new_labeled_index = self.get_disagreement_score(new_candidate_queries)
                        disagreement_scores.append(disagreement_score)
                        next_labels.append(new_labeled_index)
                actions = ["a", "b", "c"]
                selected_action_idx = np.argmax(disagreement_scores)
                selected_action = actions[selected_action_idx]
                print("disagreement_scores: ", disagreement_scores)
                print("selected_action: ", selected_action)
                self.candidate_queries = new_candidate_queries_all_actions[selected_action_idx]
                new_labeled_index = next_labels[selected_action_idx]
                if len(new_labeled_index) > self.budget - len(self.labeled_index):
                    new_labeled_index = new_labeled_index[:self.budget - len(self.labeled_index)]
                print("pick next segments", new_labeled_index)
                self.labeled_index += new_labeled_index
                print("# labeled segments", len(self.labeled_index))
                print("# positive: {}, # negative: {}".format(sum(self.labels[self.labeled_index]), len(self.labels[self.labeled_index]) - sum(self.labels[self.labeled_index])))
                # assert labeled_index does not contain duplicates
                # assert(len(self.labeled_index) == len(set(self.labeled_index)))
                # Update scores
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
                # print("test segment_selection_time_per_iter time:", time.time() - _start_segment_selection_time_per_iter)
                self.segment_selection_time += time.time() - _start_segment_selection_time
            else:
                self.candidate_queries = new_candidate_queries_a + new_candidate_queries_b + new_candidate_queries_c

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
        all_children_a = current_query_graph.get_all_children_one_action_postgres("a")
        all_children_b = current_query_graph.get_all_children_one_action_postgres("b")
        all_children_c = current_query_graph.get_all_children_one_action_postgres("c")

        new_candidate_queries_a = []
        new_candidate_queries_b = []
        new_candidate_queries_c = []

        # Compute F1 score of each candidate query
        for child in all_children_a:
            score = self.compute_query_score_postgres(child.program)
            if score > 0:
                new_candidate_queries_a.append([child, score])
                print(rewrite_program_postgres(child.program), score)
        for child in all_children_b:
            score = self.compute_query_score_postgres(child.program)
            if score > 0:
                new_candidate_queries_b.append([child, score])
                print(rewrite_program_postgres(child.program), score)
        for child in all_children_c:
            score = self.compute_query_score_postgres(child.program)
            if score > 0:
                new_candidate_queries_c.append([child, score])
                print(rewrite_program_postgres(child.program), score)
        return new_candidate_queries_a, new_candidate_queries_b, new_candidate_queries_c