from quivr.methods.vocal_postgres import VOCALPostgres
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
import psycopg2 as psycopg
import uuid

def using(point=""):
    usage=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return '''%s: mem=%s MB'''%(point, usage/1024.0 )


def compare_with_ties(x, y):
    if x < y:
        return -1
    elif x > y:
        return 1
    else:
        return random.randint(0, 1) * 2 - 1

class VOCALPostgresTwoStages(VOCALPostgres):
    def main(self):
        self.first_stage()
        self.iteration -= 1
        self.second_stage()

    def second_stage(self):
        self.candidate_queries = sorted(self.answers, key=lambda x: cmp_to_key(compare_with_ties)(x[1]), reverse=True)[:self.beam_width]
        print("beam_width queries", [(rewrite_program_postgres(query.program), score) for query, score in self.candidate_queries])

        while len(self.candidate_queries) or self.iteration == 0:
            self.output_log.append("[Step {}]".format(self.iteration))
            # Generate more queries
            _start_query_expansion_time = time.time()
            new_candidate_queries = []
            if self.multithread > 1:
                with ThreadPoolExecutor(max_workers=self.multithread) as executor:
                    for result in executor.map(self.expand_query_and_compute_score_second_stage, self.candidate_queries):
                        new_candidate_queries.extend(result)
            else:
                for candidate_query in self.candidate_queries:
                    new_candidate_queries.extend(self.expand_query_and_compute_score_second_stage(candidate_query))
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
                _start_segment_selection_time = time.time()
                for _ in range(self.samples_per_iter[self.iteration]):
                    _start_segment_selection_time_per_iter = time.time()
                    self.candidate_queries = sorted(self.candidate_queries, key=lambda x: cmp_to_key(compare_with_ties)(x[1]), reverse=True)
                    new_labeled_index = self.pick_next_segment_model_picker_postgres()
                    if len(new_labeled_index) > self.budget - len(self.labeled_index):
                        new_labeled_index = new_labeled_index[:self.budget - len(self.labeled_index)]
                    print("pick next segments", new_labeled_index)
                    self.labeled_index += new_labeled_index
                    print("# labeled segments", len(self.labeled_index))
                    print("# positive: {}, # negative: {}".format(sum(self.labels[self.labeled_index]), len(self.labels[self.labeled_index]) - sum(self.labels[self.labeled_index])))
                    # assert labeled_index does not contain duplicates
                    assert(len(self.labeled_index) == len(set(self.labeled_index)))
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
        all_children = current_query_graph.get_all_children_no_duration_refinement_postgres()

        new_candidate_queries = []

        # Compute F1 score of each candidate query
        for child in all_children:
            score = self.compute_query_score_postgres(child.program)
            print("test", rewrite_program_postgres(child.program), score, self.labeled_index)
            if score > 0:
                new_candidate_queries.append([child, score])
                print(rewrite_program_postgres(child.program), score)
        return new_candidate_queries

    def expand_query_and_compute_score_second_stage(self, current_query):
        current_query_graph, _ = current_query
        current_query = current_query_graph.program
        print("expand search space", rewrite_program_postgres(current_query))
        all_children = current_query_graph.get_all_children_only_duration_refinement_postgres()

        new_candidate_queries = []

        # Compute F1 score of each candidate query
        for child in all_children:
            score = self.compute_query_score_postgres(child.program)
            print("test", rewrite_program_postgres(child.program), score, self.labeled_index)
            if score > 0:
                new_candidate_queries.append([child, score])
                print(rewrite_program_postgres(child.program), score)
        return new_candidate_queries