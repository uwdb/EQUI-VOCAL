from quivr.methods.base_method import BaseMethod
from quivr.utils import print_program
from quivr.query_graph import QueryGraph
import numpy as np
import time
from lru import LRU
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import resource
import random
import quivr.dsl as dsl

def using(point=""):
    usage=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return '''%s: mem=%s MB
           '''%(point, usage/1024.0 )

random.seed(10)

class ExhaustiveSearch(BaseMethod):

    def __init__(self, inputs, labels, predicate_dict, max_num_atomic_predicates=5, max_depth=2, k=20, max_duration=2, multithread=1):
        self.inputs = np.array(inputs, dtype=object)
        self.labels = np.array(labels, dtype=object)
        self.predicate_dict = predicate_dict
        self.max_num_atomic_predicates = max_num_atomic_predicates
        self.max_depth = max_depth
        self.k = k
        self.max_duration = max_duration
        self.multithread = multithread

        self.query_expansion_time = 0
        self.segment_selection_time = 0
        self.retain_top_k2_queries_time = 0
        self.answers = []

        if self.multithread > 1:
            self.m = multiprocessing.Manager()
            self.lock = self.m.Lock()
        elif self.multithread == 1:
            self.lock = None
        else:
            raise ValueError("multithread must be 1 or greater")

    def run(self):
        self.labeled_index = list(range(len(self.labels)))
        _start_total_time = time.time()
        self.memoize_all_inputs = [LRU(10000) for _ in range(len(self.inputs))] # For each (input, label) pair, memoize the results of all sub-queries encountered during synthesis.
        print("LRU can maximally store {} results".format(10000))

        # Initialize program graph
        candidate_queries = []

        for pred in self.predicate_dict:
            pred_instances = []
            if self.predicate_dict[pred]:
                for param in self.predicate_dict[pred]:
                    pred_instances.append(pred(param))
            else:
                pred_instances.append(pred())
            for pred_instance in pred_instances:
                query_graph = QueryGraph(self.max_num_atomic_predicates, self.max_depth)
                query_graph.program = dsl.SequencingOperator(dsl.SequencingOperator(dsl.TrueStar(), pred_instance), dsl.TrueStar())
                query_graph.num_atomic_predicates = 1
                query_graph.depth = 1
                candidate_queries.append(query_graph)

        if self.multithread > 1:
            with ThreadPoolExecutor(max_workers=self.multithread) as executor:
                # TODO: right now only `len(self.predicate_dict)' threads are used.
                executor.map(self.dfs, candidate_queries)
        else:
            for current_query_graph in candidate_queries:
                self.dfs(current_query_graph)

        # RETURN: the list.
        print("final_answers")
        for query_graph, score in self.answers:
            print("answer", print_program(query_graph.program), score)

        print("[Runtime] total time: {}".format(time.time() - _start_total_time))

        return self.answers

    def dfs(self, current_query_graph):
        # print("expand search space", print_program(current_query_graph.program))
        all_children = current_query_graph.get_all_children_bu(self.predicate_dict, self.max_duration)
        # Compute F1 score of each candidate query
        for child in all_children:
            # answers.extend(self.dfs(child))
            self.dfs(child)
        _start_time = time.time()
        score = self.compute_query_score(current_query_graph.program)
        print("add query: {}, score: {}, time:{}".format(print_program(current_query_graph.program), score, time.time() - _start_time))
        if self.lock:
            self.lock.acquire()
        if score > 0:
            # if len(self.answers) == 0 or score >= self.answers[-1][1]:
            self.answers.append([current_query_graph, score])
            self.answers = sorted(self.answers, key=lambda x: x[1], reverse=True)
            self.answers = self.answers[:self.k]
        if self.lock:
            self.lock.release()