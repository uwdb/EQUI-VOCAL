"""
The implementation of the QUIVR baseline algorithm, which uses hard constraints to prune intermediate queries.

Algorithm (pseudocode):
queue = [??]
candidates_list = []
answer_list = []

while queue not empty:
    Q = queue.pop()
    if overapproximation(Q) doesn't match inputs:
        continue
    else if Q is a sketch:
        Computes all parameters for Q that are consistent with W (for each parameter in Q,
        compute its largest/smallest possible value when overapproximating all other parameters).
        Add them to candidates_list
    else:
        Add all the children of Q to queue

for Q in candidates_list:
    if Q matches with W, add it to answer_list

return answer_list
"""

# Adapted from https://github.com/trishullab/near

from utils import print_program
from query_graph import QueryGraph
import dsl
import copy
import numpy as np
import time
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import resource
from lru import LRU

def using(point=""):
    usage=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return '''%s: mem=%s MB
           '''%(point, usage/1024.0 )

class QUIVR:

    def __init__(self, inputs, labels, predicate_dict, max_npred, max_depth, max_duration, multithread):
        self.inputs = np.array(inputs, dtype=object)
        self.labels = np.array(labels, dtype=object)
        self.predicate_dict = predicate_dict
        self.max_npred = max_npred
        self.max_depth = max_depth
        self.max_duration = max_duration
        self.multithread = multithread
        self.answers = []

        if self.multithread > 1:
            self.m = multiprocessing.Manager()
            self.lock = self.m.Lock()
        elif self.multithread == 1:
            self.lock = None
        else:
            raise ValueError("multithread must be 1 or greater")

    def run(self):
        _start_total_time = time.time()
        self.memoize_all_inputs = [LRU(10000) for _ in range(len(self.inputs))] # For each (input, label) pair, memoize the results of all sub-queries encountered during synthesis.

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
                query_graph = QueryGraph(self.max_npred, self.max_depth)
                query_graph.program = dsl.SequencingOperator(dsl.SequencingOperator(dsl.TrueStar(), pred_instance), dsl.TrueStar())
                query_graph.num_atomic_predicates = 1
                query_graph.depth = 1
                candidate_queries.append(query_graph)

        if self.multithread > 1:
            with ThreadPoolExecutor(max_workers=self.multithread) as executor:
                # TODO: right now only `len(self.predicate_dict)' threads are used.
                for result in executor.map(self.expand_and_prune, candidate_queries):
                    self.answers.extend(result)
        else:
            for current_query_graph in candidate_queries:
                self.answers.extend(self.expand_and_prune(current_query_graph))

        # return answer
        print("answer size", len(self.answers))
        for query in self.answers:
            print("answer", print_program(query))
        total_time = time.time() - _start_total_time
        print("[Runtime] total time: {}".format(total_time))
        print(using("profile"))
        return self.answers, total_time

    def expand_and_prune(self, query_graph):
        queue = [query_graph]
        answers = []
        while len(queue) != 0:
            current_query_graph = queue.pop(0) # Pop the first element
            current_query = current_query_graph.program
            # IF: overapproximation doesn't match, prune Q
            overapproximation_for_all_inputs = True
            is_consistent = True
            for i, (input, label) in enumerate(zip(self.inputs, self.labels)):
                if self.lock:
                    self.lock.acquire()
                memoize = self.memoize_all_inputs[i]
                if self.lock:
                    self.lock.release()
                result, new_memoize = current_query.execute(input, label, memoize, {})
                if self.lock:
                    self.lock.acquire()
                self.memoize_all_inputs[i].update(new_memoize)
                if self.lock:
                    self.lock.release()
                if (result[0, len(input[0])] > 0) != label:
                    is_consistent = False
                    if label == 1:
                        overapproximation_for_all_inputs = False
                    break
            if not overapproximation_for_all_inputs:
                print("overapproximation doesn't match, prune Q")
                continue

            # If the query is consitent with inputs, add it to the answer list.
            if is_consistent:
                print("Q is consistent with inputs, add it to the answer list")
                answers.append(current_query)

                # Add all the children of Q to queue.
                all_children = current_query_graph.get_all_children_bu(self.predicate_dict, self.max_duration)
                queue.extend(all_children)
        return answers