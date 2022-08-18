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

from quivr.utils import print_program
from quivr.query_graph import QueryGraph
import quivr.dsl as dsl
import copy
import numpy as np
import time
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from lru import LRU
class QUIVR:

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

        self.enumerate_all_candidates_time = 0
        self.prune_partial_query_time = 0
        self.prune_parameter_values_time = 0
        self.find_children_time = 0
        self.answers = []

        if self.multithread > 1:
            self.m = multiprocessing.Manager()
            self.lock = self.m.Lock()
        elif self.multithread == 1:
            self.lock = None
        else:
            raise ValueError("multithread must be 1 or greater")

    def run(self):
        self.memoize_all_inputs = [LRU(10000) for _ in range(len(self.inputs))] # For each (input, label) pair, memoize the results of all sub-queries encountered during synthesis.

        # Initialize program graph
        query_graph = QueryGraph(self.max_npred, self.max_depth, topdown_or_bottomup='topdown')

        queue = [query_graph]
        self.consistent_queries = []
        while len(queue) != 0:
            current_query_graph = queue.pop(0) # Pop the last element
            current_query = current_query_graph.program
            # IF: overapproximation doesn't match, prune Q
            _start_prune_partial_query = time.time()
            overapproximation_for_all_inputs = True
            for i, (input, label) in enumerate(zip(self.inputs, self.labels)):
                memoize = self.memoize_all_inputs[i]
                result, new_memoize = current_query.execute(input, label, memoize, {})
                self.memoize_all_inputs[i].update(new_memoize)
                if (result[0, len(input[0])] > 0) != label:
                    overapproximation_for_all_inputs = False
                    break
            self.prune_partial_query_time += time.time() - _start_prune_partial_query
            if not overapproximation_for_all_inputs:
                continue
            # ELSE IF: Q is sketch, computes all parameters for Q that are consistent with inputs, and adds them to the final list.
            if current_query_graph.is_sketch(current_query):
                _start_prune_parameter_values = time.time()
                print("current_query", print_program(current_query))
                # self.fill_all_parameter_holes(current_query) # not pruning parameter values
                self.prune_parameter_values(current_query)
                self.prune_parameter_values_time += time.time() - _start_prune_parameter_values
            # ELSE: adds all the children of Q to queue.
            else:
                _start_find_children = time.time()
                all_children = current_query_graph.get_all_children("quivr")
                queue.extend(all_children)
                self.find_children_time += time.time() - _start_find_children

        # RETURN: the list. Need to enumerate over all possible queries to omit ones that are inconsistenet with W.
        answer = []
        print("candidate queries size", len(self.consistent_queries))
        # return answer
        _start = time.time()
        for candidate_query in self.consistent_queries:
            print("candidate query", print_program(candidate_query))
            matched = True
            for i, (input, label) in enumerate(zip(inputs, labels)):
                memoize = self.memoize_all_inputs[i]
                result, memoize = candidate_query.execute(input, label, memoize)
                self.memoize_all_inputs[i] = memoize
                if not (result[0, len(input[0])] > 0) == label:
                    matched = False
                    break
            if matched:
                answer.append(candidate_query)
        self.enumerate_all_candidates_time += time.time() - _start
        print("[Runtime] enumerate all candidates time: {}, prune partial query time: {}, prune parameter values time: {}, find children time: {}".format(self.enumerate_all_candidates_time, self.prune_partial_query_time, self.prune_parameter_values_time, self.find_children_time))
        print(self.memoize_all_inputs[0].keys())
        return answer

    def prune_parameter_values(self, current_query):
        # current_query is sketch (containing only parameter holes)
        # 1. Find all parameter holes
        # 2. For each parameter hole, fix it and fill in other holes with -inf or inf.
        if QueryGraph.is_complete(current_query):
            return
            # self.consistent_queries.append(current_query)
        queue = [current_query]
        while len(queue) != 0:
            current = queue.pop(0)
            for submod, functionclass in current.submodules.items():
                if isinstance(functionclass, dsl.ParameterHole):
                    orig_fclass = copy.deepcopy(current.submodules[submod])
                    value_range = functionclass.get_value_range()
                    step = functionclass.get_step()
                    # Update the current parameter hole to parameter class (but theta is still unset)
                    predicate = copy.deepcopy(functionclass.get_predicate())
                    predicate.with_hole = True
                    current.submodules[submod] = predicate
                    new_query = copy.deepcopy(current_query)
                    value_interval = self.fill_other_parameter_holes(new_query, value_range)
                    print(value_interval)
                    if value_interval[0] > value_interval[1]:
                        return
                    orig_fclass.value_range = value_interval
                    current.submodules[submod] = orig_fclass
                    continue
                if issubclass(type(functionclass), dsl.Predicate):
                    continue
                else:
                    #add submodules
                    queue.append(functionclass)
        # 3. Construct all (candidate) consistent queries
        self.fill_all_parameter_holes(current_query)

    def fill_other_parameter_holes(self, new_query, value_range):
        theta_lb = value_range[0]
        theta_ub = value_range[1]
        for i, (input, label) in enumerate(zip(self.inputs, self.labels)):
            memoize = self.memoize_all_inputs[i]
            result, new_memoize = new_query.execute(input, label, memoize, {})
            self.memoize_all_inputs[i].update(new_memoize)
            if label == 1:
                theta_ub = min(result[0, len(input[0])], theta_ub)
            if label == 0:
                theta_lb = max(result[0, len(input[0])], theta_lb)
            if theta_lb > theta_ub:
                break
        return [theta_lb, theta_ub]

    def fill_all_parameter_holes(self, current_query):
        if QueryGraph.is_complete(current_query):
            # print("here3")
            self.consistent_queries.append(current_query)
            return
        queue = [current_query]
        while len(queue) != 0:
            current = queue.pop(0)
            for submod, functionclass in current.submodules.items():
                if isinstance(functionclass, dsl.PredicateHole):
                    raise ValueError
                if issubclass(type(functionclass), dsl.Predicate):
                    continue
                if isinstance(functionclass, dsl.ParameterHole):
                    orig_fclass = copy.deepcopy(current.submodules[submod])
                    value_range = functionclass.get_value_range()
                    step = functionclass.get_step()
                    for theta in np.arange(value_range[0], value_range[1] + 1e-6, step):
                        predicate = functionclass.fill_hole(theta)
                        # print("fill hole", predicate.name, theta)
                        # replace the parameter hole with a predicate
                        current.submodules[submod] = predicate
                        # create the correct child node
                        new_query = copy.deepcopy(current_query)
                        if not QueryGraph.is_sketch(new_query):
                            raise ValueError
                            # print("here1")
                        self.fill_all_parameter_holes(new_query)
                    current.submodules[submod] = orig_fclass
                    return
                else:
                    #add submodules
                    queue.append(functionclass)