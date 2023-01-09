"""
The implementation of the QUIVR baseline algorithm, which uses hard constraints to prune intermediate queries, and the original query expansion rules.

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

from webbrowser import get
from utils import print_program, get_depth_and_npred
from query_graph import QueryGraph
import dsl as dsl
import copy
import numpy as np
from scipy import stats
import time
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import resource
import random
from lru import LRU
from collections import deque

def using(point=""):
    usage=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return '''%s: mem=%s MB'''%(point, usage/1024.0 )

class QUIVROriginal:

    def __init__(self, inputs, labels, predicate_list, max_npred, max_nontrivial, max_trivial, max_depth, max_duration, budget, multithread, lru_capacity):
        self.inputs = np.array(inputs, dtype=object)
        self.labels = np.array(labels, dtype=object)
        self.predicate_list = predicate_list
        self.max_npred = max_npred
        self.max_nontrivial = max_nontrivial
        self.max_trivial = max_trivial
        self.max_depth = max_depth
        self.max_duration = max_duration
        self.budget = budget
        self.multithread = multithread
        self.lru_capacity = lru_capacity
        self.with_kleene = True

        self.enumerate_all_possible_answers_time = 0
        self.prune_partial_query_time = 0
        self.prune_parameter_values_time = 0
        self.find_children_time = 0
        self.other_time = 0
        self.zero_step_total_time = 0
        self.sample_selection_time = 0
        self.prune_inconsistent_queries_active_learing_time = 0
        if self.multithread > 1:
            self.m = multiprocessing.Manager()
            self.lock = self.m.Lock()
        elif self.multithread == 1:
            self.lock = None
        else:
            raise ValueError("multithread must be 1 or greater")
        self.count_candidate_queries = 0
        self.count_predictions = 0
        self.output_log = []

    def run(self, init_labeled_index):
        _start_total_time = time.time()
        init_nlabels = len(init_labeled_index)
        self.labeled_index = init_labeled_index
        if self.lru_capacity:
            self.memoize_all_inputs = [LRU(self.lru_capacity) for _ in range(len(self.inputs))] # For each (input, label) pair, memoize the results of all sub-queries encountered during synthesis.
        else:
            self.memoize_all_inputs = [{} for _ in range(len(self.inputs))]

        self.output_log.append("[Step 0]")
        answer = self.zero_steps() # First iteration
        for query in answer:
            print("answer", print_program(query))
            self.output_log.append(print_program(query))
        print("[# queries]: {}".format(len(answer)))
        self.output_log.append("[# queries]: {}".format(len(answer)))
        print("[Runtime so far] {}".format(self.zero_step_total_time))
        self.output_log.append("[Runtime so far] {}".format(self.zero_step_total_time))
        print("[Memory footprint] {}".format(using("profile")))
        self.output_log.append("[Memory footprint] {}".format(using("profile")))
        print("[Count candidate queries] {}".format(self.count_candidate_queries))
        self.output_log.append("[Count candidate queries] {}".format(self.count_candidate_queries))
        print("[Count predictions] {}".format(self.count_predictions))
        self.output_log.append("[Count predictions] {}".format(self.count_predictions))

        # Sampling-based active learning: sample 100 queries and 100 unlabeled trajecotries.
        while len(self.labeled_index) < self.budget and len(answer) > 0:
            print("[Step {}]".format(len(self.labeled_index) - init_nlabels + 1))
            self.output_log.append("[Step {}]".format(len(self.labeled_index) - init_nlabels + 1))
            if len(answer) > 100:
                sampled_answer = random.sample(answer, 100)
            else:
                sampled_answer = answer

            sampled_unlabeled_trajectories = []
            for i in range(len(self.inputs)):
                if i not in self.labeled_index:
                    sampled_unlabeled_trajectories.append(i)
            sampled_unlabeled_trajectories = random.sample(sampled_unlabeled_trajectories, 100)
            _start_pick_next_sample = time.time()
            j_values = np.zeros(len(sampled_unlabeled_trajectories))
            # prediction_matrix = []
            for i in range(len(sampled_unlabeled_trajectories)):
                input = self.inputs[i]
                label = self.labels[i]
                memoize = self.memoize_all_inputs[i]
                pred_per_input = []
                for query in sampled_answer:
                    result, new_memoize = query.execute(input, -1, memoize, {})
                    self.count_predictions += 1
                    self.memoize_all_inputs[i].update(new_memoize)
                    pred_per_input.append(int(result[0, len(input[0])] > 0))
                # prediction_matrix.append(pred_per_input)
                hist, bin_edges = np.histogram(pred_per_input, bins=2)
                prob_i = hist/np.sum(hist)
                j_values[i] = np.sum(prob_i * (1 - prob_i))
            next_idx = np.argmax(j_values)
            self.sample_selection_time += time.time() - _start_pick_next_sample
            if j_values[next_idx] == 0:
                # prediction_matrix = np.array(prediction_matrix)
                for query in answer:
                    print("answer", print_program(query))
                    self.output_log.append(print_program(query))
                print("[# queries]: {}".format(len(answer)))
                self.output_log.append("[# queries]: {}".format(len(answer)))
                print("[Runtime so far] {}".format(time.time() - _start_total_time))
                self.output_log.append("[Runtime so far] {}".format(time.time() - _start_total_time))
                print("[Memory footprint] {}".format(using("profile")))
                self.output_log.append("[Memory footprint] {}".format(using("profile")))
                print("No more uncertainty")
                self.output_log.append("No more uncertainty")
                break
            self.labeled_index.append(next_idx)
            print("# labeled segments", len(self.labeled_index))
            print("# positive: {}, # negative: {}".format(sum(self.labels[self.labeled_index]), len(self.labels[self.labeled_index]) - sum(self.labels[self.labeled_index])))
            # Prune inconsistent queries
            _start_prune_inconsistent_queries_active_learing = time.time()
            updated_answer = []
            input = self.inputs[next_idx]
            label = self.labels[next_idx]
            memoize = self.memoize_all_inputs[next_idx]
            for query in answer:
                result, new_memoize = query.execute(input, -1, memoize, {})
                self.count_predictions += 1
                self.memoize_all_inputs[next_idx].update(new_memoize)
                if (result[0, len(input[0])] > 0) == label:
                    updated_answer.append(query)
            answer = updated_answer
            print("Number of queries after pruning inconsistent queries:", len(answer))
            for query in answer:
                print("answer", print_program(query))
                self.output_log.append(print_program(query))
            self.prune_inconsistent_queries_active_learing_time += time.time() - _start_prune_inconsistent_queries_active_learing
            print("[# queries]: {}".format(len(answer)))
            self.output_log.append("[# queries]: {}".format(len(answer)))
            print("[Runtime so far] {}".format(time.time() - _start_total_time))
            self.output_log.append("[Runtime so far] {}".format(time.time() - _start_total_time))
            print("[Memory footprint] {}".format(using("profile")))
            self.output_log.append("[Memory footprint] {}".format(using("profile")))
            print("[Count candidate queries] {}".format(self.count_candidate_queries))
            self.output_log.append("[Count candidate queries] {}".format(self.count_candidate_queries))
            print("[Count predictions] {}".format(self.count_predictions))
            self.output_log.append("[Count predictions] {}".format(self.count_predictions))

        total_time = time.time() - _start_total_time
        total_time_log = "[Runtime] enumerate all possible answers time: {}, prune partial query time: {}, prune parameter values time: {}, find children time: {}, other time: {}, zero steps total time: {}, sample selection time: {}, pruning in active learning time: {}, total time: {}".format(self.enumerate_all_possible_answers_time, self.prune_partial_query_time, self.prune_parameter_values_time, self.find_children_time, self.other_time, self.zero_step_total_time, self.sample_selection_time, self.prune_inconsistent_queries_active_learing_time, total_time)
        print(total_time_log)
        self.output_log.append(total_time_log)
        print("[Count candidate queries] {}".format(self.count_candidate_queries))
        self.output_log.append("[Count candidate queries] {}".format(self.count_candidate_queries))
        print("[Count predictions] {}".format(self.count_predictions))
        self.output_log.append("[Count predictions] {}".format(self.count_predictions))
        return self.output_log

    def zero_steps(self):
        _start_total_time_zero_step = time.time()

        # Initialize program graph
        query_graph = QueryGraph(self.max_npred, self.max_depth, self.max_nontrivial, self.max_trivial, self.max_duration, 2, self.predicate_list, True, topdown_or_bottomup='topdown')

        queue = deque()
        queue.append(query_graph)
        self.consistent_queries = []
        while queue:
            _start_others = time.time()
            current_query_graph = queue.pop()
            current_query = current_query_graph.program
            self.count_candidate_queries += 1
            print("current_query", print_program(current_query), get_depth_and_npred(current_query))
            print("[Memory footprint] {}".format(using("profile")))
            self.other_time += time.time() - _start_others
            # IF: overapproximation doesn't match, prune Q
            _start_prune_partial_query = time.time()
            overapproximation_for_all_inputs = True
            for i in range(len(self.inputs)):
                if i not in self.labeled_index:
                    continue
                input = self.inputs[i]
                label = self.labels[i]
                # NOTE: additional pruning?
                memoize = self.memoize_all_inputs[i]
                result, new_memoize = current_query.execute(input, label, memoize, {})
                self.count_predictions += 1
                self.memoize_all_inputs[i].update(new_memoize)
                if (result[0, len(input[0])] > 0) != label:
                    overapproximation_for_all_inputs = False
                    print("pruned")
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
                all_children = current_query_graph.get_all_children(with_parameter_hole=False, with_kleene=self.with_kleene) # Considering only conjunction and sequencing
                # all_children = current_query_graph.get_all_children(with_parameter_hole=False, with_kleene=True) # Considering conjunction, sequencing, and Kleene star
                queue.extend(all_children)
                self.find_children_time += time.time() - _start_find_children

        # RETURN: the list. Need to enumerate over all possible queries to omit ones that are inconsistenet with W.
        answer = []
        print("candidate queries size", len(self.consistent_queries))
        # return answer
        _start = time.time()
        for candidate_query in self.consistent_queries:
            matched = True
            for i in range(len(self.inputs)):
                if i not in self.labeled_index:
                    continue
                input = self.inputs[i]
                label = self.labels[i]

                memoize = self.memoize_all_inputs[i]
                result, new_memoize = candidate_query.execute(input, label, memoize, {})
                self.count_predictions += 1
                self.memoize_all_inputs[i].update(new_memoize)
                if (result[0, len(input[0])] > 0) != label:
                    matched = False
                    break
            if matched:
                answer.append(candidate_query)
        self.enumerate_all_possible_answers_time += time.time() - _start
        self.zero_step_total_time += time.time() - _start_total_time_zero_step
        return answer

    def prune_parameter_values(self, current_query):
        # current_query is sketch (containing only parameter holes)
        # 1. Find all parameter holes
        # 2. For each parameter hole, fix it and fill in other holes with -inf or inf.
        if QueryGraph.is_complete(current_query):
            self.consistent_queries.append(current_query)
            return
        queue = deque()
        queue.append(current_query)
        while queue:
            current = queue.pop()
            for submod, functionclass in current.submodules.items():
                if isinstance(functionclass, dsl.ParameterHole):
                    orig_fclass = copy.deepcopy(current.submodules[submod])
                    value_range = functionclass.get_value_range()
                    # step = functionclass.get_step()
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
        # for i, (input, label) in enumerate(zip(self.inputs[self.labeled_index], self.labels[self.labeled_index])):
        for i in range(len(self.inputs)):
            if i not in self.labeled_index:
                continue
            input = self.inputs[i]
            label = self.labels[i]
            memoize = self.memoize_all_inputs[i]
            result, new_memoize = new_query.execute(input, label, memoize, {})
            self.count_predictions += 1
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
            self.consistent_queries.append(current_query)
            return
        queue = deque()
        queue.append(current_query)
        while queue:
            current = queue.pop()
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
                        self.count_candidate_queries += 1
                        if not QueryGraph.is_sketch(new_query):
                            raise ValueError
                            # print("here1")
                        self.fill_all_parameter_holes(new_query)
                    current.submodules[submod] = orig_fclass
                    return
                else:
                    #add submodules
                    queue.append(functionclass)

    def enumerate_search_space(self):
        self.query_count = 0
        # Initialize program graph
        query_graph = QueryGraph(self.max_npred, self.max_depth, self.max_nontrivial, self.max_trivial, self.max_duration, 2, self.predicate_list, True, topdown_or_bottomup='topdown')

        queue = deque()
        queue.append(query_graph)
        self.consistent_queries = []
        while queue:
            current_query_graph = queue.pop()
            current_query = current_query_graph.program
            # print("current_query", print_program(current_query))
            # self.output_log.append("current_query: {}".format(print_program(current_query)))
            self.query_count += 1
            # ELSE IF: Q is sketch, computes all parameters for Q that are consistent with inputs, and adds them to the final list.
            if current_query_graph.is_sketch(current_query):
                if not current_query_graph.is_complete(current_query):
                    self.count_all_parameter_holes(current_query)
            # ELSE: adds all the children of Q to queue.
            else:
                all_children = current_query_graph.get_all_children(with_parameter_hole=False, with_kleene=self.with_kleene) # Considering only conjunction and sequencing
                queue.extend(all_children)
        print("[Query count]", self.query_count)
        self.output_log.append("[Query count] {}".format(self.query_count))
        return self.output_log

    def count_all_parameter_holes(self, current_query):
        if QueryGraph.is_complete(current_query):
            return
        queue = deque()
        queue.append(current_query)
        while queue:
            current = queue.pop()
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
                        # replace the parameter hole with a predicate
                        current.submodules[submod] = predicate
                        # create the correct child node
                        new_query = copy.deepcopy(current_query)
                        self.query_count += 1
                        # print("fill parameters query", print_program(new_query))
                        # self.output_log.append("fill parameters query: {}".format(print_program(new_query)))
                        if not QueryGraph.is_sketch(new_query):
                            raise ValueError
                            # print("here1")
                        self.count_all_parameter_holes(new_query)
                    current.submodules[submod] = orig_fclass
                    return
                else:
                    #add submodules
                    queue.append(functionclass)