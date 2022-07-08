"""
queue = [??]
candidates_list = []
answer_list = []

while queue not empty:
    Q = queue.pop()
    score = compute_f1_score(Q) // using over-/under-approximation
    if score < 0.5:
        continue
    else if Q is a sketch:
        Computes all parameters for Q that give f1 score > 0.5 over the inputs (for each parameter in Q,
        compute its largest/smallest possible value when over-/under-approximating all other parameters).
        Add them to candidates_list
    else:
        Add all the children of Q to queue

for Q in candidates_list:
    if compute_f1_score(Q) > 0.5, add it to answer_list

return sorted answer_list
"""

import utils
from mimetypes import init
from query_graph import QueryGraph
import dsl
import copy
import numpy as np
import time
from sklearn.metrics import f1_score
from scipy import stats

class QUIVRSoft:

    def __init__(self, max_num_programs=100, max_num_atomic_predicates=5,
        max_depth=2, k1=10, k2=20, thresh=0.5):
        self.max_num_programs = max_num_programs
        self.max_num_atomic_predicates = max_num_atomic_predicates
        self.max_depth = max_depth

        self.enumerate_all_candidates_time = 0
        self.prune_partial_query_time = 0
        self.prune_parameter_values_time = 0
        self.find_children_time = 0

        self.k1 = k1
        self.k2 = k2
        self.thresh = thresh

    def run(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels
        self.memoize_all_inputs = [{} for _ in range(len(inputs))] # For each (input, label) pair, memoize the results of all sub-queries encountered during synthesis.

        # Initialize program graph
        query_graph = QueryGraph(self.max_num_atomic_predicates, self.max_depth)

        queue = [query_graph]
        self.consistent_queries = []
        while len(queue) != 0:
            current_query_graph = queue.pop(0) # Pop the last element
            current_query = current_query_graph.program
            # print("current_query", utils.print_program(current_query))
            # IF: overapproximation doesn't match, prune Q
            _start_prune_partial_query = time.time()
            score = self.compute_query_score(current_query)
            if score < self.thresh:
                continue
            self.prune_partial_query_time += time.time() - _start_prune_partial_query
            # ELSE IF: Q is sketch, computes all parameters for Q that are consistent with inputs, and adds them to the final list.
            if current_query_graph.is_sketch(current_query):
                _start_prune_parameter_values = time.time()
                print("current_query", utils.print_program(current_query))
                self.fill_all_parameter_holes(current_query) # not pruning parameter values
                # self.prune_parameter_values(current_query, inputs, labels)
                self.prune_parameter_values_time += time.time() - _start_prune_parameter_values
            # ELSE: adds all the children of Q to queue.
            else:
                _start_find_children = time.time()
                all_children = current_query_graph.get_all_children("vocal")
                queue.extend(all_children)
                self.find_children_time += time.time() - _start_find_children

        # RETURN: the list. Need to enumerate over all possible queries to omit ones that are inconsistenet with W.
        answer = []
        print("candidate queries size", len(self.consistent_queries))
        # return answer
        _start = time.time()
        for candidate_query in self.consistent_queries:
            print("candidate query", utils.print_program(candidate_query))
            score = self.compute_query_score(candidate_query)
            answer.append([candidate_query, score])
        self.enumerate_all_candidates_time += time.time() - _start
        print("[Runtime] enumerate all candidates time: {}, prune partial query time: {}, prune parameter values time: {}, find children time: {}".format(self.enumerate_all_candidates_time, self.prune_partial_query_time, self.prune_parameter_values_time, self.find_children_time))
        print(self.memoize_all_inputs[0].keys())
        answer = sorted(answer, key=lambda x: x[1], reverse=True)
        return answer


    def prune_parameter_values(self, current_query, inputs, labels):
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
                    value_interval = self.fill_other_parameter_holes(new_query, inputs, labels, value_range)
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

    def fill_other_parameter_holes(self, new_query, inputs, labels, value_range):
        theta_lb = value_range[0]
        theta_ub = value_range[1]
        for i, (input, label) in enumerate(zip(inputs, labels)):
            memoize = self.memoize_all_inputs[i]
            result, memoize = new_query.execute(input, label, memoize)
            self.memoize_all_inputs[i] = memoize
            if label == 1:
                theta_ub = min(result[0, len(input[0])], theta_ub)
            if label == 0:
                theta_lb = max(result[0, len(input[0])], theta_lb)
            if theta_lb > theta_ub:
                break
        return [theta_lb, theta_ub]

    def fill_all_parameter_holes(self, current_query):
        if QueryGraph.is_complete(current_query):
            score = self.compute_query_score(current_query)
            if score > self.thresh:
                self.consistent_queries.append(current_query)
        queue = [current_query]
        while len(queue) != 0:
            current = queue.pop(0)
            for submod, functionclass in current.submodules.items():
                if isinstance(functionclass, dsl.PredicateHole):
                    raise ValueError
                if issubclass(type(functionclass), dsl.Predicate):
                    continue
                if isinstance(functionclass, dsl.ParameterHole):
                    assert(isinstance(functionclass.predicate, dsl.MinLength))
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

    def compute_query_score(self, current_query):
        y_pred = []
        for i, (input, label) in enumerate(zip(self.inputs, self.labels)):
            memoize = self.memoize_all_inputs[i]
            result, memoize = current_query.execute(input, label, memoize)
            y_pred.append(int(result[0, len(input[0])] > 0))
            self.memoize_all_inputs[i] = memoize
        # print(self.labels[self.labeled_index], y_pred)
        score = f1_score(list(self.labels), y_pred)
        return score

    def pick_next_segment(self, sorted_candidate_list):
        tuning_par = 1
        true_labels = np.array(self.labels)
        prediction_matrix = []
        _start = time.time()
        for i in range(len(self.inputs)):
            # print("prediction matrix", i)
            input = self.inputs[i]
            label = self.labels[i]
            memoize = self.memoize_all_inputs[i]
            pred_per_input = []
            for query_graph, _ in sorted_candidate_list:
                query = query_graph.program
                result, memoize = query.execute(input, label, memoize)
                pred_per_input.append(int(result[0, len(input[0])] > 0))
            prediction_matrix.append(pred_per_input)
        prediction_matrix = np.array(prediction_matrix)
        print("constructing prediction matrix", time.time()-_start)
        n_instances = len(true_labels)
        k = self.k1

        # Initialize
        loss_t = np.zeros(k)
        for i in range(n_instances):
            if i not in self.labeled_index:
                continue
            if true_labels[i]:
                loss_t += (np.array((prediction_matrix[i, :] != 1) * 1) * 5)
                # loss_t += (np.array((prediction_matrix_pair_level[t, :] != 1) * 1) / u_t * (n_pos + n_neg) / (2 * n_pos))
                # n_pos += 1
            else:
                loss_t += (np.array((prediction_matrix[i, :] != 0) * 1))

        eta = np.sqrt(np.log(k)/(2*(len(self.labeled_index)+1)))

        posterior_t = np.exp(-eta * (loss_t-np.min(loss_t)))
        # Note that above equation is equivalent to np.exp(-eta * loss_t).
        # `-np.min(loss_t)` is applied only to avoid entries being near zero for large eta*loss_t values before the normalization
        posterior_t  /= np.sum(posterior_t)  # normalized weight

        entropy_list = np.zeros(n_instances)
        for i in range(n_instances):
            if i in self.labeled_index:
                entropy_list[i] = -1
            else:
                # Measure the normalized entropy of the incoming data
                hist, bin_edges = np.histogram(prediction_matrix[i, :], bins=2)
                prob_i = hist/np.sum(hist)
                entropy_i = stats.entropy(prob_i, base=2) * tuning_par
                # Check if the normalized entropy is greater than 1
                if entropy_i > 1:
                    entropy_i = 1
                if entropy_i < 0:
                    entropy_i = 0
                entropy_list[i] = entropy_i
        # find argmax of entropy
        max_entropy_index = np.argmax(entropy_list)
        return max_entropy_index