"""
candidates_list = [(??, 1, 0, 0)]  // (query_graph, f1_score, npred, depth)
answer_list = []

while candidates_list not empty:
    new_candidates_list = []
    for Q, s in candidates_list:
        Compute all the children of Q and their scores.
        if child is a sketch:
            Compute all parameters for Q and their scores.
            Add them to answer_list
        else:
            Add (child, score, npred, depth) to new_candidates_list.
    Keep the top k queries based on F1 score in new_candidates_list and prune the rest.
    candidates_list = new_candidates_list

for Q, s in answer_list:
    Recompute f1 score for Q.

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
        max_depth=3, k=32):
        self.max_num_programs = max_num_programs
        self.max_num_atomic_predicates = max_num_atomic_predicates
        self.max_depth = max_depth

        self.update_answer_list_score_time = 0
        self.fill_parameter_holes_time = 0
        self.compute_score_time = 0
        self.k = k

    def run(self, inputs, labels):
        _start_total_time = time.time()
        self.inputs = inputs
        self.labels = labels
        self.memoize_all_inputs = [{} for _ in range(len(inputs))] # For each (input, label) pair, memoize the results of all sub-queries encountered during synthesis.

        # Initialize program graph
        query_graph = QueryGraph(self.max_num_atomic_predicates, self.max_depth)

        candidate_list = [[query_graph, 1, 0, 0]]
        self.consistent_queries = []
        while len(candidate_list) > 0:
            new_candidate_list = []
            for current_query_graph, score, _, _ in candidate_list:
                current_query = current_query_graph.program
                all_children = current_query_graph.get_all_children("vocal")
                for child in all_children:
                    if QueryGraph.is_sketch(child.program):
                        _start_fill_parameter_holes = time.time()
                        print("sketch query", utils.print_program(child.program))
                        self.fill_all_parameter_holes(child.program) # not pruning parameter values
                        # self.prune_parameter_values(current_query, inputs, labels)
                        self.fill_parameter_holes_time += time.time() - _start_fill_parameter_holes
                    else:
                        _start_compute_score = time.time()
                        score = self.compute_query_score(child.program)
                        new_candidate_list.append([child, score, child.num_atomic_predicates, child.depth])
                        self.compute_score_time += time.time() - _start_compute_score
            candidate_list = sorted(new_candidate_list, key=lambda x: (-x[1], -x[2], x[3]))[:self.k]
            print("candidate list count", len(candidate_list))
            for q, s, npred, depth in candidate_list:
                print("candidate query", utils.print_program(q.program), s, npred, depth)


        # RETURN: the list. Need to enumerate over all possible queries to omit ones that are inconsistenet with W.
        answer = []
        # return answer
        _start = time.time()
        for current_query, score in self.consistent_queries:
            # print("complete query", utils.print_program(current_query), score)
            score = self.compute_query_score(current_query)
            answer.append([current_query, score])
        self.update_answer_list_score_time += time.time() - _start
        answer = sorted(answer, key=lambda x: x[1], reverse=True)
        print("[Runtime] update answer list score time: {}, fill parameter holes time: {}, compute score time: {}, total time: {}".format(self.update_answer_list_score_time, self.fill_parameter_holes_time, self.compute_score_time, time.time() - _start_total_time))
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
            self.consistent_queries.append([current_query, score])
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