"""
User initializes hard constraints (e.g., must have a red cube)

labeled_data = [10 positive and negative examples]
candidates_list = [("red cube" and ??, score)]

// Generate K1 candidates from the initial query
while len(candidates_list) < K1:
    Compute all the children of Q.
    For each child Q':
        compute the score of Q' based on FN lower bound (using overapproximation), FP lower bound (using underapproximation), and structure (or depth) of query. Add (Q', score) to candidates_list

while len(labeled_data) < N:
    // Pick next data to label
    For each top K1 query in candidates_list:
        compute the weight of each query, and select the next video segment to label based on (weighted) disagreement among the top K1 queries.
        add the selected video segment to labeled_data.
    // Update scores
    For each top K2 query Q' in candidates_list:
        Update the score of Q'.
    // Expand search space
    Find the query Q with the highest score in candidates_list that is incomplete.
    Find all the children of Q.
    For each child Q':
        if Q' is a sketch:
            compute all parameters for Q' that are at least x% consistent with W (for each parameter in Q', compute its largest/smallest possible value when overapproximating all other parameters).
            add them along with their scores to candidates_list
        else:
            add (Q', score) to candidates_list
"""

from utils import print_program, str_to_program
from mimetypes import init
from query_graph import QueryGraph
import dsl
import copy
import numpy as np
import time
from sklearn.metrics import f1_score
from scipy import stats

class VOCAL:

    def __init__(self, inputs, labels, max_num_programs=100, max_num_atomic_predicates=5,
        max_depth=2, k1=10, k2=20, budget=100, thresh=0.5):
        self.max_num_programs = max_num_programs
        self.max_num_atomic_predicates = max_num_atomic_predicates
        self.max_depth = max_depth

        self.enumerate_all_candidates_time = 0
        self.prune_partial_query_time = 0
        self.prune_parameter_values_time = 0
        self.find_children_time = 0

        self.k1 = k1
        self.k2 = k2
        self.budget = budget
        self.thresh = thresh

        self.inputs = np.array(inputs, dtype=object)
        self.labels = np.array(labels, dtype=object)

    def run(self, init_labeled_index):
        self.labeled_index = init_labeled_index
        self.memoize_all_inputs = [{} for _ in range(len(self.inputs))] # For each (input, label) pair, memoize the results of all sub-queries encountered during synthesis.

        # Initialize program graph
        query_graph = QueryGraph(self.max_num_atomic_predicates, self.max_depth)

        queue = [query_graph]
        self.consistent_queries = []

        score = self.compute_query_score(query_graph.program)
        self.candidate_list = [[query_graph, score]]
        print("compute init query score")
        print(print_program(query_graph.program), score)
        print("compute init k queries scores")
        while len(self.candidate_list) < self.k1:
            current_query_graph, _ = self.candidate_list.pop(0)
            current_query = current_query_graph.program
            all_children = current_query_graph.get_all_children("vocal")
            for child in all_children:
                score = self.compute_query_score(child.program)
                self.candidate_list.append([child, score])
                print(print_program(child.program), score)

        while len(queue) != 0 and len(self.labeled_index) < self.budget:
            print("number of labeled data", len(self.labeled_index))
            sorted_candidate_list = sorted(self.candidate_list, key=lambda x: x[1], reverse=True)[:self.k1]
            video_segment_id = self.pick_next_segment(sorted_candidate_list)
            print("pick next segment", video_segment_id)
            self.labeled_index.append(video_segment_id)
            # assert labeled_index does not contain duplicates
            assert(len(self.labeled_index) == len(set(self.labeled_index)))

            sorted_idx = [i[0] for i in sorted(enumerate(self.candidate_list), key=lambda x: x[1][1], reverse=True)][:self.k2]

            for i in sorted_idx:
                self.candidate_list[i][1] = self.compute_query_score(self.candidate_list[i][0].program)

            # Expand search space
            sorted_idx = [i[0] for i in sorted(enumerate(self.candidate_list), key=lambda x: x[1][1], reverse=True)][:self.k2]
            for i in sorted_idx:
                current_query_graph = self.candidate_list[i][0]
                current_query = current_query_graph.program
                if not QueryGraph.is_complete(current_query):
                    print("expand search space", print_program(current_query))
                    del self.candidate_list[i]
                    break
            all_children = current_query_graph.get_all_children("vocal")
            for child in all_children:
                if QueryGraph.is_sketch(child.program):
                    self.fill_all_parameter_holes(child.program)
                    # self.prune_parameter_values(child.program)
                else:
                    score = self.compute_query_score(child.program)
                    self.candidate_list.append([child, score])
            print("end of iteration", sum(self.labels[self.labeled_index]), len(self.labeled_index)-sum(self.labels[self.labeled_index]))
            print("candidate list")
            for graph, score in self.candidate_list:
                print(print_program(graph.program), score)
            print("top k queries")
            n_answer = 10
            answer = []
            sorted_candidate_list = sorted(self.candidate_list, key=lambda x: x[1], reverse=True)
            for query_graph, score in sorted_candidate_list:
                if QueryGraph.is_complete(query_graph.program):
                    answer.append([query_graph.program, score])
                    if len(answer) == n_answer:
                        break
            for q, s in answer:
                print(print_program(q), s)

        # RETURN: the list.
        n_answer = 10
        answer = []
        sorted_candidate_list = sorted(self.candidate_list, key=lambda x: x[1], reverse=True)
        for query_graph, score in sorted_candidate_list:
            if QueryGraph.is_complete(query_graph.program):
                answer.append(query_graph.program)
                if len(answer) == n_answer:
                    break
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
                # TODO: should be query graph instead of query
                query_graph = QueryGraph(self.max_num_atomic_predicates, self.max_depth)
                query_graph.program = current_query
                print("add complete query", print_program(current_query))
                self.candidate_list.append([query_graph, score])
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
        # for i, (input, label) in enumerate(zip(self.inputs, self.labels)):
        for i in self.labeled_index:
            input = self.inputs[i]
            label = self.labels[i]
            memoize = self.memoize_all_inputs[i]
            result, memoize = current_query.execute(input, label, memoize)
            y_pred.append(int(result[0, len(input[0])] > 0))
            self.memoize_all_inputs[i] = memoize
        # print(self.labels[self.labeled_index], y_pred)
        score = f1_score(list(self.labels[self.labeled_index]), y_pred)
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