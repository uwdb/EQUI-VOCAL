import copy
import itertools
from collections import deque
import src.dsl as dsl
from src.utils import print_program, program_to_dsl, dsl_to_program, get_depth_and_npred

class QueryGraph(object):

    def __init__(self, dataset_name, max_npred, max_depth, max_nontrivial, max_trivial, max_duration, max_vars, predicate_list, is_trajectory,depth = 0, npred = 0, topdown_or_bottomup="bottomup"):
        self.dataset_name = dataset_name
        self.max_npred = max_npred
        self.max_depth = max_depth
        self.max_duration = max_duration
        self.max_nontrivial = max_nontrivial
        self.max_trivial = max_trivial
        self.predicate_list = predicate_list
        if topdown_or_bottomup == "bottomup":
            self.program = None
        elif topdown_or_bottomup == "topdown":
            self.program = dsl.StartOperator()
        else:
            raise ValueError("Unknown algorithm:", topdown_or_bottomup)
        if dataset_name.startswith("user_study"):
            self.duration_unit = 25
        else:
            self.duration_unit = 5

        #armadillo
        self.depth = depth #0
        self.npred = npred #0 # npred = n_trivial + n_nontrivial
        self.n_trivial = 0 # number of <true>* predicates
        self.n_nontrivial =0
        self.variables = ["o{}".format(i) for i in range(max_vars)]
        self.is_trajectory = is_trajectory

    def get_parameter_holes_and_value_space(self):
        parameter_holes = []
        value_space = []
        steps = []
        queue = deque()
        queue.append(self.program)
        while queue:
            current = queue.pop()
            # (key, value) pair
            for submod, functionclass in current.submodules.items():
                if isinstance(functionclass, dsl.PredicateHole):
                    continue
                if issubclass(type(functionclass), dsl.Predicate):
                    continue
                if isinstance(functionclass, dsl.ParameterHole):
                    parameter_holes.append(functionclass)
                    value_space.append(parameter_holes.get_value_range())
                    steps.append(parameter_holes.get_step())
                else:
                    #add submodules
                    queue.append(functionclass)
        return parameter_holes, value_space, steps

    @classmethod
    def is_sketch(cls, candidate_query):
        # print("candidate_query", candidate_query)
        # A partial query Q is a sketch if it has no predicate holes
        queue = deque()
        queue.append(candidate_query)
        while queue:
            current_function = queue.pop()
            if isinstance(current_function, dsl.PredicateHole):
                return False
            if isinstance(current_function, dsl.ParameterHole):
                continue
            if issubclass(type(current_function), dsl.Predicate):
                continue
            else:
                for submodule_name in current_function.submodules:
                    queue.append(current_function.submodules[submodule_name])
        return True

    @classmethod
    def is_complete(cls, candidate_query):
        # A partial query Q is a sketch if it has no holes
        queue = deque()
        queue.append(candidate_query)
        while queue:
            current_function = queue.pop()
            if issubclass(type(current_function), dsl.Hole):
                return False
            if issubclass(type(current_function), dsl.Predicate):
                continue
            else:
                for submodule_name in current_function.submodules:
                    queue.append(current_function.submodules[submodule_name])
        return True


    def get_all_children(self, with_parameter_hole, with_kleene):
        """ Each child of Q is obtained by filling in one predicate hole using an applicable operator.
        """
        # Fill the predicate hole that is found first (BFS).
        all_children = []
        queue = deque()
        queue.append(self.program)
        while queue:
            current = queue.pop()
            for submod, functionclass in current.submodules.items():
                if isinstance(functionclass, dsl.ParameterHole):
                    continue
                if issubclass(type(functionclass), dsl.Predicate):
                    continue
                if isinstance(functionclass, dsl.PredicateHole):
                    if with_kleene:
                        replacement_candidates = self.construct_candidates(current, submod, with_parameter_hole)
                    else:
                        replacement_candidates = self.construct_candidates_no_kleene(current, submod, with_parameter_hole)
                    orig_fclass = copy.deepcopy(current.submodules[submod])
                    for child_candidate in replacement_candidates:
                        # replace the predicate hole with a candidate
                        current.submodules[submod] = child_candidate
                        # create the correct child node
                        new_query_graph = copy.deepcopy(self)
                        new_query_graph.depth, new_query_graph.n_nontrivial, new_query_graph.n_trivial = get_depth_and_npred(new_query_graph.program)
                        new_query_graph.npred = new_query_graph.n_nontrivial + new_query_graph.n_trivial
                        # check if child program can be completed within max_depth
                        if new_query_graph.depth > new_query_graph.max_depth or new_query_graph.npred > new_query_graph.max_npred or (new_query_graph.max_nontrivial and new_query_graph.n_nontrivial > new_query_graph.max_nontrivial) or (new_query_graph.max_trivial and new_query_graph.n_trivial > new_query_graph.max_trivial):
                            continue
                        # if yes, compute costs and add to list of children
                        all_children.append(new_query_graph)
                    # once we've copied it, set current back to the original current
                    current.submodules[submod] = orig_fclass
                    return all_children
                else:
                    #add submodules
                    queue.append(functionclass)
        return all_children

    def construct_candidates(self, parent_functionclass, submod, has_paramter_hole):
        candidates = []
        if isinstance(parent_functionclass, dsl.SequencingOperator):
            if submod == "function1":
                replacement_candidates = [dsl.ConjunctionOperator, dsl.SequencingOperator, dsl.TrueStar] + list(self.predicate_list.keys())
            else:
                # Remove semantically equivalent duplicates: associativity of sequencing
                replacement_candidates = [dsl.ConjunctionOperator, dsl.TrueStar] + list(self.predicate_list.keys())
        elif isinstance(parent_functionclass, dsl.ConjunctionOperator):
            if submod == "function1":
                replacement_candidates = [dsl.ConjunctionOperator, dsl.SequencingOperator] + list(self.predicate_list.keys())
            else:
                left_child = parent_functionclass.submodules["function1"]
                # remove semantically equivalent duplicates: associativity of conjunction and commutativity of conjunction
                if isinstance(left_child, dsl.ConjunctionOperator) or isinstance(left_child, dsl.SequencingOperator):
                    replacement_candidates = [dsl.SequencingOperator] + list(self.predicate_list.keys())
                elif isinstance(left_child, dsl.KleeneOperator):
                    left_child = left_child.submodules["kleene"]
                    replacement_candidates = []
                    for pred in self.predicate_list.keys():
                        if pred() > left_child:
                            replacement_candidates.append(pred)
                elif isinstance(left_child, dsl.ParameterHole):
                    replacement_candidates = []
                    for pred in self.predicate_list.keys():
                        if pred() > left_child.get_predicate():
                            replacement_candidates.append(pred)
                elif issubclass(type(left_child), dsl.Predicate):
                    replacement_candidates = []
                    for pred in self.predicate_list.keys():
                        if pred() > left_child:
                            replacement_candidates.append(pred)
                else:
                    raise ValueError("left_child: {}".format(left_child))
        else: # Start()
            replacement_candidates = [dsl.ConjunctionOperator, dsl.SequencingOperator, dsl.TrueStar] + list(self.predicate_list.keys())
        for functionclass in replacement_candidates:
            if issubclass(type(functionclass()), dsl.Predicate):
                if has_paramter_hole and functionclass.has_theta:
                    candidate = dsl.ParameterHole(functionclass())
                    candidates.append(candidate)
                elif not has_paramter_hole and isinstance(functionclass(), dsl.MinLength):
                    candidate = dsl.ParameterHole(functionclass(max_duration=self.max_duration))
                    candidates.append(candidate)
                else:
                    # candidate = functionclass()
                    # candidates.append(candidate)
                    if isinstance(functionclass(), dsl.TrueStar):
                        candidate_with_kleene = functionclass()
                    else:
                        candidate_with_kleene = dsl.KleeneOperator(functionclass())
                    candidates.append(candidate_with_kleene)
            else:
                candidate = functionclass()
                candidates.append(candidate)
        return candidates


    def construct_candidates_no_kleene(self, parent_functionclass, submod, has_paramter_hole):
        candidates = []
        if isinstance(parent_functionclass, dsl.SequencingOperator):
            if submod == "function1":
                replacement_candidates = [dsl.ConjunctionOperator, dsl.SequencingOperator, dsl.TrueStar] + list(self.predicate_list.keys())
            else:
                # Remove semantically equivalent duplicates: associativity of sequencing
                replacement_candidates = [dsl.ConjunctionOperator, dsl.TrueStar] + list(self.predicate_list.keys())
        elif isinstance(parent_functionclass, dsl.ConjunctionOperator):
            if submod == "function1":
                replacement_candidates = [dsl.ConjunctionOperator, dsl.SequencingOperator] + list(self.predicate_list.keys())
            else:
                left_child = parent_functionclass.submodules["function1"]
                # remove semantically equivalent duplicates: associativity of conjunction and commutativity of conjunction
                if isinstance(left_child, dsl.ConjunctionOperator) or isinstance(left_child, dsl.SequencingOperator):
                    replacement_candidates = [dsl.SequencingOperator] + list(self.predicate_list.keys())
                elif isinstance(left_child, dsl.ParameterHole):
                    replacement_candidates = []
                    for pred in self.predicate_list.keys():
                        if pred() > left_child.get_predicate():
                            replacement_candidates.append(pred)
                elif issubclass(type(left_child), dsl.Predicate):
                    replacement_candidates = []
                    for pred in self.predicate_list.keys():
                        if pred() > left_child:
                            replacement_candidates.append(pred)
                else:
                    raise ValueError
        else: # Start()
            replacement_candidates = [dsl.ConjunctionOperator, dsl.SequencingOperator, dsl.TrueStar] + list(self.predicate_list.keys())
        for functionclass in replacement_candidates:
            if issubclass(type(functionclass()), dsl.Predicate):
                if (has_paramter_hole and functionclass.has_theta):
                    candidate = dsl.ParameterHole(functionclass())
                # elif not has_paramter_hole and isinstance(functionclass(), dsl.MinLength):
                #     candidate = dsl.ParameterHole(functionclass(self.max_duration))
                else:
                    candidate = functionclass()
                candidates.append(candidate)
            else:
                candidate = functionclass()
                candidates.append(candidate)
        return candidates


    def get_all_children_bu(self):
        """
        [Not used] Bottom-up expansion
        Example query:
            q = True*; p11 ^ ... ^ p1i ^ d1; True*; p21 ^ ... ^ p2j ^ d2; True*
            (Base case, one scene graph: True*; p11 ^ ... ^ p1i ^ d1; True*)
        Verbose form:
            q = Seq(Seq(Seq(Seq(True*, Duration(Conj(Conj(p13, p12), p11), theta1)), True*), Duration(Conj(Conj(p23, p22), p21), theta2)), True*)

        Output: a list of all possible expanded queries of the current query graph. Each child is rewriten into the ordered format to avoid duplicates.
        """
        all_children = []
        # Action a: Scene graph construction: add a predicate to existing scene graph (i.e., the last scene graph in the sequence).
        # Require: the last scene graph must not have duration constraint.
        if self.npred + 1 <= self.max_npred:
            for pred in self.predicate_list:
                pred_instances = []
                if self.predicate_list[pred]:
                    for param in self.predicate_list[pred]:
                        pred_instances.append(pred(param))
                else:
                    pred_instances.append(pred())

                for pred_instance in pred_instances:
                    new_query_graph = copy.deepcopy(self)
                    # 1. Find the last scene graph g2 = q.submodules["function1"].submodules["function2"] // Duration(Conj(Conj(p23, p22), p21), theta2)
                    parent_graph = [new_query_graph.program.submodules["function1"], "function2"]
                    last_graph = new_query_graph.program.submodules["function1"].submodules["function2"]
                    # 2. If g2 has duration constraint, CONTINUE
                    # NOT TRUE: locate the scene graph only: n2 = g2.submodules["duration"] // n2 = Conj(Conj(p23, p22), p21)
                    if last_graph.name == "Duration":
                        continue
                        # parent_graph = [last_graph, "duration"]
                        # last_graph = last_graph.submodules["duration"]
                    # 3. Find p23, which is the leftmost child of n2. If the predicate p24 already exists in the scene graph, skip.
                    is_duplicate_predicate = False
                    while last_graph.name == "Conjunction":
                        if last_graph.submodules["function2"] >= pred_instance:
                            is_duplicate_predicate = True
                            break
                        parent_graph = [parent_graph[0].submodules[parent_graph[1]], "function1"]
                        last_graph = last_graph.submodules["function1"] # Leftmost child of last_graph
                    if is_duplicate_predicate:
                        continue
                    if last_graph >= pred_instance:
                        continue
                    # 4. Replace p23 with Conj(p24, p23)
                    parent_graph[0].submodules[parent_graph[1]] = dsl.ConjunctionOperator(pred_instance, last_graph)
                    new_query_graph.npred += 1
                    print("Action A: ", print_program(new_query_graph.program))
                    all_children.append(new_query_graph)

        # Action b: Sequence construction: add a new scene graph (which consists of one predicate) to the end of the sequence.
        # 1. q' = Seq(Seq(q, p31), True*)
        if self.npred + 1 <= self.max_npred and self.depth + 1 <= self.max_depth:
            for pred in self.predicate_list:
                pred_instances = []
                if self.predicate_list[pred]:
                    for param in self.predicate_list[pred]:
                        pred_instances.append(pred(param))
                else:
                    pred_instances.append(pred())

                for pred_instance in pred_instances:
                    new_query_graph = copy.deepcopy(self)
                    new_query_graph.program = dsl.SequencingOperator(dsl.SequencingOperator(new_query_graph.program, pred_instance), dsl.TrueStar())
                    new_query_graph.npred += 1
                    new_query_graph.depth += 1
                    print("Action B: ", print_program(new_query_graph.program))
                    all_children.append(new_query_graph)

        # Action c: Duration refinement: increase the duration of the last scene graph in the sequence
        new_query_graph = copy.deepcopy(self)
        # 1. Find the last scene graph g2 = q.submodules["function1"].submodules["function2"] // g2 = Duration(Conj(Conj(p23, p22), p21), theta2)
        last_graph = new_query_graph.program.submodules["function1"].submodules["function2"]
        # 2. If g2 has duration constraint, increment by 1: g2.theta += 1
        if last_graph.name == "Duration":
            if last_graph.theta < self.max_duration:
                last_graph.theta += 1
                all_children.append(new_query_graph)
                print("Action C: ", print_program(new_query_graph.program))
        # 3. Else, add a duration constraint: g2' = Duration(g2, 2)
        else:
            new_query_graph.program.submodules["function1"].submodules["function2"] = dsl.DurationOperator(last_graph, 2)
            all_children.append(new_query_graph)
            print("Action C: ", print_program(new_query_graph.program))

        return all_children


    def get_all_children_unrestricted_postgres(self):
        """
        predicate_list = [{"name": "Near", "parameters": [-1.05], "nargs": 2}, {...}, ...]
        Output: a list of all possible expanded queries of the current query graph. Each child is rewriten into the ordered format to avoid duplicates.
        """
        all_children = []
        # Action a: Scene graph construction: Add a predicate to an existing region graph
        if self.npred + 1 <= self.max_npred:
            pred_instances = []
            for pred in self.predicate_list:
                # {"name": "Near", "parameters": [-1.05], "nargs": 2}
                if pred["parameters"]:
                    for param in pred["parameters"]:
                        pred_instances.append({"name": pred["name"], "parameter": param, "nargs": pred["nargs"]})
                else:
                    pred_instances.append({"name": pred["name"], "parameter": None, "nargs": pred["nargs"]})
                if pred["nargs"] > len(self.variables):
                    raise ValueError("The predicate has more variables than the number of variables in the query.")
            for pred_instance in pred_instances:
                for scene_graph_idx, dict in enumerate(self.program):
                    scene_graph = dict["scene_graph"]
                    nvars = pred_instance["nargs"]
                    # Special case: for trajectory experiment only
                    if self.is_trajectory:
                        if nvars == 1:
                            if self.dataset_name == "warsaw":
                                variables_list = [["o0"], ["o1"]]
                            else:
                                # Special case: for CLEVRER trajectory experiment, single-varible predicates can only be applied to o0
                                variables_list = [["o0"]]
                        elif nvars == 2:
                            variables_list = [["o0", "o1"]]
                    # Gneral case:
                    else:
                        variables_list = itertools.permutations(self.variables, nvars)
                    for variables in variables_list:
                        is_duplicate_predicate = False
                        for p in scene_graph:
                            if p["predicate"] == pred_instance["name"] and set(p["variables"]) == set(variables):
                                is_duplicate_predicate = True
                                break
                        if is_duplicate_predicate:
                            continue
                        new_query = copy.deepcopy(self)
                        new_query.program[scene_graph_idx]["scene_graph"].append({"predicate": pred_instance["name"], "parameter": pred_instance["parameter"], "variables": list(variables)})
                        new_query.npred += 1
                        print("Action A: ", program_to_dsl(new_query.program, not self.is_trajectory))
                        all_children.append(new_query)

        # Action b: Sequence construction: insert a new region graph consisting of one predicate into any position of the existing sequence of region graphs
        # 1. q' = Seq(Seq(q, p31), True*)
        if self.npred + 1 <= self.max_npred and self.depth + 1 <= self.max_depth:
            pred_instances = []
            for pred in self.predicate_list:
                # {"name": "Near", "parameters": [-1.05], "nargs": 2}
                if pred["parameters"]:
                    for param in pred["parameters"]:
                        pred_instances.append({"name": pred["name"], "parameter": param, "nargs": pred["nargs"]})
                else:
                    pred_instances.append({"name": pred["name"], "parameter": None, "nargs": pred["nargs"]})
            for pred_instance in pred_instances:
                nvars = pred_instance["nargs"]
                if nvars > len(self.variables):
                    raise ValueError("The predicate has more variables than the number of variables in the query.")
                # Special case: for trajectory experiment only
                if self.is_trajectory:
                    if nvars == 1:
                        if self.dataset_name == "warsaw":
                            variables_list = [["o0"], ["o1"]]
                        else:
                            # Special case: for CLEVRER trajectory experiment, single-varible predicates can only be applied to o0
                            variables_list = [["o0"]]
                    elif nvars == 2:
                        variables_list = [["o0", "o1"]]
                # Gneral case:
                else:
                    variables_list = itertools.permutations(self.variables, nvars)
                for variables in variables_list:
                    predicate = {"predicate": pred_instance["name"], "parameter": pred_instance["parameter"], "variables": list(variables)}
                    new_scene_graph = {"scene_graph": [predicate], "duration_constraint": 1}
                    for insert_idx in range(self.depth + 1):
                        new_query = copy.deepcopy(self)
                        new_query.program.insert(insert_idx, new_scene_graph)
                        new_query.npred += 1
                        new_query.depth += 1
                        print("Action B: ", program_to_dsl(new_query.program, not self.is_trajectory))
                        all_children.append(new_query)

        # Action c: Duration refinement: increase the duration of a scene graph
        for scene_graph_idx, dict in enumerate(self.program):
            scene_graph = dict["scene_graph"]
            if dict["duration_constraint"] < self.max_duration:
                new_query = copy.deepcopy(self)
                if new_query.program[scene_graph_idx]["duration_constraint"] == 1 and self.duration_unit != 1:
                    new_query.program[scene_graph_idx]["duration_constraint"] = self.duration_unit
                else:
                    new_query.program[scene_graph_idx]["duration_constraint"] += self.duration_unit
                print("Action C: ", program_to_dsl(new_query.program, not self.is_trajectory))
                all_children.append(new_query)

        # Remove duplicates
        all_children_removing_duplicates = []
        print("[all_children] before removing duplicates:", len(all_children))
        signatures = set()
        for query in all_children:
            signature = program_to_dsl(query.program, not self.is_trajectory)
            if signature not in signatures:
                query.program = dsl_to_program(signature)
                all_children_removing_duplicates.append(query)
                signatures.add(signature)
        print("[all_children] after removing duplicates:", len(all_children_removing_duplicates))
        return all_children_removing_duplicates