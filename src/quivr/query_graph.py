import copy
import dsl

class QueryGraph(object):

    def __init__(self, max_num_atomic_predicates, max_depth):
        self.max_num_atomic_predicates = max_num_atomic_predicates
        self.max_depth = max_depth
        self.max_num_atomic_predicates = 0

        start = dsl.StartOperator()
        self.program = start
        self.depth = 0
        self.num_atomic_predicates = 0

    def get_parameter_holes_and_value_space(self):
        parameter_holes = []
        value_space = []
        steps = []
        queue = [self.program]
        while len(queue) != 0:
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
        queue = [candidate_query]
        while len(queue) != 0:
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
        queue = [candidate_query]
        while len(queue) != 0:
            current_function = queue.pop()
            if issubclass(type(current_function), dsl.Hole):
                return False
            if issubclass(type(current_function), dsl.Predicate):
                continue
            else:
                for submodule_name in current_function.submodules:
                    queue.append(current_function.submodules[submodule_name])
        return True


    def get_all_children(self, algorithm):
        """ Each child of Q is obtained by filling in one predicate hole using an applicable operator.
        """
        # Fill the predicate hole that is found first (BFS).
        all_children = []
        # child_depth = self.depth + 1
        # child_num_units = self.num_units_at_depth(child_depth)
        queue = [self.program]
        while len(queue) != 0:
            current = queue.pop(0)
            for submod, functionclass in current.submodules.items():
                if isinstance(functionclass, dsl.ParameterHole):
                    continue
                if issubclass(type(functionclass), dsl.Predicate):
                    continue
                if isinstance(functionclass, dsl.PredicateHole):
                    if algorithm == "quivr":
                        replacement_candidates = self.construct_candidates_quivr(current, submod)
                    elif algorithm == "vocal":
                        replacement_candidates = self.construct_candidates_vocal(current, submod)
                    else:
                        raise ValueError("Unknown algorithm:", algorithm)
                    orig_fclass = copy.deepcopy(current.submodules[submod])
                    for child_candidate, inc_depth in replacement_candidates:
                        # replace the neural function with a candidate
                        current.submodules[submod] = child_candidate
                        # create the correct child node
                        new_query_graph = copy.deepcopy(self)
                        if (issubclass(type(functionclass), dsl.Predicate) and not isinstance(functionclass, dsl.TrueStar)) or isinstance(functionclass, dsl.ParameterHole):
                            new_query_graph.num_atomic_predicates += 1
                        new_query_graph.depth = self.depth + inc_depth
                        # check if child program can be completed within max_depth
                        if new_query_graph.depth > new_query_graph.max_depth or new_query_graph.num_atomic_predicates > new_query_graph.max_num_atomic_predicates:
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


    def construct_candidates_quivr(self, parent_functionclass, submod):
        candidates = []
        if isinstance(parent_functionclass, dsl.KleeneOperator):
            # omit nested Kleene star operators and Kleene star around sequencing; also Kleene star around <True>* or MinLength doesn't make sense
            replacement_candidates = [dsl.ConjunctionOperator, dsl.Near, dsl.Far]
        elif isinstance(parent_functionclass, dsl.SequencingOperator):
            if submod == "function1":
                # MinLength shouldn't appear in sequencing
                replacement_candidates = [dsl.ConjunctionOperator, dsl.SequencingOperator, dsl.KleeneOperator, dsl.Near, dsl.Far, dsl.TrueStar]
            else:
                # Remove sequencing MinLength; remove semantically equivalent duplicates: associativity of sequencing
                replacement_candidates = [dsl.ConjunctionOperator, dsl.KleeneOperator, dsl.Near, dsl.Far, dsl.TrueStar]
        elif isinstance(parent_functionclass, dsl.ConjunctionOperator):
            if submod == "function1":
                replacement_candidates = [dsl.ConjunctionOperator, dsl.SequencingOperator, dsl.KleeneOperator, dsl.Near, dsl.Far, dsl.MinLength]
            else:
                left_child = parent_functionclass.submodules["function1"]
                # remove semantically equivalent duplicates: associativity of conjunction and commutativity of conjunction
                if isinstance(left_child, dsl.ConjunctionOperator):
                    replacement_candidates = [dsl.SequencingOperator, dsl.KleeneOperator, dsl.Near, dsl.Far, dsl.MinLength]
                elif isinstance(left_child, dsl.SequencingOperator):
                    replacement_candidates = [dsl.SequencingOperator, dsl.KleeneOperator, dsl.Near, dsl.Far, dsl.MinLength]
                elif isinstance(left_child, dsl.KleeneOperator):
                    replacement_candidates = [dsl.KleeneOperator, dsl.Near, dsl.Far, dsl.MinLength]
                elif isinstance(left_child, dsl.TrueStar):
                    replacement_candidates = [dsl.Near, dsl.Far, dsl.MinLength]
                elif isinstance(left_child, dsl.ParameterHole):
                    if isinstance(left_child.get_predicate(), dsl.Near):
                        replacement_candidates = [dsl.Far, dsl.MinLength]
                    elif isinstance(left_child.get_predicate(), dsl.Far):
                        replacement_candidates = [dsl.MinLength]
                    elif isinstance(left_child.get_predicate(), dsl.MinLength):
                        replacement_candidates = []
                else:
                    raise ValueError
        else:
            replacement_candidates = [dsl.ConjunctionOperator, dsl.SequencingOperator, dsl.KleeneOperator, dsl.Near, dsl.Far, dsl.MinLength, dsl.TrueStar]
        for functionclass in replacement_candidates:
            if issubclass(functionclass, dsl.Predicate):
                if functionclass.has_theta:
                    candidate = dsl.ParameterHole(functionclass())
                else:
                    candidate = functionclass()
                candidates.append([candidate, 0])
            else:
                candidate = functionclass()
                candidates.append([candidate, 1])
        return candidates


    def construct_candidates_vocal(self, parent_functionclass, submod):
        candidates = []
        if isinstance(parent_functionclass, dsl.KleeneOperator):
            # omit nested Kleene star operators and Kleene star around sequencing; also Kleene star around <True>* or MinLength doesn't make sense
            replacement_candidates = [dsl.ConjunctionOperator, dsl.Near, dsl.Far]
        elif isinstance(parent_functionclass, dsl.SequencingOperator):
            if submod == "function1":
                # MinLength shouldn't appear in sequencing
                replacement_candidates = [dsl.ConjunctionOperator, dsl.SequencingOperator, dsl.KleeneOperator, dsl.Near, dsl.Far, dsl.TrueStar]
            else:
                # Remove sequencing MinLength; remove semantically equivalent duplicates: associativity of sequencing
                replacement_candidates = [dsl.ConjunctionOperator, dsl.KleeneOperator, dsl.Near, dsl.Far, dsl.TrueStar]
        elif isinstance(parent_functionclass, dsl.ConjunctionOperator):
            if submod == "function1":
                replacement_candidates = [dsl.ConjunctionOperator, dsl.SequencingOperator, dsl.KleeneOperator, dsl.Near, dsl.Far, dsl.MinLength]
            else:
                left_child = parent_functionclass.submodules["function1"]
                # remove semantically equivalent duplicates: associativity of conjunction and commutativity of conjunction
                if isinstance(left_child, dsl.ConjunctionOperator):
                    replacement_candidates = [dsl.SequencingOperator, dsl.KleeneOperator, dsl.Near, dsl.Far, dsl.MinLength]
                elif isinstance(left_child, dsl.SequencingOperator):
                    replacement_candidates = [dsl.SequencingOperator, dsl.KleeneOperator, dsl.Near, dsl.Far, dsl.MinLength]
                elif isinstance(left_child, dsl.KleeneOperator):
                    replacement_candidates = [dsl.KleeneOperator, dsl.Near, dsl.Far, dsl.MinLength]
                elif isinstance(left_child, dsl.TrueStar):
                    replacement_candidates = [dsl.Near, dsl.Far, dsl.MinLength]
                # elif isinstance(left_child, dsl.ParameterHole):
                #     if isinstance(left_child.get_predicate(), dsl.Near):
                #         replacement_candidates = [dsl.Far, dsl.MinLength]
                #     elif isinstance(left_child.get_predicate(), dsl.Far):
                #         replacement_candidates = [dsl.MinLength]
                #     elif isinstance(left_child.get_predicate(), dsl.MinLength):
                #         replacement_candidates = []
                elif isinstance(left_child, dsl.Near):
                    replacement_candidates = [dsl.Far, dsl.MinLength]
                elif isinstance(left_child, dsl.Far):
                    replacement_candidates = [dsl.MinLength]
                elif isinstance(left_child, dsl.ParameterHole):
                    # assert(isinstance(left_child.get_predicate(), dsl.MinLength))
                    replacement_candidates = []
                else:
                    print("value error", left_child)
                    # raise ValueError
                # elif isinstance(left_child, dsl.Near):
                #     replacement_candidates = [dsl.Far, dsl.MinLength]
                # elif isinstance(left_child, dsl.Far):
                #     replacement_candidates = [dsl.MinLength]
                # elif isinstance(left_child, dsl.MinLength):
                #     replacement_candidates = []
        else:
            replacement_candidates = [dsl.ConjunctionOperator, dsl.SequencingOperator, dsl.KleeneOperator, dsl.Near, dsl.Far, dsl.MinLength, dsl.TrueStar]
        for functionclass in replacement_candidates:
            if issubclass(functionclass, dsl.Predicate):
                # if functionclass.has_theta:
                #     candidate = dsl.ParameterHole(functionclass())
                # else:
                #     candidate = functionclass()
                if isinstance(functionclass(), dsl.MinLength):
                    candidate = dsl.ParameterHole(functionclass())
                else:
                    candidate = functionclass()
                candidates.append([candidate, 0])
            else:
                candidate = functionclass()
                candidates.append([candidate, 1])
        return candidates