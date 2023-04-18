# InLaneK, MinLength_theta, Dist_theta(A, B)
import math
import copy
import numpy as np
import functools
import src.utils as utils
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

# (predicate), conjunction, sequencing, Kleene star, finite k iteration
# Omit nested Kleene star operators, as well as Kleene star around sequencing.
# Use Boolean semantics

class BaseOperator:

    def __init__(self, submodules, name=""):
        self.submodules = submodules
        self.name = name

    def get_submodules(self):
        return self.submodules

    def set_submodules(self, new_submodules):
        self.submodules = new_submodules

    def get_typesignature(self):
        return self.input_type, self.output_type

    def execute(self):
        # Boolean semantics: Z (trajectories) --> B (0 or 1)
        raise NotImplementedError

class StartOperator(BaseOperator):
    def __init__(self, function1=None):
        if function1 is None:
            self.program = PredicateHole()
        else:
            self.program = function1
        submodules = { 'program' : self.program }
        super().__init__(submodules, name="Start")

    def execute(self, input, label, memoize, new_memoize):
        # dsl.StartOperator(dsl.SequencingOperator(dsl.SequencingOperator(dsl.TrueStar(), dsl.PredicateHole()), dsl.TrueStar()))
        top_level = SequencingOperator(SequencingOperator(TrueStar(), self.submodules["program"]), TrueStar())
        return top_level.execute(input, label, memoize, new_memoize)
        # return self.submodules["program"].execute(input, label, memoize, new_memoize)

class ConjunctionOperator(BaseOperator):
    def __init__(self, function1=None, function2=None):
        if function1 is None:
            function1 = PredicateHole()
        if function2 is None:
            function2 = PredicateHole()
        submodules = { "function1": function1, "function2": function2 }
        super().__init__(submodules, name="Conjunction")

    def execute(self, input, label, memoize, new_memoize):
        subquery_str = utils.print_program(self)
        if subquery_str in memoize:
            return memoize[subquery_str], new_memoize
        if subquery_str in new_memoize:
            return new_memoize[subquery_str], new_memoize
        res1, new_memoize = self.submodules["function1"].execute(input, label, memoize, new_memoize)
        res2, new_memoize = self.submodules["function2"].execute(input, label, memoize, new_memoize)
        result = np.minimum(res1, res2)

        new_memoize[subquery_str] = result

        return result, new_memoize

class SequencingOperator(BaseOperator):
    def __init__(self, function1=None, function2=None):
        if function1 is None:
            function1 = PredicateHole()
        if function2 is None:
            function2 = PredicateHole()
        submodules = { "function1": function1, "function2": function2 }
        super().__init__(submodules, name="Sequencing")

    def execute(self, input, label, memoize, new_memoize):
        subquery_str = utils.print_program(self)
        if subquery_str in memoize:
            return memoize[subquery_str], new_memoize
        if subquery_str in new_memoize:
            return new_memoize[subquery_str], new_memoize
        res1, new_memoize = self.submodules["function1"].execute(input, label, memoize, new_memoize)
        res2, new_memoize = self.submodules["function2"].execute(input, label, memoize, new_memoize)
        result = np.amax(np.minimum(res1[..., np.newaxis], res2[np.newaxis, ...]), axis=1)

        new_memoize[subquery_str] = result

        return result, new_memoize

# class IterationOperator(BaseOperator):
#     def __init__(self, k, function1=None):
#         self.k = k
#         if function1 is None:
#             function1 = PredicateHole()
#         submodules = { "iteration": function1}
#         super().__init__(submodules, name="Iteration")

#     def execute(self, input, label, memoize):
#         subquery_str = utils.print_program(self)
#         if subquery_str in memoize:
#             return memoize[subquery_str], memoize
#         result = self.execute_helper(self.k, input, label, memoize)
#         memoize[subquery_str] = result
#         return result, memoize

#     def execute_helper(self, k, input, label, memoize):
#         # TODO: this is not optimized (refer to the paper)
#         if k == 0:
#             # A matrix with inf on the diagonal and -inf elsewhere
#             return np.fill_diagonal(np.full((len(input[0]) + 1, len(input[0]) + 1), -np.inf), np.inf)

#         pow = self.execute_helper(self, k // 2, input, label)

#         if k % 2 == 0:
#             # pow * pow
#             return np.max(np.minimum(pow[..., np.newaxis], pow[np.newaxis, ...]), axis=1)
#         else:
#             # M * pow * pow
#             return np.max(np.minimum(self.submodules["iteration"].execute(input, label, memoize)[0][..., np.newaxis], np.max(np.minimum(pow[..., np.newaxis], pow[np.newaxis, ...]), axis=1)), axis=1)

# NOTE: the implementation is actually Kleene Plus. Otherwise, Quivr takes even longer.
class KleeneOperator(BaseOperator):
    def __init__(self, function1=None):
        if function1 is None:
            function1 = PredicateHole()
        submodules = { "kleene": function1}
        super().__init__(submodules, name="Kleene")

    def execute(self, input, label, memoize, new_memoize):
        subquery_str = utils.print_program(self)
        if subquery_str in memoize:
            return memoize[subquery_str], new_memoize
        if subquery_str in new_memoize:
            return new_memoize[subquery_str], new_memoize
        identity_mtx = np.full((len(input[0]) + 1, len(input[0]) + 1), -np.inf)
        np.fill_diagonal(identity_mtx, np.inf)

        if len(input[0]) == 0:
            result = np.full((len(input[0]) + 1, len(input[0]) + 1), -np.inf)
        elif len(input[0]) == 1:
            result, new_memoize = self.submodules["kleene"].execute(input, label, memoize, new_memoize)
        else:
            Q_mtx, new_memoize = self.submodules["kleene"].execute(input, label, memoize, new_memoize)
            base = np.maximum(identity_mtx, Q_mtx) # Q + I
            result = power(base, len(input[0])-1) # (Q + I)^(n-1)
            result = np.amax(np.minimum(Q_mtx[..., np.newaxis], result[np.newaxis, ...]), axis=1) # Q * (Q + I)^(n-1)

        new_memoize[subquery_str] = result

        return result, new_memoize

def power(M, n):
    if n == 1:
        return M
    elif n % 2 == 0:
        M_squared = np.amax(np.minimum(M[..., np.newaxis], M[np.newaxis, ...]), axis=1)
        return power(M_squared, n // 2)
    else:
        M_squared = np.amax(np.minimum(M[..., np.newaxis], M[np.newaxis, ...]), axis=1)
        arr = power(M_squared, n // 2)
        return np.amax(np.minimum(M[..., np.newaxis], arr[np.newaxis, ...]), axis=1)

class DurationOperator(BaseOperator):
    def __init__(self, function1, theta):
        # theta >= 2 and is an integer
        self.theta = int(theta)
        submodules = { "duration": function1 }
        super().__init__(submodules, name="Duration")

    def execute(self, input, label, memoize, new_memoize):
        # res1, memoize = self.submodules["duration"].execute(input, label, memoize)
        kleene = KleeneOperator(self.submodules["duration"])
        conj = ConjunctionOperator(kleene, MinLength(self.theta, self.theta))
        return conj.execute(input, label, memoize, new_memoize)

############### Predicate ################
@functools.total_ordering
class Predicate:
    def __init__(self, name):
        self.name = name

    def __eq__(self, other):
        return self.name == other.name

    def __lt__(self, other):
        return self.name < other.name

    def set_theta(self, theta):
        self.theta = theta

    def execute(self):
        raise NotImplementedError

class Near(Predicate):
    has_theta=True

    def __init__(self, theta=-1, step=0.2, with_hole=False):
        self.theta = theta
        self.step = step
        self.with_hole = with_hole
        super().__init__("Near")

    def init_value_range(self):
        return [-1.2, -0.8]

    def execute_with_hole(self, input, label, memoize, new_memoize):
        assert len(input) == 2
        subquery_str = utils.print_program(self)
        if subquery_str in memoize:
            return memoize[subquery_str], new_memoize
        if subquery_str in new_memoize:
            return new_memoize[subquery_str], new_memoize
        result = np.full((len(input[0]) + 1, len(input[0]) + 1), -np.inf)
        for i in range(len(input[0])):
            result[i, i + 1] = -1 * obj_distance(input[0][i], input[1][i])
        new_memoize[subquery_str] = result
        return result, new_memoize

    def execute(self, input, label, memoize, new_memoize):
        if self.with_hole:
            return self.execute_with_hole(input, label, memoize, new_memoize)

        assert len(input) == 2

        subquery_str = utils.print_program(self)
        if subquery_str in memoize:
            return memoize[subquery_str], new_memoize
        if subquery_str in new_memoize:
            return new_memoize[subquery_str], new_memoize
        result = np.full((len(input[0]) + 1, len(input[0]) + 1), -np.inf)
        for i in range(len(input[0])):
            if -1 * obj_distance(input[0][i], input[1][i]) >= self.theta:
                result[i, i + 1] = np.inf

        new_memoize[subquery_str] = result

        return result, new_memoize

class Far(Predicate):
    has_theta=True

    def __init__(self, theta=3, step=0.5, with_hole=False):
        self.theta = theta
        self.step = step
        self.with_hole = with_hole
        super().__init__("Far")

    def init_value_range(self):
        return [1, 3]

    def execute_with_hole(self, input, label, memoize, new_memoize):
        assert len(input) == 2
        subquery_str = utils.print_program(self)
        if subquery_str in memoize:
            return memoize[subquery_str], new_memoize
        if subquery_str in new_memoize:
            return new_memoize[subquery_str], new_memoize
        result = np.full((len(input[0]) + 1, len(input[0]) + 1), -np.inf)
        for i in range(len(input[0])):
            result[i, i + 1] = obj_distance(input[0][i], input[1][i])
        new_memoize[subquery_str] = result
        return result, new_memoize

    def execute(self, input, label, memoize, new_memoize):
        if self.with_hole:
            return self.execute_with_hole(input, label, memoize, new_memoize)

        assert len(input) == 2
        subquery_str = utils.print_program(self)
        if subquery_str in memoize:
            return memoize[subquery_str], new_memoize
        if subquery_str in new_memoize:
            return new_memoize[subquery_str], new_memoize
        result = np.full((len(input[0]) + 1, len(input[0]) + 1), -np.inf)
        for i in range(len(input[0])):
            if obj_distance(input[0][i], input[1][i]) >= self.theta:
                result[i, i + 1] = np.inf
        new_memoize[subquery_str] = result

        return result, new_memoize

class DirectionPredicate(Predicate):
    has_theta = False

    def __init__(self, direction_name):
        super().__init__(direction_name)

    def execute(self, direction_name, input, label, memoize, new_memoize):
        assert len(input) == 2
        subquery_str = utils.print_program(self)
        if subquery_str in memoize:
            return memoize[subquery_str], new_memoize
        if subquery_str in new_memoize:
            return new_memoize[subquery_str], new_memoize
        result = np.full((len(input[0]) + 1, len(input[0]) + 1), -np.inf)
        for i in range(len(input[0])):
            if direction_relationship(input[0][i], input[1][i], direction_name):
                result[i, i + 1] = np.inf

        new_memoize[subquery_str] = result

        return result, new_memoize

class QuadrantPredicate(Predicate):
    has_theta = False

    def __init__(self, quadrant_name):
        super().__init__(quadrant_name)

    def execute(self, quadrant_name, input, label, memoize, new_memoize):
        assert len(input) == 2
        subquery_str = utils.print_program(self)
        if subquery_str in memoize:
            return memoize[subquery_str], new_memoize
        if subquery_str in new_memoize:
            return new_memoize[subquery_str], new_memoize
        result = np.full((len(input[0]) + 1, len(input[0]) + 1), -np.inf)
        for i in range(len(input[0])):
            if quadrant_relationship(input[0][i], quadrant_name):
                result[i, i + 1] = np.inf

        new_memoize[subquery_str] = result

        return result, new_memoize


class TopQuadrant(QuadrantPredicate):
    has_theta=False

    def __init__(self):
        super().__init__("TopQuadrant")

    def execute(self, input, label, memoize, new_memoize):
        return super().execute("TopQuadrant", input, label, memoize, new_memoize)

class BottomQuadrant(QuadrantPredicate):
    has_theta=False

    def __init__(self):
        super().__init__("BottomQuadrant")

    def execute(self, input, label, memoize, new_memoize):
        return super().execute("BottomQuadrant", input, label, memoize, new_memoize)

class RightQuadrant(QuadrantPredicate):
    has_theta=False

    def __init__(self):
        super().__init__("RightQuadrant")

    def execute(self, input, label, memoize, new_memoize):
        return super().execute("RightQuadrant", input, label, memoize, new_memoize)

class LeftQuadrant(QuadrantPredicate):
    has_theta=False

    def __init__(self):
        super().__init__("LeftQuadrant")

    def execute(self, input, label, memoize, new_memoize):
        return super().execute("LeftQuadrant", input, label, memoize, new_memoize)

class FrontOf(DirectionPredicate):
    has_theta=False

    def __init__(self):
        super().__init__("FrontOf")

    def execute(self, input, label, memoize, new_memoize):
        return super().execute("FrontOf", input, label, memoize, new_memoize)

class Behind(DirectionPredicate):
    has_theta=False

    def __init__(self):
        super().__init__("Behind")

    def execute(self, input, label, memoize, new_memoize):
        return super().execute("Behind", input, label, memoize, new_memoize)

class RightOf(DirectionPredicate):
    has_theta=False

    def __init__(self):
        super().__init__("RightOf")

    def execute(self, input, label, memoize, new_memoize):
        return super().execute("RightOf", input, label, memoize, new_memoize)

class LeftOf(DirectionPredicate):
    has_theta=False

    def __init__(self):
        super().__init__("LeftOf")

    def execute(self, input, label, memoize, new_memoize):
        return super().execute("LeftOf", input, label, memoize, new_memoize)

############### Warsaw Predicate ################

class InLanePredicate(Predicate):
    has_theta = False

    def __init__(self, lane_name):
        super().__init__(lane_name)

    def execute(self, lane_name, input, label, memoize, new_memoize):
        which_object = 0 if lane_name[0] == 'A' else 1
        lane_name = lane_name[1:]
        assert len(input) == 2
        subquery_str = utils.print_program(self)
        if subquery_str in memoize:
            return memoize[subquery_str], new_memoize
        if subquery_str in new_memoize:
            return new_memoize[subquery_str], new_memoize
        result = np.full((len(input[0]) + 1, len(input[0]) + 1), -np.inf)
        for i in range(len(input[0])):
            if in_lane(input[which_object][i], lane_name):
                result[i, i + 1] = np.inf

        new_memoize[subquery_str] = result

        return result, new_memoize

class AEastward4(InLanePredicate):
    has_theta=False

    def __init__(self):
        super().__init__("AEastward4")

    def execute(self, input, label, memoize, new_memoize):
        return super().execute("AEastward4", input, label, memoize, new_memoize)

class AEastward3(InLanePredicate):
    has_theta=False

    def __init__(self):
        super().__init__("AEastward3")

    def execute(self, input, label, memoize, new_memoize):
        return super().execute("AEastward3", input, label, memoize, new_memoize)

class AEastward2(InLanePredicate):
    has_theta=False

    def __init__(self):
        super().__init__("AEastward2")

    def execute(self, input, label, memoize, new_memoize):
        return super().execute("AEastward2", input, label, memoize, new_memoize)

class AWestward2(InLanePredicate):
    has_theta=False

    def __init__(self):
        super().__init__("AWestward2")

    def execute(self, input, label, memoize, new_memoize):
        return super().execute("AWestward2", input, label, memoize, new_memoize)

class ASouthward1Upper(InLanePredicate):
    has_theta=False

    def __init__(self):
        super().__init__("ASouthward1Upper")

    def execute(self, input, label, memoize, new_memoize):
        return super().execute("ASouthward1Upper", input, label, memoize, new_memoize)

class BEastward4(InLanePredicate):
    has_theta=False

    def __init__(self):
        super().__init__("BEastward4")

    def execute(self, input, label, memoize, new_memoize):
        return super().execute("BEastward4", input, label, memoize, new_memoize)

class BEastward3(InLanePredicate):
    has_theta=False

    def __init__(self):
        super().__init__("BEastward3")

    def execute(self, input, label, memoize, new_memoize):
        return super().execute("BEastward3", input, label, memoize, new_memoize)

class BEastward2(InLanePredicate):
    has_theta=False

    def __init__(self):
        super().__init__("BEastward2")

    def execute(self, input, label, memoize, new_memoize):
        return super().execute("BEastward2", input, label, memoize, new_memoize)

class BWestward2(InLanePredicate):
    has_theta=False

    def __init__(self):
        super().__init__("BWestward2")

    def execute(self, input, label, memoize, new_memoize):
        return super().execute("BWestward2", input, label, memoize, new_memoize)

class BSouthward1Upper(InLanePredicate):
    has_theta=False

    def __init__(self):
        super().__init__("BSouthward1Upper")

    def execute(self, input, label, memoize, new_memoize):
        return super().execute("BSouthward1Upper", input, label, memoize, new_memoize)

class AStopped(Predicate):
    has_theta=True

    def __init__(self, theta=-2, step=0.5, with_hole=False):
        self.theta = theta
        self.step = step
        self.with_hole = with_hole
        super().__init__("AStopped")

    def init_value_range(self):
        return [-1, -3]

    def execute_with_hole(self, input, label, memoize, new_memoize):
        assert len(input) == 2
        subquery_str = utils.print_program(self)
        if subquery_str in memoize:
            return memoize[subquery_str], new_memoize
        if subquery_str in new_memoize:
            return new_memoize[subquery_str], new_memoize
        result = np.full((len(input[0]) + 1, len(input[0]) + 1), -np.inf)
        for i in range(len(input[0])):
            if input[0][i][4] is not np.nan and input[0][i][5] is not np.nan:
                result[i, i + 1] = -1 * math.sqrt(input[0][i][4] ** 2 + input[0][i][5] ** 2)
        new_memoize[subquery_str] = result
        return result, new_memoize

    def execute(self, input, label, memoize, new_memoize):
        if self.with_hole:
            return self.execute_with_hole(input, label, memoize, new_memoize)

        assert len(input) == 2

        subquery_str = utils.print_program(self)
        if subquery_str in memoize:
            return memoize[subquery_str], new_memoize
        if subquery_str in new_memoize:
            return new_memoize[subquery_str], new_memoize
        result = np.full((len(input[0]) + 1, len(input[0]) + 1), -np.inf)
        for i in range(len(input[0])):
            if input[0][i][4] is not np.nan and input[0][i][5] is not np.nan and -1 * math.sqrt(input[0][i][4] ** 2 + input[0][i][5] ** 2) >= self.theta:
                result[i, i + 1] = np.inf

        new_memoize[subquery_str] = result

        return result, new_memoize

class BStopped(Predicate):
    has_theta=True

    def __init__(self, theta=-2, step=0.5, with_hole=False):
        self.theta = theta
        self.step = step
        self.with_hole = with_hole
        super().__init__("BStopped")

    def init_value_range(self):
        return [-1, -3]

    def execute_with_hole(self, input, label, memoize, new_memoize):
        assert len(input) == 2
        subquery_str = utils.print_program(self)
        if subquery_str in memoize:
            return memoize[subquery_str], new_memoize
        if subquery_str in new_memoize:
            return new_memoize[subquery_str], new_memoize
        result = np.full((len(input[1]) + 1, len(input[1]) + 1), -np.inf)
        for i in range(len(input[1])):
            if input[1][i][4] is not np.nan and input[1][i][5] is not np.nan:
                result[i, i + 1] = -1 * math.sqrt(input[1][i][4] ** 2 + input[1][i][5] ** 2)
        new_memoize[subquery_str] = result
        return result, new_memoize

    def execute(self, input, label, memoize, new_memoize):
        if self.with_hole:
            return self.execute_with_hole(input, label, memoize, new_memoize)

        assert len(input) == 2

        subquery_str = utils.print_program(self)
        if subquery_str in memoize:
            return memoize[subquery_str], new_memoize
        if subquery_str in new_memoize:
            return new_memoize[subquery_str], new_memoize
        result = np.full((len(input[1]) + 1, len(input[1]) + 1), -np.inf)
        for i in range(len(input[1])):
            if input[1][i][4] is not np.nan and input[1][i][5] is not np.nan and -1 * math.sqrt(input[1][i][4] ** 2 + input[1][i][5] ** 2) >= self.theta:
                result[i, i + 1] = np.inf

        new_memoize[subquery_str] = result

        return result, new_memoize

class AHighAccel(Predicate):
    has_theta=True

    def __init__(self, theta=2, step=0.5, with_hole=False):
        self.theta = theta
        self.step = step
        self.with_hole = with_hole
        super().__init__("AHighAccel")

    def init_value_range(self):
        return [1, 3]

    def execute_with_hole(self, input, label, memoize, new_memoize):
        assert len(input) == 2
        subquery_str = utils.print_program(self)
        if subquery_str in memoize:
            return memoize[subquery_str], new_memoize
        if subquery_str in new_memoize:
            return new_memoize[subquery_str], new_memoize
        result = np.full((len(input[0]) + 1, len(input[0]) + 1), -np.inf)
        for i in range(len(input[0])):
            if input[0][i][6] is not np.nan and input[0][i][7] is not np.nan:
                result[i, i + 1] = math.sqrt(input[0][i][6] ** 2 + input[0][i][7] ** 2)
        new_memoize[subquery_str] = result
        return result, new_memoize

    def execute(self, input, label, memoize, new_memoize):
        if self.with_hole:
            return self.execute_with_hole(input, label, memoize, new_memoize)

        assert len(input) == 2

        subquery_str = utils.print_program(self)
        if subquery_str in memoize:
            return memoize[subquery_str], new_memoize
        if subquery_str in new_memoize:
            return new_memoize[subquery_str], new_memoize
        result = np.full((len(input[0]) + 1, len(input[0]) + 1), -np.inf)
        for i in range(len(input[0])):
            if input[0][i][6] is not np.nan and input[0][i][7] is not np.nan and math.sqrt(input[0][i][6] ** 2 + input[0][i][7] ** 2) >= self.theta:
                result[i, i + 1] = np.inf

        new_memoize[subquery_str] = result

        return result, new_memoize

class BHighAccel(Predicate):
    has_theta=True

    def __init__(self, theta=2, step=0.5, with_hole=False):
        self.theta = theta
        self.step = step
        self.with_hole = with_hole
        super().__init__("BHighAccel")

    def init_value_range(self):
        return [1, 3]

    def execute_with_hole(self, input, label, memoize, new_memoize):
        assert len(input) == 2
        subquery_str = utils.print_program(self)
        if subquery_str in memoize:
            return memoize[subquery_str], new_memoize
        if subquery_str in new_memoize:
            return new_memoize[subquery_str], new_memoize
        result = np.full((len(input[1]) + 1, len(input[1]) + 1), -np.inf)
        for i in range(len(input[1])):
            if input[1][i][6] is not np.nan and input[1][i][7] is not np.nan:
                result[i, i + 1] = math.sqrt(input[1][i][6] ** 2 + input[1][i][7] ** 2)
        new_memoize[subquery_str] = result
        return result, new_memoize

    def execute(self, input, label, memoize, new_memoize):
        if self.with_hole:
            return self.execute_with_hole(input, label, memoize, new_memoize)

        assert len(input) == 2

        subquery_str = utils.print_program(self)
        if subquery_str in memoize:
            return memoize[subquery_str], new_memoize
        if subquery_str in new_memoize:
            return new_memoize[subquery_str], new_memoize
        result = np.full((len(input[1]) + 1, len(input[1]) + 1), -np.inf)
        for i in range(len(input[1])):
            if input[1][i][6] is not np.nan and input[1][i][7] is not np.nan and math.sqrt(input[1][i][6] ** 2 + input[1][i][7] ** 2) >= self.theta:
                result[i, i + 1] = np.inf

        new_memoize[subquery_str] = result

        return result, new_memoize

class DistanceSmall(Predicate):
    has_theta=True

    def __init__(self, theta=-100, step=50, with_hole=False):
        self.theta = theta
        self.step = step
        self.with_hole = with_hole
        super().__init__("DistanceSmall")

    def init_value_range(self):
        return [-50, -200]

    def execute_with_hole(self, input, label, memoize, new_memoize):
        assert len(input) == 2
        subquery_str = utils.print_program(self)
        if subquery_str in memoize:
            return memoize[subquery_str], new_memoize
        if subquery_str in new_memoize:
            return new_memoize[subquery_str], new_memoize
        result = np.full((len(input[0]) + 1, len(input[0]) + 1), -np.inf)
        for i in range(len(input[0])):
            if input[0][i][0] is not np.nan and input[1][i][0] is not np.nan:
                result[i, i + 1] = -1 * obj_distance_warsaw(input[0][i], input[1][i])
        new_memoize[subquery_str] = result
        return result, new_memoize

    def execute(self, input, label, memoize, new_memoize):
        if self.with_hole:
            return self.execute_with_hole(input, label, memoize, new_memoize)

        assert len(input) == 2

        subquery_str = utils.print_program(self)
        if subquery_str in memoize:
            return memoize[subquery_str], new_memoize
        if subquery_str in new_memoize:
            return new_memoize[subquery_str], new_memoize
        result = np.full((len(input[0]) + 1, len(input[0]) + 1), -np.inf)
        for i in range(len(input[0])):
            if input[0][i][0] is not np.nan and input[1][i][0] is not np.nan and -1 * obj_distance_warsaw(input[0][i], input[1][i]) >= self.theta:
                result[i, i + 1] = np.inf

        new_memoize[subquery_str] = result

        return result, new_memoize

class Faster(Predicate):
    has_theta=True

    def __init__(self, theta=1.5, step=0.5, with_hole=False):
        self.theta = theta
        self.step = step
        self.with_hole = with_hole
        super().__init__("Faster")

    def init_value_range(self):
        return [1, 3]

    def execute_with_hole(self, input, label, memoize, new_memoize):
        assert len(input) == 2
        subquery_str = utils.print_program(self)
        if subquery_str in memoize:
            return memoize[subquery_str], new_memoize
        if subquery_str in new_memoize:
            return new_memoize[subquery_str], new_memoize
        result = np.full((len(input[0]) + 1, len(input[0]) + 1), -np.inf)
        for i in range(len(input[0])):
            if input[0][i][4] is not np.nan and input[1][i][4] is not np.nan:
                result[i, i + 1] = speed_ratio(input[0][i], input[1][i])
        new_memoize[subquery_str] = result
        return result, new_memoize

    def execute(self, input, label, memoize, new_memoize):
        if self.with_hole:
            return self.execute_with_hole(input, label, memoize, new_memoize)

        assert len(input) == 2
        subquery_str = utils.print_program(self)
        if subquery_str in memoize:
            return memoize[subquery_str], new_memoize
        if subquery_str in new_memoize:
            return new_memoize[subquery_str], new_memoize
        result = np.full((len(input[0]) + 1, len(input[0]) + 1), -np.inf)
        for i in range(len(input[0])):
            if input[0][i][4] is not np.nan and input[1][i][4] is not np.nan and speed_ratio(input[0][i], input[1][i]) >= self.theta:
                result[i, i + 1] = np.inf
        new_memoize[subquery_str] = result

        return result, new_memoize

class TrueStar(Predicate):
    has_theta=False

    def __init__(self):
        super().__init__("True*")

    def execute(self, input, label, memoize, new_memoize):
        subquery_str = utils.print_program(self)
        if subquery_str in memoize:
            return memoize[subquery_str], new_memoize
        if subquery_str in new_memoize:
            return new_memoize[subquery_str], new_memoize
        result = np.full((len(input[0]) + 1, len(input[0]) + 1), -np.inf)
        for i in range(len(input[0]) + 1):
            for j in range(i, len(input[0]) + 1):
                result[i, j] = np.inf

        new_memoize[subquery_str] = result

        return result, new_memoize

class MinLength(Predicate):
    has_theta=True

    def __init__(self, max_duration=15, theta=1, step=5, with_hole=False):
        self.theta = theta
        self.step = step
        self.with_hole = with_hole
        self.max_duration = max_duration
        super().__init__("MinLength")

    def init_value_range(self):
        return [5, self.max_duration]

    def execute_with_hole(self, input, label, memoize, new_memoize):
        subquery_str = utils.print_program(self)
        if subquery_str in memoize:
            return memoize[subquery_str], new_memoize
        if subquery_str in new_memoize:
            return new_memoize[subquery_str], new_memoize
        result = np.full((len(input[0]) + 1, len(input[0]) + 1), -np.inf)
        for i in range(len(input[0])):
            for j in range(i, len(input[0]) + 1):
                result[i, j] = j - i
        new_memoize[subquery_str] = result
        return result, new_memoize

    def execute(self, input, label, memoize, new_memoize):
        if self.with_hole:
            return self.execute_with_hole(input, label, memoize, new_memoize)

        subquery_str = utils.print_program(self)
        if subquery_str in memoize:
            return memoize[subquery_str], new_memoize
        if subquery_str in new_memoize:
            return new_memoize[subquery_str], new_memoize
        result = np.full((len(input[0]) + 1, len(input[0]) + 1), -np.inf)
        for i in range(len(input[0])):
            for j in range(i, len(input[0]) + 1):
                if j - i >= self.theta:
                    result[i, j] = np.inf

        new_memoize[subquery_str] = result

        return result, new_memoize

def obj_distance(bbox1, bbox2):
    x1, y1, x2, y2 = bbox1
    x3, y3, x4, y4 = bbox2
    cx1 = (x1 + x2) / 2
    cy1 = (y1 + y2) / 2
    cx2 = (x3 + x4) / 2
    cy2 = (y3 + y4) / 2
    return math.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) / ((x2 - x1 + x4 - x3) / 2)

def obj_distance_warsaw(bbox1, bbox2):
    x1, y1, x2, y2, _, _, _, _ = bbox1
    x3, y3, x4, y4, _, _, _, _ = bbox2
    cx1 = (x1 + x2) / 2
    cy1 = (y1 + y2) / 2
    cx2 = (x3 + x4) / 2
    cy2 = (y3 + y4) / 2
    return math.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)

def speed_ratio(bbox1, bbox2):
    _, _, _, _, v_x, v_y, _, _ = bbox1
    _, _, _, _, v_x2, v_y2, _, _ = bbox2
    return (v_x * v_x + v_y * v_y) / (v_x2 * v_x2 + v_y2 * v_y2 + 1e-6)

def direction_relationship(bbox1, bbox2, direction):
    x1, y1, x2, y2 = bbox1
    x3, y3, x4, y4 = bbox2
    cx1 = (x1 + x2) / 2
    cy1 = (y1 + y2) / 2
    cx2 = (x3 + x4) / 2
    cy2 = (y3 + y4) / 2
    if direction == "LeftOf":
        return cx1 < cx2
    elif direction == "RightOf":
        return cx1 > cx2
    elif direction == "Behind":
        return cy1 < cy2
    elif direction == "FrontOf":
        return cy1 > cy2
    else:
        raise ValueError("Invalid direction")

def quadrant_relationship(bbox1, quadrant):
    x1, y1, x2, y2 = bbox1
    cx1 = (x1 + x2) / 2
    cy1 = (y1 + y2) / 2
    if quadrant == "LeftQuadrant":
        return cx1 >= 0 and cx1 < 240
    elif quadrant == "RightQuadrant":
        return cx1 >= 240 and cx1 <= 480
    elif quadrant == "TopQuadrant":
        return cy1 >= 0 and cy1 < 160
    elif quadrant == "BottomQuadrant":
        return cy1 >= 160 and cy1 <= 320
    else:
        raise ValueError("Invalid direction")

def in_lane(bbox, lane_name):
    x1, y1, x2, y2, _, _, _, _ = bbox
    if x1 == np.nan or y1 == np.nan or x2 == np.nan or y2 == np.nan:
        return False
    eastward_4 = np.array([[0, 330], [330, 362], [789, 369], [960, 355], [960, 372], [803, 391], [351, 389], [0, 351]])
    eastward_3 = np.array([[0, 351], [351, 389], [803, 391], [960, 372], [960, 394], [838, 413], [424, 422], [153, 395], [0, 370]])
    eastward_2 = np.array([[0, 370], [153, 395], [424, 422], [838, 413], [960, 394], [960, 420], [763, 451], [414, 460], [97, 420], [0, 397]])
    westward_2 = np.array([[0, 262], [709, 213], [708, 242], [0, 288]])
    southward_1_upper = np.array([[384, 113], [414, 115], [565, 223], [543, 224]])
    eastward_4_polygon = Polygon(eastward_4)
    eastward_3_polygon = Polygon(eastward_3)
    eastward_2_polygon = Polygon(eastward_2)
    westward_2_polygon = Polygon(westward_2)
    southward_1_upper_polygon = Polygon(southward_1_upper)
    if lane_name == "Eastward4":
        return eastward_4_polygon.contains(Point((x1 + x2) / 2, y2 - 10))
    elif lane_name == "Eastward3":
        return eastward_3_polygon.contains(Point((x1 + x2) / 2, y2 - 10))
    elif lane_name == "Eastward2":
        return eastward_2_polygon.contains(Point((x1 + x2) / 2, y2 - 10))
    elif lane_name == "Westward2":
        return westward_2_polygon.contains(Point((x1 + x2) / 2, y2 - 4))
    elif lane_name == "Southward1Upper":
        return southward_1_upper_polygon.contains(Point((x1 + x2) / 2, y2))
    else:
        raise ValueError("Invalid direction")

######## Hole ###########
class Hole:
    def __init__(self, name):
        self.name = name

class PredicateHole(Hole):
    def __init__(self):
        super().__init__("PredicateHole")

    def execute(self, input, label, memoize, new_memoize):
        subquery_str = utils.print_program(self)
        if subquery_str in memoize:
            return memoize[subquery_str], new_memoize
        if subquery_str in new_memoize:
            return new_memoize[subquery_str], new_memoize
        if label == 1:
            # over-approximation
            result = np.full((len(input[0]) + 1, len(input[0]) + 1), -np.inf)
            for i in range(len(input[0]) + 1):
                for j in range(i, len(input[0]) + 1):
                    result[i, j] = np.inf
            new_memoize[subquery_str] = result
            return result, new_memoize
        else:
            # under-approximation
            result = np.full((len(input[0]) + 1, len(input[0]) + 1), -np.inf)
            new_memoize[subquery_str] = result
            return result, new_memoize

class ParameterHole(Hole):
    def __init__(self, predicate):
        super().__init__("ParameterHole*" + predicate.name)
        self.predicate = predicate
        self.value_range = predicate.init_value_range()
        self.step = predicate.step

    def get_value_range(self):
        return self.value_range

    def get_step(self):
        return self.step

    def get_predicate(self):
        return self.predicate

    def fill_hole(self, theta):
        self.predicate.set_theta(theta)
        # self.predicate.with_hole = False
        return copy.deepcopy(self.predicate)

    def execute(self, input, label, memoize, new_memoize):
        if label == 1:
            # self.predicate.set_theta(-np.inf)
            self.predicate.set_theta(self.value_range[0])
        else:
            # self.predicate.set_theta(np.inf)
            self.predicate.set_theta(self.value_range[1])
        return self.predicate.execute(input, label, memoize, new_memoize)