# InLaneK, MinLength_theta, Dist_theta(A, B)
import math
import copy
import numpy as np
import functools
import src.utils as utils
import shapely
from shapely import Point, Polygon

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
        subquery_str = utils.print_program(self)
        if subquery_str in memoize:
            return memoize[subquery_str], new_memoize
        if subquery_str in new_memoize:
            return new_memoize[subquery_str], new_memoize
        result = np.full((len(input[0]) + 1, len(input[0]) + 1), -np.inf)
        out = np.sqrt(np.square(input[0, :, 0] + input[0, :, 2] - input[1, :, 0] - input[1, :, 2]) + np.square(input[0, :, 1] + input[0, :, 3] - input[1, :, 1] - input[1, :, 3])) / (input[0, :, 2] - input[0, :, 0] + input[1, :, 2] - input[1, :, 0]) # (L,)
        indices = np.arange(len(input[0]))
        result[indices, indices+1] = -1 * out
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
        out = np.sqrt(np.square(input[0, :, 0] + input[0, :, 2] - input[1, :, 0] - input[1, :, 2]) + np.square(input[0, :, 1] + input[0, :, 3] - input[1, :, 1] - input[1, :, 3])) / (input[0, :, 2] - input[0, :, 0] + input[1, :, 2] - input[1, :, 0]) # (L,)
        indices = (out <= -1 * self.theta).nonzero()[0]
        result[indices, indices+1] = np.inf

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
        subquery_str = utils.print_program(self)
        if subquery_str in memoize:
            return memoize[subquery_str], new_memoize
        if subquery_str in new_memoize:
            return new_memoize[subquery_str], new_memoize
        result = np.full((len(input[0]) + 1, len(input[0]) + 1), -np.inf)
        out = np.sqrt(np.square(input[0, :, 0] + input[0, :, 2] - input[1, :, 0] - input[1, :, 2]) + np.square(input[0, :, 1] + input[0, :, 3] - input[1, :, 1] - input[1, :, 3])) / (input[0, :, 2] - input[0, :, 0] + input[1, :, 2] - input[1, :, 0]) # (L,)
        indices = np.arange(len(input[0]))
        result[indices, indices+1] = out
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
        out = np.sqrt(np.square(input[0, :, 0] + input[0, :, 2] - input[1, :, 0] - input[1, :, 2]) + np.square(input[0, :, 1] + input[0, :, 3] - input[1, :, 1] - input[1, :, 3])) / (input[0, :, 2] - input[0, :, 0] + input[1, :, 2] - input[1, :, 0]) # (L,)
        indices = (out >= self.theta).nonzero()[0]
        result[indices, indices+1] = np.inf
        new_memoize[subquery_str] = result

        return result, new_memoize


class TopQuadrant(Predicate):
    has_theta=False

    def __init__(self):
        super().__init__("TopQuadrant")

    def execute(self, input, label, memoize, new_memoize):
        subquery_str = utils.print_program(self)
        if subquery_str in memoize:
            return memoize[subquery_str], new_memoize
        if subquery_str in new_memoize:
            return new_memoize[subquery_str], new_memoize
        result = np.full((len(input[0]) + 1, len(input[0]) + 1), -np.inf)
        out = (input[0, :, 1] + input[0, :, 3]) / 2
        indices = np.where((out >= 0) & (out <= 160))[0]
        result[indices, indices+1] = np.inf

        new_memoize[subquery_str] = result

        return result, new_memoize

class BottomQuadrant(Predicate):
    has_theta=False

    def __init__(self):
        super().__init__("BottomQuadrant")

    def execute(self, input, label, memoize, new_memoize):
        subquery_str = utils.print_program(self)
        if subquery_str in memoize:
            return memoize[subquery_str], new_memoize
        if subquery_str in new_memoize:
            return new_memoize[subquery_str], new_memoize
        result = np.full((len(input[0]) + 1, len(input[0]) + 1), -np.inf)
        out = (input[0, :, 1] + input[0, :, 3]) / 2
        indices = np.where((out >= 160) & (out <= 320))[0]
        result[indices, indices+1] = np.inf

        new_memoize[subquery_str] = result

        return result, new_memoize

class RightQuadrant(Predicate):
    has_theta=False

    def __init__(self):
        super().__init__("RightQuadrant")

    def execute(self, input, label, memoize, new_memoize):
        subquery_str = utils.print_program(self)
        if subquery_str in memoize:
            return memoize[subquery_str], new_memoize
        if subquery_str in new_memoize:
            return new_memoize[subquery_str], new_memoize
        result = np.full((len(input[0]) + 1, len(input[0]) + 1), -np.inf)
        out = (input[0, :, 0] + input[0, :, 2]) / 2
        indices = np.where((out >= 240) & (out <= 480))[0]
        result[indices, indices+1] = np.inf

        new_memoize[subquery_str] = result

        return result, new_memoize

class LeftQuadrant(Predicate):
    has_theta=False

    def __init__(self):
        super().__init__("LeftQuadrant")

    def execute(self, input, label, memoize, new_memoize):
        subquery_str = utils.print_program(self)
        if subquery_str in memoize:
            return memoize[subquery_str], new_memoize
        if subquery_str in new_memoize:
            return new_memoize[subquery_str], new_memoize
        result = np.full((len(input[0]) + 1, len(input[0]) + 1), -np.inf)
        out = (input[0, :, 0] + input[0, :, 2]) / 2
        indices = np.where((out >= 0) & (out < 240))[0]
        result[indices, indices+1] = np.inf

        new_memoize[subquery_str] = result

        return result, new_memoize

class FrontOf(Predicate):
    has_theta=False

    def __init__(self):
        super().__init__("FrontOf")

    def execute(self, input, label, memoize, new_memoize):
        subquery_str = utils.print_program(self)
        if subquery_str in memoize:
            return memoize[subquery_str], new_memoize
        if subquery_str in new_memoize:
            return new_memoize[subquery_str], new_memoize
        result = np.full((len(input[0]) + 1, len(input[0]) + 1), -np.inf)
        out = input[0, :, 1] + input[0, :, 3] > input[1, :, 1] + input[1, :, 3]
        indices = np.where(out)[0]
        result[indices, indices+1] = np.inf

        new_memoize[subquery_str] = result

        return result, new_memoize


class Behind(Predicate):
    has_theta=False

    def __init__(self):
        super().__init__("Behind")

    def execute(self, input, label, memoize, new_memoize):
        subquery_str = utils.print_program(self)
        if subquery_str in memoize:
            return memoize[subquery_str], new_memoize
        if subquery_str in new_memoize:
            return new_memoize[subquery_str], new_memoize
        result = np.full((len(input[0]) + 1, len(input[0]) + 1), -np.inf)
        out = input[0, :, 1] + input[0, :, 3] < input[1, :, 1] + input[1, :, 3]
        indices = np.where(out)[0]
        result[indices, indices+1] = np.inf

        new_memoize[subquery_str] = result

        return result, new_memoize


class RightOf(Predicate):
    has_theta=False

    def __init__(self):
        super().__init__("RightOf")

    def execute(self, input, label, memoize, new_memoize):
        subquery_str = utils.print_program(self)
        if subquery_str in memoize:
            return memoize[subquery_str], new_memoize
        if subquery_str in new_memoize:
            return new_memoize[subquery_str], new_memoize
        result = np.full((len(input[0]) + 1, len(input[0]) + 1), -np.inf)
        out = input[0, :, 0] + input[0, :, 2] > input[1, :, 0] + input[1, :, 2]
        indices = np.where(out)[0]
        result[indices, indices+1] = np.inf

        new_memoize[subquery_str] = result

        return result, new_memoize

class LeftOf(Predicate):
    has_theta=False

    def __init__(self):
        super().__init__("LeftOf")

    def execute(self, input, label, memoize, new_memoize):
        subquery_str = utils.print_program(self)
        if subquery_str in memoize:
            return memoize[subquery_str], new_memoize
        if subquery_str in new_memoize:
            return new_memoize[subquery_str], new_memoize
        result = np.full((len(input[0]) + 1, len(input[0]) + 1), -np.inf)
        out = input[0, :, 0] + input[0, :, 2] < input[1, :, 0] + input[1, :, 2]
        indices = np.where(out)[0]
        result[indices, indices+1] = np.inf

        new_memoize[subquery_str] = result

        return result, new_memoize

############### Warsaw Predicate ################

class AEastward4(Predicate):
    has_theta=False

    def __init__(self):
        super().__init__("AEastward4")

    def execute(self, input, label, memoize, new_memoize):
        subquery_str = utils.print_program(self)
        if subquery_str in memoize:
            return memoize[subquery_str], new_memoize
        if subquery_str in new_memoize:
            return new_memoize[subquery_str], new_memoize
        eastward_4 = np.array([[0, 330], [330, 362], [789, 369], [960, 355], [960, 372], [803, 391], [351, 389], [0, 351]])
        result = np.full((len(input[0]) + 1, len(input[0]) + 1), -np.inf)
        # Point((x1 + x2) / 2, y2 - 10)
        points = shapely.points((input[0, :, 0] + input[0, :, 2])/2, input[0, :, 3] - 10)
        out = shapely.contains(Polygon(eastward_4), points)
        indices = np.where(out)[0]
        result[indices, indices+1] = np.inf

        new_memoize[subquery_str] = result

        return result, new_memoize

class AEastward3(Predicate):
    has_theta=False

    def __init__(self):
        super().__init__("AEastward3")

    def execute(self, input, label, memoize, new_memoize):
        subquery_str = utils.print_program(self)
        if subquery_str in memoize:
            return memoize[subquery_str], new_memoize
        if subquery_str in new_memoize:
            return new_memoize[subquery_str], new_memoize
        eastward_3 = np.array([[0, 351], [351, 389], [803, 391], [960, 372], [960, 394], [838, 413], [424, 422], [153, 395], [0, 370]])
        result = np.full((len(input[0]) + 1, len(input[0]) + 1), -np.inf)
        # Point((x1 + x2) / 2, y2 - 10)
        points = shapely.points((input[0, :, 0] + input[0, :, 2])/2, input[0, :, 3] - 10)
        out = shapely.contains(Polygon(eastward_3), points)
        indices = np.where(out)[0]
        result[indices, indices+1] = np.inf

        new_memoize[subquery_str] = result

        return result, new_memoize

class AEastward2(Predicate):
    has_theta=False

    def __init__(self):
        super().__init__("AEastward2")

    def execute(self, input, label, memoize, new_memoize):
        subquery_str = utils.print_program(self)
        if subquery_str in memoize:
            return memoize[subquery_str], new_memoize
        if subquery_str in new_memoize:
            return new_memoize[subquery_str], new_memoize
        eastward_2 = np.array([[0, 370], [153, 395], [424, 422], [838, 413], [960, 394], [960, 420], [763, 451], [414, 460], [97, 420], [0, 397]])
        result = np.full((len(input[0]) + 1, len(input[0]) + 1), -np.inf)
        # Point((x1 + x2) / 2, y2 - 10)
        points = shapely.points((input[0, :, 0] + input[0, :, 2])/2, input[0, :, 3] - 10)
        out = shapely.contains(Polygon(eastward_2), points)
        indices = np.where(out)[0]
        result[indices, indices+1] = np.inf

        new_memoize[subquery_str] = result

        return result, new_memoize

class AWestward2(Predicate):
    has_theta=False

    def __init__(self):
        super().__init__("AWestward2")

    def execute(self, input, label, memoize, new_memoize):
        subquery_str = utils.print_program(self)
        if subquery_str in memoize:
            return memoize[subquery_str], new_memoize
        if subquery_str in new_memoize:
            return new_memoize[subquery_str], new_memoize
        westward_2 = np.array([[0, 262], [709, 213], [708, 242], [0, 288]])
        result = np.full((len(input[0]) + 1, len(input[0]) + 1), -np.inf)
        # Point((x1 + x2) / 2, y2 - 4)
        points = shapely.points((input[0, :, 0] + input[0, :, 2])/2, input[0, :, 3] - 4)
        out = shapely.contains(Polygon(westward_2), points)
        indices = np.where(out)[0]
        result[indices, indices+1] = np.inf

        new_memoize[subquery_str] = result

        return result, new_memoize

class ASouthward1Upper(Predicate):
    has_theta=False

    def __init__(self):
        super().__init__("ASouthward1Upper")

    def execute(self, input, label, memoize, new_memoize):
        subquery_str = utils.print_program(self)
        if subquery_str in memoize:
            return memoize[subquery_str], new_memoize
        if subquery_str in new_memoize:
            return new_memoize[subquery_str], new_memoize
        southward_1_upper = np.array([[384, 113], [414, 115], [565, 223], [543, 224]])
        result = np.full((len(input[0]) + 1, len(input[0]) + 1), -np.inf)
        # Point((x1 + x2) / 2, y2)
        points = shapely.points((input[0, :, 0] + input[0, :, 2])/2, input[0, :, 3])
        out = shapely.contains(Polygon(southward_1_upper), points)
        indices = np.where(out)[0]
        result[indices, indices+1] = np.inf

        new_memoize[subquery_str] = result

        return result, new_memoize

class BEastward4(Predicate):
    has_theta=False

    def __init__(self):
        super().__init__("BEastward4")

    def execute(self, input, label, memoize, new_memoize):
        subquery_str = utils.print_program(self)
        if subquery_str in memoize:
            return memoize[subquery_str], new_memoize
        if subquery_str in new_memoize:
            return new_memoize[subquery_str], new_memoize
        eastward_4 = np.array([[0, 330], [330, 362], [789, 369], [960, 355], [960, 372], [803, 391], [351, 389], [0, 351]])
        result = np.full((len(input[0]) + 1, len(input[0]) + 1), -np.inf)
        # Point((x1 + x2) / 2, y2 - 10)
        points = shapely.points((input[1, :, 0] + input[1, :, 2])/2, input[1, :, 3] - 10)
        out = shapely.contains(Polygon(eastward_4), points)
        indices = np.where(out)[0]
        result[indices, indices+1] = np.inf

        new_memoize[subquery_str] = result

        return result, new_memoize

class BEastward3(Predicate):
    has_theta=False

    def __init__(self):
        super().__init__("BEastward3")

    def execute(self, input, label, memoize, new_memoize):
        subquery_str = utils.print_program(self)
        if subquery_str in memoize:
            return memoize[subquery_str], new_memoize
        if subquery_str in new_memoize:
            return new_memoize[subquery_str], new_memoize
        eastward_3 = np.array([[0, 351], [351, 389], [803, 391], [960, 372], [960, 394], [838, 413], [424, 422], [153, 395], [0, 370]])
        result = np.full((len(input[0]) + 1, len(input[0]) + 1), -np.inf)
        # Point((x1 + x2) / 2, y2 - 10)
        points = shapely.points((input[1, :, 0] + input[1, :, 2])/2, input[1, :, 3] - 10)
        out = shapely.contains(Polygon(eastward_3), points)
        indices = np.where(out)[0]
        result[indices, indices+1] = np.inf

        new_memoize[subquery_str] = result

        return result, new_memoize

class BEastward2(Predicate):
    has_theta=False

    def __init__(self):
        super().__init__("BEastward2")

    def execute(self, input, label, memoize, new_memoize):
        subquery_str = utils.print_program(self)
        if subquery_str in memoize:
            return memoize[subquery_str], new_memoize
        if subquery_str in new_memoize:
            return new_memoize[subquery_str], new_memoize
        eastward_2 = np.array([[0, 370], [153, 395], [424, 422], [838, 413], [960, 394], [960, 420], [763, 451], [414, 460], [97, 420], [0, 397]])
        result = np.full((len(input[0]) + 1, len(input[0]) + 1), -np.inf)
        # Point((x1 + x2) / 2, y2 - 10)
        points = shapely.points((input[1, :, 0] + input[1, :, 2])/2, input[1, :, 3] - 10)
        out = shapely.contains(Polygon(eastward_2), points)
        indices = np.where(out)[0]
        result[indices, indices+1] = np.inf

        new_memoize[subquery_str] = result

        return result, new_memoize

class BWestward2(Predicate):
    has_theta=False

    def __init__(self):
        super().__init__("BWestward2")

    def execute(self, input, label, memoize, new_memoize):
        subquery_str = utils.print_program(self)
        if subquery_str in memoize:
            return memoize[subquery_str], new_memoize
        if subquery_str in new_memoize:
            return new_memoize[subquery_str], new_memoize
        westward_2 = np.array([[0, 262], [709, 213], [708, 242], [0, 288]])
        result = np.full((len(input[0]) + 1, len(input[0]) + 1), -np.inf)
        # Point((x1 + x2) / 2, y2 - 4)
        points = shapely.points((input[1, :, 0] + input[1, :, 2])/2, input[1, :, 3] - 4)
        out = shapely.contains(Polygon(westward_2), points)
        indices = np.where(out)[0]
        result[indices, indices+1] = np.inf

        new_memoize[subquery_str] = result

        return result, new_memoize

class BSouthward1Upper(Predicate):
    has_theta=False

    def __init__(self):
        super().__init__("BSouthward1Upper")

    def execute(self, input, label, memoize, new_memoize):
        subquery_str = utils.print_program(self)
        if subquery_str in memoize:
            return memoize[subquery_str], new_memoize
        if subquery_str in new_memoize:
            return new_memoize[subquery_str], new_memoize
        southward_1_upper = np.array([[384, 113], [414, 115], [565, 223], [543, 224]])
        result = np.full((len(input[0]) + 1, len(input[0]) + 1), -np.inf)
        # Point((x1 + x2) / 2, y2)
        points = shapely.points((input[1, :, 0] + input[1, :, 2])/2, input[1, :, 3])
        out = shapely.contains(Polygon(southward_1_upper), points)
        indices = np.where(out)[0]
        result[indices, indices+1] = np.inf

        new_memoize[subquery_str] = result

        return result, new_memoize

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
        subquery_str = utils.print_program(self)
        if subquery_str in memoize:
            return memoize[subquery_str], new_memoize
        if subquery_str in new_memoize:
            return new_memoize[subquery_str], new_memoize
        result = np.full((len(input[0]) + 1, len(input[0]) + 1), -np.inf)
        out = -1 * np.sqrt(np.square(input[0, :, 4]) + np.square(input[0, :, 5]))
        indices = np.arange(len(input[0]))
        result[indices, indices+1] = out
        new_memoize[subquery_str] = result
        return result, new_memoize

    def execute(self, input, label, memoize, new_memoize):
        if self.with_hole:
            return self.execute_with_hole(input, label, memoize, new_memoize)

        subquery_str = utils.print_program(self)
        out = np.sqrt(np.square(input[0, :, 4]) + np.square(input[0, :, 5]))
        indices = np.where(out <= -1 * self.theta)[0]
        result[indices, indices+1] = np.inf

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
        subquery_str = utils.print_program(self)
        if subquery_str in memoize:
            return memoize[subquery_str], new_memoize
        if subquery_str in new_memoize:
            return new_memoize[subquery_str], new_memoize
        result = np.full((len(input[1]) + 1, len(input[1]) + 1), -np.inf)
        out = -1 * np.sqrt(np.square(input[1, :, 4]) + np.square(input[1, :, 5]))
        indices = np.arange(len(input[1]))
        result[indices, indices+1] = out
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
        result = np.full((len(input[1]) + 1, len(input[1]) + 1), -np.inf)
        out = np.sqrt(np.square(input[1, :, 4]) + np.square(input[1, :, 5]))
        indices = np.where(out <= -1 * self.theta)[0]
        result[indices, indices+1] = np.inf

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
        subquery_str = utils.print_program(self)
        if subquery_str in memoize:
            return memoize[subquery_str], new_memoize
        if subquery_str in new_memoize:
            return new_memoize[subquery_str], new_memoize
        result = np.full((len(input[0]) + 1, len(input[0]) + 1), -np.inf)
        out = np.sqrt(np.square(input[0, :, 6]) + np.square(input[0, :, 7]))
        indices = np.arange(len(input[0]))
        result[indices, indices+1] = out
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
        out = np.sqrt(np.square(input[0, :, 6]) + np.square(input[0, :, 7]))
        indices = np.where(out >= self.theta)[0]
        result[indices, indices+1] = np.inf

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
        subquery_str = utils.print_program(self)
        if subquery_str in memoize:
            return memoize[subquery_str], new_memoize
        if subquery_str in new_memoize:
            return new_memoize[subquery_str], new_memoize
        result = np.full((len(input[1]) + 1, len(input[1]) + 1), -np.inf)
        out = np.sqrt(np.square(input[1, :, 6]) + np.square(input[1, :, 7]))
        indices = np.arange(len(input[1]))
        result[indices, indices+1] = out
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
        result = np.full((len(input[1]) + 1, len(input[1]) + 1), -np.inf)
        out = np.sqrt(np.square(input[1, :, 6]) + np.square(input[1, :, 7]))
        indices = np.where(out >= 2)[0]
        result[indices, indices+1] = np.inf

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
        out = -0.5 * np.sqrt(np.square(input[0, :, 0] + input[0, :, 2] - input[1, :, 0] - input[1, :, 2]) + np.square(input[0, :, 1] + input[0, :, 3] - input[1, :, 1] - input[1, :, 3]))
        indices = np.arange(len(input[1]))
        result[indices, indices+1] = out
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
        out = 0.5 * np.sqrt(np.square(input[0, :, 0] + input[0, :, 2] - input[1, :, 0] - input[1, :, 2]) + np.square(input[0, :, 1] + input[0, :, 3] - input[1, :, 1] - input[1, :, 3]))
        indices = np.where(out <= -1 * self.theta)[0]
        result[indices, indices+1] = np.inf

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
        out = (np.square(input[0, :, 4]) + np.square(input[0, :, 5])) / (np.square(input[1, :, 4]) + np.square(input[1, :, 5]) + 1e-6)
        indices = np.arange(len(input[0]))
        result[indices, indices+1] = out
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
        out = (np.square(input[0, :, 4]) + np.square(input[0, :, 5])) / (np.square(input[1, :, 4]) + np.square(input[1, :, 5]) + 1e-6)
        indices = np.where(out >= self.theta)[0]
        result[indices, indices+1] = np.inf
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