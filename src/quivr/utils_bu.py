import logging
import os
import dsl_bu as dsl

def print_program(program):
    # if isinstance(program, dsl.Predicate) or isinstance(program, dsl.Hole):
    if issubclass(type(program), dsl.Predicate):
        if program.has_theta:
            if program.with_hole:
                return program.name + "_withHole"
            else:
                return program.name + "_" + str(abs(program.theta))
        else:
            return program.name
    if issubclass(type(program), dsl.Hole):
        return program.name
    if issubclass(type(program), dsl.DurationOperator):
        return program.name + "(" + print_program(program.submodules["duration"]) + ", " + str(program.theta) + ")"
    else:
        collected_names = []
        for submodule, functionclass in program.submodules.items():
            collected_names.append(print_program(functionclass))
        joined_names = ', '.join(collected_names)
        return program.name + "(" + joined_names + ")"

# convert a str back to an executable program; this is an inverse of print_program()
def str_to_program(program_str):
    if program_str.startswith("Near"):
        return dsl.Near(theta=-float(program_str.split("_")[1]))
    elif program_str.startswith("Far"):
        return dsl.Far(theta=float(program_str.split("_")[1]))
    elif program_str.startswith("MinLength"):
        return dsl.MinLength(theta=float(program_str.split("_")[1]))
    elif program_str.startswith("True*"):
        return dsl.TrueStar()
    else:
        idx = program_str.find("(")
        idx_r = program_str.rfind(")")
        # True*, Sequencing(Near_1.0, MinLength_10.0)
        functionclass = program_str[:idx]
        submodules = program_str[idx+1:idx_r]
        counter = 0
        submodule_list = []
        submodule_start = 0
        for i, char in enumerate(submodules):
            if char == "," and counter == 0:
                submodule_list.append(submodules[submodule_start:i])
                submodule_start = i+2
            elif char == "(":
                counter += 1
            elif char == ")":
                counter -= 1
        submodule_list.append(submodules[submodule_start:])
        if functionclass == "Sequencing":
            program_init = getattr(dsl, "SequencingOperator")
        elif functionclass == "Conjunction":
            program_init = getattr(dsl, "ConjunctionOperator")
        elif functionclass == "Kleene":
            program_init = getattr(dsl, "KleeneOperator")
        elif functionclass == "Duration":
            program_init = getattr(dsl, "DurationOperator")
            return program_init(str_to_program(submodule_list[0]), int(submodule_list[1]))
        submodule_list = [str_to_program(submodule) for submodule in submodule_list]
        return program_init(*submodule_list)