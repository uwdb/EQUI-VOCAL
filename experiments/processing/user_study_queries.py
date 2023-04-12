from experiments.processing.prepare_data_postgres import prepare_data_given_target_query, construct_train_test

from src.utils import str_to_program_postgres, rewrite_program_postgres

if __name__ == '__main__':
    query_programs = [
        [
            {'scene_graph': [{'predicate': 'Far', 'parameter': 3, 'variables': ['o0', 'o1']}, {'predicate': 'Color', 'parameter': 'red', 'variables': ['o0']}, {'predicate': 'Shape', 'parameter': 'cylinder', 'variables': ['o1']}], 'duration_constraint': 1},
            {'scene_graph': [{'predicate': 'Near', 'parameter': 1, 'variables': ['o0', 'o1']}], 'duration_constraint': 1}
        ],
        [
            {'scene_graph': [{'predicate': 'Far', 'parameter': 3, 'variables': ['o0', 'o1']}, {'predicate': 'Color', 'parameter': 'red', 'variables': ['o0']}, {'predicate': 'Shape', 'parameter': 'cylinder', 'variables': ['o1']}], 'duration_constraint': 1},
            {'scene_graph': [{'predicate': 'Near', 'parameter': 1, 'variables': ['o0', 'o1']}, {'predicate': 'TopQuadrant', 'parameter': None, 'variables': ['o2']}, {'predicate': 'RightQuadrant', 'parameter': None, 'variables': ['o2']}], 'duration_constraint': 1}
        ],
        [
            {'scene_graph': [{'predicate': 'Far', 'parameter': 3, 'variables': ['o0', 'o1']}, {'predicate': 'Color', 'parameter': 'red', 'variables': ['o0']}, {'predicate': 'Shape', 'parameter': 'cylinder', 'variables': ['o1']}], 'duration_constraint': 15},
            {'scene_graph': [{'predicate': 'Near', 'parameter': 1, 'variables': ['o0', 'o1']}, {'predicate': 'TopQuadrant', 'parameter': None, 'variables': ['o2']}, {'predicate': 'RightQuadrant', 'parameter': None, 'variables': ['o2']}], 'duration_constraint': 1}
        ],
        [
            {'scene_graph': [{'predicate': 'Color', 'parameter': 'purple', 'variables': ['o0']}, {'predicate': 'material', 'parameter': 'metal', 'variables': ['o0']}, {'predicate': 'BottomQuadrant', 'parameter': None, 'variables': ['o1']}, {'predicate': 'Behind', 'parameter': None, 'variables': ['o0', 'o1']}], 'duration_constraint': 1}
        ],
       [
            {'scene_graph': [{'predicate': 'Color', 'parameter': 'purple', 'variables': ['o0']}, {'predicate': 'material', 'parameter': 'metal', 'variables': ['o0']}, {'predicate': 'BottomQuadrant', 'parameter': None, 'variables': ['o1']}, {'predicate': 'Behind', 'parameter': None, 'variables': ['o0', 'o1']}], 'duration_constraint': 1},
            {'scene_graph': [{'predicate': 'TopQuadrant', 'parameter': None, 'variables': ['o1']}], 'duration_constraint': 1}
        ],
        [
            {'scene_graph': [{'predicate': 'Color', 'parameter': 'purple', 'variables': ['o0']}, {'predicate': 'material', 'parameter': 'metal', 'variables': ['o0']}, {'predicate': 'BottomQuadrant', 'parameter': None, 'variables': ['o1']}, {'predicate': 'Behind', 'parameter': None, 'variables': ['o0', 'o1']}], 'duration_constraint': 1},
            {'scene_graph': [{'predicate': 'TopQuadrant', 'parameter': None, 'variables': ['o1']}], 'duration_constraint': 1},
            {'scene_graph': [{'predicate': 'BottomQuadrant', 'parameter': None, 'variables': ['o2']}, {'predicate': 'RightQuadrant', 'parameter': None, 'variables': ['o2']}], 'duration_constraint': 15}
        ],
    ]
    for query_program in query_programs:
        dataset_name = "user_study_queries_scene_graph"
        query_str = rewrite_program_postgres(query_program)
        print(query_str)
        prepare_data_given_target_query(query_str, 0, 1, dataset_name, "Obj_clevrer", sampling_rate=None)
        construct_train_test("/gscratch/balazinska/enhaoz/complex_event_video/inputs/{}".format(dataset_name), n_train=500)


    # Easy: Conjunction(Conjunction(Color_red(o0), Far_3(o0, o1)), Shape_cylinder(o1)); Near_1(o0, o1)
    # Generated 1666 positive inputs and 8334 negative inputs
    # [
    #     {'scene_graph': [{'predicate': 'Far', 'parameter': 3, 'variables': ['o0', 'o1']}, {'predicate': 'Color', 'parameter': 'red', 'variables': ['o0']}, {'predicate': 'Shape', 'parameter': 'cylinder', 'variables': ['o1']}], 'duration_constraint': 1},
    #     {'scene_graph': [{'predicate': 'Near', 'parameter': 1, 'variables': ['o0', 'o1']}], 'duration_constraint': 1}
    # ]

    # Medium: Conjunction(Conjunction(Color_red(o0), Far_3(o0, o1)), Shape_cylinder(o1)); Conjunction(Conjunction(Near_1(o0, o1), RightQuadrant(o2)), TopQuadrant(o2))
    # Generated 1280 positive inputs and 8720 negative inputs
    # [
    #     {'scene_graph': [{'predicate': 'Far', 'parameter': 3, 'variables': ['o0', 'o1']}, {'predicate': 'Color', 'parameter': 'red', 'variables': ['o0']}, {'predicate': 'Shape', 'parameter': 'cylinder', 'variables': ['o1']}], 'duration_constraint': 1},
    #     {'scene_graph': [{'predicate': 'Near', 'parameter': 1, 'variables': ['o0', 'o1']}, {'predicate': 'TopQuadrant', 'parameter': None, 'variables': ['o2']}, {'predicate': 'RightQuadrant', 'parameter': None, 'variables': ['o2']}], 'duration_constraint': 1}
    # ]

    # Hard: Duration(Conjunction(Conjunction(Color_red(o0), Far_3(o0, o1)), Shape_cylinder(o1)), 15); Conjunction(Conjunction(Near_1(o0, o1), RightQuadrant(o2)), TopQuadrant(o2))
    # Generated 678 positive inputs and 9322 negative inputs
    # inputs_train 500
    # labels_train 500 34
    # inputs_test 9500
    # labels_test 9500 644
    # [
    #     {'scene_graph': [{'predicate': 'Far', 'parameter': 3, 'variables': ['o0', 'o1']}, {'predicate': 'Color', 'parameter': 'red', 'variables': ['o0']}, {'predicate': 'Shape', 'parameter': 'cylinder', 'variables': ['o1']}], 'duration_constraint': 15},
    #     {'scene_graph': [{'predicate': 'Near', 'parameter': 1, 'variables': ['o0', 'o1']}, {'predicate': 'TopQuadrant', 'parameter': None, 'variables': ['o2']}, {'predicate': 'RightQuadrant', 'parameter': None, 'variables': ['o2']}], 'duration_constraint': 1}
    # ]

    # Easy: Conjunction(Conjunction(Conjunction(Behind(o0, o1), BottomQuadrant(o1)), Color_purple(o0)), material_metal(o0))
    # Generated 2418 positive inputs and 7582 negative inputs
    # inputs_train 500
    # labels_train 500 121
    # inputs_test 9500
    # labels_test 9500 2297

    # Medium: Conjunction(Conjunction(Conjunction(Behind(o0, o1), BottomQuadrant(o1)), Color_purple(o0)), material_metal(o0)); TopQuadrant(o1)
    # Generated 1280 positive inputs and 8720 negative inputs
    # inputs_train 500
    # labels_train 500 64
    # inputs_test 9500
    # labels_test 9500 1216

    # Hard: Conjunction(Conjunction(Conjunction(Behind(o0, o1), BottomQuadrant(o1)), Color_purple(o0)), material_metal(o0)); TopQuadrant(o1); Duration(Conjunction(BottomQuadrant(o2), RightQuadrant(o2)), 15)
    # Generated 448 positive inputs and 9552 negative inputs