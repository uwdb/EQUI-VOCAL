from experiments.processing.prepare_data_postgres import prepare_data_given_target_query, construct_train_test

from src.utils import str_to_program_postgres, rewrite_program_postgres

if __name__ == '__main__':
    query_program = [
        {'scene_graph': [{'predicate': 'LeftOf', 'parameter': None, 'variables': ['o0', 'o1']}, {'predicate': 'Color', 'parameter': 'red', 'variables': ['o0']}, {'predicate': 'Color', 'parameter': 'yellow', 'variables': ['o1']}], 'duration_constraint': 1},
        {'scene_graph': [{'predicate': 'RightOf', 'parameter': None, 'variables': ['o0', 'o1']}], 'duration_constraint': 1}
    ]

    dataset_name = "demo_queries_scene_graph"
    query_str = rewrite_program_postgres(query_program)
    prepare_data_given_target_query(query_str, 0, 1, dataset_name, "Obj_clevrer", sampling_rate=None)
    construct_train_test("/gscratch/balazinska/enhaoz/complex_event_video/inputs/{}".format(dataset_name), n_train=500)