from experiments.processing.prepare_data_postgres import prepare_data_given_target_query, construct_train_test

if __name__ == '__main__':
    query_strs = [
        "Duration(Far_3(o0, o1), 5); Near_1(o0, o1); Far_3(o0, o1)",
        "Duration(LeftOf(o0, o1), 5); (Near_1.0(o0, o1), TopQuadrant(o0)); Duration(RightOf(o0, o1), 5)",
        "Duration((FrontOf(o0, o1), LeftQuadrant(o0)), 15); Duration((LeftQuadrant(o0), RightOf(o0, o1), TopQuadrant(o0)), 5)"
        ]
    for query_str in query_strs:
        prepare_data_given_target_query(query_str, 0, 1, "trajectories_duration", "Obj_trajectories", sampling_rate=None)
        construct_train_test("/gscratch/balazinska/enhaoz/complex_event_video/inputs/trajectories_duration", n_train=500)