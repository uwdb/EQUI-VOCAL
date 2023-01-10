python \
    ../experiments/analysis/evaluate_vocal.py \
    --dataset_name "synthetic_scene_graph_without_duration-npred_3-nattr_pred_1-depth_1-40" `# options: synthetic_scene_graph_easy, synthetic_scene_graph_medium, synthetic_scene_graph_hard`\
    --query_str "Conjunction(Conjunction(Conjunction(BottomQuadrant(o0), Color_green(o0)), Near_1.0(o0, o1)), RightOf(o0, o1))" \
    --method "vocal_postgres-topk" `# options: vocal_postgres_no_active_learning-topk, vocal_postgres-topk` \
    --task_name "bw" `# options: trajectory, budget, bw, k, num_init, cpu, reg_lambda` \
    --multithread 4