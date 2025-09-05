python \
    ../experiments/analysis/evaluate_vocal.py \
    --dataset_name "synthetic_scene_graph_easy" `# options: synthetic_scene_graph_easy, synthetic_scene_graph_medium, synthetic_scene_graph_hard`\
    --query_str "(BottomQuadrant(o0), Color(o0, 'green'), Near(o0, o1, 1.0), RightOf(o0, o1))" \
    --method "vocal_postgres-topk" `# options: vocal_postgres_no_active_learning-topk, vocal_postgres-topk` \
    --task_name "bw" `# options: trajectory, budget, bw, k, num_init, cpu, reg_lambda` \
    --multithread 4