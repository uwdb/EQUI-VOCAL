python \
    ../src/synthesize.py \
    --method vocal_postgres `# options: vocal_postgres, vocal_postgres_no_active_learning`\
    --n_init_pos 5 \
    --n_init_neg 5 \
    --dataset_name "synthetic_scene_graph_easy" `# options: synthetic_scene_graph_easy, synthetic_scene_graph_medium, synthetic_scene_graph_hard`\
    --query_str "Conjunction(Conjunction(Conjunction(BottomQuadrant(o0), Color_green(o0)), Near_1.0(o0, o1)), RightOf(o0, o1))" `# An Easy query` \
    --npred 7 \
    --depth 3 \
    --max_duration 15 \
    --max_vars 3 \
    --beam_width 10 \
    --pool_size 100 \
    --k 100 \
    --budget 30 \
    --multithread 4 \
    --strategy "topk" \
    --run_id 0 \
    --output_dir "../outputs_test" \
    --reg_lambda 0.001