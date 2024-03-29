python \
    ../src/synthesize.py \
    --method vocal_postgres \
    --n_init_pos 2 \
    --n_init_neg 10 \
    --dataset_name "without_duration-sampling_rate_4" `# options: without_duration-sampling_rate_4, without_duration-sampling_rate_4-fn_error_rate_0.1-fp_error_rate_0.01, trajectories_handwritten`\
    --npred 5 \
    --depth 3 \
    --max_duration 1 \
    --max_vars 2 \
    --beam_width 10 \
    --pool_size 100 \
    --k 100 \
    --budget 20 \
    --multithread 1 \
    --strategy "topk" \
    --query_str "Conjunction(Near_1(o0, o1), BottomQuadrant(o0))" `# TQ1` \
    --run_id 0 \
    --output_dir "../outputs_test" \
    --reg_lambda 0.01