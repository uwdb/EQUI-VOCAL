python main.py --root_path /home/ubuntu/3D-ResNets-PyTorch/data --video_path static_traffic_R3D_jpgs \
--result_path results_static_traffic --n_classes 2 --n_pretrain_classes 700 \
--pretrain_path models/r3d50_K_200ep.pth --ft_begin_module fc \
--model resnet --model_depth 50 --batch_size 64 --n_threads 4 --checkpoint 5 --sample_duration 16 --train_crop center --learning_rate 0.001 --weight_decay 0.001 --sample_size 112 --multistep_milestones 50 100 150