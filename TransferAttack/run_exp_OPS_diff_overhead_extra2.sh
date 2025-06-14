python main_for_ops.py --input_dir ./new_data/ --output_dir exp_data/exp_ops_overhead/ops_onely_ns_10_01/resnet18 --attack ops_only_ns --model resnet18 --GPU_ID $1 --epoch 10 --num_sample_neighbor 1 --num_sample_operator 1
python main_for_ops.py --input_dir ./new_data/ --output_dir exp_data/exp_ops_overhead/ops_onely_ns_10_02/resnet18 --attack ops_only_ns --model resnet18 --GPU_ID $1 --epoch 10 --num_sample_neighbor 2 --num_sample_operator 1
python main_for_ops.py --input_dir ./new_data/ --output_dir exp_data/exp_ops_overhead/ops_onely_ns_10_03/resnet18 --attack ops_only_ns --model resnet18 --GPU_ID $1 --epoch 10 --num_sample_neighbor 3 --num_sample_operator 1
python main_for_ops.py --input_dir ./new_data/ --output_dir exp_data/exp_ops_overhead/ops_onely_ns_10_04/resnet18 --attack ops_only_ns --model resnet18 --GPU_ID $1 --epoch 10 --num_sample_neighbor 4 --num_sample_operator 1
python main_for_ops.py --input_dir ./new_data/ --output_dir exp_data/exp_ops_overhead/ops_onely_ns_10_05/resnet18 --attack ops_only_ns --model resnet18 --GPU_ID $1 --epoch 10 --num_sample_neighbor 5 --num_sample_operator 1
python main_for_ops.py --input_dir ./new_data/ --output_dir exp_data/exp_ops_overhead/ops_onely_ns_10_10/resnet18 --attack ops_only_ns --model resnet18 --GPU_ID $1 --epoch 10 --num_sample_neighbor 10 --num_sample_operator 1
python main_for_ops.py --input_dir ./new_data/ --output_dir exp_data/exp_ops_overhead/ops_onely_ns_10_15/resnet18 --attack ops_only_ns --model resnet18 --GPU_ID $1 --epoch 10 --num_sample_neighbor 15 --num_sample_operator 1
python main_for_ops.py --input_dir ./new_data/ --output_dir exp_data/exp_ops_overhead/ops_onely_ns_10_20/resnet18 --attack ops_only_ns --model resnet18 --GPU_ID $1 --epoch 10 --num_sample_neighbor 20 --num_sample_operator 1
python main_for_ops.py --input_dir ./new_data/ --output_dir exp_data/exp_ops_overhead/ops_onely_ns_10_25/resnet18 --attack ops_only_ns --model resnet18 --GPU_ID $1 --epoch 10 --num_sample_neighbor 25 --num_sample_operator 1
python main_for_ops.py --input_dir ./new_data/ --output_dir exp_data/exp_ops_overhead/ops_onely_ns_10_30/resnet18 --attack ops_only_ns --model resnet18 --GPU_ID $1 --epoch 10 --num_sample_neighbor 30 --num_sample_operator 1

python main_for_ops.py --input_dir ./new_data/ --output_dir exp_data/exp_ops_overhead/ops_onely_ns_10_01/resnet18 --eval --GPU_ID $1
python main_for_ops.py --input_dir ./new_data/ --output_dir exp_data/exp_ops_overhead/ops_onely_ns_10_02/resnet18 --eval --GPU_ID $1
python main_for_ops.py --input_dir ./new_data/ --output_dir exp_data/exp_ops_overhead/ops_onely_ns_10_03/resnet18 --eval --GPU_ID $1
python main_for_ops.py --input_dir ./new_data/ --output_dir exp_data/exp_ops_overhead/ops_onely_ns_10_04/resnet18 --eval --GPU_ID $1
python main_for_ops.py --input_dir ./new_data/ --output_dir exp_data/exp_ops_overhead/ops_onely_ns_10_05/resnet18 --eval --GPU_ID $1
python main_for_ops.py --input_dir ./new_data/ --output_dir exp_data/exp_ops_overhead/ops_onely_ns_10_10/resnet18 --eval --GPU_ID $1
python main_for_ops.py --input_dir ./new_data/ --output_dir exp_data/exp_ops_overhead/ops_onely_ns_10_15/resnet18 --eval --GPU_ID $1
python main_for_ops.py --input_dir ./new_data/ --output_dir exp_data/exp_ops_overhead/ops_onely_ns_10_20/resnet18 --eval --GPU_ID $1
python main_for_ops.py --input_dir ./new_data/ --output_dir exp_data/exp_ops_overhead/ops_onely_ns_10_25/resnet18 --eval --GPU_ID $1
python main_for_ops.py --input_dir ./new_data/ --output_dir exp_data/exp_ops_overhead/ops_onely_ns_10_30/resnet18 --eval --GPU_ID $1

python main_for_ops.py --input_dir ./new_data/ --output_dir exp_data/exp_ops_overhead/ops_only_os_10_01/resnet18 --attack ops --model resnet18 --GPU_ID $1 --epoch 10 --beta 0. --num_sample_neighbor 1 --num_sample_operator 1
python main_for_ops.py --input_dir ./new_data/ --output_dir exp_data/exp_ops_overhead/ops_only_os_10_02/resnet18 --attack ops --model resnet18 --GPU_ID $1 --epoch 10 --beta 0. --num_sample_neighbor 1 --num_sample_operator 2
python main_for_ops.py --input_dir ./new_data/ --output_dir exp_data/exp_ops_overhead/ops_only_os_10_03/resnet18 --attack ops --model resnet18 --GPU_ID $1 --epoch 10 --beta 0. --num_sample_neighbor 1 --num_sample_operator 3
python main_for_ops.py --input_dir ./new_data/ --output_dir exp_data/exp_ops_overhead/ops_only_os_10_04/resnet18 --attack ops --model resnet18 --GPU_ID $1 --epoch 10 --beta 0. --num_sample_neighbor 1 --num_sample_operator 4
python main_for_ops.py --input_dir ./new_data/ --output_dir exp_data/exp_ops_overhead/ops_only_os_10_05/resnet18 --attack ops --model resnet18 --GPU_ID $1 --epoch 10 --beta 0. --num_sample_neighbor 1 --num_sample_operator 5
python main_for_ops.py --input_dir ./new_data/ --output_dir exp_data/exp_ops_overhead/ops_only_os_10_10/resnet18 --attack ops --model resnet18 --GPU_ID $1 --epoch 10 --beta 0. --num_sample_neighbor 1 --num_sample_operator 10
python main_for_ops.py --input_dir ./new_data/ --output_dir exp_data/exp_ops_overhead/ops_only_os_10_15/resnet18 --attack ops --model resnet18 --GPU_ID $1 --epoch 10 --beta 0. --num_sample_neighbor 1 --num_sample_operator 15
python main_for_ops.py --input_dir ./new_data/ --output_dir exp_data/exp_ops_overhead/ops_only_os_10_20/resnet18 --attack ops --model resnet18 --GPU_ID $1 --epoch 10 --beta 0. --num_sample_neighbor 1 --num_sample_operator 20
python main_for_ops.py --input_dir ./new_data/ --output_dir exp_data/exp_ops_overhead/ops_only_os_10_25/resnet18 --attack ops --model resnet18 --GPU_ID $1 --epoch 10 --beta 0. --num_sample_neighbor 1 --num_sample_operator 25
python main_for_ops.py --input_dir ./new_data/ --output_dir exp_data/exp_ops_overhead/ops_only_os_10_30/resnet18 --attack ops --model resnet18 --GPU_ID $1 --epoch 10 --beta 0. --num_sample_neighbor 1 --num_sample_operator 30

python main_for_ops.py --input_dir ./new_data/ --output_dir exp_data/exp_ops_overhead/ops_only_os_10_01/resnet18 --eval --GPU_ID $1
python main_for_ops.py --input_dir ./new_data/ --output_dir exp_data/exp_ops_overhead/ops_only_os_10_02/resnet18 --eval --GPU_ID $1
python main_for_ops.py --input_dir ./new_data/ --output_dir exp_data/exp_ops_overhead/ops_only_os_10_03/resnet18 --eval --GPU_ID $1
python main_for_ops.py --input_dir ./new_data/ --output_dir exp_data/exp_ops_overhead/ops_only_os_10_04/resnet18 --eval --GPU_ID $1
python main_for_ops.py --input_dir ./new_data/ --output_dir exp_data/exp_ops_overhead/ops_only_os_10_05/resnet18 --eval --GPU_ID $1
python main_for_ops.py --input_dir ./new_data/ --output_dir exp_data/exp_ops_overhead/ops_only_os_10_10/resnet18 --eval --GPU_ID $1
python main_for_ops.py --input_dir ./new_data/ --output_dir exp_data/exp_ops_overhead/ops_only_os_10_15/resnet18 --eval --GPU_ID $1
python main_for_ops.py --input_dir ./new_data/ --output_dir exp_data/exp_ops_overhead/ops_only_os_10_20/resnet18 --eval --GPU_ID $1
python main_for_ops.py --input_dir ./new_data/ --output_dir exp_data/exp_ops_overhead/ops_only_os_10_25/resnet18 --eval --GPU_ID $1
python main_for_ops.py --input_dir ./new_data/ --output_dir exp_data/exp_ops_overhead/ops_only_os_10_30/resnet18 --eval --GPU_ID $1
