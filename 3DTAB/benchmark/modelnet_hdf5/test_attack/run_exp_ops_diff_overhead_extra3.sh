# 设计实验，并编写实验脚本

echo "" >>result.txt

echo "1, 10" >>result.txt
python run.py verbose=False attack@attacker=ops evaluator.budget_type=linfty evaluator.budget=0.06 evaluator.hooks.1.exp_name=ops_diff_overhead_linfty_0.06 +evaluator.hooks.1.atk_name=ops_10_01_10 attacker.num_sample_operator=1 attacker.num_sample_neighbor=10
python run.py verbose=False attack@attacker=ops evaluator.budget_type=linfty evaluator.budget=0.06 evaluator.hooks.1.exp_name=ops_diff_overhead_linfty_0.06 +evaluator.hooks.1.atk_name=ops_10_01_15 attacker.num_sample_operator=1 attacker.num_sample_neighbor=15
python run.py verbose=False attack@attacker=ops evaluator.budget_type=linfty evaluator.budget=0.06 evaluator.hooks.1.exp_name=ops_diff_overhead_linfty_0.06 +evaluator.hooks.1.atk_name=ops_10_01_20 attacker.num_sample_operator=1 attacker.num_sample_neighbor=20
python run.py verbose=False attack@attacker=ops evaluator.budget_type=linfty evaluator.budget=0.06 evaluator.hooks.1.exp_name=ops_diff_overhead_linfty_0.06 +evaluator.hooks.1.atk_name=ops_10_01_25 attacker.num_sample_operator=1 attacker.num_sample_neighbor=25
python run.py verbose=False attack@attacker=ops evaluator.budget_type=linfty evaluator.budget=0.06 evaluator.hooks.1.exp_name=ops_diff_overhead_linfty_0.06 +evaluator.hooks.1.atk_name=ops_10_01_30 attacker.num_sample_operator=1 attacker.num_sample_neighbor=30

echo "2, 10" >>result.txt
python run.py verbose=False attack@attacker=ops evaluator.budget_type=linfty evaluator.budget=0.06 evaluator.hooks.1.exp_name=ops_diff_overhead_linfty_0.06 +evaluator.hooks.1.atk_name=ops_10_02_10 attacker.num_sample_operator=2 attacker.num_sample_neighbor=10
python run.py verbose=False attack@attacker=ops evaluator.budget_type=linfty evaluator.budget=0.06 evaluator.hooks.1.exp_name=ops_diff_overhead_linfty_0.06 +evaluator.hooks.1.atk_name=ops_10_02_15 attacker.num_sample_operator=2 attacker.num_sample_neighbor=15
python run.py verbose=False attack@attacker=ops evaluator.budget_type=linfty evaluator.budget=0.06 evaluator.hooks.1.exp_name=ops_diff_overhead_linfty_0.06 +evaluator.hooks.1.atk_name=ops_10_02_20 attacker.num_sample_operator=2 attacker.num_sample_neighbor=20
python run.py verbose=False attack@attacker=ops evaluator.budget_type=linfty evaluator.budget=0.06 evaluator.hooks.1.exp_name=ops_diff_overhead_linfty_0.06 +evaluator.hooks.1.atk_name=ops_10_02_25 attacker.num_sample_operator=2 attacker.num_sample_neighbor=25
python run.py verbose=False attack@attacker=ops evaluator.budget_type=linfty evaluator.budget=0.06 evaluator.hooks.1.exp_name=ops_diff_overhead_linfty_0.06 +evaluator.hooks.1.atk_name=ops_10_02_30 attacker.num_sample_operator=2 attacker.num_sample_neighbor=30

echo "3, 10" >>result.txt
python run.py verbose=False attack@attacker=ops evaluator.budget_type=linfty evaluator.budget=0.06 evaluator.hooks.1.exp_name=ops_diff_overhead_linfty_0.06 +evaluator.hooks.1.atk_name=ops_10_03_10 attacker.num_sample_operator=3 attacker.num_sample_neighbor=10
python run.py verbose=False attack@attacker=ops evaluator.budget_type=linfty evaluator.budget=0.06 evaluator.hooks.1.exp_name=ops_diff_overhead_linfty_0.06 +evaluator.hooks.1.atk_name=ops_10_03_15 attacker.num_sample_operator=3 attacker.num_sample_neighbor=15
python run.py verbose=False attack@attacker=ops evaluator.budget_type=linfty evaluator.budget=0.06 evaluator.hooks.1.exp_name=ops_diff_overhead_linfty_0.06 +evaluator.hooks.1.atk_name=ops_10_03_20 attacker.num_sample_operator=3 attacker.num_sample_neighbor=20
python run.py verbose=False attack@attacker=ops evaluator.budget_type=linfty evaluator.budget=0.06 evaluator.hooks.1.exp_name=ops_diff_overhead_linfty_0.06 +evaluator.hooks.1.atk_name=ops_10_03_25 attacker.num_sample_operator=3 attacker.num_sample_neighbor=25
python run.py verbose=False attack@attacker=ops evaluator.budget_type=linfty evaluator.budget=0.06 evaluator.hooks.1.exp_name=ops_diff_overhead_linfty_0.06 +evaluator.hooks.1.atk_name=ops_10_03_30 attacker.num_sample_operator=3 attacker.num_sample_neighbor=30

echo "4, 10" >>result.txt
python run.py verbose=False attack@attacker=ops evaluator.budget_type=linfty evaluator.budget=0.06 evaluator.hooks.1.exp_name=ops_diff_overhead_linfty_0.06 +evaluator.hooks.1.atk_name=ops_10_04_10 attacker.num_sample_operator=4 attacker.num_sample_neighbor=10
python run.py verbose=False attack@attacker=ops evaluator.budget_type=linfty evaluator.budget=0.06 evaluator.hooks.1.exp_name=ops_diff_overhead_linfty_0.06 +evaluator.hooks.1.atk_name=ops_10_04_15 attacker.num_sample_operator=4 attacker.num_sample_neighbor=15
python run.py verbose=False attack@attacker=ops evaluator.budget_type=linfty evaluator.budget=0.06 evaluator.hooks.1.exp_name=ops_diff_overhead_linfty_0.06 +evaluator.hooks.1.atk_name=ops_10_04_20 attacker.num_sample_operator=4 attacker.num_sample_neighbor=20
python run.py verbose=False attack@attacker=ops evaluator.budget_type=linfty evaluator.budget=0.06 evaluator.hooks.1.exp_name=ops_diff_overhead_linfty_0.06 +evaluator.hooks.1.atk_name=ops_10_04_25 attacker.num_sample_operator=4 attacker.num_sample_neighbor=25
python run.py verbose=False attack@attacker=ops evaluator.budget_type=linfty evaluator.budget=0.06 evaluator.hooks.1.exp_name=ops_diff_overhead_linfty_0.06 +evaluator.hooks.1.atk_name=ops_10_04_30 attacker.num_sample_operator=4 attacker.num_sample_neighbor=30

echo "10, 1" >>result.txt
python run.py verbose=False attack@attacker=ops evaluator.budget_type=linfty evaluator.budget=0.06 evaluator.hooks.1.exp_name=ops_diff_overhead_linfty_0.06 +evaluator.hooks.1.atk_name=ops_10_10_01 attacker.num_sample_operator=10 attacker.num_sample_neighbor=1
python run.py verbose=False attack@attacker=ops evaluator.budget_type=linfty evaluator.budget=0.06 evaluator.hooks.1.exp_name=ops_diff_overhead_linfty_0.06 +evaluator.hooks.1.atk_name=ops_10_10_02 attacker.num_sample_operator=10 attacker.num_sample_neighbor=2
python run.py verbose=False attack@attacker=ops evaluator.budget_type=linfty evaluator.budget=0.06 evaluator.hooks.1.exp_name=ops_diff_overhead_linfty_0.06 +evaluator.hooks.1.atk_name=ops_10_10_03 attacker.num_sample_operator=10 attacker.num_sample_neighbor=3
python run.py verbose=False attack@attacker=ops evaluator.budget_type=linfty evaluator.budget=0.06 evaluator.hooks.1.exp_name=ops_diff_overhead_linfty_0.06 +evaluator.hooks.1.atk_name=ops_10_10_04 attacker.num_sample_operator=10 attacker.num_sample_neighbor=4

echo "15, 1" >>result.txt
python run.py verbose=False attack@attacker=ops evaluator.budget_type=linfty evaluator.budget=0.06 evaluator.hooks.1.exp_name=ops_diff_overhead_linfty_0.06 +evaluator.hooks.1.atk_name=ops_10_15_01 attacker.num_sample_operator=15 attacker.num_sample_neighbor=1
python run.py verbose=False attack@attacker=ops evaluator.budget_type=linfty evaluator.budget=0.06 evaluator.hooks.1.exp_name=ops_diff_overhead_linfty_0.06 +evaluator.hooks.1.atk_name=ops_10_15_02 attacker.num_sample_operator=15 attacker.num_sample_neighbor=2
python run.py verbose=False attack@attacker=ops evaluator.budget_type=linfty evaluator.budget=0.06 evaluator.hooks.1.exp_name=ops_diff_overhead_linfty_0.06 +evaluator.hooks.1.atk_name=ops_10_15_03 attacker.num_sample_operator=15 attacker.num_sample_neighbor=3
python run.py verbose=False attack@attacker=ops evaluator.budget_type=linfty evaluator.budget=0.06 evaluator.hooks.1.exp_name=ops_diff_overhead_linfty_0.06 +evaluator.hooks.1.atk_name=ops_10_15_04 attacker.num_sample_operator=15 attacker.num_sample_neighbor=4

echo "20, 1" >>result.txt
python run.py verbose=False attack@attacker=ops evaluator.budget_type=linfty evaluator.budget=0.06 evaluator.hooks.1.exp_name=ops_diff_overhead_linfty_0.06 +evaluator.hooks.1.atk_name=ops_10_20_01 attacker.num_sample_operator=20 attacker.num_sample_neighbor=1
python run.py verbose=False attack@attacker=ops evaluator.budget_type=linfty evaluator.budget=0.06 evaluator.hooks.1.exp_name=ops_diff_overhead_linfty_0.06 +evaluator.hooks.1.atk_name=ops_10_20_02 attacker.num_sample_operator=20 attacker.num_sample_neighbor=2
python run.py verbose=False attack@attacker=ops evaluator.budget_type=linfty evaluator.budget=0.06 evaluator.hooks.1.exp_name=ops_diff_overhead_linfty_0.06 +evaluator.hooks.1.atk_name=ops_10_20_03 attacker.num_sample_operator=20 attacker.num_sample_neighbor=3
python run.py verbose=False attack@attacker=ops evaluator.budget_type=linfty evaluator.budget=0.06 evaluator.hooks.1.exp_name=ops_diff_overhead_linfty_0.06 +evaluator.hooks.1.atk_name=ops_10_20_04 attacker.num_sample_operator=20 attacker.num_sample_neighbor=4

echo "25, 1" >>result.txt
python run.py verbose=False attack@attacker=ops evaluator.budget_type=linfty evaluator.budget=0.06 evaluator.hooks.1.exp_name=ops_diff_overhead_linfty_0.06 +evaluator.hooks.1.atk_name=ops_10_25_01 attacker.num_sample_operator=25 attacker.num_sample_neighbor=1
python run.py verbose=False attack@attacker=ops evaluator.budget_type=linfty evaluator.budget=0.06 evaluator.hooks.1.exp_name=ops_diff_overhead_linfty_0.06 +evaluator.hooks.1.atk_name=ops_10_25_02 attacker.num_sample_operator=25 attacker.num_sample_neighbor=2
python run.py verbose=False attack@attacker=ops evaluator.budget_type=linfty evaluator.budget=0.06 evaluator.hooks.1.exp_name=ops_diff_overhead_linfty_0.06 +evaluator.hooks.1.atk_name=ops_10_25_03 attacker.num_sample_operator=25 attacker.num_sample_neighbor=3
python run.py verbose=False attack@attacker=ops evaluator.budget_type=linfty evaluator.budget=0.06 evaluator.hooks.1.exp_name=ops_diff_overhead_linfty_0.06 +evaluator.hooks.1.atk_name=ops_10_25_04 attacker.num_sample_operator=25 attacker.num_sample_neighbor=4

echo "30, 1" >>result.txt
python run.py verbose=False attack@attacker=ops evaluator.budget_type=linfty evaluator.budget=0.06 evaluator.hooks.1.exp_name=ops_diff_overhead_linfty_0.06 +evaluator.hooks.1.atk_name=ops_10_30_01 attacker.num_sample_operator=30 attacker.num_sample_neighbor=1
python run.py verbose=False attack@attacker=ops evaluator.budget_type=linfty evaluator.budget=0.06 evaluator.hooks.1.exp_name=ops_diff_overhead_linfty_0.06 +evaluator.hooks.1.atk_name=ops_10_30_02 attacker.num_sample_operator=30 attacker.num_sample_neighbor=2
python run.py verbose=False attack@attacker=ops evaluator.budget_type=linfty evaluator.budget=0.06 evaluator.hooks.1.exp_name=ops_diff_overhead_linfty_0.06 +evaluator.hooks.1.atk_name=ops_10_30_03 attacker.num_sample_operator=30 attacker.num_sample_neighbor=3
python run.py verbose=False attack@attacker=ops evaluator.budget_type=linfty evaluator.budget=0.06 evaluator.hooks.1.exp_name=ops_diff_overhead_linfty_0.06 +evaluator.hooks.1.atk_name=ops_10_30_04 attacker.num_sample_operator=30 attacker.num_sample_neighbor=4

echo "endexp: ops_diff_overhead_extra3" >>result.txt
