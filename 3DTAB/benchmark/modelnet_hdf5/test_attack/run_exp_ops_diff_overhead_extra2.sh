# 设计实验，并编写实验脚本

echo "" >>result.txt

python run.py verbose=False attack@attacker=ops evaluator.budget_type=linfty evaluator.budget=0.06 evaluator.hooks.1.exp_name=ops_diff_overhead_linfty_0.06 +evaluator.hooks.1.atk_name=ops_10_01_01 attacker.num_sample_operator=1 attacker.num_sample_neighbor=1
python run.py verbose=False attack@attacker=ops evaluator.budget_type=linfty evaluator.budget=0.06 evaluator.hooks.1.exp_name=ops_diff_overhead_linfty_0.06 +evaluator.hooks.1.atk_name=ops_10_02_02 attacker.num_sample_operator=2 attacker.num_sample_neighbor=2
python run.py verbose=False attack@attacker=ops evaluator.budget_type=linfty evaluator.budget=0.06 evaluator.hooks.1.exp_name=ops_diff_overhead_linfty_0.06 +evaluator.hooks.1.atk_name=ops_10_03_03 attacker.num_sample_operator=3 attacker.num_sample_neighbor=3
python run.py verbose=False attack@attacker=ops evaluator.budget_type=linfty evaluator.budget=0.06 evaluator.hooks.1.exp_name=ops_diff_overhead_linfty_0.06 +evaluator.hooks.1.atk_name=ops_10_04_04 attacker.num_sample_operator=4 attacker.num_sample_neighbor=4
python run.py verbose=False attack@attacker=ops evaluator.budget_type=linfty evaluator.budget=0.06 evaluator.hooks.1.exp_name=ops_diff_overhead_linfty_0.06 +evaluator.hooks.1.atk_name=ops_10_05_05 attacker.num_sample_operator=5 attacker.num_sample_neighbor=5
python run.py verbose=False attack@attacker=ops evaluator.budget_type=linfty evaluator.budget=0.06 evaluator.hooks.1.exp_name=ops_diff_overhead_linfty_0.06 +evaluator.hooks.1.atk_name=ops_10_06_06 attacker.num_sample_operator=6 attacker.num_sample_neighbor=6
python run.py verbose=False attack@attacker=ops evaluator.budget_type=linfty evaluator.budget=0.06 evaluator.hooks.1.exp_name=ops_diff_overhead_linfty_0.06 +evaluator.hooks.1.atk_name=ops_10_07_07 attacker.num_sample_operator=7 attacker.num_sample_neighbor=7
python run.py verbose=False attack@attacker=ops evaluator.budget_type=linfty evaluator.budget=0.06 evaluator.hooks.1.exp_name=ops_diff_overhead_linfty_0.06 +evaluator.hooks.1.atk_name=ops_10_08_08 attacker.num_sample_operator=8 attacker.num_sample_neighbor=8
python run.py verbose=False attack@attacker=ops evaluator.budget_type=linfty evaluator.budget=0.06 evaluator.hooks.1.exp_name=ops_diff_overhead_linfty_0.06 +evaluator.hooks.1.atk_name=ops_10_09_09 attacker.num_sample_operator=9 attacker.num_sample_neighbor=9
python run.py verbose=False attack@attacker=ops evaluator.budget_type=linfty evaluator.budget=0.06 evaluator.hooks.1.exp_name=ops_diff_overhead_linfty_0.06 +evaluator.hooks.1.atk_name=ops_10_10_10 attacker.num_sample_operator=10 attacker.num_sample_neighbor=10

echo "endexp: ops_diff_overhead_extra2" >>result.txt
