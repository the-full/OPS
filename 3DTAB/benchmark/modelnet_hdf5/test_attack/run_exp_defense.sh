# 设计实验，并编写实验脚本

echo "" >>result.txt

# SRS
python run.py --multirun verbose=False defense@defenser=srs attack@attacker=knn,geoa3,hit_adv evaluator.budget_type=point_linfty evaluator.budget=0.18 evaluator.hooks.1.exp_name=point_linfty_0.18
python run.py --multirun verbose=False defense@defenser=srs attack@attacker=advpc,aof,pf_attack evaluator.budget_type=point_linfty evaluator.budget=0.18 evaluator.hooks.1.exp_name=point_linfty_0.18
python run.py --multirun verbose=False defense@defenser=srs attack@attacker=ops evaluator.budget_type=point_linfty evaluator.budget=0.18 evaluator.hooks.1.exp_name=point_linfty_0.18 +evaluator.hooks.1.atk_name=ops_10_05_05 attacker.num_sample_operator=5 attacker.num_sample_neighbor=5
python run.py --multirun verbose=False defense@defenser=srs attack@attacker=ops evaluator.budget_type=point_linfty evaluator.budget=0.06 evaluator.hooks.1.exp_name=point_linfty_0.06 +evaluator.hooks.1.atk_name=ops_10_05_05 attacker.num_sample_operator=5 attacker.num_sample_neighbor=5

# SOR
python run.py --multirun verbose=False defense@defenser=sor attack@attacker=knn,geoa3,hit_adv evaluator.budget_type=point_linfty evaluator.budget=0.18 evaluator.hooks.1.exp_name=point_linfty_0.18
python run.py --multirun verbose=False defense@defenser=sor attack@attacker=advpc,aof,pf_attack evaluator.budget_type=point_linfty evaluator.budget=0.18 evaluator.hooks.1.exp_name=point_linfty_0.18
python run.py --multirun verbose=False defense@defenser=sor attack@attacker=ops evaluator.budget_type=point_linfty evaluator.budget=0.18 evaluator.hooks.1.exp_name=point_linfty_0.18 +evaluator.hooks.1.atk_name=ops_10_05_05 attacker.num_sample_operator=5 attacker.num_sample_neighbor=5
python run.py --multirun verbose=False defense@defenser=sor attack@attacker=ops evaluator.budget_type=point_linfty evaluator.budget=0.06 evaluator.hooks.1.exp_name=point_linfty_0.06 +evaluator.hooks.1.atk_name=ops_10_05_05 attacker.num_sample_operator=5 attacker.num_sample_neighbor=5

# Dup-Net 会增加点数，需要调小 batch size
python run.py --multirun verbose=False victim_models_batch_size.default=20 defense@defenser=dupnet attack@attacker=knn,geoa3,hit_adv evaluator.budget_type=point_linfty evaluator.budget=0.18 evaluator.hooks.1.exp_name=point_linfty_0.18
python run.py --multirun verbose=False victim_models_batch_size.default=20 defense@defenser=dupnet attack@attacker=advpc,aof,pf_attack evaluator.budget_type=point_linfty evaluator.budget=0.18 evaluator.hooks.1.exp_name=point_linfty_0.18
python run.py --multirun verbose=False victim_models_batch_size.default=20 defense@defenser=dupnet attack@attacker=ops evaluator.budget_type=point_linfty evaluator.budget=0.18 evaluator.hooks.1.exp_name=point_linfty_0.18 +evaluator.hooks.1.atk_name=ops_10_05_05 attacker.num_sample_operator=5 attacker.num_sample_neighbor=5
python run.py --multirun verbose=False victim_models_batch_size.default=20 defense@defenser=dupnet attack@attacker=ops evaluator.budget_type=point_linfty evaluator.budget=0.06 evaluator.hooks.1.exp_name=point_linfty_0.06 +evaluator.hooks.1.atk_name=ops_10_05_05 attacker.num_sample_operator=5 attacker.num_sample_neighbor=5

# IF-Defense
python run.py --multirun verbose=False defense@defenser=onet_opt attack@attacker=knn,geoa3,hit_adv evaluator.budget_type=point_linfty evaluator.budget=0.18 evaluator.hooks.1.exp_name=point_linfty_0.18
python run.py --multirun verbose=False defense@defenser=onet_opt attack@attacker=advpc,aof,pf_attack evaluator.budget_type=point_linfty evaluator.budget=0.18 evaluator.hooks.1.exp_name=point_linfty_0.18
python run.py --multirun verbose=False defense@defenser=onet_opt attack@attacker=ops evaluator.budget_type=point_linfty evaluator.budget=0.18 evaluator.hooks.1.exp_name=point_linfty_0.18 +evaluator.hooks.1.atk_name=ops_10_05_05 attacker.num_sample_operator=5 attacker.num_sample_neighbor=5
python run.py --multirun verbose=False defense@defenser=onet_opt attack@attacker=ops evaluator.budget_type=point_linfty evaluator.budget=0.06 evaluator.hooks.1.exp_name=point_linfty_0.06 +evaluator.hooks.1.atk_name=ops_10_05_05 attacker.num_sample_operator=5 attacker.num_sample_neighbor=5

echo "endexp: defense" >>result.txt
