# 设计实验，并编写实验脚本

# exp1: linfty 0.06
# exp2: linfty 0.045

#echo "" >>result.txt
#python run.py verbose=False evaluator.hooks.1.type=none
#echo "" >>result.txt
#
## exp1: point linfty 0.18
#echo "" >>result.txt
## 白盒攻击的可转移性
#python run.py --multirun verbose=False attack@attacker=knn,geoa3,hit_adv evaluator.budget_type=point_linfty evaluator.budget=0.18 evaluator.hooks.1.exp_name=point_linfty_0.18
## 转移攻击的可转移性(原生)
#python run.py --multirun verbose=False attack@attacker=advpc,aof,pf_attack evaluator.budget_type=point_linfty evaluator.budget=0.18 evaluator.hooks.1.exp_name=point_linfty_0.18
# 转移攻击的可转移性(图像移植而来)
python run.py --multirun verbose=False attack@attacker=fgsm,i_fgsm,ai_fgtm,mi_fgsm,ni_fgsm,vmi_fgsm,gra,pgn evaluator.budget_type=point_linfty evaluator.budget=0.18 evaluator.hooks.1.exp_name=point_linfty_0.18
# 我们的方法
python run.py --multirun verbose=False attack@attacker=ops evaluator.budget_type=point_linfty evaluator.budget=0.18 evaluator.hooks.1.exp_name=point_linfty_0.18 +evaluator.hooks.1.atk_name=ops_10_05_05 attacker.num_sample_operator=5 attacker.num_sample_neighbor=5
python run.py --multirun verbose=False attack@attacker=ops evaluator.budget_type=point_linfty evaluator.budget=0.18 evaluator.hooks.1.exp_name=point_linfty_0.18 +evaluator.hooks.1.atk_name=ops_10_10_10 attacker.num_sample_operator=10 attacker.num_sample_neighbor=10
python run.py --multirun verbose=False attack@attacker=ops evaluator.budget_type=point_linfty evaluator.budget=0.18 evaluator.hooks.1.exp_name=point_linfty_0.18 +evaluator.hooks.1.atk_name=ops_10_20_20 attacker.num_sample_operator=20 attacker.num_sample_neighbor=20
python run.py --multirun verbose=False attack@attacker=ops evaluator.budget_type=point_linfty evaluator.budget=0.18 evaluator.hooks.1.exp_name=point_linfty_0.18 +evaluator.hooks.1.atk_name=ops_10_30_30 attacker.num_sample_operator=30 attacker.num_sample_neighbor=30

echo "endexp3: point linfty 0.18" >>result.txt
