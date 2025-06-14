echo "" >>result.txt
python run.py verbose=False evaluator.hooks.1.type=none
echo "endexp: no attack" >>result.txt

# exp: linfty 0.18
echo "" >>result.txt

# White-box Attacks
python run.py --multirun verbose=False attack@attacker=knn,geoa3,hit_adv evaluator.budget_type=point_linfty evaluator.budget=0.18 evaluator.hooks.1.exp_name=point_linfty_0.18

# Transfer Attacks
python run.py --multirun verbose=False attack@attacker=advpc,aof,pf_attack evaluator.budget_type=point_linfty evaluator.budget=0.18 evaluator.hooks.1.exp_name=point_linfty_0.18

# Transfer Attacks (from images)
# python run.py --multirun verbose=False attack@attacker=fgsm,i_fgsm,ai_fgtm,mi_fgsm,ni_fgsm,vmi_fgsm evaluator.budget_type=point_linfty evaluator.budget=0.18 evaluator.hooks.1.exp_name=point_linfty_0.18

# Ours
python run.py --multirun verbose=False attack@attacker=ops evaluator.budget_type=point_linfty evaluator.budget=0.18 evaluator.hooks.1.exp_name=point_linfty_0.18 +evaluator.hooks.1.atk_name=ops_10_05_05 attacker.num_sample_operator=5 attacker.num_sample_neighbor=5

echo "endexp: point linfty 0.18" >>result.txt
