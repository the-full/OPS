# exp: linfty 0.06
echo "" >>result.txt

# White-box Attacks
# python run.py --multirun verbose=False attack@attacker=knn,geoa3,hit_adv evaluator.budget_type=linfty evaluator.budget=0.06 evaluator.hooks.1.exp_name=linfty_0.06

# Transfer Attacks
# python run.py --multirun verbose=False attack@attacker=advpc,aof,pf_attack evaluator.budget_type=linfty evaluator.budget=0.06 evaluator.hooks.1.exp_name=linfty_0.06

# Transfer Attacks (from images)
# python run.py --multirun verbose=False attack@attacker=fgsm,i_fgsm,ai_fgtm,mi_fgsm,ni_fgsm,vmi_fgsm evaluator.budget_type=linfty evaluator.budget=0.06 evaluator.hooks.1.exp_name=linfty_0.06

python run.py --multirun verbose=False attack@attacker=ops evaluator.budget_type=linfty evaluator.budget=0.06 evaluator.hooks.1.exp_name=linfty_0.06 +evaluator.hooks.1.atk_name=ops_10_05_05 attacker.num_sample_operator=5 attacker.num_sample_neighbor=5

echo "endexp: linfty 0.06" >>result.txt
