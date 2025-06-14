# Evaluate on Adversarial Training (Based on [_Admix_](https://github.com/JHL-HUST/Admix) Code)

[中文](https://github.com/the-full/OPS/blob/main/AT_old/README_zh.md)

## 1. Introduction  

This folder contains the evaluation code based on the paper: [Ensemble Adversarial Training: Attacks and Defenses](https://arxiv.org/abs/1705.07204).  

## 2. Environment Setup  

To set up the `AT_old` environment, run the following command:  

```bash  
bash install.sh  
```

This script will automatically install all required dependencies, including third-party packages and environment configurations.  

Next, download [bigfile](https://drive.google.com/file/d/1uC9ZNWR7VuDlqPzMQTmC787KB7QrPofi/view?usp=drive_link), and place the files from the `AT_old` subfolder in the appropriate locations as follows:

| Source Path(bigfile/AT_old)            | Destination Path           |
|----------------------------------------|----------------------------|
| `*`                                    | `AT_old/models/*`          |

Ensure all files are correctly placed before proceeding to the next steps.

## 3. Running the Experiments  

After completing the environment setup, follow these steps to run the experiments:  

1. Enter the `AT_old` environment:

   ```bash  
   conda activate AT_old
   ```

2. **Generate adversarial examples using TransferAttack**  
   Refer to the README.md file in the `TransferAttack` folder for detailed instructions on how to generate adversarial examples.  

3. **Run the experiment scripts**  
   Execute one of the `run_exp_*.sh` scripts to evaluate the defense methods. For example:  

   ```bash  
   bash run_exp_defense_old_AT.sh  
   ```

   Each script will execute a specific evaluation. The correspondence between the experiment scripts and the experiments in the paper is as follows:  

| Script                               | Related Paper Figure/Table          |  
|--------------------------------------|-------------------------------------|  
| `run_exp_defense_old_AT.sh`          | Table 4 (Baseline)                  |  
| `run_exp_defense_old_AT_ops.sh`      | Table 4 (OPS) |  
