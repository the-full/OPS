# 3D Transfer Attack Benchmark (Demo)

[中文](https://github.com/the-full/OPS/blob/main/3DTAB/README_zh.md)

## 1. Introduction

3DTAB (3D Transfer Attack Benchmark) is a benchmark designed to evaluate adversarial attack and defense methods on 3D point clouds. Although initially designed for assessing transfer attacks, it can also be utilized to evaluate white-box attack methods in practice.

This README outlines the steps to set up the environment, run experiments, and provides an overview of the file structure within the project.

## 2. Environment Setup

To set up the `3DTAB` environment, run the following command to install the necessary environment:

```bash
bash install.sh
```

This script will automatically install all the required dependencies, including any third-party packages and environment configurations.

Next, download [bigfile](https://drive.google.com/file/d/1uC9ZNWR7VuDlqPzMQTmC787KB7QrPofi/view?usp=drive_link), and place the files from the `3DTAB` subfolder in the appropriate locations as follows:

| Source Path(bigfile/3DTAB)             | Destination Path                                                                                    |
|----------------------------------------|-----------------------------------------------------------------------------------------------------|
| `modelnet40_ply_hdf5_2048`             | `3DTAB/asset/dataset/modelnet40_ply_hdf5_2048`                                                      |
| `modelnet40_ply_hdf5_2048_mini`        | `3DTAB/asset/dataset/modelnet40_ply_hdf5_2048_mini`                                                 |
| `ModelNetHdf5`                         | `3DTAB/asset/model_ckpt/ModelNetHdf5`                                                               |
| `pretrain/AdvPC/mn40`                  | `3DTAB/packages/ATK/ATK/attack/AdvPC/pretrain/mn40`                                                 |
| `pretrain/DUP_Net/*`                   | `3DTAB/packages/ATK/ATK/defense/DUP_Net/pretrain/*`                                               |
| `pretrain/IF-Defense/*`                | `3DTAB/packages/ATK/ATK/defense/IF-Defense/pretrain/*`                                            |

Ensure all files are correctly placed before proceeding to the next steps.

## 3. Running the Experiments

After setting up the environment, you can begin running the experiments. Follow these steps:

1. Enter the `3DTAB` environment:

   ```bash  
   conda activate 3DTAB 
   ```

2. Navigate to the appropriate dataset directory under `benchmark/{DATASET}/test_attack`. For example:

   ```bash
   cd benchmark/modelnet_hdf5/test_attack
   ```

3. Run a specific experiment script by executing one of the `run_exp_*.sh` scripts. For example:

   ```bash
   bash run_exp_main.sh
   ```

Each script will execute a specific evaluation, and the results will be saved in a file `result.txt`. The correspondence between the experiment scripts and the experiments presented in the paper is as follows:

| Script                               | Related Paper Figure/Table          |
|--------------------------------------|-----------------------------------------------|
| `run_exp_main.sh`                    | Table 3                                       |
| `run_exp_main_extra.sh`              | Table 3                                       |
| `run_exp_defense.sh`                 | Table 4                                       |
| `run_exp_ops_diff_overhead.sh`       | Figures 3 and 4                               |
| `run_exp_ops_diff_overhead_extra.sh` | Figures 3 and 4                               |
| `run_exp_ops_diff_overhead_extra2.sh`| Figures 3 and 4                               |
| `run_exp_ops_diff_overhead_extra3.sh`| Figures 3 and 4                               |

## 4. File Structure

The project follows a clear structure for easy navigation. Below is an overview of the basic file layout:

```
├── asset                  # Contains data, weights, etc., generated during benchmark construction & testing
├── benchmark
│   └── modelnet_hdf5
│       ├── test_attack    # Contains scripts for evaluating attack & defense methods
│       └── train_model    # Contains scripts for training models to build the benchmark
├── libs                   # Libraries for model dependencies (cuda/Cython)
└── packages               # Custom packages required by 3DTAB
    ├── ATK                # **Core** of the 3DTAB framework
    └── qqdm               # A progress bar package for handling long-running tasks
```

### ATK Directory Structure

The `ATK` directory contains the core functionalities for adversarial attack and defense methods. Its structure is as follows:

```
ATK
├── attack                # Code for all attack methods
├── data                  # Code for datasets
├── defense               # Code for all defense methods
├── evaluator             # Code for the evaluation framework
├── model                 # Code for all model architectures
└── utils                 # Utility functions
```
