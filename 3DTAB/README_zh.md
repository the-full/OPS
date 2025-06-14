# 3D Transfer Attack Benchmark（demo）

[英文](https://github.com/the-full/OPS/blob/main/3DTAB/README.md)

## 1. 简介

3DTAB（3D Transfer Attack Benchmark）是一个用于评估三维点云上对抗攻击与防御方法的基准测试平台。尽管最初设计用于迁移攻击的评估，但在实践中它也可以支持白盒攻击方法评估。

这个说明将介绍如何配置运行环境、如何执行实验，并提供项目文件结构的概述。

## 2. 环境配置

要配置 `3DTAB` 环境，请运行以下命令来安装所需环境：

```bash
bash install.sh
```

该脚本将自动安装所需的所有依赖，包括第三方库和环境配置。

接下来，下载 [bigfile](https://drive.google.com/file/d/1uC9ZNWR7VuDlqPzMQTmC787KB7QrPofi/view?usp=drive_link)，并将其中 `3DTAB` 子文件夹下的文件放置到以下对应位置：

| 源路径（bigfile/3DTAB）              | 目标路径                                                   |
| ------------------------------- | ------------------------------------------------------ |
| `modelnet40_ply_hdf5_2048`      | `3DTAB/asset/dataset/modelnet40_ply_hdf5_2048`         |
| `modelnet40_ply_hdf5_2048_mini` | `3DTAB/asset/dataset/modelnet40_ply_hdf5_2048_mini`    |
| `ModelNetHdf5`                  | `3DTAB/asset/model_ckpt/ModelNetHdf5`                  |
| `pretrain/AdvPC/mn40`           | `3DTAB/packages/ATK/ATK/attack/AdvPC/pretrain/mn40`    |
| `pretrain/DUP_Net/*`            | `3DTAB/packages/ATK/ATK/defense/DUP_Net/pretrain/*`    |
| `pretrain/IF-Defense/*`         | `3DTAB/packages/ATK/ATK/defense/IF-Defense/pretrain/*` |

在继续下一步之前，请确保所有文件已正确放置。

## 3. 运行实验

环境配置完成后，你可以开始运行实验。按照以下步骤进行：

1. 激活 `3DTAB` 环境：

   ```bash
   conda activate 3DTAB
   ```

2. 进入对应数据集的目录，例如：

   ```bash
   cd benchmark/modelnet_hdf5/test_attack
   ```

3. 执行某个实验脚本，例如：

   ```bash
   bash run_exp_main.sh
   ```

每个脚本会执行特定的评估，结果将保存在 `result.txt` 文件中。实验脚本与论文中实验的对应关系如下：

| 脚本名称                                  | 对应论文图表           |
| ------------------------------------- | ---------------- |
| `run_exp_main.sh`                     | 表格 Table 3       |
| `run_exp_main_extra.sh`               | 表格 Table 3       |
| `run_exp_defense.sh`                  | 表格 Table 4       |
| `run_exp_ops_diff_overhead.sh`        | 图表 Figures 3 和 4 |
| `run_exp_ops_diff_overhead_extra.sh`  | 图表 Figures 3 和 4 |
| `run_exp_ops_diff_overhead_extra2.sh` | 图表 Figures 3 和 4 |
| `run_exp_ops_diff_overhead_extra3.sh` | 图表 Figures 3 和 4 |

## 4. 文件结构

项目遵循清晰的结构，便于导航。基本文件结构如下：

```
├── asset                  # 存放数据、模型权重等文件，构建与测试基准时生成
├── benchmark
│   └── modelnet_hdf5
│       ├── test_attack    # 存放攻击与防御评估相关脚本
│       └── train_model    # 存放训练模型以构建基准的脚本
├── libs                   # 模型依赖库（cuda/Cython）
└── packages               # 3DTAB 所需的自定义包
    ├── ATK                # **3DTAB 框架的核心代码**
    └── qqdm               # 长时间任务使用的进度条库
```

### ATK 目录结构

`ATK` 目录包含了对抗攻击与防御方法的核心功能，其结构如下：

```
ATK
├── attack                # 所有攻击方法的代码
├── data                  # 数据集相关代码
├── defense               # 所有防御方法的代码
├── evaluator             # 评估框架代码
├── model                 # 各种模型结构代码
└── utils                 # 工具函数
```

## 5. 后续更新

在后续更新中，我们将提供对 3DTAB 更全面的介绍，包括其设计原理、关键特性以及在对抗攻击与防御研究中的潜在应用。我们还计划添加更多使用示例与教程，以便用户更好地使用该基准框架，并将其集成到相关研究项目中。
