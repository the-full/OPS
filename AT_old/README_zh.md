# 对抗训练上的评估（基于 [*Admix*](https://github.com/JHL-HUST/Admix) ）

[英文](https://github.com/the-full/OPS/blob/main/AT_old/README.md)

## 1. 简介

本文件夹包含了关于 [Ensemble Adversarial Training: Attacks and Defenses](https://arxiv.org/abs/1705.07204) 的评估代码。

## 2. 环境配置

要配置 `AT_old` 环境，请运行以下命令：

```bash
bash install.sh
```

该脚本会自动安装所有必要的依赖项，包括第三方库和环境配置。

接下来，下载 [bigfile](https://drive.google.com/file/d/1uC9ZNWR7VuDlqPzMQTmC787KB7QrPofi/view?usp=drive_link)，并将其中 `AT_old` 子文件夹下的文件放置到以下对应位置：

| 源路径（bigfile/AT_old） | 目标路径              |
| -------------------- | ----------------- |
| `*`                  | `AT_old/models/*` |

在继续下一步之前，请确保所有文件已正确放置。

## 3. 运行实验

完成环境配置后，按照以下步骤运行实验：

1. 激活 `AT_old` 环境：

   ```bash
   conda activate AT_old
   ```

2. **使用 TransferAttack 生成对抗样本**
   请参考 `TransferAttack` 文件夹中的 `README.md` 文件，了解如何生成对抗样本的详细说明。

3. **运行实验脚本**
   执行其中一个 `run_exp_*.sh` 脚本以评估防御方法，例如：

   ```bash
   bash run_exp_defense_old_AT.sh
   ```

   每个脚本都会执行一个特定的评估任务。实验脚本与论文中实验的对应关系如下：

| 脚本名称                            | 对应论文图表               |
| ------------------------------- | -------------------- |
| `run_exp_defense_old_AT.sh`     | 表格 Table 4（Baseline） |
| `run_exp_defense_old_AT_ops.sh` | 表格 Table 4（OPS）      |

