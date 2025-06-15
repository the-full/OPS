# 复现指南

[英文](https://github.com/the-full/OPS/blob/main/README.md)

该目录包含了我们实验中使用的全部代码。代码被组织在三个主要的子目录中：

1. **3DTAB**：该文件夹包含 3D 点云数据中使用的评估代码。
2. **AT_old**：该文件夹包含在 ImageNet 数据集上评估对抗训练防御效果的代码，所使用的方法来自[**Ensemble Adversarial Training: Attacks and Defenses**](https://arxiv.org/abs/1705.07204)。
3. **TransferAttack**：该文件夹包含图像数据中使用的评估代码。

## 环境设置

每个文件夹中的代码都需要特定的运行环境。你可以通过运行各文件夹中的 `install.sh` 脚本来创建所需环境。默认创建的环境名称与对应文件夹名称相同。

## 实验运行

论文中描述的实验被组织成以 `run_exp_*.sh` 命名的脚本。每个脚本与具体实验的对应关系，可以在各文件夹内的 `README.md` 文件中找到。

## 模型权重与数据集

运行代码所需的模型权重与数据集可以从 [bigfile](https://drive.google.com/file/d/1-npsCNCYf3j_URhTqQSnnWMjMRirkg8U/view?usp=drive_link) 下载。关于如何放置这些权重的详细说明，请参考各自文件夹中的 `README.md` 文件。

## 用于验证的对抗样本

为方便验证，我们提供了在 **TransferAttack** 实验中生成的所有对抗样本，可从[此链接](https://drive.google.com/file/d/1lImziwWRpRF5IU5dFNV3fh2ZSKmz7m38/view?usp=drive_link)下载。

## 致谢

项目代码参考了许多优秀工作的代码，我们在此表示由衷的感谢：

- 基准的搭建和攻击的评估流程参考了 [TransferAttack](https://github.com/Trustworthy-AI-Group/TransferAttack)
- 基于迭代的攻击方法的代码实现以及 registry 的设计参考了 [ARES](https://github.com/thu-ml/ares)
- 基于 C&W Attack 的攻击方法的攻击流程参考了 [GeoA3](https://github.com/Gorilla-Lab-SCUT/GeoA3) 和 [3d-adv-pc](https://github.com/xiangchong1/3d-adv-pc)
- 针对集成对抗训练防御下的模型的评估代码参考了 [admix](https://github.com/JHL-HUST/Admix)

- 3DTAB 中的模型、攻击方法和防御方法的代码实现参考了如下仓库：

    - **模型**：[PointNet](https://github.com/fxia22/pointnet.pytorch)、[PointNet++](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)、[PointConv](https://github.com/DylanWusee/pointconv_pytorch.git)、[PointCNN](https://github.com/hxdengBerkeley/PointCNN.Pytorch)、[CurveNet](https://github.com/tiangexiang/CurveNet.git)、[Point-Transformers](https://github.com/qq456cvb/Point-Transformers.git)、[PointMLP](https://github.com/ma-xu/pointMLP-pytorch.git)、[Repsurf](https://github.com/hancyran/RepSurf.git)、[PointCat](https://github.com/shikiw/PointCAT)、[Pointcept](https://github.com/Pointcept/Pointcept)、[vnn](https://github.com/FlyingGiraffe/vnn)
    - **攻击**：[PF-Attack](https://github.com/HeBangYan/PF-Attack)、[SI-Adv](https://github.com/shikiw/SI-Adv.git)、[HiT-ADV](https://github.com/TRLou/HiT-ADV.git)、[ai-pointnet-attack](https://github.com/jinyier/ai_pointnet_attack.git)、[advpc](https://github.com/ajhamdi/AdvPC.git)、
    - **防御**：[IF-Defense](https://github.com/Wuziyi616/IF-Defense.git)、[SI-Adv](https://github.com/shikiw/SI-Adv.git) {提供了 pytorch 版本的 DUP-Net 防御}

## 联系方式

如果你对论文或代码有任何疑问，请联系 [zhazineedamail@163.com](mailto:zhazineedamail@163.com)）。
