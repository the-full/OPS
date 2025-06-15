# Reproduce Guide

[中文](https://github.com/the-full/OPS/blob/main/README_zh.md)

This directory contains all the code used in our experiments. It is organized into three main subdirectories:

1. **3DTAB**: This folder contains the code for evaluating 3D point cloud data.
2. **AT_old**: This folder includes code for evaluating the results of adversarial training defenses from [**Ensemble Adversarial Training: Attacks and Defenses**](https://arxiv.org/abs/1705.07204) on ImageNet datasets.
3. **TransferAttack**: This folder contains code for evaluating transferability in images.

## Setting Up the Environment

Each folder requires a specific environment to run the corresponding code. You can create the required environment by running the `install.sh` script in each folder. The default environment name is the same as the folder name.

## Running the Experiments

The experiments described in the paper are organized into scripts named `run_exp_*.sh`. The specific mapping of these scripts to the experiments can be found in the corresponding `README.md` file within each folder.

## Model Weights and Dataset

The model weights and dataset required for running the code can be found at [bigfile](https://drive.google.com/file/d/1-npsCNCYf3j_URhTqQSnnWMjMRirkg8U/view?usp=drive_link). For detailed instructions on where to place the weights, refer to the `README.md` file in each respective folder.

## Adversarial Examples for Validation

For convenience, we provide all the adversarial examples generated during the experiments in **TransferAttack**, which can be download from [here](https://drive.google.com/file/d/1lImziwWRpRF5IU5dFNV3fh2ZSKmz7m38/view?usp=drive_link).


## Credits

Our code is inspired by many excellent works. We sincerely thank the contributors of the following projects:

* The benchmark setup and attack evaluation process are based on [TransferAttack](https://github.com/Trustworthy-AI-Group/TransferAttack).
* The implementation of iterative attack methods and registry design is based on [ARES](https://github.com/thu-ml/ares).
* The attack pipeline for C\&W-based methods draws from [GeoA3](https://github.com/Gorilla-Lab-SCUT/GeoA3) and [3d-adv-pc](https://github.com/xiangchong1/3d-adv-pc).
* The evaluation code for models trained with ensemble adversarial training is adapted from [admix](https://github.com/JHL-HUST/Admix).
* The implementations of models, attack methods, and defense methods in 3DTAB reference the following repositories:

  * **Models**:
    [PointNet](https://github.com/fxia22/pointnet.pytorch),
    [PointNet++](https://github.com/yanx27/Pointnet_Pointnet2_pytorch),
    [PointConv](https://github.com/DylanWusee/pointconv_pytorch.git),
    [PointCNN](https://github.com/hxdengBerkeley/PointCNN.Pytorch),
    [CurveNet](https://github.com/tiangexiang/CurveNet.git),
    [Point-Transformers](https://github.com/qq456cvb/Point-Transformers.git),
    [PointMLP](https://github.com/ma-xu/pointMLP-pytorch.git),
    [Repsurf](https://github.com/hancyran/RepSurf.git),
    [PointCat](https://github.com/shikiw/PointCAT),
    [Pointcept](https://github.com/Pointcept/Pointcept),
    [vnn](https://github.com/FlyingGiraffe/vnn)

  * **Attacks**:
    [PF-Attack](https://github.com/HeBangYan/PF-Attack),
    [SI-Adv](https://github.com/shikiw/SI-Adv.git),
    [HiT-ADV](https://github.com/TRLou/HiT-ADV.git),
    [ai-pointnet-attack](https://github.com/jinyier/ai_pointnet_attack.git),
    [advpc](https://github.com/ajhamdi/AdvPC.git)

  * **Defenses**:
    [IF-Defense](https://github.com/Wuziyi616/IF-Defense.git),
    [SI-Adv](https://github.com/shikiw/SI-Adv.git) (provides a PyTorch implementation of the DUP-Net defense)

## Contact

If you have any questions about the paper or the code, please contact [zhazineedamail@163.com](mailto:zhazineedamail@163.com).

