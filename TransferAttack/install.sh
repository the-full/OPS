#!/bin/bash

ROOT_DIR=$(pwd)

if [ -z "$USER_CONDA_BASE" ]; then
  CONDA_PATH=$(which conda)
  if [ -z "$CONDA_PATH" ]; then
    echo "Conda is not installed or not found in PATH."
    exit 1
  fi
  USER_CONDA_BASE=$(dirname $(dirname $CONDA_PATH))
fi

echo $USER_CONDA_BASE

# NOTE: create TransferAttack env
conda create -n TransferAttack pip python==3.9 -y
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate TransferAttack

# NOTE: install pytorch
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia -y
# NOTE: fix numpy==2.* not compatible for pytorch 2.0
pip install numpy==1.*

pip install tqdm pandas timm scikit-optimize matplotlib
