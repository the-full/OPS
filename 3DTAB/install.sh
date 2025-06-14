#!/bin/bash

ROOT_DIR=$(pwd)
PACKAGE_DIR=$(readlink -f "./packages/")
LIBRARY_DIR=$(readlink -f "./libs/")

if [ -z "$USER_CONDA_BASE" ]; then
  CONDA_PATH=$(which conda)
  if [ -z "$CONDA_PATH" ]; then
    echo "Conda is not installed or not found in PATH."
    exit 1
  fi
  USER_CONDA_BASE=$(dirname $(dirname $CONDA_PATH))
fi

echo $USER_CONDA_BASE

echo "S0: Creating 3DTAB environment..."
conda create -n 3DTAB pip python==3.9 -y
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate 3DTAB

echo "S1: Installing PyTorch & PyTorch lightning..."
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia -y
conda install lightning -c conda-forge -y
# NOTE: Resolve compatibility issue with numpy==2.* for PyTorch
pip install numpy==1.*

echo "S2: Installing hydra for configuration management..."
pip install hydra-core

echo "S3: Installing ninja & Cython for building extensions..."
pip install ninja Cython

echo "S4: Installing extension libraries..."
cd "$LIBRARY_DIR"

cd pointops                         # for some models
python setup.py build_ext --inplace # compile
pip install .                       # install
cd ..

cd pointops2                        # for some models
python setup.py build_ext --inplace # compile
pip install .                       # install
cd ..

cd pointops_repsurf                 # for Repsurf
python setup.py build_ext --inplace # compile
pip install .                       # install
cd ..

cd pointops_rscnn                   # for RSCNN
python setup.py build_ext --inplace # compile
pip install .                       # install
cd ..

cd im2mesh                          # for IF-Defense
python setup.py build_ext --inplace # compile
pip install .                       # install
cd ..

# NOTE: install pytorch3d for ATK/utils/ops.py and others
echo "S5: Installing pytorch3d..."
pip install git+https://github.com/facebookresearch/pytorch3d.git@stable

echo "S6:  Installing our package: ATK..."
cd $PACKAGE_DIR

cd ./ATK                      # for 3DTAB
pip install -e . --use-pep517 # editable install
cd ..

cd ./qqdm     # for better progress bar
pip install . # no need for editable install
cd ..

echo "S7: Installing timm, pytorch-geometric(pyg), and others for PointTransformer-v2..."
cd $ROOT_DIR

conda install timm -y
conda install pyg=*=*cu* -c pyg -y
conda install pytorch-cluster -c pyg -y
conda install pytorch-scatter -c pyg -y

echo "S8: Installing tensorboardX for lightning training..."
pip install -U tensorboardX
