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

# NOTE: create AT_old env
conda create -n AT_old pip python==3.6.13 -y
source "$USER_CONDA_BASE/etc/profile.d/conda.sh"
conda activate AT_old

pip install -r requirements.txt
