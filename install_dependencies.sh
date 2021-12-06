#!/bin/bash
# Install PyTorch and Python Packages

# conda create -n pymarl python=3.7 -y
# conda activate pymarl

conda install pytorch torchvision cudatoolkit=10.2 -c pytorch -y
pip install wandb sacred numpy scipy matplotlib seaborn pyyaml pygame pytest probscale imageio snakeviz tensorboard-logger
pip install git+https://github.com/oxwhirl/smac.git
