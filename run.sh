#!/bin/bash
HASH=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 4 | head -n 1)
GPU=$1
name=${USER}_pymarl_GPU_${GPU}_${HASH}
WANDB_API_KEY=$(cat $WANDB_API_KEY_FILE)
echo "Launching container named '${name}' on GPU '${GPU}'"
# Launches a docker container using our image, and runs the provided command

if hash nvidia-docker 2>/dev/null; then
  cmd=nvidia-docker
else
  cmd=docker
fi

NV_GPU="$GPU" ${cmd} run \
    --name $name \
    --user $(id -u) \
    -e WANDB_API_KEY=$WANDB_API_KEY \
    -v `pwd`:/home/minsun/pymarl \
    -t pymarl:stochastic_test \
    ${@:2}
