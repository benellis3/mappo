#!/bin/bash
HASH=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 4 | head -n 1)
name=${USER}_pymarl_GPU_${GPU}_${HASH}

echo "Launching container named '${name}' on GPU '${GPU}'"
# Launches a docker container using our image, and runs the provided command

if hash nvidia-docker 2>/dev/null; then
  cmd=nvidia-docker
else
  cmd=docker
fi

SCRIPT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

${cmd} run -d --rm \
    --name $name \
    --security-opt="apparmor=unconfined" --cap-add=SYS_PTRACE \
    --net host \
    --user $(id -u) \
    -e CUDA_LAUNCH_BLOCKING=1 \
    -v $SCRIPT_PATH:/pymarl \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v `pwd`/mongodb:/data/db \
    -e DISPLAY=unix$DISPLAY \
    -t pymarl \
    ${@:1}