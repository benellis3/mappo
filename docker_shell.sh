#!/usr/bin/env bash

if hash nvidia-docker 2>/dev/null; then
  cmd=nvidia-docker
else
  cmd=docker
fi

NV_GPU=0 ${cmd} run --rm --user $(id -u) -v `pwd`:/home/minsun/pymarl -it pymarl:ppo
