#!/bin/bash

echo "Launching container locally"
# Launches a docker container using our image, and runs the provided command

docker run \
    --user $(id -u) \
    -v `pwd`:/home/minsun/pymarl \
    --network=host \
    pymarl:ppo \
    ${@:2}
