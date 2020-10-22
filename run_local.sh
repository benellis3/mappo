#!/bin/bash
HASH=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 4 | head -n 1)
name=${USER}_pymarl_local_${HASH}

echo "Launching container named '${name}' locally"
# Launches a docker container using our image, and runs the provided command

docker run \
    --name $name \
    --user $(id -u) \
    -v `pwd`:/home/minsun/pymarl \
    --network=host \
    -t pymarl:ppo \
    ${@:2}
