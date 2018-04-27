#!/bin/bash
echo "Killing all docker containers with a name  matching ${USER}_deepmarl_GPU_*"
docker rm $(docker stop $(docker ps -a -q --filter name=${USER}_pymarl_GPU_ --format="{{.ID}}"))
