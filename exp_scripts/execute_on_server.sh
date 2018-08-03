#!/bin/bash
# Kills the existing docker container and restarts a new detached container

GPU_ID=$1
N_REPEAT=$2
PARAMS="${@:3}"

# I am not sshed into the server in my home directory

if [ -z "$EXP_DIR" ]
then
    EXP_DIR=~
fi

echo "EXP_DIR: $EXP_DIR"

# Update the git repo
echo "Updating git repo"
#mkdir -p $EXP_DIR/deepmarl
cd $EXP_DIR/pymarl
# echo "REMEMBER TO ALLOW UPDATING OF THE REPO IN execute_on_server.sh AFTER TESTING!"
git fetch -q origin
git reset --hard origin/refactor -q

# Run the experiment $N_REPEAT times on GPU $GPU
bash ./run.sh $GPU_ID "bash exp_scripts/repeat_exp.sh $N_REPEAT $PARAMS"
