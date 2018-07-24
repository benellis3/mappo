#!/bin/bash
# Kills all docker containers running the deepmarl/pytorch image.
# Rebuilds docker image

if [ -z "$EXP_DIR" ]
then
    EXP_DIR=~
fi

echo "EXP_DIR: $EXP_DIR"
cd $EXP_DIR

# Check for the deepmarl repo, if it is not there then clone it
if [ ! -d "pymarl" ]; then
    echo "Cloning pymarl repo"
    # If this doesn't work on a server you might have to manually connect and clone the repo for the first time
    git clone git@github.com:oxwhirl/pymarl.git
fi

cd $EXP_DIR/pymarl

git fetch -q origin
git reset --hard origin/tabz -q

# Kill the docker container
./kill.sh

# Rebuild the docker container
./build.sh