#!/bin/bash
# Calls reset_server.sh or execute_on_server.sh through ssh

usage()
{
    echo "bash exp_scripts/run_on_server.sh SERVER GPU_ID N_REPEAT CFG or SERVER reset"
}

if [ -z "$1" ]; then usage; exit 1;
else
    SERVER=$1
fi

if [ -z "$2" ]; then usage; exit 1;
else
    GPU_ID=$2
fi

if [ -z "$3" ]; then usage; exit 1;
else
    N_REPEAT=$3
fi

PARAMS="${@:4}"

if [ $GPU_ID == "reset" ]; then
    echo "Resetting Server ${SERVER}"
    ssh -A $SERVER 'bash -s' < exp_scripts/reset_server.sh
else
    echo "Running exps on Server ${SERVER} on GPU ${GPU_ID} ${N_REPEAT} times"
    ssh -A $SERVER "EXP_DIR=${EXP_DIR} bash -s" < exp_scripts/execute_on_server.sh $GPU_ID $N_REPEAT $PARAMS
fi
