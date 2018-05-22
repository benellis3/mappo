#!/bin/bash

TAG=$1
N_REPEAT=$2
TIMESTAMP=`date "+%d/%m/%y-%H:%M:%S"`
echo "Experiment ${TAG} time stamp: ${TIMESTAMP}"
./wendelin_4agent_6x6.sh ${TAG} ${TIMESTAMP} ${N_REPEAT} &