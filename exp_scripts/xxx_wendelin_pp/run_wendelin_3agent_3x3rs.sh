#!/bin/bash

TAG=$1
N_REPEAT=$2
TIMESTAMP=`date "+%d/%m/%y-%H:%M:%S"`
echo "Experiment ${TAG} time stamp: ${TIMESTAMP}"
./wendelin_3agent_3x3rs.sh ${TAG} ${TIMESTAMP} ${N_REPEAT} &