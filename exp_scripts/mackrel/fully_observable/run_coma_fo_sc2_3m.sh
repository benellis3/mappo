#!/bin/bash

TAG=$1
N_REPEAT=$2
TIMESTAMP=`date "+%d/%m/%y-%H:%M:%S"`
echo "Experiment ${TAG} time stamp: ${TIMESTAMP}"
./coma_fo_sc2_3m.sh ${TAG} ${TIMESTAMP} ${N_REPEAT} &
