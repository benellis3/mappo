#!/bin/bash

TAG=$1
N_REPEAT=$2
TIMESTAMP=`date "+%d/%m/%y-%H:%M:%S"`
echo "Experiment ${TAG} time stamp: ${TIMESTAMP}"
./coma_jakob_sc2_sf16.sh ${TAG} ${TIMESTAMP} ${N_REPEAT} &