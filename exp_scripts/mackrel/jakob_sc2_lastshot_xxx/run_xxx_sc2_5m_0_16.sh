#!/bin/bash

TAG=$1
N_REPEAT=$2
TIMESTAMP=`date "+%d/%m/%y-%H:%M:%S"`
echo "Experiment ${TAG} time stamp: ${TIMESTAMP}"
./xxx_jakob_sc2_5m_0_16.sh ${TAG} ${TIMESTAMP} ${N_REPEAT} &