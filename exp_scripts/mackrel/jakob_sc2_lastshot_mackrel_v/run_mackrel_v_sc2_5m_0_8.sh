#!/bin/bash

TAG=$1
N_REPEAT=$2
TIMESTAMP=`date "+%d/%m/%y-%H:%M:%S"`
echo "Experiment ${TAG} time stamp: ${TIMESTAMP}"
./mackrel_v_jakob_sc2_5m_0_8.sh ${TAG} ${TIMESTAMP} ${N_REPEAT} &