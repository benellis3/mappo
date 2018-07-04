#!/bin/bash

TAG=$1
N_REPEAT=$2
TIMESTAMP=`date "+%d/%m/%y-%H:%M:%S"`
echo "Experiment ${TAG} time stamp: ${TIMESTAMP}"
./flounderl_jakob_pp_4agents_6x6.sh ${TAG} ${TIMESTAMP} ${N_REPEAT} &