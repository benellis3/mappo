#!/bin/bash

TAG=$1
N_REPEAT=$2
TIMESTAMP=`date "+%d/%m/%y-%H:%M:%S"`
echo "Experiment ${TAG} time stamp: ${TIMESTAMP}"
./baseline1.sh ${TAG} ${TIMESTAMP} ${N_REPEAT} &
#./baseline2.sh ${TAG} ${TIMESTAMP} ${N_REPEAT} & ./baseline3.sh ${TAG} ${TIMESTAMP} ${N_REPEAT}