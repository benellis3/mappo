#!/bin/bash

TAG=$1
N_REPEAT=$2
TIMESTAMP=`date "+%d/%m/%y-%H:%M:%S"`
echo "Experiment ${TAG} time stamp: ${TIMESTAMP}"
./exp1.sh ${TAG} ${TIMESTAMP} ${N_REPEAT} & ./exp2.sh ${TAG} ${TIMESTAMP} ${N_REPEAT} & ./exp3.sh ${TAG} ${TIMESTAMP} ${N_REPEAT}
