#!/bin/bash

NREPS=$1
./baseline1.sh ${NREPS} & ./baseline2.sh ${NREPS} & ./baseline3.sh ${NREPS}