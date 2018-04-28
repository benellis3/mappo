#!/bin/bash

NREPS=$1
./exp1_sc2.sh  ${NREPS} & ./exp1_sc2.sh with pomace_exp_variant=2 & ./exp2_sc2.sh with pomace_exp_variant=1 ${NREPS} & ./exp2_sc2.sh with pomace_exp_variant=2