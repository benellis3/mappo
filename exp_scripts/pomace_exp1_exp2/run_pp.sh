#!/bin/bash

NREPS=$1
./exp1_pp.sh  ${NREPS} & ./exp1_pp.sh with pomace_exp_variant=2 & ./exp2_pp.sh with pomace_exp_variant=1 ${NREPS} & ./exp2_pp.sh with pomace_exp_variant=2