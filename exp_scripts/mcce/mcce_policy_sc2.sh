#!/bin/bash
# To be called from inside Docker container for now

cd /deepmarl/src
export PYTHONPATH=$PYTHONPATH:/deepmarl/src/utils/blitzz

N_REPEAT=$1
N_GPUS=`nvidia-smi -L | wc -l`
N_UPPER=`expr $N_GPUS - 1`

for i in $(seq 1 $N_REPEAT); do
  GPU_ID=`shuf -i0-${N_UPPER} -n1`
  echo "Starting repeat number $i on GPU $GPU_ID"
  NV_GPU=${GPU_ID} ../.././docker.sh python3 /deepmarl/src/main.py --exp_name=mcce_policy_coma_sc2 with name=mcce_policy_coma_sc2__repeat${i} &
done
