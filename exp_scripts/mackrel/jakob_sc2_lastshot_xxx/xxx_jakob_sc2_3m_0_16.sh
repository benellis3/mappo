#!/bin/bash
# To be called from inside Docker container for now

#cd /pymarl/src
export PYTHONPATH=$PYTHONPATH:/pymarl/src

TAG=$1
TIMESTAMP=$2
N_REPEAT=$3
N_GPUS=`nvidia-smi -L | wc -l`
N_UPPER=`expr $N_GPUS - 1`


for j in $(seq 1 $N_REPEAT); do
  GPU_ID=`shuf -i0-${N_UPPER} -n1`
  echo "Starting repeat number ${j} on GPU $GPU_ID"
  NV_GPU=${GPU_ID} ../.././docker.sh python3 /pymarl/src/main.py --exp_name="xxx_jakob_sc2_3m__lastshot_0_16" with name=${TAG}:0__xxx_jakob_sc2_3m__lastshot_0_16__${TIMESTAMP}__repeat${j} use_tensorboard=False &
  sleep 10s
done
