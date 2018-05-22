#!/bin/bash
# To be called from inside Docker container for now

#cd /pymarl/src
export PYTHONPATH=$PYTHONPATH:/pymarl/src

filename=$(basename -- "$0")
extension="${filename##*.}"
filename="${filename%.*}"

TAG=$1
TIMESTAMP=$2
N_REPEAT=$3
N_GPUS=`nvidia-smi -L | wc -l`
N_UPPER=`expr $N_GPUS - 1`

for i in $(seq 1 $N_REPEAT); do
  GPU_ID=`shuf -i0-${N_UPPER} -n1`
  echo "Starting repeat number $i on GPU $GPU_ID"
  NV_GPU=${GPU_ID} ../../.././docker.sh python3 /pymarl/src/main.py --exp_name="xxx_jakob_sc2_3m" with batch_size=7 batch_size_run=7 lr_agent_level1=0.001 lr_agent_level2=0.001 lr_agent_level3=0.001 name=${TAG}__${filename}__${TIMESTAMP}__repeat${i} use_tensorboard=False &
  sleep 10s
done
