#!/bin/bash
# To be called from inside Docker container for now

#cd /pymarl/src
export PYTHONPATH=$PYTHONPATH:/pymarl/src

TAG=$1
TIMESTAMP=$2
N_REPEAT=$3
N_GPUS=`nvidia-smi -L | wc -l`
N_UPPER=`expr $N_GPUS - 1`

for i in $(seq 1 $N_REPEAT); do
  GPU_ID=`shuf -i0-${N_UPPER} -n1`
  echo "Starting repeat number $i on GPU $GPU_ID"
  NV_GPU=${GPU_ID} ../.././docker.sh python3 /pymarl/src/main.py --exp_name="xxx_pp" with debug_mode=level2_actions_only env_args.n_agents=2 "env_args.agent_obs=[1,1]" "env_args.predator_prey_shape=[4,4]" env_args.predator_prey_toroidal=False name=${TAG}__xxx_debug_level2only_pp_${TIMESTAMP}__repeat${i} use_tensorboard=False &
  sleep 10s
done
