#!/bin/bash
# debug=
# debug=echo
trap 'onCtrlC' INT

function onCtrlC () {
  echo 'Ctrl+C is captured'
  for pid in $(jobs -p); do
    kill -9 $pid
  done
  
  kill -HUP $( ps -A -ostat,ppid | grep -e '^[Zz]' | awk '{print $2}')
  exit 1
}

config=$1  # qmix
tag=$2
units=${3:-6}   # MMM2 left out
# unit=5
offset=0
maps=${8:-sc2_gen_protoss_epo,sc2_gen_terran_epo,sc2_gen_zerg_epo}
threads=${4:-18} # 2
args=${5:-}    # ""
gpus=${6:-0,1,2,3,4,5,6,7}    # 0,1,2,3,4,5,6,7
times=${7:-3}   # 5
prob_obs_enemy=${9:-0.0,0.5,1.0}
lr=0.0005
clipping_range=0.1
dist=surrounded_and_reflect

maps=(${maps//,/ })
gpus=(${gpus//,/ })
args=(${args//,/ })
units=(${units//,/ })
prob_obs_enemy=(${prob_obs_enemy//,/ })

if [ ! $config ] || [ ! $tag ]; then
    echo "Please enter the correct command."
    echo "bash run.sh config_name map_name_list (experinments_threads_num arg_list gpu_list experinments_num)"
    exit 1
fi

echo "CONFIG:" $config
echo "MAP LIST:" ${maps[@]}
echo "THREADS:" $threads
echo "ARGS:"  ${args[@]}
echo "GPU LIST:" ${gpus[@]}
echo "TIMES:" $times


# run parallel
count=0
for prob in "${prob_obs_enemy[@]}"; do
	for map in "${maps[@]}"; do
		for((i=0;i<times;i++)); do
			for unit in "${units[@]}"; do
				gpu=${gpus[$(($count % ${#gpus[@]}))]}  
				group="${config}-${tag}"
				# enemies=$(($unit + $offset))
				enemies=5
				$debug ./run_docker.sh $gpu python3 src/main.py --no-mongo --config="$config" --env-config="$map" with env_args.capability_config.n_units=$unit env_args.capability_config.n_enemies=$enemies env_args.capability_config.start_positions.dist_type="$dist" env_args.prob_obs_enemy=$prob group="$group" clip_range=$clipping_range lr_actor=$lr use_wandb=True save_model=False "${args[@]}" &

				count=$(($count + 1))     
				if [ $(($count % $threads)) -eq 0 ]; then
					wait
				fi
				# for random seeds
				sleep $((RANDOM % 3 + 3))
			done 
		done
	done
done
wait
