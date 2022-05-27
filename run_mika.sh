#!/bin/bash
# debug=
debug=echo
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
units=${3:-10,5,20}   # adding 5 and 20 would be better
clipping_range=${9:-0.1,0.05,0.15}
lr=${10:-0.0007,0.0004,0.0001}
maps=${8:-sc2_gen_protoss,sc2_gen_zerg,sc2_gen_terran}
threads=${4:-18} # 2
args=${5:-}    # ""
times=${7:-3}   # could use fewer seeds and instead run more numbers of units.

maps=(${maps//,/ })
args=(${args//,/ })
units=(${units//,/ })
lrs=(${lr//,/ })
clipping_ranges=(${clipping_range//,/ })

if [ ! $config ] || [ ! $tag ]; then
    echo "Please enter the correct command."
    echo "bash run.sh config_name map_name_list (experinments_threads_num arg_list gpu_list experinments_num)"
    exit 1
fi

# run parallel
count=0
for lr in "${lrs[@]}"; do
    for clipping_range in "${clipping_ranges[@]}"; do
        for map in "${maps[@]}"; do
            for((i=0;i<times;i++)); do
                for unit in "${units[@]}"; do
                    group="${config}-${map}-${tag}"
                    echo python3 src/main.py --no-mongo --config="$config" --env-config="$map" with env_args.capability_config.n_units=$unit env_args.capability_config.start_positions.n_enemies=$unit group="$group" clip_range=$clipping_range lr_actor=$lr use_wandb=True save_model=True "${args[@]}"

                    count=$(($count + 1))
                    if [ $(($count % $threads)) -eq 0 ]; then
                        wait
                    fi
        	done
            done
        done
    done
done
wait
