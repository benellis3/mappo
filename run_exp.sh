#!/bin/bash
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
maps=${3:-2s3z,2s_vs_1sc,1c3s5z,3s5z,10m_vs_11m,8m_vs_9m,5m_vs_6m,3s_vs_5z,5m_vs_6m,bane_vs_bane,2c_vs_64zg,corridor,3s5z_vs_3s6z,27m_vs_30m,6h_vs_8z}   # MMM2 left out
threads=${4:-28} # 2
args=${5:-}    # ""
gpus=${6:-0,1,2,3,4,5,6,7}    # 0,1
times=${7:-3}   # 5

maps=(${maps//,/ })
gpus=(${gpus//,/ })
args=(${args//,/ })

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
for map in "${maps[@]}"; do
    for((i=0;i<times;i++)); do
        gpu=${gpus[$(($count % ${#gpus[@]}))]}  
        group="${config}-${tag}"
        ./run.sh $gpu python3 src/main.py --no-mongo --config="$config" --env-config=sc2 with env_args.map_name="$map" group="$group" "${args[@]}" &

        count=$(($count + 1))     
        if [ $(($count % $threads)) -eq 0 ]; then
            wait
        fi
        # for random seeds
        sleep $((RANDOM % 60 + 60))
    done
done
wait
