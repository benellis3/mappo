cat task.txt | xargs -n 6 -P 20 \
    sh -c 'bash run.sh $0 python3 src/main.py --config=$1 --env-config=sc2 --no-mongo with env_args.map_name=$2 t_max=$3 name=$4 label=$5'