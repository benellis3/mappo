cat task.txt | xargs -n 7 -P 24 \
    sh -c 'bash run.sh $0 python3 src/main.py --config=$1 --env-config=sc2 with env_args.map_name=$2 t_max=$3 name=$4 label=$5 kl_coeff_beta=$6 divergence_mode=mean'