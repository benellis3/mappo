cat task.txt | xargs -n 4 -P 16 \
    sh -c 'bash run.sh $0 python3 src/main.py --config=$1 --env-config=matrix_game name=$2 label=$3'