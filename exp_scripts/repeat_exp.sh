#!/bin/bash

N_REPEAT=$1
PARAMS="${@:2}"

for i in $(seq 1 $N_REPEAT); do
  echo "Starting repeat number $i"
  python3 src/main.py $PARAMS
done
