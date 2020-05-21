N_REPEAT=$1

for i in $(seq 1 $N_REPEAT); do
  echo "Starting repeat number $i"
  python3 src/main.py --config=ppo --env-config=sc2 with name=ppo__ppo_3m_2020_05_21__$i
done
