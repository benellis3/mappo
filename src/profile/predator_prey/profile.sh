P=`pwd`
cd ../../
python3 -m cProfile -o $P/predator_prey.out main.py --exp_name="iql_pp3x3_3agentcap" with t_max=100000
#python3  main.py --exp_name="iql_pp3x3_3agentcap" with t_max=2000000
