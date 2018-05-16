P=`pwd`
REL_PATH=`pwd`"/../../"
export PYTHONPATH=$REL_PATH:$PYTHONPATH
ROOT_PATH=`pwd`"/../../../"
cd $ROOT_PATH
N_ITER=32

#python3 -m cProfile -o $P/loop.out ${REL_PATH}main.py --exp_name="coma_sc2_3m" with t_max=10000 batch_size=${N_ITER} batch_size_run=${N_ITER} runner_test_batch_size=${N_ITER}
#python3 -m cProfile -o $P/subprocess.out ${REL_PATH}main.py --exp_name="coma_sc2_3m" with t_max=10000 batch_size=${N_ITER} batch_size_run=${N_ITER} runner_test_batch_size=${N_ITER} n_loops_per_thread_or_sub_or_main_process=1 n_subprocesses=${N_ITER}
#python3 -m cProfile -o $P/pred_prey.out ${REL_PATH}main.py --exp_name="coma_pp3x3_3agentcap" with t_max=10000 batch_size=${N_ITER} batch_size_run=${N_ITER} runner_test_batch_size=${N_ITER}
python3 -m cProfile -o $P/subprocessX.out ${REL_PATH}main.py --exp_name="xxx_pp" with t_max=30000 batch_size=${N_ITER} batch_size_run=${N_ITER} runner_test_batch_size=${N_ITER}
