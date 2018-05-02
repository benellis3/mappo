P=`pwd`
cd ../../../

python3 -m cProfile -o $P/blitzz.out main.py with env=pred_prey n_agents=4 t_max=240000 learner=coma env_args.prey_movement=escape "env_args.predator_prey_shape=(3,3)" env_args.nagent_capture_enabled=False tensorboard=True name="33pp_comatest_dqn" target_update_interval=10000 agent=basic agent_model=DQN observe=True observe_db=True action_selector="multinomial" use_blitzz=True obs_last_action=True batch_size=64 batch_size_run=32 test_interval=5000 epsilon_start=1.0 epsilon_finish=0.05 epsilon_time_length=100000 epsilon_decay="exp" lr=5e-3 test_nepisode=50
