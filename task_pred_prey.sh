./run.sh 0 python3 src/main.py --config=ppo_rnn --env-config=pred_prey with name="ppo_central_v" label="ppo_pred_prey_test_mingefei" is_central_value=True &
./run.sh 1 python3 src/main.py --config=ppo_rnn --env-config=pred_prey with name="ppo_central_v" label="ppo_pred_prey_test_mingefei" is_central_value=True &
./run.sh 2 python3 src/main.py --config=ppo_rnn --env-config=pred_prey with name="ppo_central_v" label="ppo_pred_prey_test_mingefei" is_central_value=True &
./run.sh 0 python3 src/main.py --config=ppo_rnn --env-config=pred_prey with name="actor_critic" label="ppo_pred_prey_test_mingefei" actor_critic_mode=True &
./run.sh 1 python3 src/main.py --config=ppo_rnn --env-config=pred_prey with name="actor_critic" label="ppo_pred_prey_test_mingefei" actor_critic_mode=True &
./run.sh 2 python3 src/main.py --config=ppo_rnn --env-config=pred_prey with name="actor_critic" label="ppo_pred_prey_test_mingefei" actor_critic_mode=True &
