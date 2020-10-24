./run.sh 0 python3 src/main.py --config=ppo_rnn --env-config=pred_prey with name="ppo_pred_prey" label="ppo_pred_prey_test_mingefei" &
./run.sh 1 python3 src/main.py --config=qmix --env-config=pred_prey with name="qmix_pred_prey" label="ppo_pred_prey_test_mingefei" &
