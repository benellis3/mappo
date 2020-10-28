# ./run.sh 1 python3 src/main.py --config=ppo_rnn --env-config=stag_hunt with name="ppo_correct" label="ppo_stag_hunt_test_mingefei" &
# ./run.sh 2 python3 src/main.py --config=ppo_rnn --env-config=stag_hunt with name="ppo_correct" label="ppo_stag_hunt_test_mingefei" &
# ./run.sh 3 python3 src/main.py --config=ppo_rnn --env-config=stag_hunt with name="ppo_correct" label="ppo_stag_hunt_test_mingefei" &
./run.sh 3 python3 src/main.py --config=ppo_rnn --env-config=stag_hunt with is_central_value=True name="ppo_rnn_correct" label="ppo_stag_hunt_test_mingefei" &
./run.sh 4 python3 src/main.py --config=ppo_rnn --env-config=stag_hunt with is_central_value=True name="ppo_rnn_correct" label="ppo_stag_hunt_test_mingefei" &
./run.sh 5 python3 src/main.py --config=ppo_rnn --env-config=stag_hunt with is_central_value=True name="ppo_rnn_correct" label="ppo_stag_hunt_test_mingefei" &
