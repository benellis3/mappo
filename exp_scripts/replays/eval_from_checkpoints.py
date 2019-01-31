import subprocess
import os
import pprint
import shutil

root_dir = "/Users/mika/workspace/pymarl-dev/smac_arxiv_coma_qmix_models"

method = "qmix"
min_timesteps = 10000000
load_step = 0  # max
test_nepisode = 21

result_dir = "/Users/mika/Desktop/smac_arxiv_runs"
if os.path.exists(result_dir):
    shutil.rmtree(result_dir)
os.makedirs(result_dir)

def get_map_name(exp):
    return exp.split("__")[1]

for server_name in os.listdir(root_dir):

    if os.path.isdir(server_name):
        continue  # ignore

    server_path = os.path.join(root_dir, server_name)

    for exp in sorted(os.listdir(server_path)):

        if not exp.startswith(method):
            continue  # ignore

        map_name = get_map_name(exp)

        map_file = os.path.join(result_dir, map_name + ".txt")
        if not os.path.isfile(map_file):
            # Create an empty file
            with open(map_file, 'w+') as f:
                print(map_file + " created")

        exp_path = os.path.join(server_path, exp)

        checkpoint = None
        for cp in os.listdir(exp_path):
            if int(cp) > min_timesteps:
                checkpoint = cp

        if checkpoint is None:
            continue  # ignore

        command = "python3 src/main.py --config={} --env-config=sc2 with env_args.map_name={} checkpoint_path={} load_step={} evaluate=True save_replay=False test_nepisode={} no-mongo=True runner=episode".format(
            method,
            map_name,
            exp_path,
            load_step,
            test_nepisode
        )

        subprocess.call(command.split())

