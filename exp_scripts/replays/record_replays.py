import os
import subprocess
import re

directory = 'qmix'

for filename in os.listdir(directory):
    if filename.endswith(".SC2Replay"):

        replay_name = os.path.join(directory, filename)
        base = os.path.splitext(filename)[0]

        map_name = " ".join(base.split('_2018')[0].split('_')[1:])

        subprocess.call(["clear"])
        subprocess.call(["figlet", map_name])

        cmd = "python3 -m pysc2.bin.play --norender --rgb_minimap_size 0 --window_size 2880,1800 --replay {}".format(replay_name)

        subprocess.call(cmd.split())
