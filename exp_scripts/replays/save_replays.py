import subprocess
import os
import pprint

maps = {
    "3m":
    {
        "coma": "",
        "iql": "",
        "qmix": "brown/qmix__3m__2018-11-27_21-44-12"
    },
    "8m":
    {
        "coma": "",
        "iql": "",
        "qmix": "brown/qmix__8m__2018-12-01_08-00-51"
    },
    "25m":
    {
        "coma": "",
        "iql": "",
        "qmix": "dgx1/qmix__25m__2018-12-02_06-00-59"
    },
    "5m_6m":
    {
        "coma": "",
        "iql": "",
        "qmix": "woma/qmix__5m_6m__2018-11-30_14-11-07"
    },
    "8m_9m":
    {
        "coma": "",
        "iql": "",
        "qmix": "woma/qmix__8m_9m__2018-11-28_22-04-57"
    },
    "10m_11m":
    {
        "coma": "",
        "iql": "",
        "qmix": "savitar/qmix__10m_11m__2018-11-28_16-30-44"
    },
    "27m_30m":
    {
        "coma": "",
        "iql": "",
        "qmix": "savitar/qmix__27m_30m__2018-11-30_14-08-47"
    },
    "MMM":
    {
        "coma": "",
        "iql": "",
        "qmix": "gandalf/qmix__MMM__2018-12-03_12-22-52"
    },
    "MMM2":
    {
        "coma": "",
        "iql": "",
        "qmix": "savitar/qmix__MMM2__2018-11-29_08-30-08"
    },
    "2s3z":
    {
        "coma": "",
        "iql": "",
        "qmix": "dgx1/qmix__2s3z__2018-11-30_19-54-29"
    },
    "3s5z":
    {
        "coma": "",
        "iql": "",
        "qmix": "gandalf/qmix__3s5z__2018-12-02_08-11-59"
    },
    "3s5z_3s6z":
    {
        "coma": "",
        "iql": "",
        "qmix": "savitar/qmix__3s5z_3s6z__2018-11-28_14-48-35"
    },
    "3s_vs_3z":
    {
        "coma": "",
        "iql": "",
        "qmix": "savitar/qmix__3s_vs_3z__2018-11-27_21-53-29"
    },
    "3s_vs_4z":
    {
        "coma": "",
        "iql": "",
        "qmix": "gandalf/qmix__3s_vs_4z__2018-11-30_19-33-35"
    },
    "3s_vs_5z":
    {
        "coma": "",
        "iql": "",
        "qmix": "brown/qmix__3s_vs_5z__2018-11-30_21-39-27"
    },
    "micro_2M_Z":
    {
        "coma": "",
        "iql": "",
        "qmix": "brown/qmix__micro_2M_Z__2018-11-27_21-55-29"
    },
    "micro_corridor":
    {
        "coma": "",
        "iql": "",
        "qmix": "gollum/qmix__micro_corridor__2018-11-29_18-43-15"
    },
    "micro_focus":
    {
        "coma": "",
        "iql": "",
        "qmix": "gollum/qmix__micro_focus__2018-11-30_16-49-48"
    },
    "micro_retarget":
    {
        "coma": "",
        "iql": "",
        "qmix": "gollum/qmix__micro_retarget__2018-11-28_13-55-48"
    },
    "micro_baneling":
    {
        "coma": "",
        "iql": "",
        "qmix": "gandalf/qmix__micro_baneling__2018-11-27_21-56-16"
    },
    "micro_bane":
    {
        "coma": "",
        "iql": "",
        "qmix": "dgx1/qmix__micro_bane__2018-12-03_17-23-20"
    },
    "micro_colossus":
    {
        "coma": "",
        "iql": "",
        "qmix": "gandalf/qmix__micro_colossus__2018-12-04_18-21-13"
    },
}

pprint.pprint(maps)

dir_prefix = "/Users/mika/workspace/pymarl-dev/smac_arxiv_coma_qmix_models"

for i, map_name in enumerate(maps.keys()):

    print("\n", " Map {} - {}/{} ".format(map_name, i, len(maps)).center(60, "*"), "\n")

    experiment = maps[map_name]

    if map_name != "25m":
        continue

    for method, path in experiment.items():

        if not path:
            continue

        checkpoint_path = os.path.join(dir_prefix, path)

        load_step = 0  # max
        test_nepisode = 2

        command = "python3 src/main.py --config={} --env-config=sc2 with env_args.map_name={} checkpoint_path={} load_step={} evaluate=True save_replay=True test_nepisode={} env_args.save_replay_prefix={} no-mongo=True runner=episode env_args.debug=True".format(
            method,
            map_name,
            checkpoint_path,
            load_step,
            test_nepisode,
            method + '_' + map_name
        )

        if not os.path.isdir(checkpoint_path):
            print("Model {} does not exist!".format(checkpoint_path))

        print("Executing the following command:\n", command)

        subprocess.call(command.split())

