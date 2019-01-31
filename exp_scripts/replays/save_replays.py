import subprocess
import os
import pprint

maps = {
    "3m":
    {
        "qmix": "brown/qmix__3m__2018-11-28_13-08-08",
    },
    "8m":
    {
        "qmix": "brown/qmix__8m__2018-12-01_04-16-28",
    },
    "25m":
    {
        "qmix": "dgx1/qmix__25m__2018-12-02_03-59-59",
    },
    "5m_6m":
    {
        "qmix": "gandalf/qmix__5m_6m__2018-11-30_00-51-25",
    },
    "8m_9m":
    {
        "qmix": "woma/qmix__8m_9m__2018-11-30_08-06-54",
    },
    "10m_11m":
    {
        "qmix": "savitar/qmix__10m_11m__2018-11-30_06-50-09",
    },
    "27m_30m":
    {
        "qmix": "savitar/qmix__27m_30m__2018-11-30_14-08-47",
    },
    "MMM":
    {
        "qmix": "gandalf/qmix__MMM__2018-12-01_13-12-55",
    },
    "MMM2":
    {
        "qmix": "savitar/qmix__MMM2__2018-11-29_12-45-12",
    },
    "2s3z":
    {
        "qmix": "dgx1/qmix__2s3z__2018-11-30_02-44-28",
    },
    "3s5z":
    {
        "qmix": "gandalf/qmix__3s5z__2018-11-30_05-42-52",
    },
    "3s5z_3s6z":
    {
        "qmix": "savitar/qmix__3s5z_3s6z__2018-11-27_21-53-00",
    },
    "3s_vs_3z":
    {
        "qmix": "savitar/qmix__3s_vs_3z__2018-11-28_11-21-17",
    },
    "3s_vs_4z":
    {
        "qmix": "gandalf/qmix__3s_vs_4z__2018-11-29_23-16-18",
    },
    "3s_vs_5z":
    {
        "qmix": "brown/qmix__3s_vs_5z__2018-11-30_02-13-17",
    },
    "micro_2M_Z":
    {
        "qmix": "brown/qmix__micro_2M_Z__2018-11-30_07-10-04",
    },
    "micro_corridor":
    {
        "qmix": "gollum/qmix__micro_corridor__2018-11-29_18-43-15",
    },
    "micro_focus":
    {
        "qmix": "gollum/qmix__micro_focus__2018-11-30_03-30-34",
    },
    "micro_retarget":
    {
        "qmix": "gollum/qmix__micro_retarget__2018-11-29_02-27-19",
    },
    "micro_baneling":
    {
        "qmix": "gandalf/qmix__micro_baneling__2018-11-30_12-37-24",
    },
    "micro_bane":
    {
        "qmix": "dgx1/qmix__micro_bane__2018-12-05_01-23-13r",
    },
    "micro_colossus":
    {
        "qmix": "gandalf/qmix__micro_colossus__2018-11-27_21-57-32",
    },
}

pprint.pprint(maps)

dir_prefix = "/Users/mika/workspace/pymarl-dev/smac_arxiv_coma_qmix_models"

for i, map_name in enumerate(maps.keys()):

    print("\n", " Map {} - {}/{} ".format(map_name, i, len(maps)).center(60, "*"), "\n")

    experiment = maps[map_name]

    for method, path in experiment.items():

        checkpoint_path = os.path.join(dir_prefix, path)

        load_step = 0  # max
        test_nepisode = 2

        if map_name != "micro_2M_Z":
            continue

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

