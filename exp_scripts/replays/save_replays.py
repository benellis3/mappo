import subprocess
import os
import pprint

maps = {
    "3m":
    {
        "qmix": "brown/qmix__3m__2018-11-27_21-44-12"
    },
    "8m":
    {
        "qmix": "brown/qmix__8m__2018-12-01_01-13-10"
    },
    "25m":
    {
        "qmix": "dgx1/qmix__25m__2018-11-30_13-19-59"
    },
    "5m_6m":
    {
        "qmix": "woma/qmix__5m_6m__2018-11-28_17-09-20"
    },
    "8m_9m":
    {
        "qmix": "savitar/qmix__8m_9m__2018-11-29_23-54-30"
    },
    "10m_11m":
    {
        "qmix": "savitar/qmix__10m_11m__2018-11-28_16-30-44"
    },
    "27m_30m":
    {
        "qmix": "savitar/qmix__27m_30m__2018-11-30_14-08-47"
    },
    "MMM":
    {
        "qmix": "gandalf/qmix__MMM__2018-11-27_21-48-03"
    },
    "MMM2":
    {
        "qmix": "savitar/qmix__MMM2__2018-11-28_17-21-45"
    },
    "2s3z":
    {
        "qmix": "dgx1/qmix__2s3z__2018-11-27_21-46-40"
    },
    "3s5z":
    {
        "qmix": "gandalf/qmix__3s5z__2018-11-27_21-47-22"
    },
    "3s5z_3s6z":
    {
        "qmix": "savitar/qmix__3s5z_3s6z__2018-11-28_14-48-35"
    },
    "3s_vs_3z":
    {
        "qmix": "savitar/qmix__3s_vs_3z__2018-11-27_21-53-29"
    },
    "3s_vs_4z":
    {
        "qmix": "gandalf/qmix__3s_vs_4z__2018-11-30_19-33-35"
    },
    "3s_vs_5z":
    {
        "qmix": "brown/qmix__3s_vs_5z__2018-11-29_07-10-51"
    },
    "micro_2M_Z":
    {
        "qmix": "brown/qmix__micro_2M_Z__2018-11-27_21-55-29"
    },
    "micro_corridor":
    {
        "qmix": "gollum/qmix__micro_corridor__2018-11-29_18-43-15"
    },
    "micro_focus":
    {
        "qmix": "gollum/qmix__micro_focus__2018-11-30_16-49-48"
    },
    "micro_retarget":
    {
        "qmix": "gollum/qmix__micro_retarget__2018-11-29_06-51-21"
    },
    "micro_baneling":
    {
        "qmix": "gandalf/qmix__micro_baneling__2018-11-27_21-56-16"
    },
    "micro_bane":
    {
        "qmix": "dgx1/qmix__micro_bane__2018-12-03_17-23-20"
    },
    "micro_colossus":
    {
        "qmix": "gandalf/qmix__micro_colossus__2018-11-29_13-08-00"
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

