import subprocess
import os
import pprint

maps = {
    "micro_corridor" : {"qmix" : "qmix__micro_corridor__2018-11-11_14-14-18",
                        "iql": "iql__micro_corridor__2018-11-10_02-24-14"},
    "3s5z_3s6z": {"qmix": "qmix_parallel__2018-11-08_01-26-50"},
    "MMM2": {"qmix": "qmix__MMM2__2018-11-14_13-53-07"},
    "3s_5z": {"coma": "coma__3s_5z__2018-11-11_17-16-49",
              "qmix": "qmix_parallel__2018-11-07_18-47-14"},
    "5m_6m": {"qmix": "qmix_parallel__2018-11-07_01-43-50",
              "coma": "coma__5m_6m__2018-11-12_08-00-25"},
    "3s_vs_5z": {"iql": "iql__3s_vs_5z__2018-11-11_15-04-54",
                 "qmix": "qmix__3s_vs_5z__2018-11-14_04-28-25"},
    "8m_9m": {"qmix": "qmix__8m_9m__2018-11-14_10-38-51"},
    "10m_11m": {"qmix": "qmix_parallel__2018-11-08_05-28-54",
                "iql": "iql__10m_11m__2018-11-14_03-35-03",
                "coma": "coma__10m_11m__2018-11-13_18-58-15"},
    "2s_3z": {"iql": "iql__2s_3z__2018-11-13_19-57-27",
              "qmix": "qmix__2s_3z__2018-11-14_04-38-10",
              "coma": "coma__2s_3z__2018-11-14_12-43-37"},
    "3m_3m": {"iql": "iql__3m_3m__2018-11-14_03-11-22",
              "coma": "coma__3m_3m__2018-11-14_07-00-54",
              "qmix": "qmix__3m_3m__2018-11-14_03-54-53"},
    "3s_vs_3z": {"qmix": "qmix__3s_vs_3z__2018-11-14_20-36-01",
                 "iql": "iql__3s_vs_3z__2018-11-14_15-13-51",
                 "coma": "coma__3s_vs_3z__2018-11-14_08-15-09"},
    "3s_vs_4z": {"iql": "iql__3s_vs_4z__2018-11-11_10-51-00",
                 "qmix": "qmix__3s_vs_4z__2018-11-14_04-24-07"},
    "8m_8m": {"iql": "iql__8m_8m__2018-11-14_00-06-30",
              "coma": "coma__8m_8m__2018-11-14_15-20-24",
              "qmix": "qmix__8m_8m__2018-11-14_13-24-22"},
    "MMM": {"coma": "coma__MMM__2018-11-13_20-33-38",
            "qmix": "qmix__MMM__2018-11-14_14-21-08",
            "iql": "iql__MMM__2018-11-14_04-26-02"},
    "micro_2M_Z": {"iql": "iql__micro_2M_Z__2018-11-14_11-09-31",
                   "coma": "coma__micro_2M_Z__2018-11-14_12-52-08"},
                   #"qmix": "qmix_parallel_micro1__2018-10-19_10-39-33"}, we have this already
    "micro_baneling": {"coma": "coma__micro_baneling__2018-11-10_02-37-10",
                       "qmix": "qmix__micro_baneling__2018-11-14_06-21-39"},
    "micro_colossus2" : {"qmix": "qmix__micro_colossus2__2018-11-13_22-01-06"},
    "micro_focus" : {"qmix": "qmix__micro_focus__2018-11-14_07-47-00"},
    "micro_retarget" : {"qmix": "qmix__micro_retarget__2018-11-11_02-36-11",
                        "iql": "iql__micro_retarget__2018-11-10_02-22-13",
                        "coma": "coma__micro_retarget__2018-11-10_02-34-17"}
}

pprint.pprint(maps)

dir_prefix = "/Users/mika/Downloads/models2"

for i, map_name in enumerate(maps.keys()):

    print("\n", " Map {} - {}/{} ".format(map_name, i, len(maps)).center(60, "*"), "\n")

    experiment = maps[map_name]

    for method, path in experiment.items():

        #if method == "qmix" or method == "iql":
        #    continue

        checkpoint_path = os.path.join(dir_prefix, path)

        load_step = 10**7
        test_nepisode = 5

        command = "python3 src/main.py --config={} --env-config=sc2 with env_args.map_name={} checkpoint_path={} load_step={} evaluate=True save_replay=True test_nepisode={} env_args.save_replay_prefix={} no-mongo=True runner=episode".format(
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

