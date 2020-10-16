# Fast MVC-style deep MARL framework

Includes implementations of algorithms:
- COMA
- IQL
- VDN (untested)
- QMIX (untested)

## Installation instructions

Build the Dockerfile using 
```
cd docker
./build.sh
```

Set up StarCraft II. Download this specific version here:
(SC2.3.16.1)[http://blzdistsc2-a.akamaihd.net/Linux/SC2.3.16.1.zip]
move to 
> coma/3rdparty/StarCraftII

(and unzip of course using password `iagreetotheeula`)

and then copy 
> /src/envs/starcraft2/maps

to the 

> 3rdparty/StarCraftII/Maps/Melee 
maps folder (which you will have to create first).

## Run an experiment 
```
# run ppo experiments on map MMM2
./run.sh 0 python3 src/main.py --config=ppo_conv1d --env-config=sc2framestack with env_args.map_name=MMM2
```

Run one of the EXPERIMENTs from the folder `src/config/experiments`
on a specific GPU using some special PARAMETERS:
```
cd pymarl/src
../run.sh <GPU> python3 main.py --exp_name=<EXPERIMENT> with <PARAMETERS>
```

Keep an eye on your docker containers, they will be named
`<USER>_pymarl_GPU_<GPU>_<RANDOM>`:
```
docker ps
```

If you do not want them anymore, kill a container named `NAME` with
```
docker kill <NAME>
docker rm <NAME>
```

If you want to get rid of *all* your containers, execute 
```
pymarl/kill.sh
```

## Run SC2 baselines

Run 

> exp_scripts/coma_baselines/run.sh <Number of runs per scenario, e.g. 5>

Results are automatically logged to both tensorboard

> tensorboard --logdir=./results/tb_logs

and, if MongoDB has been set up (see main.py for config details), a database will be created.

If MongoDB is not available, then Sacred will produce output files under

> ./results/sacred

### Copyright and usage restrictions

This framework has been designed, and all core components have been implemented by:
Copyright 2018, Christian Schroeder de Witt (WhiRL, Torr Vision Lab, University of Oxford)
caschroeder@outlook.com

PLEASE KEEP CONFIDENTIAL - USE FOR EDUCATIONAL / ACADEMIC PURPOSES ONLY
