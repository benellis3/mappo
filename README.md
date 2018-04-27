# Fast MVC-style deep MARL framework

Work in progress implementation of Counterfactual Multi-Agent Policy Gradients
(should be stable now)

## Installation instructions

Build the Dockerfile using 
> $ ./run.sh

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

## Run SC2 baselines

Run 

> exp_scripts/coma_baselines/run.sh <Number of runs per scenario, e.g. 5>

Results are automatically logged to both tensorboard

> tensorboard --logdir=./results/tb_logs

and, if MongoDB has been set up (see main.py for config details), a database will be created.

If MongoDB is not available, then Sacred will produce output files under

> ./results/sacred

### Copyright and usage restrictions

All code in this repo (except where explicitely denoted otherwise):
Copyright 2018, Christian Schroeder de Witt (WhiRL, Torr Vision Lab, University of Oxford)
caschroeder@outlook.com

PLEASE KEEP CONFIDENTIAL
