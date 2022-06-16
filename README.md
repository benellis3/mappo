# Fast MVC-style deep MARL framework

Includes implementations of algorithms:
- Independent PPO

## Installation instructions

Build the Dockerfile using 
```
cd docker
./build.sh
```

Set up StarCraft II by running `./install_sc2.sh`. Make sure to install the `32x32_flat.SC2Map` by copying it to the `$SC2PATH/Maps` directory from 
the [smacv2 repo](https://github.com/oxwhirl/smacv2).

## Running Hyperparameter Tuning

1. Make sure to either set `use_wandb=False` or to change the `project` and `entity` in `src/config/default.yaml`.
2. Build the docker file as above.
3. Install StarCraft II as above.
4. Run the `./run_exp.sh` script.


### Copyright and usage restrictions

This framework has been designed, and all core components have been implemented by:
Copyright 2018, Christian Schroeder de Witt (WhiRL, Torr Vision Lab, University of Oxford)
caschroeder@outlook.com
