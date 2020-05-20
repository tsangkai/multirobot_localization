# Multirobot Localization Simulation

This is the simulation code for the paper "Resilient Multirobot Cooperative Localization with Explicit Communication" submitted to *IEEE Transaction on Robotics*.

## Usage

All the simulation parameters are specified in `sim_env.py`.

For GS (global-state) algorithms, one can directly run `gs_ci_sim.py` or `gs_sci_sim.py`.

For LS (local-state) algorithms, one can run `ls_sim.py` with argument `cen`, `bda`, or `ci` to specify which algorithm will be applied.

## Simulation Result

![Performance plot](plot/performance.png)

This simulation uses the topology presented in our paper.


![Performance plot](plot/performance_dr.png)

This simulation with motion propagation update only.