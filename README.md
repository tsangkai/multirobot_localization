# Multirobot Localization Simulation

This is the simulation code for the paper "Resilient Multirobot Cooperative Localization with Explicit Communication" submitted to *IEEE Transaction on Robotics*.

## Multirobot Cooperative Localization Algorithm based on Covariance Intersection

This is our algorithm developed in the paper.

## Other Multirobot Cooperative Localization Algorithms

We simulate 4 other algorithms for comparision. We rename and classify them to emphasize the structural difference. The first category is the local state (LS) algorithms, where each robot only tracks its own spatial state. The other category is the global state (GS) algorithms, where each robot tracks the state of the entire robot team.

### LS-Cen

### GS-SCI

We simulate the algorithm based on our proposed structure. However, the communication update is realized by the split covariance intersection in [].

## Usage

All the simulation parameters are specified in `sim_env.py`. One can specify the random seed here as well.

For GS algorithms, one can directly run `gs_ci_sim.py` or `gs_sci_sim.py`.

For LS algorithms, one can run `ls_sim.py` with argument `cen`, `bda`, or `ci` to specify which algorithm will be applied.

## Covariance Boundedness

## Observation and Communication Topologies


## TODO

- generalize the number of robots

## Reference