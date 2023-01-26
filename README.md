# OoD-Control: Out-of-Distribution Generalization for Flight Control
We implement our algorithm and baseline algorithms on two models: inverted pendulum and quadrotor.

## Inverted pendulum
`cd pendulum` for inverted pendulum experiment.

Run `run.py` for the complete training and testing task. You can add `--logs 1` to keep the logs in `./logs` and `./F_logs`. After the logs are created, you can run `plot_result.py` to plot the results in `./pics`.

## Quadrotor
`cd quadrotor` for quadrotor experiment.

Run `run.py` for the complete training and testing task. You can add `--logs 1` to keep the logs in `./logs` and `--trace {fig8, hover, sin, spiral}` for different trajectories.

After the logentry is created, you can run `plot_result.py --trace {fig8, hover, sin, spiral}` to create 3D traces `./traces` and 2D projection in `./projections`. 

For both experiments, the argument parse `--wind {breeze, strong_breeze, gale}` is used to determine the wind velocity in the experiment.
