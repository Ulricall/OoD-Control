# OoD-Control: Out-of-Distribution Generalization for Flight Control
We implement our algorithm and baseline algorithms on two models: inverted pendulum and quadrotor.

## Inverted pendulum
`cd pendulum` for inverted pendulum experiment.

To simply run the experiment under a certain setting, just run `run.py` for the complete training and testing task. You can also run the experiment under different settings.
Examples:
```python
# simply run
python run.py
# assign a wind condition
python run.py --wind breeze
# do not keep the log:
python run.py --logs 0
```
After the logs are created, you can run `plot_result.py` to plot the results in `./pics`.

## Quadrotor
`cd quadrotor` for quadrotor experiment.

Similarly, run `run.py` for simple start. More settings are shown in following examples.
```python
# simply run
python run.py
# do not keep the log
python run.py --logs 0
# assign a trajectory
python run.py --trace fig8
# assign a wind condition
python run.py --wind strong_breeze
```
Trajectory can be chosen from `{fig8, hover, sin, spiral}`.
After the logentry is created, you can run `plot_result.py --trace {fig8, hover, sin, spiral}` to create 3D traces in `./traces` and 2D projection in `./projections`. 

For both experiments, the argument parse `--wind {breeze, strong_breeze, gale}` is used to determine the wind velocity in the experiment.
