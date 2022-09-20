# Example scripts

The scripts provided here demonstrate key features of EvoTorch for new users and serve as starting points for solving custom problems.

---
## Black-box optimization

- [bbo_vectorized.py](./bbo_vectorized.py): demonstrates single objective black-box optimization using a distribution-based algorithm, accelerated using vectorization on a single GPU/CPU.
- [moo_parallel.py](./moo_parallel.py): demonstrates multi-objective optimization using parallelization on all CPU cores without vectorization.

---
## Reinforcement Learning

- [rl_gym.py](./rl_gym.py): demonstrates how to solve a simple [Gym](https://www.gymlibrary.ml/) problem using the PGPE algorithm and ClipUp optimizer.

---
## Paper re-implementation (RL)

The script [rl_clipup.py](./rl_clipup.py) reproduces experiments from the following [paper](https://arxiv.org/abs/2008.02387):
```
Nihat Engin Toklu, Paweł Liskowski, and Rupesh Kumar Srivastava.
"ClipUp: a simple and powerful optimizer for distribution-based policy evolution."
International Conference on Parallel Problem Solving from Nature. Springer, Cham, 2020.
https://arxiv.org/abs/2008.02387
```
It allows you to train policies for the Lunar Lander, Walker-2D and Humanoid environments from [Gym](https://www.gymlibrary.ml/) as well as the [PyBullet](https://pybullet.org/) Humanoid.


### Requirements

```bash
# Necessary: Used for configs and logging results to files or databases
pip install sacred
# Optional: box2d for Lunar Lander, mujoco for Walker-2D and Humanoid
pip install 'gym[box2d,mujoco]'
# Optional: For PyBullet Humanoid
pip install pybullet
```

### Running experiments

The script `rl_clipup.py` is a command line utility for training and saving policies for classical gym environments. Its basic usage is as follows:

```bash
python rl_clipup.py -F RESULTS_DIR           \
    with env_name=GYM_ENVIRONMENT_ID  \
         policy=POLICY                \
         save_interval=SAVE_INTERVAL  \
         hyperparameter1=value1       \
         hyperparameter2=value2       \
         ...
```

where:

- `RESULTS_DIR` is the directory into which the logs and the artifacts (pickle files containing the evolved policies and metadata) will be saved.
- `GYM_ENVIRONMENT_ID` is a string such as `"LunarLander-v2"` or `"Humanoid-v4"`, etc., indicating which environment is to be solved.
- `POLICY` is a string which determines the architecture of the neural network that will be trained. By default, this is set as a single-hidden-layered network (the hidden layer having 16 neurons), represented by the string: `"Linear(obs_length, 16) >> Tanh() >> Linear(16, act_length)"` (where `obs_length` represents the length of the observation vector, and `act_length` represents the length of the action vector). To use a linear policy with bias, one can set this string as: `"Linear(obs_length, act_length)"`. Alternatively, for having a linear policy without bias, one can set this string as: `"Linear(obs_length, act_length, bias=False)"`.
- `SAVE_INTERVAL` can be an integer `n`, which means that at every `n` generations, the current state of the evolved policy will be saved to disk. `SAVE_INTERVAL` can also be set as "last", which means that the evolved policy will be saved only at the end of the entire run. For large experiments, it is recommended to set this as an integer such as 10, or 50, etc.

To see the full list of available hyperparameters with their explanations and their default values, one can use the following shell command:

```bash
python rl_clipup.py print_config
```

One can also see the effects of changing hyperparameter(s), without executing the experiment, via:

```bash
python rl_clipup.py print_config with hyperparameter1=newvalue1 hyperparameter2=newvalue2 ...
```

### Available pre-configurations

`rl_clipup.py` comes with pre-configurations for certain environments.

For example, the following shell command:

```bash
python rl_clipup.py -F RESULTS_DIR with lunarlander
```

...evolves a LunarLanderContinuous-v2 and saves it into `RESULTS_DIR`.

Also, the following command:

```bash
python rl_clipup.py -F RESULTS_DIR with pybullet_humanoid save_interval=50
```

...starts an evolutionary computation run for solving `pybullet_envs:HumanoidBulletEnv-v0`, and saves the policy at every 50 generations. Such an explicit `save_interval` value is recommended for pybullet humanoid, since the computational experiments for this environment last long, and one might want to look at how the current agent is behaving without having to wait until the end of the run.

Other available pre-configurations are `walker` (for the MuJoCo environment `Walker-v4`) and `humanoid` (for the MuJoCo environment `Humanoid-v4`).

### Script for testing and/or visualizing policies: `rl_enjoy.py`

The policies trained/evolved by `rl_clipup.py` are saved as pickle files. These pickle files can then be tested with the help of `rl_enjoy.py`.
The simplest usage is:

```bash
python rl_enjoy.py PICKLE_FILE
```

...which loads the policy saved in the specified pickle file, runs and visualizes it in an episode of its environment, and prints the obtained total reward.

For further help, one might use the following shell command:

```bash
python rl_enjoy.py --help
```
