# Neuroevolution for `gym` Environments

A common use case for neuroevolution is to evolve neural network _policies_ that maximize the _episodic return_ (the total reward) of an agent in a Reinforcement Learning (RL) environment.
EvoTorch provides built-in support for environments that support the commonly-used [Gym](https://www.gymlibrary.dev/) API.
In this API, the evaluation of the `policy` for a given `environment` and `policy` typically follows:

```python
episodic_return = 0.0
terminated, truncated = False, False
observation, info = environment.reset()
while not (terminated or truncated):
    action = policy(observation)
    observation, step_reward, terminated, truncated, info = environment.step(action)
    episodic_return += step_reward
```

where `episodic_return` is the value we wish to maximize, and the `policy` is represented by a (PyTorch) neural network that wish to train.

!!! note "Old vs New Gym API"
    Gym's long-used API changed in Sept 2022 with the release of [v0.26.0](https://github.com/openai/gym/releases/tag/0.26.0). EvoTorch supports environments defined using both new and old APIs. 

## [GymNE][evotorch.neuroevolution.gymne.GymNE] and [VecGymNE][evotorch.neuroevolution.gymne.GymNE]

EvoTorch provides two custom `Problem` classes with very similar arguments for easily applying and scaling up neuroevolution across CPUs and GPUs:

* [GymNE][evotorch.neuroevolution.gymne.GymNE]: This class can be used for any Gym environment. Each problem actor (configured using the `num_actors` argument) maintains an instance of the environment to use for evaluation of each policy network in the population. Thus, this class uses parallelization but not vectorization.
* [VecGymNE][evotorch.neuroevolution.gymne.GymNE]: This class is specially designed for use with [_vectorized_ environments](https://www.gymlibrary.dev/content/vectorising/). In addition to potentially exploiting vectorization for environment simulators, **this class further vectorizes policy evaluations using [functorch](https://pytorch.org/functorch/stable/)** making it possible to fully utilize accelerators such as GPUs for neuroevolution. This is the recommended class to use for environments from massively parallel simulators such as [Brax](https://github.com/google/brax) and [IsaacGym](https://github.com/NVIDIA-Omniverse/OmniIsaacGymEnvs).

For the simplest cases, you can create a reinforcement learning problem simply by specifying the name of the environment. For example,

```python
from evotorch.neuroevolution import GymNE

problem = GymNE(
    # Name of the environment
    env_name="LunarLanderContinuous-v2",
    # Linear policy mapping observations to actions
    network="Linear(obs_length, act_length)",
    # Use 4 available CPUs. Note that you can modify this value,
    # or use 'max' to exploit all available GPUs
    num_actors=4,
)
```

will create a [GymNE][evotorch.neuroevolution.gymne.GymNE] instance for the [`"LunarLanderContinuous-v2"` environment](https://www.gymlibrary.ml/environments/box2d/lunar_lander/) with a `Linear` policy which takes `obs_length` inputs (the number of observations) and returns `act_length` actions (the number of actions). In general, [GymNE][evotorch.neuroevolution.gymne.GymNE] automatically provides both `obs_length`, `act_length` and `obs_space` ([the observation spaces of the policy](https://www.gymlibrary.ml/content/api/#gym.Env.observation_space)) to the instantiation of the policy, meaning that you can also define classes with respect to the dimensions of the environment:

```python
from gym.spaces import Space
import torch


class CustomPolicy(torch.nn.Module):
    def __init__(self, obs_length: int, act_length: int, obs_space: Space):
        super().__init__()
        self.lin1 = torch.nn.Linear(obs_length, 32)
        self.act = torch.nn.Tanh()
        self.lin2 = torch.nn.Linear(32, act_length)

    def forward(self, data):
        return self.lin2(self.act(self.lin1(data)))


problem = GymNE(
    env_name="LunarLanderContinuous-v2",
    network=CustomPolicy,
    num_actors=4,
)
```

You can specify additional arguments to pass to the instantiation of the environment, as you would pass [key-word arguments to `gym.make`](https://www.gymlibrary.ml/environments/box2d/lunar_lander/#arguments), using the `env_config` dictionary. For example:

```python
problem = GymNE(
    env_name="LunarLanderContinuous-v2",
    env_config={
        "gravity": -1e-5,
    },
    network=CustomPolicy,
)
```

will effectively disable `gravity` in the `"LunarLanderContinuous-v2"` environment.

It should be noted that [GymNE][evotorch.neuroevolution.gymne.GymNE] (and [VecGymNE][evotorch.neuroevolution.gymne.VecGymNE]) has its own method, `to_policy()`, which you should use instead of `parameterize_net`. This method wraps `parameterize_net()`, but adds any additional layers for observation normalization and action clipping as specified by the problem and environment. Therefore, you should generally use `to_policy()` for [GymNE][evotorch.neuroevolution.gymne.GymNE] and [VecGymNE][evotorch.neuroevolution.gymne.VecGymNE], rather than `parameterize_net()`.

[GymNE][evotorch.neuroevolution.gymne.GymNE] has a number of useful arguments that will help you to recreate experiments from neuroevolution literature:

## Controlling the Number of Episodes

The `num_episodes` argument allows you to evaluate individual networks repeatedly and have their episodic returns averaged. This is particularly useful when studying noisy environments, and when using population-based evolutionary algorithms whose selection procedures and elitism mechanisms may be more sensitive to noise. For example, instantiating the problem

```python
problem = GymNE(
    env_name="LunarLanderContinuous-v2",
    network=CustomPolicy,
    num_actors=4,
    num_episodes=5,
)
```

will specify that each solution should be evaluated $5$ times with their episodic returns averaged, rather than just the default behaviour of evaluating the return from a single episode.

## Using Observation Normalization

In [recent neuroevolution studies](https://arxiv.org/pdf/1703.03864.pdf), observation normalization has been observed to be particularly helpful. Observation normalization tracks the expectation $\mathbb{E}[o_i]$ and variance $\mathbb{V}[o_i]$ for each observation variable $o_i$ as observations are drawn from the environment. Then the observation passed to the policy is the modified:

$$o_i' = \frac{o_i - \mathbb{E}[o_i]}{\sqrt{\mathbb{V}[o_i]}}$$

While in practice this means that the problem is non-stationary, as the expection and variance of each variable is updated as new observations are drawn, the normalizing effect on the observations generally makes successful configuration of neuroevolution substantially easier. You can enable observation normalization using the boolean flag `observation_normalization` e.g.

```python
problem = GymNE(
    env_name="LunarLanderContinuous-v2",
    network=CustomPolicy,
    num_actors=4,
    observation_normalization=True,
)
```

And if you then evaluate a batch (to ensure the observation normalization statistics are initialized) and print the `problem`'s policy:

```python
problem.evaluate(problem.generate_batch(2))
print(problem.to_policy(problem.make_zeros(problem.solution_length)))
```

???+ abstract "Output"
    ```
    Sequential(
        (0): ObsNormLayer()
        (1): CustomPolicy(
            (lin1): Linear(in_features=8, out_features=32, bias=True)
            (act): Tanh()
            (lin2): Linear(in_features=32, out_features=2, bias=True)
        )
        (2): ActClipLayer()
    )
    ```

you will observe that the policy contains an [ObsNormLayer][evotorch.neuroevolution.net.rl.ObsNormLayer] which automatically applies observation normalization to the input to the policy, and an [ActClipLayer][evotorch.neuroevolution.net.rl.ActClipLayer] which automatically clips the actions to the space of the environment.

## Modifying the step reward

A number of `gym` environments use an `alive_bonus`: a scalar value that is added to the `step_reward` in each step to encourage RL agents to survive for longer. In evolutionary RL, however, [it has been observed](https://arxiv.org/pdf/2008.02387.pdf) that this `alive_bonus` is detrimental and creates unhelpful local optimal. While you can of course disabled particular rewards with the `env_config` argument when available, we also provide direct support for you to decrease the `step_reward` by a scalar amount.

For example, the `"Humanoid-v4"` environment [has an `alive_bonus` value of 5](https://www.gymlibrary.ml/environments/mujoco/humanoid/#rewards). You can easily offset this using the `decrease_rewards_by` keyword argument:

```python
problem = GymNE(
    env_name="Humanoid-v4",
    network=CustomPolicy,
    decrease_rewards_by=5.0,
)
```

which will cause each step to return `5.0` less reward.
