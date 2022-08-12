# Neuroevolution for `gym` Environments

## Overview of [GymNE][evotorch.neuroevolution.gymne.GymNE]

[`gym` environments](https://www.gymlibrary.ml/) are a mainstay of reinforcement learning literature. When attempting to learn agents for these environments with neuroevolution, we typically use an episodic reward. For a given `environment` and `policy`, the evaluation of the `policy` typically follows:

```python
episodic_reward = 0.0
done = False
observation = environment.reset()
while not done:
    action = policy(observation)
    observation, step_reward, done, info = environment.step(action)
    episodic_reward += step_reward
```

where `episodic_reward` is a value learn to maximize.

EvoTorch provides direct support for Neuroevolution of agents for `gym` environments through the [GymNE][evotorch.neuroevolution.gymne.GymNE] class. This class exploits the `gym.make` function, meaning that you can create a reinforcement learning problem simply by passing the name of the environment. For example,

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

Will create a [GymNE][evotorch.neuroevolution.gymne.GymNE] instance for the [`"LunarLanderContinuous-v2"` environment](https://www.gymlibrary.ml/environments/box2d/lunar_lander/) with a `Linear` policy which takes `obs_length` inputs (the number of observations) and returns `act_length` actions (the number of actions). In general, [GymNE][evotorch.neuroevolution.gymne.GymNE] automatically provides both `obs_length`, `act_length` and `obs_space` ([the observation spaces of the policy](https://www.gymlibrary.ml/content/api/#gym.Env.observation_space)) to the instantiation of the policy, meaning that you can also define classes with respect to the dimensions of the environment:

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

It should be noted that [GymNE][evotorch.neuroevolution.gymne.GymNE] has its own function, `to_policy`, which you should use instead of `parameterize_net`. This function wraps `parameterize_net`, but adds any additional layers for observation normalization and action clipping as specified by the problem and environment. Therefore, you should generally use `to_policy` for [GymNE][evotorch.neuroevolution.gymne.GymNE], rather than `parameterize_net`.

[GymNE][evotorch.neuroevolution.gymne.GymNE] has a number of useful arguments that will help you to recreate experiments from neuroevolution literature:

## Controlling the Number of Episodes

Firstly, there is the `num_episodes` argument, which allows you to evaluate individual networks repeatedly and have their episodic rewards averaged. This is particularly useful when studying noisy environments, and when using population-based evolutionary algorithms whose selection procedures and elitism mechanisms may be more sensitive to noise. For example, instantiating the problem

```python
problem = GymNE(
    env_name="LunarLanderContinuous-v2",
    network=CustomPolicy,
    num_actors=4,
    num_episodes=5,
)
```

will specify that each solution should be evaluated $5$ times with their episodic rewards averaged, rather than just the default behaviour of evaluating the reward on a single episode.

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
    ```bash
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
