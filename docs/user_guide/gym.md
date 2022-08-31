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

## Working with stateful policies

When working with reinforcement-learning settings it is common to use a stateful policy network, such as RNNs and LSTMs. EvoTorch expects this to be handled within the definition of your policy network. For example, a simple RNN which maintains its internal hidden state is implemented as:

```py
from torch import nn
class SimpleRNN(nn.Module):

    def __init__(self, obs_length: int, act_length: int, obs_space: Space, hidden_dim: int = 32):
        super().__init__()
        # First linear layer takes both observations and previous hidden state as input
        self.lin1 = torch.nn.Linear(obs_length + hidden_dim, hidden_dim)
        # Activation is Tanh
        self.act = torch.nn.Tanh()
        # Second linear layer maps hidden state to actions
        self.lin2 = torch.nn.Linear(hidden_dim, act_length)

        # Initially the hidden state is stored as None
        self._hidden_state = None
        self._hidden_dim = hidden_dim

    def forward(self, observations):
        # If the hidden state is not initialized, initialize it and maintain device and dtype consistency
        if self._hidden_state is None:
            self._hidden_state = torch.zeros(self._hidden_dim, device = observations.device, dtype = observations.dtype)
        # Concatenate observations with hidden state as input to self.lin1
        policy_input = torch.cat(observations, self._hidden_state, dim = -1)
        # Update the internal hidden state
        self._hidden_state = self.act(self.lin1(policy_input))
        # Compute the actions from the internal hidden state
        act = self.lin2(self._hidden_state)
        return act

    def reset(self):
        # Whenever the environment is reset, this method will be called to
        # Resetting self._hidden_state to None means that the hidden state from previous episodes will be cleared
        self._hidden_state = None
```

The critical point to note is the addition of the `reset()` method. Whenever an episode ends and the `gym` environment is reset, `GymProblem` will crawl your policy module and call the `reset()` method of any module within it (including itself). You can use this functionality to clear memory from the previous episode so that solutions are evaluated independently from previous evaluations. In the above example `SimpleRNN`, the policy maintains a `_hidden_state` variable which is reset via the `reset()` method whenever the environment is also reset, so that no hidden state is kept between independent episodes. 

To make usage of stateful policies, we provide wrapped versions of `torch.nn.RNN`  and `torch.nn.LSTM`, which are [evotorch.neuroevolution.net.RecurrentNet][evotorch.neuroevolution.net.RecurrentNet] and [evotorch.neuroevolution.net.LSTMNet][evotorch.neuroevolution.net.RecurrentNet], respectively. These wrapped versions work exactly the same as their underlying `torch` equivalents, except that they internally handle the hidden state of the recurrent network and therefore only require a single input e.g. the observations from the environment or some earlier layer in the policy. These wrapped classes also handle the implementation of the `reset()` method. You can use these special layers in the string representation too:

```py
from evotorch.neuroevolution import GymNE

problem = GymNE(
    env_name="LunarLanderContinuous-v2",
    # Recurrent layer with hidden dimension 64, followed by linear mapping to actions
    network="RecurrentNet(obs_length, 64) >> Linear(64, act_length)",
    num_actors=4,
)
```