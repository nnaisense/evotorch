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

## [GymNE][evotorch.neuroevolution.gymne.GymNE] and [VecGymNE][evotorch.neuroevolution.vecgymne.VecGymNE]

EvoTorch provides two custom `Problem` classes with very similar arguments for easily applying and scaling up neuroevolution across CPUs and GPUs:

* [GymNE][evotorch.neuroevolution.gymne.GymNE]: This class can be used for any Gym environment. Each problem actor (configured using the `num_actors` argument) maintains an instance of the environment to use for evaluation of each policy network in the population. Thus, this class uses parallelization but not vectorization.
* [VecGymNE][evotorch.neuroevolution.vecgymne.VecGymNE]: This class is specially designed for use with [_vectorized_ environments](https://www.gymlibrary.dev/content/vectorising/). In addition to potentially exploiting vectorization for environment simulators, **this class further vectorizes policy evaluations using [functorch](https://pytorch.org/functorch/stable/)** making it possible to fully utilize accelerators such as GPUs for neuroevolution. This is the recommended class to use for environments from massively parallel simulators such as [Brax](https://github.com/google/brax) and [IsaacGym](https://github.com/NVIDIA-Omniverse/OmniIsaacGymEnvs).

!!! info "Brax and IsaacGym environments"
    Brax environments are supported out-of-the-box by `VecGymNE` and can be used to instantiate a problem object by appending `brax::` to an available environment name, such as `brax::humanoid`. For further details regarding Brax environments, see the dedicated example notebook in the repository (`examples/notebooks`). Out-of-the-box support for IsaacGym environments is under development.
 
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

will create a [GymNE][evotorch.neuroevolution.gymne.GymNE] instance for the [`"LunarLanderContinuous-v2"` environment](https://www.gymlibrary.dev/environments/box2d/lunar_lander/) with a `Linear` policy which takes `obs_length` inputs (the number of observations) and returns `act_length` actions (the number of actions).
You can also tell [GymNE][evotorch.neuroevolution.gymne.GymNE] to use a custom [PyTorch module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module) subclass as the policy. Usage of `GymNE` with a custom policy could look like this:


```python
import torch


class CustomPolicy(torch.nn.Module):
    def __init__(self, obs_length: int, act_length: int):
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

Notice that in the example code above, `CustomPolicy` expects two arguments: `obs_length` and `act_length`. `GymNE` will inspect the expected arguments of the custom policy and automatically provide values for them. A custom policy class might expect none, some, or all of the following arguments:

- `obs_length`: Length of the observation vector, as an integer.
- `act_length`: Length of the action vector, as an integer.
- `obs_shape`: Shape of the observation space, as a tuple of integers.
- `act_shape`: Shape of the action space, as a tuple of integers.
- `obs_space`: The observation space, as a [Box](https://www.gymlibrary.dev/api/spaces/#box).
- `act_space`: The action space, as a [Box](https://www.gymlibrary.dev/api/spaces/#box). Please note that, even if the gym environment's action space is discrete, this will be given as a Box. The reason is that `GymNE` always expects the policy network to produce tensors of real numbers (whose shape is specified by the given Box).

You can specify additional arguments to pass to the instantiation of the environment, as you would pass [key-word arguments to `gym.make`](https://www.gymlibrary.dev/environments/box2d/lunar_lander/#arguments), using the `env_config` dictionary. For example:

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

[GymNE][evotorch.neuroevolution.gymne.GymNE] and [VecGymNE][evotorch.neuroevolution.vecgymne.VecGymNE] provide a helper method named `to_policy(...)` to convert a solution (where the solution can be a 1-dimensional tensor or an instance of [Solution][evotorch.core.Solution]) to a policy network. While one could also use the methods `parameterize_net(...)` or `make_net(...)` for similar purposes, it is recommended to use `to_policy(...)`, because `to_policy` will wrap the policy network with observation normalization and action clipping modules to make sure that the inputs to the network are properly processed and the produced actions do not violate the boundaries of the action space.
Further remarks regarding the differences between `parameterize_net(...)` and `to_policy(...)` are: (i) `parameterize_net(...)` is not available in `VecGymNE`, and (ii) `parameterize_net(...)` can be considered a lower level method and strictly expects PyTorch tensors (not `Solution` objects).

[GymNE][evotorch.neuroevolution.gymne.GymNE] and [VecGymNE][evotorch.neuroevolution.vecgymne.VecGymNE] have a number of useful arguments that will help you to recreate experiments from neuroevolution literature:

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
    ActClipWrapperModule(
      (wrapped_module): ObsNormWrapperModule(
        (wrapped_module): CustomPolicy(
          (lin1): Linear(in_features=8, out_features=32, bias=True)
          (act): Tanh()
          (lin2): Linear(in_features=32, out_features=2, bias=True)
        )
        (normalizer): ObsNormLayer()
      )
    )
    ```

you will observe that the policy contains an [ObsNormWrapperModule][evotorch.neuroevolution.net.rl.ObsNormWrapperModule] which automatically applies observation normalization to the input to the policy, and an [ActClipWrapperModule][evotorch.neuroevolution.net.rl.ActClipWrapperModule] which automatically clips the actions to the space of the environment.

## Modifying the step reward

A number of `gym` environments use an `alive_bonus`: a scalar value that is added to the `step_reward` in each step to encourage RL agents to survive for longer. In evolutionary RL, however, [it has been observed](https://arxiv.org/pdf/2008.02387.pdf) that this `alive_bonus` is detrimental and creates unhelpful local optimal. While you can of course disabled particular rewards with the `env_config` argument when available, we also provide direct support for you to decrease the `step_reward` by a scalar amount.

For example, the `"Humanoid-v4"` environment [has an `alive_bonus` value of 5](https://www.gymlibrary.dev/environments/mujoco/humanoid/#rewards). You can easily offset this using the `decrease_rewards_by` keyword argument:

```python
problem = GymNE(
    env_name="Humanoid-v4",
    network=CustomPolicy,
    decrease_rewards_by=5.0,
)
```

which will cause each step to return `5.0` less reward.
