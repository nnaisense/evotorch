# Distribution-based evolutionary algorithms: advanced usage

A distribution-based evolutionary algorithm is an algorithm which samples its population from a probability distribution.
It can also be considered that these algorithms perform gradient-based search although they do not analytically derive the gradients.
Instead of deriving, they _estimate_ the gradients with the help of the fitnesses of the sampled solutions.
The algorithms [PGPE][evotorch.algorithms.distributed.gaussian.PGPE] (Policy Gradients with Parameter-based Exploration), [SNES][evotorch.algorithms.distributed.gaussian.SNES] (Separable Natural Evolution Strategies), and [CEM][evotorch.algorithms.distributed.gaussian.CEM] (Cross Entropy Method) belong to the class of distribution-based algorithms.

EvoTorch provides certain features that can scale up the distribution-based search algorithms to handle large computational requirements of challenging tasks.

## Distributed mode

When the problem at hand is parallelized across multiple CPUs, the default behavior of a distribution-based search algorithm is as follows:

- For each generation:
  - Sample the population from the current search distribution
  - Split the population to sub-populations and send them to remote actors for parallelized evaluation
  - Collect the evaluation results
  - Estimate the gradients for the parameters of the search distribution (where the gradients are estimated in the direction of more promising solutions)
  - Update the search distribution according to the gradients

Although this behavior is a sensible default for many cases, there are some other cases in which one might be able to squeeze further performance.
In more details, imagine solving a challenging optimization problem which requires a huge population size.
With such a huge population size, sub-populations to be sent to the remote actors will also be huge, and will result in a significant interprocess communication.
To decrease such interprocess communication, EvoTorch provides an alternative way of parallelization which is called "distributed mode".
In distributed mode, the following behavior is enabled:

- For each generation:
  - Send the current search distribution's parameters to remote actors
    - Upon receiving the distribution parameters, each actor will generate its own sub-population and estimate its own gradient
  - Collect the gradients estimated by the actors, and compute the final gradient, which is the average of the collected gradients
  - Update the search distribution according to the gradients

In this distributed mode, the interprocess communication is reduced to sending distribution parameters and gradients.
To switch to the distributed mode, it is enough to set the keyword argument `distributed` as `True` while instantiating a distribution-based algorithm, as shown below:

```python
from evotorch.algorithms import PGPE

# Somehow instantiate the problem
problem = Problem(
    ...,
    num_actors="max",  # use all available CPUs
)

# Instantiate the PGPE algorithm in distributed mode
# (also works with SNES and CEM)
searcher = PGPE(
    problem,
    ...,
    distributed=True,
)
```

## Using distributed mode for efficient multi-GPU optimization

Some problems can be configured in such a way that each actor is assigned a GPU, therefore multiple GPUs can be used for efficient parallelized evaluation.
For example, the supervised learning neuro-evolution problem, [SupervisedNE][evotorch.neuroevolution.supervisedne.SupervisedNE], is configured by default to allocate a GPU for each actor if a GPU driver recognized by PyTorch is available.
Other than [SupervisedNE][evotorch.neuroevolution.supervisedne.SupervisedNE], any Problem with its `num_gpus_per_actor` set as a positive real number (or set as "max") is configured to allocate GPU(s) for its actors (see the reference documentation of the [Problem][evotorch.core.Problem] class for GPU allocation details).

When a search algorithm in distributed mode is used on such a multi-GPU problem, each actor will do the following on the GPU assigned to itself:

- Sample a sub-population
- Evaluate the solutions of the sub-population
- Estimate the gradients for the parameters of the search distribution

Except for the interprocess communication of distribution parameters and gradients, all operations therefore are done on the GPUs with the help of the actors.
The fact that most operations are done on GPUs also means that the data transfer between CPU and GPUs are minimal.

Therefore, one is encouraged to try the distributed mode when dealing with a [SupervisedNE][evotorch.neuroevolution.supervisedne.SupervisedNE] instance (or a similar multi-GPU problem).

## Efficient reinforcement learning

Some reinforcement learning problems are subject to constraints, and when an agent breaks those constraints while running, its episode is terminated immediately (before reaching the actual desired length of an episode).
Among popular constrained reinforcement learning problems are locomotion tasks such as MuJoCo humanoid (e.g. Humanoid-v4) where the constraints dictate that the humanoid body must be on its feet and standing tall. When the humanoid body falls down, the constraint is violated and the episode is terminated (or truncated).

With constrained reinforcement learning problems, the total number of simulator interactions can be used as an interesting indication:

- If the total number of simulator interactions of the population is small, then we might be in a bad region of the search space, because the agents represented by the population were unsuccessful and therefore they broke the constraints early, leading to truncated episodes and diminished number of simulator interactions
- If the total number of simulator interactions of the population is large, then we might be in a good region of the search space, because the agents represented by the population were successful and their episodes were longer, learing to higher number of simulator interactions

This becomes an opportunity to adapt the population size according to how promising our current region in the search space is.
In more detail, with adaptive population size, we keep sampling and evaluating more solutions until a certain threshold of interactions is reached.
This way, in bad regions, more solutions are sampled and we get more reliable gradients for escaping such bad regions.

An EvoTorch algorithm can be instantiated in adaptive population size mode with the help of these two keyword arguments:

- `num_interactions`: Number of simulator interactions threshold. More solutions will be sampled until this threshold is reached. Each re-sampling will be of the base population size given via `popsize`.
- `popsize_max`: If the current population size reaches this point, re-sampling will be stopped.

For example, the following configuration enables adaptive population size for solving Humanoid-v4:

```python
from evotorch.algorithms import PGPE
from evotorch.neuroevolution import GymNE
from evotorch.logging import StdOutLogger


# Declare the problem to solve
problem = GymNE(
    env_name="Humanoid-v4",  # Solve the Humanoid-v4 task
    network="Linear(obs_length, act_length)",  # Linear policy
    observation_normalization=True,  # Normalize the policy inputs
    decrease_rewards_by=5.0,  # Decrease each reward by 5.0
    num_actors="max",  # Use all available CPUs
)


# Instantiate a PGPE algorithm to solve the problem
searcher = PGPE(
    problem,
    # Base population size
    popsize=200,
    # For each generation, sample more solutions until the
    # number of simulator interactions reaches the given
    # threshold.
    # In this example, we compute the threshold by multiplying
    # the base populatio size (200) by the episode length (1000)
    # and then taking 75% of the value.
    # This means, more or less, if 75% of the population can
    # reach the ends of their episodes, we do not need to
    # re-sample. Otherwise, we re-sample.
    num_interactions=int(200 * 1000 * 0.75),
    # Stop re-sampling solutions if the current population size
    # reaches or exceeds this number.
    popsize_max=3200,
    # Learning rates
    center_learning_rate=0.0075,
    stdev_learning_rate=0.1,
    # Radius of the initial search distribution
    radius_init=0.27,
    # Use the ClipUp optimizer with the specified maximum speed
    optimizer="clipup",
    optimizer_config={"max_speed": 0.15},
)
```

Note that adaptive population size mode and the distributed mode (via `distributed=True`) can be enabled at the same time. When both modes are on, each actor will have its share of `num_interactions` and `popsize_max` thresholds, and will keep re-sampling its own local solutions.
One might want to enable both of these modes when encountering a challenging constrained reinforcement learning problems requiring large population sizes (and large interaction thresholds).
