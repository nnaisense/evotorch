<p align="center">
    <a href="https://evotorch.ai" rel="nofollow">
        <img src="https://raw.githubusercontent.com/nnaisense/evotorch/master/docs/assets/evotorch.svg" />
    </a>
</p>

<div align="center">
    <a href="https://www.python.org/" rel="nofollow">
        <img src="https://img.shields.io/pypi/pyversions/evotorch" alt="Python" />
    </a>
    <a href="https://pypi.org/project/evotorch/" rel="nofollow">
        <img src="https://img.shields.io/pypi/v/evotorch" alt="PyPI" />
    </a>
    <a href="https://github.com/nnaisense/evotorch/blob/master/LICENSE" rel="nofollow">
        <img src="https://img.shields.io/pypi/l/evotorch" alt="License" />
    </a>
    <a href="https://docs.evotorch.ai" rel="nofollow">
        <img src="https://github.com/nnaisense/evotorch/actions/workflows/docs.yaml/badge.svg" alt="Build" />
    </a>
    <a href="https://github.com/nnaisense/evotorch/actions/workflows/test.yaml" rel="nofollow">
        <img src="https://github.com/nnaisense/evotorch/actions/workflows/test.yaml/badge.svg" alt="Test" />
    </a>
    <a href="https://github.com/psf/black" rel="nofollow">
        <img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black" />
    </a>
    <a href="https://results.pre-commit.ci/latest/github/nnaisense/evotorch/master" rel="nofollow">
        <img src="https://results.pre-commit.ci/badge/github/nnaisense/evotorch/master.svg" alt="pre-commit.ci status" />
    </a>
</div>

---

Welcome to the EvoTorch project!
EvoTorch is an open source evolutionary computation library developed at [NNAISENSE](https://nnaisense.com), built on top of [PyTorch](https://pytorch.org/).
See the [documentation](https://docs.evotorch.ai) for in-depth guidance about using EvoTorch, and [join us on Slack](https://join.slack.com/t/evotorch/shared_invite/zt-1hcj9prrl-wQBMX4JtaB6WdGKSDjZGXw) for discussions.

Get started by installing EvoTorch:
```
pip install evotorch
```

With EvoTorch, one can solve various optimization problems, regardless of whether they are differentiable (i.e. allow gradient descent). Among the problem types that are solvable with EvoTorch are:
- Black-box optimization problems (continuous or discrete)
- Reinforcement learning tasks
- Supervised learning tasks

Various evolutionary computation algorithms are available in EvoTorch:
- **Distribution-based search algorithms:**
    - **PGPE:** Policy Gradients with Parameter-based Exploration.
    - **XNES:** Exponential Natural Evolution Strategies.
    - **SNES:** Separable Natural Evolution Strategies.
    - **CEM:** Cross Entropy Method.
- **Population-based search algorithms:**
    - **SteadyStateGA:** A fully elitist genetic algorithm implementation. Also supports multiple objectives, in which case it behaves like **NSGA-II**.
    - **CoSyNE:** Cooperative Synapse Neuroevolution.

Since all of these algorithms are implemented in PyTorch, they benefit from use of vectorization and parallelization on GPUs, drastically speeding up optimization when GPUs are available.
Using [Ray](https://github.com/ray-project/ray), EvoTorch scales these algorithms even further by splitting the workload across:
- multiple CPUs
- multiple GPUs
- multiple computers in a Ray cluster

# Examples

Below are some code examples that demonstrate the API of EvoTorch.

## A black-box optimization example

Any objective function defined to work with PyTorch can be used directly with EvoTorch.
A non-vectorized objective function simply receives a solution as a 1-dimensional torch tensor, and returns a fitness as a scalar.
A vectorized objective function receives a batch of solutions as a 2-dimensional torch tensor, and returns a 1-dimensional tensor of fitnesses.
The following example demonstrates how to define and solve the classical Rastrigin problem.

```python
from evotorch import Problem
from evotorch.algorithms import SNES
from evotorch.logging import StdOutLogger, PandasLogger
import math
import matplotlib.pyplot as plt
import torch

# Declare the objective function
def rastrigin(x: torch.Tensor) -> torch.Tensor:
    A = 10
    (_, n) = x.shape
    return A * n + torch.sum((x**2) - A * torch.cos(2 * math.pi * x), 1)


# Declare the problem
problem = Problem(
    "min",
    rastrigin,
    initial_bounds=(-5.12, 5.12),
    solution_length=100,
    vectorized=True,
    # device="cuda:0"  # enable this line if you wish to use GPU
)

# Initialize the SNES algorithm to solve the problem
searcher = SNES(problem, popsize=1000, stdev_init=10.0)

# Initialize a standard output logger, and a pandas logger
_ = StdOutLogger(searcher, interval=10)
pandas_logger = PandasLogger(searcher)

# Run SNES for the specified amount of generations
searcher.run(2000)

# Get the progress of the evolution into a DataFrame with the
# help of the PandasLogger, and then plot the progress.
pandas_frame = pandas_logger.to_dataframe()
pandas_frame["best_eval"].plot()
plt.show()
```

## A reinforcement learning example

The following example demonstrates how to solve reinforcement learning tasks that are available through the gym library.

```python
from evotorch.algorithms import PGPE
from evotorch.logging import StdOutLogger, PicklingLogger
from evotorch.neuroevolution import GymNE

# Declare the problem to solve
problem = GymNE(
    env="Humanoid-v4",  # Solve the Humanoid-v4 task
    network="Linear(obs_length, act_length)",  # Linear policy
    observation_normalization=True,  # Normalize the policy inputs
    decrease_rewards_by=5.0,  # Decrease each reward by 5.0
    num_actors="max",  # Use all available CPUs
    # num_actors=4,    # Explicit setting. Use 4 actors.
)

# Instantiate a PGPE algorithm to solve the problem
searcher = PGPE(
    problem,
    # Base population size
    popsize=200,
    # For each generation, sample more solutions until the
    # number of simulator interactions reaches this threshold
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

# Instantiate a standard output logger
_ = StdOutLogger(searcher)

# Optional: Instantiate a logger to pickle and save the results periodically.
# In this example, among the saved results will be the center of the search
# distribution, since we are using PGPE which is distribution-based.
_ = PicklingLogger(searcher, interval=10)

# Run the algorithm for the specified amount of generations
searcher.run(500)

# Get the center point of the search distribution,
# obtain a policy out of that point, and visualize the
# agent using that policy.
center_solution = searcher.status["center"]
trained_policy = problem.make_net(center_solution)
problem.visualize(trained_policy)
```

More examples can be found [here](examples/).

# Authors

- [Nihat Engin Toklu](https://github.com/engintoklu)
- [Timothy Atkinson](https://github.com/NaturalGradient)
- [Vojtech Micka](https://github.com/Higgcz)
- [Rupesh Kumar Srivastava](https://github.com/flukeskywalker)
