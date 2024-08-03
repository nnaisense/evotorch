# Copyright 2024 NNAISENSE SA
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Purely functional implementations of optimization algorithms.

**Reasoning.**
PyTorch has a functional API within its namespace `torch.func`.
In addition to allowing one to choose a pure functional programming style,
`torch.func` enables powerful batched operations via `torch.func.vmap`.

To be able to work with the functional programming style of `torch.func`,
EvoTorch introduces functional implementations of evolutionary search
algorithms and optimizers within the namespace
`evotorch.algorithms.functional`.
These algorithm implementations are compatible with `torch.func.vmap`,
and therefore they can perform batched evolutionary searches
(e.g. they can work on not just a single population, but on batches
of populations). Such batched searches can be helpful in the following
scenarios:

**Scenario 1: Nested optimization.**
The main optimization problem at hand might have internal optimization
problems. Therefore, when the main optimization problem's fitness function is
reached, the internal optimization problem will have to be solved for each
solution of the main problem. In such a scenario, one might want to use a
functional evolutionary search for the inner optimization problem, so that
a batch of populations is formed where each batch item represents a separate
population associated with a separate solution of the main problem.

**Scenario 2: Batched hyperparameter search.**
If the user is interested in using a search algorithm that has a functional
implementation, the user might want to implement a hyperparameter search
in such a way that there is a batch of hyperparameters (instead of just
a single set of hyperparameters), and the search is performed on a
batch of populations. In such a setting, each population within the population
batch is associated with a different hyperparameter set within the
hyperparameter batch.

**Example: cross entropy method.**
Let us assume that we have the following fitness function, whose output we
wish to minimize:

```python
import torch


def f(x: torch.Tensor) -> torch.Tensor:
    assert x.ndim == 2, "Please pass `x` as a 2-dimensional tensor"
    return torch.sum(x**2, dim=-1)
```

Let us initialize our search from a random point:

```python
solution_length = 1000
center_init = torch.randn(solution_length, dtype=torch.float32) * 10
```

Now we can initialize our cross entropy method like this:

```python
from evotorch.algorithms.functional import cem, cem_ask, cem_tell

state = cem(
    #
    # Center point of the initial search distribution:
    center_init=center_init,
    #
    #
    # Standard deviation of the initial search distribution:
    stdev_init=10.0,
    #
    #
    # Top half of the population are to be chosen as parents:
    parenthood_ratio=0.5,
    #
    #
    # We wish to minimize the fitnesses:
    objective_sense="min",
    #
    #
    # A standard deviation item is not allowed to change more than
    # 1% of its original value:
    stdev_max_change=0.01,
)
```

At this point, we have an initial state of our cross entropy method search,
stored by the variable `state`. Now, we can implement a loop and perform
multiple generations of evolutionary search like this:

```python
num_generations = 1000

for generation in range(1, 1 + num_generations):
    # Ask for a new population (of size 1000) from cross entropy method
    solutions = cem_ask(state, popsize=1000)

    # At this point, `solutions` is a regular PyTorch tensor, ready to be
    # passed to the function `f`.
    # `solutions` is a 2-dimensional tensor of shape (N, L) where `N`
    # is the number of solutions, and `L` is the length of a solution.
    # Our example fitness function `f` is implemented in such a way that
    # we can pass our 2-dimensional `solutions` tensor into it directly.
    # We will receive `fitnesses` as a 1-dimensional tensor of length `N`.
    fitnesses = f(solutions)

    # Let us report the mean of fitnesses to see the progress
    print("Generation:", generation, "  Mean of fitnesses:", torch.mean(fitnesses))

    # Now, we inform cross entropy method of the latest state of the search,
    # the latest population, and the latest fitnesses, so that it can give us
    # the next state of the search.
    state = cem_tell(state, solutions, fitnesses)
```

At the end of the evolutionary search (or, actually, at any point), one can
analyze the `state` tuple to get information about the current status of the
search distribution. These state tuples are named tuples, and therefore, the
data they store are labeled.
In the case of cross entropy method, the latest center of the search
distribution can be obtained via:

```python
latest_center = state.center

# Note, in the case of pgpe, this would be:
# latest_center = state.optimizer_state.center
```

**Notes on manipulating the evolutionary search.**
If, at any point of the search, you would like to change a hyperparameter,
you can do so by creating a modified copy of your latest `state` tuple,
and pass it to the ask method of your evolutionary search (which,
in the case of cross entropy method, is `cem_ask`).
Similarly, if you wish to change the center point of the search, you can
pass a modified state tuple containing the new center point to `cem_ask`.

**Notes on batching.**
In regular non-batched cases, functional search algorithms expect the
`center_init` argument as a 1-dimensional tensor. If `center_init` is given
as a tensor with 2 or more dimensions, the extra leftmost dimensions will
be considered as batch dimensions, and therefore the evolutionary search
itself will be batched (which means that the ask method of the search
algorithm will return a batch of populations). Furthermore, certain
hyperparameters can also be given in batches. See the specific
documentation of the functional algorithms to see which hyperparameters
support batching.

When working with batched populations, it is important to make sure that
the fitness function can work with arbitrary amount of dimensions (not just
2 dimensions). One way to implement such fitness functions with the help
of the [rowwise][evotorch.decorators.rowwise] decorator:

```python
from evotorch.decorators import rowwise


@rowwise
def f(x: torch.Tensor) -> torch.Tensor:
    return torch.sum(x**2)
```

When decorated with `@rowwise`, we can implement our function as if the
tensor `x` is a 1-dimensional tensor. If the decorated `f` receives `x`
not as a vector, but as a matrix, then it will do the same operation
on each row of the matrix, in a vectorized manner. If `x` has 3 or
more dimensions, they will be considered as extra batch dimensions,
affecting the shape of the resulting tensor.

**Example: gradient-based search.**
This namespace also provides functional implementations of various gradient
based optimizers. The reasoning behind the existence of these implementations
is two-fold: (i) these optimizers are used by the functional `pgpe`
implementation (for handling the momentum); and (ii) having these optimizers
with a similar API allows user to switch back-and-forth between evolutionary
and gradient-based search for solving the same problem, hopefully without
having to change the code too much.

Let us consider the same fitness function again, in its `@rowwise` form so
that it can work with a single vector or a batch of such vectors:

```python
from evotorch.decorators import rowwise


@rowwise
def f(x: torch.Tensor) -> torch.Tensor:
    return torch.sum(x**2)
```

To solve this optimization problem using the Adam optimizer, one can
do the following:

```python
from evotorch.algorithms.functional import adam, adam_ask, adam_tell
from torch.func import grad

# Prepare an initial search point
solution_length = 1000
center_init = torch.randn(solution_length, dtype=torch.float32) * 10


# Initialize the Adam optimizer
state = adam(
    center_init=center_init,
    center_learning_rate=0.001,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
)


num_iterations = 1000

for iteration in range(1, 1 + num_iterations):
    # Get the current search point of the Adam search
    center = adam_ask(state)

    # Get the gradient.
    # Negative, because we want to minimize f.
    gradient = -(grad(f)(center))

    # Inform the Adam optimizer of the gradient to follow, and get the next
    # state of the search
    state = adam_tell(state, follow_grad=gradient)


# Store the final solution
final_solution = adam_ask(state)
# or, alternatively:
# final_solution = state.center
```

**Solving a stateful Problem object using functional algorithms.**
If you wish to solve a stateful [Problem][evotorch.core.Problem]
using a functional optimization algorithm, you can obtain a callable evaluator
out of that Problem object, and then use it for computing the fitnesses.
See the following example:

```python
from evotorch import Problem, SolutionBatch
from evotorch.algorithms.functional import cem, cem_ask, cem_tell


class MyProblem(Problem):
    def __init__(self): ...

    def _evaluate_batch(self, batch: SolutionBatch):
        # Stateful batch evaluation code goes here
        ...


# Instantiate the problem
problem = MyProblem()

# Make a callable fitness evaluator
fproblem = problem.make_callable_evaluator()

# Make an initial solution
center_init = torch.randn(problem.solution_length, dtype=torch.float32) * 10


# Prepare a cross entropy method search
state = cem(
    center_init=center_init,
    stdev_init=10.0,
    parenthood_ratio=0.5,
    objective_sense="min",
    stdev_max_change=0.01,
)


num_generations = 1000
for generation in range(1, 1 + num_generations):
    # Get a population
    solutions = cem_ask(state, popsize=1000)

    # Call the evaluator to get the fitnesses
    fitnesses = fproblem(solutions)

    # Let us report the mean of fitnesses to see the progress
    print("Generation:", generation, "  Mean of fitnesses:", torch.mean(fitnesses))

    # Now, we inform cross entropy method of the latest state of the search,
    # the latest population, and the latest fitnesses, so that it can give us
    # the next state of the search.
    state = cem_tell(state, solutions, fitnesses)


# Center of the latest search distribution
latest_center = state.center
```
"""

from .funcadam import adam, adam_ask, adam_tell
from .funccem import cem, cem_ask, cem_tell
from .funcclipup import clipup, clipup_ask, clipup_tell
from .funcpgpe import pgpe, pgpe_ask, pgpe_tell
from .funcsgd import sgd, sgd_ask, sgd_tell

__all__ = [
    "adam",
    "adam_ask",
    "adam_tell",
    "cem",
    "cem_ask",
    "cem_tell",
    "clipup",
    "clipup_ask",
    "clipup_tell",
    "pgpe",
    "pgpe_ask",
    "pgpe_tell",
    "sgd",
    "sgd_ask",
    "sgd_tell",
]
