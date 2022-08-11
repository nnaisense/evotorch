# Using Algorithms

## Basic Interface

Once a [SearchAlgorithm][evotorch.algorithms.searchalgorithm.SearchAlgorithm] instance has been created, e.g.

```python
from evotorch import Problem
from evotorch.algorithms import SNES
import torch


def sphere(x: torch.Tensor) -> torch.Tensor:
    return torch.sum(x.pow(2.0))


problem = Problem("min", sphere, solution_length=10, initial_bounds=(-1, 1))
searcher = SNES(problem, stdev_init=5)
```

the main usage is to step the algorithm forward by a single generation

```python
searcher.step()
```

However, a common use-case is to run a [SearchAlgorithm][evotorch.algorithms.searchalgorithm.SearchAlgorithm] instance for many generations. In this case, the `run` function provides an easy interface for this,

```python
searcher.run(num_generations=100)
```

!!! important

    The above call to `run` will not produce any output as the `searcher` has no attached loggers.

## Accessing the Status

Each [SearchAlgorithm][evotorch.algorithms.searchalgorithm.SearchAlgorithm] instance maintains a `status` dictionary which tracks various information about the current status of the `searcher`. You can discover the available status information for a specific class that inherits [SearchAlgorithm][evotorch.algorithms.searchalgorithm.SearchAlgorithm] using,

```python
print([k for k in searcher.iter_status_keys()])
```

Each status property can be accessed by its name

```python
best_discovered_solution = searcher.status["best"]
```

All algorithms currently implemented in EvoTorch applied to single-objective problems will *at least* have the following status properties:

- `'best'`, the best discovered solution so far.
- `'worst'`, the worst discovered solution so far.
- `'best_eval'`, the fitness of the best discovered solution so far.
- `'worst_eval'`, the fitness of the worst discovered solution so far.
- `'pop_best'`, the best solution in the population.
- `'pop_best_eval'`, the fitness of the best solution in the population.
- `'mean_eval'`, the mean fitness of the population.
- `'median_eval'`, the best solution in the population.

## Changing Data-types and Devices

In EvoTorch, problems can be specified to use a specific `torch` device and data type. When a [SearchAlgorithm][evotorch.algorithms.searchalgorithm.SearchAlgorithm] instance is created, it is passed a [Problem][evotorch.core.Problem] instance, and in doing so, inherits the device and data type from the [Problem][evotorch.core.Problem].

This is easy to observe, as running the following code,

```python
prob = Problem(
    "min",
    sphere,
    solution_length=10,
    initial_bounds=(-1, 1),
    dtype=torch.float16,
    device="cuda:0",
)
searcher = SNES(prob, stdev_init=5)
searcher.run(10)
```

and then printing the center of the search distribution using the `status` dictionary, will show that the resulting tensor is now using the `dtype` and `device` that was specified in the instantiation of `prob`

```python
print(searcher.status["center"])
```

???+ abstract "Output"
    ```bash
    ReadOnlyTensor(
        tensor([-0.5435,  1.5527, -2.0664,  1.2012,  0.2749, -1.7686, -0.4634,  5.5039,
                -1.4092, -0.7285, -0.3555, -1.5322, -0.9805, -1.7363, -4.3633, -3.1953,
                -3.3008,  0.5483,  0.3359,  0.0964,  2.3184, -2.7031,  1.4873,  0.2109,
                1.4775], device='cuda:0', dtype=torch.float16)
    )
    ```
