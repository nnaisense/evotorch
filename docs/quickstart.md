---
hide:
  - navigation
---
# Quickstart

This section gives a brief overview of the EvoTorch API to solve a simple minimisation problem.

There are four main stages to using EvoTorch:

1. Creating a problem to solve.
2. Creating a searcher to optimise the problem.
3. Attaching loggers to the searcher
4. Running the algorithm

Let's start by importing the relevant packages.


```python
from evotorch import Problem
from evotorch.algorithms import SNES
from evotorch.logging import StdOutLogger
import torch
```


## Problem definition

For this simple example, we're going to consider the classic 'sphere' minimisation problem. The objective is to find a $d$-dimensional vector $x*$ that minimises

\[
f(x) = \sum_{i=1}^d x_i^2.
\]

Implementing this in PyTorch we have


```python
def sphere(x: torch.Tensor) -> torch.Tensor:
    return torch.sum(x.pow(2.0))
```

To make this function visible to EvoTorch's algorithms, we simply wrap it up as a [Problem][evotorch.core.Problem] instance. To do this, we will need to specify that we want to minimise (`"min"`) the function, and the `solution_length` is $d$, in this case $d=10$. We will also specify that the initial bounds for solutions is in the range $(-1, 1)$, so that our algorithm knows roughly where to start.


```python
problem = Problem("min", sphere, solution_length=10, initial_bounds=(-1, 1))
```

## Creating a searcher

Now we can search for solutions for the problem we've defined. In this example, we'll use the [Separable Natural Evolution Strategies](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.300.1836&rep=rep1&type=pdf) algorithm with default parameters, and we will specify that the initial standard deviation (scale) of the search distribution is 5 with `stdev_init=5`.


```python
searcher = SNES(problem, stdev_init=5)
```

## Attaching a logger

To keep an eye on what's happening as we run the algorithm, we'll also create a logger. In this case, we'll use the [StdOutLogger][evotorch.logging.StdOutLogger] which will print the status of the evolutionary algorithm to the terminal.


```python
logger = StdOutLogger(searcher)
```

## Running the searcher

Now we can run the algorithm for one iteration by calling the `searcher.step()` method.


```python
searcher.step()
```

???+ abstract "Output"
    ```bash
            iter : 1
        mean_eval : 268.27362060546875
      median_eval : 269.3234558105469
    pop_best_eval : 119.95197296142578
        best_eval : 119.95197296142578
      worst_eval : 473.68804931640625
    ```

Or if we want to, we can run it for as many iterations as we want using the `searcher.run()` method. Let's try running it for 3 iterations.


```python
searcher.run(3)
```

???+ abstract "Output"
    ```bash
            iter : 2
        mean_eval : 244.1479034423828
      median_eval : 223.21856689453125
    pop_best_eval : 128.8501434326172
        best_eval : 119.95197296142578
      worst_eval : 473.68804931640625

            iter : 3
        mean_eval : 276.6123352050781
      median_eval : 207.94456481933594
    pop_best_eval : 88.03515625
        best_eval : 88.03515625
      worst_eval : 688.3544921875

            iter : 4
        mean_eval : 284.18206787109375
      median_eval : 224.14187622070312
    pop_best_eval : 83.8626937866211
        best_eval : 83.8626937866211
      worst_eval : 688.3544921875
    ```

## A complete script

Putting everything together we have:

```python
from evotorch import Problem
from evotorch.algorithms import SNES
from evotorch.logging import StdOutLogger
import torch

# Define a function to minimize
def sphere(x: torch.Tensor) -> torch.Tensor:
    return torch.sum(x.pow(2.0))


# Define a Problem instance wrapping the function
# Solutions have length 10
problem = Problem("min", sphere, solution_length=10, initial_bounds=(-1, 1))

# Instantiate a searcher
searcher = SNES(problem, stdev_init=5)

# Create a logger
logger = StdOutLogger(searcher)

# Evolve!
searcher.run(3)
```

## Next steps

Now that you have completed your first evolutionary learning with EvoTorch, we recommend that you continue to our [User Guide](user_guide/general_usage.md). Alternatively, you can take a look at our [Examples](examples/index.md) if you want to dive into some advanced use-cases.
