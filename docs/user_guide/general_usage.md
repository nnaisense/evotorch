# General Usage

In EvoTorch, code typically consists of 3 modular components:

- The searcher, a [SearchAlgorithm][evotorch.algorithms.searchalgorithm.SearchAlgorithm] instance which will iteratively generate populations of candidate solutions, evaluate those populations using the fitness function, and then update its own internal variables to attempt to produce higher-quality solutions in the subsequent iterations.
- The problem, a [Problem][evotorch.core.Problem] instance which will receive populations from the searcher and update the populations' individual solutions with computed fitness values. The [Problem][evotorch.core.Problem] class supports various approaches to vectorisation and/or parallelisation using `ray` actor pools, meaning that the problem can be implemented in a highly efficient manner with little effort.
- Any number of loggers, each an instance of [Logger][evotorch.logging.Logger], which observe the status of the searcher and process that status in various useful ways, such as printing the searcher status to the standard output or storing statistics about the evolutionary run in a remote logging system such as [Sacred](https://sacred.readthedocs.io/en/stable/quickstart.html) or [MLFlow](https://mlflow.org/).

The relationship between these three components is outlined below:

![evotorch](../evotorch_architecture.svg)

The general usage of EvoTorch therefore roughly follows the sketch below,

```python
from evotorch import Problem
from evotorch.algorithms import SNES
from evotorch.logging import StdOutLogger, PandasLogger
import torch

# Create a Problem instance to solve
def sphere(x: torch.Tensor) -> torch.Tensor:
    return torch.sum(x.pow(2.))
problem = Problem(
    'min',
    sphere,
    solution_length = 10,
    initial_bounds = (-1, 1),
)

# Create a SearchAlgorithm instance to optimise the Problem instance
searcher = SNES(problem, stdev_init = 5)

# Create loggers as desired
stdout_logger = StdOutLogger(searcher)  # Status printed to the stdout
...
pandas_logger = PandasLogger(searcher)  # Status stored in a Pandas dataframe

# Run the algorithm for as many iterations as desired
searcher.run(10)

# Process the information accumulated by the loggers.
...
progress = pandas_logger.to_dataframe()
progress.mean_eval.plot() # Display a graph of the evolutionary progress by using the pandas data frame
...
```
