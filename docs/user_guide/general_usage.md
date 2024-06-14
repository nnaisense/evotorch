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
    return torch.sum(x.pow(2.0))


problem = Problem("min", sphere, solution_length=10, initial_bounds=(-1, 1))

# Create a SearchAlgorithm instance to optimise the Problem instance
searcher = SNES(problem, stdev_init=5)

# Create loggers as desired
stdout_logger = StdOutLogger(searcher)  # Status printed to the stdout
...
pandas_logger = PandasLogger(searcher)  # Status stored in a Pandas dataframe

# Run the algorithm for as many iterations as desired
searcher.run(10)

# Process the information accumulated by the loggers.
...
progress = pandas_logger.to_dataframe()
progress.mean_eval.plot()  # Plot the evolutionary progress
...

# We now analyze the current status using the dictionary-like status object.
# The status object allows one to get evaluation results (i.e. fitness, etc.)
# and decision values of the best solution in the population
# (via the key "pop_best"), center of the search distribution
# (via the key "center", in case a distribution-based evolutionary
# algorithm is being used), etc.

for status_key in searcher.iter_status_keys():
    print("===", status_key, "===")
    print(searcher.status[status_key])
    print()
```

## Extracting and analyzing results

We now discuss additional ways of analyzing the results of the evolutionary computation. The following code snippets represent possible continuations to the code example above (or to codes similar to it). Therefore, we will continue to refer to the search algorithm as `searcher`.

If the evolutionary algorithm that was used is distribution-based (such as [SNES][evotorch.algorithms.distributed.gaussian.SNES], [XNES][evotorch.algorithms.distributed.gaussian.XNES], [CEM][evotorch.algorithms.distributed.gaussian.CEM], [PGPE][evotorch.algorithms.distributed.gaussian.PGPE], [CMAES][evotorch.algorithms.distributed.cmaes.CMAES]), the status object includes an item with key `"center"`, representing the center (i.e. mean) of the search distribution as a [ReadOnlyTensor][evotorch.tools.readonlytensor.ReadOnlyTensor]:

```python
center_point_as_tensor = searcher.status["center"]
```

Algorithms such as [Cosyne][evotorch.algorithms.ga.Cosyne], [GeneticAlgorithm][evotorch.algorithms.ga.GeneticAlgorithm], etc. do not have a search distribution, therefore, they do not have a center point. However, the best solution of their last population can be obtained via the status key `"pop_best"`, as a [Solution][evotorch.core.Solution] object:

```python
solution_object = searcher.status["pop_best"]
decision_values_as_tensor = solution_object.values
evals_as_tensor = solution_object.evals  # fitness(es), evaluation data, etc.
```

If the [Problem][evotorch.core.Problem] object was initialized with `store_solution_stats=True` (which is enabled by default when the device of the Problem is "cpu"), the solution with the best fitness ever observed is available via the status key `"best"`:

```python
best_sln = searcher.status["best"]
best_sln_decision_values = best_solution.values
best_sln_evals = best_sln.evals  # fitness(s), evaluation data, etc.
```

Unless the search algorithm was initialized to work across remote actors (via `distributed=True`), the search algorithm keeps its last population accessible via the attribute named `population`.

```python
for solution in searcher.population:
    print("Decision values:", solution.values)
    print("Evaluation:", solution.evals)  # fitnesses, evaluation data, etc.
    print()
```
