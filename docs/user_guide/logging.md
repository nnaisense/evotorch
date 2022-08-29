# Using Loggers

Loggers allow automatic accumulation, storage and visualisation of the `status` dictionary of a [SearchAlgorithm][evotorch.algorithms.searchalgorithm.SearchAlgorithm] instance. When we create a [Logger][evotorch.logging.Logger] instance, we attach it to a [SearchAlgorithm][evotorch.algorithms.searchalgorithm.SearchAlgorithm] instance, and the internal code will do the rest.

```python
from evotorch import Problem
from evotorch.algorithms import SNES
import torch


def sphere(x: torch.Tensor) -> torch.Tensor:
    return torch.sum(x.pow(2.0))


problem = Problem("min", sphere, solution_length=10, initial_bounds=(-1, 1))
searcher = SNES(problem, stdev_init=5)

# Create the logger, attaching it to the searcher
from evotorch.logging import StdOutLogger

logger = StdOutLogger(searcher)
```

Once a [Logger][evotorch.logging.Logger] instance has been created and attached to a [SearchAlgorithm][evotorch.algorithms.searchalgorithm.SearchAlgorithm] instance `searcher`, it will log `searcher.status` every time `searcher.step()` is called.

Most loggers also support the argument `interval`. This allows you to control how often the [Logger][evotorch.logging.Logger] instance is updated. For example, if we instead do

```python
logger = StdOutLogger(searcher, interval=10)
```

then `logger` will only log to the stdout every $10$ iterations.

## Direct Logging with StdOutLogger

The [StdOutLogger][evotorch.logging.StdOutLogger] class facilitates the logging of the status directly to the standard output. For example, calling `searcher.run(3)` with a [StdOutLogger][evotorch.logging.StdOutLogger] attached to `searcher` will print the `status` dictionary of the searcher in each of the 3 calls to `step()`, producing an output such as below:

```python
logger = StdOutLogger(searcher)
searcher.run(3)
```
???+ abstract "Output"
    ```
                iter : 1
            mean_eval : 166.8264923095703
        median_eval : 134.81417846679688
        pop_best_eval : 74.62313079833984
            best_eval : 74.62313079833984
        worst_eval : 274.99029541015625

                iter : 2
            mean_eval : 241.93557739257812
        median_eval : 201.2202911376953
        pop_best_eval : 125.85945892333984
            best_eval : 74.62313079833984
        worst_eval : 413.692626953125

                iter : 3
            mean_eval : 199.49822998046875
        median_eval : 199.6777801513672
        pop_best_eval : 93.25483703613281
            best_eval : 74.62313079833984
        worst_eval : 413.692626953125
    ```

## Locally Collecting the Logs via Pandas

One might want to collect the logs into a local progress report for performing analysis and plotting training/evolution curves.
A [PandasLogger][evotorch.logging.PandasLogger] is provided which collects such logs into a `pandas.DataFrame` during the execution of the evolutionary algorithm.

Given that one has a [SearchAlgorithm][evotorch.algorithms.searchalgorithm.SearchAlgorithm] instance (e.g. SNES, PGPE, etc.) stored by the variable `searcher`, a [PandasLogger][evotorch.logging.PandasLogger] can be instantiated as shown below:

```python
from evotorch.logging import PandasLogger

# Instantiate a PandasLogger before the generations are executed
pandas_logger = PandasLogger(searcher)

# Run the evolutionary process (for 100 generations)
searcher.run(100)
```

When a local progress report is desired (most probably after the evolutionary algorithm is finished), one can obtain a `pandas.DataFrame` via:

```python
my_data_frame = pandas_logger.to_dataframe()
```

The mean evaluation (represented by that status key "mean_eval") can then be plotted via:

```python
import matplotlib.pyplot as plt

my_data_frame["mean_eval"].plot()
plt.show()
```

## Remote Logging with `mlflow`

There is also a logger class named `MlflowLogger` which logs the metrics via the `mlflow` library.
With the help of `mlflow`, the logs can be stored into the local disk or into a remote server.

In the simplest case, a script which stores its logs with the help of `mlflow` looks like this:

```python
from evotorch import Problem
from evotorch.algorithms import SNES
from evotorch.logginer import StdOutLogger, MlFlowLoger

# Somehow instantiate the problem
problem = Problem(...)

# Somehow instantiate the search algorithm
searcher = SNES(problem, ...)

# Instantiate a standard output logger so that the local screen
# shows the progress.
# Instantiating a standard output logger is not mandatory, but
# one might wish to monitor the progress from the standard output.
_ = StdOutLogger(searcher)

# In addition, instantiate an MlflowLogger so that the logs are stored
# via mlflow.
import mlflow

client = mlflow.tracking.MlflowClient()  # Create the Mlflow client
run = mlflow.start_run()  # Start an mlflow run to log to
_ = MlflowLogger(searcher, client=client, run=run)

# Run the search algorithm
searcher.run(100)
```

## Remote Logging with `sacred`

As an alternative to `mlflow`, one might want to log the metrics with the help of the `sacred` library.
The basic structure of a script using `sacred` is as follows:

```python
from evotorch import Problem
from evotorch.algorithms import SNES
from evotorch.logginer import StdOutLogger, SacredLogger

from sacred import Experiment
from evotorch.tools import SuppressSacredExperiment


# Instantiate an experiment only if this Python file is executed
# as a script.
if __name__ == "__main__":
    ex = Experiment()
else:
    ex = SuppressSacredExperiment()


@ex.automain
def main():
    # Somehow instantiate the problem
    problem = Problem(...)

    # Somehow instantiate the search algorithm
    searcher = SNES(problem, ...)

    # Instantiate a standard output logger so that the local screen
    # shows the progress.
    # Instantiating a standard output logger is not mandatory, but
    # one might wish to monitor the progress from the standard output.
    _ = StdOutLogger(searcher)

    # In addition, instantiate a SacredLogger so that the logs are stored
    # via the sacred library.
    # The SacredLogger is bound to the SearchAlgorithm instance `searcher`
    # and the experiment `ex`.
    # The main result of the experiment is declared as the status item
    # with the key "mean_eval".
    _ = SacredLogger(searcher, ex, "mean_eval")

    # Run the search algorithm
    searcher.run(100)
```
