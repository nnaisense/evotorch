# Logging

Loggers allow automatic accumulation, storage and visualisation of the `status` dictionary of a [SearchAlgorithm][evotorch.algorithms.searchalgorithm.SearchAlgorithm] instance.
Creating a logger is as simple as creating an instance of one of the [supported loggers](#supported-loggers) and attaching it to the [SearchAlgorithm][evotorch.algorithms.searchalgorithm.SearchAlgorithm].

```python title="Simple example of StdOutLogger" hl_lines="14 16"
from evotorch import Problem
from evotorch.algorithms import SNES
import torch


def sphere(x: torch.Tensor) -> torch.Tensor:
    return torch.sum(x.pow(2.0))


problem = Problem("min", sphere, solution_length=10, initial_bounds=(-1, 1))
searcher = SNES(problem, stdev_init=5)

# Create the logger, attaching it to the searcher
from evotorch.logging import StdOutLogger

_ = StdOutLogger(searcher)
```

Once the logger is attached to the [SearchAlgorithm][evotorch.algorithms.searchalgorithm.SearchAlgorithm] instance `searcher`, it will log `searcher.status` every time `searcher.step()` is called.

Most loggers also support the argument `interval`. This allows you to control how often the logger is updated. For example we can change the above code to log to the stdout every **10** iterations.

```python
_ = StdOutLogger(searcher, interval=10)
```

## Supported loggers

Here is a list of loggers that are currently supported by EvoTorch.

| Logger                                            | Description                                                                  |
| ------------------------------------------------- | ---------------------------------------------------------------------------- |
| [StdOutLogger](#logging-to-the-stdout)            | Logs to the stdout                                                           |
| [PicklingLogger](#logging-to-a-pickle-file)       | Logs to a local file system as pickle file                                   |
| [PandasLogger](#logging-to-a-pandas-dataframe)    | Logs to a local file system as [Pandas]()https://pandas.pydata.org DataFrame |
| [MlflowLogger](#logging-with-mlflow)              | Logs using [MLFlow](https://mlflow.org)                                      |
| [SacredLogger](#logging-with-sacred)              | Logs using [Sacred](https://github.com/IDSIA/sacred)                         |
| [NeptuneLogger](#logging-with-neptune)            | Logs using [Neptune](https://neptune.ai/)                                    |
| [WandbLogger](#logging-with-weights-biases)       | Logs using [Weights & Biases](https://wandb.ai)                              |

## Logging to the stdout

The [StdOutLogger][evotorch.logging.StdOutLogger] class facilitates the logging of the status directly to the standard output.
For example, calling `searcher.run(3)` with a [StdOutLogger][evotorch.logging.StdOutLogger] attached to `searcher` will print the `status` dictionary of the searcher in each of the 3 calls to `step()`, producing an output such as below:

```python
from evotorch.logging import StdOutLogger

...
_ = StdOutLogger(searcher)
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

## Logging to a Pickle file

Using the logger [PicklingLogger][evotorch.logging.PicklingLogger] you can log the current results of the [SearchAlgorithm][evotorch.algorithms.searchalgorithm.SearchAlgorithm] instance to a local file system as a pickle file.

The pickled data includes the current **center solution** and the **best solution** (if available).

```python
from evotorch.logging import PicklingLogger

...
_ = PicklingLogger(searcher, interval=10)
```

!!! tip

    If the problem being solved is a **reinforcement learning task**, then the pickled data also includes the **observation normalization** data and the **policy**.

## Logging to a Pandas DataFrame

One might want to collect the logs into a local progress report for performing analysis and plotting training/evolution curves.
A [PandasLogger][evotorch.logging.PandasLogger] is provided which collects such logs into a `pandas.DataFrame` during the execution of the evolutionary algorithm.

Given that one has a [SearchAlgorithm][evotorch.algorithms.searchalgorithm.SearchAlgorithm] instance (e.g. SNES, PGPE, etc.) stored by the variable `searcher`, a [PandasLogger][evotorch.logging.PandasLogger] can be instantiated as shown below:

```python
from evotorch.logging import PandasLogger

...

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

??? example "Full Example"

    ```python title="pandas_logging.py"
    from evotorch import Problem
    from evotorch.algorithms import SNES
    from evotorch.logging import StdOutLogger, PandasLogger
    import torch


    # Create a Problem instance to solve
    def sphere(x: torch.Tensor) -> torch.Tensor:
        return torch.sum(x.pow(2.0))


    problem = Problem("min", sphere, solution_length=10, initial_bounds=(-1, 1))
    searcher = SNES(problem, stdev_init=5)

    # Instantiate a standard output logger so that the local screen
    # shows the progress.
    # Instantiating a standard output logger is not mandatory, but
    # one might wish to monitor the progress from the standard output.
    _ = StdOutLogger(searcher)

    # Instantiate a PandasLogger before the generations are executed
    pandas_logger = PandasLogger(searcher)

    # Run the evolutionary process (for 100 generations)
    searcher.run(100)

    # Obtain the data frame
    my_data_frame = pandas_logger.to_dataframe()

    # Display a graph of the evolutionary progress by using the pandas data frame
    my_data_frame["mean_eval"].plot()
    ```

## Logging with MLFlow

If you are using [MLFlow](https://mlflow.org), you can log the results of the evolutionary algorithm to MLFlow using the [MlflowLogger][evotorch.logging.MlflowLogger].

!!! tip

    Thanks to MLFlow you can log the results locally or to a remote server. ([Read more in MLFlow documentation](https://mlflow.org/docs/latest/tracking.html#where-runs-are-recorded))


!!! warning "Additional Requirements"

    The `mlflow` package must be installed for this logger to work.
    ```
    pip install mlflow
    ```

In order to use the [MlflowLogger][evotorch.logging.MlflowLogger], you just need to create a MLFlow Client and Run object:

```python
from evotorch.logging import MlflowLogger

...

# Import mlflow to create the run
import mlflow

# Create the MLFlow client
client = mlflow.tracking.MlflowClient()

# Start an MLFlow run to log to
run = mlflow.start_run()

# Create an MlflowLogger instance
_ = MlflowLogger(searcher, client=client, run=run)
```

??? example "Full Example"

    ```python title="mlflow_logging.py"
    from evotorch import Problem
    from evotorch.algorithms import SNES
    from evotorch.logging import StdOutLogger, MlflowLogger
    import torch


    # Create a Problem instance to solve
    def sphere(x: torch.Tensor) -> torch.Tensor:
        return torch.sum(x.pow(2.0))


    problem = Problem("min", sphere, solution_length=10, initial_bounds=(-1, 1))
    searcher = SNES(problem, stdev_init=5)

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

## Logging with Sacred

If you are using [Sacred](https://sacred.readthedocs.io/en/stable/), you can log the results of the evolutionary algorithm to Sacred using the [SacredLogger][evotorch.logging.SacredLogger].

!!! warning "Additional Requirements"

    The `sacred` package must be installed for this logger to work.
    ```
    pip install sacred
    ```

The basic structure of a script using `sacred` is as follows:

```python
from sacred import Experiment
from evotorch.tools import SuppressSacredExperiment
from evotorch.logging import SacredLogger


if __name__ == "__main__":
    ex = Experiment()
else:
    ex = SuppressSacredExperiment()


@ex.automain
def main():
    ...
    # In addition, instantiate a SacredLogger so that the logs are stored
    # via the sacred library.
    # The SacredLogger is bound to the SearchAlgorithm instance `searcher`
    # and the experiment `ex`.
    # The main result of the experiment is declared as the status item
    # with the key "mean_eval".
    _ = SacredLogger(searcher, ex, "mean_eval")
```


??? example "Full Example"

    ```python title="sacred_logging.py"
    from evotorch import Problem
    from evotorch.algorithms import SNES
    from evotorch.logging import StdOutLogger, SacredLogger
    import torch

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
        problem = Problem("min", sphere, solution_length=10, initial_bounds=(-1, 1))
        searcher = SNES(problem, stdev_init=5)

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

## Logging with Neptune

If you are using [Neptune](https://neptune.ai), you can log the results of the evolutionary algorithm to Neptune using the [NeptuneLogger][evotorch.logging.NeptuneLogger].
Usage of [NeptuneLogger][evotorch.logging.NeptuneLogger] is as simple as creating a Neptune Run object and passing it to the logger class at instantiation.

!!! warning "Additional Requirements"

    The `neptune-client` package must be installed for this logger to work.
    ```
    pip install neptune-client
    ```

```python
from evotorch.logging import NeptuneLogger

logger = NeptuneLogger(searcher, project='workspace-name/project-name')

# User can explicitly log other parameters, metrics etc.
logger.run["..."].log(...)
```

??? example "Full Example"

    ```python title="neptune_logging.py"
    from evotorch import Problem
    from evotorch.algorithms import SNES
    from evotorch.logging import NeptuneLogger

    problem = Problem("min", sphere, solution_length=10, initial_bounds=(-1, 1))
    searcher = SNES(problem, stdev_init=5)

    # Instantiate the Neptune logger
    # You can pass additional parameters that you would normally pass to neptune.init_run(...)
    # via further keyword arguments
    logger = NeptuneLogger(searcher, project='workspace-name/project-name')

    # User can explicitly log other parameters, metrics etc.
    # logger.run["..."].log(...)

    # Run the search algorithm
    searcher.run(100)

    # Stop the neptune run
    run.stop()
    ```

## Logging with Weights & Biases

If you are using [Weights & Biases](https://wandb.ai), you can log the results of the evolutionary algorithm to Weights & Biases using the [WandbLogger][evotorch.logging.WandbLogger].

!!! warning "Additional Requirements"

    The `wandb` package must be installed for this logger to work.
    ```
    pip install wandb
    ```

```python
from evotorch.logging import WandbLogger

...
# Instantiate the W&B logger
# You can also pass any other parameters that are expected by wandb.init(...)
# As an example, we pass the 'project' parameter to set the project name
_ = WandbLogger(searcher, project="project-name")
```

??? example "Full Example"

    ```python title="wandb_logging.py"
    from evotorch import Problem
    from evotorch.algorithms import SNES
    from evotorch.logging import WandbLogger

    problem = Problem("min", sphere, solution_length=10, initial_bounds=(-1, 1))
    searcher = SNES(problem, stdev_init=5)

    # Instantiate the W&B logger
    # You can pass additional parameters that you would normally pass to wandb.init(...)
    # via further keyword arguments
    _ = WandbLogger(searcher, project="project-name")

    # Run the search algorithm
    searcher.run(100)

    # Stop the neptune run
    run.stop()
    ```
