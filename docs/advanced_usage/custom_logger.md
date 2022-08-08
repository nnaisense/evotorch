While we provide a number of logging functionalities out-of-the-box, there are many cases where you may wish to create your own logger. This short tutorial will walk you through the steps to achieve this.

## General Structure

A [Logger][evotorch.logging.Logger] object has quite a simple structure: it requires only an `__init__` method that accepts a [SearchAlgorithm][evotorch.algorithms.searchalgorithm.SearchAlgorithm] instance, and a `_log` method to handle the status that is returned from the [SearchAlgorithm][evotorch.algorithms.searchalgorithm.SearchAlgorithm]:

```python
from evotorch.logging import Logger

class MyLogger(Logger):

    def __init__(self, searcher):
        super().__init__(searcher)

        ...
        # any additional desired initialisation
        ...

    def _log(self, status: dict):
        ...
        # do some work on the status dictionary
        ...
```

From only these two methods, the [Logger][evotorch.logging.Logger] instance will work for any [SearchAlgorithm][evotorch.algorithms.searchalgorithm.SearchAlgorithm] instance, simply by attaching it at instantiation,

```python
my_logger = MyLogger(searcher)
searcher.step()
```

## A Simple Example

Let's suppose that we wish to use `matplotlib` functionality to create a simple chart that tracks the status of our algorithm in real-time. To do this, we first import `matplotlib` using the `'TkAgg'` back-end and in interactive mode.

```python
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
plt.ion()
```

When we initialise our custom logger, `LivePlotter`, we will pass it a `searcher` to attach to, and a named status variable to plot `target_status`.

```python
class LivePlotter(Logger):

    def __init__(self, searcher, target_status: str):

        # Call the super constructor
        super().__init__(searcher)

        # Set up the target status
        self._target_status = target_status

        ...
```

Additionally, we will set up a `matplotlib` `Figure` instance that we can interact with as we receive data from the `searcher`.

```python
        # Create a figure and axis
        self._fig = plt.figure(figsize=(10, 4), dpi=80)
        self._ax = self._fig.add_subplot(111)

        # Set the labels of the x and y axis
        self._ax.set_xlabel('iter')
        self._ax.set_ylabel(target_status)

        # Create a line with (initially) no data in it
        self._line, = self._ax.plot([],[])

        # Update the TkAgg window name to something more interesting
        self._fig.canvas.manager.window.title(f'LivePlotter: {target_status}')

        ...
```

We will also want to keep track of all iterations and status values seen so far by the `logger`.

```python
        self._iter_hist = []
        self._status_hist = []
```

When `_log` gets called from the searcher, we are passed the `status` dictionary. We can use this to update the histories of iterations and status values.

```python
    def _log(self, status: dict):

        # Update the histories of the status
        self._iter_hist.append(status['iter'])
        self._status_hist.append(status[self._target_status])

        ...
```

Then we simply need to update the data in the figure `self._fig`

```python
        # Update the x and y data
        self._line.set_xdata(self._iter_hist)
        self._line.set_ydata(self._status_hist)

        # Rescale the limits of the x and y axis
        self._ax.set_xlim(0.99, status['iter'])
        self._ax.set_ylim(min(self._status_hist) * 0.99, max(self._status_hist) * 1.01)

        # Draw the figure and flush its events
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()

        # Sleeping here will make the updates easier to watch
        time.sleep(0.05)
```

Putting all of this together we have

```python
from evotorch.logging import Logger
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
plt.ion()

class LivePlotter(Logger):

    def __init__(self, searcher, target_status: str):

        # Call the super constructor
        super().__init__(searcher)

        # Set up the target status
        self._target_status = target_status

        # Create a figure and axis
        self._fig = plt.figure(figsize=(10, 4), dpi=80)
        self._ax = self._fig.add_subplot(111)

        # Set the labels of the x and y axis
        self._ax.set_xlabel('iter')
        self._ax.set_ylabel(target_status)

        # Create a line with (initially) no data in it
        self._line, = self._ax.plot([],[])

        # Update the TkAgg window name to something more interesting
        self._fig.canvas.manager.window.title(f'LivePlotter: {target_status}')

        self._iter_hist = []
        self._status_hist = []

    def _log(self, status: dict):

        # Update the histories of the status
        self._iter_hist.append(status['iter'])
        self._status_hist.append(status[self._target_status])

        # Update the x and y data
        self._line.set_xdata(self._iter_hist)
        self._line.set_ydata(self._status_hist)

        # Rescale the limits of the x and y axis
        self._ax.set_xlim(0.99, status['iter'])
        self._ax.set_ylim(min(self._status_hist) * 0.99, max(self._status_hist) * 1.01)

        # Draw the figure and flush its events
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()

        # Sleeping here will make the updates easier to watch
        time.sleep(0.05)
```

If we now create a simple problem and associated `searcher`,

```python
from evotorch import Problem
from evotorch.algorithms import SNES
import torch

def sphere(x: torch.Tensor) -> torch.Tensor:
    return torch.sum(x.pow(2.))

problem = Problem(
    'min',
    sphere,
    solution_length = 10,
    initial_bounds = (-1, 1),
)
searcher = SNES(problem, stdev_init = 5)
```

attaching an instance of our custom logger to plot the `'mean_eval'` status and running the searcher,

```python
mean_eval_logger = LivePlotter(searcher, 'mean_eval')
searcher.run(200)
```

should create a plot that is updated in real-time!

<p align="center">
  <img src="../custom_logger.gif" alt="animated" />
</p>
