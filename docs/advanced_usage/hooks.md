# Using Hooks

Hooks are a powerful tool that allow you to inject code into various stages of the evolutionary process. You can add hooks to both [Problem][evotorch.core.Problem] instances and [SearchAlgorithm][evotorch.algorithms.searchalgorithm.SearchAlgorithm] instances. This page will demonstrate how you can use hooks to add additional functionality to your evolutionary pipeline.

## Problem Hooks

### `before_eval_hook`

You can inject code before the evaluation of a [SolutionBatch][evotorch.core.SolutionBatch] using the `before_eval_hook` property. You can append functions to this property that take, as argument, a [SolutionBatch][evotorch.core.SolutionBatch] instance. Let us suppose that you want to catch scenarios where your custom [SearchAlgorithm][evotorch.algorithms.searchalgorithm.SearchAlgorithm] has produced `inf` decision values. Let's define this as a function:

```python
from evotorch import SolutionBatch
import torch


def abort_if_inf(batch: SolutionBatch):
    if torch.any(torch.isinf(batch.values)):
        raise ValueError("SearchAlgorithm created inf values!")
```

Adding this hook to a problem class is straightforward. Consider the sphere problem:

```python
from evotorch import Problem


def sphere(x: torch.Tensor) -> torch.Tensor:
    return torch.sum(x.pow(2.0))


problem = Problem("min", sphere, solution_length=10, initial_bounds=(-1, 1))
```

You can add the `abort_if_inf` function as a hook to execute before a [SolutionBatch][evotorch.core.SolutionBatch] is evaluated using:

```python
problem.before_eval_hook.append(abort_if_inf)
```

Checking that the hook is now in use is as simple as attempting to pass a [SolutionBatch][evotorch.core.SolutionBatch] to the `problem` instance which contains at least 1 `inf` value, which will cause the `ValueError` we designed earlier to be triggered::

```python
dodgy_batch = problem.generate_batch(10)  # Generate a SolutionBatch of size 10
dodgy_batch.access_values()[1, 1] = float("inf")  # Inject a rogue inf value
problem.evaluate(dodgy_batch)  # Pass the SolutionBatch to the problem to evaluate
```

???+ abstract "Output"
    ```bash
    ValueError: SearchAlgorithm created inf values!
    ```

### `after_eval_hook`

Similarly, you can inject code to execute immediately *after* a [SolutionBatch][evotorch.core.SolutionBatch] instance has been evaluated using the `after_eval_hook` property. This hook works exactly as `before_eval_hook`, except that it also allows us to return a `dict` instance which will be added to the `status` of the [Problem][evotorch.core.Problem] and therefore the `status` of a [SearchAlgorithm][evotorch.algorithms.searchalgorithm.SearchAlgorithm] instance using that problem. Let's consider the scenario where we wish to track the memory usage on our hardware every time a [SolutionBatch][evotorch.core.SolutionBatch] instance is called. We can define a function which returns this information as a `dict`:

```python
import psutil


def report_memory_usage(batch: SolutionBatch):
    return {"mem usage": psutil.virtual_memory().percent}
```

and add it to our `problem` just as before:

```python
problem.after_eval_hook.append(report_memory_usage)
```

If we now create a [SearchAlgorithm][evotorch.algorithms.searchalgorithm.SearchAlgorithm] instance and optimize for a few iterations with the [StdOutLogger][evotorch.logging.StdOutLogger] we will observe that `"mem_usage"`, the memory usage of the machine, is now being reported in the `searcher.status` and therefore by the [StdOutLogger][evotorch.logging.StdOutLogger]:

```python
from evotorch.algorithms import SNES
from evotorch.logging import StdOutLogger

searcher = SNES(problem, stdev_init=1.0)
logger = StdOutLogger(searcher)
searcher.run(3)
```

???+ abstract "Output"
    ```bash
            iter : 1
        mean_eval : 12.302145004272461
      median_eval : 12.144983291625977
    pop_best_eval : 6.008919715881348
        best_eval : 4.496272563934326
      worst_eval : 22.681987762451172
        mem usage : 30.1

            iter : 2
        mean_eval : 8.87697696685791
      median_eval : 8.475287437438965
    pop_best_eval : 4.878612041473389
        best_eval : 4.496272563934326
      worst_eval : 22.681987762451172
        mem usage : 30.1

            iter : 3
        mean_eval : 12.097428321838379
      median_eval : 11.44112491607666
    pop_best_eval : 4.2585768699646
        best_eval : 4.2585768699646
      worst_eval : 22.681987762451172
        mem usage : 30.1
    ```

## SearchAlgorithm Hooks

### `before_step_hook`

Much like [Problem][evotorch.core.Problem] allows you to inject hooks before and after evaluation, [SearchAlgorithm][evotorch.algorithms.searchalgorithm.SearchAlgorithm] allows you to inject hooks before and after the call to `_step`. You can inject code before `_step` using the `before_step_hook` property. This function takes no arguments, but we can easily exploit `class` definitions or `global` variables to achieve complex behaviour. Let's suppose that we want to keep track of the variables $\mu$ and $\sigma$, corresponding to `searcher.status['center']` and `searcher.status['stdev']`, respectively, using global variables `mu` and `sigma`:

```python
mu = None
sigma = None


def update_global_distribution_vars():
    global mu, sigma, searcher
    mu = searcher.status["center"].clone()
    sigma = searcher.status["stdev"].clone()
```

If we now add this hook to our `searcher` and run it for some iterations we can observe the global `mu` and `sigma` variables changing every time we step the searcher:

```python
searcher = SNES(problem, stdev_init=1.0)
searcher.before_step_hook.append(update_global_distribution_vars)
for _ in range(3):
    searcher.step()
    print(f"Last mu: {mu}")
    print(f"Last sigma: {sigma}")
```

???+ abstract "Output"
    ```bash
    Last mu: tensor([ 0.5075, -0.0859, -0.4649,  0.6813,  0.1237, -0.5578, -0.8896,  0.2800,
            -0.4438,  0.1856])
    Last sigma: tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
    Last mu: tensor([ 0.5075, -0.0859, -0.4649,  0.6813,  0.1237, -0.5578, -0.8896,  0.2800,
            -0.4438,  0.1856])
    Last sigma: tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
    Last mu: tensor([ 1.0853,  0.3743, -0.2238,  0.4539,  0.3474, -0.2821, -0.1601,  0.4664,
            -1.0065,  0.1527])
    Last sigma: tensor([1.0659, 1.0077, 0.9795, 0.9146, 0.8209, 0.9982, 0.9484, 0.9260, 0.7501,
            1.1041])
    ```

!!! important

    [SNES][evotorch.algorithms.distributed.gaussian.SNES] will not update the search distribution in its first step as, initially, its population contains unevaluated solutions.

### `after_step_hook`

You can also inject code after the call to `_step` using `after_step_hook`. This works identically to `before_step_hook` except that, much like `after_eval_hook` for the [Problem][evotorch.core.Problem] class you may also return a `dict` which will be added to the `searcher`'s `status` dictionary. For example, let's use the `update_global_distribution_vars` as before, but now also add a hook to apply after the step which calculated the euclidean movement in $\mu$:

```python
def euclidean_movement():
    global searcher, mu
    return {"mu_movement": torch.norm(searcher.status["center"] - mu).item()}
```

If we now add this to `after_step_hook`, add a [StdOutLogger][evotorch.logging.StdOutLogger] instance and run for some iterations we will see that the `status` dictionary now tracks the euclidean movement in $\mu$:

```python
logger = StdOutLogger(searcher)
searcher.after_step_hook.append(euclidean_movement)
searcher.run(3)
```

???+ abstract "Output"
    ```bash
            iter : 4
        mean_eval : 12.15910816192627
    pop_best_eval : 5.9899001121521
      median_eval : 11.250519752502441
        best_eval : 0.4812171459197998
      worst_eval : 26.787078857421875
        mem usage : 31.0
      mu_movement : 1.1693975925445557

            iter : 5
        mean_eval : 8.20405387878418
    pop_best_eval : 3.2686986923217773
      median_eval : 6.675857067108154
        best_eval : 0.4812171459197998
      worst_eval : 26.787078857421875
        mem usage : 31.0
      mu_movement : 1.1225664615631104

            iter : 6
        mean_eval : 7.373251914978027
    pop_best_eval : 3.3824539184570312
      median_eval : 6.133779048919678
        best_eval : 0.4812171459197998
      worst_eval : 26.787078857421875
        mem usage : 31.0
      mu_movement : 1.0265130996704102
    ```
