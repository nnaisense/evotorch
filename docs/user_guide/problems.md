# Defining Problems

## Basic Usage

One of the most important components of EvoTorch is the definition of problems. To define problems, we use the [Problem][evotorch.core.Problem] class, which provides various advanced functionality, including vectorisation, GPU usage, [Ray](https://www.ray.io/) parallelisation and variable population sizes, out of the box. The most basic usage of the [Problem][evotorch.core.Problem] class is simply to pass to it a function to minimize or maximize. In the following documentation, we will focus on minimization of the Sphere function

\[
f(x) = \sum_{i=1}^d x_i^2.
\]

which is implemented in PyTorch as,

```python
import torch


def sphere(x: torch.Tensor) -> torch.Tensor:
    return torch.sum(x.pow(2.0))
```

With only this function definition, we can create an EvoTorch [Problem][evotorch.core.Problem] instance and start learning,

```python
from evotorch import Problem
from evotorch.algorithms import SNES
from evotorch.logging import StdOutLogger

problem = Problem("min", sphere, solution_length=10, initial_bounds=(-1, 1))

searcher = SNES(problem, stdev_init=5)
logger = StdOutLogger(searcher)
searcher.run(10)
```

If we instead want to maximize the function, we can instead instantiate the [Problem][evotorch.core.Problem] instance,

```python
problem = Problem("max", sphere, solution_length=10, initial_bounds=(-1, 1))
```

and if we want to make the instantiation explicit, we can use the keyword arguments `objective_sense` and `objective_func`,

```python
problem = Problem(
    objective_sense="min",
    objective_func=sphere,
    solution_length=10,
    initial_bounds=(-1, 1),
)
```

## Vectorising Problems

One of the most straight-forward ways to accelerate evolution is to use *vectorised* problems. In a typical EA implementation, the population is stored as a list of vectors, so that fitnesses can be evaluated with a simple `for` loop

```python
fitnesses = [f(x) for x in population]
```

In EvoTorch, the population is stored as a `torch` tensor of shape $N \times d$ (or, to use the PyTorch notation, of shape `torch.Size([N, d])`) where $N$ is the population size and $d$ is the problem dimensionality. If it is possible to define a fitness function that can evaluate *all* $N$ solutions at once, as is often possible when the fitness function is defined in terms of PyTorch operators, then significant speedups can be achieved by letting the low-level C implementation of PyTorch do much of the work. To demonstrate this, let's vectorise the `sphere` function from earlier:

```python
def vectorised_sphere(xs: torch.Tensor) -> torch.Tensor:
    return torch.sum(xs.pow(2.0), dim=-1)
```

By specifying that we want to sum across the last dimension, we return an $N$ dimensional vector of fitnesses, rather than a single fitness value. Using this new vectorised function is as simple as using the `vectorized` flag in the instantiation of the [Problem][evotorch.core.Problem].

```python
problem = Problem(
    objective_sense="min",
    objective_func=vectorised_sphere,
    vectorized=True,
    solution_length=10,
    initial_bounds=(-1, 1),
)
```

## Creating Custom Problem Classes

While many fitness functions can be expressed as a callable function $f: \mathbb{R}^d \rightarrow \mathbb{R}$ which can be passed to a [Problem][evotorch.core.Problem] instance at instantiation using the `objective_func` keyword-argument, there are also many cases were we wish to create custom [Problem][evotorch.core.Problem] classes which can be stateful and parameterisable. In this case, we can create a new class that inherits from the [Problem][evotorch.core.Problem] class.

To demonstrate this, we will consider the $d$-dimensional Rastrigin problem, where the center of the function is offset by a randomly chosen vector $x'$,


\[
f(x) = Ad + \sum_{i=1}^d z_i^2 - A \cos (2 \pi z_i).
\]

where

\[
z = x - x'
\]

We can create a new class which defines this problem and randomly chooses $x'$ at instantiation. To do this, we only need to define the `__init__` and `_evaluate` methods.

```python
from evotorch import Problem, Solution
import torch
import math


class OffsetRastrigin(Problem):
    def __init__(self, d: int = 25, A: int = 10):

        super().__init__(
            objective_sense="min",
            solution_length=d,
            initial_bounds=(-1, 1),
        )

        # Store the A parameter for evaluation
        self._A = A
        # Generate a random offset with center 0 and standard deviation 1
        self._x_prime = self.make_gaussian(d, center=0.0, stdev=1.0)

    def _evaluate(self, solution: Solution):
        x = solution.values
        z = x - self._x_prime
        f = (self._A * self.solution_length) + torch.sum(
            z.pow(2.0) - self._A * torch.cos(2 * math.pi * z)
        )
        solution.set_evals(f)
```

This [Problem][evotorch.core.Problem] class can be used just like any other, reparameterising it as needed

```python
from evotorch.algorithms import SNES
from evotorch.logging import StdOutLogger

prob = OffsetRastrigin(d=14, A=5)
searcher = SNES(prob, stdev_init=5)
logger = StdOutLogger(searcher)
searcher.run(10)
```

Let's break down what is happening in the `_evaluate` method definition.

1. This method receives an instance of [Solution][evotorch.core.Solution], the data type used to store and manipulate individual members of the population in EvoTorch.
2. The [Solution][evotorch.core.Solution] instance's `values` method is called, which returns the $d$-dimensional vector `x` that represents the solution.
3. This vector `x` is evaluated according to the above formula to give fitness `f`.
4. The [Solution][evotorch.core.Solution] instance's `set_evals` method is called with argument `f`. This stores the fitness value `f` within the [Solution][evotorch.core.Solution] instance so that it can be used by the `searcher` in the next iteration.

For more detail on interacting with [Solution][evotorch.core.Solution] instances, please refer to [the relevant advanced usage guide](../advanced_usage/solution_batch.md).

## Vectorising Custom Problems

Much like the base [Problem][evotorch.core.Problem] class, it is straight-forward to introduce fitness vectorisation when creating a custom [Problem][evotorch.core.Problem] class. To do this, we simply override the `_evaluate_batch` method, rather than the `_evaluate` method.

```python
from evotorch import SolutionBatch


class VecOffsetRastrigin(Problem):
    def __init__(self, d: int = 25, A: int = 10):

        super().__init__(
            objective_sense="min",
            solution_length=d,
            initial_bounds=(-1, 1),
        )

        # Store the A parameter for evaluation
        self._A = A
        # Generate a random offset with center 0 and standard deviation 1
        self._x_prime = self.make_gaussian((1, d), center=0.0, stdev=1.0)

    def _evaluate_batch(self, solutions: SolutionBatch):
        xs = solutions.values
        zs = xs - self._x_prime
        fs = (self._A * self.solution_length) + torch.sum(
            zs.pow(2.0) - self._A * torch.cos(2 * math.pi * zs), dim=-1
        )
        solutions.set_evals(fs)
```

All that has changed is that rather than receiving a [Solution][evotorch.core.Solution] instance, we are now receiving a [SolutionBatch][evotorch.core.SolutionBatch] instance which consists of $N$ solutions. The call to `values` instead yields a $N \times d$ tensor `xs`, and by appropriately rewriting the line that computes `fs` so that the result is a $N$-dimensional vector, we can straightforwardly set the fitness values of the entire batch of solutions with `set_evals`.

## Working with Data Types and Devices

The [Problem][evotorch.core.Problem] class supports different `torch` data types and devices, with the `dtype` and `device` keyword arguments, respectively. For example, we can specify that we wish to use 16-bit floating point values on the first available CUDA-capable device in the initialisation of the class.

One way to accelerate solution evaluation is to use CUDA-capable devices to compute the fitness values. In EvoTorch, this can be done easily using the `device` flag. By default, the device flag is set to `'cpu'`, so that the problem (and any searcher attached to it) will run everything on the CPU. Assuming there is at least one CUDA-capable device available, we can instead use,

```python
problem = Problem(
    objective_sense="min",
    objective_func=vectorised_sphere,
    vectorized=True,
    solution_length=10,
    initial_bounds=(-1, 1),
    dtype=torch.float16,
    device="cuda:0",
)
```

When working with different data types and device, EvoTorch searchers will use those data types and devices in their own computations to ensure that everything is compatible when the [Problem][evotorch.core.Problem] instance is called. In practice, particularly in high dimensions and with large population sizes, using a CUDA-capable device can yield significant speedups. A particularly important example of this is in the case of neuroevolution.

Similarly, we can use different data types and devices within custom [Problem][evotorch.core.Problem] classes:

```python
class OffsetRastrigin16(Problem):
    def __init__(self, d: int = 25, A: int = 10):

        super().__init__(
            objective_sense="min",
            solution_length=d,
            initial_bounds=(-1, 1),
            dtype=torch.float16,
            device="cuda:0",
        )

        ...
```

For a similar reason, when we are creating `torch` tensors that we will use within the evaluation, it is also important to ensure they share the same data type and device. For this reason, [Problem][evotorch.core.Problem] instances support a number of `torch.Tensor` generation methods out-of-the-box. Earlier, we used the method `make_gaussian` to generate a random center $x'$ for the fitness evaluation to use. This method will generate a sample from a Gaussian distribution with the shape, center and standard deviation specified, but of particular relevance, it will ensure that the generated sample uses the device and data type associated with the [Problem][evotorch.core.Problem] instance. There are a number of similar methods available, which can be found detailed in the [API reference][evotorch].
