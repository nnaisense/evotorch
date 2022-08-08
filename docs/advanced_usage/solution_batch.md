# Manipulating Solutions

For many advanced use-cases in EvoTorch, it's essential to directly work with populations of solutions. In EvoTorch, solutions are represented through the [Solution][evotorch.core.Solution] class, and groups (populations) of solutions are represented through the [SolutionBatch][evotorch.core.SolutionBatch] class. This tutorial will demonstrate a variety of features of these classes. To begin, consider the simple sphere problem:

```python
from evotorch import Problem, Solution, SolutionBatch
import torch

def sphere(x: torch.Tensor) -> torch.Tensor:
    return torch.sum(x.pow(2.))

problem = Problem(
    'min',
    sphere,
    solution_length = 2,
    initial_bounds = (-1, 1),
)
```

## Creating and using SolutionBatch instances

A [SolutionBatch][evotorch.core.SolutionBatch] instance can be generated directly from the problem itself, with decision values sampled uniformly according to the `initial_bounds` parameter. Calling

```python
batch = problem.generate_batch(5)
```

will create `batch`, an instance of [SolutionBatch][evotorch.core.SolutionBatch] consisting of 5 [Solution][evotorch.core.Solution] instances. These solutions can be viewed in terms of their $5 \times 2$ tensor representation using the `values` attribute:

```python
print(batch.values)
```

???+ abstract "Output"
    ```python
    ReadOnlyTensor(
        tensor([[-0.3779, -0.0910],
                [-0.7356,  0.8931],
                [ 0.0012, -0.1703],
                [ 0.4683,  0.8081],
                [-0.1409, -0.6965]])
    )
    ```

Note that the decision values are in the initial range `(-1, 1)` that was specified when the problem was created. The fitness values of the batch can be accessed with the `evals` attribute, and will initially be filled with `nan` values as the population has not yet been evaluated

```python
print(batch.evals)
```

???+ abstract "Output"
    ```python
    ReadOnlyTensor(
        tensor([[nan],
                [nan],
                [nan],
                [nan],
                [nan]])
    )
    ```

Calling `problem.evaluate(batch)` will fill the fitness values with solutions' computed fitnesses:

```python
problem.evaluate(batch)
print(batch.evals)
```

???+ abstract "Output"
    ```python
    ReadOnlyTensor(
        tensor([[0.1511],
                [1.3387],
                [0.0290],
                [0.8723],
                [0.5049]])
    )
    ```

The decision values can be overidden using the `set_values` function. Setting them with random gaussian values,

```python
batch.set_values(problem.make_gaussian((5, 2)))
```

will update the values

```python
print(batch.values)
```
???+ abstract "Output"
    ```python
    ReadOnlyTensor(
        tensor([[ 0.1206, -2.0179],
                [-0.0508, -0.5009],
                [-1.3668, -0.4877],
                [ 1.0296, -1.1230],
                [-0.0574, -0.4883]])
    )
    ```

and clear the fitness values

```python
print(batch.evals)
```
???+ abstract "Output"
    ```python
    ReadOnlyTensor(
        tensor([[nan],
                [nan],
                [nan],
                [nan],
                [nan]])
    )
    ```

The fitness values can also be set with the `set_evals` method to give updated fitnesses,

```python
batch.set_evals(problem.make_uniform(5))
print(batch.evals)
```

???+ abstract "Output"
    ```python
    ReadOnlyTensor(
        tensor([[0.5828],
                [0.5208],
                [0.7184],
                [0.5949],
                [0.7678]])
    )
    ```

The `values` and `evals` properties yield read-only tensors that cannot be modified. In some advanced use-cases, you can directly access the underlying tensors using the `access_values` and `access_evals` methods. However, care should be taken when using these methods. By default, [SolutionBatch][evotorch.core.SolutionBatch] assumes that a call to `access_values` means that the underlying tensor will be modified, and the fitnesses are therefore cleared and reset to `nan`,

```python
values = batch.access_values()
print(batch.evals)
```

???+ abstract "Output"
    ```python
    ReadOnlyTensor(
        tensor([[nan],
                [nan],
                [nan],
                [nan],
                [nan]])
    )
    ```

However, this behaviour can be disabled with the `keep_evals` flag, e.g.

```python
batch.set_evals(problem.make_uniform(5))
values = batch.access_values(keep_evals = True)
print(batch.evals)
```
???+ abstract "Output"
    ```python
    ReadOnlyTensor(
        tensor([[0.3775],
                [0.2806],
                [0.1781],
                [0.3854],
                [0.4929]])
    )
    ```

If you wish to make an exact copy of a [SolutionBatch][evotorch.core.SolutionBatch] instance, you can do so with

```python
cloned_batch = batch.clone()
```

The new instance `cloned_batch` will have identical `values` and `evals` to `batch`, but will share no memory locations, and can be freely modified without modifying its parent `batch`.

## Slicing and Indexing

[SolutionBatch][evotorch.core.SolutionBatch] instances can be arbitrarily sliced and indexed. For example,

```python
last_3 = batch[2:5]
```

will return a [SolutionBatch][evotorch.core.SolutionBatch] instance `last_3` which is a *view* of `batch` e.g. memory data locations with it. Thus we can see that the decision values are exactly the decision values of the last 3 solutions of `batch`

```python
print(last_3.values)
```
???+ abstract "Output"
    ```python
    ReadOnlyTensor(
        tensor([[-1.3668, -0.4877],
                [ 1.0296, -1.1230],
                [-0.0574, -0.4883]])
    )
    ```

and the evaluations are exactly the evaluations of the last 3 solutions of `batch`

```python
print(last_3.evals)
```
???+ abstract "Output"
    ```python
    ReadOnlyTensor(
        tensor([[0.1781],
                [0.3854],
                [0.4929]])
    )
    ```

A similar behaviour can be achieved with a list of indexes e.g. `subbatch = batch[[0, 1, 3]]` will give a solution batch that is a view of the solutions at indices 0, 1 and 3. When a sub-batch is updated, as can be achieved using `set_values` and `set_evals` or `access_values` and `access_evals` followed by modification,

```python
last_3.set_values(problem.make_gaussian(3, 2))
print(last_3.values)
```

???+ abstract "Output"
    ```python
    ReadOnlyTensor(
        tensor([[ 0.2202,  1.5362],
                [ 0.5641, -0.0477],
                [ 0.8012, -1.3022]])
    )
    ```

the parent solution batch `batch` is also modified.

```python
print(batch.values)
```

???+ abstract "Output"
    ```python
    ReadOnlyTensor(
        tensor([[ 0.1206, -2.0179],
                [-0.0508, -0.5009],
                [ 0.2202,  1.5362],
                [ 0.5641, -0.0477],
                [ 0.8012, -1.3022]])
    )
    ```

[SolutionBatch][evotorch.core.SolutionBatch] instances of the same [Problem][evotorch.core.Problem] class can also be arbitrarily concatenated. For example, if we now create a second [SolutionBatch][evotorch.core.SolutionBatch] with 3 instances,

```python
second_batch = problem.generate_batch(3)
print(second_batch.values)
```

???+ abstract "Output"
    ```python
    ReadOnlyTensor(
        tensor([[ 0.0433, -0.6526],
                [ 0.9731,  0.2553],
                [ 0.7482, -0.6220]])
    )
    ```

and then concatenate `last_3` with `second_batch`, using either the `SolutionBatch.cat` or `batch.concat` methods,

```python
new_batch = SolutionBatch.cat([last_3, second_batch])
print(new_batch.values)
```

???+ abstract "Output"
    ```python
    ReadOnlyTensor(
        tensor([[ 0.2202,  1.5362],
                [ 0.5641, -0.0477],
                [ 0.8012, -1.3022],
                [ 0.0433, -0.6526],
                [ 0.9731,  0.2553],
                [ 0.7482, -0.6220]])
    )
    ```

Creating a new instance of [SolutionBatch][evotorch.core.SolutionBatch], `new_batch`, in this way is copying, so further modification of `new_batch` will not modify the parent batches `batch` and `second_batch`.

## Accessing and Manipulating individual Solutions

An individual [Solution][evotorch.core.Solution] instance can be accessed by indexing into the batch. For example, indexing the first [Solution][evotorch.core.Solution] will yield the solution at index 1 of `batch`

```python
solution1 = batch[1]
print(solution1)
```

???+ abstract "Output"
    ```python
    <Solution values=tensor([-0.0508, -0.5009]), evals=tensor([0.2806])>
    ```

A [Solution][evotorch.core.Solution] instance has the same properties `values` and `evals` and methods `set_values`, `set_evals`, `access_values` and `access_evals` as [SolutionBatch][evotorch.core.SolutionBatch], except that they now work in one dimension, rather than two. For example, accessing the values yields a 1-dimensional read-only tensor of decision values

```python
print(solution1.values)
```

???+ abstract "Output"
    ```python
    ReadOnlyTensor(tensor([-0.0508, -0.5009]))
    ```

A [Solution][evotorch.core.Solution] instance is a view of its parent [SolutionBatch][evotorch.core.SolutionBatch], so modifying it will not only modify the solution,

```python
solution1.set_values(torch.tensor([1, 2]))
solution1.set_evals(0.5)
print(solution1)
```

???+ abstract "Output"
    ```python
    <Solution values=tensor([1., 2.]), evals=tensor([0.5000])>
    ```

but will also modify `batch`,

```python
print(batch.values)
```

???+ abstract "Output"
    ```python
    ReadOnlyTensor(
        tensor([[ 0.1206, -2.0179],
                [ 1.0000,  2.0000],
                [ 0.2202,  1.5362],
                [ 0.5641, -0.0477],
                [ 0.8012, -1.3022]])
    )
    ```

A [Solution][evotorch.core.Solution] instance can also be cloned with the `clone()` method e.g.


```python
cloned_sol = solution1.clone()
```

The cloned solution is no longer a view of `batch` and can be freely modified. To obtain the new, single-solution, [SolutionBatch][evotorch.core.SolutionBatch] instance of the cloned solution, you can use `cloned_sol.to_batch()`.
