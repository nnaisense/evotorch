# Problem Parallelization

Not every fitness function can be so straight-forwardly vectorized or placed on a CUDA-capable device. To provide a significant speedup in this scenario, EvoTorch supports parallelization using [Ray](https://www.ray.io/) out-of-the-box. This page guides you through the various use cases of Ray.

## Using Multiple Actors

To get started using Ray, you simply need to change the `num_actors` argument.

```python
import torch
from evotorch import Problem


def sphere(x: torch.Tensor) -> torch.Tensor:
    return torch.sum(x.pow(2.0))


problem = Problem(
    objective_sense="min",
    objective_func=sphere,
    solution_length=10,
    initial_bounds=(-1, 1),
    num_actors=4,
)
```

by changing `num_actors > 1`, the [Problem][evotorch.core.Problem] instance will automatically spawn Ray actors the first time an evaluation is called. Each actor will be sent a sub-batch of the population, evaluate that sub-batch and return their associated fitness values.

You should note that even if the [Problem][evotorch.core.Problem] instance is vectorized, either through the decorator @[vectorized][evotorch.decorators.vectorized] or through the custom definition of the `_evaluate_batch` function, each Ray actor will continue to evaluate solutions in a vectorized manner, working on sub-batches of the [SolutionBatch][evotorch.core.SolutionBatch] passed to `problem.evaluate` which is automatically split by the main [Problem][evotorch.core.Problem] instance.

By using Ray for parallelization by default, EvoTorch therefore supports deployment of [Problem][evotorch.core.Problem] instances to large clusters, across multiple machines and CPUs. However, by default, EvoTorch will only be able to exploit the resources available on the single machine. For further guidance on setting up a Ray node to use a cluster, visit the library's [official documentation](https://docs.ray.io/en/latest/ray-core/configure.html) and refer to [our own short tutorial](../advanced_usage/ray_cluster.md) on the topic for tips on getting started.

## Using Ray with GPUs

In our guide on [Defining Problems](problems.md), we demonstrated the use of CUDA-capable devices using the `device` argument. However, as Ray communicates between actors on the CPU, it is recommended that when you have `num_actors > 1`, you use the default value `device = 'cpu'` so that the main [Problem][evotorch.core.Problem] instance remains on the CPU, and resultingly, [SolutionBatch][evotorch.core.SolutionBatch] instances created by [SearchAlgorithm][evotorch.algorithms.searchalgorithm.SearchAlgorithm] instances attached to the problem will also be on the CPU.

This does not, however, mean that you cannot still use CUDA-capable devices for evaluation within the individual Ray actors.
When working with a custom [Problem][evotorch.core.Problem] subclass, from within the methods `_evaluate(self, ...)` or `_evaluate_batch(self, ...)`, a GPU device can be obtained (as a `torch.device` instance) via the property `self.aux_device` (where `aux_device` stands for "auxiliary device"). If this custom problem at hand is not parallelized through Ray, the `aux_device` property will return the first visible CUDA-capable device within the main execution environment. If the problem is parallelized through Ray, the `aux_device` property will return the first visible CUDA-capable device within the environment of the actor (therefore, for each actor, it will return the device assigned to that actor). For example, consider this simple custom problem:

```python
def vectorized_sphere(x: torch.Tensor) -> torch.Tensor:
    return torch.sum(x.pow(2.0), dim=-1)


class VecSphere(Problem):
    def _evaluate_batch(self, solutions: SolutionBatch):
        xs = solutions.values.to(self.aux_device)
        fs = vectorized_sphere(xs)
        solutions.set_evals(fs.to(solutions.device))
```

when a [SolutionBatch][evotorch.core.SolutionBatch] is passed to a `VecSphere` instance's `_evaluate_batch` method, its `values` property is moved to the local `aux_device` for evaluation. Once the decision values have been evaluated (using the `vectorized_sphere` function in this case),  then their fitnesses are moved back to the original device of `solutions` and assigned as the fitness evaluations.

To use the GPU within Ray actors, simply use the `num_gpus_per_actor` keyword argument to specify how many GPUs will be allocated to each actor. For example, if you have 2 GPUs, you can do:

```python
problem = VecSphere(
    objective_sense="min",
    solution_length=10,
    initial_bounds=(-1, 1),
    num_actors=4,
    num_gpus_per_actor=0.5,
)
```

so that the 4 Ray actors will each be assigned half of a GPU. Then when a [SolutionBatch][evotorch.core.SolutionBatch] to be evaluated is passed to `problem`, EvoTorch will split it into sub-batches, pass each sub-batch to a Ray actor which will then evaluate the sub-batch. In the case of `VecSphere`, this means that each of the sub-batches will then be moved to each Ray actor's local auxilliary CUDA-capable device, evaluated, and then returned to the main problem instance.

If you do not wish to create custom problem classes for the sake of multi-GPU parallelization, EvoTorch provides a decorator named @[on_aux_device][evotorch.decorators.on_aux_device] to be used on the fitness function. This decorator informs the problem instance that the fitness function is meant to be used on the auxiliary device. With this decorator, the multi-GPU example above can be re-written as follows:

```python
from evotorch.decorators import vectorized, on_aux_device


@vectorized
@on_aux_device
def vectorized_sphere(x: torch.Tensor) -> torch.Tensor:
    return torch.sum(x.pow(2.0), dim=-1)


problem = VecSphere(
    objective_sense="min",
    objective_func=vectorized_sphere,
    solution_length=10,
    initial_bounds=(-1, 1),
    num_actors=4,
    num_gpus_per_actor=0.5,
)
```

!!! info

    Instead of using @[on_aux_device][evotorch.decorators.on_aux_device], you might prefer being more explicit about which device you want to enable for your fitness function. In that case, you could decorate your fitness function with `@on_device(device)` where `device` is either a string (like `"cuda:0"`) or a device object (like `torch.device("cuda:0")`); or with `@on_cuda` to refer to the default cuda device; or with `@on_cuda(i)` where `i` is an integer referring to the cuda device index. See the decorators @[on_device][evotorch.decorators.on_device] and @[on_cuda][evotorch.decorators.on_cuda] for details.

## Common Use-cases

We have provided a number of special values for the arguments `num_actors` and `num_gpus_per_actor` to easily support a number of common use-cases:

### "I do not want Ray-based parallelization"

To avoid creation of Ray actors, you can set `num_actors` as `None` (or as 0, or as 1, which are equivalent).

!!! info

    For optimization problem classes pre-defined in EvoTorch (including the most basic class [Problem][evotorch.core.Problem]), this is the default. Because these mentioned classes do not make assumptions about whether or not parallelization is needed for the problem at hand. Of course, you can obtain parallelization for such a problem by manually setting `num_actors` to an integer, or to one of the special values listed below.

### "I want to use all available CPUs"

To use all available CPUs, use the `num_actors = 'max'` argument. This will automatically detect the number of available CPUs on your machine/Ray cluster and set `num_actors` to that value.

!!! tip

    It is a very common practice to parallelize reinforcement learning tasks across multiple CPUs for shortening the required execution time when running the necessary episodes. Therefore, when using [GymNE][evotorch.neuroevolution.gymne.GymNE], you might want to set `num_actors = 'max'` if the underlying machine is fully dedicated to the experiment.

### "I want to assign the maximum available GPUs, split across my actors"

To use all available GPUs, use the `num_gpus_per_actor = 'max'` argument. This will automatically detect the number of GPUs on your machine/Ray cluster and assign `num_gpus_per_actor` to the total number of GPUs divided by `num_actors`.

### "I want to use all available CPUs and GPUs"

To use all available compute on your machine, set both `num_actors = 'max'` and `num_gpus_per_actor' = 'max'`.

### "I want to use as many CPUs as I have GPUs"

To create an actor per GPU, use the `num_actors = 'num_gpus'` argument. This will automatically detect the number of GPUs on your machine/Ray cluster and assign `num_actors` to that value and `num_gpus_per_actor = 1`.

!!! tip

    When dealing with neuro-evolution (using the generic neuroevolution problem class, [NEProblem][evotorch.neuroevolution.neproblem.NEProblem], or using the supervised learning problem class, [SupervisedNE][evotorch.neuroevolution.supervisedne.SupervisedNE], or using the vectorized reinforcement learning problem class [VecGymNE][evotorch.neuroevolution.vecgymne.VecGymNE]) on dedicated machines with multiple GPUs, it is usually desired to split the workload of evaluating the neural network parameters across multiple GPUs. In those cases, you might want to consider setting `num_actors = "num_gpus"`.

!!! tip

    Sometimes, for some neuro-evolution problems that are small enough, it can be performance-wise more beneficial to run everything on a single GPU (both the evaluations of the networks and the evolutionary algorithm itself). In such cases, you might want to set `device = 'cuda'` (with a device index if desired, e.g., `device = 'cuda:0'`) and ensure that `num_actors` is left as None (or is set as 0 or 1).


### "I want one-to-one mapping between CPUs and GPUs"

To create multiple actors and to configure each actor to allocate one of the GPUs entirely for itself, use the `num_actors = 'num_devices'` argument. This is similar to `num_actors = 'num_gpus'`, however, it is not the same, because the setting `num_actors = 'num_devices'` takes into account both the number of CPUs and of GPUs. In more details, with `num_actors = 'num_devices'`, the following steps are taken automatically:

1. The number of CPUs are counted.
2. The number of GPUs are counted.
3. The minimum value among the number of CPUs and number of GPUs is computed. Let us call this value `n`.
4. `n` actors are created.
5. Each actor is assigned a GPU. This way, it is ensured that each actor gets an entire GPU to itself, while also ensuring that the number of actors do not exceed the number of available CPUs.


###  "I want to freely use GPUs within my Ray actors"

Ray automatically sets/modifies the `CUDA_VISIBLE_DEVICES` environment variable so that each Ray actor can only see its allocated GPUs. You can override this by setting `num_gpus_per_actors = 'all'`, in which case each actor will be able to see every available CUDA-capable device.

!!! info

    With `num_gpus_per_actors` set as `'all'`, since all the GPUs are visible to all actors, the `aux_device` property can only guess which auxiliary device is being targeted. The simple guess made by `aux_device` in this case is `'cuda:I'` where `I` is the index of the actor. This might be an erroneous guess if the number of actors are more than the number of GPUs. Therefore, with `num_gpus_per_actor` set as `'all'` it is recommended that the users do not heavily rely on `aux_device`, and instead introduce their own case-specific rules for sharing/using the GPUs. Alternatively, if the users wish to rely on the property `aux_device`, they might want to consider another option from this list of common use-cases.

A general remark is that, for all these special values for `num_actors` which count the number of CPUs (and GPUs), if it turns out that the number of actors to be created is only 1 (most probably because there is only one CPU provided by the ray cluster), then no actor will be created (and therefore there won't be any GPU assignment to any actor). This is because having only 1 actor would not bring any parallelization benefit, while still bringing the performance overhead of interprocess communication.
