# Problem Parallelization

Not every fitness function can be so straight-forwardly vectorised or placed on a CUDA-capable device. To provide a significant speedup in this scenario, EvoTorch supports parallelisation using [Ray](https://www.ray.io/) out-of-the-box. This page guides you through the various use cases of Ray.

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

You should note that even if the [Problem][evotorch.core.Problem] instance is vectorized, either through the `vectorized = True` boolean flag or through the custom definition of the `_evaluate_batch` function, each Ray actor will continue to evaluate solutions in a vectorized manner, working on sub-batches of the [SolutionBatch][evotorch.core.SolutionBatch] passed to `problem.evaluate` which is automatically split by the main [Problem][evotorch.core.Problem] instance.

By using Ray for parallelisation by default, EvoTorch therefore supports deployment of [Problem][evotorch.core.Problem] instances to large clusters, across multiple machines and CPUs. However, by default, EvoTorch will only be able to exploit the resources available on the single machine. For further guidance on setting up a Ray node to use a cluster, visit the library's [official documentation](https://docs.ray.io/en/latest/ray-core/configure.html) and refer to [our own short tutorial](../advanced_usage/ray_cluster.md) on the topic for tips on getting started.

## Using Ray with GPUs

In our guide on [Defining Problems](problems.md), we demonstrated the use of CUDA-capable devices using the `device` argument. However, as Ray communicates between actors on the CPU, it is recommended that when you have `num_actors > 1`, you use the default value `device = 'cpu'` so that the main [Problem][evotorch.core.Problem] instance remains on the CPU, and resultingly, [SolutionBatch][evotorch.core.SolutionBatch] instances created by [SearchAlgorithm][evotorch.algorithms.searchalgorithm.SearchAlgorithm] instances attached to the problem will also be on the CPU.

This does not, however, mean that you cannot still use CUDA-capable devices for evaluation within the individual Ray actors. To use the GPU, simply use the `num_gpus_per_actor` keyword argument to specify how many GPUs will be allocated to each actor. For example, if you have 2 GPUs, you can do:

```python
problem = Problem(
    objective_sense="min",
    objective_func=sphere,
    solution_length=10,
    initial_bounds=(-1, 1),
    num_actors=4,
    num_gpus_per_actor=0.5,
)
```

so that the 4 Ray actors will each be assigned half of a GPU. Then when a [SolutionBatch][evotorch.core.SolutionBatch] to be evaluated is passed to `problem`, EvoTorch will split it into sub-batches, pass each sub-batch to a Ray actor which will move it to the corresponding GPU memory and evaluate the sub-batch on the GPU. The computed fitness values will be moved back to CPU and returned to `problem` which will then assign the fitnesses of the full [SolutionBatch][evotorch.core.SolutionBatch].

There might be certain problems in which, although the main device is CPU (because the `device` keyword argument was left as None or was explicitly set as "cpu"), you might wish to speed-up certain parts of the fitness evaluation by transferring some of the computation to a GPU.
When working with a custom [Problem][evotorch.core.Problem] subclass, from within the methods `_evaluate(self, ...)` or `_evaluate_batch(self, ...)`, such a GPU device can be obtained (as a `torch.device` instance) via the property `self.aux_device` (where `aux_device` stands for "auxiliary device"). If this custom problem at hand is not parallelised, the `aux_device` property will return the first visible CUDA-capable device within the main execution environment. If the problem is parallelised, the `aux_device` property will return the first visible CUDA-capable device within the environment of the actor (therefore, for each actor, it will return the device assigned to that actor).

## Common Use-cases

We have provided a number of special values for the arguments `num_actors` and `num_gpus_per_actor` to easily support a number of common use-cases:

### "I do not want Ray-based parallelization"

To avoid creation of Ray actors, you can set `num_actors` as `None` (or as 0, or as 1, which are equivalent).

!!! info

    For regular optimization problems expressed by the class [Problem][evotorch.core.Problem], this is the default. Because the [Problem][evotorch.core.Problem] class does not make assumptions about whether or not parallelization is needed for the regular problem at hand. Of course, you can obtain parallelization for such a problem by manually setting `num_actors` to an integer, or to one of the special values listed below.

### "I want to use all available CPUs"

To use all available CPUs, use the `num_actors = 'max'` argument. This will automatically detect the number of available CPUs on your machine/Ray cluster and set `num_actors` to that value.

!!! info

    It is a very common practice to parallelize reinforcement learning tasks across multiple CPUs for shortening the required execution time when running the necessary episodes. Therefore, the option `num_actors = 'max'` is the default for [GymNE][evotorch.neuroevolution.gymne.GymNE] (the problem class which expresses tasks of solving classical CPU-bound gym environments).

### "I want to assign the maximum available GPUs, split across my actors"

To use all available GPUs, use the `num_gpus_per_actor = 'max'` argument. This will automatically detect the number of GPUs on your machine/Ray cluster and assign `num_gpus_per_actor` to the total number of GPUs divided by `num_actors`.

### "I want to use all available CPUs and GPUs"

To use all available compute on your machine, set both `num_actors = 'max'` and `num_gpus_per_actor' = 'max'`.

### "I want to use as many CPUs as I have GPUs"

To create an actor per GPU, use the `num_actors = 'num_gpus'` argument. This will automatically detect the number of GPUs on your machine/Ray cluster and assign `num_actors` to that value and `num_gpus_per_actor = 1`.

### "I want one-to-one mapping between CPUs and GPUs"

To create multiple actors and to configure each actor to allocate one of the GPUs entirely for itself, use the `num_actors = 'num_devices'` argument. This is similar to `num_actors = 'num_gpus'`, however, it is not the same, because the setting `num_actors = 'num_devices'` takes into account both the number of CPUs and of GPUs. In more details, with `num_actors = 'num_devices'`, the following steps are taken automatically:

1. The number of CPUs are counted.
2. The number of GPUs are counted.
3. The minimum value among the number of CPUs and number of GPUs is computed. Let us call this value `n`.
4. `n` actors are created.
5. Each actor is assigned a GPU. This way, it is ensured that each actor gets an entire GPU to itself, while also ensuring that the number of actors do not exceed the number of available CPUs.

!!! info

    When dealing with neuro-evolution, it is usually desired to split the workload of evaluating the neural network parameters across multiple GPUs. For this reason, `num_actors = 'num_devices'` is the default option for [NEProblem][evotorch.neuroevolution.neproblem.NEProblem] (the generic neuro-evolution problem type) and for [SupervisedNE][evotorch.neuroevolution.supervisedne.SupervisedNE] (the supervised learning problem type).

!!! tip

    Sometimes, for some neuro-evolution problems that are small enough, it can be performance-wise more beneficial to run everything on a single GPU (both the evaluations of the networks and the evolutionary algorithm itself). In such cases, just setting `device = 'cuda'` (with a device index if desired, e.g., `device = 'cuda:0'`) should be enough. The setting `num_actors = 'num_devices'` refrains from creating Ray actors if `device` is not left as None and is instead set to a GPU.

###  "I want to freely use GPUs within my Ray actors"

Ray automatically sets/modifies the `CUDA_VISIBLE_DEVICES` environment variable so that each Ray actor can only see its allocated GPUs. You can override this by setting `num_gpus_per_actors = 'all'`, in which case each actor will be able to see every available CUDA-capable device.

!!! info

    With `num_gpus_per_actors` set as `'all'`, since all the GPUs are visible to all actors, the `aux_device` property can only guess which auxiliary device is being targeted. The simple guess made by `aux_device` in this case is `'cuda:I'` where `I` is the index of the actor. This might be an erroneous guess if the number of actors are more than the number of GPUs. Therefore, with `num_gpus_per_actor` set as `'all'` it is recommended that the users do not heavily rely on `aux_device`, and instead introduce their own case-specific rules for sharing/using the GPUs. Alternatively, if the users wish to rely on the property `aux_device`, they might want to consider another option from this list of common use-cases.

A general remark is that, for all these special values for `num_actors` which count the number of CPUs (and GPUs), if it turns out that the number of actors to be created is only 1 (most probably because there is only one CPU provided by the ray cluster), then no actor will be created (and therefore there won't be any GPU assignment to any actor). This is because having only 1 actor would not bring any parallelization benefit, while still bringing the performance overhead of interprocess communication.
