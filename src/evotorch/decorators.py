# Copyright 2022 NNAISENSE SA
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Module defining decorators for evotorch."""

from typing import Callable, Iterable, Optional, Union

import torch

from .tools import Device


def _simple_decorator(
    decorator: Union[str, Callable], args: Iterable, decorator_name: Optional[str] = None
) -> Callable:
    """
    Internal helper function for writing decorators.

    This helper function assumes that the decorators themselves do not expect
    arguments.

    Let us imagine that we have the following decorator:

    ```python
    def mydecorator(fn: Callable) -> Callable:
        decorated = ...  # decorate fn somehow
        return decorated
    ```

    This function defined above would work as follows:

    ```
    # Usage (a)

    @mydecorator
    def myfunction(...):
        ...
    ```

    However, in most common cases, the following wouldn't work:

    ```
    # Usage (b)

    @mydecorator()  # <- note the parentheses
    def myfunction(...):
        ...
    ```

    Instead, now imagine that `mydecorator` is written like this:

    ```python
    def mydecorator(*args):  # <- outer decorator
        def decorator(fn: Callable) -> Callable:  # <- inner decorator
            decorated = ...  # decorate fn somehow
            return decorated

        return _simple_decorator(decorator, args)
    ```

    The `_simple_decorator` helper ensures that `mydecorator` can now work
    with or without parentheses. In other words, both usage (a) and usage
    (b) can now work.

    Another feature of `_simple_decorator` is that it makes it easy to write
    decorators whose sole purpose is to patch the decorated function with a new
    attribute (whose value will be True). Please see the following example:

    ```python
    def my_attribute_adding_decorator(*args):
        return _simple_decorator("__some_new_attribute__", args)
    ```

    Now, let us use our new decorator on a function:

    ```
    @my_attribute_adding_decorator
    def myfunction(...):
        ...
    ```

    As a result of this, `myfunction` now has an additional attribute named
    `__some_new_attribute__` whose value is True. This can be checked via:

    ```python
    print(myfunction.__some_new_attribute__)
    ```

    which would produce the output `True`.

    Args:
        decorator: This can be a Callable object, in which case this argument
            represents the actual function that decorates its received argument
            (the inner decorator).
            Alternatively, this can be a string, in which case it will be
            assumed that the desired inner decorator is a function which
            patches its received callable object with a new attribute, the
            name of this new attribute being represented the given string,
            and the value of this new attribute being True.
        args: The positional arguments that were received by the outermost
            function (by the outer decorator).
        decorator_name: Optionally the name of the (outer) decorator function,
            as string. If given, then this name will appear in the produced
            error messages when the number of arguments is unexpected.
    Returns:
        The inner decorator, if the outer decorator received no arguments;
        the decorated function, if the outer decorator received the function
        to decorate.
    """

    if isinstance(decorator, str):
        # If the decorator argument was given as a string, replace it with an inner decorator function
        # which adds the decorated object a new attribute of the specified name.
        attrib_name = decorator

        def decorator(fn: Callable) -> Callable:
            setattr(fn, attrib_name, True)
            return fn

    # The following variable represents the number of positional arguments received by the outer decorator.
    nargs = len(args)

    if nargs == 0:
        # This is the case where the outer decorator function received no positional arguments.
        # Most probably, the user used the outer decoration function like this:
        #
        #     @my_outer_decorator()  # <- note the parentheses
        #     def f(...):
        #         ...
        #
        # Because we do not have the function to be decorated yet, we return the inner decorator itself.
        return decorator
    elif nargs == 1:
        # This is the case where the outer decorated received a single positional argument.
        # We assume that this single positional argument represents the function to be decorated.
        # Most probably, the user used the outer decoration function like this:
        #
        #     @my_outer_decorator
        #     def f(...):
        #         ...
        #
        # Because now we know which function to decorate, we apply the inner decorator on this function,
        # and return the decorated function.
        return decorator(args[0])
    else:
        # This is the case where the outer decorator received an unexpected number of arguments.
        # We raise a TypeError to let the user know.
        subject = "function" if decorator_name is None else f"`{decorator_name}`"
        raise TypeError(f"The decorator {subject} received unexpected positional arguments")


def pass_info(*args) -> Callable:
    """
    Decorates a callable so that the neuroevolution problem class (e.g. GymNE) will
    pass information regarding the task at hand, in the form of keyword arguments.

    This decorator adds a new attribute named `__evotorch_pass_info__` to the
    decorated callable object, sets this new attribute to True, and then returns
    the callable object itself. Upon seeing this attribute with the value `True`,
    a neuroevolution problem class sends extra information as keyword arguments.

    For example, in the case of [GymNE][evotorch.neuroevolution.GymNE] or
    [VecGymNE][evotorch.neuroevolution.VecGymNE], the passed information would
    include dimensions of the observation and action spaces.

    Example:
        ```python
        @pass_info
        class MyModule(nn.Module):
            def __init__(self, obs_length: int, act_length: int, **kwargs):
                # Because MyModule is decorated with @pass_info, it receives
                # keyword arguments related to the environment "CartPole-v0",
                # including obs_length and act_length.
                ...


        problem = GymNE(
            "CartPole-v0",
            network=MyModule,
            ...,
        )
        ```

    Args:
        fn_or_class (Callable): Function or class to decorate

    Returns:
        Callable: Decorated function or class
    """
    return _simple_decorator("__evotorch_pass_info__", args, decorator_name="pass_info")


def on_device(device: Device) -> Callable:
    """
    Decorator that informs a problem object that this function wants to
    receive its solutions on the specified device.

    What this decorator does is that it injects a `device` attribute onto
    the decorated callable object. Then, this callable object itself is
    returned. Upon seeing the `device` attribute, the `evaluate(...)` method
    of the [Problem][evotorch.core.Problem] object will attempt to move the
    solutions to that device.

    Let us imagine a fitness function `f` whose definition looks like:

    ```python
    import torch


    def f(x: torch.Tensor) -> torch.Tensor:
        return torch.sum(x, dim=-1)
    ```

    In its not-yet-decorated form, the function `f` would be given `x` on the
    default device of the associated problem object. However, if one decorates
    `f` as follows:

    ```python
    from evotorch.decorators import on_device


    @on_device("cuda:0")
    def f(x: torch.Tensor) -> torch.Tensor:
        return torch.sum(x, dim=-1)
    ```

    then the Problem object will first move `x` onto the device cuda:0, and
    then will call `f`.

    This decorator is useful on multi-GPU settings. For details, please see
    the following example:

    ```python
    from evotorch import Problem
    from evotorch.decorators import on_device


    @on_device("cuda")
    def f(x: torch.Tensor) -> torch.Tensor:
        ...


    problem = Problem(
        "min",
        f,
        num_actors=4,
        num_gpus_per_actor=1,
        device="cpu",
    )
    ```

    In the example code above, we assume that there are 4 GPUs available.
    The main device of the problem is "cpu", which means the populations
    will be generated on the cpu. When evaluating a population, the population
    will be split into 4 subbatches (because we have 4 actors), and each
    subbatch will be sent to an actor. Thanks to the decorator `@on_device`,
    the [Problem][evotorch.core.Problem] instance on each actor will first move
    its [SolutionBatch][evotorch.core.SolutionBatch] to the cuda device visible
    to its actor, and then the fitness function `f` will perform its evaluation
    operations on that [SolutionBatch][evotorch.core.SolutionBatch] on the
    the visible cuda. In summary, the actors will use their associated cuda
    devices to evaluate the fitnesses of the solutions in parallel.

    This decorator can also be used to decorate the method `_evaluate` or
    `_evaluate_batch` belonging to a custom subclass of
    [Problem][evotorch.core.Problem]. Please see the example below:

    ```python
    from evotorch import Problem


    class MyCustomProblem(Problem):
        def __init__(self):
            super().__init__(
                ...,
                device="cpu",  # populations will be created on the cpu
                ...,
            )

        @on_device("cuda")  # fitness evaluations will happen on cuda
        def _evaluate_batch(self, solutions: SolutionBatch):
            fitnesses = ...
            solutions.set_evals(fitnesses)
    ```

    The attribute `device` that is added by this decorator can be used to
    query the fitness device, and also to modify/update it:

    ```python
    @on_device("cpu")
    def f(x: torch.Tensor) -> torch.Tensor:
        ...


    print(f.device)  # Prints: torch.device("cpu")
    f.device = "cuda:0"  # Evaluations will be done on cuda:0 from now on
    ```

    Args:
        device: The device on which the decorated fitness function will work.
    """

    # Take the `torch.device` counterpart of `device`
    device = torch.device(device)

    def decorator(fn: Callable) -> Callable:
        setattr(fn, "__evotorch_on_device__", True)
        setattr(fn, "device", device)
        return fn

    return decorator


def on_cuda(*args) -> Callable:
    """
    Decorator that informs a problem object that this function wants to
    receive its solutions on a cuda device (optionally of the specified
    cuda index).

    Decorating a fitness function like this:

    ```
    @on_cuda
    def f(...):
        ...
    ```

    is equivalent to:

    ```
    @on_device("cuda")
    def f(...):
        ...
    ```

    Decorating a fitness function like this:

    ```
    @on_cuda(0)
    def f(...):
        ...
    ```

    is equivalent to:

    ```
    @on_device("cuda:0")
    def f(...):
        ...
    ```

    Please see the documentation of [on_device][evotorch.decorators.on_device]
    for further details.

    Args:
        args: An optional positional arguments using which one can specify
            the index of the cuda device to use.
    """

    # Get the number of arguments
    nargs = len(args)

    if nargs == 0:
        # If the number of arguments is 0, then we assume that we are in this situation:
        #
        #     @on_cuda()
        #     def f(...):
        #         ...
        #
        # There is no specified index, and we are not yet given which object to decorate.
        # Therefore, we set both of them as None.
        index = None
        fn = None
    elif nargs == 1:
        # The number of arguments is 1. We begin by storing that single argument using a variable named `arg`.
        arg = args[0]

        if isinstance(arg, Callable):
            # If the argument is a callable object, we assume that we are in this situation:
            #
            #     @on_cuda
            #     def f(...):
            #         ...

            # We are not given a cuda index
            index = None

            # We are given our function to decorate. We store that function using a variable named `fn`.
            fn = arg
        else:
            # If the argument is not a callable object, we assume that it is a cuda index, and that we are in the
            # following situation:
            #
            #     @on_cuda(index)
            #     def f(...):
            #         ...

            # We are given a cuda index. After making sure that it is an integer, we store it by a variable named
            # `index`.
            index = int(arg)

            # At this moment, we do not know the function that is being decorated. So, we set `fn` as None.
            fn = None
    else:
        # If the number of arguments is neither 0 nor 1, then this is an unexpected case.
        # We raise an error to inform the user.
        raise TypeError("`on_cuda(...)` received invalid number of arguments")

    # Prepare the device as "cuda"
    device_str = "cuda"

    if index is not None:
        # If a cuda index is given, then we add ":N" (where N is the index) to the end of `device_str`.
        device_str += ":" + str(index)

    # Prepare the decorator function which, upon being called with a function argument, wraps that function.
    decorator = on_device(device_str)

    # If the function that is being decorated is not known yet (i.e. if `fn` is None), then we return the
    # decorator function. If the function is known, then we decorate and return it.
    return decorator if fn is None else decorator(fn)


def on_aux_device(*args) -> Callable:
    """
    Decorator that informs a problem object that this function wants to
    receive its solutions on the auxiliary device of the problem.

    According to its default (non-overriden) implementation, a problem
    object returns `torch.device("cuda")` as its auxiliary device if
    PyTorch's cuda backend is available and if there is a visible cuda
    device. Otherwise, the auxiliary device is returned as
    `torch.device("cpu")`.
    The auxiliary device is meant as a secondary device (in addition
    to the main device reported by the problem object's `device`
    attribute) used mainly for boosting the performance of fitness
    evaluations.
    This decorator, therefore, tells a problem object that the fitness
    function requests to receive its solutions on this secondary device.

    What this decorator does is that it injects a new attribute named
    `__evotorch_on_aux_device__` onto the decorated callable object,
    then sets that new attribute to `True`, and then return the decorated
    callable object itself. Upon seeing this new attribute with the
    value `True`, a [Problem][evotorch.core.Problem] object will attempt
    to move the solutions to its auxiliary device before calling the
    decorated fitness function.

    Let us imagine a fitness function `f` whose definition looks like:

    ```python
    import torch


    def f(x: torch.Tensor) -> torch.Tensor:
        return torch.sum(x, dim=-1)
    ```

    In its not-yet-decorated form, the function `f` would be given `x` on the
    main device of the associated problem object. However, if one decorates
    `f` as follows:

    ```python
    from evotorch.decorators import on_aux_device


    @on_aux_device
    def f(x: torch.Tensor) -> torch.Tensor:
        return torch.sum(x, dim=-1)
    ```

    then the Problem object will first move `x` onto its auxiliary device,
    then will call `f`.

    This decorator is useful on multi-GPU settings. For details, please see
    the following example:

    ```python
    from evotorch import Problem
    from evotorch.decorators import on_aux_device


    @on_aux_device
    def f(x: torch.Tensor) -> torch.Tensor:
        ...


    problem = Problem(
        "min",
        f,
        num_actors=4,
        num_gpus_per_actor=1,
        device="cpu",
    )
    ```

    In the example code above, we assume that there are 4 GPUs available.
    The main device of the problem is "cpu", which means the populations
    will be generated on the cpu. When evaluating a population, the population
    will be split into 4 subbatches (because we have 4 actors), and each
    subbatch will be sent to an actor. Thanks to the decorator `@on_aux_device`,
    the [Problem][evotorch.core.Problem] instance on each actor will first move
    its [SolutionBatch][evotorch.core.SolutionBatch] to its auxiliary device
    visible to the actor, and then the fitness function will perform its
    fitness evaluations on that device. In summary, the actors will use their
    associated auxiliary devices (most commonly "cuda") to evaluate the
    fitnesses of the solutions in parallel.

    This decorator can also be used to decorate the method `_evaluate` or
    `_evaluate_batch` belonging to a custom subclass of
    [Problem][evotorch.core.Problem]. Please see the example below:

    ```python
    from evotorch import Problem


    class MyCustomProblem(Problem):
        def __init__(self):
            super().__init__(
                ...,
                device="cpu",  # populations will be created on the cpu
                ...,
            )

        @on_aux_device("cuda")  # evaluations will be on the auxiliary device
        def _evaluate_batch(self, solutions: SolutionBatch):
            fitnesses = ...
            solutions.set_evals(fitnesses)
    ```
    """
    return _simple_decorator("__evotorch_on_aux_device__", args, decorator_name="on_aux_device")


def vectorized(*args) -> Callable:
    """
    Decorates a fitness function so that the problem object (which can be an instance
    of [evotorch.Problem][evotorch.core.Problem]) will send the fitness function a 2D
    tensor containing all the solutions, instead of a 1D tensor containing a single
    solution.

    What this decorator does is that it adds the decorated fitness function a new
    attribute named `__evotorch_vectorized__`, the value of this new attribute being
    True. Upon seeing this new attribute, the problem object will send this function
    multiple solutions so that vectorized operations on multiple solutions can be
    performed by this fitness function.

    Let us imagine that we have the following fitness function which works on a
    single solution `x`, and returns a single fitness value:

    ```python
    import torch


    def f(x: torch.Tensor) -> torch.Tensor:
        return torch.sum(x**2)
    ```

    ...and let us now define the optimization problem associated with this fitness
    function:

    ```python
    p1 = Problem("min", f, initial_bounds=(-10.0, 10.0), solution_length=5)
    ```

    While the fitness function `f` and the definition `p1` form a valid problem
    description, it does not use PyTorch to its full potential in terms of performance.
    If we were to request the evaluation results on a population of solutions via
    `p1.evaluate(population)`, `p1` would use a classic `for` loop to evaluate every
    single solution within `population` one by one.
    We could greatly increase our performance by:
    (i) re-defining our fitness function in a vectorized manner, i.e. in such a way
    that it will operate on many solutions and compute all of their fitnesses at once;
    (ii) label our fitness function via `@vectorized`, so that the problem object
    will be aware that this new fitness function expects `n` solutions and returns
    `n` fitnesses. The re-designed and labeled fitness function looks like this:

    ```python
    from evotorch.decorators import vectorized


    @vectorized
    def f2(x: torch.Tensor) -> torch.Tensor:
        return torch.sum(x**2, dim=-1)
    ```

    The problem description for `f2` is:

    ```python
    p2 = Problem("min", f2, initial_bounds=(-10.0, 10.0), solution_length=5)
    ```

    In this last example, `p2` will realize that `f2` is decorated via `@vectorized`,
    and will send it `n` solutions, and will receive and process `n` fitnesses.
    """
    return _simple_decorator("__evotorch_vectorized__", args, decorator_name="vectorized")
