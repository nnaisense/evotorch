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

from numbers import Number
from typing import Callable, Iterable, Optional, Union

import numpy as np
import torch

from .tools import Device

try:
    from torch.func import vmap
except ImportError:
    from functorch import vmap


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
    def f(x: torch.Tensor) -> torch.Tensor: ...


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
    def f(x: torch.Tensor) -> torch.Tensor: ...


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
    def f(x: torch.Tensor) -> torch.Tensor: ...


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


def expects_ndim(  # noqa: C901
    *expected_ndims,
    allow_smaller_ndim: bool = False,
    randomness: str = "error",
) -> Callable:
    """
    Decorator to declare the number of dimensions for each positional argument.

    Let us imagine that we have a function `f(a, b)`, where `a` and `b` are
    PyTorch tensors. Let us also imagine that the function `f` is implemented
    in such a way that `a` is assumed to be a 2-dimensional tensor, and `b`
    is assumed to be a 1-dimensional tensor. In this case, the function `f`
    can be decorated as follows:

    ```python
    from evotorch.decorators import expects_ndim


    @expects_ndim(2, 1)
    def f(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor: ...
    ```

    Once decorated like this, the function `f` will gain the following
    additional behaviors:

    - If less-than-expected number of dimensions are provided either for
      `a` or for `b`, an error will be raised (unless the decorator
      is provided with the keyword argument `allow_smaller_ndim=True`)
    - If either `a` or `b` are given as tensors that have extra leftmost
      dimensions, those dimensions will be assumed as batch dimensions,
      and therefore, the function `f` will run in a vectorized manner
      (with the help of `vmap` behind the scene), and the result will be
      a tensor with extra leftmost dimension(s), representing a batch
      of resulting tensors.
    - For convenience, numpy arrays and scalar data that are subclasses
      of `numbers.Number` will be converted to PyTorch tensors first, and
      then will be processed.

    To be able to take advantage of this decorator, please ensure that the
    decorated function is a `vmap`-friendly function. Please also ensure
    that the decorated function expects positional arguments only.

    **Randomness.**
    Like in `torch.func.vmap`, the behavior of the decorated function in
    terms of randomness can be configured via a keyword argument named
    `randomness`:

    ```python
    @expects_ndim(2, 1, randomness="error")
    def f(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor: ...
    ```

    If `randomness` is set as "error", then, when there is batching, any
    attempt to generate random data using PyTorch will raise an error.
    If `randomness` is set as "different", then, a random generation
    operation such as `torch.randn(...)` will produce a `BatchedTensor`,
    where each batch item has its own re-sampled data.
    If `randomness` is set as "same", then, a random generation operation
    such as `torch.randn(...)` will produce a non-batched tensor containing
    random data that is sampled only once.

    **Alternative usage.**
    `expects_ndim` has an alternative interface that allows one to use it
    as a tool for temporarily wrapping/transforming other functions. Let us
    consider again our example function `f`. Instead of using the decorator
    syntax, one can do:

    ```python
    result = expects_ndim(f, (2, 1))(a, b)
    ```

    which will temporarily wrap the function `f` with the additional behaviors
    mentioned above, and immediately call it with the arguments `a` and `b`.
    """

    if (len(expected_ndims) == 2) and isinstance(expected_ndims[0], Callable) and isinstance(expected_ndims[1], tuple):
        func_to_wrap, expected_ndims = expected_ndims
        return expects_ndim(*expected_ndims, allow_smaller_ndim=allow_smaller_ndim, randomness=randomness)(func_to_wrap)

    expected_ndims = tuple(
        (None if expected_arg_ndim is None else int(expected_arg_ndim)) for expected_arg_ndim in expected_ndims
    )

    def expects_ndim_decorator(fn: Callable):
        def expects_ndim_decorated(*args):
            # The inner class below is responsible for accumulating the dtype and device info of the tensors
            # encountered across the arguments received by the decorated function.
            # Such dtype and device information will be used if one of the considered arguments is given as a native
            # scalar object (i.e. float), when converting that native scalar object to a PyTorch tensor.
            class tensor_info:
                # At first, we initialize the set of encountered dtype and device info as None.
                # They will be lazily filled if we ever need such information.
                encountered_dtypes: Optional[set] = None
                encountered_devices: Optional[set] = None

                @classmethod
                def update(cls):
                    # Collect and fill the dtype and device information if it is not filled yet.
                    if (cls.encountered_dtypes is None) or (cls.encountered_devices is None):
                        cls.encountered_dtypes = set()
                        cls.encountered_devices = set()
                        for expected_arg_ndim, arg in zip(expected_ndims, args):
                            if (expected_arg_ndims is not None) and isinstance(arg, torch.Tensor):
                                # If the argument has a declared expected ndim, and also if it is a PyTorch tensor,
                                # then we add its dtype and device information to the sets `encountered_dtypes` and
                                # `encountered_devices`.
                                cls.encountered_dtypes.add(arg.dtype)
                                cls.encountered_devices.add(arg.device)

                @classmethod
                def _get_unique_dtype(cls, error_msg: str) -> torch.dtype:
                    # Ensure that there is only one `dtype` and return it.
                    # If there is not exactly one dtype, then raise an error.
                    if len(cls.encountered_dtypes) == 1:
                        [dtype] = cls.encountered_dtypes
                        return dtype
                    else:
                        raise TypeError(error_msg)

                @classmethod
                def _get_unique_device(cls, error_msg: str) -> torch.device:
                    # Ensure that there is only one `device` and return it.
                    # If there is not exactly one device, then raise an error.
                    if len(cls.encountered_devices) == 1:
                        [device] = cls.encountered_devices
                        return device
                    else:
                        raise TypeError(error_msg)

                @classmethod
                def convert_scalar_to_tensor(cls, scalar: Number) -> torch.Tensor:
                    # This class method aims to convert a scalar to a PyTorch tensor.
                    # The dtype and device of the tensor counterpart of the scalar will be taken from the dtype and
                    # device information of the other tensors encountered so far.

                    # First, we update the dtype and device information that can be collected from the arguments.
                    cls.update()

                    # Get the device used by the tensor arguments.
                    device = cls._get_unique_device(
                        f"The function decorated with `expects_ndim` received the scalar argument {scalar}."
                        f" However, this scalar argument cannot be converted to a PyTorch tensor, because it is not"
                        " clear to which device should this scalar be moved."
                        " This might happen when none of the other considered arguments is a tensor,"
                        " or when there are multiple tensor arguments with conflicting devices."
                        f" Devices encountered across all the considered arguments are: {cls.encountered_devices}."
                        " To make this error go away, please consider making sure that other tensor arguments have a"
                        " consistent device, or passing this scalar as a PyTorch tensor so that no conversion is"
                        " needed."
                    )

                    if isinstance(scalar, (bool, np.bool_)):
                        # If the given scalar argument is a boolean, we declare the dtype of its tensor counterpart as
                        # torch.bool.
                        dtype = torch.bool
                    else:
                        # If the given scalar argument is not a boolean, we declare the dtype of its tensor counterpart
                        # as the dtype that is observed across the other arguments.
                        dtype = cls._get_unique_dtype(
                            f" The function decorated with `expects_ndim` received the scalar argument {scalar}."
                            " However, this scalar argument cannot be converted to a PyTorch tensor, because it is not"
                            " clear by which dtype should this scalar be represented in its tensor form."
                            " This might happen when none of the other considered arguments is a tensor,"
                            " or when there are multiple tensor arguments with different dtypes."
                            f" dtypes encountered across all the considered arguments are {cls.encountered_dtypes}."
                            " To make this error go away, please consider making sure that other tensor arguments have"
                            " a consistent dtype, or passing this scalar as a PyTorch tensor so that no conversion is"
                            " needed."
                        )

                    # Finally, using our new dtype and new device, we convert the scalar to a tensor.
                    return torch.as_tensor(scalar, dtype=dtype, device=device)

            # First, we want to make sure that each positional argument is a PyTorch tensor.
            # So, we initialize `new_args` as an empty list, which will be filled with the tensor counterparts
            # of the original positional arguments.
            new_args = []

            for i_arg, (expected_arg_ndims, arg) in enumerate(zip(expected_ndims, args)):
                if (expected_arg_ndims is None) or isinstance(arg, torch.Tensor):
                    # In this case, either the expected number of dimensions is given as None (indicating that the user
                    # does not wish any batching nor any conversion for this argument), or the argument is already
                    # a PyTorch tensor (so, no conversion to tensor needs to be done).
                    # We do not have to do anything in this case.
                    pass
                elif isinstance(arg, (Number, np.bool_)):
                    # If the argument is a scalar `Number`, we convert it to a PyTorch tensor, the dtype and the device
                    # of it being determined with the help of the inner class `tensor_info`.
                    arg = tensor_info.convert_scalar_to_tensor(arg)
                elif isinstance(arg, np.ndarray):
                    # If the argument is a numpy array, we convert it to a PyTorch tensor.
                    arg = torch.as_tensor(arg)
                else:
                    # This is the case where an object of an unrecognized type is received. We do not know how to
                    # process this argument, and, naively trying to convert it to a PyTorch tensor could fail, or
                    # could generate an unexpected result. So, we raise an error.
                    raise TypeError(f"Received an argument of unexpected type: {arg} (of type {type(arg)})")

                if (expected_arg_ndims is not None) and (arg.ndim < expected_arg_ndims) and (not allow_smaller_ndim):
                    # This is the case where the currently analyzed positional argument has less-than-expected number
                    # of dimensions, and we are not in the allow-smaller-ndim mode. So, we raise an error.
                    raise ValueError(
                        f"The argument with index {i_arg} has the shape {arg.shape}, having {arg.ndim} dimensions."
                        f" However, it was expected as a tensor with {expected_arg_ndims} dimensions."
                    )

                # At this point, we know that `arg` is a proper PyTorch tensor. So, we add it into `new_args`.
                new_args.append(arg)

            wrapped_fn = fn
            num_args = len(new_args)
            wrapped_ndims = [
                (None if expected_arg_ndim is None else arg.ndim)
                for expected_arg_ndim, arg in zip(expected_ndims, new_args)
            ]

            # The following loop will run until we know that no `vmap` is necessary.
            while True:
                # Within each iteration, at first, we assume that `vmap` is not necessary, and therefore, for each
                # positional argument, the batching dimension is `None` (which means no argument will be batched).
                needs_vmap = False
                in_dims = [None for _ in new_args]

                for i_arg in range(num_args):
                    # For each positional argument with index `i_arg`, we check whether or not there are extra leftmost
                    # dimensions.

                    if (wrapped_ndims[i_arg] is not None) and (wrapped_ndims[i_arg] > expected_ndims[i_arg]):
                        # This is the case where the number of dimensions associated with this positional argument is
                        # greater than its expected number of dimensions.

                        # We take note that there is at least one positional argument which requires `vmap`.
                        needs_vmap = True

                        # We declare that this argument's batching dimension is 0 (i.e. its leftmost dimension).
                        in_dims[i_arg] = 0

                        # Now that we marked the leftmost dimension of this argument as the batching dimension, we
                        # should not consider this dimension in the next iteration of this `while` loop. So, we
                        # decrease its number of not-yet-handled dimensions by 1.
                        wrapped_ndims[i_arg] -= 1

                if needs_vmap:
                    # This is the case where there was at least one positional argument that needs `vmap`.
                    # Therefore, we wrap the function via `vmap`.
                    # Note that, after this `vmap` wrapping, if some of the positional arguments still have extra
                    # leftmost dimensions, another level of `vmap`-wrapping will be done by the next iteration of this
                    # `while` loop.
                    wrapped_fn = vmap(wrapped_fn, in_dims=tuple(in_dims), randomness=randomness)
                else:
                    # This is the case where no positional argument with extra leftmost dimension was found.
                    # Either the positional arguments were non-batched to begin with, or the `vmap`-wrapping of the
                    # previous iterations of this `while` loop were sufficient. Therefore, we are now ready to quit
                    # this loop.
                    break

            # Run the `vmap`-wrapped counterpart of the function and return its result
            return wrapped_fn(*new_args)

        return expects_ndim_decorated

    return expects_ndim_decorator


def rowwise(*args, randomness: str = "error") -> Callable:
    """
    Decorate a vector-expecting function to make it support batch dimensions.

    To be able to decorate a function via `@rowwise`, the following conditions
    are required to be satisfied:
    (i) the function expects a single positional argument, which is a PyTorch
    tensor;
    (ii) the function is implemented with the assumption that the tensor it
    receives is a vector (i.e. is 1-dimensional).

    Let us consider the example below:

    ```python
    @rowwise
    def f(x: torch.Tensor) -> torch.Tensor:
        return torch.sum(x**2)
    ```

    Notice how the implementation of the function `f` assumes that its argument
    `x` is 1-dimensional, and based on that assumption, omits the `dim`
    keyword argument when calling `torch.sum(...)`.

    Upon receiving a 1-dimensional tensor, this decorated function `f` will
    perform its operations on the vector `x`, like how it would work without
    the decorator `@rowwise`.
    Upon receiving a 2-dimensional tensor, this decorated function `f` will
    perform its operations on each row of `x`.
    Upon receiving a tensor with 3 or more dimensions, this decorated function
    `f` will interpret its input as a batch of matrices, and perform its
    operations on each matrix within the batch.

    **Defining fitness functions for Problem objects.**
    The decorator `@rowwise` can be used for defining a fitness function for a
    [Problem][evotorch.core.Problem] object. The advantage of doing so is to be
    able to implement the fitness function with the simple assumption that the
    input is a vector (that stores decision values for a single solution),
    and the output is a scalar (that represents the fitness of the solution).
    The decorator `@rowwise` also flags the decorated function (like
    `@vectorized` does), so, the fitness function is used correctly by the
    `Problem` instance, in a vectorized manner. See the example below:

    ```python
    @rowwise
    def fitness(decision_values: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(torch.sum(decision_values**2))


    my_problem = Problem("min", fitness, ...)
    ```

    In the example above, thanks to the decorator `@rowwise`, `my_problem` will
    use `fitness` in a vectorized manner when evaluating a `SolutionBatch`,
    even though `fitness` is defined in terms of a single solution.

    **Randomness.**
    Like in `torch.func.vmap`, the behavior of the decorated function in
    terms of randomness can be configured via a keyword argument named
    `randomness`:

    ```python
    @rowwise(randomness="error")
    def f(x: torch.Tensor) -> torch.Tensor: ...
    ```

    If `randomness` is set as "error", then, when there is batching, any
    attempt to generate random data using PyTorch will raise an error.
    If `randomness` is set as "different", then, a random generation
    operation such as `torch.randn(...)` will produce a `BatchedTensor`,
    where each batch item has its own re-sampled data.
    If `randomness` is set as "same", then, a random generation operation
    such as `torch.randn(...)` will produce a non-batched tensor containing
    random data that is sampled only once.
    """
    num_args = len(args)

    if num_args == 0:
        immediately_decorate = False
    elif num_args == 1:
        immediately_decorate = True
    else:
        raise TypeError("`rowwise` received invalid number of positional arguments")

    def decorator(fn: Callable) -> Callable:  # <- inner decorator
        decorated = expects_ndim(fn, (1,), randomness=randomness)
        decorated.__evotorch_vectorized__ = True
        return decorated

    return decorator(args[0]) if immediately_decorate else decorator
