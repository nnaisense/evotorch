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

from collections.abc import Iterable, Sequence
from numbers import Number
from typing import Callable, Optional, Union

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


def on_device(device: Device, *, move_only_from_cpu: bool = False) -> Callable:
    """
    Transform a function so that it will compute on the specified device.

    A function decorated via `@on_device` will first move its positional
    arguments to the specified device, then perform the operations listed
    within the body of the original function definition, and then move
    the result back to the most encountered device within its arguments.

    For a function to be decorated via `@on_device`, the assumption is that
    its positional arguments and its output are of these types:

    - Pytorch tensor
    - `ReadOnlyTensor`
    - `TensorFrame`
    - `ObjectArray`
    - shallow (non-nested) sequence or dictionary-like container consisting of
        objects that are instances of the types listed above

    Additionally, a `device` attribute is added onto the decorated counterpart
    of the function. This `device` attribute is not meant for changing, but for
    informing an observer regarding where the computation will take place.

    **Note.**
    Although an `on_device`-decorated function moves its arguments to the
    specified target device for encouraging the computation to take place on
    that device, it is still possible for the inner body of the function to
    move the tensors to any device.

    **Special behavior for evaluation methods of Problem objects.**
    In addition to simple functions, these specific methods of a `Problem`
    class can be decorated via `@on_device`:

    - `_evaluate`
    - `_evaluate_batch`

    If the decorated function receives a Problem object as its first argument,
    and a Solution or a SolutionBatch as its second argument, the decorator
    will assume that the decorated function is one of the methods listed above,
    and will do nothing other than simply passing the arguments to the original
    version of the decorated function. Instead, it is the `Problem` object
    which moves the solutions to the correct device by looking at the `device`
    attribute created by the `@on_device` decorator.

    Decorating arbitrary methods (other than these solution or batch evaluation
    methods of the `Problem` class) is not supported.

    **Example usage 1.**

    Assuming that the cuda device is available:

    ```python
    from evotorch.decorators import on_device


    @on_device("cuda")
    def my_function(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Thanks to the decorator, x and y should be on the 'cuda' device.
        result = x + y

        return result  # the result will be moved back to the most encountered
        # device among the original x and y tensors.
    ```

    **Example usage 2.**

    Assuming that the cuda device is available:

    ```python
    import torch
    from evotorch.decorators import on_device
    from evotorch import Problem, SolutionBatch


    class SphereProblem(Problem):
        def __init__(self):
            super().__init__(
                objective_sense="min",
                solution_length=20,
                initial_bounds=(-1.0, 1.0),
                dtype=torch.float32,
                device="cpu",  # the populations are to be stored on the cpu
            )

        @on_device("cuda")
        def _evaluate_batch(self, batch: SolutionBatch):
            # Upon seeing that this method is decorated by `@on_device`,
            # the `Problem` object will move the `batch` to the cuda device
            # while calling this method.
            # Therefore, the computation below is expected to happen on cuda.
            evals = torch.sum(batch.values**2.0, dim=-1)
            batch.set_evals(evals)
    ```

    Args:
        device: The device to which the arguments will be moved.
        move_only_from_cpu: If this True, only the tensors which are on the
            cpu will be moved to the specified target tensor.
    """

    from .core import Problem, Solution, SolutionBatch
    from .tools._shallow_containers import most_favored_device_among_arguments, move_shallow_container_to_device

    # Make sure that the device is expressed as an instance of `torch.device`
    device = torch.device(device)

    def decorator(original_behavior: Callable) -> Callable:

        def modified_behavior(*args) -> Callable:

            is_evaluation_method = False
            if isinstance(args[0], Problem):
                if (len(args) == 2) and isinstance(args[1], (Solution, SolutionBatch)):
                    is_evaluation_method = True
                else:
                    raise TypeError(
                        " The function decorated by `@on_device` (or `@on_aux_device` or `@on_cuda`) has received"
                        " a Problem object as its first argument."
                        " In this case, it is assumed that the decorated function is an overriden version"
                        " of the method `Problem._evaluate(self, solution: Solution)`"
                        " or `Problem._evaluate_batch(self, batch: SolutionBatch)`."
                        " However, either the number of arguments or the type of the received non-self argument"
                        " is unexpected."
                    )

            if is_evaluation_method:
                # This seems to be an evaluation method (like, e.g. _evaluate_batch).
                # In this case, we assume that the Problem object, while calling this method, already saw the
                # `device` attribute of the decorated function, and did the necessary move operations on the
                # solution batch.
                # So, we just pass the positional arguments to the original function:
                return original_behavior(*args)

            # Get the most favored device among the tensors of the received arguments.
            # This most favored device is the target device for the produced output.
            result_device = most_favored_device_among_arguments(args, slightly_favor_cpu=True)

            # Move each argument to the target device, and apply the wrapped function on the moved data.
            result_value = original_behavior(*[move_shallow_container_to_device(arg, device=device) for arg in args])

            # Move the result back to the most favored device among the input arguments.
            result_value = move_shallow_container_to_device(result_value, device=result_device)

            # Finally, we return the result here.
            return result_value

        if hasattr(original_behavior, "__evotorch_vectorized__"):
            modified_behavior.__evotorch_vectorized__ = original_behavior.__evotorch_vectorized__
        modified_behavior.device = device
        if move_only_from_cpu:
            modified_behavior.__evotorch_move_only_from_cpu__ = True
        return modified_behavior

    return decorator


def on_aux_device(*args) -> Callable:
    """
    Transform a function so that it will compute on the auxiliary device.

    By default, the auxiliary device is cuda if cuda is available, and
    cpu if cuda is not available.

    A function decorated via `@on_aux_device` will first move its positional
    arguments to the auxiliary device if their original device is the cpu,
    then perform the operations listed within the body of the original function
    definition, and then move the result back to the most encountered device
    within its arguments.

    For a function to be decorated via `@on_aux_device`, the assumption is that
    its positional arguments and its output are of these types:

    - Pytorch tensor
    - `ReadOnlyTensor`
    - `TensorFrame`
    - `ObjectArray`
    - shallow (non-nested) sequence or dictionary-like container consisting of
        objects that are instances of the types listed above

    Additionally, a `device` attribute is added onto the decorated counterpart
    of the function. This `device` attribute is not meant for changing, but for
    informing an observer regarding where the computation will take place.
    An attribute `__evotorch_on_aux_device__=True` is also registered to the
    decorated function, to inform to an outside observer that the function is
    decorated via `@on_aux_device`.

    **Note.**
    Although an `on_aux_device`-decorated function moves its cpu-residing
    arguments to the auxiliary device for encouraging the computation to take
    place on that auxiliary device, it is still possible for the inner body of
    the function to move the tensors to any device.

    **Special behavior for evaluation methods of Problem objects.**
    In addition to simple functions, these specific methods of a `Problem`
    class can be decorated via `@on_aux_device`:

    - `_evaluate`
    - `_evaluate_batch`

    If the decorated function receives a Problem object as its first argument,
    and a Solution or a SolutionBatch as its second argument, the decorator
    will assume that the decorated function is one of the methods listed above,
    and will do nothing other than simply passing the arguments to the original
    version of the decorated function. Instead, it is the `Problem` object
    which moves the solutions to its own auxiliary device by looking at its own
    `aux_device` property.

    Decorating arbitrary methods (other than these solution or batch evaluation
    methods of the `Problem` class) is not supported.

    **Example usage 1.**

    ```python
    from evotorch.decorators import on_device


    @on_aux_device
    def my_function(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Thanks to the decorator, x and y will be on the cuda device
        # if cuda is available.
        result = x + y

        return result  # the result will be moved back to the most encountered
        # device among the original x and y tensors.
    ```

    **Example usage 2.**

    ```python
    import torch
    from evotorch.decorators import on_device
    from evotorch import Problem, SolutionBatch


    class SphereProblem(Problem):
        def __init__(self):
            super().__init__(
                objective_sense="min",
                solution_length=20,
                initial_bounds=(-1.0, 1.0),
                dtype=torch.float32,
                device="cpu",  # the populations are to be stored on the cpu
            )

        @on_device("cuda")
        def _evaluate_batch(self, batch: SolutionBatch):
            # Upon seeing that this method is decorated by `@on_aux_device`,
            # the `Problem` object will move the `batch` to the auxiliary
            # device declared by its property named `aux_device`.
            evals = torch.sum(batch.values**2.0, dim=-1)
            batch.set_evals(evals)
    ```
    """

    num_args = len(args)

    if num_args == 0:
        func_to_wrap = None
    elif num_args == 1:
        [func_to_wrap] = args
    else:
        raise TypeError("`on_aux_device` received an unexpected number of positional arguments")

    target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def decorator(fn: Callable) -> Callable:
        fn = on_device(target_device, move_only_from_cpu=True)(fn)
        fn.__evotorch_on_aux_device__ = True
        return fn

    result = decorator
    if func_to_wrap is not None:
        result = result(func_to_wrap)
    return result


def on_cuda(*args) -> Callable:
    """
    Transform a function so that it will compute on the specified cuda device.

    A function decorated via `@on_cuda` will first move its positional
    arguments to the specified cuda device, then perform the operations listed
    within the body of the original function definition, and then move
    the result back to the most encountered device within its arguments.

    For a function to be decorated via `@on_cuda`, the assumption is that
    its positional arguments and its output are of these types:

    - Pytorch tensor
    - `ReadOnlyTensor`
    - `TensorFrame`
    - `ObjectArray`
    - shallow (non-nested) sequence or dictionary-like container consisting of
        objects that are instances of the types listed above

    Additionally, a `device` attribute is added onto the decorated counterpart
    of the function. This `device` attribute is not meant for changing, but for
    informing an observer regarding where the computation will take place.

    **Note.**
    Although an `on_cuda`-decorated function moves its arguments to the
    specified cuda device for encouraging the computation to take place on
    cuda, it is still possible for the inner body of the function to move
    the tensors to any device.

    **Special behavior for evaluation methods of Problem objects.**
    In addition to simple functions, these specific methods of a `Problem`
    class can be decorated via `@on_cuda`:

    - `_evaluate`
    - `_evaluate_batch`

    If the decorated function receives a Problem object as its first argument,
    and a Solution or a SolutionBatch as its second argument, the decorator
    will assume that the decorated function is one of the methods listed above,
    and will do nothing other than simply passing the arguments to the original
    version of the decorated function. Instead, it is the `Problem` object
    which moves the solutions to the correct cuda device by looking at the
    `device` attribute created by the `@on_cuda` decorator.

    Decorating arbitrary methods (other than these solution or batch evaluation
    methods of the `Problem` class) is not supported.

    **Example usage 1.**

    Assuming that the cuda device is available:

    ```python
    from evotorch.decorators import on_device


    @on_cuda  # Note: could also be, e.g., @on_cuda(0) for 'cuda:0'
    def my_function(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Thanks to the decorator, x and y should be on the 'cuda' device.
        result = x + y

        return result  # the result will be moved back to the most encountered
        # device among the original x and y tensors.
    ```

    **Example usage 2.**

    Assuming that the cuda device is available:

    ```python
    import torch
    from evotorch.decorators import on_device
    from evotorch import Problem, SolutionBatch


    class SphereProblem(Problem):
        def __init__(self):
            super().__init__(
                objective_sense="min",
                solution_length=20,
                initial_bounds=(-1.0, 1.0),
                dtype=torch.float32,
                device="cpu",  # the populations are to be stored on the cpu
            )

        @on_cuda  # Note: could also be, e.g., @on_cuda(0) for 'cuda:0'
        def _evaluate_batch(self, batch: SolutionBatch):
            # Upon seeing that this method is decorated by `@on_cuda`,
            # the `Problem` object will move the `batch` to the cuda device
            # while calling this method.
            # Therefore, the computation below is expected to happen on cuda.
            evals = torch.sum(batch.values**2.0, dim=-1)
            batch.set_evals(evals)
    ```
    """
    num_args = len(args)

    if num_args == 0:
        func_to_wrap = None
        target_device = torch.device("cuda")
    elif num_args == 1:
        [first_arg] = args
        if isinstance(first_arg, Callable):
            func_to_wrap = first_arg
            target_device = torch.device("cuda")
        else:
            func_to_wrap = None
            target_device = torch.device("cuda", int(first_arg))
    else:
        raise TypeError("`on_cuda` received an unexpected number of positional arguments")

    decorator = on_device(target_device)

    if func_to_wrap is None:
        return decorator
    else:
        return decorator(func_to_wrap)


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
        if hasattr(fn, "__evotorch_distribute__"):
            raise ValueError(
                "Cannot apply `@expects_ndim` or `@rowwise` on a function"
                " that was previously subjected to `@distribute`"
            )

        def expects_ndim_decorated(*args):
            # The inner class below is responsible for accumulating the dtype and device info of the tensors
            # encountered across the arguments received by the decorated function.
            # Such dtype and device information will be used if one of the considered arguments is given as a native
            # scalar object (i.e. float), when converting that native scalar object to a PyTorch tensor.
            class tensor_info:
                # At first, we initialize the set of encountered dtype and device info as None.
                # They will be lazily filled if we ever need such information.
                encountered_dtypes: set | None = None
                encountered_devices: set | None = None

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


def distribute(
    *arguments,
    num_actors: str | int | None = None,
    chunk_size: int | None = None,
    num_gpus_per_actor: int | float | str | None = None,
    devices: Sequence[bool] | None = None,
) -> Callable:
    """
    Transform a function such that its computations are distributed.

    Let us assume that we have the following function which expects two tensors
    as arguments, and returns a new tensor, with the constraint that the
    leftmost dimension sizes of all these tensors (of its input arguments and
    and of its returned output) are the same:

    ```python
    def update_and_concat(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        assert a.ndim == 2
        assert b.ndim == 2

        # ====
        # Let us imagine some very heavy computation here which modifies a and b
        # such that their values are updated, but their sizes remain the same.
        ...
        # ====

        return torch.hstack([a, b])
    ```

    Let us now imagine that, because of the heavy computation part, we want to
    run this function in a distributed manner, across two cuda devices.
    To achieve this, we can decorate this function as follows:

    ```python
    @distribute(devices=["cuda:0", "cuda:1"])
    def update_and_concat(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor: ...
    ```

    The decorated version of this function, upon being called for the first
    time, will do the following:

    - create two remote actors, one for `cuda:0`, one for `cuda:1`;
    - split the input arguments `a` and `b` into 2 chunks (because of 2 actors)
      along their leftmost dimensions;
    - send the first chunk of arguments to the actor dedicated to `cuda:0`
      and the second chunk of arguments to the actor dedicated to `cuda:1`;
    - initiate parallelized computation across the actors (each actor
      moving its received chunk of arguments to its assigned device);
    - collect the resulting chunks produced by the actors and combine the
      chunks.

    The finally collected and combined result is the output of the decorated
    function.

    The following types are supported for splitting into chunks of arguments
    and for combining to form the final output:
    - `torch.Tensor`
    - `evotorch.tools.ReadOnlyTensor`
    - `evotorch.tools.ObjectArray`
    - `evotorch.tools.TensorFrame`
    - a (non-nested) dictionary-like object (i.e. Mapping) in which the values
      are `Tensor`, `ReadOnlyTensor`, `ObjectArray` or `TensorFrame`
    - a (non-nested) sequence in which the values are `Tensor`,
      `ReadOnlyTensor`, `ObjectArray`, `TensorFrame`

    **Combining with other decorators.**
    A function that was previously decorated via `@expects_ndim` or `@rowwise`
    or `@torch.vmap` can be decorated via `@distribute`. However, the opposite
    is NOT true (e.g. a function that was previously decorated via
    `@distribute` cannot be then decorated via `@expects_ndim`).

    **Inline function transformation.**
    The `distribute` function can also be used in this alternative form if
    decoration is not desired:

    ```python
    distributed_update_and_concat = distribute(
        update_and_concat, devices=["cuda:0", "cuda:1"]
    )
    ```

    **Alternative ways of declaring number of actors.**
    Like in the example above, if we have two cuda devices and we want to
    explicitly target them, we decorate our function like this:

    ```python
    @distribute(devices=["cuda:0", "cuda:1"])
    def update_and_concat(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor: ...
    ```

    The devices do not have to be different. For example, for having 4
    actors which share the available CPUs, one could do:

    ```python
    @distribute(devices=["cpu", "cpu", "cpu", "cpu"])
    def update_and_concat(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor: ...
    ```

    For distributing the computation across 4 GPUs:

    ```python
    @distribute(num_actors=4, num_gpus_per_actor=1)
    def update_and_concat(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor: ...
    ```

    For having two actors, each using the half of a single GPU:

    ```python
    @distribute(num_actors=2, num_gpus_per_actor=0.5)
    def update_and_concat(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor: ...
    ```

    For having an actor for each available GPU:

    ```python
    @distribute(num_actors="num_gpus", num_gpus_per_actor=1)
    def update_and_concat(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor: ...
    ```

    For having `n` actors, where `n` is the minimum between the number of
    CPUs and the number of GPUs:

    ```python
    @distribute(num_actors="num_devices", num_gpus_per_actor=1)
    def update_and_concat(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor: ...
    ```

    For having a CPU-only actor for each available CPU:

    ```python
    @distribute(num_actors="num_cpus")  # or: num_actors="max"
    def update_and_concat(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # Note: setting `num_actors` without setting `num_gpus_per_actor`
        # will cause the actors to be CPU-only (they will not see the GPUs).
        ...
    ```

    **Specifying which argument to split into chunks.**
    Sometimes one has to distribute a function in which some of the arguments
    are not tensors, but are flags to configure the behavior of the function.
    While sending the arguments to remote actors, such flags are usually
    expected to be duplicated, instead of being sent in chunks.
    As an example, please take a look at the function below:

    ```python
    def update_and_combine(
        a: torch.Tensor, b: torch.Tensor, combine_how: str
    ) -> torch.Tensor:
        assert a.ndim == 2
        assert b.ndim == 2

        # ====
        # Let us imagine some very heavy computation here which modifies a and b
        # such that their values are updated, but their sizes remain the same.
        ...
        # ====

        if combine_how == "add":
            return a + b
        elif combine_how == "hstack":
            return torch.hstack([a, b])
        else:
            raise ValueError("Unsupported combine_how value")
    ```

    In the case of this example, we inform `@distribute` that the first two
    positional arguments are to be split into chunks, and the third positional
    argument is to be duplicated:

    ```python
    @distribute(True, True, False, devices=...)
    def update_and_combine(
        a: torch.Tensor, b: torch.Tensor, combine_how: str
    ) -> torch.Tensor: ...
    ```

    Notice how `@distribute` is given booleans as positional arguments.
    The first boolean (True) tells that the first argument of
    `update_and_combine`, `a`, is to be split into chunks.
    The second boolean (True) tells that the second argument of
    `update_and_combine`, `b`, is to be split into chunks.
    The third boolean (False) tells that the third argument of
    `update_and_combine`, `combine_how`, is to be duplicated
    (i.e. to be sent as it is, instead of being split into chunks).

    The non-decorator alternative looks like this:

    ```python
    dist_update_and_combine = distribute(
        update_and_combine, (True, True, False), devices=...
    )
    ```

    **Specifying a chunk size.**
    The `@distribute` decorator has an optional integer argument named
    `chunk_size`. If this is given, then the original arguments will be
    split into chunks with at most this given size.

    Example:

    ```python
    @distribute(devices=["cpu", "cpu"], chunk_size=10)
    def function_to_be_distributed(x: torch.Tensor) -> torch.Tensor: ...


    large_data = ...  # some large tensor here

    # The call below will split `large_data` into chunks.
    # Each chunk is a subtensor of `large_data`, and the leftmost dimension
    # size of each chunk is at most 10.
    # Parallelized processing of these chunks will be scheduled for the two
    # available remote actors.
    result = function_to_be_distributed(large_data)
    ```

    **Distributing across multiple computers.**
    This `@distribute` decorator uses the `ray` library for parallelizing
    the wrapped function. Thanks to this, if the program is placed upon
    a `ray`-powered cluster consisting of multiple computers (and also
    if the main program has addressed and initialized the `ray` cluster using
    `ray.init` before executing this decorator), the computation of the
    wrapped function will be distributed across all the devices that are
    visible to the cluster.

    **NOTE.**
    If a distributed counterpart of a function cannot be created due to its
    distribution configuration (e.g. if one sets `num_actors` as 1 or 0, or if
    one sets `num_actors` as `"num_gpus"` when there is only 1 GPU available),
    an error will be raised.
    """

    from ._distribute import DecoratorForDistributingFunctions

    if (len(arguments) == 1) and isinstance(arguments[0], Callable):
        function_to_decorate = arguments[0]
        split_arguments = None
    elif (len(arguments) == 2) and isinstance(arguments[0], Callable) and isinstance(arguments[1], tuple):
        function_to_decorate = arguments[0]
        split_arguments = arguments[1]
    else:
        function_to_decorate = None
        split_arguments = arguments

    result = DecoratorForDistributingFunctions(
        split_arguments=split_arguments,
        num_actors=num_actors,
        chunk_size=chunk_size,
        num_gpus_per_actor=num_gpus_per_actor,
        devices=devices,
    )

    if function_to_decorate is not None:
        result = result(function_to_decorate)

    return result
