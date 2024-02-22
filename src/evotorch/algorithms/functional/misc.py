# Copyright 2024 NNAISENSE SA
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

from typing import Callable, Iterable, NamedTuple, Optional, Union

import torch


class OptimizerFunctions(NamedTuple):
    initialize: Callable
    ask: Callable
    tell: Callable


def get_functional_optimizer(optimizer: Union[str, tuple]) -> tuple:
    """
    Get a tuple of optimizer-related functions, from the given optimizer name.

    For example, if the given string is "adam", the returned tuple will be
    `(adam, adam_ask, adam_tell)`, where
    [adam][evotorch.algorithms.functional.funcadam.adam]
    is the function that will initialize the Adam optimizer,
    [adam_ask][evotorch.algorithms.functional.funcadam.adam_ask]
    is the function that will get the current search point as a tensor, and
    [adam_tell][evotorch.algorithms.functional.funcadam.adam_tell]
    is the function that will expect the gradient and will return the updated
    state of the Adam search after applying the given gradient.

    In addition to "adam", the strings "clipup" and "sgd" are also supported.

    If the given optimizer is a 3-element tuple, then, the three elements
    within the tuple are assumed to be the initialization, ask, and tell
    functions of a custom optimizer, and those functions are returned
    in the same order.

    Args:
        optimizer: The optimizer name as a string, or a 3-element tuple
            representing the functions related to the optimizer.
    Returns:
        A 3-element tuple in the form
        `(optimizer, optimizer_ask, optimizer_tell)`, where each element
        is a function, the first one being responsible for initializing
        the optimizer and returning its first state.
    """
    from .funcadam import adam, adam_ask, adam_tell
    from .funcclipup import clipup, clipup_ask, clipup_tell
    from .funcsgd import sgd, sgd_ask, sgd_tell

    if optimizer == "adam":
        return OptimizerFunctions(initialize=adam, ask=adam_ask, tell=adam_tell)
    elif optimizer == "clipup":
        return OptimizerFunctions(initialize=clipup, ask=clipup_ask, tell=clipup_tell)
    elif optimizer in ("sgd", "sga", "momentum"):
        return OptimizerFunctions(initialize=sgd, ask=sgd_ask, tell=sgd_tell)
    elif isinstance(optimizer, str):
        raise ValueError(f"Unrecognized functional optimizer name: {optimizer}")
    elif isinstance(optimizer, Iterable):
        a, b, c = optimizer
        return OptimizerFunctions(initialize=a, ask=b, tell=c)
    else:
        raise TypeError(
            f"`get_functional_optimizer(...)` received an unrecognized argument: {repr(optimizer)}"
            f" (of type {type(optimizer)})"
        )


def _get_stdev_init(
    *,
    center_init: torch.Tensor,
    stdev_init: Optional[Union[float, Iterable]] = None,
    radius_init: Optional[Union[float, Iterable]] = None,
) -> torch.Tensor:
    """
    Internal helper function for getting the standard deviation vector.

    This utility function is used by the functional implementations of
    the algorithms `cem` and `pgpe`.

    Given a `center_init` tensor and the arguments `stdev_init` and
    `radius_init`, this helper function returns a `stdev_init` tensor ready
    to be used by the functional evolutionary algorithm with a compatible
    dtype, device, and shape.

    Args:
        center_init: The center point for the initial search distribution.
        stdev_init: Standard deviation as None, or as a scalar, or as a vector,
            or as a batch of vectors. If left as None, it will be assumed that
            the user wishes to express the coverage area of the initial search
            distribution via `radius_init` instead.
        radius_init: Radius of the initial search distribution as None, or as a
            scalar, or as a batch of scalars. If left as None, it will be
            assumed that the user wishes to express the coverage area of the
            initial search distribution via `stdev_init` instead.
    Returns:
        Standard deviation for the initial search distribution, as a vector
        in the non-batched case, or as a tensor with 2 or more dimensions
        in the batched case.
    """

    if not isinstance(center_init, torch.Tensor):
        raise TypeError(
            "While computing/validating the initial standard deviation of the functional search algorithm,"
            " the argument `center_init` was encountered as something other than a tensor."
            " The argument `center_init` is expected as a tensor."
        )

    # Get the dtype and the device of the tensor `center_init`. It will be ensured that the standard deviation tensor
    # is also of this dtype and on this device.
    dtype = center_init.dtype
    device = center_init.device

    # The length of a solution is understood from the `center_init` tensor's rightmost dimension's size.
    solution_length = center_init.shape[-1]

    if (stdev_init is None) and (radius_init is None):
        raise ValueError(
            "Both `stdev_init` and `radius_init` are encountered as None."
            " Please provide one of them so that the standard deviation of the initial search distribution is clear."
        )
    elif (stdev_init is not None) and (radius_init is None):
        # This is the case where `stdev_init` is given and `radius_init` is omitted.
        stdev_init = torch.as_tensor(stdev_init, dtype=dtype, device=device)
        if stdev_init.ndim == 0:
            # This is the case where `stdev_init` is given as a scalar. We convert it to a vector (by repeating its
            # original scalar form).
            stdev_init = stdev_init.repeat(solution_length)
        else:
            # This is the case where `stdev_init` is not a scalar. We make sure that its rightmost dimension's size
            # is compatible with the solution length.
            if stdev_init.shape[-1] != solution_length:
                raise ValueError(
                    "The shape of `stdev_init` does not seem compatible with the shape of `center_init`."
                    f" The shape of `center_init` is {center_init.shape},"
                    f" implying a solution length of {solution_length}."
                    f" However, the shape of `stdev_init` is {stdev_init.shape}."
                    f" Please ensure that `stdev_init` is either a scalar, or is a vector of length {solution_length},"
                    f" or is a batch of vectors where the rightmost dimension's size is {solution_length}."
                )
        return stdev_init
    elif (stdev_init is None) and (radius_init is not None):
        # This is the case where `radius_init` is given and `stdev_init` is omitted.
        # We make a standard deviation vector (or a batch of standard deviation vectors) such that the norm of the
        # the vector(s) is/are equal to the given radius value(s).
        radius_init = torch.as_tensor(radius_init, dtype=dtype, device=device)
        stdev_element = torch.sqrt((radius_init**2) / solution_length)
        final_stdev = stdev_element[..., None] * torch.ones(solution_length)
        return final_stdev
    else:
        raise ValueError(
            "Both `stdev_init` and `radius_init` are encountered as values other than None."
            " Please specify either `stdev_init` or `radius_init`, but not both."
        )
