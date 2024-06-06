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

"""Miscellaneous utility functions"""

import functools
import inspect
import logging
import math
import sys
from collections.abc import Mapping
from numbers import Integral, Number, Real
from typing import Any, Callable, Dict, Iterable, NamedTuple, Optional, Type, Union

import numpy as np
import torch
from torch import nn

DType = Union[str, torch.dtype, np.dtype, Type]
Device = Union[str, torch.device]
DTypeAndDevice = NamedTuple("DTypeAndDevice", dtype=DType, device=Device)
Size = Union[int, torch.Size]
RealOrVector = Union[float, Iterable[float], torch.Tensor]
Vector = Union[Iterable[float], torch.Tensor]
BatchableScalar = Union[Number, np.ndarray, torch.Tensor]
BatchableVector = Union[torch.Tensor, np.ndarray]


try:
    import sacred
except ImportError:
    sacred = None


class SuppressSacredExperiment:
    @staticmethod
    def empty_decorator(f):
        return f

    config = empty_decorator
    capture = empty_decorator
    command = empty_decorator
    automain = empty_decorator
    main = empty_decorator


def expect_none(msg_prefix: str, **kwargs):
    """
    Expect the values associated with the given keyword arguments
    to be None. If not, raise error.

    Args:
        msg_prefix: Prefix of the error message.
        kwargs: Keyword arguments whose values are expected to be None.
    Raises:
        ValueError: if at least one of the keyword arguments has a value
            other than None.
    """
    for k, v in kwargs.items():
        if v is not None:
            raise ValueError(f"{msg_prefix}: expected `{k}` as None, however, it was found to be {repr(v)}")


def to_torch_dtype(dtype: DType) -> torch.dtype:
    """
    Convert the given string or the given numpy dtype to a PyTorch dtype.
    If the argument is already a PyTorch dtype, then the argument is returned
    as it is.

    Returns:
        The dtype, converted to a PyTorch dtype.
    """
    if isinstance(dtype, str) and hasattr(torch, dtype):
        attrib_within_torch = getattr(torch, dtype)
    else:
        attrib_within_torch = None

    if isinstance(attrib_within_torch, torch.dtype):
        return attrib_within_torch
    elif isinstance(dtype, torch.dtype):
        return dtype
    elif dtype is Any or dtype is object:
        raise TypeError(f"Cannot make a numeric tensor with dtype {repr(dtype)}")
    else:
        return torch.from_numpy(np.array([], dtype=dtype)).dtype


def to_numpy_dtype(dtype: DType) -> np.dtype:
    """
    Convert the given string or the given PyTorch dtype to a numpy dtype.
    If the argument is already a numpy dtype, then the argument is returned
    as it is.

    Returns:
        The dtype, converted to a numpy dtype.
    """
    if isinstance(dtype, torch.dtype):
        return torch.tensor([], dtype=dtype).numpy().dtype
    elif is_dtype_object(dtype):
        return np.dtype(object)
    elif isinstance(dtype, np.dtype):
        return dtype
    else:
        return np.dtype(dtype)


def is_dtype_object(dtype: DType) -> bool:
    """
    Return True if the given dtype is `object` or `Any`.

    Returns:
        True if the given dtype is `object` or `Any`; False otherwise.
    """
    if isinstance(dtype, str):
        return dtype in ("object", "Any", "O")
    elif dtype is object or dtype is Any:
        return True
    else:
        return False


def is_sequence(x: Any) -> bool:
    """
    Return True if `x` is a sequence.
    Note that this function considers `str` and `bytes` as scalars,
    not as sequences.

    Args:
        x: The object whose sequential nature is being queried.
    Returns:
        True if `x` is a sequence; False otherwise.
    """
    if isinstance(x, (str, bytes)):
        return False
    elif isinstance(x, (np.ndarray, torch.Tensor)):
        return x.ndim > 0
    elif isinstance(x, Iterable):
        return True
    else:
        return False


def is_dtype_integer(t: DType) -> bool:
    """
    Return True if the given dtype is an integer type.

    Args:
        t: The dtype, which can be a dtype string, a numpy dtype,
            or a PyTorch dtype.
    Returns:
        True if t is an integer type; False otherwise.
    """
    t: np.dtype = to_numpy_dtype(t)
    return t.kind.startswith("u") or t.kind.startswith("i")


def is_dtype_float(t: DType) -> bool:
    """
    Return True if the given dtype is an float type.

    Args:
        t: The dtype, which can be a dtype string, a numpy dtype,
            or a PyTorch dtype.
    Returns:
        True if t is an float type; False otherwise.
    """
    t: np.dtype = to_numpy_dtype(t)
    return t.kind.startswith("f")


def is_dtype_bool(t: DType) -> bool:
    """
    Return True if the given dtype is an bool type.

    Args:
        t: The dtype, which can be a dtype string, a numpy dtype,
            or a PyTorch dtype.
    Returns:
        True if t is a bool type; False otherwise.
    """
    t: np.dtype = to_numpy_dtype(t)
    return t.kind.startswith("b")


def is_dtype_real(t: DType) -> bool:
    """
    Return True if the given dtype represents real numbers
    (i.e. if dtype is an integer type or is a float type).

    Args:
        t: The dtype, which can be a dtype string, a numpy dtype,
            or a PyTorch dtype.
    Returns:
        True if t represents a real numbers type; False otherwise.
    """
    return is_dtype_float(t) or is_dtype_integer(t)


def is_integer(x: Any) -> bool:
    """
    Return True if `x` is an integer.

    Note that this function does NOT consider booleans as integers.

    Args:
        x: An object whose type is being queried.
    Returns:
        True if `x` is an integer; False otherwise.
    """
    if is_bool(x):
        return False
    elif isinstance(x, Integral):
        return True
    elif isinstance(x, (torch.Tensor, np.ndarray)):
        if x.ndim > 0:
            return False
        else:
            return is_dtype_integer(x.dtype)
    else:
        return False


def is_real(x: Any) -> bool:
    """
    Return True if `x` is a real number.

    Note that this function does NOT consider booleans as real numbers.

    Args:
        x: An object whose type is being queried.
    Returns:
        True if `x` is a real number; False otherwise.
    """
    if is_bool(x):
        return False
    elif isinstance(x, Real):
        return True
    elif isinstance(x, (torch.Tensor, np.ndarray)):
        if x.ndim > 0:
            return False
        else:
            return is_dtype_real(x.dtype)
    else:
        return False


def is_bool(x: Any) -> bool:
    """
    Return True if `x` represents a bool.

    Args:
        x: An object whose type is being queried.
    Returns:
        True if `x` is a bool; False otherwise.
    """
    if isinstance(x, (bool, np.bool_)):
        return True
    elif isinstance(x, (torch.Tensor, np.ndarray)):
        if x.ndim > 0:
            return False
        else:
            return is_dtype_bool(x.dtype)
    else:
        return False


def is_integer_vector(x: Any) -> bool:
    """
    Return True if `x` is a vector consisting of integers.

    Args:
        x: An object whose elements' types are to be queried.
    Returns:
        True if the elements of `x` are integers; False otherwise.
    """
    if isinstance(x, (torch.Tensor, np.ndarray)):
        if x.ndim != 1:
            return False
        else:
            return is_dtype_integer(x.dtype)
    elif isinstance(x, Iterable):
        for item in x:
            if not is_integer(item):
                return False
        return True
    else:
        return False


def is_bool_vector(x: Any) -> bool:
    """
    Return True if `x` is a vector consisting of bools.

    Args:
        x: An object whose elements' types are to be queried.
    Returns:
        True if the elements of `x` are bools; False otherwise.
    """
    if isinstance(x, (torch.Tensor, np.ndarray)):
        if x.ndim != 1:
            return False
        else:
            return is_dtype_bool(x.dtype)
    elif isinstance(x, Iterable):
        for item in x:
            if not is_bool(item):
                return False
        return True
    else:
        return False


def is_real_vector(x: Any) -> bool:
    """
    Return True if `x` is a vector consisting of real numbers.

    Args:
        x: An object whose elements' types are to be queried.
    Returns:
        True if the elements of `x` are real numbers; False otherwise.
    """
    if isinstance(x, (torch.Tensor, np.ndarray)):
        if x.ndim != 1:
            return False
        else:
            return is_dtype_real(x.dtype)
    elif isinstance(x, Iterable):
        for item in x:
            if not is_real(item):
                return False
        return True
    else:
        return False


def cast_tensors_in_container(
    container: Any,
    *,
    dtype: Optional[DType] = None,
    device: Optional[Device] = None,
    memo: Optional[dict] = None,
) -> Any:
    """
    Cast and/or transfer all the tensors in a Python container.

    Args:
        dtype: If given as a dtype and not as None, then all the PyTorch
            tensors in the container will be cast to this dtype.
        device: If given as a device and not as None, then all the PyTorch
            tensors in the container will be copied to this device.
        memo: Optionally a memo dictionary to handle shared objects and
            circular references. In most scenarios, when calling this
            function from outside, this is expected as None.
    Returns:
        A new copy of the original container in which the tensors have the
        desired dtype and/or device.
    """

    if memo is None:
        memo = {}

    container_id = id(container)
    if container_id in memo:
        return memo[container_id]

    cast_kwargs = {}
    if dtype is not None:
        cast_kwargs["dtype"] = to_torch_dtype(dtype)
    if device is not None:
        cast_kwargs["device"] = device

    def call_self(sub_container: Any) -> Any:
        return cast_tensors_in_container(sub_container, dtype=dtype, device=device, memo=memo)

    if isinstance(container, torch.Tensor):
        result = torch.as_tensor(container, **cast_kwargs)
        memo[container_id] = result
    elif (container is None) or isinstance(container, (Number, str, bytes, bool)):
        result = container
    elif isinstance(container, set):
        result = set()
        memo[container_id] = result
        for x in container:
            result.add(call_self(x))
    elif isinstance(container, Mapping):
        result = {}
        memo[container_id] = result
        for k, v in container.items():
            result[k] = call_self(v)
    elif isinstance(container, tuple):
        result = []
        memo[container_id] = result
        for x in container:
            result.append(call_self(x))
        if hasattr(container, "_fields"):
            result = type(container)(*result)
        else:
            result = type(container)(result)
        memo[container_id] = result
    elif isinstance(container, Iterable):
        result = []
        memo[container_id] = result
        for x in container:
            result.append(call_self(x))
    else:
        raise TypeError(f"Encountered an object of unrecognized type: {type(container)}")

    return result


def dtype_of_container(
    container: Any, *, visited: Optional[dict] = None, visiting: Optional[str] = None
) -> Optional[torch.dtype]:
    """
    Get the dtype of the given container.

    It is assumed that the given container stores PyTorch tensors from
    which the dtype information will be extracted.
    If the container contains only basic types like int, float, string,
    bool, or None, or if the container is empty, then the returned dtype
    will be None.
    If the container contains unrecognized objects, an error will be
    raised.

    Args:
        container: A sequence or a dictionary of objects from which the
            dtype information will be extracted.
        visited: Optionally a dictionary which stores the (sub)containers
            which are already visited. In most cases, when this function
            is called from outside, this is expected as None.
        visiting: Optionally a set which stores the (sub)containers
            which are being visited. This set is used to prevent recursion
            errors while handling circular references. In most cases,
            when this function is called from outside, this argument is
            expected as None.
    Returns:
        The dtype if available, None otherwise.
    """

    container_id = id(container)

    if visited is None:
        visited = {}

    if container_id in visited:
        return visited[container_id]

    if visiting is None:
        visiting = set()

    if container_id in visiting:
        return None

    class result:
        dtype: Optional[torch.dtype] = None

        @classmethod
        def update(cls, new_dtype: Optional[torch.dtype]):
            if new_dtype is not None:
                if cls.dtype is None:
                    cls.dtype = new_dtype
                else:
                    if new_dtype != cls.dtype:
                        raise ValueError(f"Encountered tensors whose `dtype`s mismatch: {new_dtype}, {cls.dtype}")

    def call_self(sub_container):
        return dtype_of_container(sub_container, visited=visited, visiting=visiting)

    if isinstance(container, torch.Tensor):
        result.update(container.dtype)
    elif (container is None) or isinstance(container, (Number, str, bytes, bool)):
        pass
    elif isinstance(container, Mapping):
        visiting.add(container_id)
        try:
            for _, v in container.items():
                result.update(call_self(v))
        finally:
            visiting.remove(container_id)
    elif isinstance(container, Iterable):
        visiting.add(container_id)
        try:
            for v in container:
                result.update(call_self(v))
        finally:
            visiting.remove(container_id)
    else:
        raise TypeError(f"Encountered an object of unrecognized type: {type(container)}")

    visited[container_id] = result.dtype
    return result.dtype


def device_of_container(
    container: Any, *, visited: Optional[dict] = None, visiting: Optional[str] = None
) -> Optional[torch.device]:
    """
    Get the device of the given container.

    It is assumed that the given container stores PyTorch tensors from
    which the device information will be extracted.
    If the container contains only basic types like int, float, string,
    bool, or None, or if the container is empty, then the returned device
    will be None.
    If the container contains unrecognized objects, an error will be
    raised.

    Args:
        container: A sequence or a dictionary of objects from which the
            device information will be extracted.
        visited: Optionally a dictionary which stores the (sub)containers
            which are already visited. In most cases, when this function
            is called from outside, this is expected as None.
        visiting: Optionally a set which stores the (sub)containers
            which are being visited. This set is used to prevent recursion
            errors while handling circular references. In most cases,
            when this function is called from outside, this argument is
            expected as None.
    Returns:
        The device if available, None otherwise.
    """
    container_id = id(container)

    if visited is None:
        visited = {}

    if container_id in visited:
        return visited[container_id]

    if visiting is None:
        visiting = set()

    if container_id in visiting:
        return None

    class result:
        device: Optional[torch.device] = None

        @classmethod
        def update(cls, new_device: Optional[torch.device]):
            if new_device is not None:
                if cls.device is None:
                    cls.device = new_device
                else:
                    if new_device != cls.device:
                        raise ValueError(f"Encountered tensors whose `device`s mismatch: {new_device}, {cls.device}")

    def call_self(sub_container):
        return device_of_container(sub_container, visited=visited, visiting=visiting)

    if isinstance(container, torch.Tensor):
        result.update(container.device)
    elif (container is None) or isinstance(container, (Number, str, bytes, bool)):
        pass
    elif isinstance(container, Mapping):
        visiting.add(container_id)
        try:
            for _, v in container.items():
                result.update(call_self(v))
        finally:
            visiting.remove(container_id)
    elif isinstance(container, Iterable):
        visiting.add(container_id)
        try:
            for v in container:
                result.update(call_self(v))
        finally:
            visiting.remove(container_id)
    else:
        raise TypeError(f"Encountered an object of unrecognized type: {type(container)}")

    visited[container_id] = result.device
    return result.device


@torch.no_grad()
def clone(x: Any, *, memo: Optional[dict] = None) -> Any:
    """
    Get a deep copy of the given object.

    The cloning is done in no_grad mode.

    When this function is used on read-only containers (e.g. ReadOnlyTensor,
    ImmutableContainer, etc.), the created clones preserve their read-only
    behaviors. For creating a mutable clone of an immutable object,
    use their `clone()` method instead.

    Returns:
        The deep copy of the given object.
    """
    from .cloning import deep_clone

    if memo is None:
        memo = {}
    return deep_clone(x, otherwise_deepcopy=True, memo=memo)


@torch.no_grad()
def ensure_tensor_length_and_dtype(
    t: Any,
    length: int,
    dtype: DType,
    about: Optional[str] = None,
    *,
    allow_scalar: bool = False,
    device: Optional[Device] = None,
) -> Any:
    """
    Return the given sequence as a tensor while also confirming its
    length, dtype, and device.
    If the given object is already a tensor conforming to the desired
    length, dtype, and device, the object will be returned as it is
    (there will be no copying).

    Args:
        t: The tensor, or a sequence which is convertible to a tensor.
        length: The length to which the tensor is expected to conform.
        dtype: The dtype to which the tensor is expected to conform.
        about: The prefix for the error message. Can be left as None.
        allow_scalar: Whether or not to accept scalars in addition
            to vector of the desired length.
            If `allow_scalar` is False, then scalars will be converted
            to sequences of the desired length. The sequence will contain
            the same scalar, repeated.
            If `allow_scalar` is True, then the scalar itself will be
            converted to a PyTorch scalar, and then will be returned.
        device: The device in which the sequence is to be stored.
            If the given sequence is on a different device than the
            desired device, a copy on the correct device will be made.
            If device is None, the default behavior of `torch.tensor(...)`
            will be used, that is: if `t` is already a tensor, the result
            will be on the same device, otherwise, the result will be on
            the cpu.
    Returns:
        The sequence whose correctness in terms of length, dtype, and
        device is ensured.
    Raises:
        ValueError: if there is a length mismatch.
    """
    device_args = {}
    if device is not None:
        device_args["device"] = device

    t = as_tensor(t, dtype=dtype, **device_args)

    if t.ndim == 0:
        if allow_scalar:
            return t
        else:
            return t.repeat(length)
    else:
        if t.ndim != 1 or len(t) != length:
            if about is not None:
                err_prefix = about + ": "
            else:
                err_prefix = ""
            raise ValueError(
                f"{err_prefix}Expected a 1-dimensional tensor of length {length}, but got a tensor with shape: {t.shape}"
            )
        return t


@torch.no_grad()
def clip_tensor(
    x: torch.Tensor,
    lb: Optional[Union[float, Iterable]] = None,
    ub: Optional[Union[float, Iterable]] = None,
    ensure_copy: bool = True,
) -> torch.Tensor:
    """
    Clip the values of a tensor with respect to the given bounds.

    Args:
        x: The PyTorch tensor whose values will be clipped.
        lb: Lower bounds, as a PyTorch tensor.
            Can be None if there are no lower bounds.
        ub: Upper bounds, as a PyTorch tensor.
            Can be None if there are no upper bonuds.
        ensure_copy: If `ensure_copy` is True, the result will be
            a clipped copy of the original tensor.
            If `ensure_copy` is False, and both `lb` and `ub`
            are None, then there is nothing to do, so, the result
            will be the original tensor itself, not a copy of it.
    Returns:
        The clipped tensor.
    """
    result = x
    if lb is not None:
        lb = torch.as_tensor(lb, dtype=x.dtype, device=x.device)
        result = torch.max(result, lb)
    if ub is not None:
        ub = torch.as_tensor(ub, dtype=x.dtype, device=x.device)
        result = torch.min(result, ub)
    if ensure_copy and result is x:
        result = x.clone()
    return result


@torch.no_grad()
def modify_tensor(
    original: torch.Tensor,
    target: torch.Tensor,
    lb: Optional[Union[float, torch.Tensor]] = None,
    ub: Optional[Union[float, torch.Tensor]] = None,
    max_change: Optional[Union[float, torch.Tensor]] = None,
    in_place: bool = False,
) -> torch.Tensor:
    """Return the modified version of the original tensor, with bounds checking.

    Args:
        original: The original tensor.
        target: The target tensor which contains the values to replace the
            old ones in the original tensor.
        lb: The lower bound(s), as a scalar or as an tensor.
            Values below these bounds are clipped in the resulting tensor.
            None means -inf.
        ub: The upper bound(s), as a scalar or as an tensor.
            Value above these bounds are clipped in the resulting tensor.
            None means +inf.
        max_change: The ratio of allowed change.
            In more details, when given as a real number r,
            modifications are allowed only within
            ``[original-(r*abs(original)) ... original+(r*abs(original))]``.
            Modifications beyond this interval are clipped.
            This argument can also be left as None if no such limitation
            is needed.
        in_place: Provide this as True if you wish the modification to be
            done within the original tensor. The default value of this
            argument is False, which means, the original tensor is not
            changed, and its modified version is returned as an independent
            copy.
    Returns:
        The modified tensor.
    """
    if (lb is None) and (ub is None) and (max_change is None):
        # If there is no restriction regarding how the tensor
        # should be modified (no lb, no ub, no max_change),
        # then we simply use the target values
        # themselves for modifying the tensor.

        if in_place:
            original[:] = target
            return original
        else:
            return target
    else:
        # If there are some restriction regarding how the tensor
        # should be modified, then we turn to the following
        # operations

        def convert_to_tensor(x, tensorname: str):
            if isinstance(x, torch.Tensor):
                converted = x
            else:
                converted = torch.as_tensor(x, dtype=original.dtype, device=original.device)
            if converted.ndim == 0 or converted.shape == original.shape:
                return converted
            else:
                raise IndexError(
                    f"Argument {tensorname}: shape mismatch."
                    f" Shape of the original tensor: {original.shape}."
                    f" Shape of {tensorname}: {converted.shape}."
                )

        if lb is None:
            # If lb is None, then it should be taken as -inf
            lb = convert_to_tensor(float("-inf"), "lb")
        else:
            lb = convert_to_tensor(lb, "lb")

        if ub is None:
            # If ub is None, then it should be taken as +inf
            ub = convert_to_tensor(float("inf"), "ub")
        else:
            ub = convert_to_tensor(ub, "ub")

        if max_change is not None:
            # If max_change is provided as something other than None,
            # then we update the lb and ub so that they are tight
            # enough to satisfy the max_change restriction.

            max_change = convert_to_tensor(max_change, "max_change")
            allowed_amounts = torch.abs(original) * max_change
            allowed_lb = original - allowed_amounts
            allowed_ub = original + allowed_amounts
            lb = torch.max(lb, allowed_lb)
            ub = torch.min(ub, allowed_ub)

        ## If in_place is given as True, the clipping (that we are about
        ## to perform), should be in-place.
        # more_config = {}
        # if in_place:
        #    more_config['out'] = original
        #
        ## Return the clipped version of the target values
        # return torch.clamp(target, lb, ub, **more_config)

        result = torch.max(target, lb)
        result = torch.min(result, ub)

        if in_place:
            original[:] = result
            return original
        else:
            return result


def _modify_vector_using_bounds(
    target: torch.Tensor,
    lb: torch.Tensor,
    ub: torch.Tensor,
) -> torch.Tensor:
    # Strictly expect `target` as a 1-dimensional tensor, and get its length
    [target_length] = target.shape

    # Ensure that `lb` and `ub` are 1-dimensional tensors, and get their lengths
    if lb.ndim == 0:
        lb = lb.expand(target.shape)
    if ub.ndim == 0:
        ub = ub.expand(target.shape)
    [lb_length] = lb.shape
    [ub_length] = ub.shape

    # Verify that the lengths of the vectors match
    if target_length != lb_length:
        raise ValueError("The lower bound (`lb`) has a different length than the given `target`")
    if target_length != ub_length:
        raise ValueError("The upper bound (`ub`) has a different length than the given `target`")

    return torch.min(torch.max(target, lb), ub)


def _modify_vector_using_max_change(
    original: torch.Tensor,
    target: torch.Tensor,
    max_change: torch.Tensor,
) -> torch.Tensor:
    # Strictly expect `original` and `target` as 1-dimensional tensors, and get their lengths
    [original_length] = original.shape
    [target_length] = target.shape

    # Ensure that `max_change` is a 1-dimensional tensor, and get its length
    if max_change.ndim == 0:
        max_change = max_change.expand(target.shape)
    [max_change_length] = max_change.shape

    # Verify that the lengths of the vectors match
    if target_length != original_length:
        raise ValueError("The `target` vector and the `original` vector have different lengths")
    if original_length != max_change_length:
        raise ValueError("The length of `max_change` is different than the length of `original`")

    max_diff = torch.abs(original) * max_change
    return torch.min(torch.max(target, original - max_diff), original + max_diff)


def modify_vector(
    original: torch.Tensor,
    target: torch.Tensor,
    *,
    lb: Optional[Union[float, torch.Tensor]] = None,
    ub: Optional[Union[float, torch.Tensor]] = None,
    max_change: Optional[Union[float, torch.Tensor]] = None,
) -> torch.Tensor:
    """
    Return the modified version(s) of the vector(s), with bounds checking.

    This function is similar to `modify_tensor`, but it has the following
    different behaviors:

    - Assumes that all of its arguments are either vectors, or are batches
      of vectors. If some or more of its arguments have 2 or more dimensions,
      those arguments will be considered as batches, and the computation will
      be vectorized to return a batch of results.
    - Designed to be `vmap`-friendly.
    - Designed for functional programming paradigm, and therefore lacks the
      in-place modification option.
    """
    from ..decorators import expects_ndim

    if max_change is None:
        result = target
    else:
        result = expects_ndim(_modify_vector_using_max_change, (1, 1, 1), allow_smaller_ndim=True)(
            original, target, max_change
        )

    if (lb is None) and (ub is None):
        pass  # no strict boundaries, so, nothing more to do
    elif (lb is not None) and (ub is not None):
        result = expects_ndim(_modify_vector_using_bounds, (1, 1, 1), allow_smaller_ndim=True)(result, lb, ub)
    else:
        raise ValueError(
            "`modify_vector` expects either with `lb` and `ub` given together, or with both of them omitted."
            " Having only `lb` or only `ub` is not supported."
        )
    return result


def empty_tensor_like(
    source: Any,
    *,
    shape: Optional[Union[tuple, int]] = None,
    length: Optional[int] = None,
    dtype: Optional[DType] = None,
    device: Optional[Device] = None,
) -> Any:
    """
    Make an empty tensor with attributes taken from a source tensor.

    The source tensor can be a PyTorch tensor, or an ObjectArray.

    Unlike `torch.empty_like(...)`, this function allows one to redefine the
    shape and/or length of the new empty tensor.

    Args:
        source: The source tensor whose shape, dtype, and device will be used
            by default for the new empty tensor.
        shape: If given as None (which is the default), then the shape of the
            source tensor will be used for the new empty tensor.
            If given as a tuple or a `torch.Size` instance, then the new empty
            tensor will be in this given shape instead.
            This argument cannot be used together with `length`.
        length: If given as None (which is the default), then the length of
            the new empty tensor will be equal to the length of the source
            tensor (where length here means the size of the outermost
            dimension, i.e., what is returned by `len(...)`).
            If given as an integer, the length of the empty tensor will be
            this given length instead.
            This argument cannot be used together with `shape`.
        dtype: If given as None, the dtype of the new empty tensor will be
            the dtype of the source tensor.
            If given as a `torch.dtype` instance, then the dtype of the
            tensor will be this given dtype instead.
        device: If given as None, the device of the new empty tensor will be
            the device of the source tensor.
            If given as a `torch.device` instance, then the device of the
            tensor will be this given device instead.
    Returns:
        The new empty tensor.
    """
    from .objectarray import ObjectArray

    if isinstance(source, ObjectArray):
        if length is not None and shape is None:
            n = int(length)
        elif shape is not None and length is None:
            if isinstance(shape, Iterable):
                if len(shape) != 1:
                    raise ValueError(
                        f"An ObjectArray must always be 1-dimensional."
                        f" Therefore, this given shape is incompatible: {shape}"
                    )
                n = int(shape[0])
        elif length is None and shape is None:
            n = len(source)
        else:
            raise ValueError("`length` and `shape` cannot be used together")

        if device is not None:
            if str(device) != "cpu":
                raise ValueError(
                    f"An ObjectArray can only be allocated on cpu. However, the specified `device` is: {device}."
                )

        if dtype is not None:
            if not is_dtype_object(dtype):
                raise ValueError(
                    f"The dtype of an ObjectArray can only be `object`. However, the specified `dtype` is: {dtype}."
                )

        return ObjectArray(n)
    elif isinstance(source, torch.Tensor):
        if length is not None:
            if shape is not None:
                raise ValueError("`length` and `shape` cannot be used together")
            if source.ndim == 0:
                raise ValueError("`length` can only be used when the source tensor is at least 1-dimensional")
            newshape = [int(length)]
            newshape.extend(source.shape[1:])
            shape = tuple(newshape)

        if not ((dtype is None) or isinstance(dtype, torch.dtype)):
            dtype = to_torch_dtype(dtype)

        return torch.empty(
            source.shape if shape is None else shape,
            dtype=(source.dtype if dtype is None else dtype),
            device=(source.device if device is None else device),
        )
    else:
        raise TypeError(f"The source tensor is of an unrecognized type: {type(source)}")


class ErroneousResult:
    """
    Representation of a caught error being returned as a result.
    """

    def __init__(self, error: Exception):
        self.error = error

    def _to_string(self) -> str:
        return f"<{type(self).__name__}, error: {self.error}>"

    def __str__(self) -> str:
        return self._to_string()

    def __repr__(self) -> str:
        return self._to_string()

    def __bool__(self) -> bool:
        return False

    @staticmethod
    def call(f, *args, **kwargs) -> Any:
        """
        Call a function with the given arguments.
        If the function raises an error, wrap the error in an ErroneousResult
        object, and return that ErroneousResult object instead.

        Returns:
            The result of the function if there was no error,
            or an ErroneousResult if there was an error.
        """
        try:
            result = f(*args, **kwargs)
        except Exception as ex:
            result = ErroneousResult(ex)
        return result


def is_tensor_on_cpu(tensor) -> bool:
    """
    Return True of the given PyTorch tensor or ObjectArray is on cpu.
    """
    return str(tensor.device) == "cpu"


def numpy_copy(x: Iterable, dtype: Optional[DType] = None) -> np.ndarray:
    """
    Return a numpy copy of the given iterable.

    The newly created numpy array will be mutable, even if the
    original iterable object is read-only.

    Args:
        x: Any Iterable whose numpy copy will be returned.
        dtype: The desired dtype. Can be given as a numpy dtype,
            as a torch dtype, or a native dtype (e.g. int, float),
            or as a string (e.g. "float32").
            If left as None, dtype will be determined according
            to the data contained by the original iterable object.
    Returns:
        The numpy copy of the original iterable object.
    """
    from .objectarray import ObjectArray

    needs_casting = dtype is not None

    if isinstance(x, ObjectArray):
        result = x.numpy()
    elif isinstance(x, torch.Tensor):
        result = x.cpu().clone().numpy()
    elif isinstance(x, np.ndarray):
        result = x.copy()
    else:
        needs_casting = False
        result = np.array(x, dtype=dtype)

    if needs_casting:
        result = result.astype(dtype)

    return result


@torch.jit.script
def multiply_rows_by_scalars(multipliers: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    # Imagine the multipliers is:
    #     [m0 m1 m2]

    # and the tensor x is:
    #     [ x00 x01 x02 x03 ]
    #     [ x10 x11 x12 x13 ]
    #     [ x20 x21 x22 x23 ]

    # The result of this function:
    #     [ m0*x00   m0*x01   m0*x02   m0*x03 ]
    #     [ m1*x10   m1*x11   m1*x12   m1*x13 ]
    #     [ m2*x20   m2*x21   m2*x22   m2*x23 ]

    return (multipliers * x.T).T


@torch.jit.script
def rowwise_sum(x: torch.Tensor) -> torch.Tensor:
    # Apply a summation on the rows of the 2D tensor x,
    # and return a resulting row.
    return torch.sum(x, 0)


def split_workload(workload: int, num_actors: int) -> list:
    """
    Split a workload among actors.

    By "workload" what is meant is the total amount of a work,
    this amount being expressed by an integer.
    For example, if the "work" is the evaluation of a population,
    the "workload" would usually be the population size.

    Args:
        workload: Total amount of work, as an integer.
        num_actors: Number of actors (i.e. remote workers) among
            which the workload will be distributed.
    Returns:
        A list of integers. The i-th item of the returned list
        expresses the suggested workload for the i-th actor.
    """
    base_workload = workload // num_actors
    extra_workload = workload % num_actors
    result = [base_workload] * num_actors
    for i in range(extra_workload):
        result[i] += 1
    return result


def make_tensor(
    data: Any,
    *,
    dtype: Optional[DType] = None,
    device: Optional[Device] = None,
    read_only: bool = False,
) -> Iterable:
    """
    Make a new tensor.

    This function can be used to create PyTorch tensors, or ObjectArray
    instances with or without read-only behavior.

    The following example creates a 2-dimensional PyTorch tensor:

        my_tensor = make_tensor(
            [[1, 2], [3, 4]],
            dtype="float32",    # alternatively, torch.float32
            device="cpu",
        )

    The following example creates an ObjectArray from a list that contains
    arbitrary data:

        my_obj_tensor = make_tensor(["a_string", (1, 2)], dtype=object)

    Args:
        data: The data to be converted to a tensor.
            If one wishes to create a PyTorch tensor, this can be anything
            that can be stored by a PyTorch tensor.
            If one wishes to create an `ObjectArray` and therefore passes
            `dtype=object`, then the provided `data` is expected as an
            `Iterable`.
        dtype: Optionally a string (e.g. "float32"), or a PyTorch dtype
            (e.g. torch.float32), or `object` or "object" (as a string)
            or `Any` if one wishes to create an `ObjectArray`.
            If `dtype` is not specified, it will be assumed that the user
            wishes to create a PyTorch tensor (not an `ObjectArray`) and
            then `dtype` will be inferred from the provided `data`
            (according to the default behavior of PyTorch).
        device: The device in which the tensor will be stored.
            If `device` is not specified, it will be understood from the
            given `data` (according to the default behavior of PyTorch).
        read_only: Whether or not the created tensor will be read-only.
            By default, this is False.
    Returns:
        A PyTorch tensor or an ObjectArray.
    """
    from .objectarray import ObjectArray
    from .readonlytensor import as_read_only_tensor

    if (dtype is not None) and is_dtype_object(dtype):
        if not hasattr(data, "__len__"):
            data = list(data)
        n = len(data)
        result = ObjectArray(n)
        result[:] = data
    else:
        kwargs = {}
        if dtype is not None:
            kwargs["dtype"] = to_torch_dtype(dtype)
        if device is not None:
            kwargs["device"] = device
        result = torch.tensor(data, **kwargs)

    if read_only:
        result = as_read_only_tensor(result)

    return result


def make_empty(
    *size: Size,
    dtype: Optional[DType] = None,
    device: Optional[Device] = None,
) -> Iterable:
    """
    Make an empty tensor.

    Args:
        size: Shape of the empty tensor to be created.
            expected as multiple positional arguments of integers,
            or as a single positional argument containing a tuple of
            integers.
            Note that when the user wishes to create an `ObjectArray`
            (i.e. when `dtype` is given as `object`), then the size
            is expected as a single integer, or as a single-element
            tuple containing an integer (because `ObjectArray` can only
            be one-dimensional).
        dtype: Optionally a string (e.g. "float32") or a PyTorch dtype
            (e.g. torch.float32) or, for creating an `ObjectArray`,
            "object" (as string) or `object` or `Any`.
            If `dtype` is not specified, the default choice of
            `torch.empty(...)` is used, that is, `torch.float32`.
        device: The device in which the new empty tensor will be stored.
            If not specified, "cpu" will be used.
    Returns:
        The new empty tensor, which can be a PyTorch tensor or an
        `ObjectArray`.
    """
    from .objectarray import ObjectArray

    if (dtype is not None) and is_dtype_object(dtype):
        if (device is None) or (str(device) == "cpu"):
            if len(size) == 1:
                size = size[0]
            return ObjectArray(size)
        else:
            return ValueError(
                f"Invalid device for ObjectArray: {repr(device)}. Note: an ObjectArray can only be stored on 'cpu'."
            )
    else:
        kwargs = {}
        if dtype is not None:
            kwargs["dtype"] = to_torch_dtype(dtype)
        if device is not None:
            kwargs["device"] = device
        return torch.empty(*size, **kwargs)


def _out_tensor(
    *size: Size,
    out: Optional[torch.Tensor] = None,
    dtype: Optional[DType] = None,
    device: Optional[Device] = None,
) -> torch.Tensor:
    if out is None:
        out = make_empty(*size, dtype=dtype, device=device)
    else:
        if len(size) >= 1:
            raise ValueError(
                f"When `out` is provided (i.e. not None), the positional `size` arguments were not expected."
                f" However, `size` arguments were received as {repr(size)}."
            )
        expect_none("when `out` is provided (i.e. not None)", dtype=dtype, device=device)
    return out


def _out_tensor_for_random_operation(
    *size: Size,
    out: Optional[torch.Tensor] = None,
    dtype: Optional[DType] = None,
    device: Optional[Device] = None,
) -> torch.Tensor:
    if out is None:
        out = torch.empty(tuple(), dtype=dtype, device=device).expand(*size)
        out = out + make_batched_false_for_vmap(out.device)
    else:
        if len(size) >= 1:
            raise ValueError(
                f"When `out` is provided (i.e. not None), the positional `size` arguments were not expected."
                f" However, `size` arguments were received as {repr(size)}."
            )
        expect_none("when `out` is provided (i.e. not None)", dtype=dtype, device=device)
    return out


def _scalar_requested(*size: Size) -> bool:
    return (len(size) == 1) and isinstance(size[0], tuple) and (len(size[0]) == 0)


def _scalar_tensor(
    value: Union[int, float],
    *,
    out: Optional[torch.Tensor] = None,
    dtype: Optional[DType] = None,
    device: Optional[Device] = None,
) -> torch.Tensor:
    expect_none("when creating a scalar (with size specified as an empty tuple)", out=out)
    result = make_empty((1,), dtype=dtype, device=device)
    result[0] = value
    return result[0]


def make_zeros(
    *size: Size,
    out: Optional[torch.Tensor] = None,
    dtype: Optional[DType] = None,
    device: Optional[Device] = None,
) -> torch.Tensor:
    """
    Make a new tensor filled with 0, or fill an existing tensor with 0.

    The following example creates a float32 tensor filled with 0 values,
    of shape (3, 5):

        zero_values = make_zeros(3, 5, dtype="float32")

    The following example fills an existing tensor with 0s:

        make_zeros(out=existing_tensor)

    Args:
        size: Size of the new tensor to be filled with 0.
            This can be given as multiple positional arguments, each such
            positional argument being an integer, or as a single positional
            argument of a tuple, the tuple containing multiple integers.
            Note that, if the user wishes to fill an existing tensor with
            0 values, then no positional argument is expected.
        out: Optionally, the tensor to be filled by 0 values.
            If an `out` tensor is given, then no `size` argument is expected.
        dtype: Optionally a string (e.g. "float32") or a PyTorch dtype
            (e.g. torch.float32).
            If `dtype` is not specified, the default choice of
            `torch.empty(...)` is used, that is, `torch.float32`.
            If an `out` tensor is specified, then `dtype` is expected
            as None.
        device: The device in which the new tensor will be stored.
            If not specified, "cpu" will be used.
            If an `out` tensor is specified, then `device` is expected
            as None.
    Returns:
        The created or modified tensor after placing 0 values.
    """
    if _scalar_requested(*size):
        return _scalar_tensor(0, out=out, dtype=dtype, device=device)
    else:
        out = _out_tensor(*size, out=out, dtype=dtype, device=device)
        out.zero_()
        return out


def make_ones(
    *size: Size,
    out: Optional[torch.Tensor] = None,
    dtype: Optional[DType] = None,
    device: Optional[Device] = None,
) -> torch.Tensor:
    """
    Make a new tensor filled with 1, or fill an existing tensor with 1.

    The following example creates a float32 tensor filled with 1 values,
    of shape (3, 5):

        zero_values = make_ones(3, 5, dtype="float32")

    The following example fills an existing tensor with 1s:

        make_ones(out=existing_tensor)

    Args:
        size: Size of the new tensor to be filled with 1.
            This can be given as multiple positional arguments, each such
            positional argument being an integer, or as a single positional
            argument of a tuple, the tuple containing multiple integers.
            Note that, if the user wishes to fill an existing tensor with
            1 values, then no positional argument is expected.
        out: Optionally, the tensor to be filled by 1 values.
            If an `out` tensor is given, then no `size` argument is expected.
        dtype: Optionally a string (e.g. "float32") or a PyTorch dtype
            (e.g. torch.float32).
            If `dtype` is not specified, the default choice of
            `torch.empty(...)` is used, that is, `torch.float32`.
            If an `out` tensor is specified, then `dtype` is expected
            as None.
        device: The device in which the new tensor will be stored.
            If not specified, "cpu" will be used.
            If an `out` tensor is specified, then `device` is expected
            as None.
    Returns:
        The created or modified tensor after placing 1 values.
    """
    if _scalar_requested(*size):
        return _scalar_tensor(1, out=out, dtype=dtype, device=device)
    else:
        out = _out_tensor(*size, out=out, dtype=dtype, device=device)
        out[:] = 1
        return out


def make_nan(
    *size: Size,
    out: Optional[torch.Tensor] = None,
    dtype: Optional[DType] = None,
    device: Optional[Device] = None,
) -> torch.Tensor:
    """
    Make a new tensor filled with NaN, or fill an existing tensor with NaN.

    The following example creates a float32 tensor filled with NaN values,
    of shape (3, 5):

        nan_values = make_nan(3, 5, dtype="float32")

    The following example fills an existing tensor with NaNs.

        make_nan(out=existing_tensor)

    Args:
        size: Size of the new tensor to be filled with NaNs.
            This can be given as multiple positional arguments, each such
            positional argument being an integer, or as a single positional
            argument of a tuple, the tuple containing multiple integers.
            Note that, if the user wishes to fill an existing tensor with
            NaN values, then no positional argument is expected.
        out: Optionally, the tensor to be filled by NaN values.
            If an `out` tensor is given, then no `size` argument is expected.
        dtype: Optionally a string (e.g. "float32") or a PyTorch dtype
            (e.g. torch.float32).
            If `dtype` is not specified, the default choice of
            `torch.empty(...)` is used, that is, `torch.float32`.
            If an `out` tensor is specified, then `dtype` is expected
            as None.
        device: The device in which the new tensor will be stored.
            If not specified, "cpu" will be used.
            If an `out` tensor is specified, then `device` is expected
            as None.
    Returns:
        The created or modified tensor after placing NaN values.
    """
    if _scalar_requested(*size):
        return _scalar_tensor(float("nan"), out=out, dtype=dtype, device=device)
    else:
        out = _out_tensor(*size, out=out, dtype=dtype, device=device)
        out[:] = float("nan")
        return out


def make_I(
    size: Optional[int] = None,
    *,
    out: Optional[torch.Tensor] = None,
    dtype: Optional[DType] = None,
    device: Optional[Device] = None,
) -> torch.Tensor:
    """
    Make a new identity matrix (I), or change an existing tensor into one.

    The following example creates a 3x3 identity matrix:

        identity_matrix = make_I(3, dtype="float32")

    The following example changes an already existing square matrix such that
    its values will store an identity matrix:

        make_I(out=existing_tensor)

    Args:
        size: A single integer or a tuple containing a single integer,
            where the integer specifies the length of the target square
            matrix. In this context, "length" means both rowwise length
            and columnwise length, since the target is a square matrix.
            Note that, if the user wishes to fill an existing tensor with
            identity values, then `size` is expected to be left as None.
        out: Optionally, the existing tensor whose values will be changed
            so that they represent an identity matrix.
            If an `out` tensor is given, then `size` is expected as None.
        dtype: Optionally a string (e.g. "float32") or a PyTorch dtype
            (e.g. torch.float32).
            If `dtype` is not specified, the default choice of
            `torch.empty(...)` is used, that is, `torch.float32`.
            If an `out` tensor is specified, then `dtype` is expected
            as None.
        device: The device in which the new tensor will be stored.
            If not specified, "cpu" will be used.
            If an `out` tensor is specified, then `device` is expected
            as None.
    Returns:
        The created or modified tensor after placing the I matrix values
    """
    if size is None:
        if out is None:
            raise ValueError(
                "When the `size` argument is missing, the function `make_I(...)` expects an `out` tensor."
                " However, the `out` argument was received as None."
            )
        size = tuple()
    else:
        if isinstance(size, tuple):
            if len(size) == 1:
                size = size[0]
            else:
                raise ValueError(
                    f"When the `size` argument is given as a tuple,"
                    f" the function `make_I(...)` expects this tuple to contain exactly one element."
                    f" The received tuple is {size}."
                )
        n = int(size)
        size = (n, n)
    out = _out_tensor(*size, out=out, dtype=dtype, device=device)
    out.zero_()
    out.fill_diagonal_(1)
    return out


def _generator_of(generator: Any) -> torch.Generator:
    if isinstance(generator, torch.Generator):
        return generator
    elif hasattr(generator, "generator"):
        return generator.generator
    else:
        raise TypeError(
            f"The provided `generator` was expected as a `torch.Generator`"
            f" or as an object which has a `generator` attribute (e.g. a `Problem` object)."
            f" However, `generator` was received as an incompatible object: {repr(generator)}"
        )


def _generator_kwargs(generator: Any) -> torch.Generator:
    return {} if generator is None else {"generator": _generator_of(generator)}


def make_uniform(
    *size: Size,
    lb: Optional[RealOrVector] = None,
    ub: Optional[RealOrVector] = None,
    out: Optional[torch.Tensor] = None,
    dtype: Optional[DType] = None,
    device: Optional[Device] = None,
    generator: Any = None,
) -> torch.Tensor:
    """
    Make a new or existing tensor filled by uniformly distributed values.
    Both lower and upper bounds are inclusive.
    This function can work with both float and int dtypes.

    Args:
        size: Size of the new tensor to be filled with uniformly distributed
            values. This can be given as multiple positional arguments, each
            such positional argument being an integer, or as a single
            positional argument of a tuple, the tuple containing multiple
            integers. Note that, if the user wishes to fill an existing
            tensor instead, then no positional argument is expected.
        lb: Lower bound for the uniformly distributed values.
            Can be a scalar, or a tensor.
            If not specified, the lower bound will be taken as 0.
            Note that, if one specifies `lb`, then `ub` is also expected to
            be explicitly specified.
        ub: Upper bound for the uniformly distributed values.
            Can be a scalar, or a tensor.
            If not specified, the upper bound will be taken as 1.
            Note that, if one specifies `ub`, then `lb` is also expected to
            be explicitly specified.
        out: Optionally, the tensor to be filled by uniformly distributed
            values. If an `out` tensor is given, then no `size` argument is
            expected.
        dtype: Optionally a string (e.g. "float32") or a PyTorch dtype
            (e.g. torch.float32).
            If `dtype` is not specified, the default choice of
            `torch.empty(...)` is used, that is, `torch.float32`.
            If an `out` tensor is specified, then `dtype` is expected
            as None.
        device: The device in which the new tensor will be stored.
            If not specified, "cpu" will be used.
            If an `out` tensor is specified, then `device` is expected
            as None.
        generator: Pseudo-random number generator to be used when sampling
            the values. Can be a `torch.Generator`, or an object with
            a `generator` attribute (such as `Problem`).
            If left as None, the global generator of PyTorch will be used.
    Returns:
        The created or modified tensor after placing the uniformly
        distributed values.
    """

    scalar_requested = _scalar_requested(*size)
    if scalar_requested:
        size = (1,)

    def _invalid_bound_args():
        raise ValueError(
            f"Expected both `lb` and `ub` as None, or both `lb` and `ub` as not None."
            f" It appears that one of them is None, while the other is not."
            f" lb: {repr(lb)}."
            f" ub: {repr(ub)}."
        )

    out = _out_tensor_for_random_operation(*size, out=out, dtype=dtype, device=device)
    gen_kwargs = _generator_kwargs(generator)

    def _cast_bounds():
        nonlocal lb, ub
        lb = torch.as_tensor(lb, dtype=out.dtype, device=out.device)
        ub = torch.as_tensor(ub, dtype=out.dtype, device=out.device)

    if out.dtype == torch.bool:
        out.random_(**gen_kwargs)
        if (lb is None) and (ub is None):
            pass  # nothing to do
        elif (lb is not None) and (ub is not None):
            _cast_bounds()
            lb_shape_matches = lb.shape == out.shape
            ub_shape_matches = ub.shape == out.shape
            if (not lb_shape_matches) or (not ub_shape_matches):
                all_false = torch.zeros_like(out)
                if not lb_shape_matches:
                    lb = lb | all_false
                if not ub_shape_matches:
                    ub = ub | all_false
            mask_for_always_false = (~lb) & (~ub)
            mask_for_always_true = lb & ub
            out[mask_for_always_false] = False
            out[mask_for_always_true] = True
        else:
            _invalid_bound_args()
    elif out.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        out.random_(**gen_kwargs)
        if (lb is None) and (ub is None):
            out %= 2
        elif (lb is not None) and (ub is not None):
            _cast_bounds()
            diff = (ub - lb) + 1
            out -= lb
            out %= diff
            out += lb
        else:
            _invalid_bound_args()
    else:
        out.uniform_(**gen_kwargs)
        if (lb is None) and (ub is None):
            pass  # nothing to do
        elif (lb is not None) and (ub is not None):
            _cast_bounds()
            diff = ub - lb
            out *= diff
            out += lb
        else:
            _invalid_bound_args()

    if scalar_requested:
        out = out[0]

    return out


def make_gaussian(
    *size: Size,
    center: Optional[RealOrVector] = None,
    stdev: Optional[RealOrVector] = None,
    symmetric: bool = False,
    out: Optional[torch.Tensor] = None,
    dtype: Optional[DType] = None,
    device: Optional[Device] = None,
    generator: Any = None,
) -> torch.Tensor:
    """
    Make a new or existing tensor filled by Gaussian distributed values.
    This function can work only with float dtypes.

    Args:
        size: Size of the new tensor to be filled with Gaussian distributed
            values. This can be given as multiple positional arguments, each
            such positional argument being an integer, or as a single
            positional argument of a tuple, the tuple containing multiple
            integers. Note that, if the user wishes to fill an existing
            tensor instead, then no positional argument is expected.
        center: Center point (i.e. mean) of the Gaussian distribution.
            Can be a scalar, or a tensor.
            If not specified, the center point will be taken as 0.
            Note that, if one specifies `center`, then `stdev` is also
            expected to be explicitly specified.
        stdev: Standard deviation for the Gaussian distributed values.
            Can be a scalar, or a tensor.
            If not specified, the standard deviation will be taken as 1.
            Note that, if one specifies `stdev`, then `center` is also
            expected to be explicitly specified.
        symmetric: Whether or not the values should be sampled in a
            symmetric (i.e. antithetic) manner.
            The default is False.
        out: Optionally, the tensor to be filled by Gaussian distributed
            values. If an `out` tensor is given, then no `size` argument is
            expected.
        dtype: Optionally a string (e.g. "float32") or a PyTorch dtype
            (e.g. torch.float32).
            If `dtype` is not specified, the default choice of
            `torch.empty(...)` is used, that is, `torch.float32`.
            If an `out` tensor is specified, then `dtype` is expected
            as None.
        device: The device in which the new tensor will be stored.
            If not specified, "cpu" will be used.
            If an `out` tensor is specified, then `device` is expected
            as None.
        generator: Pseudo-random number generator to be used when sampling
            the values. Can be a `torch.Generator`, or an object with
            a `generator` attribute (such as `Problem`).
            If left as None, the global generator of PyTorch will be used.
    Returns:
        The created or modified tensor after placing the Gaussian
        distributed values.
    """
    scalar_requested = _scalar_requested(*size)
    if scalar_requested:
        size = (1,)

    out = _out_tensor_for_random_operation(*size, out=out, dtype=dtype, device=device)
    gen_kwargs = _generator_kwargs(generator)

    if symmetric:
        leftmost_dim = out.shape[0]
        if (leftmost_dim % 2) != 0:
            raise ValueError(
                f"Symmetric sampling cannot be done if the leftmost dimension of the target tensor is odd."
                f" The shape of the target tensor is: {repr(out.shape)}."
            )
        out[0::2, ...].normal_(**gen_kwargs)
        out[1::2, ...] = out[0::2, ...]
        out[1::2, ...] *= -1
    else:
        out.normal_(**gen_kwargs)

    if (center is None) and (stdev is None):
        pass  # do nothing
    elif (center is not None) and (stdev is not None):
        stdev = torch.as_tensor(stdev, dtype=out.dtype, device=out.device)
        out *= stdev
        center = torch.as_tensor(center, dtype=out.dtype, device=out.device)
        out += center
    else:
        raise ValueError(
            f"Please either specify none of `stdev` and `center`, or both of them."
            f" Currently, `center` is {center}"
            f" and `stdev` is {stdev}."
        )

    if scalar_requested:
        out = out[0]

    return out


def make_randint(
    *size: Size,
    n: Union[int, float, torch.Tensor],
    out: Optional[torch.Tensor] = None,
    dtype: Optional[DType] = None,
    device: Optional[Device] = None,
    generator: Any = None,
) -> torch.Tensor:
    """
    Make a new or existing tensor filled by random integers.
    The integers are uniformly distributed within `[0 ... n-1]`.
    This function can be used with integer or float dtypes.

    Args:
        size: Size of the new tensor to be filled with uniformly distributed
            values. This can be given as multiple positional arguments, each
            such positional argument being an integer, or as a single
            positional argument of a tuple, the tuple containing multiple
            integers. Note that, if the user wishes to fill an existing
            tensor instead, then no positional argument is expected.
        n: Number of choice(s) for integer sampling.
            The lowest possible value will be 0, and the highest possible
            value will be n - 1.
            `n` can be a scalar, or a tensor.
        out: Optionally, the tensor to be filled by the random integers.
            If an `out` tensor is given, then no `size` argument is
            expected.
        dtype: Optionally a string (e.g. "int64") or a PyTorch dtype
            (e.g. torch.int64).
            If `dtype` is not specified, torch.int64 will be used.
        device: The device in which the new tensor will be stored.
            If not specified, "cpu" will be used.
            If an `out` tensor is specified, then `device` is expected
            as None.
        generator: Pseudo-random number generator to be used when sampling
            the values. Can be a `torch.Generator`, or an object with
            a `generator` attribute (such as `Problem`).
            If left as None, the global generator of PyTorch will be used.
    Returns:
            The created or modified tensor after placing the uniformly
            distributed values.
    """
    scalar_requested = _scalar_requested(*size)
    if scalar_requested:
        size = (1,)

    if (dtype is None) and (out is None):
        dtype = torch.int64
    out = _out_tensor_for_random_operation(*size, out=out, dtype=dtype, device=device)
    gen_kwargs = _generator_kwargs(generator)
    out.random_(**gen_kwargs)
    out %= n

    if scalar_requested:
        out = out[0]

    return out


def as_tensor(x: Any, *, dtype: Optional[DType] = None, device: Optional[Device] = None) -> Iterable:
    """
    Get the tensor counterpart of the given object `x`.

    This function can be used to convert native Python objects to tensors:

        my_tensor = as_tensor([1.0, 2.0, 3.0], dtype="float32")

    One can also use this function to convert an existing tensor to another
    dtype:

        my_new_tensor = as_tensor(my_tensor, dtype="float16")

    This function can also be used for moving a tensor from one device to
    another:

        my_gpu_tensor = as_tensor(my_tensor, device="cuda:0")

    This function can also create ObjectArray instances when dtype is
    given as `object` or `Any` or "object" or "O".

        my_objects = as_tensor([1, {"a": 3}], dtype=object)

    Args:
        x: Any object to be converted to a tensor.
        dtype: Optionally a string (e.g. "float32") or a PyTorch dtype
            (e.g. torch.float32) or, for creating an `ObjectArray`,
            "object" (as string) or `object` or `Any`.
            If `dtype` is not specified, the default behavior of
            `torch.as_tensor(...)` will be used, that is, dtype will be
            inferred from `x`.
        device: The device in which the resulting tensor will be stored.
    Returns:
        The tensor counterpart of the given object `x`.
    """
    from .objectarray import ObjectArray

    if (dtype is None) and isinstance(x, (torch.Tensor, ObjectArray)):
        if (device is None) or (str(device) == "cpu"):
            return x
        else:
            raise ValueError(
                f"An ObjectArray cannot be moved into a device other than 'cpu'. The received device is: {device}."
            )
    elif is_dtype_object(dtype):
        if (device is not None) and (str(device) != "cpu"):
            raise ValueError(
                f"An ObjectArray cannot be created on a device other than 'cpu'. The received device is: {device}."
            )
        if isinstance(x, ObjectArray):
            return x
        else:
            x = list(x)
            n = len(x)
            result = ObjectArray(n)
            result[:] = x
            return result
    else:
        dtype = to_torch_dtype(dtype)
        return torch.as_tensor(x, dtype=dtype, device=device)


def stdev_from_radius(radius: float, solution_length: int) -> float:
    """
    Get elementwise standard deviation from a given radius.

    Sometimes, for a distribution-based search algorithm, the user might
    choose to configure the initial coverage area of the search distribution
    not via standard deviation, but via a radius value, as was done in the
    study of Toklu et al. (2020).
    This function takes the desired radius value and the solution length of
    the problem at hand, and returns the elementwise standard deviation value.
    Let us name this returned standard deviation value as `s`.
    When a new Gaussian distribution is constructed such that its initial
    standard deviation is `[s, s, s, ...]` (the length of this vector being
    equal to the solution length), this constructed distribution's radius
    corresponds with the desired radius.

    Here, the "radius" of a Gaussian distribution is defined as the norm
    of the standard deviation vector. In the case of a standard normal
    distribution, this radius formulation serves as a simplified approximation
    to `E[||Normal(0, I)||]` (for which a closer approximation is used in
    the study of Hansen & Ostermeier (2001)).

    Reference:

        Toklu, N.E., Liskowski, P., Srivastava, R.K. (2020).
        ClipUp: A Simple and Powerful Optimizer
        for Distribution-based Policy Evolution.
        Parallel Problem Solving from Nature (PPSN 2020).

        Nikolaus Hansen, Andreas Ostermeier (2001).
        Completely Derandomized Self-Adaptation in Evolution Strategies.

    Args:
        radius: The radius whose elementwise standard deviation counterpart
            will be returned.
        solution_length: Length of a solution for the problem at hand.
    Returns:
        An elementwise standard deviation value `s`, such that a Gaussian
        distribution constructed with the standard deviation `[s, s, s, ...]`
        has the desired radius.
    """
    radius = float(radius)
    solution_length = int(solution_length)
    return math.sqrt((radius**2) / solution_length)


def to_stdev_init(
    *,
    solution_length: int,
    stdev_init: Optional[RealOrVector] = None,
    radius_init: Optional[RealOrVector] = None,
) -> RealOrVector:
    """
    Ask for both standard deviation and radius, return the standard deviation.

    It is very common among the distribution-based search algorithms to ask
    for both standard deviation and for radius for initializing the coverage
    area of the search distribution. During their initialization phases,
    these algorithms must check which one the user provided (radius or
    standard deviation), and return the result as the standard deviation
    so that a Gaussian distribution can easily be constructed.

    This function serves as a helper function for such search algorithms
    by performing these actions:

    - If the user provided a standard deviation and not a radius, then this
      provided standard deviation is simply returned.
    - If the user provided a radius and not a standard deviation, then this
      provided radius is converted to its standard deviation counterpart,
      and then returned.
    - If both standard deviation and radius are missing, or they are both
      given at the same time, then an error is raised.

    Args:
        solution_length: Length of a solution for the problem at hand.
        stdev_init: Standard deviation. If one wishes to provide a radius
            instead, then `stdev_init` is expected as None.
        radius_init: Radius. If one wishes to provide a standard deviation
            instead, then `radius_init` is expected as None.
    Returns:
        The standard deviation for the search distribution to be constructed.
    """
    if (stdev_init is not None) and (radius_init is None):
        return stdev_init
    elif (stdev_init is None) and (radius_init is not None):
        return stdev_from_radius(radius_init, solution_length)
    elif (stdev_init is None) and (radius_init is None):
        raise ValueError(
            "Received both `stdev_init` and `radius_init` as None."
            " Please provide a value either for `stdev_init` or for `radius_init`."
        )
    else:
        raise ValueError(
            "Found both `stdev_init` and `radius_init` with values other than None."
            " Please provide a value either for `stdev_init` or for `radius_init`, but not for both."
        )


def ensure_ray():
    """
    Ensure that the ray parallelization engine is initialized.
    If ray is already initialized, this function does nothing.
    """
    import ray

    if not ray.is_initialized():
        ray.init()


def dtype_of(x: Any) -> DType:
    """
    Get the dtype of the given object.

    Args:
        x: The object whose dtype is being queried.
            The object can be a PyTorch tensor, or a PyTorch module
            (in which case the dtype of the first parameter tensor
            will be returned), or an ObjectArray (in which case
            the returned dtype will be `object`), or any object with
            the attribute `dtype`.
    Returns:
        The dtype of the given object.
    """
    if isinstance(x, nn.Module):
        result = None
        for param in x.parameters():
            result = param.dtype
            break
        if result is None:
            raise ValueError(f"Cannot determine the dtype of the module {x}")
        return result
    else:
        return x.dtype


def device_of(x: Any) -> Device:
    """
    Get the device of the given object.

    Args:
        x: The object whose device is being queried.
            The object can be a PyTorch tensor, or a PyTorch module
            (in which case the device of the first parameter tensor
            will be returned), or an ObjectArray (in which case
            the returned device will be the cpu device), or any object
            with the attribute `device`.
    Returns:
        The device of the given object.
    """
    if isinstance(x, nn.Module):
        result = None
        for param in x.parameters():
            result = param.device
            break
        if result is None:
            raise ValueError(f"Cannot determine the device of the module {x}")
        return result
    else:
        return x.device


def pass_info_if_needed(f: Callable, info: Dict[str, Any]) -> Callable:
    """
    Pass additional arguments into a callable, the info dictionary is unpacked
    and passed as additional keyword arguments only if the policy is decorated
    with the [pass_info][evotorch.decorators.pass_info] decorator.

    Args:
        f (Callable): The callable to be called.
        info (Dict[str, Any]): The info to be passed to the callable.
    Returns:
        Callable: The callable with extra arguments
    Raises:
        TypeError: If the callable is decorated with the [pass_info][evotorch.decorators.pass_info] decorator,
            but its signature does not match the expected signature.
    """
    if hasattr(f, "__evotorch_pass_info__"):
        try:
            sig = inspect.signature(f)
            sig.bind_partial(**info)
        except TypeError:
            raise TypeError(
                "Callable {f} is decorated with @pass_info, but it doesn't expect some of the extra arguments "
                f"({', '.join(info.keys())}). Hint: maybe you forgot to add **kwargs to the function signature?"
            )
        except Exception:
            pass

        return functools.partial(f, **info)
    else:
        return f


def set_default_logger_config(
    logger_name: str = "evotorch",
    logger_level: int = logging.INFO,
    show_process: bool = True,
    show_lineno: bool = False,
    override: bool = False,
):
    """
    Configure the "EvoTorch" Python logger to print to the console with default format.

    The logger will be configured to print to all messages with level INFO or lower to stdout and all
    messages with level WARNING or higher to stderr.

    The default format is:
    ```
    [2022-11-23 22:28:47] INFO     <75935>   evotorch:      This is a log message
    {asctime}             {level}  {process} {logger_name}: {message}
    ```
    The format can be slightly customized by passing `show_process=False` to hide Process ID or `show_lineno=True` to
    show the filename and line number of the log message instead of the Logger Name.

    This function should be called before any other logging is performed, otherwise the default configuration will
    not be applied. If the logger is already configured, this function will do nothing unless `override=True` is passed,
    in which case the logger will be reconfigured.

    Args:
        logger_name: Name of the logger to configure.
        logger_level: Level of the logger to configure.
        show_process: Whether to show the process name in the log message.
        show_lineno: Whether to show the filename with the line number in the log message or just the name of the logger.
        override: Whether to override the logger configuration if it has already been configured.
    """
    logger = logging.getLogger(logger_name)

    if not override and logger.hasHandlers():
        # warn user that the logger is already configured
        logger.warning(
            "The logger is already configured. "
            "The default configuration will not be applied. "
            "Call `set_default_logger_config` with `override=True` to override the current configuration."
        )
        return
    elif override:
        # remove all handlers
        for handler in logger.handlers:
            logger.removeHandler(handler)

    logger.setLevel(logger_level)
    logger.propagate = False

    formatter = logging.Formatter(
        "[{asctime}] "
        + "{levelname:<8s} "
        + ("<{process:5d}> " if show_process else "")
        + ("{filename}:{lineno}: " if show_lineno else "{name}: ")
        + "{message}",
        datefmt="%Y-%m-%d %H:%M:%S",
        style="{",
    )

    _stdout_handler = logging.StreamHandler(sys.stdout)
    _stdout_handler.addFilter(lambda log_record: log_record.levelno < logging.WARNING)
    _stdout_handler.setFormatter(formatter)
    logger.addHandler(_stdout_handler)

    _stderr_handler = logging.StreamHandler(sys.stderr)
    _stderr_handler.addFilter(lambda log_record: log_record.levelno >= logging.WARNING)
    _stderr_handler.setFormatter(formatter)
    logger.addHandler(_stderr_handler)


def message_from(sender: object, message: Any) -> str:
    """
    Prepend the sender object's name and id to a string message.

    Let us imagine that we have a class named `Example`:

    ```python
    from evotorch.tools import message_from


    class Example:
        def say_hello(self):
            print(message_from(self, "Hello!"))
    ```

    Let us now instantiate this class and use its `say_hello` method:

    ```python
    ex = Example()
    ex.say_hello()
    ```

    The output becomes something like this:

    ```
    Instance of `Example` (id:...) -- Hello!
    ```

    Args:
        sender: The object which produces the message
        message: The message, as something that can be converted to string
    Returns:
        The new message string, with the details regarding the sender object
        inserted to the beginning.
    """
    sender_type = type(sender).__name__
    sender_id = id(sender)

    return f"Instance of `{sender_type}` (id:{sender_id}) -- {message}"


if hasattr(torch.Tensor, "untyped_storage"):

    def _storage_ptr(x: Iterable) -> int:
        return x.untyped_storage().data_ptr()

else:

    def _storage_ptr(x: Iterable) -> int:
        return x.storage().data_ptr()


def storage_ptr(x: Iterable) -> int:
    """
    Get the pointer to the underlying storage of a tensor of an ObjectArray.

    Calling `storage_ptr(x)` is equivalent to `x.untyped_storage().data_ptr()`.

    Args:
        x: A regular PyTorch tensor, or a ReadOnlyTensor, or an ObjectArray.
    Returns:
        The address of the underlying storage.
    """
    return _storage_ptr(x)


def make_batched_false_for_vmap(device: Device) -> torch.Tensor:
    """
    Get `False`, properly batched if inside `vmap(..., randomness='different')`.

    **Reasoning.**
    Imagine we have the following function:

    ```python
    import torch


    def sample_and_shift(target_shape: tuple, shift: torch.Tensor) -> torch.Tensor:
        result = torch.empty(target_shape, device=x.device)
        result.normal_()
        result += shift
        return result
    ```

    which allocates an empty tensor, then fills it with samples from the
    standard normal distribution, then shifts the samples and returns the
    result. An important implementation detail regarding this example function
    is that all of its operations are in-place (i.e. the method `normal_()`
    and the operator `+=` work on the given pre-allocated tensor).

    Let us now imagine that we have a batch of shift tensors, and we would like
    to generate multiple shifted sample tensors. Ideally, such a batched
    operation could be done by transforming the example function with the help
    of `vmap`:

    ```python
    from torch.func import vmap

    batched_sample_and_shift = vmap(sample_and_shift, in_dims=0, randomness="different")
    ```

    where the argument `randomness="different"` tells PyTorch that for each
    batch item, we want to generate different samples (instead of just
    duplicating the same samples across the batch dimension(s)).
    Such a re-sampling approach is usually desired in applications where
    preserving stochasticity is crucial, evolutionary computation being one
    of such case.

    Now let us call our transformed function:

    ```python
    batch_of_shifts = ...  # a tensor like `shift`, but with an extra leftmost
    # dimension for the batches

    # Will fail:
    batched_results = batched_sample_and_shift(shape_goes_here, batch_of_shifts)
    ```

    At this point, we observe that `batched_sample_and_shift` fails.
    The reason for this failure is that the function first allocates an empty
    tensor, then tries to perform random sampling in an in-place manner.
    The first allocation via `empty` is not properly batched (it is not aware
    of the active `vmap`), so, when we later call `.normal_()` on it,
    there is no room for the data that would be re-sampled for each batch item.
    To remedy this, we could modify our original function slightly:

    ```python
    import torch


    def sample_and_shift2(target_shape: tuple, shift: torch.Tensor) -> torch.Tensor:
        result = torch.empty(target_shape, device=x.device)
        result = result + result.make_batched_false_for_vmap(x.device)
        result.normal_()
        result += shift
        return result
    ```

    In this modified function, right after making an initial allocation, we add
    onto it a batched false, and re-assign the result to the variable `result`.
    Thanks to being the result of an interaction with a batched false, the new
    `result` variable is now properly batched (if we are inside
    `vmap(..., randomness="different")`. Now, let us transform our function:

    ```python
    from torch.func import vmap

    batched_sample_and_shift2 = vmap(sample_and_shift2, in_dims=0, randomness="different")
    ```

    The following code should now work:

    ```python
    batch_of_shifts = ...  # a tensor like `shift`, but with an extra leftmost
    # dimension for the batches

    # Should work:
    batched_results = batched_sample_and_shift2(shape_goes_here, batch_of_shifts)
    ```

    Args:
        device: The target device on which the batched `False` will be created
    Returns:
        A scalar tensor having the value `False`. This returned tensor will be
        a batch of scalar tensors (i.e. a `BatchedTensor`) if we are inside
        `vmap(..., randomness="different")`.
    """
    return torch.randint(0, 1, tuple(), dtype=torch.bool, device=device)
