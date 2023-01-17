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

"""
Utility functions for evotorch-related unit testing.
"""

from numbers import Real
from typing import Any, Iterable, Optional, Type, Union

import numpy as np
import torch


class TestingError(Exception):
    """
    An Exception type to be raised when a unit testing function
    encounters illegal arguments.
    """

    pass


_MaybeDType = Optional[Union[np.dtype, torch.dtype, str, Type]]


def _to_numpy_dtype(dtype: _MaybeDType) -> Optional[np.dtype]:
    from .tools import to_numpy_dtype

    return None if dtype is None else to_numpy_dtype(dtype)


def _to_torch_dtype(dtype: _MaybeDType) -> Optional[torch.dtype]:
    from .tools import to_torch_dtype

    return None if dtype is None else to_torch_dtype(dtype)


def _to_numpy(x: Iterable, *, dtype: _MaybeDType = None) -> np.ndarray:
    from .core import Solution, SolutionBatch
    from .tools import to_numpy_dtype

    dtype = _to_numpy_dtype(dtype)
    do_casting = True

    if isinstance(x, np.ndarray):
        result = x
    elif isinstance(x, torch.Tensor):
        result = x.cpu().numpy()
    elif isinstance(x, (Solution, SolutionBatch)):
        result = x.values.cpu().numpy()
    else:
        do_casting = False
        if dtype is None:
            dtype = "float32"
        result = np.array(x, dtype=dtype)

    if do_casting and (dtype is not None):
        result = result.astype(to_numpy_dtype(dtype))

    return result


def _to_torch(x: Iterable, *, dtype: Optional[Union[np.dtype, torch.dtype, str]] = None) -> torch.Tensor:
    from .core import Solution, SolutionBatch
    from .tools import to_torch_dtype

    dtype = _to_torch_dtype(dtype)
    do_casting = True

    if isinstance(x, np.ndarray):
        result = torch.from_numpy(x)
    elif isinstance(x, torch.Tensor):
        result = x
    elif isinstance(x, (Solution, SolutionBatch)):
        result = x.values
    else:
        do_casting = False
        if dtype is None:
            dtype = torch.float32
        result = torch.tensor(x, dtype=dtype)

    if do_casting and (dtype is not None):
        result = torch.as_tensor(result, dtype=dtype)

    return result


def assert_allclose(
    actual: Iterable,
    desired: Iterable,
    *,
    rtol: Optional[float] = None,
    atol: Optional[float] = None,
    equal_nan: bool = True,
):
    """
    This function is similar to `numpy.testing.assert_allclose(...)` except
    that `atol` and `rtol` are keyword-only arguments (which encourages
    one to be more explicit when writing tests) and that the default dtype
    is "float32" when the provided arguments are neither numpy arrays nor
    torch tensors. Having "float32" as the default target dtype is a behavior
    that is compatible with PyTorch.

    This function first casts `actual` into the dtype of `desired`, then
    uses numpy's `assert_allclose(...)` for testing the closeness of the
    values.

    Args:
        actual: An iterable of numbers.
        desired: An iterable of numbers. These numbers represent the values
            that we expect the `actual` to contain. If the numbers contained
            by `actual` are significantly different than `desired`, the
            assertion will fail.
        rtol: Relative tolerance.
            Can be left as None if only `atol` is to be used.
            See the documentation of `numpy.testing.assert_allclose(...)`
            for details about how `rtol` affects the tolerance.
        atol: Absolute tolerance.
            Can be left as None if only `rtol` is to be used.
            See the documentation of `numpy.testing.assert_allclose(...)`
            for details about how `atol` affects the tolerance.
        equal_nan: If True, `nan` values will be counted as equal.
    Raises:
        AssertionError: if the numerical difference between `actual`
            and `desired` are beyond the tolerance expressed by `atol`
            and `rtol`.
        TestingError: If both `rtol` and `atol` are given as None.
    """

    if rtol is None and atol is None:
        raise TestingError(
            "Both `rtol` and `atol` were found to be None. Please either specify `rtol`, `atol`, or both."
        )
    elif rtol is None:
        rtol = 0.0
    elif atol is None:
        atol = 0.0

    desired = _to_numpy(desired)
    actual = _to_numpy(actual, dtype=desired.dtype)

    np.testing.assert_allclose(actual, desired, rtol=rtol, atol=atol, equal_nan=bool(equal_nan))


def assert_almost_between(
    x: Iterable, lb: Union[float, Iterable], ub: Union[float, Iterable], *, atol: Optional[float] = None
):
    """
    Assert that the given Iterable has its values between the desired bounds.

    Args:
        x: An Iterable containing numeric (float) values.
        lb: Lower bound for the desired interval.
            Can be a scalar or an iterable of values.
        ub: Upper bound for the desired interval.
            Can be a scalar or an iterable of values.
        atol: Absolute tolerance. If given, then the effective interval will
            be `[lb-atol; ub+atol]` instead of `[lb; ub]`.
    Raises:
        AssertionError: if any element of `x` violates the boundaries.
    """

    x = _to_numpy(x)
    lb = _to_numpy(lb)
    ub = _to_numpy(ub)

    if lb.shape != x.shape:
        lb = np.broadcast_to(lb, x.shape)
    if ub.shape != x.shape:
        ub = np.broadcast_to(ub, x.shape)

    lb = np.asarray(lb, dtype=x.dtype)
    ub = np.asarray(ub, dtype=x.dtype)

    if atol is not None:
        atol = float(atol)
        tolerant_lb = lb - atol
        tolerant_ub = ub + atol
    else:
        tolerant_lb = lb
        tolerant_ub = ub

    assert np.all((x >= tolerant_lb) & (x <= tolerant_ub)), (
        f"The provided array is not within the desired boundaries."
        f"Provided array: {x}. Lower bound: {lb}. Upper bound: {ub}. Absolute tolerance: {atol}."
    )


def assert_dtype_matches(x: Iterable, dtype: Union[str, Type, np.dtype, torch.dtype]):
    """
    Assert that the dtype of `x` is compatible with the given `dtype`.

    Args:
        x: An object with `dtype` attribute (e.g. can be numpy array,
            a torch tensor, an ObjectArray, a Solution, etc.)
        dtype: The dtype which `x` is expected to have.
            Can be given as a string, as a numpy dtype, as a torch dtype,
            or as a native type (e.g. int, float, bool, object).
    Raises:
        AssertionError: if `x` has a different dtype.
    """
    actual_dtype = x.dtype

    if isinstance(actual_dtype, torch.dtype):
        actual_dtype = torch.tensor([], dtype=actual_dtype).numpy().dtype
    else:
        actual_dtype = np.dtype(actual_dtype)

    if dtype == "Any" or dtype is Any:
        dtype = np.dtype(object)
    elif isinstance(dtype, torch.dtype):
        dtype = torch.tensor([], dtype=dtype).numpy().dtype
    else:
        dtype = np.dtype(dtype)

    assert dtype == actual_dtype, f"dtype mismatch. Encountered dtype: {actual_dtype}, expected dtype: {dtype}"


def assert_shape_matches(x: Iterable, shape: Union[tuple, int]):
    """
    Assert that the dtype of `x` matches the given shape

    Args:
        x: An object which can be converted to a PyTorch tensor.
        shape: A tuple, or a torch.Size, or an integer.
    Raises:
        AssertionError: if there is a shape mismatch.
    """
    if isinstance(x, torch.Tensor):
        pass  # nothing to do
    elif isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    else:
        x = torch.tensor(x)

    if not isinstance(shape, Iterable):
        shape = (int(shape),)

    assert x.shape == shape, f"Encountered a shape mismatch. Shape of the tensor: {x.shape}. Expected shape: {shape}"


def assert_eachclose(x: Iterable, value: Any, *, rtol: Optional[float] = None, atol: Optional[float] = None):
    """
    Assert that the given tensor or array consists of a single value.

    Args:
        x: The tensor in which each value will be compared against `value`
        value: A scalar
    Raises:
        AssertionError: if at least one value is different enough
    """

    # If the given scalar is not a Real, then try to cast it to float
    if not isinstance(value, Real):
        value = float(value)

    x = _to_numpy(x)
    desired = np.empty_like(x)
    desired[:] = value

    assert_allclose(x, desired, rtol=rtol, atol=atol)
