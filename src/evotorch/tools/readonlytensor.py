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

from typing import Any, Callable, Iterable, Mapping, Optional, Union

import numpy as np
import torch
from packaging.version import Version

# Determine the PyTorch version without failing when __version__ has an unexpected value.
# Perhaps such unexpected values could be encountered when using a custom/modified version
# of PyTorch, and we do not wish this module to fail in those scenarios.
_torch_older_than_1_12 = Version(torch.__version__) < Version("1.12")


class ReadOnlyTensor(torch.Tensor):
    """
    A special type of tensor which is read-only.

    This is a subclass of `torch.Tensor` which explicitly disallows
    operations that would cause in-place modifications.

    Since ReadOnlyTensor if a subclass of `torch.Tensor`, most
    non-destructive PyTorch operations are on this tensor are supported.

    Cloning a ReadOnlyTensor using the `clone()` method or Python's
    `deepcopy(...)` function results in a regular PyTorch tensor.

    Reshaping or slicing operations might return a ReadOnlyTensor if the
    result ends up being a view of the original ReadOnlyTensor; otherwise,
    the returned tensor is a regular `torch.Tensor`.
    """

    def __getattribute__(self, attribute_name: str) -> Any:
        if (
            isinstance(attribute_name, str)
            and attribute_name.endswith("_")
            and (not ((attribute_name.startswith("__")) and (attribute_name.endswith("__"))))
        ):
            raise AttributeError(
                f"A ReadOnlyTensor explicitly disables all members whose names end with '_'."
                f" Cannot access member {repr(attribute_name)}."
            )
        else:
            return super().__getattribute__(attribute_name)

    def __cannot_modify(self, *ignore, **ignore_too):
        raise TypeError("The contents of a ReadOnlyTensor cannot be modified")

    __setitem__ = __cannot_modify
    __iadd__ = __cannot_modify
    __iand__ = __cannot_modify
    __idiv__ = __cannot_modify
    __ifloordiv__ = __cannot_modify
    __ilshift__ = __cannot_modify
    __imatmul__ = __cannot_modify
    __imod__ = __cannot_modify
    __imul__ = __cannot_modify
    __ior__ = __cannot_modify
    __ipow__ = __cannot_modify
    __irshift__ = __cannot_modify
    __isub__ = __cannot_modify
    __itruediv__ = __cannot_modify
    __ixor__ = __cannot_modify

    if _torch_older_than_1_12:
        # Define __str__ and __repr__ for when using PyTorch 1.11 or older.
        # With PyTorch 1.12, overriding __str__ and __repr__ are not necessary.
        def __to_string(self) -> str:
            s = super().__repr__()
            if "\n" not in s:
                return f"ReadOnlyTensor({super().__repr__()})"
            else:
                indenter = " " * 4
                s = (indenter + s.replace("\n", "\n" + indenter)).rstrip()
                return f"ReadOnlyTensor(\n{s}\n)"

        __str__ = __to_string
        __repr__ = __to_string

    def clone(self, *, preserve_read_only: bool = False) -> torch.Tensor:
        result = super().clone()
        if not preserve_read_only:
            result = result.as_subclass(torch.Tensor)
        return result

    def __mutable_if_independent(self, other: torch.Tensor) -> torch.Tensor:
        from .misc import storage_ptr

        self_ptr = storage_ptr(self)
        other_ptr = storage_ptr(other)
        if self_ptr != other_ptr:
            other = other.as_subclass(torch.Tensor)
        return other

    def __getitem__(self, index_or_slice) -> torch.Tensor:
        result = super().__getitem__(index_or_slice)
        return self.__mutable_if_independent(result)

    def reshape(self, *args, **kwargs) -> torch.Tensor:
        result = super().reshape(*args, **kwargs)
        return self.__mutable_if_independent(result)

    def numpy(self) -> np.ndarray:
        arr: np.ndarray = torch.Tensor.numpy(self)
        arr.flags["WRITEABLE"] = False
        return arr

    def __array__(self, *args, **kwargs) -> np.ndarray:
        arr: np.ndarray = super().__array__(*args, **kwargs)
        arr.flags["WRITEABLE"] = False
        return arr

    def __copy__(self):
        return self.clone(preserve_read_only=True)

    def __deepcopy__(self, memo):
        return self.clone(preserve_read_only=True)

    @classmethod
    def __torch_function__(cls, func: Callable, types: Iterable, args: tuple = (), kwargs: Optional[Mapping] = None):
        if (kwargs is not None) and ("out" in kwargs):
            if isinstance(kwargs["out"], ReadOnlyTensor):
                raise TypeError(
                    f"The `out` keyword argument passed to {func} is a ReadOnlyTensor."
                    f" A ReadOnlyTensor explicitly fails when referenced via the `out` keyword argument of any torch"
                    f" function."
                    f" This restriction is for making sure that the torch operations which could normally do in-place"
                    f" modifications do not operate on ReadOnlyTensor instances."
                )
        return super().__torch_function__(func, types, args, kwargs)


def _device_and_dtype_kwargs(*, dtype: Optional[torch.dtype], device: Optional[Union[str, torch.device]]) -> dict:
    result = {}
    if dtype is not None:
        result["dtype"] = dtype
    if device is not None:
        result["device"] = device
    return result


def read_only_tensor(
    x: Any, *, dtype: Optional[torch.dtype] = None, device: Optional[Union[str, torch.device]] = None
) -> Iterable:
    """
    Make a ReadOnlyTensor from the given object.

    The provided object can be a scalar, or an Iterable of numeric data,
    or an ObjectArray.

    This function can be thought as the read-only counterpart of PyTorch's
    `torch.tensor(...)` function.

    Args:
        x: The object from which the new ReadOnlyTensor will be made.
        dtype: The dtype of the new ReadOnlyTensor (e.g. torch.float32).
        device: The device in which the ReadOnlyTensor will be stored
            (e.g. "cpu").
    Returns:
        The new read-only tensor.
    """
    from .objectarray import ObjectArray

    kwargs = _device_and_dtype_kwargs(dtype=dtype, device=device)
    if isinstance(x, ObjectArray):
        if len(kwargs) != 0:
            raise ValueError(
                f"read_only_tensor(...): when making a read-only tensor from an ObjectArray,"
                f" the arguments `dtype` and `device` were not expected."
                f" However, the received keyword arguments are: {kwargs}."
            )
        return x.get_read_only_view()
    else:
        return torch.as_tensor(x, **kwargs).as_subclass(ReadOnlyTensor)


def as_read_only_tensor(
    x: Any, *, dtype: Optional[torch.dtype] = None, device: Optional[Union[str, torch.device]] = None
) -> Iterable:
    """
    Convert the given object to a ReadOnlyTensor.

    The provided object can be a scalar, or an Iterable of numeric data,
    or an ObjectArray.

    This function can be thought as the read-only counterpart of PyTorch's
    `torch.as_tensor(...)` function.

    Args:
        x: The object to be converted to a ReadOnlyTensor.
        dtype: The dtype of the new ReadOnlyTensor (e.g. torch.float32).
            If this argument is not specified, dtype will be inferred from `x`.
            For example, if `x` is a PyTorch tensor or a numpy array, its
            existing dtype will be kept.
        device: The device in which the ReadOnlyTensor will be stored
            (e.g. "cpu").
            If this argument is not specified, the device which is storing
            the original `x` will be re-used.
    Returns:
        The read-only counterpart of the provided object.
    """
    from .objectarray import ObjectArray

    kwargs = _device_and_dtype_kwargs(dtype=dtype, device=device)
    if isinstance(x, ObjectArray):
        if len(kwargs) != 0:
            raise ValueError(
                f"read_only_tensor(...): when making a read-only tensor from an ObjectArray,"
                f" the arguments `dtype` and `device` were not expected."
                f" However, the received keyword arguments are: {kwargs}."
            )
        return x.get_read_only_view()
    else:
        return torch.as_tensor(x, **kwargs).as_subclass(ReadOnlyTensor)
