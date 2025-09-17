# Copyright 2025 NNAISENSE SA
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
Internal utility functions for operating on shallow (non-nested) containers.
"""


from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

from . import ObjectArray, TensorFrame

TensorLike = torch.Tensor | TensorFrame | ObjectArray


def _move_tensorframe(x: TensorFrame, *, device: str | torch.device, move_only_from_cpu: bool = False) -> TensorFrame:
    """
    Move a TensorFrame to the specified device.

    Args:
        x: The TensorFrame to be moved to the specified device.
        device: The target device.
        move_only_from_cpu: If True, only columns that are on cpu will be moved.
    Returns:
        The new TensorFrame with its columns moved.
    """

    if move_only_from_cpu:
        # This is the case where we only move the columns that are on the cpu.
        device_to_be_enforced = x.has_enforced_device
        original_is_read_only = x.is_read_only

        new_devices = set()  # the set which is to store the devices after the moving operation
        coldata = {}  # the dictionary which is to store the new columns
        for colname in x.columns:
            # Move each column on the cpu to the target device
            coldata[colname] = _move(x[colname], device=device, move_only_from_cpu=move_only_from_cpu)
            # Add the new column's device into the set of new devices
            new_devices.add(coldata[colname].device)

        device_kwarg = {}
        if device_to_be_enforced and (len(new_devices) == 1):
            # If the original TensorFrame has an enforced device, and the number of new devices is 1,
            # then we declare this new device as the enforced device of the new TensorFrame.
            device_kwarg["device"] = device

        # Prepare the new TensorFrame and return it.
        return TensorFrame(coldata, read_only=original_is_read_only, **device_kwarg)
    else:
        # This is the case where we move all columns of the TensorFrame.
        device_to_be_enforced = x.has_enforced_device
        x = x.to(device=device)
        if not device_to_be_enforced:
            x = x.without_enforced_device()
        return x


def _move(x: TensorLike, *, device: str | torch.device, move_only_from_cpu: bool = False) -> TensorLike:
    """
    Move a tensor or a TensorFrame or an ObjectArray to the target device.

    Note that an ObjectArray cannot really be moved to any device other
    than the cpu, so, any instance of ObjectArray is simply returned
    as it is.

    Args:
        x: A tensor, or a TensorFrame, or an ObjectArray subject to moving.
        device: The target device.
        move_only_from_cpu: If True, the tensor, or the columns of the given
            TensorFrame will be moved only if they currently reside on the cpu.
            If False, all the tensor data will be moved to the target device.
    Returns:
        The counterpart of the original tensor/array after the moving.
    """
    if isinstance(x, TensorFrame):
        return _move_tensorframe(x, device=device, move_only_from_cpu=move_only_from_cpu)
    if isinstance(x, ObjectArray):
        return x

    if move_only_from_cpu and (x.device != torch.device("cpu")):
        # If we are to move the tensor only when it is on the cpu, and also we observe that it is NOT on the cpu,
        # then we simply return the tensor as it is, without moving it.
        return x

    return x.to(device=device)


def move_shallow_container_to_device(
    x: TensorLike | Sequence[TensorLike] | Mapping[Any, TensorLike],
    *,
    device: str | torch.device,
    move_only_from_cpu: bool = False,
) -> TensorLike | Sequence[TensorLike] | Mapping[Any, TensorLike]:
    """
    Move a tensor or a shallow container of tensors to the given device.

    Args:
        x: A tensor or a TensorFrame or an ObjectArray, or a dictionary-like
            object or a sequence of tensors/arrays. Any encountered tensors
            and `TensorFrame`s within `x` will be moved to the given `device`.
            Any encountered ObjectArray within `x` will be put back as it is
            (without raising any error), since an ObjectArray can reside only
            on the cpu.
        device: The target device that can be given as an instance of `str`
            or `torch.device`.
        move_only_from_cpu: If this is given as True, then the tensors/arrays
            will be moved to the target device ONLY if they currently reside
            on the cpu.
    Returns:
        The counterpart of `x` that resides on the given device.
    """

    def move(obj: TensorLike) -> TensorLike:
        return _move(obj, device=device, move_only_from_cpu=move_only_from_cpu)

    if isinstance(x, (torch.Tensor, TensorFrame, ObjectArray)):
        result = move(x)
    elif isinstance(x, (str, np.str_, bytes, bytearray)):
        raise TypeError(f"Cannot move an object of type {type(x)} to the device {device}")
    elif isinstance(x, Mapping):
        result = {}
        for k, v in x.items():
            if isinstance(v, (torch.Tensor, TensorFrame, ObjectArray)):
                result[k] = move(v)
            else:
                raise TypeError(
                    "While trying to move the tensors within a dictionary-like object,"
                    f" encountered an element of this unexpected type: {type(v)}"
                )
    elif isinstance(x, Sequence):
        result = []
        for item in x:
            if isinstance(item, (torch.Tensor, TensorFrame, ObjectArray)):
                result.append(move(item))
            else:
                raise TypeError(
                    "While trying to move the tensors within a sequence,"
                    f" encountered an element of this unexpected type: {type(item)}"
                )
    else:
        raise TypeError(f"Cannot move an object of type {type(x)} to the device {device}")

    return result


def _update_dict_additively(left: dict, right: dict):
    """
    Additively update the left dict using the items of the right dict.

    In more details, if a key within the right dictionary does not exist within
    the left dictionary, that item is put into the left dictionary. On the
    other hand, if a key within the right dictionary does exist within the left
    dictionary, the value within the right dictionary is added (using the `+=`
    operator) onto the value of the left dictionary.

    This function returns nothing, and the update is done in-place.

    Args:
        left: The dictionary to be updated.
        right: The dictionary whose values will be used to update the left
            dictionary.
    """
    for k, v in right.items():
        if k in left:
            left[k] += v
        else:
            left[k] = v


def count_devices_within_shallow_container(
    x: TensorLike | Mapping[Any, TensorLike] | Sequence[TensorLike],
    *,
    _already_within_container: bool = False,
) -> dict[torch.device, float]:
    """
    Given a shallow (non-nested) container of tensors, count devices in it.

    The returned object is a dictionary in which each key is a `torch.device`,
    and each value represents how many times a device is encountered.

    Args:
        x: A shallow container in which the devices are to be counted.
        _already_within_container: For internal usage.
    Returns:
        The dictionary which stores the device counts.
    """

    devices = {}

    if isinstance(x, (torch.Tensor, ObjectArray)):
        _update_dict_additively(devices, {x.device: 1.0})
    elif isinstance(x, TensorFrame):
        for col in x.columns:
            _update_dict_additively(devices, {x[col].device: 1.0})
    elif isinstance(x, (Sequence, Mapping)):
        if _already_within_container:
            raise TypeError("Nested containers are not supported")
        if isinstance(x, (str, np.str_, bytes, bytearray)):
            raise TypeError(f"Unsupported type: {type(x)}")
        if isinstance(x, tuple) and hasattr(x, "_fields"):
            raise TypeError("Named tuples are not supported")
        if isinstance(x, Mapping):
            values_of_x = x.values()
        else:
            values_of_x = x
        for v in values_of_x:
            _update_dict_additively(
                devices,
                count_devices_within_shallow_container(
                    v,
                    _already_within_container=True,
                ),
            )
    else:
        raise TypeError(f"Encountered an object of this unexpected type: {type(x)}")

    return devices


def most_favored_device_among_arguments(
    args: Sequence[TensorLike | Mapping[Any, TensorLike] | Sequence[TensorLike]],
    *,
    slightly_favor_cpu: bool = True,
) -> torch.device:
    """
    Given arguments in a tuple, find the most favored PyTorch device.

    It is expected that the arguments consist of PyTorch tensors or
    `TensorFrame`s or `ObjectArray`s, or shallow (non-nested) sequences or
    dictionary-like containers containing tensors and/or `TensorFrame`s
    and/or `ObjectArray`s.

    Args:
        x: A tuple of arguments.
        slightly_favor_cpu: If this is given as True, and if there are
            multiple devices that are encountered equally, and if cpu
            is one of those devices, then cpu will be picked.
    Returns:
        The most favored torch.device.
    """

    weights = {}

    if not isinstance(args, Sequence):
        raise TypeError(f"`args` was received as an instance of this unexpected type: {type(args)}")
    if isinstance(args, tuple) and hasattr(args, "_fields"):
        raise TypeError("Providing `args` as a named tuple is not supported")

    for arg in args:
        _update_dict_additively(
            weights,
            count_devices_within_shallow_container(arg),
        )

    if slightly_favor_cpu:
        _update_dict_additively(weights, {"cpu": 0.1})

    device_with_max_weight = torch.device("cpu")
    max_weight = 0.0
    for d, w in weights.items():
        if w > max_weight:
            device_with_max_weight = d
            max_weight = w

    return device_with_max_weight
