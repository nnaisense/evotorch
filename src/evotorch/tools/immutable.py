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

from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence, Set
from numbers import Number
from typing import Any, Optional, Union

import numpy as np
import torch

from .cloning import ReadOnlyClonable, deep_clone
from .recursiveprintable import RecursivePrintable


def _is_basic_data(x: Any) -> bool:
    return (x is None) or isinstance(x, (str, Number))


def _numpy_array_stores_objects(x: np.ndarray) -> bool:
    return x.dtype == np.dtype(object)


def _is_numpy_array_immutable(x: np.ndarray) -> bool:
    return (not _numpy_array_stores_objects(x)) and (not x.flags["WRITEABLE"])


def is_immutable_container_or_tensor(x: Any) -> bool:
    from .objectarray import ObjectArray
    from .readonlytensor import ReadOnlyTensor

    return (
        isinstance(x, (ImmutableContainer, ReadOnlyTensor))
        or (isinstance(x, np.ndarray) and _is_numpy_array_immutable(x))
        or (isinstance(x, ObjectArray) and x.is_read_only)
    )


def as_immutable(x: Any, *, memo: Optional[dict] = None) -> Any:
    from .objectarray import ObjectArray
    from .readonlytensor import ReadOnlyTensor

    if memo is None:
        memo = {}

    x_id = id(x)
    if x_id in memo:
        return memo[x_id]

    put_into_memo = True

    if is_immutable_container_or_tensor(x) or _is_basic_data(x):
        put_into_memo = False
        result = x
    elif isinstance(x, torch.Tensor):
        result = x.clone().as_subclass(ReadOnlyTensor)
    elif isinstance(x, ObjectArray):
        result = x.clone().get_read_only_view()
    elif isinstance(x, np.ndarray):
        if _numpy_array_stores_objects(x):
            result = ObjectArray(len(x))
            for i, element in enumerate(x):
                result[i] = element
            result = result.get_read_only_view()
        else:
            result = x.copy()
            result.flags["WRITEABLE"] = False
            result = result
    elif isinstance(x, Mapping):
        result = ImmutableDict(x, memo)
    elif isinstance(x, set):
        result = ImmutableSet(x, memo=memo)
    elif isinstance(x, Iterable):
        result = ImmutableList(x, memo=memo)
    else:
        raise TypeError(
            f"as_immutable(...) encountered an object of unsupported type."
            f" Encountered object: {repr(x)}."
            f" Type of the encountered object: {repr(type(x))}."
        )

    if put_into_memo:
        memo[x_id] = result

    return result


def mutable_copy(x: Any, *, memo: Optional[dict] = None) -> Any:
    from .objectarray import ObjectArray
    from .readonlytensor import ReadOnlyTensor

    if memo is None:
        memo = {}

    x_id = id(x)
    if x_id in memo:
        return memo[x_id]

    if _is_basic_data(x):
        result = x
    elif isinstance(x, (ReadOnlyTensor, ObjectArray)):
        result = x.clone(preserve_read_only=False)
    elif isinstance(x, np.ndarray):
        result = x.copy()
    elif isinstance(x, torch.Tensor):
        result = x.clone()
    elif isinstance(x, ImmutableContainer):
        result = x.clone(memo=memo, preserve_read_only=False)
    else:
        raise TypeError(f"Encountered an object of unrecognized type. The object is {repr(x)}. Its type is {type(x)}")

    if (x_id not in memo) and (result is not x):
        memo[x_id] = result

    return result


class ImmutableContainer(ReadOnlyClonable, RecursivePrintable):
    def _get_cloned_state(self, *, memo: dict) -> dict:
        return deep_clone(self.__dict__, otherwise_deepcopy=True, memo=memo)

    def __setstate__(self, state: dict):
        self.__dict__.update(state)

        # After pickling and unpickling, numpy arrays become mutable.
        # Since we are dealing with immutable containers here, we need to forcefully make all numpy arrays read-only.
        if isinstance(self, Mapping):
            all_values = self.values()
        elif isinstance(self, Sequence):
            all_values = self
        else:
            raise NotImplementedError

        for v in all_values:
            if isinstance(v, np.ndarray):
                v.flags["WRITEABLE"] = False


class ImmutableSequence(ImmutableContainer, Sequence):
    def __init__(self, x: Iterable, *, memo: Optional[dict] = None):
        if memo is None:
            memo = {}
        memo[id(x)] = self
        self.__data: list = [as_immutable(item, memo=memo) for item in x]

    def __len__(self) -> int:
        return len(self.__data)

    def __getitem__(self, i):
        return self.__data[i]

    def __add__(self, other: Iterable) -> "ImmutableSequence":
        cls = type(self)
        return cls(list(self) + list(other))

    def __radd__(self, other: Iterable) -> "ImmutableSequence":
        cls = type(self)
        return cls(list(other) + list(self))

    def __eq__(self, other: Iterable) -> bool:
        self_length = len(self)
        self_last = self_length - 1
        other_count = 0
        for i, other_item in enumerate(other):
            other_count += 1
            if i > self_last:
                return False
            if self[i] != other_item:
                return False
        return other_count == self_length

    def __ne__(self, other: Iterable) -> bool:
        return not self.__eq__(other)


class ImmutableList(ImmutableSequence):
    def _get_mutable_clone(self, *, memo: dict) -> list:
        result = []
        memo[id(self)] = result
        for x in self:
            result.append(mutable_copy(x, memo=memo))
        return result


class ImmutableSet(ImmutableContainer, Set):
    def __init__(self, x: Iterable, *, memo: Optional[dict] = None):
        if memo is None:
            memo = {}
        memo[id(x)] = self
        self.__data: set = set([as_immutable(item, memo=memo) for item in x])

    def __contains__(self, x: Any) -> bool:
        return as_immutable(x) in self.__data

    def __iter__(self):
        for item in self.__data:
            yield item

    def __len__(self) -> int:
        return len(self.__data)

    def _get_mutable_clone(self, *, memo: dict) -> set:
        result = set()
        memo[id(self)] = result
        for x in self:
            result.add(mutable_copy(x, memo=memo))
        return result


def _acceptable_key(x: Any) -> Union[int, str]:
    from .misc import is_integer

    if is_integer(x):
        return int(x)
    elif isinstance(x, str):
        return x
    elif isinstance(x, (tuple, ImmutableSequence)):
        return tuple([_acceptable_key(k) for k in x])
    else:
        raise TypeError(
            f"The object to be used as a key within an `ImmutableDict` must"
            f" be of one of these types: `int`, `str`, `tuple` (where the"
            f" tuple can only contain `int`, `str` and/or `tuple` instances)."
            f" The received key candidate does not seem to satisfy this"
            f" constraint."
            f" The unnaccepted key candidate is: {repr(x)}."
            f" The type of the unnaccepted key is: {repr(type(x))}."
        )


class ImmutableDict(ImmutableContainer, Mapping):
    def __init__(self, *args, **kwargs):
        if len(args) == 1:
            x = args[0]
            memo = {}
        elif len(args) == 2:
            x, memo = args
            if memo is None:
                memo = {}
        else:
            raise TypeError("Wrong number of positional arguments. Expected 1 or 2 arguments.")

        memo[id(x)] = self

        self.__data: OrderedDict = OrderedDict()
        iterator = x.items() if isinstance(x, Mapping) else x
        for k, v in iterator:
            self.__data[_acceptable_key(k)] = as_immutable(v, memo=memo)
        for k, v in kwargs.items():
            self.__data[str(k)] = as_immutable(v, memo=memo)

    def __getitem__(self, k):
        k = _acceptable_key(k)
        return self.__data[k]

    def __iter__(self):
        for k in self.__data.keys():
            yield k

    def __len__(self) -> int:
        return len(self.__data)

    def _get_mutable_clone(self, *, memo: dict) -> dict:
        result = {}
        memo[id(self)] = result
        for k, v in self.items():
            k = mutable_copy(k, memo=memo)
            v = mutable_copy(v, memo=memo)
            result[k] = v
        return result
