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
from typing import Any, Union

import numpy as np
import torch


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


def as_immutable(x: Any) -> Any:
    from .objectarray import ObjectArray
    from .readonlytensor import ReadOnlyTensor

    if is_immutable_container_or_tensor(x) or _is_basic_data(x):
        return x
    elif isinstance(x, torch.Tensor):
        return x.clone().as_subclass(ReadOnlyTensor)
    elif isinstance(x, ObjectArray):
        return x.clone().get_read_only_view()
    elif isinstance(x, np.ndarray):
        if _numpy_array_stores_objects(x):
            result = ObjectArray(len(x))
            for i, element in enumerate(x):
                result[i] = element
            return result.get_read_only_view()
        else:
            result = x.copy()
            result.flags["WRITEABLE"] = False
            return result
    elif isinstance(x, Mapping):
        return ImmutableDict(x)
    elif isinstance(x, tuple):
        return ImmutableTuple(x)
    elif isinstance(x, set):
        return ImmutableSet(x)
    elif isinstance(x, Iterable):
        return ImmutableList(x)
    else:
        raise TypeError(
            f"as_immutable(...) encountered an object of unsupported type."
            f" Encountered object: {repr(x)}."
            f" Type of the encountered object: {repr(type(x))}."
        )


def mutable_copy(x: Any) -> Any:
    from .objectarray import ObjectArray
    from .readonlytensor import ReadOnlyTensor

    if _is_basic_data(x):
        return x
    elif is_immutable_container_or_tensor(x):
        if isinstance(x, ImmutableContainer):
            return x.clone()
        elif isinstance(x, np.ndarray):
            return x.copy()
        else:
            return x.clone()
    else:
        raise TypeError(
            f"mutable_copy(...) encountered an object of unsupported type."
            f" Encountered object: {repr(x)}."
            f" Type of the encountered object: {repr(type(x))}."
        )


class ImmutableContainer:
    def clone(self) -> Iterable:
        raise NotImplementedError


class ImmutableSequence(ImmutableContainer, Sequence):
    def __init__(self, x: Iterable):
        self.__data: list = [as_immutable(item) for item in x]

    def __len__(self) -> int:
        return len(self.__data)

    def __getitem__(self, i):
        return self.__data[i]

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.__data})"

    def __add__(self, other: Iterable) -> "ImmutableSequence":
        cls = type(self)
        return cls(list(self) + list(other))

    def __radd__(self, other: Iterable) -> "ImmutableSequence":
        cls = type(self)
        return cls(list(other) + list(self))

    def clone(self) -> Iterable:
        pass

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
    def clone(self) -> list:
        return [mutable_copy(x) for x in self]


class ImmutableTuple(ImmutableSequence):
    def clone(self) -> tuple:
        return tuple([mutable_copy(x) for x in self])


class ImmutableSet(ImmutableContainer, Set):
    def __init__(self, x: Iterable):
        self.__data: set = set([as_immutable(item) for item in x])

    def __contains__(self, x: Any) -> bool:
        return as_immutable(x) in self.__data

    def __iter__(self):
        for item in self.__data:
            yield item

    def __len__(self) -> int:
        return len(self.__data)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.__data})"

    def clone(self) -> Iterable:
        return set([mutable_copy(x) for x in self])


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
    def __init__(self, x: Iterable, **kwargs):
        self.__data: OrderedDict = OrderedDict()
        iterator = x.items() if isinstance(x, Mapping) else x
        for k, v in iterator:
            self.__data[_acceptable_key(k)] = as_immutable(v)
        for k, v in kwargs.items():
            self.__data[str(k)] = as_immutable(v)

    def __getitem__(self, k):
        k = _acceptable_key(k)
        return self.__data[k]

    def __iter__(self):
        for k in self.__data.keys():
            yield k

    def __len__(self) -> int:
        return len(self.__data)

    def __repr__(self) -> str:
        type_name = type(self).__name__
        contents = ", ".join([repr(k) + ": " + repr(v) for (k, v) in self.__data.items()])
        return type_name + "({" + contents + "})"

    def clone(self) -> "ImmutableDict":
        result = {}
        for k, v in self.items():
            k = mutable_copy(k)
            v = mutable_copy(v)
            result[k] = v
        return result
