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
from copy import deepcopy
from numbers import Number
from typing import Any, Optional

import numpy as np
import torch


def deep_clone(  # noqa: C901
    x: Any,
    *,
    otherwise_deepcopy: bool = False,
    otherwise_return: bool = False,
    otherwise_fail: bool = False,
    memo: Optional[dict] = None,
) -> Any:
    """
    A recursive cloning function similar to the standard `deepcopy`.

    The difference between `deep_clone(...)` and `deepcopy(...)` is that
    `deep_clone(...)`, while recursively traversing, will run the `.clone()`
    method on the PyTorch tensors it encounters, so that the cloned tensors
    are forcefully detached from their storages (instead of cloning those
    storages as well).

    At the moment of writing this documentation, the current behavior of
    PyTorch tensors upon being deep-copied is to clone themselves AND their
    storages. Therefore, if a PyTorch tensor is a slice of a large tensor
    (which has a large storage), then the large storage will also be
    deep-copied, and the newly made clone of the tensor will point to a newly
    made large storage. One might instead prefer to clone tensors in such a
    way that the newly made tensor points to a newly made storage that
    contains just enough data for the tensor (with the unused data being
    dropped). When such a behavior is desired, one can use this
    `deep_clone(...)` function.

    Upon encountering a read-only and/or immutable data, this function will
    NOT modify the read-only behavior. For example, the deep-clone of a
    ReadOnlyTensor is still a ReadOnlyTensor, and the deep-clone of a
    read-only numpy array is still a read-only numpy array. Note that this
    behavior is different than the `clone()` method of a ReadOnlyTensor
    and the `copy()` method of a numpy array. The reason for this
    protective behavior is that since this is a deep-cloning operation,
    the encountered tensors and/or arrays might be the components of the root
    object, and changing their read-only attributes might affect the integrity
    of this root object.

    The `deep_clone(...)` function needs to know what to do when an object
    of unrecognized type is encountered. Therefore, the user is expected to
    set one of these arguments as True (and leave the others as False):
    `otherwise_deepcopy`, `otherwise_return`, `otherwise_fail`.

    Args:
        x: The object which will be deep-cloned. This object can be a standard
            Python container (i.e. list, tuple, dict, set), an instance of
            Problem, Solution, SolutionBatch, ObjectArray, ImmutableContainer,
            Clonable, and also any other type of object if either the argument
            `otherwise_deepcopy` or the argument `otherwise_return` is set as
            True.
        otherwise_deepcopy: Setting this as True means that, when an
            unrecognized object is encountered, that object will be
            deep-copied. To handle shared and cyclic-referencing objects,
            the `deep_clone(...)` function stores its own memo dictionary.
            When the control is given to the standard `deepcopy(...)`
            function, the memo dictionary of `deep_clone(...)` will be passed
            to `deepcopy`.
        otherwise_return: Setting this as True means that, when an
            unrecognized object is encountered, that object itself will be
            returned (i.e. will be a part of the created clone).
        otherwise_fail: Setting this as True means that, when an unrecognized
            object is encountered, a TypeError will be raised.
        memo: Optionally a dictionary. In most scenarios, when this function
            is called from outside, this is expected to be left as None.
    Returns:
        The newly made clone of the original object.
    """
    from .objectarray import ObjectArray
    from .readonlytensor import ReadOnlyTensor

    if memo is None:
        # If a memo dictionary was not given, make a new one now.
        memo = {}

    # Get the id of the object being cloned.
    x_id = id(x)

    if x_id in memo:
        # If the id of the object being cloned is already in the memo dictionary, then this object was previously
        # cloned. We just return that clone.
        return memo[x_id]

    # Count how many of the arguments `otherwise_deepcopy`, `otherwise_return`, and `otherwise_fail` was set as True.
    # In this context, we call these arguments as fallback behaviors.
    fallback_behaviors = (otherwise_deepcopy, otherwise_return, otherwise_fail)
    enabled_behavior_count = sum(1 for behavior in fallback_behaviors if behavior)

    if enabled_behavior_count == 0:
        # If none of the fallback behaviors was enabled, then we raise an error.
        raise ValueError(
            "The action to take with objects of unrecognized types is not known because"
            " none of these arguments was set as True: `otherwise_deepcopy`, `otherwise_return`, `otherwise_fail`."
            " Please set one of these arguments as True."
        )
    elif enabled_behavior_count == 1:
        # If one of the fallback behaviors was enabled, then we received our expected input. We do nothing here.
        pass
    else:
        # If the number of enabled fallback behaviors is an unexpected value. then we raise an error.
        raise ValueError(
            f"The following arguments were received, which is conflicting: otherwise_deepcopy={otherwise_deepcopy},"
            f" otherwise_return={otherwise_return}, otherwise_fail={otherwise_fail}."
            f" Please set exactly one of these arguments as True and leave the others as False."
        )

    # This inner function specifies how the deep_clone function should call itself.
    def call_self(obj: Any) -> Any:
        return deep_clone(
            obj,
            otherwise_deepcopy=otherwise_deepcopy,
            otherwise_return=otherwise_return,
            otherwise_fail=otherwise_fail,
            memo=memo,
        )

    # Below, we handle the cloning behaviors case by case.
    if (x is None) or (x is NotImplemented) or (x is Ellipsis):
        result = deepcopy(x)
    elif isinstance(x, (Number, str, bytes, bytearray)):
        result = deepcopy(x, memo=memo)
    elif isinstance(x, np.ndarray):
        result = x.copy()
        result.flags["WRITEABLE"] = x.flags["WRITEABLE"]
    elif isinstance(x, (ObjectArray, ReadOnlyClonable)):
        result = x.clone(preserve_read_only=True, memo=memo)
    elif isinstance(x, ReadOnlyTensor):
        result = x.clone(preserve_read_only=True)
    elif isinstance(x, torch.Tensor):
        result = x.clone()
    elif isinstance(x, Clonable):
        result = x.clone(memo=memo)
    elif isinstance(x, (dict, OrderedDict)):
        result = type(x)()
        memo[x_id] = result
        for k, v in x.items():
            result[call_self(k)] = call_self(v)
    elif isinstance(x, list):
        result = type(x)()
        memo[x_id] = result
        for item in x:
            result.append(call_self(item))
    elif isinstance(x, set):
        result = type(x)()
        memo[x_id] = result
        for item in x:
            result.add(call_self(item))
    elif isinstance(x, tuple):
        result = []
        memo[x_id] = result
        for item in x:
            result.append(call_self(item))
        if hasattr(x, "_fields"):
            result = type(x)(*result)
        else:
            result = type(x)(result)
        memo[x_id] = result
    else:
        # If the object is not recognized, we use the fallback behavior.
        if otherwise_deepcopy:
            result = deepcopy(x, memo=memo)
        elif otherwise_return:
            result = x
        elif otherwise_fail:
            raise TypeError(f"Do not know how to clone {repr(x)} (of type {type(x)}).")
        else:
            raise RuntimeError("The function `deep_clone` reached an unexpected state. This might be a bug.")

    if (x_id not in memo) and (result is not x):
        # If the newly made clone is still not in the memo dictionary AND the "clone" is not just a reference to the
        # original object, we make sure that it is in the memo dictionary.
        memo[x_id] = result

    # Finally, the result is returned.
    return result


class Clonable:
    """
    A base class allowing inheriting classes define how they should be cloned.

    Any class inheriting from Clonable gains these behaviors:
    (i) A new method named `.clone()` becomes available;
    (ii) `__deepcopy__` and `__copy__` work as aliases for `.clone()`;
    (iii) A new method, `_get_cloned_state(self, *, memo: dict)` is now
    defined and needs to be implemented by the inheriting class.

    The method `_get_cloned_state(...)` expects a dictionary named `memo`,
    which maps from the ids of already cloned objects to their clones.
    If `_get_cloned_state(...)` is to use `deep_clone(...)` or `deepcopy(...)`
    within itself, this `memo` dictionary can be passed to these functions.
    The return value of `_get_cloned_state(...)` is a dictionary, which will
    be used as the `__dict__` of the newly made clone.
    """

    def _get_cloned_state(self, *, memo: dict) -> dict:
        raise NotImplementedError

    def clone(self, *, memo: Optional[dict] = None) -> "Clonable":
        """
        Get a clone of this object.

        Args:
            memo: Optionally a dictionary which maps from the ids of the
                already cloned objects to their clones. In most scenarios,
                when this method is called from outside, this can be left
                as None.
        Returns:
            The clone of the object.
        """
        if memo is None:
            memo = {}

        self_id = id(self)
        if self_id in memo:
            return memo[self_id]

        new_object = object.__new__(type(self))
        memo[id(self)] = new_object

        new_object.__dict__.update(self._get_cloned_state(memo=memo))
        return new_object

    def __copy__(self) -> "Clonable":
        return self.clone()

    def __deepcopy__(self, memo: Optional[dict]):
        if memo is None:
            memo = {}
        return self.clone(memo=memo)


class Serializable(Clonable):
    """
    Base class allowing the inheriting classes become Clonable and picklable.

    Any class inheriting from `Serializable` becomes `Clonable` (since
    `Serializable` is a subclass of `Clonable`) and therefore is expected to
    define its own `_get_cloned_state(...)` (see the documentation of the
    class `Clonable` for details).

    A `Serializable` class gains a behavior for its `__getstate__`. In this
    already defined and implemented `__getstate__` method, the resulting
    dictionary of `_get_cloned_state(...)` is used as the state dictionary.
    Therefore, for `Serializable` objects, the behavior defined in their
    `_get_cloned_state(...)` methods affect how they are pickled.

    Classes inheriting from `Serializable` are `evotorch.Problem`,
    `evotorch.Solution`, `evotorch.SolutionBatch`, and
    `evotorch.distributions.Distribution`. In their `_get_cloned_state(...)`
    implementations, these classes use `deep_clone(...)` on themselves to make
    sure that their contained PyTorch tensors are copied using the `.clone()`
    method, ensuring that those tensors are detached from their old storages
    during the cloning operation. Thanks to being `Serializable`, their
    contained tensors are detached from their old storages both at the moment
    of copying/cloning AND at the moment of pickling.
    """

    def __getstate__(self) -> dict:
        memo = {id(self): self}
        return self._get_cloned_state(memo=memo)


class ReadOnlyClonable(Clonable):
    """
    Clonability base class for read-only and/or immutable objects.

    This is a base class specialized for the immutable containers of EvoTorch.
    These immutable containers have two behaviors for cloning:
    one where the read-only attribute is preserved and one where a mutable
    clone is created.

    Upon being copied or deep-copied (using the standard Python functions),
    the newly made clones are also read-only. However, when copied using the
    `clone(...)` method, the newly made clone is mutable by default
    (unless the `clone(...)` method was used with `preserve_read_only=True`).
    This default behavior of the `clone(...)` method was inspired by the
    `copy()` method of numpy arrays (the inspiration being that the `.copy()`
    of a read-only numpy array will not be read-only anymore).

    Subclasses of `evotorch.immutable.ImmutableContainer` inherit from
    `ReadOnlyClonable`.
    """

    def _get_mutable_clone(self, *, memo: dict) -> Any:
        raise NotImplementedError

    def clone(self, *, memo: Optional[dict] = None, preserve_read_only: bool = False) -> Any:
        """
        Get a clone of this read-only object.

        Args:
            memo: Optionally a dictionary which maps from the ids of the
                already cloned objects to their clones. In most scenarios,
                when this method is called from outside, this can be left
                as None.
            preserve_read_only: Whether or not to preserve the read-only
                behavior in the clone.
        Returns:
            The clone of the object.
        """
        if memo is None:
            memo = {}
        if preserve_read_only:
            return super().clone(memo=memo)
        else:
            return self._get_mutable_clone(memo=memo)

    def __copy__(self) -> Any:
        return self.clone(preserve_read_only=True)

    def __deepcopy__(self, memo: Optional[dict]) -> Any:
        if memo is None:
            memo = {}
        return self.clone(memo=memo, preserve_read_only=True)
