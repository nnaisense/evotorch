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
This module contains the ObjectArray class, which is an array-like
data structure with an interface similar to PyTorch tensors,
but with an ability to store arbitrary type of data (not just numbers).
"""

from collections.abc import Sequence
from typing import Any, Iterable, Optional

import numpy as np
import torch

from .misc import Device, DType, Size, clone, is_integer, is_sequence, storage_ptr
from .recursiveprintable import RecursivePrintable


class ObjectArrayStorage:
    def __init__(self, source: "ObjectArray"):
        self._source = source

    def data_ptr(self) -> int:
        return id(self._source._objects)


class ObjectArray(Sequence, RecursivePrintable):
    """
    An object container with an interface similar to PyTorch tensors.

    It is strictly one-dimensional, and supports advanced indexing and
    slicing operations supported by PyTorch tensors.

    An ObjectArray can store `None` values, strings, numbers, booleans,
    lists, sets, dictionaries, PyTorch tensors, and numpy arrays.

    When a container (such as a list, dictionary, set, is placed into an
    ObjectArray, an immutable clone of this container is first created, and
    then this newly created immutable clone gets stored within the
    ObjectArray. This behavior is to prevent accidental modification of the
    stored data.

    When a numeric array (such as a PyTorch tensor or a numpy array with a
    numeric dtype) is placed into an ObjectArray, the target ObjectArray
    first checks if the numeric array is read-only. If the numeric array
    is indeed read-only, then the array is put into the ObjectArray as it
    is. If the array is not read-only, then a read-only clone of the
    original numeric array is first created, and then this clone gets
    stored by the ObjectArray. This behavior has the following implications:
    (i) even when an ObjectArray is shared by multiple components of the
    program, the risk of accidental modification of the stored data through
    this shared ObjectArray is significantly reduced as the stored numeric
    arrays are read-only;
    (ii) although not recommended, one could still forcefully modify the
    numeric arrays stored by an ObjectArray by explicitly casting them as
    mutable arrays
    (in the case of a numpy array, one could forcefully set the WRITEABLE
    flag, and, in the case of a ReadOnlyTensor, one could forcefully cast it
    as a regular PyTorch tensor);
    (iii) if an already read-only array `x` is placed into an ObjectArray,
    but `x` shares its memory with a mutable array `y`, then the contents
    of the ObjectArray can be affected by modifying `y`.
    The implication (ii) is demonstrated as follows:

    ```python
    objs = ObjectArray(1)  # a single-element ObjectArray

    # Place a numpy array into objs:
    objs[0] = np.array([1, 2, 3], dtype=float)

    # At this point, objs[0] is a read-only numpy array.
    # objs[0] *= 2   # <- Not allowed

    # Possible but NOT recommended:
    objs.flags["WRITEABLE"] = True
    objs[0] *= 2
    ```

    The implication (iii) is demonstrated as follows:

    ```python
    objs = ObjectArray(1)  # a single-element ObjectArray

    # Make a new mutable numpy array
    y = np.array([1, 2, 3], dtype=float)

    # Make a read-only view to y:
    x = y[:]
    x.flags["WRITEABLE"] = False

    # Place x into objs.
    objs[0] = x

    # At this point, objs[0] is a read-only numpy array.
    # objs[0] *= 2   # <- Not allowed

    # During the operation of setting its 0-th item, the ObjectArray
    # `objs` did not clone `x` because `x` was already read-only.
    # However, the contents of `x` could actually be modified because
    # `x` shares its memory with the mutable array `y`.

    # Possible but NOT recommended:
    y *= 2  # This affects both x and objs!
    ```

    When a numpy array of dtype object is placed into an ObjectArray,
    a read-only ObjectArray copy of the original array will first be
    created, and then, this newly created ObjectArray will be stored
    by the outer ObjectArray.

    An ObjectArray itself has a read-only mode, so that, in addition to its
    stored data, the ObjectArray itself can be protected against undesired
    modifications.

    An interesting feature of PyTorch: if one slices a tensor A and the
    result is a new tensor B, and if B is sharing storage memory with A,
    then A.untyped_storage().data_ptr() and B.untyped_storage().data_ptr()
    will return the same pointer. This means, one can compare the storage
    pointers of A and B and see whether or not the two are sharing memory.
    ObjectArray was designed to have this exact behavior, so that one
    can understand if two ObjectArray instances are sharing memory.
    Note that NumPy does NOT have such a behavior. In more details,
    a NumPy array C and a NumPy array D could report different pointers
    even when D was created via a basic slicing operation on C.
    """

    def __init__(
        self,
        size: Optional[Size] = None,
        *,
        slice_of: Optional[tuple] = None,
    ):
        """
        `__init__(...)`: Instantiate a new ObjectArray.

        Args:
            size: Length of the ObjectArray. If this argument is present and
                is an integer `n`, then the resulting ObjectArray will be
                of length `n`, and will be filled with `None` values.
                This argument cannot be used together with the keyword
                argument `slice_of`.
            slice_of: Optionally a tuple in the form
                `(original_object_tensor, slice_info)`.
                When this argument is present, then the resulting ObjectArray
                will be a slice of the given `original_object_tensor` (which
                is expected as an ObjectArray instance). `slice_info` is
                either a `slice` instance, or a sequence of integers.
                The resulting ObjectArray might be a view of
                `original_object_tensor` (i.e. it might share its memory with
                `original_object_tensor`).
                This keyword argument cannot be used together with the
                argument `size`.
        """
        if size is not None and slice_of is not None:
            raise ValueError("Expected either `size` argument or `slice_of` argument, but got both.")
        elif size is None and slice_of is None:
            raise ValueError("Expected either `size` argument or `slice_of` argument, but got none.")
        elif size is not None:
            if not is_sequence(size):
                length = size
            elif isinstance(size, (np.ndarray, torch.Tensor)) and (size.ndim > 1):
                raise ValueError(f"Invalid size: {size}")
            else:
                [length] = size
            length = int(length)
            self._indices = torch.arange(length, dtype=torch.int64)
            self._objects = [None] * length
        elif slice_of is not None:
            source: ObjectArray

            source, slicing = slice_of

            if not isinstance(source, ObjectArray):
                raise TypeError(
                    f"`slice_of`: The first element was expected as an ObjectArray."
                    f" But it is of type {repr(type(source))}"
                )

            if isinstance(slicing, tuple) or is_integer(slicing):
                raise TypeError(f"Invalid slice: {slicing}")

            self._indices = source._indices[slicing]
            self._objects = source._objects

            if storage_ptr(self._indices) != storage_ptr(source._indices):
                self._objects = clone(self._objects)

        self._device = torch.device("cpu")
        self._read_only = False

    @property
    def shape(self) -> Size:
        """Shape of the ObjectArray, as a PyTorch Size tuple."""
        return self._indices.shape

    def size(self) -> Size:
        """
        Get the size of the ObjectArray, as a PyTorch Size tuple.

        Returns:
            The size (i.e. the shape) of the ObjectArray.
        """
        return self._indices.size()

    @property
    def ndim(self) -> int:
        """
        Number of dimensions handled by the ObjectArray.
        This is equivalent to getting the length of the size tuple.
        """
        return self._indices.ndim

    def dim(self) -> int:
        """
        Get the number of dimensions handled by the ObjectArray.
        This is equivalent to getting the length of the size tuple.

        Returns:
            The number of dimensions, as an integer.
        """
        return self._indices.dim()

    def numel(self) -> int:
        """
        Number of elements stored by the ObjectArray.

        Returns:
            The number of elements, as an integer.
        """
        return self._indices.numel()

    def repeat(self, *sizes) -> "ObjectArray":
        """
        Repeat the contents of this ObjectArray.

        For example, if we have an ObjectArray `objs` which stores
        `["hello", "world"]`, the following line:

            objs.repeat(3)

        will result in an ObjectArray which stores:

            `["hello", "world", "hello", "world", "hello", "world"]`

        Args:
            sizes: Although this argument is named `sizes` to be compatible
                with PyTorch, what is expected here is a single positional
                argument, as a single integer, or as a single-element
                tuple.
                The given integer (which can be the argument itself, or
                the integer within the given single-element tuple),
                specifies how many times the stored sequence will be
                repeated.
        Returns:
            A new ObjectArray which repeats the original one's values
        """

        if len(sizes) != 1:
            type_name = type(self).__name__
            raise ValueError(
                f"The `repeat(...)` method of {type_name} expects exactly one positional argument."
                f" This is because {type_name} supports only 1-dimensional storage."
                f" The received positional arguments are: {sizes}."
            )
        if isinstance(sizes, tuple):
            if len(sizes) == 1:
                sizes = sizes[0]
            else:
                type_name = type(self).__name__
                raise ValueError(
                    f"The `repeat(...)` method of {type_name} can accept a size tuple with only one element."
                    f" This is because {type_name} supports only 1-dimensional storage."
                    f" The received size tuple is: {sizes}."
                )
        num_repetitions = int(sizes[0])
        self_length = len(self)
        result = ObjectArray(num_repetitions * self_length)

        source_index = 0
        for result_index in range(len(result)):
            result[result_index] = self[source_index]
            source_index = (source_index + 1) % self_length

        return result

    @property
    def device(self) -> Device:
        """
        The device which stores the elements of the ObjectArray.
        In the case of ObjectArray, this property always returns
        the CPU device.

        Returns:
            The CPU device, as a torch.device object.
        """
        return self._device

    @property
    def dtype(self) -> DType:
        """
        The dtype of the elements stored by the ObjectArray.
        In the case of ObjectArray, the dtype is always `object`.
        """
        return object

    def __getitem__(self, i: Any) -> Any:
        if is_integer(i):
            index = int(self._indices[i])
            return self._objects[index]
        else:
            indices = self._indices[i]

            same_ptr = storage_ptr(indices) == storage_ptr(self._indices)

            result = ObjectArray(len(indices))

            if same_ptr:
                result._indices[:] = indices
                result._objects = self._objects
            else:
                result._objects = []
                for index in indices:
                    result._objects.append(self._objects[int(index)])

            result._read_only = self._read_only

            return result

    def __setitem__(self, i: Any, x: Any):
        self.set_item(i, x)

    def set_item(self, i: Any, x: Any, *, memo: Optional[dict] = None):
        """
        Set the i-th item of the ObjectArray as x.

        Args:
            i: An index or a slice.
            x: The object that will be put into the ObjectArray.
            memo: Optionally a dictionary which maps from the ids of the
                already placed objects to their clones within ObjectArray.
                In most scenarios, when this method is called from outside,
                this can be left as None.
        """
        from .immutable import as_immutable

        if memo is None:
            memo = {}

        memo[id(self)] = self

        if self._read_only:
            raise ValueError("This ObjectArray is read-only, therefore, modification is not allowed.")

        if is_integer(i):
            index = int(self._indices[i])
            self._objects[index] = as_immutable(x, memo=memo)
        else:
            indices = self._indices[i]
            if not isinstance(x, Iterable):
                raise TypeError(f"Expected an iterable, but got {repr(x)}")

            if indices.ndim != 1:
                raise ValueError(
                    "Received indices that would change the dimensionality of the ObjectArray."
                    " However, an ObjectArray can only be 1-dimensional."
                )

            slice_refers_to_whole_array = (len(indices) == len(self._indices)) and torch.all(indices == self._indices)
            if slice_refers_to_whole_array:
                memo[id(x)] = self

            if not hasattr(x, "__len__"):
                x = list(x)

            if len(x) != len(indices):
                raise TypeError(
                    f"The slicing operation refers to {len(indices)} elements."
                    f" However, the given objects sequence has {len(x)} elements."
                )

            for q, obj in enumerate(x):
                index = int(indices[q])
                self._objects[index] = as_immutable(obj, memo=memo)

    def __len__(self) -> int:
        return len(self._indices)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def clone(self, *, preserve_read_only: bool = False, memo: Optional[dict] = None) -> Iterable:
        """
        Get a deep copy of the ObjectArray.

        Args:
            preserve_read_only: Whether or not to preserve the read-only
                attribute. Note that the default value is False, which
                means that the newly made clone will NOT be read-only
                even if the original ObjectArray is.
            memo: Optionally a dictionary which maps from the ids of the
                already cloned objects to their clones.
                In most scenarios, when this method is called from outside,
                this can be left as None.
        Returns:
            The clone of the original ObjectArray.
        """
        from .cloning import deep_clone

        if memo is None:
            memo = {}

        self_id = id(self)
        if self_id in memo:
            return memo[self_id]

        if not preserve_read_only:
            return self.numpy(memo=memo)
        else:
            result = ObjectArray(len(self))
            memo[self_id] = result

            for i, item in enumerate(self):
                result[i] = deep_clone(item, otherwise_deepcopy=True, memo=memo)

            return result

    def __copy__(self) -> "ObjectArray":
        return self.clone(preserve_read_only=True)

    def __deepcopy__(self, memo: Optional[dict]) -> "ObjectArray":
        if memo is None:
            memo = {}
        return self.clone(preserve_read_only=True, memo=memo)

    def __setstate__(self, state: dict):
        self.__dict__.update(state)

        # After pickling and unpickling, numpy arrays become mutable.
        # Since we are dealing with immutable containers here, we need to forcefully make all numpy arrays read-only.
        for v in self:
            if isinstance(v, np.ndarray):
                v.flags["WRITEABLE"] = False

    # def __getstate__(self) -> dict:
    #     from .cloning import deep_clone
    #     self_id = id(self)
    #     memo = {self_id: self}
    #     cloned_dict = deep_clone(self.__dict__, otherwise_deepcopy=True, memo=memo)
    #     return cloned_dict

    def get_read_only_view(self) -> "ObjectArray":
        """
        Get a read-only view of this ObjectArray.
        """
        result = self[:]
        result._read_only = True
        return result

    @property
    def is_read_only(self) -> bool:
        """
        True if this ObjectArray is read-only; False otherwise.
        """
        return self._read_only

    def storage(self) -> ObjectArrayStorage:
        return ObjectArrayStorage(self)

    def untyped_storage(self) -> ObjectArrayStorage:
        return ObjectArrayStorage(self)

    def numpy(self, *, memo: Optional[dict] = None) -> np.ndarray:
        """
        Convert this ObjectArray to a numpy array.

        The resulting numpy array will have its dtype set as `object`.
        This new array itself and its contents will be mutable (those
        mutable objects being the copies of their immutable sources).

        Returns:
            The numpy counterpart of this ObjectArray.
        """
        from .immutable import mutable_copy

        if memo is None:
            memo = {}

        n = len(self)
        result = np.empty(n, dtype=object)

        memo[id(self)] = result

        for i, item in enumerate(self):
            result[i] = mutable_copy(item, memo=memo)

        return result

    @staticmethod
    def from_numpy(ndarray: np.ndarray) -> "ObjectArray":
        """
        Convert a numpy array of dtype `object` to an `ObjectArray`.

        Args:
            ndarray: The numpy array that will be converted to `ObjectArray`.
        Returns:
            The ObjectArray counterpart of the given numpy array.
        """
        if isinstance(ndarray, np.ndarray):
            if ndarray.dtype == np.dtype(object):
                n = len(ndarray)
                result = ObjectArray(n)
                for i, element in enumerate(ndarray):
                    result[i] = element
                return result
            else:
                raise ValueError(
                    f"The dtype of the given array was expected as `object`."
                    f" However, the dtype was encountered as {ndarray.dtype}."
                )
        else:
            raise TypeError(f"Expected a `numpy.ndarray` instance, but received an object of type {type(ndarray)}.")
