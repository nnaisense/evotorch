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

# flake8: noqa: C901

"""
This namespace contains data structures whose underlying storages are
contiguous and therefore vectorization-friendly.
"""

from collections import namedtuple
from numbers import Number
from typing import Any, Iterable, Optional, Union

import torch

from .misc import Device, DType, to_torch_dtype

Numbers = Union[Number, Iterable[Number]]


def do_where(mask: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.ndim == 0 and b.ndim > 0:
        a = a.expand(b.shape)
    elif a.ndim > 0 and b.ndim == 0:
        b = b.expand(a.shape)

    if a.shape != b.shape:
        raise RuntimeError(
            f"The shapes of the tensors `a` and `b` mismatch."
            f" The shape of `a` is {a.shape}. The shape of `b` is {b.shape}."
        )
    if mask.ndim > a.ndim:
        raise RuntimeError(
            f" Number of dimensions of the boolean mask cannot be more than the number of dimensions of `a` and `b`."
            f" The shape of the boolean mask: {mask.shape}."
            f" The shape of the tensors: {a.shape}."
        )
    if mask.shape != a.shape[: mask.ndim]:
        raise RuntimeError(
            f"The leftmost parts of the given tensors mismatch the shape of the boolean mask."
            f" The shape of the boolean mask: {mask.shape}."
            f" The shape of the tensors: {a.shape}."
        )
    mask = mask.reshape(mask.shape + tuple((1 for _ in range(a.ndim - mask.ndim))))
    return torch.where(mask, a, b)


class CMemory:
    """
    Representation of a batchable contiguous memory.

    This container can be seen as a batchable primitive dictionary where the
    keys are allowed either as integers or as tuples of integers. Please also
    note that, a memory block for each key is already allocated, meaning that
    unlike a dictionary of Python, each key already exists and is associated
    with a tensor.

    Let us consider an example where we have 5 keys, and each key is associated
    with a tensor of length 7. Such a memory could be allocated like this:

    ```python
    memory = CMemory(7, num_keys=5)
    ```

    Our allocated memory can be visualized as follows:

    ```text
     _______________________________________
    | key 0 -> [ empty tensor of length 7 ] |
    | key 1 -> [ empty tensor of length 7 ] |
    | key 2 -> [ empty tensor of length 7 ] |
    | key 3 -> [ empty tensor of length 7 ] |
    | key 4 -> [ empty tensor of length 7 ] |
    |_______________________________________|
    ```

    Let us now sample a Gaussian noise and put it into the 0-th slot:

    ```python
    memory[0] = torch.randn(7)  # or: memory[torch.tensor(0)] = torch.randn(7)
    ```

    which results in:

    ```text
     _________________________________________
    | key 0 -> [ Gaussian noise of length 7 ] |
    | key 1 -> [ empty tensor of length 7   ] |
    | key 2 -> [ empty tensor of length 7   ] |
    | key 3 -> [ empty tensor of length 7   ] |
    | key 4 -> [ empty tensor of length 7   ] |
    |_________________________________________|
    ```

    Let us now consider another example where we deal with not a single CMemory,
    but with a CMemory batch. For the sake of this example, let us say that our
    desired batch size is 3. The allocation of such a batch would be as
    follows:

    ```python
    memory_batch = CMemory(7, num_keys=5, batch_size=3)
    ```

    Our memory batch can be visualized like this:

    ```text
     __[ batch item 0 ]_____________________
    | key 0 -> [ empty tensor of length 7 ] |
    | key 1 -> [ empty tensor of length 7 ] |
    | key 2 -> [ empty tensor of length 7 ] |
    | key 3 -> [ empty tensor of length 7 ] |
    | key 4 -> [ empty tensor of length 7 ] |
    |_______________________________________|

     __[ batch item 1 ]_____________________
    | key 0 -> [ empty tensor of length 7 ] |
    | key 1 -> [ empty tensor of length 7 ] |
    | key 2 -> [ empty tensor of length 7 ] |
    | key 3 -> [ empty tensor of length 7 ] |
    | key 4 -> [ empty tensor of length 7 ] |
    |_______________________________________|

     __[ batch item 2 ]_____________________
    | key 0 -> [ empty tensor of length 7 ] |
    | key 1 -> [ empty tensor of length 7 ] |
    | key 2 -> [ empty tensor of length 7 ] |
    | key 3 -> [ empty tensor of length 7 ] |
    | key 4 -> [ empty tensor of length 7 ] |
    |_______________________________________|
    ```

    If we wish to set the 0-th element of each batch item, we could do:

    ```python
    memory_batch[0] = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
        ],
    )
    ```

    and the result would be:

    ```text
     __[ batch item 0 ]_____________________
    | key 0 -> [ 0. 0. 0. 0. 0. 0. 0.     ] |
    | key 1 -> [ empty tensor of length 7 ] |
    | key 2 -> [ empty tensor of length 7 ] |
    | key 3 -> [ empty tensor of length 7 ] |
    | key 4 -> [ empty tensor of length 7 ] |
    |_______________________________________|

     __[ batch item 1 ]_____________________
    | key 0 -> [ 1. 1. 1. 1. 1. 1. 1.     ] |
    | key 1 -> [ empty tensor of length 7 ] |
    | key 2 -> [ empty tensor of length 7 ] |
    | key 3 -> [ empty tensor of length 7 ] |
    | key 4 -> [ empty tensor of length 7 ] |
    |_______________________________________|

     __[ batch item 2 ]_____________________
    | key 0 -> [ 2. 2. 2. 2. 2. 2. 2.     ] |
    | key 1 -> [ empty tensor of length 7 ] |
    | key 2 -> [ empty tensor of length 7 ] |
    | key 3 -> [ empty tensor of length 7 ] |
    | key 4 -> [ empty tensor of length 7 ] |
    |_______________________________________|
    ```

    Continuing from the same example, if we wish to set the slot with key 1
    in the 0th batch item, slot with key 2 in the 1st batch item, and
    slot with key 3 in the 2nd batch item, all in one go, we could do:

    ```python
    # Longer version: memory_batch[torch.tensor([1, 2, 3])] = ...

    memory_batch[[1, 2, 3]] = torch.tensor(
        [
            [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
            [6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0],
            [7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0],
        ],
    )
    ```

    Our updated memory batch would then look like this:

    ```text
     __[ batch item 0 ]_____________________
    | key 0 -> [ 0. 0. 0. 0. 0. 0. 0.     ] |
    | key 1 -> [ 5. 5. 5. 5. 5. 5. 5.     ] |
    | key 2 -> [ empty tensor of length 7 ] |
    | key 3 -> [ empty tensor of length 7 ] |
    | key 4 -> [ empty tensor of length 7 ] |
    |_______________________________________|

     __[ batch item 1 ]_____________________
    | key 0 -> [ 1. 1. 1. 1. 1. 1. 1.     ] |
    | key 1 -> [ empty tensor of length 7 ] |
    | key 2 -> [ 6. 6. 6. 6. 6. 6. 6.     ] |
    | key 3 -> [ empty tensor of length 7 ] |
    | key 4 -> [ empty tensor of length 7 ] |
    |_______________________________________|

     __[ batch item 2 ]_____________________
    | key 0 -> [ 2. 2. 2. 2. 2. 2. 2.     ] |
    | key 1 -> [ empty tensor of length 7 ] |
    | key 2 -> [ empty tensor of length 7 ] |
    | key 3 -> [ 7. 7. 7. 7. 7. 7. 7.     ] |
    | key 4 -> [ empty tensor of length 7 ] |
    |_______________________________________|
    ```

    Conditional modifications via boolean masks is also supported.
    For example, the following update on our `memory_batch`:

    ```python
    memory_batch.set_(
        [4, 3, 1],
        torch.tensor(
            [
                [8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0],
                [9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0],
                [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
            ]
        ),
        where=[True, True, False],  # or: where=torch.tensor([True,True,False]),
    )
    ```

    would result in:

    ```text
     __[ batch item 0 ]_____________________
    | key 0 -> [ 0. 0. 0. 0. 0. 0. 0.     ] |
    | key 1 -> [ 5. 5. 5. 5. 5. 5. 5.     ] |
    | key 2 -> [ empty tensor of length 7 ] |
    | key 3 -> [ empty tensor of length 7 ] |
    | key 4 -> [ 8. 8. 8. 8. 8. 8. 8.     ] |
    |_______________________________________|

     __[ batch item 1 ]_____________________
    | key 0 -> [ 1. 1. 1. 1. 1. 1. 1.     ] |
    | key 1 -> [ empty tensor of length 7 ] |
    | key 2 -> [ 6. 6. 6. 6. 6. 6. 6.     ] |
    | key 3 -> [ 9. 9. 9. 9. 9. 9. 9.     ] |
    | key 4 -> [ empty tensor of length 7 ] |
    |_______________________________________|

     __[ batch item 2 ]_____________________
    | key 0 -> [ 2. 2. 2. 2. 2. 2. 2.     ] |
    | key 1 -> [ empty tensor of length 7 ] |
    | key 2 -> [ empty tensor of length 7 ] |
    | key 3 -> [ 7. 7. 7. 7. 7. 7. 7.     ] |
    | key 4 -> [ empty tensor of length 7 ] |
    |_______________________________________|
    ```

    Please notice above that the slot with key 1 of the batch item 2 was not
    modified because its corresponding mask value was given as False.
    """

    def __init__(
        self,
        *size: Union[int, tuple, list],
        num_keys: Union[int, tuple, list],
        key_offset: Optional[Union[int, tuple, list]] = None,
        batch_size: Optional[Union[int, tuple, list]] = None,
        batch_shape: Optional[Union[int, tuple, list]] = None,
        fill_with: Optional[Numbers] = None,
        dtype: Optional[DType] = None,
        device: Optional[Device] = None,
        verify: bool = True,
    ):
        """
        `__init__(...)`: Initialize the CMemory.

        Args:
            size: Size of a tensor associated with a key, expected as an
                integer, or as multiple positional arguments (each positional
                argument being an integer), or as a tuple of integers.
            num_keys: How many keys (and therefore how many slots) will the
                memory have. If given as an integer `n`, then there will be `n`
                slots in the memory, and to access a slot one will need to use
                an integer key `k` (where, by default, the minimum acceptable
                `k` is 0 and the maximum acceptable `k` is `n-1`).
                If given as a tuple of integers, then the number of slots in
                the memory will be computed as the product of all the integers
                in the tuple, and a key will be expected as a tuple.
                For example, when `num_keys` is `(3, 5)`, there will be 15
                slots in the memory (where, by default, the minimum acceptable
                key will be `(0, 0)` and the maximum acceptable key will be
                `(2, 4)`.
            key_offset: Optionally can be used to shift the integer values of
                the keys. For example, if `num_keys` is 10, then, by default,
                the minimum key is 0 and the maximum key is 9. But together
                with `num_keys=10`, if `key_offset` is given as 1, then the
                minimum key will be 1 and the maximum key will be 10.
                This argument can also be used together with a tuple-valued
                `num_keys`. For example, with `num_keys` set as `(3, 5)`,
                if `key_offset` is given as 1, then the minimum key value
                will be `(1, 1)` (instead of `(0, 0)`) and the maximum key
                value will be `(3, 5)` (instead of `(2, 4)`).
                Also, with a tuple-valued `num_keys`, `key_offset` can be
                given as a tuple, to shift the key values differently for each
                item in the tuple.
            batch_size: If given as None, then this memory will not be batched.
                If given as an integer `n`, then this object will represent
                a contiguous batch containing `n` memory blocks.
                If given as a tuple `(size0, size1, ...)`, then this object
                will represent a contiguous batch of memory, shape of this
                batch being determined by the given tuple.
            batch_shape: Alias for the argument `batch_size`.
            fill_with: Optionally a numeric value using which the values will
                be initialized. If no initialization is needed, then this
                argument can be left as None.
            dtype: The `dtype` of the memory tensor.
            device: The device on which the memory will be allocated.
            verify: If True, then explicit checks will be done to verify
                that there are no indexing errors. Can be set as False for
                performance.
        """
        self._dtype = torch.float32 if dtype is None else to_torch_dtype(dtype)
        self._device = torch.device("cpu") if device is None else torch.device(device)
        self._verify = bool(verify)

        if isinstance(num_keys, (list, tuple)):
            if len(num_keys) < 2:
                raise RuntimeError(
                    f"When expressed via a list or a tuple, the length of `num_keys` must be at least 2."
                    f" However, the encountered `num_keys` is {repr(num_keys)}, whose length is {len(num_keys)}."
                )
            self._multi_key = True
            self._num_keys = tuple((int(n) for n in num_keys))
            self._internal_key_shape = torch.Size(self._num_keys)
        else:
            self._multi_key = False
            self._num_keys = int(num_keys)
            self._internal_key_shape = torch.Size([self._num_keys])
        self._internal_key_ndim = len(self._internal_key_shape)

        if key_offset is None:
            self._key_offset = None
        else:
            if self._multi_key:
                if isinstance(key_offset, (list, tuple)):
                    key_offset = [int(n) for n in key_offset]
                    if len(key_offset) != len(self._num_keys):
                        raise RuntimeError("The length of `key_offset` does not match the length of `num_keys`")
                else:
                    key_offset = [int(key_offset) for _ in range(len(self._num_keys))]
                self._key_offset = torch.as_tensor(key_offset, dtype=torch.int64, device=self._device)
            else:
                if isinstance(key_offset, (list, tuple)):
                    raise RuntimeError("`key_offset` cannot be a sequence of integers when `num_keys` is a scalar")
                else:
                    self._key_offset = torch.as_tensor(int(key_offset), dtype=torch.int64, device=self._device)

        if self._verify:
            if self._multi_key:
                self._min_key = torch.zeros(len(self._num_keys), dtype=torch.int64, device=self._device)
                self._max_key = torch.tensor(list(self._num_keys), dtype=torch.int64, device=self._device) - 1
            else:
                self._min_key = torch.tensor(0, dtype=torch.int64, device=self._device)
                self._max_key = torch.tensor(self._num_keys - 1, dtype=torch.int64, device=self._device)
            if self._key_offset is not None:
                self._min_key += self._key_offset
                self._max_key += self._key_offset
        else:
            self._min_key = None
            self._max_key = None

        nsize = len(size)
        if nsize == 0:
            self._value_shape = torch.Size([])
        elif nsize == 1:
            if isinstance(size[0], (tuple, list)):
                self._value_shape = torch.Size((int(n) for n in size[0]))
            else:
                self._value_shape = torch.Size([int(size[0])])
        else:
            self._value_shape = torch.Size((int(n) for n in size))
        self._value_ndim = len(self._value_shape)

        if (batch_size is None) and (batch_shape is None):
            batch_size = None
        elif (batch_size is not None) and (batch_shape is None):
            pass
        elif (batch_size is None) and (batch_shape is not None):
            batch_size = batch_shape
        else:
            raise RuntimeError(
                "Encountered both `batch_shape` and `batch_size` at the same time."
                " None of them or one of them can be accepted, but not both of them at the same time."
            )

        if batch_size is None:
            self._batch_shape = torch.Size([])
        elif isinstance(batch_size, (tuple, list)):
            self._batch_shape = torch.Size((int(n) for n in batch_size))
        else:
            self._batch_shape = torch.Size([int(batch_size)])
        self._batch_ndim = len(self._batch_shape)

        self._for_all_batches = tuple(
            (
                torch.arange(self._batch_shape[i], dtype=torch.int64, device=self._device)
                for i in range(self._batch_ndim)
            )
        )

        self._data = torch.empty(
            self._batch_shape + self._internal_key_shape + self._value_shape,
            dtype=(self._dtype),
            device=(self._device),
        )

        if fill_with is not None:
            self._data[:] = fill_with

    @property
    def _is_dtype_bool(self) -> bool:
        return self._data.dtype is torch.bool

    def _key_must_be_valid(self, key: torch.Tensor) -> torch.Tensor:
        lb_satisfied = key >= self._min_key
        ub_satisfied = key <= self._max_key
        all_satisfied = lb_satisfied & ub_satisfied
        if not torch.all(all_satisfied):
            raise KeyError("Encountered invalid key(s)")

    def _get_key(self, key: Numbers, where: Optional[torch.Tensor] = None) -> torch.Tensor:
        key = torch.as_tensor(key, dtype=torch.int64, device=self._data.device)
        expected_shape = self.batch_shape + self.key_shape
        if key.shape == expected_shape:
            result = key
        elif key.shape == self.key_shape:
            result = key.expand(expected_shape)
        else:
            raise RuntimeError(f"The key tensor has an incompatible shape: {key.shape}")
        if where is not None:
            min_key = (
                torch.tensor(0, dtype=torch.int64, device=self._data.device) if self._min_key is None else self._min_key
            )
            key = do_where(where, key, min_key.expand(expected_shape))
        if self._verify:
            self._key_must_be_valid(key)

        return result

    def _get_value(self, value: Numbers) -> torch.Tensor:
        value = torch.as_tensor(value, dtype=self._data.dtype, device=self._data.device)
        expected_shape = self.batch_shape + self.value_shape
        if value.shape == expected_shape:
            return value
        elif (value.ndim == 0) or (value.shape == self.value_shape):
            return value.expand(expected_shape)
        else:
            raise RuntimeError(f"The value tensor has an incompatible shape: {value.shape}")
        return value

    def _get_where(self, where: Numbers) -> torch.Tensor:
        where = torch.as_tensor(where, dtype=torch.bool, device=self._data.device)
        if where.shape != self.batch_shape:
            raise RuntimeError(
                f"The boolean mask `where` has an incompatible shape: {where.shape}."
                f" Acceptable shape is: {self.batch_shape}"
            )
        return where

    def prepare_key_tensor(self, key: Numbers) -> torch.Tensor:
        """
        Return the tensor-counterpart of a key.

        Args:
            key: A key which can be a sequence of integers or a PyTorch tensor
                with an integer dtype.
                The shape of the given key must conform with the `key_shape`
                of this memory object.
                To address to a different key in each batch item, the shape of
                the given key can also have extra leftmost dimensions expressed
                by `batch_shape`.
        Returns:
            A copy of the key that is converted to PyTorch tensor.
        """
        return self._get_key(key)

    def prepare_value_tensor(self, value: Numbers) -> torch.Tensor:
        """
        Return the tensor-counterpart of a value.

        Args:
            value: A value that can be a numeric sequence or a PyTorch tensor.
                The shape of the given value must conform with the
                `value_shape` of this memory object.
                To express a different value for each batch item, the shape of
                the given value can also have extra leftmost dimensions
                expressed by `value_shape`.
        Returns:
            A copy of the given value(s), converted to PyTorch tensor.
        """
        return self._get_value(value)

    def prepare_where_tensor(self, where: Numbers) -> torch.Tensor:
        """
        Return the tensor-counterpart of a boolean mask.

        Args:
            where: A boolean mask expressed as a sequence of bools or as a
                boolean PyTorch tensor.
                The shape of the given mask must conform with the batch shape
                that is expressed by the property `batch_shape`.
        Returns:
            A copy of the boolean mask, converted to PyTorch tensor.
        """
        return self._get_where(where)

    def _get_address(self, key: Numbers, where: Optional[torch.Tensor] = None) -> tuple:
        key = self._get_key(key, where=where)
        if self._key_offset is not None:
            key = key - self._key_offset
        if self._multi_key:
            keys = tuple((key[..., j] for j in range(self._internal_key_ndim)))
            return self._for_all_batches + keys
        else:
            return self._for_all_batches + (key,)

    def get(self, key: Numbers) -> torch.Tensor:
        """
        Get the value(s) associated with the given key(s).

        Args:
            key: A single key, or multiple keys (where the leftmost dimension
                of the given keys conform with the `batch_shape`).
        Returns:
            The value(s) associated with the given key(s).
        """
        address = self._get_address(key)
        return self._data[address]

    def set_(self, key: Numbers, value: Numbers, where: Optional[Numbers] = None):
        """
        Set the value(s) associated with the given key(s).

        Args:
            key: A single key, or multiple keys (where the leftmost dimension
                of the given keys conform with the `batch_shape`).
            value: The new value(s).
            where: Optionally a boolean mask whose shape matches `batch_shape`.
                If a `where` mask is given, then modifications will happen only
                on the memory slots whose corresponding mask values are True.
        """
        where = None if where is None else self._get_where(where)
        address = self._get_address(key, where=where)
        value = self._get_value(value)

        if where is None:
            self._data[address] = value
        else:
            old_value = self._data[address]
            new_value = value
            self._data[address] = do_where(where, new_value, old_value)

    def add_(self, key: Numbers, value: Numbers, where: Optional[Numbers] = None):
        """
        Add value(s) onto the existing values of slots with the given key(s).

        Args:
            key: A single key, or multiple keys (where the leftmost dimension
                of the given keys conform with the `batch_shape`).
            value: The value(s) that will be added onto the existing value(s).
            where: Optionally a boolean mask whose shape matches `batch_shape`.
                If a `where` mask is given, then modifications will happen only
                on the memory slots whose corresponding mask values are True.
        """
        where = None if where is None else self._get_where(where)
        address = self._get_address(key, where=where)
        value = self._get_value(value)

        if where is None:
            if self._is_dtype_bool:
                self._data[address] |= value
            else:
                self._data[address] += value
        else:
            if self._is_dtype_bool:
                mask_shape = self._batch_shape + tuple((1 for _ in range(self._value_ndim)))
                self._data[address] |= value & where.reshape(mask_shape)
            else:
                self._data[address] += do_where(where, value, torch.tensor(0, dtype=value.dtype, device=value.device))

    def add_circular_(self, key: Numbers, value: Numbers, mod: Numbers, where: Optional[Numbers] = None):
        """
        Increase the values of the specified slots in a circular manner.

        This operation combines the add and modulo operations.
        Circularly adding `value` onto `x` with a modulo `mod` means:
        `x = (x + value) % mod`.

        Args:
            key: A single key, or multiple keys (where the leftmost dimension
                of the given keys conform with the `batch_shape`).
            value: The value(s) that will be added onto the existing value(s).
            mod: After the raw adding operation, the modulos according to this
                `mod` argument will be computed and placed.
            where: Optionally a boolean mask whose shape matches `batch_shape`.
                If a `where` mask is given, then modifications will happen only
                on the memory slots whose corresponding mask values are True.
        """
        where = None if where is None else self._get_where(where)
        address = self._get_address(key, where=where)
        value = self._get_value(value)
        mod = self._get_value(mod)

        if self._is_dtype_bool:
            raise ValueError("Circular addition is not supported for dtype `torch.bool`")

        if where is None:
            self._data[address] = (self._data[address] + value) % mod
        else:
            old_value = self._data[address]
            new_value = (old_value + value) % mod
            self._data[address] = do_where(where, new_value, old_value)

    def multiply_(self, key: Numbers, value: Numbers, where: Optional[Numbers] = None):
        """
        Multiply the existing values of slots with the given key(s).

        Args:
            key: A single key, or multiple keys (where the leftmost dimension
                of the given keys conform with the `batch_shape`).
            value: The value(s) that will be used as the multiplier(s).
            where: Optionally a boolean mask whose shape matches `batch_shape`.
                If a `where` mask is given, then modifications will happen only
                on the memory slots whose corresponding mask values are True.
        """
        where = None if where is None else self._get_where(where)
        address = self._get_address(key, where=where)
        value = self._get_value(value)

        if where is None:
            if self._is_dtype_bool:
                self._data[address] &= value
            else:
                self._data[address] += value
        else:
            if self._is_dtype_bool:
                self._data[address] &= do_where(
                    where, value, torch.tensor(True, dtype=value.dtype, device=value.device)
                )
            else:
                self._data[address] *= do_where(where, value, torch.tensor(1, dtype=value.dtype, device=value.device))

    def subtract_(self, key: Numbers, value: Numbers, where: Optional[Numbers] = None):
        """
        Subtract value(s) from existing values of slots with the given key(s).

        Args:
            key: A single key, or multiple keys (where the leftmost dimension
                of the given keys conform with the `batch_shape`).
            value: The value(s) that will be subtracted from existing value(s).
            where: Optionally a boolean mask whose shape matches `batch_shape`.
                If a `where` mask is given, then modifications will happen only
                on the memory slots whose corresponding mask values are True.
        """
        self.add_(key, -value, where)

    def divide_(self, key: Numbers, value: Numbers, where: Optional[Numbers] = None):
        """
        Divide the existing values of slots with the given key(s).

        Args:
            key: A single key, or multiple keys (where the leftmost dimension
                of the given keys conform with the `batch_shape`).
            value: The value(s) that will be used as divisor(s).
            where: Optionally a boolean mask whose shape matches `batch_shape`.
                If a `where` mask is given, then modifications will happen only
                on the memory slots whose corresponding mask values are True.
        """
        self.multiply_(key, 1 / value, where)

    def __getitem__(self, key: Numbers) -> torch.Tensor:
        """
        Get the value(s) associated with the given key(s).

        Args:
            key: A single key, or multiple keys (where the leftmost dimension
                of the given keys conform with the `batch_shape`).
        Returns:
            The value(s) associated with the given key(s).
        """
        return self.get(key)

    def __setitem__(self, key: Numbers, value: Numbers):
        """
        Set the value(s) associated with the given key(s).

        Args:
            key: A single key, or multiple keys (where the leftmost dimension
                of the given keys conform with the `batch_shape`).
            value: The new value(s).
        """
        self.set_(key, value)

    @property
    def data(self) -> torch.Tensor:
        """
        The entire value tensor
        """
        return self._data

    @property
    def key_shape(self) -> torch.Size:
        """
        Shape of a key
        """
        return torch.Size([self._internal_key_ndim]) if self._multi_key else torch.Size([])

    @property
    def key_ndim(self) -> int:
        """
        Number of dimensions of a key
        """
        return 1 if self._multi_key else 0

    @property
    def batch_shape(self) -> torch.Size:
        """
        Batch size of this memory object
        """
        return self._batch_shape

    @property
    def batch_ndim(self) -> int:
        """
        Number of dimensions expressed by `batch_shape`
        """
        return self._batch_ndim

    @property
    def is_batched(self) -> bool:
        """
        True if this CMemory object is batched; False otherwise.
        """
        return self._batch_ndim > 0

    @property
    def value_shape(self) -> torch.Size:
        """
        Tensor shape of a single value
        """
        return self._value_shape

    @property
    def value_ndim(self) -> int:
        """
        Number of dimensions expressed by `value_shape`
        """
        return self._value_ndim

    @property
    def dtype(self) -> torch.dtype:
        """
        `dtype` of the value tensor
        """
        return self._data.dtype

    @property
    def device(self) -> torch.device:
        """
        The device on which this memory object lives
        """
        return self._data.device


class Structure:
    """
    A mixin class for vectorized structures.

    This mixin class assumes that the inheriting structure has a protected
    attribute `_data` which is either a `CMemory` object or another
    `Structure`. With this assumption, this mixin class provides certain
    methods and properties to bring a unified interface for all vectorized
    structures provided in this namespace.
    """

    _data: Union[CMemory, "Structure"]

    @property
    def value_shape(self) -> torch.Size:
        """
        Shape of a single value
        """
        return self._data.value_shape

    @property
    def value_ndim(self) -> int:
        """
        Number of dimensions expressed by `value_shape`
        """
        return self._data.value_ndim

    @property
    def batch_shape(self) -> torch.Size:
        """
        Batch size of this structure
        """
        return self._data.batch_shape

    @property
    def batch_ndim(self) -> int:
        """
        Number of dimensions expressed by `batch_shape`
        """
        return self._data.batch_ndim

    @property
    def is_batched(self) -> bool:
        """
        True if this structure is batched; False otherwise.
        """
        return self._batch_ndim > 0

    @property
    def dtype(self) -> torch.dtype:
        """
        `dtype` of the values
        """
        return self._data.dtype

    @property
    def device(self) -> torch.device:
        """
        The device on which this structure lives
        """
        return self._data.device

    def prepare_value_tensor(self, value: Numbers) -> torch.Tensor:
        """
        Return the tensor-counterpart of a value.

        Args:
            value: A value that can be a numeric sequence or a PyTorch tensor.
                The shape of the given value must conform with the
                `value_shape` of this memory object.
                To express a different value for each batch item, the shape of
                the given value can also have extra leftmost dimensions
                expressed by `value_shape`.
        Returns:
            A copy of the given value(s), converted to PyTorch tensor.
        """
        return self._data.prepare_value_tensor(value)

    def prepare_where_tensor(self, where: Numbers) -> torch.Tensor:
        """
        Return the tensor-counterpart of a boolean mask.

        Args:
            where: A boolean mask expressed as a sequence of bools or as a
                boolean PyTorch tensor.
                The shape of the given mask must conform with the batch shape
                that is expressed by the property `batch_shape`.
        Returns:
            A copy of the boolean mask, converted to PyTorch tensor.
        """
        return self._data.prepare_where_tensor(where)

    def _get_value(self, value: Numbers) -> torch.Tensor:
        return self._data.prepare_value_tensor(value)

    def _get_where(self, where: Numbers) -> torch.Tensor:
        return self._data.prepare_where_tensor(where)

    def __contains__(self, x: Any) -> torch.Tensor:
        raise TypeError("This structure does not support the `in` operator")


class CDict(Structure):
    """
    Representation of a batchable dictionary.

    This structure is very similar to a `CMemory`, but with the additional
    behavior of separately keeping track of which keys exist and which keys
    do not exist.

    Let us consider an example where we have 5 keys, and each key is associated
    with a tensor of length 7. Such a dictionary could be allocated like this:

    ```python
    dictnry = CDict(7, num_keys=5)
    ```

    Our allocated dictionary can be visualized as follows:

    ```text
     _______________________________________
    | key 0 -> ( missing )                  |
    | key 1 -> ( missing )                  |
    | key 2 -> ( missing )                  |
    | key 3 -> ( missing )                  |
    | key 4 -> ( missing )                  |
    |_______________________________________|
    ```

    Let us now sample a Gaussian noise and put it into the 0-th slot:

    ```python
    dictnry[0] = torch.randn(7)  # or: dictnry[torch.tensor(0)] = torch.randn(7)
    ```

    which results in:

    ```text
     _________________________________________
    | key 0 -> [ Gaussian noise of length 7 ] |
    | key 1 -> ( missing )                    |
    | key 2 -> ( missing )                    |
    | key 3 -> ( missing )                    |
    | key 4 -> ( missing )                    |
    |_________________________________________|
    ```

    Let us now consider another example where we deal with not a single
    dictionary but with a dictionary batch. For the sake of this example, let
    us say that our desired batch size is 3. The allocation of such a batch
    would be as follows:

    ```python
    dict_batch = CDict(7, num_keys=5, batch_size=3)
    ```

    Our dictionary batch can be visualized like this:

    ```text
     __[ batch item 0 ]_____________________
    | key 0 -> ( missing )                  |
    | key 1 -> ( missing )                  |
    | key 2 -> ( missing )                  |
    | key 3 -> ( missing )                  |
    | key 4 -> ( missing )                  |
    |_______________________________________|

     __[ batch item 1 ]_____________________
    | key 0 -> ( missing )                  |
    | key 1 -> ( missing )                  |
    | key 2 -> ( missing )                  |
    | key 3 -> ( missing )                  |
    | key 4 -> ( missing )                  |
    |_______________________________________|

     __[ batch item 2 ]_____________________
    | key 0 -> ( missing )                  |
    | key 1 -> ( missing )                  |
    | key 2 -> ( missing )                  |
    | key 3 -> ( missing )                  |
    | key 4 -> ( missing )                  |
    |_______________________________________|
    ```

    If we wish to set the 0-th element of each batch item, we could do:

    ```python
    dict_batch[0] = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
        ],
    )
    ```

    and the result would be:

    ```text
     __[ batch item 0 ]_____________________
    | key 0 -> [ 0. 0. 0. 0. 0. 0. 0.     ] |
    | key 1 -> ( missing )                  |
    | key 2 -> ( missing )                  |
    | key 3 -> ( missing )                  |
    | key 4 -> ( missing )                  |
    |_______________________________________|

     __[ batch item 1 ]_____________________
    | key 0 -> [ 1. 1. 1. 1. 1. 1. 1.     ] |
    | key 1 -> ( missing )                  |
    | key 2 -> ( missing )                  |
    | key 3 -> ( missing )                  |
    | key 4 -> ( missing )                  |
    |_______________________________________|

     __[ batch item 2 ]_____________________
    | key 0 -> [ 2. 2. 2. 2. 2. 2. 2.     ] |
    | key 1 -> ( missing )                  |
    | key 2 -> ( missing )                  |
    | key 3 -> ( missing )                  |
    | key 4 -> ( missing )                  |
    |_______________________________________|
    ```

    Continuing from the same example, if we wish to set the slot with key 1
    in the 0th batch item, slot with key 2 in the 1st batch item, and
    slot with key 3 in the 2nd batch item, all in one go, we could do:

    ```python
    # Longer version: dict_batch[torch.tensor([1, 2, 3])] = ...

    dict_batch[[1, 2, 3]] = torch.tensor(
        [
            [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
            [6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0],
            [7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0],
        ],
    )
    ```

    Our updated dictionary batch would then look like this:

    ```text
     __[ batch item 0 ]_____________________
    | key 0 -> [ 0. 0. 0. 0. 0. 0. 0.     ] |
    | key 1 -> [ 5. 5. 5. 5. 5. 5. 5.     ] |
    | key 2 -> ( missing )                  |
    | key 3 -> ( missing )                  |
    | key 4 -> ( missing )                  |
    |_______________________________________|

     __[ batch item 1 ]_____________________
    | key 0 -> [ 1. 1. 1. 1. 1. 1. 1.     ] |
    | key 1 -> ( missing )                  |
    | key 2 -> [ 6. 6. 6. 6. 6. 6. 6.     ] |
    | key 3 -> ( missing )                  |
    | key 4 -> ( missing )                  |
    |_______________________________________|

     __[ batch item 2 ]_____________________
    | key 0 -> [ 2. 2. 2. 2. 2. 2. 2.     ] |
    | key 1 -> ( missing )                  |
    | key 2 -> ( missing )                  |
    | key 3 -> [ 7. 7. 7. 7. 7. 7. 7.     ] |
    | key 4 -> ( missing )                  |
    |_______________________________________|
    ```

    Conditional modifications via boolean masks is also supported.
    For example, the following update on our `dict_batch`:

    ```python
    dict_batch.set_(
        [4, 3, 1],
        torch.tensor(
            [
                [8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0],
                [9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0],
                [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
            ]
        ),
        where=[True, True, False],  # or: where=torch.tensor([True,True,False]),
    )
    ```

    would result in:

    ```text
     __[ batch item 0 ]_____________________
    | key 0 -> [ 0. 0. 0. 0. 0. 0. 0.     ] |
    | key 1 -> [ 5. 5. 5. 5. 5. 5. 5.     ] |
    | key 2 -> ( missing )                  |
    | key 3 -> ( missing )                  |
    | key 4 -> [ 8. 8. 8. 8. 8. 8. 8.     ] |
    |_______________________________________|

     __[ batch item 1 ]_____________________
    | key 0 -> [ 1. 1. 1. 1. 1. 1. 1.     ] |
    | key 1 -> ( missing )                  |
    | key 2 -> [ 6. 6. 6. 6. 6. 6. 6.     ] |
    | key 3 -> [ 9. 9. 9. 9. 9. 9. 9.     ] |
    | key 4 -> ( missing )                  |
    |_______________________________________|

     __[ batch item 2 ]_____________________
    | key 0 -> [ 2. 2. 2. 2. 2. 2. 2.     ] |
    | key 1 -> ( missing )                  |
    | key 2 -> ( missing )                  |
    | key 3 -> [ 7. 7. 7. 7. 7. 7. 7.     ] |
    | key 4 -> ( missing )                  |
    |_______________________________________|
    ```

    Please notice above that the slot with key 1 of the batch item 2 was not
    modified because its corresponding mask value was given as False.

    After all these modifications, querying whether or not an element with
    key 0 would give us the following output:

    ```text
    >>> dict_batch.contains(0)
    torch.tensor([True, True, True], dtype=torch.bool)
    ```

    which means that, for each dictionary within the batch, an element with
    key 0 exists. The same query for the key 3 would give us:

    ```text
    >>> dict_batch.contains(3)
    torch.tensor([False, True, True], dtype=torch.bool)
    ```

    which means that the 0-th dictionary within the batch does not have an
    element with key 3, but the dictionaries 1 and 2 do have their elements
    with that key.
    """

    def __init__(
        self,
        *size: Union[int, tuple, list],
        num_keys: Union[int, tuple, list],
        key_offset: Optional[Union[int, tuple, list]] = None,
        batch_size: Optional[Union[int, tuple, list]] = None,
        batch_shape: Optional[Union[int, tuple, list]] = None,
        dtype: Optional[DType] = None,
        device: Optional[Device] = None,
        verify: bool = True,
    ):
        """
        `__init__(...)`: Initialize the CDict.

        Args:
            size: Size of a tensor associated with a key, expected as an
                integer, or as multiple positional arguments (each positional
                argument being an integer), or as a tuple of integers.
            num_keys: How many keys (and therefore how many slots) can the
                dictionary have. If given as an integer `n`, then there will be
                `n` slots available in the dictionary, and to access a slot one
                will need to use an integer key `k` (where, by default, the
                minimum acceptable `k` is 0 and the maximum acceptable `k` is
                `n-1`). If given as a tuple of integers, then the number of slots
                available in the dictionary will be computed as the product of
                all the integers in the tuple, and a key will be expected as a
                tuple. For example, when `num_keys` is `(3, 5)`, there will be
                15 slots available in the dictionary (where, by default, the
                minimum acceptable key will be `(0, 0)` and the maximum
                acceptable key will be `(2, 4)`.
            key_offset: Optionally can be used to shift the integer values of
                the keys. For example, if `num_keys` is 10, then, by default,
                the minimum key is 0 and the maximum key is 9. But together
                with `num_keys=10`, if `key_offset` is given as 1, then the
                minimum key will be 1 and the maximum key will be 10.
                This argument can also be used together with a tuple-valued
                `num_keys`. For example, with `num_keys` set as `(3, 5)`,
                if `key_offset` is given as 1, then the minimum key value
                will be `(1, 1)` (instead of `(0, 0)`) and the maximum key
                value will be `(3, 5)` (instead of `(2, 4)`).
                Also, with a tuple-valued `num_keys`, `key_offset` can be
                given as a tuple, to shift the key values differently for each
                item in the tuple.
            batch_size: If given as None, then this dictionary will not be
                batched. If given as an integer `n`, then this object will
                represent a contiguous batch containing `n` dictionary blocks.
                If given as a tuple `(size0, size1, ...)`, then this object
                will represent a contiguous batch of dictionary, shape of this
                batch being determined by the given tuple.
            batch_shape: Alias for the argument `batch_size`.
            fill_with: Optionally a numeric value using which the values will
                be initialized. If no initialization is needed, then this
                argument can be left as None.
            dtype: The `dtype` of the values stored by this CDict.
            device: The device on which the dictionary will be allocated.
            verify: If True, then explicit checks will be done to verify
                that there are no indexing errors. Can be set as False for
                performance.
        """
        self._data = CMemory(
            *size,
            num_keys=num_keys,
            key_offset=key_offset,
            batch_size=batch_size,
            batch_shape=batch_shape,
            dtype=dtype,
            device=device,
            verify=verify,
        )

        self._exist = CMemory(
            num_keys=num_keys,
            key_offset=key_offset,
            batch_size=batch_size,
            batch_shape=batch_shape,
            dtype=torch.bool,
            device=device,
            verify=verify,
        )

    def get(self, key: Numbers, default: Optional[Numbers] = None) -> torch.Tensor:
        """
        Get the value(s) associated with the given key(s).

        Args:
            key: A single key, or multiple keys (where the leftmost dimension
                of the given keys conform with the `batch_shape`).
            default: Optionally can be specified as the fallback value for when
                the element(s) with the given key(s) do not exist.
        Returns:
            The value(s) associated with the given key(s).
        """
        if default is None:
            return self._data[key]
        else:
            exist = self._exist[key]
            default = self._get_value(default)
            return do_where(exist, self._data[key], default)

    def set_(self, key: Numbers, value: Numbers, where: Optional[Numbers] = None):
        """
        Set the value(s) associated with the given key(s).

        Args:
            key: A single key, or multiple keys (where the leftmost dimension
                of the given keys conform with the `batch_shape`).
            value: The new value(s).
            where: Optionally a boolean mask whose shape matches `batch_shape`.
                If a `where` mask is given, then modifications will happen only
                on the memory slots whose corresponding mask values are True.
        """
        self._data.set_(key, value, where)
        self._exist.set_(key, True, where)

    def add_(self, key: Numbers, value: Numbers, where: Optional[Numbers] = None):
        """
        Add value(s) onto the existing values of slots with the given key(s).

        Note that this operation does not change the existence flags of the
        keys. In other words, if element(s) with `key` do not exist, then
        they will still be flagged as non-existent after this operation.

        Args:
            key: A single key, or multiple keys (where the leftmost dimension
                of the given keys conform with the `batch_shape`).
            value: The value(s) that will be added onto the existing value(s).
            where: Optionally a boolean mask whose shape matches `batch_shape`.
                If a `where` mask is given, then modifications will happen only
                on the memory slots whose corresponding mask values are True.
        """
        self._data.add_(key, value, where)

    def subtract_(self, key: Numbers, value: Numbers, where: Optional[Numbers] = None):
        """
        Subtract value(s) from existing values of slots with the given key(s).

        Note that this operation does not change the existence flags of the
        keys. In other words, if element(s) with `key` do not exist, then
        they will still be flagged as non-existent after this operation.

        Args:
            key: A single key, or multiple keys (where the leftmost dimension
                of the given keys conform with the `batch_shape`).
            value: The value(s) that will be subtracted from existing value(s).
            where: Optionally a boolean mask whose shape matches `batch_shape`.
                If a `where` mask is given, then modifications will happen only
                on the memory slots whose corresponding mask values are True.
        """
        self._data.subtract_(key, value, where)

    def divide_(self, key: Numbers, value: Numbers, where: Optional[Numbers] = None):
        """
        Divide the existing values of slots with the given key(s).

        Note that this operation does not change the existence flags of the
        keys. In other words, if element(s) with `key` do not exist, then
        they will still be flagged as non-existent after this operation.

        Args:
            key: A single key, or multiple keys (where the leftmost dimension
                of the given keys conform with the `batch_shape`).
            value: The value(s) that will be used as divisor(s).
            where: Optionally a boolean mask whose shape matches `batch_shape`.
                If a `where` mask is given, then modifications will happen only
                on the memory slots whose corresponding mask values are True.
        """
        self._data.divide_(key, value, where)

    def multiply_(self, key: Numbers, value: Numbers, where: Optional[Numbers] = None):
        """
        Multiply the existing values of slots with the given key(s).

        Note that this operation does not change the existence flags of the
        keys. In other words, if element(s) with `key` do not exist, then
        they will still be flagged as non-existent after this operation.

        Args:
            key: A single key, or multiple keys (where the leftmost dimension
                of the given keys conform with the `batch_shape`).
            value: The value(s) that will be used as the multiplier(s).
            where: Optionally a boolean mask whose shape matches `batch_shape`.
                If a `where` mask is given, then modifications will happen only
                on the memory slots whose corresponding mask values are True.
        """
        self._data.multiply_(key, value, where)

    def contains(self, key: Numbers) -> torch.Tensor:
        """
        Query whether or not the element(s) with the given key(s) exist.

        Args:
            key: A single key, or multiple keys (where the leftmost dimension
                of the given keys conform with the `batch_shape`).
        Returns:
            A boolean tensor indicating whether or not the element(s) with the
            specified key(s) exist.
        """
        return self._exist[key]

    def __getitem__(self, key: Numbers) -> torch.Tensor:
        """
        Get the value(s) associated with the given key(s).

        Args:
            key: A single key, or multiple keys (where the leftmost dimension
                of the given keys conform with the `batch_shape`).
        Returns:
            The value(s) associated with the given key(s).
        """
        return self.get(key)

    def __setitem__(self, key: Numbers, value: Numbers):
        """
        Set the value(s) associated with the given key(s).

        Args:
            key: A single key, or multiple keys (where the leftmost dimension
                of the given keys conform with the `batch_shape`).
            value: The new value(s).
        """
        self.set_(key, value)

    def clear(self, where: Optional[torch.Tensor] = None):
        """
        Clear the dictionaries.

        In the context of this data structure, to "clear" means to set the
        status for each key to non-existent.

        Args:
            where: Optionally a boolean tensor, specifying which dictionaries
                within the batch should be cleared. If this argument is omitted
                (i.e. left as None), then all dictionaries will be cleared.
        """
        if where is None:
            self._exist.data[:] = False
        else:
            where = self._get_where(where)
            all_false = torch.tensor(False, dtype=torch.bool, device=self._exist.device).expand(self._exist.shape)
            self._exist.data[:] = do_where(where, all_false, self._exist.data[:])

    @property
    def data(self) -> torch.Tensor:
        """
        The entire value tensor
        """
        return self._data.data


_InfoForAddingElement = namedtuple("_InfoForAddingElement", ["valid_move", "to_be_declared_non_empty"])
_InfoForRemovingElement = namedtuple("_InfoForRemovingElement", ["valid_move", "to_be_declared_empty"])


class CList(Structure):
    """
    Representation of a batchable, contiguous, variable-length list structure.

    This CList structure works with a pre-allocated contiguous block of memory
    with a separately stored length. In the batched case, each batch item
    has its own length.

    This structure supports negative indexing (meaning that -1 refers to the
    last item, -2 refers to the second last item, etc.).

    Let us imagine that we need a list where each element has a shape `(3,)`,
    and our maximum length is 5.
    Such a list could be instantiated via:

    ```python
    lst = CList(3, max_length=5)
    ```

    In its initial state, the list is empty, which can be visualized like:

    ```text
     _______________________________________________________________
    | index  |    0     |    1     |    2     |    3     |    4     |
    | values | <unused> | <unused> | <unused> | <unused> | <unused> |
    |________|__________|__________|__________|__________|__________|
    ```

    We can add elements into our list like this:

    ```python
    lst.append_(torch.tensor([1.0, 2.0, 3.0]))
    lst.append_(torch.tensor([4.0, 5.0, 6.0]))
    ```

    After these two push operations, our list looks like this:

    ```text
     __________________________________________________________________
    | index  |      0     |     1     |    2     |    3     |    4     |
    | values | [1. 2. 3.] | [4. 5. 6] | <unused> | <unused> | <unused> |
    |________|____________|___________|__________|__________|__________|
    ```

    Here, `lst[0]` returns `[1. 2. 3.]` and `lst[1]` returns `[4. 5. 6.]`.
    A `CList` also supports negative indices, allowing `lst[-1]` to return
    `[4. 5. 6.]` (the last element) and `lst[-2]` to return `[1. 2. 3.]`
    (the second last element).

    One can also create a batch of lists. Let us imagine that we wish to
    create a batch of lists such that the batch size is 4, length of an
    element is 3, and the maximum length is 5. Such a batch can be created
    as follows:

    ```python
    list_batch = CList(3, max_length=5, batch_size=4)
    ```

    Our batch can be visualized like this:

    ```text
     __[ batch item 0 ]_____________________________________________
    | index  |    0     |    1     |    2     |    3     |    4     |
    | values | <unused> | <unused> | <unused> | <unused> | <unused> |
    |________|__________|__________|__________|__________|__________|

     __[ batch item 1 ]_____________________________________________
    | index  |    0     |    1     |    2     |    3     |    4     |
    | values | <unused> | <unused> | <unused> | <unused> | <unused> |
    |________|__________|__________|__________|__________|__________|

     __[ batch item 2 ]_____________________________________________
    | index  |    0     |    1     |    2     |    3     |    4     |
    | values | <unused> | <unused> | <unused> | <unused> | <unused> |
    |________|__________|__________|__________|__________|__________|

     __[ batch item 3 ]_____________________________________________
    | index  |    0     |    1     |    2     |    3     |    4     |
    | values | <unused> | <unused> | <unused> | <unused> | <unused> |
    |________|__________|__________|__________|__________|__________|
    ```

    Let us now add `[1. 1. 1.]` to the batch item 0, `[2. 2. 2.]` to the batch
    item 1, and so on:

    ```python
    list_batch.append_(
        torch.tensor(
            [
                [1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
                [3.0, 3.0, 3.0],
                [4.0, 4.0, 4.0],
            ]
        )
    )
    ```

    After these operations, `list_batch` looks like this:

    ```text
     __[ batch item 0 ]_______________________________________________
    | index  |    0       |    1     |    2     |    3     |    4     |
    | values | [1. 1. 1.] | <unused> | <unused> | <unused> | <unused> |
    |________|____________|__________|__________|__________|__________|

     __[ batch item 1 ]_______________________________________________
    | index  |    0       |    1     |    2     |    3     |    4     |
    | values | [2. 2. 2.] | <unused> | <unused> | <unused> | <unused> |
    |________|____________|__________|__________|__________|__________|

     __[ batch item 2 ]_______________________________________________
    | index  |    0       |    1     |    2     |    3     |    4     |
    | values | [3. 3. 3.] | <unused> | <unused> | <unused> | <unused> |
    |________|____________|__________|__________|__________|__________|

     __[ batch item 3 ]_______________________________________________
    | index  |    0       |    1     |    2     |    3     |    4     |
    | values | [4. 4. 4.] | <unused> | <unused> | <unused> | <unused> |
    |________|____________|__________|__________|__________|__________|
    ```

    We can also use a boolean mask to add to only some of the lists within
    the batch:

    ```python
    list_batch.append_(
        torch.tensor(
            [
                [5.0, 5.0, 5.0],
                [6.0, 6.0, 6.0],
                [7.0, 7.0, 7.0],
                [8.0, 8.0, 8.0],
            ]
        ),
        where=torch.tensor([True, False, False, True]),
    )
    ```

    which would update our batch of lists like this:

    ```text
     __[ batch item 0 ]_________________________________________________
    | index  |    0       |    1       |    2     |    3     |    4     |
    | values | [1. 1. 1.] | [5. 5. 5.] | <unused> | <unused> | <unused> |
    |________|____________|____________|__________|__________|__________|

     __[ batch item 1 ]_________________________________________________
    | index  |    0       |    1       |    2     |    3     |    4     |
    | values | [2. 2. 2.] |  <unused>  | <unused> | <unused> | <unused> |
    |________|____________|____________|__________|__________|__________|

     __[ batch item 2 ]_________________________________________________
    | index  |    0       |    1       |    2     |    3     |    4     |
    | values | [3. 3. 3.] |  <unused>  | <unused> | <unused> | <unused> |
    |________|____________|____________|__________|__________|__________|

     __[ batch item 3 ]_________________________________________________
    | index  |    0       |    1       |    2     |    3     |    4     |
    | values | [4. 4. 4.] | [8. 8. 8.] | <unused> | <unused> | <unused> |
    |________|____________|____________|__________|__________|__________|
    ```

    Please notice above how the batch items 1 and 2 were not modified because
    their corresponding boolean values in the `where` tensor were given as
    `False`.

    After all these modifications we would get the following results:

    ```text
    >>> list_batch[0]
    torch.tensor(
        [[1. 1. 1.],
         [2. 2. 2.],
         [3. 3. 3.],
         [4. 4. 4.]]
    )

    >>> list_batch[[1, 0, 0, 1]]
    torch.tensor(
        [[5. 5. 5.],
         [2. 2. 2.],
         [3. 3. 3.],
         [8. 8. 8.]]
    )

    >>> list_batch[[-1, -1, -1, -1]]
    torch.tensor(
        [[5. 5. 5.],
         [2. 2. 2.],
         [3. 3. 3.],
         [8. 8. 8.]]
    )
    ```

    Note that this CList structure also supports the ability to insert to the
    beginning, or to remove from the beginning. These operations internally
    shift the addresses for the beginning of the data within the underlying
    memory, and therefore, they are not any more costly than adding to or
    removing from the end of the list.
    """

    def __init__(
        self,
        *size: Union[int, list, tuple],
        max_length: int,
        batch_size: Optional[Union[int, tuple, list]] = None,
        batch_shape: Optional[Union[int, tuple, list]] = None,
        dtype: Optional[DType] = None,
        device: Optional[Device] = None,
        verify: bool = True,
    ):
        self._verify = bool(verify)
        self._max_length = int(max_length)

        self._data = CMemory(
            *size,
            num_keys=self._max_length,
            batch_size=batch_size,
            batch_shape=batch_shape,
            dtype=dtype,
            device=device,
            verify=False,
        )

        self._begin, self._end = [
            CMemory(
                num_keys=1,
                batch_size=batch_size,
                batch_shape=batch_shape,
                dtype=torch.int64,
                device=device,
                verify=False,
                fill_with=-1,
            )
            for _ in range(2)
        ]

        if "float" in str(self._data.dtype):
            self._pop_fallback = float("nan")
        else:
            self._pop_fallback = 0

        if self._begin.batch_ndim == 0:
            self._all_zeros = torch.tensor(0, dtype=torch.int64, device=self._begin.device)
        else:
            self._all_zeros = torch.zeros(1, dtype=torch.int64, device=self._begin.device).expand(
                self._begin.batch_shape
            )

    def _is_empty(self) -> torch.Tensor:
        # return (self._begin[self._all_zeros] == -1) & (self._end[self._all_zeros] == -1)
        return self._begin[self._all_zeros] == -1

    def _has_one_element(self) -> torch.Tensor:
        begin = self._begin[self._all_zeros]
        end = self._end[self._all_zeros]
        return (begin == end) & (begin >= 0)

    def _is_full(self) -> torch.Tensor:
        begin = self._begin[self._all_zeros]
        end = self._end[self._all_zeros]
        return ((end - begin) % self._max_length) == (self._max_length - 1)

    @staticmethod
    def _considering_where(other_mask: torch.Tensor, where: Optional[torch.Tensor]) -> torch.Tensor:
        return other_mask if where is None else other_mask & where

    def _get_info_for_adding_element(self, where: Optional[torch.Tensor]) -> _InfoForAddingElement:
        is_empty = self._is_empty()
        is_full = self._is_full()
        to_be_declared_non_empty = self._considering_where(is_empty, where)
        if self._verify:
            invalid_move = self._considering_where(is_full, where)
            if torch.any(invalid_move):
                raise IndexError("Some of the queues are full, and therefore elements cannot be added to them")
        valid_move = self._considering_where((~is_empty) & (~is_full), where)
        return _InfoForAddingElement(valid_move=valid_move, to_be_declared_non_empty=to_be_declared_non_empty)

    def _get_info_for_removing_element(self, where: Optional[torch.Tensor]) -> _InfoForRemovingElement:
        is_empty = self._is_empty()
        has_one_element = self._has_one_element()
        if self._verify:
            invalid_move = self._considering_where(is_empty, where)
            if torch.any(invalid_move):
                raise IndexError(
                    "Some of the queues are already empty, and therefore elements cannot be removed from them"
                )
        to_be_declared_empty = self._considering_where(has_one_element, where)
        valid_move = self._considering_where((~is_empty) & (~has_one_element), where)
        return _InfoForRemovingElement(valid_move=valid_move, to_be_declared_empty=to_be_declared_empty)

    def _move_begin_forward(self, where: Optional[torch.Tensor]):
        valid_move, to_be_declared_empty = self._get_info_for_removing_element(where)
        self._begin.set_(self._all_zeros, -1, where=to_be_declared_empty)
        self._end.set_(self._all_zeros, -1, where=to_be_declared_empty)
        self._begin.add_circular_(self._all_zeros, 1, self._max_length, where=valid_move)

    def _move_end_forward(self, where: Optional[torch.Tensor]):
        valid_move, to_be_declared_non_empty = self._get_info_for_adding_element(where)
        self._begin.set_(self._all_zeros, 0, where=to_be_declared_non_empty)
        self._end.set_(self._all_zeros, 0, where=to_be_declared_non_empty)
        self._end.add_circular_(self._all_zeros, 1, self._max_length, where=valid_move)

    def _move_begin_backward(self, where: Optional[torch.Tensor]):
        valid_move, to_be_declared_non_empty = self._get_info_for_adding_element(where)
        self._begin.set_(self._all_zeros, 0, where=to_be_declared_non_empty)
        self._end.set_(self._all_zeros, 0, where=to_be_declared_non_empty)
        self._begin.add_circular_(self._all_zeros, -1, self._max_length, where=valid_move)

    def _move_end_backward(self, where: Optional[torch.Tensor]):
        valid_move, to_be_declared_empty = self._get_info_for_removing_element(where)
        self._begin.set_(self._all_zeros, -1, where=to_be_declared_empty)
        self._end.set_(self._all_zeros, -1, where=to_be_declared_empty)
        self._end.add_circular_(self._all_zeros, -1, self._max_length, where=valid_move)

    def _get_key(self, key: Numbers) -> torch.Tensor:
        key = torch.as_tensor(key, dtype=torch.int64, device=self._data.device)
        batch_shape = self._data.batch_shape
        if key.shape != batch_shape:
            if key.ndim == 0:
                key = key.expand(self._data.batch_shape)
            else:
                raise ValueError(
                    f"Expected the keys of shape {batch_shape}, but received them in this shape: {key.shape}"
                )
        return key

    def _is_underlying_key_valid(self, underlying_key: torch.Tensor) -> torch.Tensor:
        within_valid_range = (underlying_key >= 0) & (underlying_key < self._max_length)
        begin = self._begin[self._all_zeros]
        end = self._end[self._all_zeros]
        empty = self._is_empty()
        non_empty = ~empty
        larger_end = non_empty & (end > begin)
        smaller_end = non_empty & (end < begin)
        same_begin_end = (begin == end) & (~empty)
        valid = within_valid_range & (
            (same_begin_end & (underlying_key == begin))
            | (larger_end & (underlying_key >= begin) & (underlying_key <= end))
            | (smaller_end & ((underlying_key <= end) | (underlying_key >= begin)))
        )
        return valid

    def _mod_underlying_key(self, underlying_key: torch.Tensor, *, verify: Optional[bool] = None) -> torch.Tensor:
        verify = self._verify if verify is None else verify
        if self._verify:
            where_negative = underlying_key < 0
            where_too_large = underlying_key >= self._max_length
            underlying_key = underlying_key.clone()
            underlying_key[where_negative] += self._max_length
            underlying_key[where_too_large] -= self._max_length
        else:
            underlying_key = underlying_key % self._max_length

        return underlying_key

    def _get_underlying_key(
        self,
        key: Numbers,
        *,
        verify: Optional[bool] = None,
        return_validity: bool = False,
        where: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple]:
        if where is not None:
            where = self._get_where(where)
        verify = self._verify if verify is None else verify
        key = self._get_key(key)
        underlying_key_for_pos_index = self._begin[self._all_zeros] + key
        underlying_key_for_neg_index = self._end[self._all_zeros] + key + 1
        underlying_key = torch.where(key >= 0, underlying_key_for_pos_index, underlying_key_for_neg_index)
        underlying_key = self._mod_underlying_key(underlying_key, verify=verify)

        if verify or return_validity:
            valid = self._is_underlying_key_valid(underlying_key)
        else:
            valid = None

        if verify:
            okay = valid if where is None else valid | (~where)
            if not torch.all(okay):
                raise IndexError("Encountered invalid index/indices")

        if return_validity:
            return underlying_key, valid
        else:
            return underlying_key

    def get(self, key: Numbers, default: Optional[Numbers] = None) -> torch.Tensor:
        """
        Get the value(s) from the specified element(s).

        Args:
            key: The index/indices pointing to the element(s) whose value(s)
                is/are queried.
            default: Default value(s) to be returned for when the specified
                index/indices are invalid and/or out of range.
        Returns:
            The value(s) stored by the element(s).
        """
        if default is None:
            underlying_key = self._get_underlying_key(key)
            return self._data[underlying_key]
        else:
            default = self._data._get_value(default)
            underlying_key, valid_key = self._get_underlying_key(key, verify=False, return_validity=True)
            return do_where(valid_key, self._data[underlying_key % self._max_length], default)

    def __getitem__(self, key: Numbers) -> torch.Tensor:
        """
        Get the value(s) from the specified element(s).

        Args:
            key: The index/indices pointing to the element(s) whose value(s)
                is/are queried.
        Returns:
            The value(s) stored by the element(s).
        """
        return self.get(key)

    def _apply_modification_method(
        self, method_name: str, key: Numbers, value: Numbers, where: Optional[Numbers] = None
    ):
        underlying_key = self._get_underlying_key(key, where=where)
        getattr(self._data, method_name)(underlying_key, value, where)

    def set_(self, key: Numbers, value: Numbers, where: Optional[Numbers] = None):
        """
        Set the element(s) addressed to by the given key(s).

        Args:
            key: The index/indices tensor.
            value: The new value(s).
            where: Optionally a boolean mask. When provided, only the elements
                whose corresponding mask value(s) is/are True will be subject
                to modification.
        """
        self._apply_modification_method("set_", key, value, where)

    def __setitem__(self, key: Numbers, value: Numbers):
        """
        Set the element(s) addressed to by the given key(s).

        Args:
            key: The index/indices tensor.
            value: The new value(s).
        """
        self._apply_modification_method("set_", key, value)

    def add_(self, key: Numbers, value: Numbers, where: Optional[Numbers] = None):
        """
        Add to the element(s) addressed to by the given key(s).

        Please note that the word "add" is used in the arithmetic sense
        (i.e. in the sense of performing addition). For putting a new
        element into this list, please see the method `append_(...)`.

        Args:
            key: The index/indices tensor.
            value: The value(s) that will be added onto the existing
                element(s).
            where: Optionally a boolean mask. When provided, only the elements
                whose corresponding mask value(s) is/are True will be subject
                to modification.
        """
        self._apply_modification_method("add_", key, value, where)

    def subtract_(self, key: Numbers, value: Numbers, where: Optional[Numbers] = None):
        """
        Subtract from the element(s) addressed to by the given key(s).

        Args:
            key: The index/indices tensor.
            value: The value(s) that will be subtracted from the existing
                element(s).
            where: Optionally a boolean mask. When provided, only the elements
                whose corresponding mask value(s) is/are True will be subject
                to modification.
        """
        self._apply_modification_method("subtract_", key, value, where)

    def multiply_(self, key: Numbers, value: Numbers, where: Optional[Numbers] = None):
        """
        Multiply the element(s) addressed to by the given key(s).

        Args:
            key: The index/indices tensor.
            value: The value(s) that will be used as the multiplier(s) on the
                existing element(s).
            where: Optionally a boolean mask. When provided, only the elements
                whose corresponding mask value(s) is/are True will be subject
                to modification.
        """
        self._apply_modification_method("multiply_", key, value, where)

    def divide_(self, key: Numbers, value: Numbers, where: Optional[Numbers] = None):
        """
        Divide the element(s) addressed to by the given key(s).

        Args:
            key: The index/indices tensor.
            value: The value(s) that will be used as the divisor(s) on the
                existing element(s).
            where: Optionally a boolean mask. When provided, only the elements
                whose corresponding mask value(s) is/are True will be subject
                to modification.
        """
        self._apply_modification_method("divide_", key, value, where)

    def append_(self, value: Numbers, where: Optional[Numbers] = None):
        """
        Add new item(s) to the end(s) of the list(s).

        The length(s) of the updated list(s) will increase by 1.

        Args:
            value: The element that will be added to the list.
                In the non-batched case, this element is expected as a tensor
                whose shape matches `value_shape`.
                In the batched case, this value is expected as a batch of
                elements with extra leftmost dimensions (those extra leftmost
                dimensions being expressed by `batch_shape`).
            where: Optionally a boolean mask whose shape matches `batch_shape`.
                If a `where` mask is given, then additions will happen only
                on the lists whose corresponding mask values are True.
        """
        where = None if where is None else self._get_where(where)
        self._move_end_forward(where)
        self.set_(-1, value, where=where)

    def push_(self, value: Numbers, where: Optional[Numbers] = None):
        """
        Alias for the method `append_(...)`.
        We provide this alternative name so that users who wish to use this
        CList structure like a stack will be able to use familiar terminology.
        """
        return self.append_(value, where=where)

    def appendleft_(self, value: Numbers, where: Optional[Numbers] = None):
        """
        Add new item(s) to the beginning point(s) of the list(s).

        The length(s) of the updated list(s) will increase by 1.

        Args:
            value: The element that will be added to the list.
                In the non-batched case, this element is expected as a tensor
                whose shape matches `value_shape`.
                In the batched case, this value is expected as a batch of
                elements with extra leftmost dimensions (those extra leftmost
                dimensions being expressed by `batch_shape`).
            where: Optionally a boolean mask whose shape matches `batch_shape`.
                If a `where` mask is given, then additions will happen only
                on the lists whose corresponding mask values are True.
        """
        where = None if where is None else self._get_where(where)
        self._move_begin_backward(where)
        self.set_(0, value, where=where)

    def pop_(self, where: Optional[Numbers] = None):
        """
        Pop the last item(s) from the ending point(s) list(s).

        The length(s) of the updated list(s) will decrease by 1.

        Args:
            where: Optionally a boolean mask whose shape matches `batch_shape`.
                If a `where` mask is given, then the pop operations will happen
                only on the lists whose corresponding mask values are True.
        Returns:
            The popped item(s).
        """
        where = None if where is None else self._get_where(where)
        result = self.get(-1, default=self._pop_fallback)
        self._move_end_backward(where)
        return result

    def popleft_(self, where: Optional[Numbers] = None):
        """
        Pop the last item(s) from the beginning point(s) list(s).

        The length(s) of the updated list(s) will decrease by 1.

        Args:
            where: Optionally a boolean mask whose shape matches `batch_shape`.
                If a `where` mask is given, then the pop operations will happen
                only on the lists whose corresponding mask values are True.
        Returns:
            The popped item(s).
        """
        where = None if where is None else self._get_where(where)
        result = self.get(0, default=self._pop_fallback)
        self._move_begin_forward(where)
        return result

    def clear(self, where: Optional[torch.Tensor] = None):
        """
        Clear the list(s).

        In the context of this data structure, to "clear" means to reduce their
        lengths to 0.

        Args:
            where: Optionally a boolean tensor, specifying which lists within
                the batch will be cleared. If this argument is omitted (i.e.
                left as None), then all of the lists will be cleared.
        """
        if where is None:
            self._begin.data[:] = -1
            self._end.data[:] = -1
        else:
            where = self._get_where(where)
            all_minus_ones = torch.tensor(-1, dtype=torch.int64, device=self._begin.device).expand(self._begin.shape)
            self._begin.data[:] = do_where(where, all_minus_ones, self._begin.data)
            self._end.data[:] = do_where(where, all_minus_ones, self._end.data)

    @property
    def data(self) -> torch.Tensor:
        """
        The underlying tensor which stores all the data
        """
        return self._data.data

    @property
    def length(self) -> torch.Tensor:
        """
        The length(s) of the list(s)
        """
        is_empty = self._is_empty()
        is_full = self._is_full()
        result = ((self._end[self._all_zeros] - self._begin[self._all_zeros]) % self._max_length) + 1
        result[is_empty] = 0
        result[is_full] = self._max_length
        return result

    @property
    def max_length(self) -> int:
        """
        Maximum length for the list(s)
        """
        return self._max_length


class CBag(Structure):
    """
    An integer bag from which one can do sampling without replacement.

    Let us imagine that we wish to create a bag whose maximum length (i.e.
    whose maximum number of contained elements) is 5. For this, we can do:

    ```python
    bag = CBag(max_length=5)
    ```

    which gives us an empty bag (i.e. a bag in which all pre-allocated slots
    are empty):

    ```
     _________________________________________________
    |         |         |         |         |         |
    | <empty> | <empty> | <empty> | <empty> | <empty> |
    |_________|_________|_________|_________|_________|
    ```

    Given that the maximum length for this bag is 5, the default set of
    acceptable values for this bag is 0, 1, 2, 3, 4. Let us put three values
    into our bag:

    ```
    bag.push_(torch.tensor(1))
    bag.push_(torch.tensor(3))
    bag.push_(torch.tensor(4))
    ```

    After these push operations, our bag can be visualized like this:

    ```
     _________________________________________________
    |         |         |         |         |         |
    |   1     |   3     |   4     | <empty> | <empty> |
    |_________|_________|_________|_________|_________|
    ```

    Let us now sample an element from this bag:

    ```python
    sampled1 = bag.pop_()
    ```

    Because this is the first time we are sampling from this bag, the elements
    will be first shuffled. Let us assume that the shuffling resulted in:

    ```
     _________________________________________________
    |         |         |         |         |         |
    |   3     |   1     |   4     | <empty> | <empty> |
    |_________|_________|_________|_________|_________|
    ```

    Given this shuffed state, our call to `pop_(...)` will pop the leftmost
    element (3 in this case). Therefore, the value of `sampled1` will be 3
    (as a scalar PyTorch tensor), and the state of the bag after the pop
    operation will be:

    ```
     _________________________________________________
    |         |         |         |         |         |
    |   1     |   4     | <empty> | <empty> | <empty> |
    |_________|_________|_________|_________|_________|
    ```

    Let us keep sampling until the bag is empty:

    ```python
    sampled2 = bag.pop_()
    sampled3 = bag.pop_()
    ```

    The value of `sampled2` becomes 1, and the value of `sampled3` becomes 4.

    This class can also represent a contiguous batch of bags. As an example,
    let us create 4 bags, each of length 5:

    ```python
    bag_batch = CBag(batch_size=4, max_length=5)
    ```

    After this instantiation, `bag_batch` can be visualized like this:

    ```
     __[ batch item 0 ]_______________________________
    |         |         |         |         |         |
    | <empty> | <empty> | <empty> | <empty> | <empty> |
    |_________|_________|_________|_________|_________|

     __[ batch item 1 ]_______________________________
    |         |         |         |         |         |
    | <empty> | <empty> | <empty> | <empty> | <empty> |
    |_________|_________|_________|_________|_________|

     __[ batch item 2 ]_______________________________
    |         |         |         |         |         |
    | <empty> | <empty> | <empty> | <empty> | <empty> |
    |_________|_________|_________|_________|_________|

     __[ batch item 3 ]_______________________________
    |         |         |         |         |         |
    | <empty> | <empty> | <empty> | <empty> | <empty> |
    |_________|_________|_________|_________|_________|
    ```

    We can add values to our batch like this:

    ```python
    bag_batch.push_(torch.tensor([3, 2, 3, 1]))
    bag_batch.push_(torch.tensor([3, 1, 1, 4]))
    ```

    which would result in:

    ```
     __[ batch item 0 ]_______________________________
    |         |         |         |         |         |
    |   3     |   3     | <empty> | <empty> | <empty> |
    |_________|_________|_________|_________|_________|

     __[ batch item 1 ]_______________________________
    |         |         |         |         |         |
    |   2     |   1     | <empty> | <empty> | <empty> |
    |_________|_________|_________|_________|_________|

     __[ batch item 2 ]_______________________________
    |         |         |         |         |         |
    |   3     |   1     | <empty> | <empty> | <empty> |
    |_________|_________|_________|_________|_________|

     __[ batch item 3 ]_______________________________
    |         |         |         |         |         |
    |   1     |   4     | <empty> | <empty> | <empty> |
    |_________|_________|_________|_________|_________|
    ```

    We can also add values only to some of the bags within the batch:

    ```
    bag_batch.push_(
        torch.tensor([0, 2, 1, 0]),
        where=torch.tensor([True, True, False, False])),
    )
    ```

    which would result in:

    ```
     __[ batch item 0 ]_______________________________
    |         |         |         |         |         |
    |   3     |   3     |   0     | <empty> | <empty> |
    |_________|_________|_________|_________|_________|

     __[ batch item 1 ]_______________________________
    |         |         |         |         |         |
    |   2     |   1     |   2     | <empty> | <empty> |
    |_________|_________|_________|_________|_________|

     __[ batch item 2 ]_______________________________
    |         |         |         |         |         |
    |   3     |   1     | <empty> | <empty> | <empty> |
    |_________|_________|_________|_________|_________|

     __[ batch item 3 ]_______________________________
    |         |         |         |         |         |
    |   1     |   4     | <empty> | <empty> | <empty> |
    |_________|_________|_________|_________|_________|
    ```

    Notice that the batch items 2 and 3 were not affected, because their
    corresponding values in the `where` tensor were given as False.

    Let us now assume that we wish to obtain a sample from each bag. We can do:

    ```python
    sample_batch1 = bag_batch.pop_()
    ```

    Since this is the first sampling operation on this bag batch, each bag
    will first be shuffled. Let us assume that the shuffling resulted in:

    ```
     __[ batch item 0 ]_______________________________
    |         |         |         |         |         |
    |   0     |   3     |   3     | <empty> | <empty> |
    |_________|_________|_________|_________|_________|

     __[ batch item 1 ]_______________________________
    |         |         |         |         |         |
    |   1     |   2     |   2     | <empty> | <empty> |
    |_________|_________|_________|_________|_________|

     __[ batch item 2 ]_______________________________
    |         |         |         |         |         |
    |   3     |   1     | <empty> | <empty> | <empty> |
    |_________|_________|_________|_________|_________|

     __[ batch item 3 ]_______________________________
    |         |         |         |         |         |
    |   4     |   1     | <empty> | <empty> | <empty> |
    |_________|_________|_________|_________|_________|
    ```

    Given this shuffled state, the pop operation takes the leftmost element
    from each bag. Therefore, the value of `sample_batch1` becomes a
    1-dimensional tensor containing `[0, 1, 3, 4]`. Once the pop operation
    is completed, the state of the batch of bags becomes:

    ```
     __[ batch item 0 ]_______________________________
    |         |         |         |         |         |
    |   3     |   3     | <empty> | <empty> | <empty> |
    |_________|_________|_________|_________|_________|

     __[ batch item 1 ]_______________________________
    |         |         |         |         |         |
    |   2     |   2     | <empty> | <empty> | <empty> |
    |_________|_________|_________|_________|_________|

     __[ batch item 2 ]_______________________________
    |         |         |         |         |         |
    |   1     | <empty> | <empty> | <empty> | <empty> |
    |_________|_________|_________|_________|_________|

     __[ batch item 3 ]_______________________________
    |         |         |         |         |         |
    |   1     | <empty> | <empty> | <empty> | <empty> |
    |_________|_________|_________|_________|_________|
    ```

    Now, if we wish to pop only from some of the bags, we can do:

    ```python
    sample_batch2 = bag_batch.pop_(
        where=torch.tensor([True, False, True, False]),
    )
    ```

    which makes the value of `sample_batch2` a 1-dimensional tensor containing
    `[3, 2, 1, 1]` (the leftmost element for each bag). The state of our batch
    of bags will become:

    ```
     __[ batch item 0 ]_______________________________
    |         |         |         |         |         |
    |   3     | <empty> | <empty> | <empty> | <empty> |
    |_________|_________|_________|_________|_________|

     __[ batch item 1 ]_______________________________
    |         |         |         |         |         |
    |   2     |   2     | <empty> | <empty> | <empty> |
    |_________|_________|_________|_________|_________|

     __[ batch item 2 ]_______________________________
    |         |         |         |         |         |
    | <empty> | <empty> | <empty> | <empty> | <empty> |
    |_________|_________|_________|_________|_________|

     __[ batch item 3 ]_______________________________
    |         |         |         |         |         |
    |   1     | <empty> | <empty> | <empty> | <empty> |
    |_________|_________|_________|_________|_________|
    ```

    Notice that the batch items 1 and 3 were not modified, because their
    corresponding values in the `where` argument were given as False.
    """

    def __init__(
        self,
        *,
        max_length: int,
        value_range: Optional[tuple] = None,
        batch_size: Optional[Union[int, tuple, list]] = None,
        batch_shape: Optional[Union[int, tuple, list]] = None,
        generator: Any = None,
        dtype: Optional[DType] = None,
        device: Optional[Device] = None,
        verify: bool = True,
    ):
        """
        Initialize the CBag.

        Args:
            max_length: Maximum length (i.e. maximum capacity for storing
                elements).
            value_range: Optionally expected as a tuple of integers in the
                form `(a, b)` where `a` is the lower bound and `b` is the
                exclusive upper bound for the range of acceptable integer
                values. If this argument is omitted, the range will be
                `(0, n)` where `n` is `max_length`.
            batch_size: Optionally an integer or a size tuple, for when
                one wishes to create not just a single bag, but a batch
                of bags.
            batch_shape: Alias for the argument `batch_size`.
            generator: Optionally an instance of `torch.Generator` or any
                object with an attribute (or a property) named `generator`
                (in which case it will be expected that this attribute will
                provide the actual `torch.Generator` instance). If this
                argument is provided, then the shuffling operation will use
                this generator. Otherwise, the global generator of PyTorch
                will be used.
            dtype: dtype for the values contained by the bag(s).
                By default, the dtype is `torch.int64`.
            device: The device on which the bag(s) will be stored.
                By default, the device is `torch.device("cpu")`.
            verify: Whether or not to do explicit checks for the correctness
                of the operations (against popping from an empty bag or
                pushing into a full bag). By default, this is True.
                If you are sure that such errors will not occur, you might
                turn this to False for getting a performance gain.
        """

        if dtype is None:
            dtype = torch.int64
        else:
            dtype = to_torch_dtype(dtype)
            if dtype not in (torch.int16, torch.int32, torch.int64):
                raise RuntimeError(
                    f"CBag currently supports only torch.int16, torch.int32, and torch.int64."
                    f" This dtype is not supported: {repr(dtype)}."
                )

        self._gen_kwargs = {}
        if generator is not None:
            if isinstance(generator, torch.Generator):
                self._gen_kwargs["generator"] = generator
            else:
                generator = generator.generator
                if generator is not None:
                    self._gen_kwargs["generator"] = generator

        max_length = int(max_length)

        self._data = CList(
            max_length=max_length,
            batch_size=batch_size,
            batch_shape=batch_shape,
            dtype=dtype,
            device=device,
            verify=verify,
        )

        if value_range is None:
            a = 0
            b = max_length
        else:
            a, b = value_range

        self._low_item = int(a)
        self._high_item = int(b)  # upper bound is exclusive
        self._choice_count = self._high_item - self._low_item
        self._bignum = self._choice_count + 1

        if self._low_item < 1:
            self._shift = 1 - self._low_item
        else:
            self._shift = 0

        self._empty = self._low_item - 1
        self._data.data[:] = self._empty
        self._sampling_phase: bool = False

    def push_(self, value: Numbers, where: Optional[Numbers] = None):
        """
        Push new value(s) into the bag(s).

        Args:
            value: The value(s) to be pushed into the bag(s).
            where: Optionally a boolean tensor. If this is given, then only
                the bags with their corresponding boolean flags set as True
                will be affected.
        """
        if self._sampling_phase:
            raise RuntimeError("Cannot put a new element into the CBag after calling `sample_(...)`")
        self._data.push_(value, where)

    def _shuffle(self):
        dtype = self._data.dtype
        device = self._data.device
        nrows, ncols = self._data.data.shape

        try:
            gaussian_noise = torch.randn(nrows, ncols, dtype=torch.float32, device=device, **(self._gen_kwargs))
            noise = gaussian_noise.argsort().to(dtype=dtype) * self._bignum
            self._data.data[:] += torch.where(
                self._data.data != self._empty, self._shift + noise, torch.tensor(0, dtype=dtype, device=device)
            )
            self._data.data[:] = self._data.data.sort(dim=-1, descending=True, stable=False).values
        finally:
            self._data.data[:] %= self._bignum
            self._data.data[:] -= self._shift

    def pop_(self, where: Optional[Numbers] = None) -> torch.Tensor:
        """
        Sample value(s) from the bag(s).

        Upon being called for the first time, this method will cause the
        contained elements to be shuffled.

        Args:
            where: Optionally a boolean tensor. If this is given, then only
                the bags with their corresponding boolean flags set as True
                will be affected.
        """
        if not self._sampling_phase:
            self._shuffle()
            self._sampling_phase = True
        return self._data.pop_(where)

    def clear(self):
        """
        Clear the bag(s).
        """
        self._data.data[:] = self._empty
        self._data.clear()
        self._sampling_phase = False

    @property
    def length(self) -> torch.Tensor:
        """
        The length(s) of the bag(s)
        """
        return self._data.length

    @property
    def data(self) -> torch.Tensor:
        """
        The underlying data tensor
        """
        return self._data.data
