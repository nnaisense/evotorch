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


from collections.abc import Callable, Iterable, Mapping, Sequence
from itertools import chain
from numbers import Integral
from threading import Lock
from typing import Any, NamedTuple

import numpy as np
import torch
from ray.util import ActorPool

from .core import Problem, SolutionBatch
from .tools import ObjectArray, TensorFrame
from .tools._shallow_containers import move_shallow_container_to_device

TensorLike = torch.Tensor | TensorFrame | ObjectArray


class _TensorSplittingResult(NamedTuple):
    chunks: list[TensorLike]
    original_size: int


def _split_tensor(
    x: TensorLike,
    num_actors: int,
    *,
    chunk_size: int | None = None,
    expect_size: int | None = None,
    target_device: str | torch.device | None = None,
) -> _TensorSplittingResult:
    """
    Split a tensor, or a TensorFrame, or an ObjectArray into chunks.

    Args:
        x: The tensor or ObjectArray or TensorFrame to be split into chunks.
        num_actors: Number of remote actors, as an integer that is at least 2.
            If `chunk_size` is not provided (i.e. left as None), the number
            of chunks and the size of each chunk will be determined by this
            `num_actors`.
        chunk_size: The size of a chunk when splitting a tensor/array into
            chunks. If this is provided, then this will be the main factor for
            determining the chunk size and also the number of chunks.
            Can be left as None if the number of chunks and the chunk size is
            to be determined by `num_actors` instead.
        expect_size: If provided, the leftmost dimension of the given tensor
            or array or the number of rows of the given TensorFrame will be
            compared to this given number. If the size of `x` does not match
            `expect_size`, an error will be raised.
        target_device: If provided, the chunks will be moved into this device
            (except when `x` is an `ObjectArray` which will then stay on the
            cpu regardless of the given `target_device`).
    Returns:
        A named tuple in which the attribute `chunks` stores the chunks in a
        list, and `original_size` represents the size of `x` before the
        splitting operation.
    """
    if not isinstance(x, (torch.Tensor, TensorFrame, ObjectArray)):
        raise TypeError(f"Expected a tensor or a TensorFrame or an ObjectArray, but got an instance of {type(x)}")

    if isinstance(x, torch.Tensor) and (x.ndim == 0):
        raise ValueError("Cannot split a 0-dimensional tensor into chunks")

    if (target_device is not None) and isinstance(x, (torch.Tensor, TensorFrame)):
        # If we are given a target device, we move the original `x` to that device, so that its chunks will be
        # on that device as well.
        x = x.to(device=target_device)

    tensor_size = len(x)

    if (expect_size is not None) and (tensor_size != expect_size):
        raise ValueError("While trying to split tensors into chunks, encountered incompatible tensor sizes")

    # Compute the chunk sizes
    if chunk_size is None:
        if tensor_size == 0:
            raise ValueError("Cannot split a tensor whose leftmost dimension size is 0")
        elif tensor_size < num_actors:
            chunk_sizes = [1 for _ in range(tensor_size)]
        else:
            min_chunk_size = tensor_size // num_actors
            remaining = tensor_size % num_actors
            chunk_sizes = [min_chunk_size for _ in range(num_actors)]
            for i in range(remaining):
                chunk_sizes[i] += 1
    else:
        if chunk_size >= tensor_size:
            raise ValueError(
                "Cannot split the tensor into chunks because the given chunk size"
                " is larger than or equal to the original tensor size."
            )
        min_num_chunks = tensor_size // chunk_size
        last_chunk_size = tensor_size % chunk_size
        chunk_sizes = [chunk_size for _ in range(min_num_chunks)]
        if last_chunk_size > 0:
            chunk_sizes.append(last_chunk_size)

    # Prepare the chunks
    chunks = []
    i = 0
    j = 0
    for chunk_size in chunk_sizes:
        j = i + chunk_size
        if isinstance(x, (torch.Tensor, ObjectArray)):
            chunk = x[i:j]
        elif isinstance(x, TensorFrame):
            chunk = x.pick[i:j, :]
        else:
            raise TypeError("Execution should not have reached this point. This is most probably a bug.")
        chunks.append(chunk)
        i = j

    return _TensorSplittingResult(chunks=chunks, original_size=tensor_size)


class _DictSplittingResult(NamedTuple):
    chunks: list[dict[Any, TensorLike]]
    original_size: int


def _split_dict(
    x: Mapping[Any, TensorLike],
    num_actors: int,
    *,
    chunk_size: int | None = None,
    expect_size: int | None = None,
    target_device: str | torch.device | None = None,
) -> _DictSplittingResult:
    """
    Split the tensors/`TensorFrame`s/`ObjectArray`s in a dictionary-like object.

    Args:
        x: A shallow (non-nested) dictionary-like object (that is an instance
            of `collections.abc.Mapping`) that contains tensors and/or
            `TensorFrame`s and/or `ObjectArray`s.
        num_actors: Number of remote actors, as an integer that is at least 2.
            If `chunk_size` is not provided (i.e. left as None), the number
            of chunks and the size of each chunk will be determined by this
            `num_actors`.
        chunk_size: The size of a chunk when splitting a tensor/array into
            chunks. If this is provided, then this will be the main factor for
            determining the chunk size and also the number of chunks.
            Can be left as None if the number of chunks and the chunk size is
            to be determined by `num_actors` instead.
        expect_size: If provided, the leftmost dimension of the contained
            tensors or `ObjectArray`s or the number of rows of the contained
            `TensorFrame`s will be compared to this given number.
            If the tensor/array sizes within `x` do not match `expect_size`, an
            error will be raised.
        target_device: If provided, the chunks will be moved into this device
            (except when an `ObjectArray` is encountered which will be kept
            on the cpu regardless of the given `target_device`).
    Returns:
        A named tuple in which the attribute `chunks` stores a list of
        dictionaries (the values in each dictionary being the chunks of
        tensors/arrays), `original_size` represents the original tensor/array
        size within `x` before the splitting operation.
    """

    if len(x) == 0:
        raise ValueError(
            "Cannot split the tensor values into chunks within the given dictionary,"
            " because the given dictionary is empty"
        )

    dict_chunks: list[dict[Any, TensorLike]] | None = None
    original_tensor_size: int | None = expect_size
    num_chunks: int | None = None
    for k, v in x.items():
        chunks, tensor_size = _split_tensor(
            v, num_actors, chunk_size=chunk_size, expect_size=original_tensor_size, target_device=target_device
        )

        if original_tensor_size is None:
            original_tensor_size = tensor_size

        if dict_chunks is None:
            num_chunks = len(chunks)
            dict_chunks = [{} for _ in range(num_chunks)]

        for i_chunk in range(num_chunks):
            dict_chunks[i_chunk][k] = chunks[i_chunk]

    return _DictSplittingResult(chunks=dict_chunks, original_size=original_tensor_size)


class _SequenceSplittingResult(NamedTuple):
    chunks: list[list[TensorLike] | tuple[TensorLike, ...]]
    original_size: int


def _split_sequence(
    x: Sequence[TensorLike],
    num_actors: int,
    *,
    chunk_size: int | None = None,
    expect_size: int | None = None,
    target_device: str | torch.device | None = None,
) -> _SequenceSplittingResult:
    """
    Split the tensors/`TensorFrame`s/`ObjectArray`s in a sequence.

    Args:
        x: A shallow (non-nested) sequence (that is an instance of
            `collections.abc.Sequence`) that contains tensors and/or
            `TensorFrame`s and/or `ObjectArray`s.
        num_actors: Number of remote actors, as an integer that is at least 2.
            If `chunk_size` is not provided (i.e. left as None), the number
            of chunks and the size of each chunk will be determined by this
            `num_actors`.
        chunk_size: The size of a chunk when splitting a tensor/array into
            chunks. If this is provided, then this will be the main factor for
            determining the chunk size and also the number of chunks.
            Can be left as None if the number of chunks and the chunk size is
            to be determined by `num_actors` instead.
        expect_size: If provided, the leftmost dimension of the contained
            tensors or `ObjectArray`s or the number of rows of the contained
            `TensorFrame`s will be compared to this given number.
            If the tensor/array sizes within `x` do not match `expect_size`, an
            error will be raised.
        target_device: If provided, the chunks will be moved into this device.
            (except when an `ObjectArray` is encountered which will be kept
            on the cpu regardless of the given `target_device`).
    Returns:
        A named tuple in which the attribute `chunks` is a list of sequences
        (the items within each sequence being the chunks of tensors/arrays),
        `original_size` represents the original tensor/array size within `x`
        before the splitting operation.
    """
    result_must_be_tuple = False
    if isinstance(x, tuple):
        if hasattr(x, "_fields"):
            raise TypeError("Named tuples are not supported")
        result_must_be_tuple = True

    if len(x) == 0:
        raise ValueError(
            "Cannot split the tensor values into chunks within the given sequence,"
            " because the given sequence is empty"
        )

    sequence_chunks: list[list[TensorLike]] | None = None
    original_tensor_size: int | None = expect_size
    num_chunks: int | None
    for v in x:
        chunks, tensor_size = _split_tensor(
            v, num_actors, chunk_size=chunk_size, expect_size=original_tensor_size, target_device=target_device
        )

        if original_tensor_size is None:
            original_tensor_size = tensor_size

        if sequence_chunks is None:
            num_chunks = len(chunks)
            sequence_chunks = [[] for _ in range(num_chunks)]

        for i_chunk in range(num_chunks):
            sequence_chunks[i_chunk].append(chunks[i_chunk])

    if result_must_be_tuple:
        sequence_chunks = [tuple(item) for item in sequence_chunks]

    return _SequenceSplittingResult(chunks=sequence_chunks, original_size=original_tensor_size)


def split_into_chunks(
    x: TensorLike | Sequence[TensorLike] | Mapping[Any, TensorLike],
    num_actors: int,
    *,
    chunk_size: int | None = None,
    expect_size: int | None = None,
    target_device: str | torch.device | None = None,
) -> _TensorSplittingResult | _DictSplittingResult | _SequenceSplittingResult:
    """
    Split into chunks a tensor/ObjectArray/TensorFrame or a container of them.

    Args:
        x: A tensor, or a `TensorFrame`, or an `ObjectArray`, or a shallow
            (non-nested) dictionary-like container or a sequence containing one
            or more tensor/`TensorFrame`/`ObjectArray`. This is the input that
            is subject to splitting into chunks.
        num_actors: Number of remote actors, as an integer that is at least 2.
            If `chunk_size` is not provided (i.e. left as None), the number
            of chunks and the size of each chunk will be determined by this
            `num_actors`.
        chunk_size: The size of a chunk when splitting a tensor/array into
            chunks. If this is provided, then this will be the main factor for
            determining the chunk size and also the number of chunks.
            Can be left as None if the number of chunks and the chunk size is
            to be determined by `num_actors` instead.
        expect_size: If provided, the leftmost dimension of the contained
            tensors or `ObjectArray`s or the number of rows of the contained
            `TensorFrame`s will be compared to this given number.
            If the tensor/array sizes within `x` do not match `expect_size`, an
            error will be raised.
        target_device: If provided, the chunks will be moved into this device.
            (except when an `ObjectArray` is encountered which will be kept
            on the cpu regardless of the given `target_device`).
    Returns:
        A named tuple in which `chunks` is a list containing the chunks of `x`,
        `original_size` is the original size of `x`.
    """
    if expect_size is not None:
        expect_size = int(expect_size)
    if isinstance(x, (str, np.str_, bytes, bytearray)):
        # Here, we actively prevent objects that are technically instances of collections.abc.Sequence
        # but cannot contain any tensor/TensorFrame/ObjectArray
        raise TypeError(f"Unsupported type: {type(x)}")
    elif isinstance(x, Mapping):
        result = _split_dict(x, num_actors, chunk_size=chunk_size, expect_size=expect_size, target_device=target_device)
    elif isinstance(x, Sequence):
        result = _split_sequence(
            x, num_actors, chunk_size=chunk_size, expect_size=expect_size, target_device=target_device
        )
    elif isinstance(x, (torch.Tensor, TensorFrame, ObjectArray)):
        result = _split_tensor(
            x, num_actors, chunk_size=chunk_size, expect_size=expect_size, target_device=target_device
        )
    else:
        raise TypeError(f"Unsupported type: {type(x)}")

    return result


def split_arguments_into_chunks(
    args: Sequence,
    split_arguments: Sequence[bool],
    num_actors: int,
    *,
    chunk_size: int | None = None,
    target_device: str | torch.device | None = None,
) -> list:
    """
    Split the specified arguments within the given sequence into chunks.

    Splittable arguments are tensors, `TensorFrame`s, `ObjectArray`s,
    or shallow dictionary-like containers or sequences consisting of
    tensors and/or `TensorFrame`s and/or `ObjectArray`s.

    Args:
        args: A sequence (e.g. list or tuple) of arguments.
        split_arguments: A sequence of booleans. Within this sequence,
            if the i-th element is True, then the i-th element of `args`
            is subject to being split into chunks. On the other hand,
            if the i-th element of `split_arguments` is False, then the
            i-th element of `args` is going to be duplicated as it is,
            instead of being split.
        num_actors: Number of remote actors, as an integer that is at least 2.
            If `chunk_size` is not provided (i.e. left as None), the number
            of chunks and the size of each chunk will be determined by this
            `num_actors`.
        chunk_size: The size of a chunk when splitting a tensor/array into
            chunks. If this is provided, then this will be the main factor for
            determining the chunk size and also the number of chunks.
            Can be left as None if the number of chunks and the chunk size is
            to be determined by `num_actors` instead.
        target_device: If provided, the chunks will be moved into this device.
            (except when an `ObjectArray` is encountered which will be kept
            on the cpu regardless of the given `target_device`).
    Returns:
        A list of argument chunks. This returned list has the same length
        with `args`, but within it, each element is a list of chunks.
    """
    if isinstance(args, tuple) and hasattr(args, "_fields"):
        raise TypeError("`args` cannot be given in the form of a named tuple")
    if not isinstance(args, Sequence):
        raise TypeError(f"Expected `args` as a Sequence, but received it as an instance of {type(args)}")
    if isinstance(args, (str, np.str_, bytes, bytearray)):
        # Here, we actively prevent `args` from being given as instances of types that are technically
        # sequences, but that cannot contain arguments.
        raise TypeError(f"Received `args` as an instance of {type(args)}, which is not supported")

    num_args = len(args)
    if len(split_arguments) != num_args:
        raise TypeError(f"Expected {len(split_arguments)} positional arguments, but got {num_args}")

    # Understand which arguments are subject to splitting, and which arguments are subject to duplication
    arg_indices_to_split = []
    arg_indices_to_duplicate = []
    for i_arg, split_arg in enumerate(split_arguments):
        if split_arg:
            arg_indices_to_split.append(i_arg)
        else:
            arg_indices_to_duplicate.append(i_arg)

    if len(arg_indices_to_split) == 0:
        raise ValueError("None of the positional arguments were marked for being split into chunks")

    # The following list is to store chunks for each argument
    result = [None for _ in range(num_args)]

    # Loop over the arguments to split first
    original_size = None
    num_chunks = None
    for i_arg in arg_indices_to_split:
        # Split the argument into chunks
        chunks, tensor_size = split_into_chunks(
            args[i_arg], num_actors, chunk_size=chunk_size, expect_size=original_size, target_device=target_device
        )
        # Make sure that we know the original size and the number of chunks
        if original_size is None:
            original_size = tensor_size
            num_chunks = len(chunks)
        # Put the chunks
        result[i_arg] = chunks

    # Now that we have split all the arguments that are marked for splitting and that we know what is the chunk size,
    # we can now loop over the arguments that are marked for duplication.
    for i_arg in arg_indices_to_duplicate:
        result[i_arg] = [args[i_arg] for _ in range(num_chunks)]

    return result


def _all_are_instances(objects: Iterable, type_or_types: type | tuple[type, ...]) -> bool:
    """
    Return True if all the given objects match the given type(s).

    Args:
        objects: An iterable of objects whose types are being queried
        type_or_types: A type or a tuple of types
    Returns:
        True if the types of the given `objects` match the provided
        `type_or_types`;
        False otherwise.
    """
    for obj in objects:
        if not isinstance(obj, type_or_types):
            return False
    return True


def _all_are_non_scalars(tensors: Iterable[torch.Tensor]) -> bool:
    """
    Return True if `tensors` are all non-scalars (having 1 or more dimensions).

    Args:
        tensors: An iterable of PyTorch tensors.
    Returns:
        True if all `tensors` are non-scalars; False otherwise.
    """
    for t in tensors:
        if t.ndim == 0:
            return False
    return True


def _ensure_chunk_lengths_are_valid(objects_with_length: Sequence, expected_lengths: Sequence[int] | None):
    """
    Ensure that the lengths of the objects match the desired lengths.

    Args:
        objects_with_length: A sequence of objects that have lengths (i.e.
            that have the method `__len__`).
        expected_lengths: A sequence of integers. The length of the i-th
            element within `objects_with_length` must match the i-th integer
            within `expected_lengths`.
    Raises:
        ValueError: if the lengths do not match.
    """
    if expected_lengths is None:
        return
    for obj, expected_len in zip(objects_with_length, expected_lengths, strict=True):
        if len(obj) != expected_len:
            raise ValueError("Received a chunk with an unexpected size")


def _stack_chunked_tensors(
    chunks: Sequence[TensorLike], *, expect_chunk_sizes: Sequence[int] | None = None
) -> TensorLike:
    """
    Stack chunks of tensors/`TensorFrame`s/`ObjectArray`s.

    If `chunks` consists of tensors, those tensors will be concatenated along
    their leftmost dimensions.
    If `chunks` consists of `TensorFrame`s, those frames will be vertically
    stacked.
    If `chunks` consists of `ObjectArray`s, those arrays will be concatenated.

    Args:
        chunks: A sequence consisting of tensors/`TensorFrame`s/`ObjectArray`s.
        expect_chunk_sizes: If given, it will be expected that the size of the
            i-th item within `chunks` matches the i-th integer within
            `expect_chunk_sizes`.
    Returns:
        The combined tensor/TensorFrame/ObjectArray.
    Raises:
        ValueError: if the i-th chunk does not have the size specified by the
            i-th integer within the provided `expect_chunk_sizes`.
    """
    from .tools import as_tensor

    if not isinstance(chunks, Sequence):
        raise TypeError(f"`chunks` was expected as a Sequence, but it was received as an instance of {type(chunks)}")
    if isinstance(chunks, tuple) and hasattr(chunks, "_fields"):
        raise TypeError("`chunks` in the form of a named tuple is not supported")
    if not isinstance(chunks, list):
        chunks = list(chunks)

    num_chunks = len(chunks)
    if num_chunks == 0:
        raise ValueError("Cannot operate on an empty list of chunks")
    if (expect_chunk_sizes is not None) and (num_chunks != len(expect_chunk_sizes)):
        raise ValueError("Received an unexpected number of chunks")

    resulting_stack: torch.Tensor | TensorFrame

    if _all_are_instances(chunks, torch.Tensor):
        if _all_are_non_scalars(chunks):
            _ensure_chunk_lengths_are_valid(chunks, expect_chunk_sizes)
            resulting_stack = torch.cat(chunks)
        else:
            raise ValueError("Received a chunk in the form of a scalar tensor, which is unexpected")
    elif _all_are_instances(chunks, ObjectArray):
        _ensure_chunk_lengths_are_valid(chunks, expect_chunk_sizes)
        resulting_stack = as_tensor(list(chain(chunks)), dtype=object)
        got_read_only = False
        for chunk in chunks:
            if chunk.is_read_only:
                got_read_only = True
                break
        if got_read_only:
            resulting_stack = resulting_stack.get_read_only_view()
    elif _all_are_instances(chunks, TensorFrame):
        _ensure_chunk_lengths_are_valid(chunks, expect_chunk_sizes)
        for i_chunk, chunk in enumerate(chunks):
            if i_chunk == 0:
                resulting_stack = chunk
            else:
                resulting_stack = resulting_stack.vstack(chunk)
    else:
        raise TypeError("Encountered some unsupported types in the chunk, or their types are inconsistent")

    return resulting_stack


def _keys_of_all_dicts(dicts: Iterable[Mapping]) -> list:
    """
    Get the keys of all dictionary-like objects.

    Args:
        dicts: An iterable of dictionary-like objects (i.e. of instances of
            `collections.abc.Mapping`).
    Returns:
        The keys of the given dictionary-like objects.
    Raises:
        KeyError: if the keys of the given dictionary-like objects are not
            consistent.
        ValueError: if the given iterable does not provide any dictionary.
    """
    key_list: list | None = None
    key_set: set | None = None
    for d in dicts:
        if key_set is None:
            key_list = list(d.keys())
            key_set = set(d.keys())
        else:
            if set(d.keys()) != key_set:
                raise KeyError("The dictionaries have inconsistent keys")
    if key_list is None:
        raise ValueError("Cannot get the keys from an empty iterable of dictionaries")
    return key_list


def _stack_chunked_dicts(
    chunks: Sequence[Mapping[Any, TensorLike]], *, expect_chunk_sizes: Sequence[int] | None = None
) -> dict[Any, TensorLike]:
    """
    Combine multiple dictionaries of chunked tensors into a single dictionary.

    Args:
        chunks: A sequence of dictionary-like objects (i.e. of instances of
            `collections.abc.Mapping`), in which each dictionary-like object
            represents a single chunk.
        expect_chunk_sizes: If provided, each tensor/TensorFrame/ObjectArray
            within the i-th dictionary-like object will be expected to have
            its size equal to the i-th integer within `expect_chunk_sizes`.
    Returns:
        A dictionary in which all the tensors/arrays are combined.
    """
    if not isinstance(chunks, Sequence):
        raise TypeError(f"`chunks` was expected as a Sequence, but it was received as an instance of {type(chunks)}")
    if not _all_are_instances(chunks, Mapping):
        raise TypeError("Some or all of the elements within the given sequence are not dictionaries")
    keys = _keys_of_all_dicts(chunks)
    return {
        k: _stack_chunked_tensors([dict_chunk[k] for dict_chunk in chunks], expect_chunk_sizes=expect_chunk_sizes)
        for k in keys
    }


def _length_of_all_sequences(sequences: Iterable[Sequence]) -> int:
    """
    Get the length of the given sequences.

    Args:
        sequences: An iterable of sequences. Within this iterable, each
            sequence's length will be checked.
    Returns:
        The length of the given sequences.
    Raises:
        ValueError: if the given sequences have inconsistent lengths, or if the
            iterable does not provide any sequence at all.
    """
    n: int | None = None
    for s in sequences:
        if n is None:
            n = len(s)
        else:
            if len(s) != n:
                raise ValueError("The sequences have inconsistent lengths")
    if n is None:
        raise ValueError("Cannot get the sequence length from an empty iterable of sequences")
    return n


def _stack_chunked_sequences(
    chunks: Sequence[Sequence[TensorLike]], *, expect_chunk_sizes: Sequence[int] | None = None
) -> list[TensorLike]:
    """
    Combine multiple sequences of chunked tensors into a single list.

    Args:
        chunks: A sequence consisting of sequences of
            tensors/`TensorFrame`s/`ObjectArray`s.
        expect_chunk_sizes: If this is provided, the
            tensors/`TensorFrame`s/`ObjectArray`s within the i-th sequence
            is expected to have their lengths equal to the i-th integer within
            `expect_chunk_sizes`.
    Returns:
        A list in which all the tensors/arrays are combined.
    """
    if not isinstance(chunks, Sequence):
        raise TypeError(f"`chunks` was expected as a Sequence, but it was received as an instance of {type(chunks)}")
    if not _all_are_instances(chunks, Sequence):
        raise TypeError("Some or all of the elements within the given sequence are not sequences")

    sequence_maker = list
    for c in chunks:
        if isinstance(c, tuple):
            sequence_maker = tuple
            if hasattr(c, "_fields"):
                raise TypeError("Chunks in the form of named tuple are not supported")

    n = _length_of_all_sequences(chunks)
    result = sequence_maker(
        _stack_chunked_tensors([sequence_chunk[k] for sequence_chunk in chunks], expect_chunk_sizes=expect_chunk_sizes)
        for k in range(n)
    )

    return result


def stack_chunks(
    chunks: Sequence[Sequence[TensorLike]] | Sequence[Mapping[Any, TensorLike]] | Sequence[TensorLike],
    *,
    expect_chunk_sizes: Sequence[int] | None = None,
) -> list[TensorLike] | Mapping[Any, TensorLike] | TensorLike:
    """
    Stack the given tensors/arrays across the given chunks.

    Each chunk can be a tensor or a TensorFrame or an ObjectArray, or a
    dictionary-like object or a sequence consisting of
    tensors/`TensorFrame`s/`ObjectArray`s.

    Args:
        chunks: A sequence in which each item is a chunk.
        expect_chunk_sizes: If this is given, the tensor/array sizes within
            the i-th chunk will be expected to be equal to the i-th integer
            within `expect_chunk_sizes`.
    Returns:
        A combined tensor/array, or a dictionary-like object or a sequence
        which contains combined tensors/arrays.
    Raises:
        ValueError: if the tensor/array lengths do not match with what is
            specified within `expect_chunk_sizes`.
    """
    if _all_are_instances(chunks, Sequence):
        return _stack_chunked_sequences(chunks, expect_chunk_sizes=expect_chunk_sizes)
    elif _all_are_instances(chunks, Mapping):
        return _stack_chunked_dicts(chunks, expect_chunk_sizes=expect_chunk_sizes)
    elif _all_are_instances(chunks, (torch.Tensor, ObjectArray, TensorFrame)):
        return _stack_chunked_tensors(chunks, expect_chunk_sizes=expect_chunk_sizes)
    else:
        raise TypeError(
            "Received a sequence in which some or all elements have unsupported types,"
            " or in which the element types are inconsistent"
        )


class _FunctionWrapInfo(NamedTuple):
    function: Callable
    num_actors: str | int | None
    chunk_size: int | None
    num_gpus_per_actor: int | float | str | None
    split_arguments: tuple[bool, ...]
    devices: tuple[torch.device, ...]


class _LockForTheMainProcess:
    """
    A lock (in the context of threading) that is meant for the main process.

    Just like a regular `threading.Lock`, the instances of this class can be
    used with the help of a `with` statement:

    ```python
    my_lock = _LockForTheMainProcess()

    ...

    with my_lock:
        ...  # critical actions go here
    ```

    The differences of this class from `threading.Lock` are as follows:

    - Instances of this class are picklable (no error will be raised).
    - Although picklable, based on the assumption that this type of lock
        is meant only for the main process, the locking capabilities
        of the instances of this class will disappear once they are
        pickled and unpickled.
    - If the locking capabilities of an instance of this class have
        disappeared, trying the `with` statement on them will cause an
        error.
    - Objects containing the instances of this class can be serialized
        and distributed by the `ray` library. However, the actual locking
        capabilities will be available only to the ones on the main process.
    """

    def __init__(self):
        self._lock: Lock | None = Lock()

    def _ensure_lock_exists(self):
        if self._lock is None:
            selfname = type(self).__name__
            raise RuntimeError(f"This {selfname} was pickled and then unpickled. It cannot be used anymore as a lock.")

    def __enter__(self):
        self._ensure_lock_exists()
        self._lock.acquire()

    def __exit__(self, exc_type, exc_value, traceback):
        self._ensure_lock_exists()
        self._lock.release()

    def __getstate__(self) -> dict:
        result = {}
        for k, v in self.__dict__.items():
            if k == "_lock":
                result[k] = None
            else:
                result[k] = v
        return result


def _loosely_find_leftmost_dimension_size(
    x: TensorLike | Sequence[TensorLike] | Mapping[Any, TensorLike], *, _recurse: bool = True
) -> int:
    """
    Find the leftmost dimension of the given tensors/arrays.

    If `x` is given as a tensor or as a TensorFrame or as an ObjectArray,
    its leftmost dimension's size will be returned.
    If `x` is given as a sequence or as a dictionary-like object, the leftmost
    dimension size of the first tensor/array encountered within it will be
    returned.

    This function assumes that the tensor/array size consistency within the
    given container is checked elsewhere. With this assumption in mind,
    consistency check will not be performed by this function.

    Args:
        x: A tensor or a TensorFrame or an ObjectArray, or a sequence or a
            dictionary-like object containing tensors/arrays.
        _recurse: For internal usage.
    Returns:
        An integer representing the leftmost dimension size.
    """
    if isinstance(x, (ObjectArray, TensorFrame)):
        return len(x)
    elif isinstance(x, torch.Tensor):
        return x.shape[0]
    elif isinstance(x, (str, np.str_, bytes, bytearray)):
        raise TypeError(f"Received a sequence of this unexpected type: {type(x)}")
    elif isinstance(x, (Mapping, Sequence)):
        if not _recurse:
            raise TypeError("Found a container when expecting a tensor or a TensorFrame or an ObjectArray")
        if len(x) == 0:
            raise ValueError("Encountered an empty container, which is unexpected")
        if isinstance(x, Mapping):
            elements = x.values()
        else:
            elements = x
        first_element = next(iter(elements))
        return _loosely_find_leftmost_dimension_size(first_element, _recurse=False)
    else:
        raise TypeError(f"Encountered an object of this unexpected type: {type(x)}")


class _Wrapped:
    functions: dict[_FunctionWrapInfo, Callable] = {}
    lock = _LockForTheMainProcess()


class _DistributedFunctionHandler(Problem):
    """
    Handler for a function that is decorated via `@distribute`.

    Although this handler is not meant to express an optimization problem,
    it is built as a subclass of `evotorch.Problem`, for taking advantage
    of multi-actor parallelization capabilities of the Problem class.

    **How does it work internally?**
    This handler declares itself as a dummy optimization problem which
    requires parallelization. The configuration arguments it receives regarding
    parallelization are passed to the initializer of its parent class,
    `Problem`. Additionally, upon its initialization, it receives the original
    form of the decorated function and stores a reference to that function
    within itself.

    Once this handler receives a request to execute the referenced function
    in a distributed (i.e. parallelized) manner (via its method
    `call_wrapped_function`), it forces its superclass (Problem) to create
    remote actors by performing a dummy solution batch evaluation.
    Once the remote actors are created, the input arguments to the wrapped
    function are split into chunks, those chunks are then sent to the
    remote actors along with a request to apply the wrapped function on them,
    and finally the results of the actors are collected and combined.
    Note that the wrapped function is called by the actors in parallel, which
    is the main goal of this handler.
    """

    def __init__(
        self,
        *,
        function: Callable,
        num_actors: str | int | None = None,
        chunk_size: int | None = None,
        num_gpus_per_actor: int | float | str | None,
        split_arguments: tuple[bool, ...],
        devices: tuple[torch.device, ...],
    ):
        """
        `__init__(...)`: Initialize the `_DistributedFunctionHandler`.

        Args:
            function: The reference to the original form of the function
                to be distributed across multiple remote actors.
            num_actors: Number of remote actors.
            chunk_size: Optionally, the size of a chunk as an integer.
                If this is given, then the original arguments will be split
                into chunks with at most this given size.
            num_gpus_per_actor: Number of GPUs to be allocated by each actor.
            split_arguments_into_chunks: A tuple of booleans, in which the i-th
                boolean says if the i-th positional argument for the wrapped
                function is expected as split into chunks (True), or is to be
                duplicated for each remote actor (False).
                If this is given as an empty tuple, it will be assumed that
                all the positional arguments are to be split into chunks.
            devices: A tuple of devices. If this tuple is not empty, then the
                i-th actor will use the i-th device listed within `devices`.
                If this argument is to be provided as a non-empty tuple,
                and if `devices` are going to be other than just cpus,
                then it is highly recommended to set `num_gpus_per_actor` as
                "all", so that the same devices will be visible to all actors.
                The `@distribute` decorator, when using this handler class
                internally, automatically sets `num_gpus_per_actor` as "all"
                when a non-empty `devices` argument is provided.
        """
        self.__function = function
        self.__chunk_size = chunk_size
        self.__split_arguments = split_arguments
        self.__devices = devices
        self.__parallelized = False
        self.__parallelization_lock = _LockForTheMainProcess()
        self.__actor_pool = None

        super().__init__(
            objective_sense="min",
            solution_length=2,
            initial_bounds=(-1.0, 1.0),
            dtype=torch.float32,
            device="cpu",
            num_actors=num_actors,
            num_gpus_per_actor=num_gpus_per_actor,
            store_solution_stats=False,
        )

    def _evaluate_batch(self, x: SolutionBatch):
        """
        Just a filler batch evaluation procedure.
        """
        z = torch.zeros(len(x), dtype=x.eval_dtype, device=x.device)
        x.set_evals(z)

    def _ensure_dummy_problem_is_parallelized(self):
        """
        Internal method for ensuring that the remote actors are created.
        """
        if self.is_remote:
            # If we are on a remote actor, this check is not necessary. We just exit the function.
            return

        with self.__parallelization_lock:
            if not self.__parallelized:
                # This is the case where the problem has not been parallelized yet (i.e. we do not have actors yet).
                # To trigger the creation of the actors, we generate a dummy SolutionBatch and evaluate it.
                # The creation of the actors is then managed by the `evaluate` method of the parent Problem class.
                dummy_batch = SolutionBatch(self, popsize=1)
                self.evaluate(dummy_batch)
                self.__parallelized = True

        if (self.actors is None) or (len(self.actors) < 2):
            raise RuntimeError(
                "Failed to create the distributed counterpart of the original function."
                " Hint: this can happen if the arguments given to the `@distribute` decorator imply a non-distributed"
                " environment, e.g., if one sets `num_actors='num_gpus'` when one has only 1 GPU,"
                " or if one sets `num_actors` as an integer that is smaller than 2."
            )

        # NOTE: do we need this, or could we actually use the actor pool of the underlying Problem?
        self.__actor_pool = ActorPool(self.actors)

    def _iter_split_arguments(self, args: Sequence):
        num_split_arguments = len(self.__split_arguments)
        if num_split_arguments == 0:
            for _ in range(len(args)):
                yield True
        else:
            if num_split_arguments != len(args):
                raise TypeError(
                    f"The number of received positional arguments ({len(args)})"
                    f" is different than what is expected ({len(self.__split_arguments)})"
                )

            for split_arg in self.__split_arguments:
                yield split_arg

    def _call_wrapped_function_remotely(self, task_index: int, args: tuple) -> tuple[int, Any]:
        """
        Internal helper method for calling the wrapped function on an actor.

        Args:
            task_index: The index of the task.
            args: Positional arguments to be passed to the wrapped function.
                The positional arguments that were marked to be split into
                chunks will be moved to the accelerator device associated with
                this actor.
        Returns:
            A tuple in the form `(task_index, result)` where `task_index` is
            the index of the task that was given, and `result` is the result
            of the wrapped function, moved back to the cpu.
        """

        if self.is_main:
            raise RuntimeError("This function should not be executed from the main actor")

        num_explicit_devices = len(self.__devices)
        prepared_args = []

        for split_arg, arg in zip(self._iter_split_arguments(args), args):
            if split_arg:
                if num_explicit_devices > 0:
                    target_device = self.__devices[self.actor_index % num_explicit_devices]
                else:
                    target_device = self.aux_device
                prepared_arg = move_shallow_container_to_device(arg, device=target_device)
            else:
                prepared_arg = arg
            prepared_args.append(prepared_arg)

        result = self.__function(*prepared_args)

        # Move the result of this function back to the cpu, and return it.
        return task_index, move_shallow_container_to_device(result, device="cpu")

    def call_wrapped_function(self, *args) -> Any:
        """
        Run the wrapped function across the remote actors.

        Args:
            args: Positional arguments to be passed to the wrapped function.
                If this class was initialized with a non-empty
                `split_arguments` tuple: i-th argument will be split into
                chunks if the i-th element within `split_arguments` is True,
                and the i-th chunk of arguments will be sent to the i-th actor.
                If this class was initialized with an empty `split_arguments`
                tuple: it will be assumed that all positional arguments are
                to be split into chunks.
                Note also that each actor will move its received chunks
                to its own associated accelerator device before applying the
                wrapped function on them.
        Returns:
            Combined result of the parallel computation of the remote actors.
            The results will be on the cpu.
        """

        if len(args) == 0:
            raise TypeError("Calling a distributed function without any positional arguments is not supported")
        if not self.is_main:
            raise RuntimeError("This method must be executed only from within the main actor")
        self._ensure_dummy_problem_is_parallelized()

        first_split_arg_index = None
        for i_arg, split_arg in enumerate(self._iter_split_arguments(args)):
            if split_arg:
                first_split_arg_index = i_arg
                break
        if first_split_arg_index is None:
            raise ValueError(
                "None of the arguments is marked for being split into chunks, which is not a supported configuration."
            )

        # split the arguments into chunks, BUT ONLY IF the argument is marked via `split_arguments`
        chunked_args = split_arguments_into_chunks(
            args,
            list(self._iter_split_arguments(args)),
            self.num_actors,
            chunk_size=self.__chunk_size,
            target_device="cpu",
        )
        num_chunks = len(chunked_args[first_split_arg_index])

        args_per_task = [[arg_chunk[i_task] for arg_chunk in chunked_args] for i_task in range(num_chunks)]
        chunk_size_per_task = [
            _loosely_find_leftmost_dimension_size(args_per_task[i_task][first_split_arg_index])
            for i_task in range(num_chunks)
        ]

        call_args_per_task = [
            ["_call_wrapped_function_remotely", [i_task, args_per_task[i_task]], {}] for i_task in range(num_chunks)
        ]

        unordered_map_result = list(
            self.__actor_pool.map_unordered(
                (lambda actor, chunk: actor.call.remote(*chunk)),
                call_args_per_task,
            )
        )

        assert len(unordered_map_result) == num_chunks

        ordered_map_result = [None for _ in range(num_chunks)]
        for i_task, returned_chunk in unordered_map_result:
            ordered_map_result[i_task] = returned_chunk

        # collect the remote results and combine the tensors
        result = stack_chunks(ordered_map_result, expect_chunk_sizes=chunk_size_per_task)

        return result


class _DistributedFunction:
    """
    A function that was decorated via `@distribute`.

    Please use the `@distribute` decorator instead of instantiating this class
    manually.
    """

    def __init__(self, wrap_info: _FunctionWrapInfo):
        self.wrap_info = wrap_info
        self.problem = _DistributedFunctionHandler(
            function=wrap_info.function,
            num_actors=wrap_info.num_actors,
            chunk_size=wrap_info.chunk_size,
            num_gpus_per_actor=wrap_info.num_gpus_per_actor,
            split_arguments=wrap_info.split_arguments,
            devices=wrap_info.devices,
        )
        if hasattr(self.wrap_info.function, "__evotorch_vectorized__"):
            self.__evotorch_vectorized__ = self.wrap_info.function.__evotorch_vectorized__
        if hasattr(self.wrap_info.function, "__evotorch_pass_info__"):
            self.__evotorch_pass_info__ = self.wrap_info.function.__evotorch_pass_info__
        self.__evotorch_distribute__ = True

    def __call__(self, *args):
        return self.problem.call_wrapped_function(*args)


def _prepare_distributed_function(
    function: Callable,
    *,
    split_arguments: Sequence[bool] | np.ndarray | torch.Tensor | None = None,
    num_actors: int | str | None = None,
    chunk_size: int | None = None,
    num_gpus_per_actor: int | float | str | None = None,
    devices: Sequence[torch.device | str],
) -> Callable:
    if split_arguments is None:
        split_arguments = tuple()

    if (not isinstance(split_arguments, Sequence)) or (isinstance(split_arguments, (str, np.str_, bytes, bytearray))):
        raise TypeError(
            f"`split_arguments` was expected as a Sequence of booleans, not as an instance of {repr(split_arguments)}"
        )

    if isinstance(split_arguments, tuple) and hasattr(split_arguments, "_fields"):
        raise ValueError("`split_arguments` in the form of named tuples is not supported")

    if len(split_arguments) > 0:
        # We are being extra careful here for ensuring that `split_arguments` is a sequence of booleans.
        # We want to actively prevent unexpected behavior that could be caused by these mistakes:
        # - providing argument indices instead of a sequence of booleans
        # - providing one or more argument names as strings, instead of a sequence of booleans
        _actual_split_arguments = []
        for split_arg in split_arguments:
            if isinstance(split_arg, torch.Tensor) and (split_arg.ndim == 0):
                _actual_split_arguments.append(bool(split_arg.to(device="cpu")))
            elif isinstance(split_arg, (bool, np.bool_)):
                _actual_split_arguments.append(bool(split_arg))
            else:
                raise TypeError("`split_arguments` was expected to contain booleans only")
        split_arguments = tuple(_actual_split_arguments)

    if devices is None:
        if (num_actors is None) or ((isinstance(num_actors, Integral)) and (num_actors <= 1)):
            raise ValueError(
                "The argument `devices` was received as None."
                " When `devices` is None, `num_actors` is expected as an integer that is at least 2."
                f" However, the given value of `num_actors` is {repr(num_actors)}."
            )
        devices = tuple()
    else:
        if isinstance(devices, tuple) and hasattr(devices, "_fields"):
            raise ValueError("`devices` in the form of a named tuple is not supported")
        devices = tuple(torch.device(item) for item in devices)
        num_devices = len(devices)
        if num_devices == 0:
            raise ValueError("`devices` cannot be given as an empty sequence")
        if num_actors is None:
            num_actors = num_devices
        else:
            raise ValueError(
                "The `argument` devices was received as provided as a value other than None."
                " When `devices` is not None, `num_actors` is expected to be left as None."
                f" However, it was received as {repr(num_actors)}."
            )

        # We are given an explicit sequence of devices.
        # Therefore, we assume that the actors must be able to see all the accelerator devices,
        # and therefore override `num_gpus_per_actor` as "all".
        if num_gpus_per_actor is None:
            num_gpus_per_actor = "all"
        else:
            raise ValueError(
                "The `argument` devices was received as provided as a value other than None."
                " When `devices` is not None, `num_gpus_per_actor` is expected to be left as None."
                f" However, it was received as {repr(num_gpus_per_actor)}."
            )

    # Prepare a wrap_info tuple which stores information about which function was wrapped with what configuration.
    wrap_info = _FunctionWrapInfo(
        function=function,
        split_arguments=split_arguments,
        num_actors=num_actors,
        chunk_size=chunk_size,
        num_gpus_per_actor=num_gpus_per_actor,
        devices=devices,
    )

    with _Wrapped.lock:
        if wrap_info in _Wrapped.functions:
            # According to our global wrapped functions dictionary, if this particular function was wrapped before
            # with these exact settings, we return the already wrapped version of the function.
            result = _Wrapped.functions[wrap_info]
        else:
            # If this is the first time we are wrapping this function with these settings, then we create a wrapped
            # version of this function, and put it into our global wrapped functions dictionary.
            result = _DistributedFunction(wrap_info)
            _Wrapped.functions[wrap_info] = result

    return result


class DecoratorForDistributingFunctions:
    """
    Parameterized wrapper for making distributed counterparts of functions.

    It is highly recommended to use the `@distribute` decorator instead.
    """

    def __init__(
        self,
        *,
        split_arguments: Sequence[bool] | np.ndarray | torch.Tensor | None = None,
        num_actors: str | int | None = None,
        chunk_size: int | None = None,
        num_gpus_per_actor: int | float | str | None = None,
        devices: Sequence[bool] | None = None,
    ):
        self.split_arguments = split_arguments
        self.num_actors = num_actors
        self.chunk_size = chunk_size
        self.num_gpus_per_actor = num_gpus_per_actor
        self.devices = devices

    def __call__(self, function: Callable) -> Callable:
        return _prepare_distributed_function(
            function,
            split_arguments=self.split_arguments,
            num_actors=self.num_actors,
            chunk_size=self.chunk_size,
            num_gpus_per_actor=self.num_gpus_per_actor,
            devices=self.devices,
        )
