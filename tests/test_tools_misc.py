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

from typing import Any

import numpy as np
import pytest
import torch
from torch import FloatTensor

from evotorch.decorators import pass_info
from evotorch.testing import assert_allclose, assert_dtype_matches
from evotorch.tools import misc
from evotorch.tools.objectarray import ObjectArray


def test_expect_none():
    with pytest.raises(ValueError):
        misc.expect_none("Testing", a=1, b=2)

    with pytest.raises(ValueError):
        misc.expect_none("Testing", a=None, b=2)

    with pytest.raises(ValueError):
        misc.expect_none("Testing", a=1, b=None)

    misc.expect_none("Testing", a=None, b=None)


@pytest.mark.parametrize(
    "type_,expected,err",
    [
        (object, None, TypeError),
        (Any, None, TypeError),
        ("object", None, TypeError),
        (np.float16, torch.float16, None),
        (np.float32, torch.float32, None),
        (np.float64, torch.float64, None),
        ("float16", torch.float16, None),
        ("float32", torch.float32, None),
        ("float64", torch.float64, None),
        (np.int16, torch.int16, None),
        (np.int32, torch.int32, None),
        (np.int64, torch.int64, None),
        ("int16", torch.int16, None),
        ("int32", torch.int32, None),
        ("int64", torch.int64, None),
    ],
)
def test_to_torch_dtype(type_, expected, err):
    if err is not None:
        with pytest.raises(err):
            misc.to_torch_dtype(type_)
    else:
        assert misc.to_torch_dtype(type_) is expected


@pytest.mark.parametrize(
    "type_,expected",
    [
        (torch.int16, np.dtype("int16")),
        (torch.int32, np.dtype("int32")),
        (torch.int64, np.dtype("int64")),
        (torch.float16, np.dtype("float16")),
        (torch.float32, np.dtype("float32")),
        (torch.float64, np.dtype("float64")),
        (np.int16, np.dtype("int16")),
        (np.int32, np.dtype("int32")),
        (np.int64, np.dtype("int64")),
        (np.float16, np.dtype("float16")),
        (np.float32, np.dtype("float32")),
        (np.float64, np.dtype("float64")),
        (np.dtype("int16"), np.dtype("int16")),
        (np.dtype("int32"), np.dtype("int32")),
        (np.dtype("int64"), np.dtype("int64")),
        (np.dtype("float16"), np.dtype("float16")),
        (np.dtype("float32"), np.dtype("float32")),
        (np.dtype("float64"), np.dtype("float64")),
        ("int16", np.dtype("int16")),
        ("int32", np.dtype("int32")),
        ("int64", np.dtype("int64")),
        ("float16", np.dtype("float16")),
        ("float32", np.dtype("float32")),
        ("float64", np.dtype("float64")),
        (object, np.dtype("O")),
        ("object", np.dtype("O")),
        (np.dtype("object"), np.dtype("O")),
        (np.dtype(object), np.dtype("O")),
        (np.dtype("O"), np.dtype("O")),
        (Any, np.dtype("O")),
    ],
)
def test_to_numpy_dtype(type_, expected):
    assert misc.to_numpy_dtype(type_) == expected


@pytest.mark.parametrize(
    "type_,expected",
    [
        (Any, True),
        (object, True),
        ("Any", True),
        ("object", True),
        ("O", True),
        ("int8", False),
        (np.int8, False),
        (np.dtype("int8"), False),
        (torch.int8, False),
        ("int16", False),
        (np.int16, False),
        (np.dtype("int16"), False),
        (torch.int16, False),
        ("int32", False),
        (np.int32, False),
        (np.dtype("int32"), False),
        (torch.int32, False),
        ("int64", False),
        (np.int64, False),
        (np.dtype("int64"), False),
        (torch.int64, False),
        ("float16", False),
        (np.float16, False),
        (np.dtype("float16"), False),
        (torch.float16, False),
        ("float32", False),
        (np.float32, False),
        (np.dtype("float32"), False),
        (torch.float32, False),
        ("float64", False),
        (np.float64, False),
        (np.dtype("float64"), False),
        (torch.float64, False),
    ],
)
def test_is_dtype_object(type_, expected):
    assert misc.is_dtype_object(type_) is expected


@pytest.mark.parametrize(
    "obj,expected",
    [
        ("string", False),
        (b"bytes", False),
        (10, False),
        (10.0, False),
        ([1, 2], True),
        ((1, 2), True),
        (np.int32(0), False),
        (np.float32(0.0), False),
        (np.array(10), False),
        (np.array(10.0), False),
        (torch.tensor(10), False),
        (torch.tensor(10.0), False),
        (np.array([1, 2]), True),
        (np.array([1.0, 2.0]), True),
        (torch.tensor([1, 2]), True),
        (torch.tensor([1.0, 2.0]), True),
        (np.array([[1, 2], [3, 4]]), True),
        (np.array([[1.0, 2.0], [3.0, 4.0]]), True),
        (torch.tensor([[1, 2], [3, 4]]), True),
        (torch.tensor([[1.0, 2.0], [3.0, 4.0]]), True),
    ],
)
def test_is_sequence(obj, expected):
    assert misc.is_sequence(obj) is expected


@pytest.mark.parametrize(
    "type_,expected",
    [
        (Any, False),
        (object, False),
        ("Any", False),
        ("object", False),
        ("O", False),
        ("int8", True),
        (np.int8, True),
        (np.dtype("int8"), True),
        (torch.int8, True),
        ("int16", True),
        (np.int16, True),
        (np.dtype("int16"), True),
        (torch.int16, True),
        ("int32", True),
        (np.int32, True),
        (np.dtype("int32"), True),
        (torch.int32, True),
        ("int64", True),
        (np.int64, True),
        (np.dtype("int64"), True),
        (torch.int64, True),
        ("uint8", True),
        (np.uint8, True),
        (np.dtype("uint8"), True),
        (torch.uint8, True),
        ("uint16", True),
        (np.uint16, True),
        (np.dtype("uint16"), True),
        ("uint32", True),
        (np.uint32, True),
        (np.dtype("uint32"), True),
        ("uint64", True),
        (np.uint64, True),
        (np.dtype("uint64"), True),
        ("float16", False),
        (np.float16, False),
        (np.dtype("float16"), False),
        (torch.float16, False),
        ("float32", False),
        (np.float32, False),
        (np.dtype("float32"), False),
        (torch.float32, False),
        ("float64", False),
        (np.float64, False),
        (np.dtype("float64"), False),
        (torch.float64, False),
        ("bool", False),
        (bool, False),
        (np.bool_, False),
        (torch.bool, False),
    ],
)
def test_is_dtype_integer(type_, expected):
    assert misc.is_dtype_integer(type_) is expected


@pytest.mark.parametrize(
    "type_,expected",
    [
        (Any, False),
        (object, False),
        ("Any", False),
        ("object", False),
        ("O", False),
        ("int8", False),
        (np.int8, False),
        (np.dtype("int8"), False),
        (torch.int8, False),
        ("int16", False),
        (np.int16, False),
        (np.dtype("int16"), False),
        (torch.int16, False),
        ("int32", False),
        (np.int32, False),
        (np.dtype("int32"), False),
        (torch.int32, False),
        ("int64", False),
        (np.int64, False),
        (np.dtype("int64"), False),
        (torch.int64, False),
        ("uint8", False),
        (np.uint8, False),
        (np.dtype("uint8"), False),
        (torch.uint8, False),
        ("uint16", False),
        (np.uint16, False),
        (np.dtype("uint16"), False),
        ("uint32", False),
        (np.uint32, False),
        (np.dtype("uint32"), False),
        ("uint64", False),
        (np.uint64, False),
        (np.dtype("uint64"), False),
        ("float16", True),
        (np.float16, True),
        (np.dtype("float16"), True),
        (torch.float16, True),
        ("float32", True),
        (np.float32, True),
        (np.dtype("float32"), True),
        (torch.float32, True),
        ("float64", True),
        (np.float64, True),
        (np.dtype("float64"), True),
        (torch.float64, True),
        ("bool", False),
        (bool, False),
        (np.bool_, False),
        (torch.bool, False),
    ],
)
def test_is_dtype_float(type_, expected):
    assert misc.is_dtype_float(type_) is expected


@pytest.mark.parametrize(
    "type_,expected",
    [
        (Any, False),
        (object, False),
        ("Any", False),
        ("object", False),
        ("O", False),
        ("int8", False),
        (np.int8, False),
        (np.dtype("int8"), False),
        (torch.int8, False),
        ("int16", False),
        (np.int16, False),
        (np.dtype("int16"), False),
        (torch.int16, False),
        ("int32", False),
        (np.int32, False),
        (np.dtype("int32"), False),
        (torch.int32, False),
        ("int64", False),
        (np.int64, False),
        (np.dtype("int64"), False),
        (torch.int64, False),
        ("uint8", False),
        (np.uint8, False),
        (np.dtype("uint8"), False),
        (torch.uint8, False),
        ("uint16", False),
        (np.uint16, False),
        (np.dtype("uint16"), False),
        ("uint32", False),
        (np.uint32, False),
        (np.dtype("uint32"), False),
        ("uint64", False),
        (np.uint64, False),
        (np.dtype("uint64"), False),
        ("float16", False),
        (np.float16, False),
        (np.dtype("float16"), False),
        (torch.float16, False),
        ("float32", False),
        (np.float32, False),
        (np.dtype("float32"), False),
        (torch.float32, False),
        ("float64", False),
        (np.float64, False),
        (np.dtype("float64"), False),
        (torch.float64, False),
        ("bool", True),
        (bool, True),
        (np.bool_, True),
        (torch.bool, True),
    ],
)
def test_is_dtype_bool(type_, expected):
    assert misc.is_dtype_bool(type_) is expected


@pytest.mark.parametrize(
    "type_,expected",
    [
        (Any, False),
        (object, False),
        ("Any", False),
        ("object", False),
        ("O", False),
        ("int8", True),
        (np.int8, True),
        (np.dtype("int8"), True),
        (torch.int8, True),
        ("int16", True),
        (np.int16, True),
        (np.dtype("int16"), True),
        (torch.int16, True),
        ("int32", True),
        (np.int32, True),
        (np.dtype("int32"), True),
        (torch.int32, True),
        ("int64", True),
        (np.int64, True),
        (np.dtype("int64"), True),
        (torch.int64, True),
        ("uint8", True),
        (np.uint8, True),
        (np.dtype("uint8"), True),
        (torch.uint8, True),
        ("uint16", True),
        (np.uint16, True),
        (np.dtype("uint16"), True),
        ("uint32", True),
        (np.uint32, True),
        (np.dtype("uint32"), True),
        ("uint64", True),
        (np.uint64, True),
        (np.dtype("uint64"), True),
        ("float16", True),
        (np.float16, True),
        (np.dtype("float16"), True),
        (torch.float16, True),
        ("float32", True),
        (np.float32, True),
        (np.dtype("float32"), True),
        (torch.float32, True),
        ("float64", True),
        (np.float64, True),
        (np.dtype("float64"), True),
        (torch.float64, True),
        ("bool", False),
        (bool, False),
        (np.bool_, False),
        (torch.bool, False),
    ],
)
def test_is_dtype_real(type_, expected):
    assert misc.is_dtype_real(type_) is expected


@pytest.mark.parametrize(
    "obj,expected",
    [
        (10, True),
        (np.int8(10), True),
        (np.int16(10), True),
        (np.int32(10), True),
        (np.int64(10), True),
        (np.uint8(10), True),
        (np.uint16(10), True),
        (np.uint32(10), True),
        (np.uint64(10), True),
        (np.array(10), True),
        (torch.tensor(10), True),
        (10.0, False),
        (np.float16(10.0), False),
        (np.float32(10.0), False),
        (np.float64(10.0), False),
        (np.array(10.0), False),
        (torch.tensor(10.0), False),
        (True, False),
        (False, False),
        (np.array(False), False),
        (torch.tensor(False), False),
        (np.array([1, 2]), False),
        (torch.tensor([1, 2]), False),
        (np.array([1.0, 2.0]), False),
        (torch.tensor([1.0, 2.0]), False),
    ],
)
def test_is_integer(obj, expected):
    assert misc.is_integer(obj) is expected


@pytest.mark.parametrize(
    "obj,expected",
    [
        (10, True),
        (np.int8(10), True),
        (np.int16(10), True),
        (np.int32(10), True),
        (np.int64(10), True),
        (np.uint8(10), True),
        (np.uint16(10), True),
        (np.uint32(10), True),
        (np.uint64(10), True),
        (np.array(10), True),
        (torch.tensor(10), True),
        (10.0, True),
        (np.float16(10.0), True),
        (np.float32(10.0), True),
        (np.float64(10.0), True),
        (np.array(10.0), True),
        (torch.tensor(10.0), True),
        (True, False),
        (False, False),
        (np.array(False), False),
        (torch.tensor(False), False),
        (np.array([1, 2]), False),
        (torch.tensor([1, 2]), False),
        (np.array([1.0, 2.0]), False),
        (torch.tensor([1.0, 2.0]), False),
    ],
)
def test_is_real(obj, expected):
    assert misc.is_real(obj) is expected


@pytest.mark.parametrize(
    "obj,expected",
    [
        (10, False),
        (np.int8(10), False),
        (np.int16(10), False),
        (np.int32(10), False),
        (np.int64(10), False),
        (np.uint8(10), False),
        (np.uint16(10), False),
        (np.uint32(10), False),
        (np.uint64(10), False),
        (np.array(10), False),
        (torch.tensor(10), False),
        (10.0, False),
        (np.float16(10.0), False),
        (np.float32(10.0), False),
        (np.float64(10.0), False),
        (np.array(10.0), False),
        (torch.tensor(10.0), False),
        (True, True),
        (False, True),
        (np.array(False), True),
        (torch.tensor(False), True),
        (np.array([False, False]), False),
        (torch.tensor([False, False]), False),
        (np.array([1, 2]), False),
        (torch.tensor([1, 2]), False),
        (np.array([1.0, 2.0]), False),
        (torch.tensor([1.0, 2.0]), False),
    ],
)
def test_is_bool(obj, expected):
    assert misc.is_bool(obj) is expected


@pytest.mark.parametrize(
    "obj,expected",
    [
        (1, False),
        (np.array(1), False),
        (torch.tensor(1), False),
        ([1, 2, 3], True),
        ((1, 2, 3), True),
        (np.array([1, 2, 3]), True),
        (torch.tensor([1, 2, 3]), True),
        (np.array([[1, 2], [3, 4]]), False),
        (torch.tensor([[1, 2], [3, 4]]), False),
        (np.array(1.0), False),
        (torch.tensor(1.0), False),
        ([1.0, 2.0, 3.0], False),
        ((1.0, 2.0, 3.0), False),
        (np.array([1.0, 2.0, 3.0]), False),
        (torch.tensor([1.0, 2.0, 3.0]), False),
        (np.array([[1.0, 2.0], [3.0, 4.0]]), False),
        (torch.tensor([[1.0, 2.0], [3.0, 4.0]]), False),
        (np.array(True), False),
        (torch.tensor(True), False),
        ([True, False, True], False),
        ((True, False, True), False),
        (np.array([True, False, True]), False),
        (torch.tensor([True, False, True]), False),
        (np.array([[True, False], [True, False]]), False),
        (torch.tensor([[True, False], [True, False]]), False),
    ],
)
def test_is_integer_vector(obj, expected):
    assert misc.is_integer_vector(obj) is expected


@pytest.mark.parametrize(
    "obj,expected",
    [
        (1, False),
        (np.array(1), False),
        (torch.tensor(1), False),
        ([1, 2, 3], True),
        ((1, 2, 3), True),
        (np.array([1, 2, 3]), True),
        (torch.tensor([1, 2, 3]), True),
        (np.array([[1, 2], [3, 4]]), False),
        (torch.tensor([[1, 2], [3, 4]]), False),
        (np.array(1.0), False),
        (torch.tensor(1.0), False),
        ([1.0, 2.0, 3.0], True),
        ((1.0, 2.0, 3.0), True),
        (np.array([1.0, 2.0, 3.0]), True),
        (torch.tensor([1.0, 2.0, 3.0]), True),
        (np.array([[1.0, 2.0], [3.0, 4.0]]), False),
        (torch.tensor([[1.0, 2.0], [3.0, 4.0]]), False),
        (np.array(True), False),
        (torch.tensor(True), False),
        ([True, False, True], False),
        ((True, False, True), False),
        (np.array([True, False, True]), False),
        (torch.tensor([True, False, True]), False),
        (np.array([[True, False], [True, False]]), False),
        (torch.tensor([[True, False], [True, False]]), False),
    ],
)
def test_is_real_vector(obj, expected):
    assert misc.is_real_vector(obj) is expected


@pytest.mark.parametrize(
    "obj,expected",
    [
        (1, False),
        (np.array(1), False),
        (torch.tensor(1), False),
        ([1, 2, 3], False),
        ((1, 2, 3), False),
        (np.array([1, 2, 3]), False),
        (torch.tensor([1, 2, 3]), False),
        (np.array([[1, 2], [3, 4]]), False),
        (torch.tensor([[1, 2], [3, 4]]), False),
        (np.array(1.0), False),
        (torch.tensor(1.0), False),
        ([1.0, 2.0, 3.0], False),
        ((1.0, 2.0, 3.0), False),
        (np.array([1.0, 2.0, 3.0]), False),
        (torch.tensor([1.0, 2.0, 3.0]), False),
        (np.array([[1.0, 2.0], [3.0, 4.0]]), False),
        (torch.tensor([[1.0, 2.0], [3.0, 4.0]]), False),
        (np.array(True), False),
        (torch.tensor(True), False),
        ([True, False, True], True),
        ((True, False, True), True),
        (np.array([True, False, True]), True),
        (torch.tensor([True, False, True]), True),
        (np.array([[True, False], [True, False]]), False),
        (torch.tensor([[True, False], [True, False]]), False),
    ],
)
def test_is_bool_vector(obj, expected):
    assert misc.is_bool_vector(obj) is expected


def test_ensure_tensor_length_and_dtype():
    inputs = (
        [1, 2, 3],
        (1, 2, 3),
        np.array([1, 2, 3]),
        torch.tensor([1, 2, 3]),
    )

    for x in inputs:
        t = misc.ensure_tensor_length_and_dtype(x, length=3, dtype="float32")
        assert t.dtype is torch.float32
        assert len(t) == 3

    scalar_ones = (1, np.array(1), torch.tensor(1))

    for x in scalar_ones:
        t = misc.ensure_tensor_length_and_dtype(x, length=3, dtype="int64")
        assert t.dtype is torch.int64
        assert len(t) == 3
        for element in t:
            assert int(element) == 1

    for x in scalar_ones:
        t = misc.ensure_tensor_length_and_dtype(x, length=3, allow_scalar=True, dtype="int64")
        assert t.dtype is torch.int64
        assert t.ndim == 0
        assert int(t) == 1

    erroneous_inputs = (
        [1, 2],
        (1, 2),
        [1, 2, 3, 4],
        (1, 2, 3, 4),
        [
            [1, 2, 3],
            [4, 5, 6],
        ],
        torch.tensor(
            [
                [1, 2],
                [3, 4],
            ]
        ),
        np.array(
            [
                [1, 2],
                [3, 4],
            ]
        ),
    )

    for x in erroneous_inputs:
        with pytest.raises(ValueError):
            misc.ensure_tensor_length_and_dtype(x, length=3, dtype="float32")


@pytest.mark.parametrize(
    "x,lb,ub,expected",
    [
        (FloatTensor([1, -1, 3]), 0, 2, FloatTensor([1, 0, 2])),
        (FloatTensor([1, -1, 3]), 0, None, FloatTensor([1, 0, 3])),
        (FloatTensor([1, -1, 3]), None, 2, FloatTensor([1, -1, 2])),
        (
            FloatTensor([1, -1, 3]),
            FloatTensor([1.5, 0, -1]),
            FloatTensor([2.5, 1, 2.5]),
            FloatTensor([1.5, 0, 2.5]),
        ),
    ],
)
def test_clip_tensor(x, lb, ub, expected):
    actual = misc.clip_tensor(x, lb, ub)
    assert_allclose(actual, expected, atol=0.00001)


# fmt: off
@pytest.mark.parametrize(
    "x,target,lb,ub,max_change,expected",
    [
        (
            FloatTensor([10, 11, 12]),   # x
            FloatTensor([0, 21, 22]),    # target
            None,                        # lb
            None,                        # ub
            None,                        # max_change
            FloatTensor([0, 21, 22]),    # expected
        ),
        (
            FloatTensor([10, 11, 12]),   # x
            FloatTensor([0, 21, 22]),    # target
            5,                           # lb
            None,                        # ub
            None,                        # max_change
            FloatTensor([5, 21, 22]),    # expected
        ),
        (
            FloatTensor([10, 11, 12]),   # x
            FloatTensor([0, 21, 22]),    # target
            5,                           # lb
            20,                          # ub
            None,                        # max_change
            FloatTensor([5, 20, 20]),    # expected
        ),
        (
            FloatTensor([10, 11, 12]),   # x
            FloatTensor([0, 21, 22]),    # target
            None,                        # lb
            None,                        # ub
            0.5,                         # max_change
            FloatTensor([5, 16.5, 18]),  # expected
        ),
        (
            FloatTensor([10, 11, 12]),   # x
            FloatTensor([0, 21, 22]),    # target
            7,                           # lb
            17,                          # ub
            0.5,                         # max_change
            FloatTensor([7, 16.5, 17]),  # expected
        ),
    ],
)
def test_modify_tensor(x, target, lb, ub, max_change, expected):
    actual = misc.modify_tensor(x, target, lb=lb, ub=ub, max_change=max_change)
    assert_allclose(actual, expected, atol=0.00001)
# fmt: on


@pytest.mark.parametrize(
    "x,shape,length,dtype",
    [
        (FloatTensor([1, 2, 3, 4]), None, None, None),
        (FloatTensor([1, 2, 3, 4]), (5, 5), None, None),
        (FloatTensor([1, 2, 3, 4]), None, 20, None),
        (FloatTensor([1, 2, 3, 4]), None, None, torch.float16),
        (FloatTensor([1, 2, 3, 4]), None, None, torch.float16),
        (ObjectArray(2), (5,), None, None),
        (ObjectArray(2), (5, 3), None, None),
        (ObjectArray(2), None, 20, None),
    ],
)
def test_empty_tensor_like(x, shape, length, dtype):
    def do():
        return misc.empty_tensor_like(x, shape=shape, length=length, dtype=dtype)

    if ((shape is not None) and (length is not None)) or (
        isinstance(x, ObjectArray) and (shape is not None) and (len(shape) >= 2)
    ):
        with pytest.raises(ValueError):
            do()
    else:
        result = do()

        if shape is None and length is None:
            assert result.shape == x.shape
        elif shape is not None and length is not None:
            assert False, "Should have raised error"
        elif shape is not None:
            assert result.shape == shape
        elif length is not None:
            assert result.ndim == 1
            assert len(result) == length

        if dtype is None:
            assert result.dtype == x.dtype
        else:
            assert result.dtype == dtype


@pytest.mark.parametrize("x", [FloatTensor([1, 2, 3]), [1, 2, 3], (1, 2, 3), np.array([1.0, 2, 3])])
def test_numpy_copy(x):
    for type_ in ("float16", "float32"):
        copied = misc.numpy_copy(x, type_)
        copied[:] = 0

        assert_allclose(x, FloatTensor([1, 2, 3]), atol=0.00001)
        assert_dtype_matches(copied, type_)


def test_pass_info_if_needed_undecorated():
    def f(a):
        return f"{a}"

    assert misc.pass_info_if_needed(f, {})(a="aa") == "aa"
    assert misc.pass_info_if_needed(f, {"b": "bb"})(a="aa") == "aa"
    assert misc.pass_info_if_needed(f, {"b": "bb", "c": "cc"})(a="aa") == "aa"


def test_pass_info_if_needed_decorated():
    @pass_info
    def g(a, b, c="c"):
        return f"{a}{b}{c}"

    assert misc.pass_info_if_needed(g, {"b": "bb"})(a="aa") == "aabbc"
    assert misc.pass_info_if_needed(g, {"b": "bb", "c": "cc"})(a="aa") == "aabbcc"
    assert misc.pass_info_if_needed(g, {"b": "bb", "c": "cc"})(a="aa", c="ccc") == "aabbccc"

    with pytest.raises(TypeError):
        misc.pass_info_if_needed(g, {})(a="aa")
