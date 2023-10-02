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

from copy import copy, deepcopy

import numpy as np
import pytest

from evotorch.tools import clone
from evotorch.tools import immutable as imm
from evotorch.tools import make_tensor, storage_ptr
from evotorch.tools.objectarray import ObjectArray


def test_making():
    # Make a new ObjectArray with the help of `make_tensor(...)`
    objs = make_tensor(["a", "b"], dtype=object)

    # The result must be an ObjectArray
    assert isinstance(objs, ObjectArray)

    # The elements must be "a" and "b"
    assert len(objs) == 2
    assert objs[0] == "a"
    assert objs[1] == "b"


def test_sharing_memory():
    # Make a new ObjectArray of 10 elements
    x = ObjectArray(10)

    # Fill the ObjectArray with 0, 1, ... 9.
    x[:] = range(10)

    # Make a new ObjectArray which is the slice of the first one
    y = x[3:5]

    # The two tensors should be sharing memory
    assert storage_ptr(x) == storage_ptr(y)

    # Change the elements of y
    y[:] = [0, 0]

    # x should be affected by the change as well
    assert x[3] == 0
    assert x[4] == 0


def test_cloning():
    def _got_same_length_same_values(a, b):
        n = len(a)
        assert n == len(b)
        for i in range(n):
            assert a[i] == b[i]

    def _got_same_length_different_values(a, b):
        n = len(a)
        assert n == len(b)
        for i in range(n):
            assert a[i] != b[i]

    # Make a new ObjectArray of 10 elements
    x = ObjectArray(10)

    # Fill the ObjectArray with zeros
    x[:] = [0 for _ in range(10)]

    # Make a copy of the original tensor
    y = copy(x)
    _got_same_length_same_values(x, y)
    # When we change the copy, the original should NOT be updated
    y[:] = [1 for _ in range(10)]
    _got_same_length_different_values(x, y)

    # Make a deepcopy of the original tensor
    y = deepcopy(x)
    _got_same_length_same_values(x, y)
    # When we change the deepcopy, the original should NOT be updated
    y[:] = [1 for _ in range(10)]
    _got_same_length_different_values(x, y)

    # Make a clone of the original tensor
    y = x.clone()
    _got_same_length_same_values(x, y)
    # When we change the clone, the original should NOT be updated
    y[:] = [1 for _ in range(10)]
    _got_same_length_different_values(x, y)


def test_cloning2():
    # Make a new ObjectArray and fill it with lists
    x = ObjectArray(2)
    x[0] = [1, 2]
    x[1] = [3, 4]

    # Make a clone of the original ObjectArray
    y = x.clone()

    # The elements must be equal
    assert x[0] == y[0]
    assert x[1] == y[1]

    # But those elements must NOT be the same objects in memory
    assert x[0] is not y[0]
    assert x[1] is not y[0]


def test_immutability():
    # Prepare data to put into an ObjectArray
    a = set([1, 2])
    b = {"x": 3, "y": [4, 5], "z": [6, 7]}

    # Make a new ObjectArray
    objs = make_tensor([a, b], dtype=object)

    # The elements in the ObjectArray instance must be equal to the originals
    assert objs[0] == a
    assert objs[1] == b

    # The elements in the ObjectArray must be the immutable counterparts of their originals
    assert isinstance(objs[0], imm.ImmutableSet)
    assert isinstance(objs[1], imm.ImmutableDict)
    assert isinstance(objs[1]["y"], imm.ImmutableList)
    assert isinstance(objs[1]["z"], imm.ImmutableList)

    # An ObjectArray can be converted to a numpy array
    objs2 = objs.numpy()
    assert isinstance(objs2, np.ndarray)
    assert objs2.dtype == np.dtype(object)
    assert objs2[0] == a
    assert objs2[1] == b

    # When converted to a numpy array, everything is mutable again
    assert isinstance(objs2[0], set)
    assert isinstance(objs2[1], dict)
    assert isinstance(objs2[1]["y"], list)
    assert isinstance(objs2[1]["z"], list)

    # The immutable elements in the ObjectArray cannot be modified
    with pytest.raises(AttributeError):
        objs[0].add(0)  # An ImmutableSet does not have an `add` method
    with pytest.raises(TypeError):
        objs[1]["x"] = 30  # An ImmutableDict does not support item assignment
    with pytest.raises(AttributeError):
        objs[1]["y"].append(50)  # An ImmutableList does not have an `append` method
    with pytest.raises(TypeError):
        objs[1]["y"][0] = 50  # An ImmutableList does not support item assignment


def test_read_only_object_tensor():
    # Make a new ObjectArray
    objs = make_tensor([1, 2, 3], dtype=object)

    # Create a read-only view to the first ObjectArray
    ro = objs.get_read_only_view()

    # The two ObjectArray must share memory
    assert storage_ptr(objs) == storage_ptr(ro)

    # The two ObjectArray instances must be equal, elementwise
    assert len(objs) == len(ro)
    for i in range(len(objs)):
        objs[i] == ro[i]

    # The read-only ObjectArray cannot be modified
    for i in range(len(ro)):
        with pytest.raises(ValueError):
            ro[i] = 10

    # The original one can be changed and the read-only view will be affected
    objs[0] = 10
    assert ro[0] == 10
