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


import itertools
import pickle
from collections import namedtuple
from copy import copy, deepcopy

import numpy as np
import pytest
import torch

from evotorch import Problem
from evotorch.distributions import ExpGaussian, ExpSeparableGaussian, SeparableGaussian, SymmetricSeparableGaussian
from evotorch.testing import assert_allclose, assert_eachclose
from evotorch.tools import ObjectArray, ReadOnlyTensor, make_tensor
from evotorch.tools.cloning import deep_clone
from evotorch.tools.immutable import ImmutableDict, ImmutableList


def test_deep_cloning():
    tolerance = 1e-4

    x = torch.randn(1000, 10)
    y = x[2:5]
    a = x[2:5]
    b = x[12:15]

    data = (y, {"a": a, "b": b})

    clone_of_data = deep_clone(data, otherwise_fail=True)
    clone_of_y = clone_of_data[0]
    clone_of_a = clone_of_data[1]["a"]
    clone_of_b = clone_of_data[1]["b"]

    assert_allclose(clone_of_y, y, atol=tolerance)
    assert_allclose(clone_of_a, a, atol=tolerance)
    assert_allclose(clone_of_b, b, atol=tolerance)

    # The deep-clone operation detaches the tensors from their original storages.
    # Therefore, unlike their originals y and a, clone_of_y and clone_of_a should NOT share storage.
    clone_of_y[:] = -1
    clone_of_a[:] = 1

    assert_eachclose(clone_of_y, -1, atol=tolerance)
    assert_eachclose(clone_of_a, 1, atol=tolerance)


def test_objectarray_cloning():
    tolerance = 1e-4

    pytorch_tensor = torch.randn(20)
    numpy_array = np.random.randn(5)

    original_data = [
        "a",
        {
            1: [10, 20],
            2: [20, 40],
            3: numpy_array,
        },
        pytorch_tensor,
    ]

    arr = make_tensor(original_data, dtype=object)

    # Test whether or not the data was properly cloned while putting into the ObjectArray.
    # The elements in the ObjectArray are expected to have the same values with their originals, while not being
    # the same objects.
    assert arr[0] == "a"

    assert isinstance(arr[1], ImmutableDict)
    assert arr[1][1] == [10, 20]
    assert arr[1][1] is not original_data[1][1]
    assert isinstance(arr[1][1], ImmutableList)

    assert arr[1][2] == [20, 40]
    assert arr[1][2] is not original_data[1][2]
    assert isinstance(arr[1][2], ImmutableList)

    assert_allclose(arr[1][3], numpy_array, atol=tolerance)
    assert arr[1][3] is not numpy_array
    assert not arr[1][3].flags["WRITEABLE"]  # The clone of numpy_array is expected to be read-only

    assert_allclose(arr[2], pytorch_tensor, atol=tolerance)
    assert arr[2] is not pytorch_tensor
    assert isinstance(arr[2], ReadOnlyTensor)  # The clone of pytorch_tensor is expected to be read-only

    # Now we use the `.clone()` method to obtain a mutable clone of the ObjectArray
    mutable_clone = arr.clone()

    # The elements in the newly made mutable clone are expected to be copies of the original data.
    # These elements are expected to have the same values with their originals, while not being the same objects.

    assert isinstance(mutable_clone, np.ndarray)
    assert mutable_clone.flags["WRITEABLE"]
    assert mutable_clone[0] == "a"

    assert isinstance(mutable_clone[1], dict)
    assert mutable_clone[1][1] == [10, 20]
    assert isinstance(mutable_clone[1][1], list)
    assert mutable_clone[1][1] is not arr[1][1]
    assert mutable_clone[1][1] is not original_data[1][1]

    assert mutable_clone[1][2] == [20, 40]
    assert isinstance(mutable_clone[1][2], list)
    assert mutable_clone[1][2] is not arr[1][2]
    assert mutable_clone[1][2] is not original_data[1][2]

    assert_allclose(mutable_clone[1][3], numpy_array, atol=tolerance)
    assert mutable_clone[1][3] is not arr[1][3]
    assert mutable_clone[1][3] is not numpy_array

    assert_allclose(mutable_clone[2], pytorch_tensor, atol=tolerance)
    assert mutable_clone[2] is not arr[2]
    assert mutable_clone[2] is not pytorch_tensor


def _assert_equals_but_not_same(a, b, atol=None):
    assert a is not b
    if atol is None:
        assert a == b
    else:
        assert_allclose(a, b, atol=atol)


class CloningMethods:
    @staticmethod
    def clone_via_method(x):
        return x.clone()

    @staticmethod
    def clone_via_copy(x):
        return copy(x)

    @staticmethod
    def clone_via_deepcopy(x):
        return deepcopy(x)

    @staticmethod
    def clone_via_clone_func(x):
        from evotorch.tools import clone

        return clone(x)

    @staticmethod
    def deep_clone(x):
        return deep_clone(x, otherwise_deepcopy=True)

    @staticmethod
    def pickle_and_unpickle(x):
        return pickle.loads(pickle.dumps(x))


@pytest.mark.parametrize(
    "clone_func",
    [
        CloningMethods.clone_via_clone_func,
        CloningMethods.deep_clone,
    ],
)
def test_cloning_container_with_cycle(clone_func):
    tolerance = 1e-4

    pytorch_tensor = torch.randn(5)

    a = {"a": 1, "b": pytorch_tensor, "c": [1, 2], "d": pytorch_tensor}
    a["c"].append(a)
    a["e"] = a["c"]

    MyNamedTuple = namedtuple("MyNamedTuple", ("aa", "bb"))
    a["f"] = MyNamedTuple(100, 200)
    a["g"] = a["f"]

    clone_of_a = clone_func(a)

    assert a is not clone_of_a

    assert set(a.keys()) == set(clone_of_a.keys())

    assert a["a"] == clone_of_a["a"]
    _assert_equals_but_not_same(a["b"], clone_of_a["b"], atol=tolerance)

    assert a["c"] is not clone_of_a["c"]
    assert a["c"][:2] == clone_of_a["c"][:2]
    assert clone_of_a["c"][2] is clone_of_a

    # Assert that the namedtuple was cloned correctly
    assert clone_of_a["f"] is not a["f"]
    assert clone_of_a["g"] is not a["g"]
    assert clone_of_a["f"] == a["f"]
    assert set(clone_of_a["f"]._fields) == set(a["f"]._fields)

    # Some elements in the original object are expected to be referred to twice, just like how their original
    # counterparts were referred to twice.
    assert clone_of_a["b"] is clone_of_a["d"]
    assert clone_of_a["c"] is clone_of_a["e"]
    assert clone_of_a["f"] is clone_of_a["g"]


@pytest.mark.parametrize(
    "clone_func",
    [
        CloningMethods.clone_via_clone_func,
        CloningMethods.clone_via_copy,
        CloningMethods.clone_via_deepcopy,
        CloningMethods.clone_via_method,
        CloningMethods.deep_clone,
        CloningMethods.pickle_and_unpickle,
    ],
)
def test_cloning_problem_with_cycle(clone_func):
    tolerance = 1e-4

    problem = Problem(
        "min",
        torch.linalg.norm,
        initial_bounds=(-1.0, 1.0),
        solution_length=5,
        dtype="float32",
    )

    some_tensor = torch.FloatTensor([10, 20, 30])
    some_data = (some_tensor[:], [1, 2, problem])
    problem.some_tensor = some_tensor
    problem.some_data = some_data

    clone_of_problem = clone_func(problem)

    assert_allclose(clone_of_problem.some_tensor, problem.some_tensor, atol=tolerance)
    assert_allclose(clone_of_problem.some_data[0], problem.some_data[0], atol=tolerance)
    assert_allclose(clone_of_problem.some_tensor, clone_of_problem.some_data[0], atol=tolerance)

    # Check whether or not the cloning was done correctly
    assert len(clone_of_problem.some_data) == len(problem.some_data)
    assert len(clone_of_problem.some_data[-1]) == len(problem.some_data[-1])
    assert clone_of_problem.some_data[-1][0] == problem.some_data[-1][0]
    assert clone_of_problem.some_data[-1][1] == problem.some_data[-1][1]

    # Although their values are the same, the inner list of `some_data` must be a clone (not the same object)
    assert clone_of_problem.some_data[-1] is not problem.some_data[-1]

    # The clone of the problem object is expected to point to itself like the original one does
    assert clone_of_problem is not problem
    assert clone_of_problem.some_data[-1][-1] is clone_of_problem

    # After the cloning, we expect that the tensors are independent: they do not share storage anymore
    clone_of_problem.some_tensor[:] = -1
    clone_of_problem.some_data[0][:] = 1
    assert_eachclose(clone_of_problem.some_tensor, -1, atol=tolerance)
    assert_eachclose(clone_of_problem.some_data[0], 1, atol=tolerance)


@pytest.mark.parametrize(
    "clone_func,expect_mutable",
    [
        (CloningMethods.clone_via_clone_func, False),
        (CloningMethods.clone_via_copy, False),
        (CloningMethods.clone_via_deepcopy, False),
        (CloningMethods.clone_via_method, True),
        (CloningMethods.deep_clone, False),
        (CloningMethods.pickle_and_unpickle, False),
    ],
)
def test_cloning_objectarray_with_cycles(clone_func, expect_mutable):
    # In this test, we prepare a list with self cycles in it.
    # We first clone it by putting into an ObjectArray.
    # We then clone this ObjectArray again by using various cloning methods.
    # If all of our cloning methods are implemented correctly, we will get the expected results listed below.

    tolerance = 1e-4

    pytorch_tensor = torch.randn(20)
    numpy_array = np.random.randn(5)

    original_data = [
        "one",
        "two",
        pytorch_tensor,
    ]

    inside_dict = {
        "a": 1,
        "b": 2,
        "c": numpy_array,
    }

    inside_dict["d"] = inside_dict

    original_data.append(inside_dict)
    original_data.append(original_data)
    original_data.append(numpy_array)

    # The final state of original_data is:
    #
    # [
    #     "one",                         # 0
    #     "two",                         # 1
    #     pytorch_tensor,                # 2
    #     {                              # 3
    #         "a": 1,
    #         "b": 2,
    #         "c": numpy_array,
    #         "d": <the dict itself>,
    #     },
    #     <the list itself>,             # 4
    #     numpy_array,                   # 5
    # ]

    arr = make_tensor(original_data, dtype=object)
    clone_of_arr = clone_func(arr)

    # The clone of the array is expected to have the same number of elements
    assert len(clone_of_arr) == len(arr)

    # The clone of the inner dictionary is expected to have the same keys
    assert set(clone_of_arr[3].keys()) == set(arr[3].keys())

    # The integers stored via the keys "a" and "b" in the cloned dictionary must be equal
    assert clone_of_arr[3]["a"] == arr[3]["a"]
    assert clone_of_arr[3]["b"] == arr[3]["b"]

    # The numpy array was referred to in the original array twice.
    # We expect that the clone of the original array also refers to its own numpy array clone twice.
    assert isinstance(clone_of_arr[3]["c"], np.ndarray)
    assert isinstance(clone_of_arr[5], np.ndarray)
    assert clone_of_arr[3]["c"] is clone_of_arr[5]

    # We expect that the numpy array was also cloned. It is not the same with its original.
    assert_allclose(clone_of_arr[5], numpy_array, atol=tolerance)
    assert clone_of_arr[5] is not numpy_array
    assert clone_of_arr[5] is not arr[5]

    # We expect that the pytorch tensor was also cloned. It is not the same with its original.
    assert_allclose(clone_of_arr[2], pytorch_tensor, atol=tolerance)
    assert isinstance(clone_of_arr[2], torch.Tensor)
    assert clone_of_arr[2] is not pytorch_tensor
    assert clone_of_arr[2] is not arr[2]

    if expect_mutable:
        assert isinstance(clone_of_arr, np.ndarray)
        assert not isinstance(clone_of_arr[2], ReadOnlyTensor)
        assert isinstance(clone_of_arr[3], dict)
    else:
        assert isinstance(clone_of_arr, ObjectArray)
        assert isinstance(clone_of_arr[2], ReadOnlyTensor)
        assert isinstance(clone_of_arr[3], ImmutableDict)
        assert isinstance(clone_of_arr[4], ObjectArray)
        assert isinstance(clone_of_arr[5], np.ndarray)

        if clone_func is not CloningMethods.pickle_and_unpickle:
            assert not clone_of_arr[5].flags["WRITEABLE"]


@pytest.mark.parametrize(
    "distcls,clone_func",
    itertools.product(
        (SeparableGaussian, SymmetricSeparableGaussian, ExpGaussian, ExpSeparableGaussian),
        (
            CloningMethods.clone_via_clone_func,
            CloningMethods.clone_via_copy,
            CloningMethods.clone_via_deepcopy,
            CloningMethods.clone_via_method,
            CloningMethods.pickle_and_unpickle,
        ),
    ),
)
def test_cloning_gaussian_distribution(distcls, clone_func):
    tolerance = 1e-4

    n = 20
    mu = torch.randn(n)
    sigma = torch.randn(n)
    params = {"mu": mu, "sigma": sigma}

    dist = distcls(params)
    clone_of_dist = clone_func(dist)

    mu = dist.mu
    sigma = dist.sigma

    clone_of_mu = clone_of_dist.mu
    clone_of_sigma = clone_of_dist.sigma

    _assert_equals_but_not_same(clone_of_mu, mu, atol=tolerance)
    _assert_equals_but_not_same(clone_of_sigma, sigma, atol=tolerance)
