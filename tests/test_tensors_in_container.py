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


import pytest
import torch

from evotorch.testing import assert_allclose
from evotorch.tools import cast_tensors_in_container, device_of_container, dtype_of_container


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64, torch.int32, torch.int64))
def test_dtype_of_container(dtype):
    x = {
        "a": torch.tensor([1, 2, 3], dtype=dtype),
        "b": [1, 2],
        "c": (
            torch.tensor([10, 20], dtype=dtype),
            "some string",
        ),
    }

    assert dtype_of_container(x) == dtype


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64, torch.int32, torch.int64))
def test_dtype_of_container_cyclic(dtype):
    x = {
        "a": torch.tensor([1, 2, 3], dtype=dtype),
        "b": [1, 2],
        "c": (
            torch.tensor([10, 20], dtype=dtype),
            "some string",
        ),
    }

    x["b"].append(x)
    x["d"] = x["a"]

    assert dtype_of_container(x) == dtype


def test_dtype_of_container_none():
    x = {
        "a": [1, 2, 3],
        "b": [1, 2],
        "c": (
            [10, 20],
            "some string",
        ),
    }

    x["b"].append(x)
    x["d"] = x["a"]

    assert dtype_of_container(x) is None


def test_device_of_container():
    device = torch.device("cpu")

    x = {
        "a": torch.tensor([1, 2, 3], device=device),
        "b": [1, 2],
        "c": (
            torch.tensor([10, 20], device=device),
            "some string",
        ),
    }

    assert device_of_container(x) == device


def test_device_of_container_cyclic():
    device = torch.device("cpu")

    x = {
        "a": torch.tensor([1, 2, 3], device=device),
        "b": [1, 2],
        "c": (
            torch.tensor([10, 20], device=device),
            "some string",
        ),
    }

    x["b"].append(x)
    x["d"] = x["a"]

    assert device_of_container(x) == device


def test_device_of_container_none():
    x = {
        "a": [1, 2, 3],
        "b": [1, 2],
        "c": (
            [10, 20],
            "some string",
        ),
    }

    x["b"].append(x)
    x["d"] = x["a"]

    assert device_of_container(x) is None


def _assert_equals_but_not_same(a, b, atol):
    assert a is not b
    assert_allclose(a, b, atol=atol)


@pytest.mark.parametrize(
    "from_dtype,to_dtype",
    (
        (torch.float32, torch.float64),
        (torch.float64, torch.float32),
    ),
)
def test_cast_tensors_in_container(from_dtype, to_dtype):
    tolerance = 1e-4

    x = {
        "a": torch.randn(10, dtype=from_dtype),
        "b": [1, 2],
        "c": (
            torch.randn(5, dtype=from_dtype),
            "some string",
        ),
    }

    # Make another element point to an existing tensor
    x["d"] = x["c"][0]

    # Introduce a cyclic reference
    x["e"] = x

    # Cast all the tensors in x
    y = cast_tensors_in_container(x, dtype=to_dtype)

    # Assert that the resulting dictionary has the same keys with the original
    assert set(y.keys()) == set(x.keys())

    # Assert that the resulting dictionary is NOT the same dictionary with the original
    assert y is not x

    # Assert that the cloning was done correctly
    assert isinstance(y["a"], torch.Tensor)
    _assert_equals_but_not_same(y["a"], x["a"], atol=tolerance)

    assert isinstance(y["b"], list)
    assert y["b"] is not x["b"]
    assert len(y["b"]) == len(x["b"])
    assert y["b"] == x["b"]

    assert isinstance(y["c"], tuple)
    assert y["c"] is not x["c"]
    assert len(y["c"]) == len(x["c"])
    _assert_equals_but_not_same(y["c"][0], x["c"][0], atol=tolerance)
    assert y["c"][1] == x["c"][1]

    assert y["d"] is not x["d"]
    assert y["d"] is y["c"][0]

    assert y["e"] is not x
    assert y["e"] is y
