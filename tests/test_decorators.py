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

import numpy as np
import pytest
import torch

from evotorch.decorators import distribute, on_aux_device, on_cuda, on_device, pass_info, vectorized
from evotorch.tools import ObjectArray, as_tensor


@pytest.mark.parametrize(
    "decorator, attribute",
    [
        (pass_info, "__evotorch_pass_info__"),
        (pass_info(), "__evotorch_pass_info__"),
        (on_aux_device, "__evotorch_on_aux_device__"),
        (on_aux_device(), "__evotorch_on_aux_device__"),
        (on_device("cpu"), "__evotorch_on_device__"),
        (on_cuda, "__evotorch_on_device__"),
        (on_cuda(), "__evotorch_on_device__"),
        (vectorized, "__evotorch_vectorized__"),
        (vectorized(), "__evotorch_vectorized__"),
    ],
)
def test_decorator_sets_attribute_to_true(decorator, attribute):
    def g():
        pass

    g = decorator(g)

    assert hasattr(g, attribute)
    assert getattr(g, attribute) is True


@pytest.mark.parametrize("decorator", [pass_info, vectorized])
def test_decorating_fails_with_too_many_args(decorator):
    def g(x):
        pass

    with pytest.raises(TypeError):
        decorator("foo", 2)(g)


@pytest.mark.parametrize("decorator", [pass_info, on_aux_device, on_device("cpu"), vectorized])
def test_decorator_does_not_modify_function(decorator):
    test_matrix = torch.LongTensor(
        [
            [1, 2],
            [3, 4],
        ]
    )

    def g(x: torch.Tensor) -> torch.Tensor:
        return 2 * x

    g = decorator(g)

    result = g(test_matrix).to(device="cpu")

    assert bool(torch.all(result == (2 * test_matrix)))


@pytest.mark.parametrize("decorator", [pass_info, vectorized])
def test_decorator_preserves_signature(decorator):
    def g(x: float, y: int) -> float:
        return x + y

    g = decorator(g)

    assert g.__annotations__ == {"x": float, "y": int, "return": float}


@pytest.mark.parametrize("decorator", [pass_info, vectorized])
def test_decorator_preserves_docstring(decorator):
    def g():
        """Docstring"""
        pass

    g = decorator(g)

    assert g.__doc__ == "Docstring"


@pytest.mark.parametrize("decorator", [pass_info, vectorized])
def test_decorator_preserves_name(decorator):
    def g():
        pass

    g = decorator(g)

    assert g.__name__ == "g"


@pytest.mark.parametrize("device", ["cpu", "cuda", "cuda:0", "cuda:1"])
def test_on_device(device):
    @on_device(device)
    def g():
        pass

    assert hasattr(g, "device")
    assert g.device == torch.device(device)


@pytest.mark.parametrize(
    "cuda, expected",
    [
        (None, "cuda"),
        (0, "cuda:0"),
        (1, "cuda:1"),
        (2, "cuda:2"),
    ],
)
def test_on_cuda(cuda, expected):
    def g():
        pass

    if cuda is None:
        g = on_cuda(g)
    else:
        g = on_cuda(cuda)(g)

    assert hasattr(g, "device")
    assert g.device == torch.device(expected)


def test_on_device_moves_input_tensors():

    @on_device("meta")
    def f(x: torch.Tensor) -> torch.Tensor:
        if x.device == torch.device("meta"):
            x = torch.ones_like(x, device="cpu")
        return torch.sum(x)

    input_tensor = torch.arange(10, dtype=torch.int64, device="cpu")
    result = f(input_tensor)

    assert int(torch.sum(result)) == len(input_tensor)


@pytest.mark.parametrize("decoration_form", [True, False])
def test_on_device_chunking(decoration_form: bool):

    input_tensor = torch.LongTensor(
        [
            [1, 2, 3],
            [4, 5, 6],
            [10, 20, 30],
            [40, 50, 60],
            [-1, -2, -3],
        ]
    )

    def f(x: torch.Tensor) -> torch.Tensor:
        return torch.sum(x, dim=-1)

    chunk_size = 2
    if decoration_form:

        @on_device("cpu", chunk_size=chunk_size)
        def chunking_f(x: torch.Tensor) -> torch.Tensor:
            return f(x)

    else:
        chunking_f = on_device(f, device="cpu", chunk_size=chunk_size)

    recombined_result = chunking_f(input_tensor)
    expected_result = f(input_tensor)

    assert recombined_result.shape == expected_result.shape
    assert bool(torch.all(recombined_result == expected_result))


@pytest.mark.parametrize(
    "decoration_form, distribute_config, chunk_size",
    [
        (True, {"devices": ["cpu", "cpu"]}, None),
        (True, {"num_actors": 2}, None),
        (False, {"devices": ["cpu", "cpu"]}, None),
        (False, {"num_actors": 2}, None),
        (True, {"devices": ["cpu", "cpu"]}, 2),
        (True, {"num_actors": 2}, 2),
        (False, {"devices": ["cpu", "cpu"]}, 2),
        (False, {"num_actors": 2}, 2),
    ],
)
def test_distribute(decoration_form: bool, distribute_config: dict, chunk_size: int | None):

    input_tensor = torch.LongTensor(
        [
            [1, 2, 3],
            [4, 5, 6],
            [10, 20, 30],
            [40, 50, 60],
            [-1, -2, -3],
            [-4, -5, -6],
            [-30, -60, -90],
        ]
    )

    def f(x: torch.Tensor) -> torch.Tensor:
        return torch.sum(x, dim=-1)

    if decoration_form:

        @distribute(**distribute_config)
        def distributed_f(x: torch.Tensor) -> torch.Tensor:
            return f(x)

    else:
        distributed_f = distribute(f, **distribute_config)

    recombined_result = distributed_f(input_tensor)
    expected_result = f(input_tensor)

    assert recombined_result.shape == expected_result.shape
    assert bool(torch.all(recombined_result == expected_result))


def test_distribute_with_objectarray():

    input_array = as_tensor(
        [
            [1, 2, 3],
            [5, 6],
            [10, 20, 30, 40],
            [100],
        ],
        dtype=object,
    )

    def f(x: ObjectArray) -> ObjectArray:
        n = len(x)
        y = ObjectArray(n)
        for i in range(n):
            y[i] = sum(x[i])
        return y

    distributed_f = distribute(f, devices=["cpu", "cpu"])

    recombined_result = distributed_f(input_array)
    expected_result = f(input_array)

    assert isinstance(recombined_result, ObjectArray)
    assert isinstance(expected_result, ObjectArray)
    assert len(recombined_result) == len(expected_result)
    assert np.all(expected_result.numpy() == recombined_result.numpy())
