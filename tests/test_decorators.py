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

import pytest
import torch

from evotorch.decorators import on_aux_device, on_cuda, on_device, pass_info, vectorized


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


@pytest.mark.parametrize("decorator", [pass_info, on_aux_device, on_device, on_cuda, vectorized])
def test_decorating_fails_with_too_many_args(decorator):
    def g(x):
        pass

    with pytest.raises(TypeError):
        decorator("foo", 2)(g)


@pytest.mark.parametrize("decorator", [pass_info, on_aux_device, on_device("cpu"), on_cuda, vectorized])
def test_decorator_does_not_modify_function(decorator):
    def g():
        return 42

    g = decorator(g)

    assert g() == 42


@pytest.mark.parametrize("decorator", [pass_info, on_aux_device, on_device("cpu"), on_cuda, vectorized])
def test_decorator_preserves_signature(decorator):
    def g(x: float, y: int) -> float:
        return x + y

    g = decorator(g)

    assert g.__annotations__ == {"x": float, "y": int, "return": float}


@pytest.mark.parametrize("decorator", [pass_info, on_aux_device, on_device("cpu"), on_cuda, vectorized])
def test_decorator_preserves_docstring(decorator):
    def g():
        """Docstring"""
        pass

    g = decorator(g)

    assert g.__doc__ == "Docstring"


@pytest.mark.parametrize("decorator", [pass_info, on_aux_device, on_device("cpu"), on_cuda, vectorized])
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
