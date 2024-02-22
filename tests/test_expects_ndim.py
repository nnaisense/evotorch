# Copyright 2024 NNAISENSE SA
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

from evotorch import testing
from evotorch.decorators import expects_ndim, rowwise


@expects_ndim(0, 1)
def f(c: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return c * torch.max(x)


@expects_ndim(None, 0, 1)
def typed_f(dtype: torch.dtype, c: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    assert c.dtype == dtype
    return c * torch.max(x)


@rowwise
def rowwise_max(x: torch.Tensor) -> torch.Tensor:
    return torch.max(x)


def test_expects_ndim_without_batching():
    c = torch.as_tensor(2, dtype=torch.float32)
    x = torch.tensor([3, -2, 1], dtype=torch.float32)

    desired = torch.tensor(6, dtype=torch.float32)

    testing.assert_allclose(f(c, x), desired, atol=1e-4)


def test_expects_ndim_with_batched_vector():
    c = torch.as_tensor(2, dtype=torch.float32)
    x = torch.tensor(
        [
            [0, -1, 2],
            [3, -2, 1],
            [-4, 6, -1],
        ],
        dtype=torch.float32,
    )

    desired = torch.tensor([4, 6, 12], dtype=torch.float32)

    testing.assert_allclose(f(c, x), desired, atol=1e-4)


def test_expects_ndim_with_batched_multiplier():
    c = torch.tensor([-1, 2], dtype=torch.float32)
    x = torch.tensor([3, -2, 1], dtype=torch.float32)

    desired = torch.tensor([-3, 6], dtype=torch.float32)

    testing.assert_allclose(f(c, x), desired, atol=1e-4)


def test_expects_ndim_with_matching_batch_dims():
    c = torch.tensor([-1, 1, 2], dtype=torch.float32)
    x = torch.tensor(
        [
            [0, -1, 2],
            [3, -2, 1],
            [-4, 6, -1],
        ],
        dtype=torch.float32,
    )

    desired = torch.tensor([-2, 3, 12], dtype=torch.float32)

    testing.assert_allclose(f(c, x), desired, atol=1e-4)


def test_expects_ndim_with_multibatch():
    c = torch.tensor(
        [
            [-1, 1, 2],
            [1, -1, -2],
        ],
        dtype=torch.float32,
    )
    x = torch.tensor(
        [
            [0, -1, 2],
            [3, -2, 1],
            [-4, 6, -1],
        ],
        dtype=torch.float32,
    )

    desired = torch.tensor(
        [
            [-2, 3, 12],
            [2, -3, -12],
        ],
        dtype=torch.float32,
    )

    testing.assert_allclose(f(c, x), desired, atol=1e-4)


def test_expects_ndim_with_non_tensor_scalar():
    for c in (True, np.bool_(True), 1, 1.0):
        for dtype in (torch.float32, torch.float64):
            x = torch.tensor([3, -2, 1], dtype=dtype)
            desired = torch.as_tensor(3, dtype=dtype)

            if isinstance(c, (bool, np.bool_)):
                expected_dtype = torch.bool
            else:
                expected_dtype = x.dtype

            testing.assert_allclose(typed_f(expected_dtype, c, x), desired, atol=1e-4)


def test_rowwise():
    with pytest.raises(ValueError):
        rowwise_max(torch.as_tensor(1.0))

    testing.assert_allclose(rowwise_max(torch.FloatTensor([1, 3, 2])), 3, rtol=1e-4)

    testing.assert_allclose(
        rowwise_max(
            torch.FloatTensor(
                [
                    [1, 3, 2],
                    [-1, 0, -9],
                ]
            ),
        ),
        torch.FloatTensor([3, 0]),
        rtol=1e-4,
    )
