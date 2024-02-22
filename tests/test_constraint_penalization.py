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

import torch

from evotorch.tools import constraints


def test_penalty():
    a = torch.as_tensor(2.0)
    b = torch.as_tensor(1.0)

    p = constraints.penalty(a, "<=", b, penalty_sign="+", linear=1.0, step=10.0, exp=2.0, exp_inf=5000.0)
    assert p > 10.0

    p = constraints.penalty(a, "==", b, penalty_sign="+", linear=1.0, step=10.0, exp=2.0, exp_inf=5000.0)
    assert p > 10.0

    p = constraints.penalty(a, ">=", b, penalty_sign="+", linear=1.0, step=10.0, exp=2.0, exp_inf=5000.0)
    assert torch.abs(p) < 1e-8


def test_batched_penalty():
    a = torch.tensor([-2.0, 2.0], dtype=torch.float32)
    b = torch.as_tensor(1.0, dtype=torch.float32)

    p = constraints.penalty(a, "<=", b, penalty_sign="+", linear=1.0, step=10.0, exp=2.0, exp_inf=5000.0)
    assert p.shape == (2,)
    assert torch.abs(p[0]) < 1e-8
    assert p[1] > 10.0

    p = constraints.penalty(a, "==", b, penalty_sign="+", linear=1.0, step=10.0, exp=2.0, exp_inf=5000.0)
    assert p.shape == (2,)
    assert p[0] > 10.0
    assert p[1] > 10.0

    p = constraints.penalty(a, ">=", b, penalty_sign="+", linear=1.0, step=10.0, exp=2.0, exp_inf=5000.0)
    assert p.shape == (2,)
    assert p[0] > 10.0
    assert torch.abs(p[1]) < 1e-8


def test_penalization_amount():
    a = torch.as_tensor(4.0)
    b = torch.as_tensor(1.0)

    p = constraints.penalty(a, "<=", b, penalty_sign="-", linear=1.0)
    desired_p = -(a - b)
    assert torch.abs(p - desired_p) < 1e-8

    p = constraints.penalty(a, "<=", b, penalty_sign="-", step=10.0)
    desired_p = -10.0
    assert torch.abs(p - desired_p) < 1e-8

    p = constraints.penalty(a, "<=", b, penalty_sign="-", linear=1.0, step=10.0)
    desired_p = -((a - b) + 10.0)
    assert torch.abs(p - desired_p) < 1e-8

    p = constraints.penalty(a, "<=", b, penalty_sign="-", exp=4.0)
    desired_p = -((a - b) ** 4.0)
    assert torch.abs(p - desired_p) < 1e-8


def test_positive_log_barrier():
    a = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    b = torch.tensor(4.0, dtype=torch.float32)
    infinity = 1000.0

    p = constraints.log_barrier(a, "<=", b, penalty_sign="+", inf=infinity)
    assert p.shape == a.shape

    for i in range(1, len(p)):
        assert p[i] > p[i - 1]

    assert torch.abs(p[-1] - infinity) < 1e-8


def test_negative_log_barrier():
    a = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    b = torch.tensor(4.0, dtype=torch.float32)
    infinity = 1000.0

    p = constraints.log_barrier(a, "<=", b, penalty_sign="-", inf=infinity)
    assert p.shape == a.shape

    for i in range(1, len(p)):
        assert p[i] < p[i - 1]

    assert torch.abs(p[-1] - (-infinity)) < 1e-8
