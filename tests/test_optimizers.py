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


from typing import Any, Optional, Type

import pytest
import torch

from evotorch.optimizers import SGD, Adam, ClipUp, get_optimizer_class
from evotorch.testing import assert_allclose


def test_clipup_api():
    with pytest.raises(ValueError):
        ClipUp(solution_length=5, dtype="float32", stepsize=-0.1)  # negative stepsize should fail

    with pytest.raises(ValueError):
        ClipUp(solution_length=5, dtype="float32", stepsize=0.1, momentum=-0.9)  # negative momentum should fail

    with pytest.raises(ValueError):
        ClipUp(solution_length=5, dtype="float32", stepsize=0.1, max_speed=-1.0)  # negative max_speed should fail


def test_adam_api():
    # The Adam instantiations below should fail because the class Adam expects beta1 and beta2 to be given together

    with pytest.raises(ValueError):
        Adam(solution_length=5, dtype="float32", beta1=0.9, beta2=None)

    with pytest.raises(ValueError):
        Adam(solution_length=5, dtype="float32", beta1=None, beta2=0.999)


@pytest.mark.parametrize(
    "stepsize,momentum,max_speed",
    [
        [0.1, 0.9, None],
        [0.1, 0.95, 0.3],
        [0.2, 0.5, 0.3],
    ],
)
def test_clipup_movement(stepsize: float, momentum: float, max_speed: Optional[float]):
    tolerance = 1e-4

    def normalize(step: torch.Tensor) -> torch.Tensor:
        return step / torch.linalg.norm(step)

    def clip(update: torch.Tensor, max_speed: float) -> torch.Tensor:
        update_norm = torch.linalg.norm(update)
        if update_norm > max_speed:
            update = normalize(update) * max_speed
        return update

    def clipup_step(
        vel: torch.Tensor,
        grad: torch.Tensor,
        stepsize: float,
        momentum: float = 0.9,
        max_speed: Optional[float] = None,
    ) -> torch.Tensor:
        if max_speed is None:
            max_speed = stepsize * 2.0
        raw_update = stepsize * normalize(grad)
        vel = clip(momentum * vel + raw_update, max_speed)
        return vel

    grads = torch.tensor(
        [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 0],
            [1, 0, 1],
            [-1, -3, 0],
            [-4, -1, 0],
            [0, 0, -2],
        ],
        dtype=torch.float32,
    )

    num_steps, length = grads.shape
    dtype = grads.dtype

    x0 = torch.zeros(length, dtype=dtype)

    clipup = ClipUp(solution_length=length, dtype=dtype, stepsize=stepsize, momentum=momentum, max_speed=max_speed)

    x = x0.clone()

    desired_vel = torch.zeros(length, dtype=dtype)
    desired_x = x0.clone()

    for t in range(num_steps):
        grad = grads[t, :]

        x = x + clipup.ascent(grad)

        desired_vel = clipup_step(desired_vel, grad, stepsize=stepsize, momentum=momentum, max_speed=max_speed)
        desired_x = desired_x + desired_vel

    assert_allclose(x, desired_x, atol=tolerance)


@pytest.mark.parametrize(
    "optimizer_name,desired_class,raises",
    [
        ["clipup", ClipUp, False],
        ["clipsgd", ClipUp, False],
        ["clipsga", ClipUp, False],
        ["adam", Adam, False],
        ["sgd", SGD, False],
        ["sga", SGD, False],
        ["unknown", Any, True],
    ],
)
def test_get_optimizer_class(optimizer_name: str, desired_class: Type, raises: bool):
    kwargs = dict(solution_length=5, dtype="float32", stepsize=0.1)

    if raises:
        with pytest.raises(ValueError):
            get_optimizer_class(optimizer_name)
    else:
        optimizer = get_optimizer_class(optimizer_name)(**kwargs)
        assert isinstance(optimizer, desired_class)

        optimizer2 = get_optimizer_class(optimizer_name, optimizer_config=kwargs)()
        assert isinstance(optimizer2, desired_class)
