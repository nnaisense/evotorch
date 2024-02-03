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

from typing import NamedTuple, Optional

import torch

from ...decorators import expects_ndim
from ...tools import BatchableScalar, BatchableVector


class ClipUpState(NamedTuple):
    center: torch.Tensor
    velocity: torch.Tensor
    center_learning_rate: torch.Tensor
    momentum: torch.Tensor
    max_speed: torch.Tensor


def clipup(
    *,
    center_init: BatchableVector,
    momentum: BatchableScalar = 0.9,
    center_learning_rate: Optional[BatchableScalar] = None,
    max_speed: Optional[BatchableScalar] = None,
) -> ClipUpState:
    """
    Initialize the ClipUp optimizer and return its initial state.

    Reference:

        Toklu, N. E., Liskowski, P., & Srivastava, R. K. (2020, September).
        ClipUp: A Simple and Powerful Optimizer for Distribution-Based Policy Evolution.
        In International Conference on Parallel Problem Solving from Nature (pp. 515-527).
        Springer, Cham.

    Args:
        center_init: Starting point for the ClipUp search.
            Expected as a PyTorch tensor with at least 1 dimension.
            If there are 2 or more dimensions, the extra leftmost dimensions
            are interpreted as batch dimensions.
        center_learning_rate: Learning rate (i.e. the step size) for the ClipUp
            updates. Can be a scalar or a multidimensional tensor.
            If given as a tensor with multiple dimensions, those dimensions
            will be interpreted as batch dimensions.
        max_speed: Maximum speed, expected as a scalar. The euclidean norm
            of the velocity (i.e. of the update vector) is not allowed to
            exceed `max_speed`.
            If given as a tensor with multiple dimensions, those dimensions
            will be interpreted as batch dimensions.
    """
    center_init = torch.as_tensor(center_init)
    dtype = center_init.dtype
    device = center_init.device

    def as_tensor(x) -> torch.Tensor:
        return torch.as_tensor(x, dtype=dtype, device=device)

    if (center_learning_rate is None) and (max_speed is None):
        raise ValueError("Both `center_learning_rate` and `max_speed` is missing. At least one of them is needed.")
    elif (center_learning_rate is not None) and (max_speed is None):
        center_learning_rate = as_tensor(center_learning_rate)
        max_speed = center_learning_rate * 2.0
    elif (center_learning_rate is None) and (max_speed is not None):
        max_speed = as_tensor(max_speed)
        center_learning_rate = max_speed / 2.0
    else:
        center_learning_rate = as_tensor(center_learning_rate)
        max_speed = as_tensor(max_speed)

    velocity = torch.zeros_like(center_init)
    momentum = as_tensor(momentum)

    return ClipUpState(
        center=center_init,
        velocity=velocity,
        center_learning_rate=center_learning_rate,
        momentum=momentum,
        max_speed=max_speed,
    )


@expects_ndim(1, 1, 1, 0, 0, 0)
def _clipup_step(
    g: torch.Tensor,
    center: torch.Tensor,
    velocity: torch.Tensor,
    center_learning_rate: torch.Tensor,
    momentum: torch.Tensor,
    max_speed: torch.Tensor,
) -> tuple:
    velocity = (momentum * velocity) + (center_learning_rate * (g / torch.norm(g)))
    vnorm = torch.norm(velocity)
    must_clip = (vnorm > max_speed).expand(velocity.shape)
    velocity = torch.where(must_clip, max_speed * (velocity / vnorm), velocity)
    center = center + velocity
    return velocity, center


def clipup_ask(state: ClipUpState) -> torch.Tensor:
    """
    Get the search point stored by the given `ClipUpState`.

    Args:
        state: The current state of the ClipUp optimizer.
    Returns:
        The search point as a 1-dimensional tensor in the non-batched case,
        or as a multi-dimensional tensor if the ClipUp search is batched.
    """
    return state.center


def clipup_tell(state: ClipUpState, *, follow_grad: BatchableVector) -> ClipUpState:
    """
    Tell the ClipUp optimizer the current gradient to get its next state.

    Args:
        state: The current state of the ClipUp optimizer.
        follow_grad: Gradient at the current point of the Adam search.
            Can be a 1-dimensional tensor in the non-batched case,
            or a multi-dimensional tensor in the batched case.
    Returns:
        The updated state of ClipUp with the given gradient applied.
    """
    velocity, center = _clipup_step(
        follow_grad,
        state.center,
        state.velocity,
        state.center_learning_rate,
        state.momentum,
        state.max_speed,
    )

    return ClipUpState(
        center=center,
        velocity=velocity,
        center_learning_rate=state.center_learning_rate,
        momentum=state.momentum,
        max_speed=state.max_speed,
    )
