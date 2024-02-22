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


class SGDState(NamedTuple):
    center: torch.Tensor
    velocity: torch.Tensor
    center_learning_rate: torch.Tensor
    momentum: torch.Tensor


def sgd(
    *,
    center_init: BatchableVector,
    center_learning_rate: BatchableScalar,
    momentum: Optional[BatchableScalar] = None,
) -> SGDState:
    """
    Initialize the gradient ascent/descent search and get its initial state.

    Reference regarding the momentum behavior:

        Polyak, B. T. (1964).
        Some methods of speeding up the convergence of iteration methods.
        USSR Computational Mathematics and Mathematical Physics, 4(5):1–17.

    Args:
        center_init: Starting point for the gradient ascent/descent.
            Expected as a PyTorch tensor with at least 1 dimension.
            If there are 2 or more dimensions, the extra leftmost dimensions
            are interpreted as batch dimensions.
        center_learning_rate: Learning rate (i.e. the step size) for gradient
            ascent/descent. Can be a scalar or a multidimensional tensor.
            If given as a tensor with multiple dimensions, those dimensions
            will be interpreted as batch dimensions.
        momentum: Momentum coefficient, expected as a scalar.
            If provided as a scalar, Polyak-style momentum will be enabled.
            If given as a tensor with multiple dimensions, those dimensions
            will be interpreted as batch dimensions.
    """
    center_init = torch.as_tensor(center_init)
    dtype = center_init.dtype
    device = center_init.device

    def as_tensor(x) -> torch.Tensor:
        return torch.as_tensor(x, dtype=dtype, device=device)

    velocity = torch.zeros_like(center_init)
    center_learning_rate = as_tensor(center_learning_rate)
    momentum = as_tensor(0.0) if momentum is None else as_tensor(momentum)

    return SGDState(
        center=center_init,
        velocity=velocity,
        center_learning_rate=center_learning_rate,
        momentum=momentum,
    )


@expects_ndim(1, 1, 1, 0, 0)
def _sgd_step(
    g: torch.Tensor,
    center: torch.Tensor,
    velocity: torch.Tensor,
    center_learning_rate: torch.Tensor,
    momentum: torch.Tensor,
) -> tuple:
    velocity = (momentum * velocity) + (center_learning_rate * g)
    center = center + velocity
    return velocity, center


def sgd_ask(state: SGDState) -> torch.Tensor:
    """
    Get the search point stored by the given `SGDState`.

    Args:
        state: The current state of gradient ascent/descent.
    Returns:
        The search point as a 1-dimensional tensor in the non-batched case,
        or as a multi-dimensional tensor if the search is batched.
    """
    return state.center


def sgd_tell(state: SGDState, *, follow_grad: BatchableVector) -> SGDState:
    """
    Tell the gradient ascent/descent the current gradient to get its next state.

    Args:
        state: The current state of gradient ascent/descent.
        follow_grad: Gradient at the current point of the search.
            Can be a 1-dimensional tensor in the non-batched case,
            or a multi-dimensional tensor in the batched case.
    Returns:
        The updated state of gradient ascent/descent, with the given gradient
        applied.
    """
    velocity, center = _sgd_step(
        follow_grad,
        state.center,
        state.velocity,
        state.center_learning_rate,
        state.momentum,
    )

    return SGDState(
        center=center,
        velocity=velocity,
        center_learning_rate=state.center_learning_rate,
        momentum=state.momentum,
    )
