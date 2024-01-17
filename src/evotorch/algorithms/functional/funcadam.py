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

from typing import NamedTuple

import torch

from ...decorators import expects_ndim
from ...tools import BatchableScalar, BatchableVector


class AdamState(NamedTuple):
    center: torch.Tensor
    center_learning_rate: torch.Tensor
    beta1: torch.Tensor
    beta2: torch.Tensor
    epsilon: torch.Tensor
    m: torch.Tensor
    v: torch.Tensor
    t: torch.Tensor


def adam(
    *,
    center_init: BatchableVector,
    center_learning_rate: BatchableScalar = 0.001,
    beta1: BatchableScalar = 0.9,
    beta2: BatchableScalar = 0.999,
    epsilon: BatchableScalar = 1e-8,
) -> AdamState:
    """
    Initialize an Adam optimizer and return its initial state.

    Reference:

        Kingma, D. P. and J. Ba (2015).
        Adam: A method for stochastic optimization.
        In Proceedings of 3rd International Conference on Learning Representations.

    Args:
        center_init: Starting point for the Adam search.
            Expected as a PyTorch tensor with at least 1 dimension.
            If there are 2 or more dimensions, the extra leftmost dimensions
            are interpreted as batch dimensions.
        center_learning_rate: Learning rate (i.e. the step size) for the Adam
            updates. Can be a scalar or a multidimensional tensor.
            If given as a tensor with multiple dimensions, those dimensions
            will be interpreted as batch dimensions.
        beta1: beta1 hyperparameter for the Adam optimizer.
            Can be a scalar or a multidimensional tensor.
            If given as a tensor with multiple dimensions, those dimensions
            will be interpreted as batch dimensions.
        beta2: beta2 hyperparameter for the Adam optimizer.
            Can be a scalar or a multidimensional tensor.
            If given as a tensor with multiple dimensions, those dimensions
            will be interpreted as batch dimensions.
        epsilon: epsilon hyperparameter for the Adam optimizer.
            Can be a scalar or a multidimensional tensor.
            If given as a tensor with multiple dimensions, those dimensions
            will be interpreted as batch dimensions.
    Returns:
        A named tuple of type `AdamState`, representing the initial state
        of the Adam optimizer.
    """
    center_init = torch.as_tensor(center_init)
    dtype = center_init.dtype
    device = center_init.device

    def as_tensor(x) -> torch.Tensor:
        return torch.as_tensor(x, dtype=dtype, device=device)

    center_learning_rate = as_tensor(center_learning_rate)
    beta1 = as_tensor(beta1)
    beta2 = as_tensor(beta2)
    epsilon = as_tensor(epsilon)

    m = torch.zeros_like(center_init)
    v = torch.zeros_like(center_init)
    t = torch.zeros(center_init.shape[:-1], dtype=dtype, device=device)

    return AdamState(
        center=center_init,
        center_learning_rate=center_learning_rate,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon,
        m=m,
        v=v,
        t=t,
    )


@expects_ndim(1, 1, 0, 0, 0, 0, 1, 1, 0)
def _adam_step(
    g: torch.Tensor,
    center: torch.Tensor,
    center_learning_rate: torch.Tensor,
    beta1: torch.Tensor,
    beta2: torch.Tensor,
    epsilon: torch.Tensor,
    m: torch.Tensor,
    v: torch.Tensor,
    t: torch.Tensor,
) -> tuple:
    t = t + 1
    m = (beta1 * m) + ((1 - beta1) * g)
    v = (beta2 * v) + ((1 - beta2) * (g**2.0))
    mhat = m / (1 - (beta1**t))
    vhat = v / (1 - (beta2**t))
    center_update = center_learning_rate * mhat / (torch.sqrt(vhat) + epsilon)
    center = center + center_update
    return center, m, v, t


def adam_ask(state: AdamState) -> torch.Tensor:
    """
    Get the search point stored by the given `AdamState`.

    Args:
        state: The current state of the Adam optimizer.
    Returns:
        The search point as a 1-dimensional tensor in the non-batched case,
        or as a multi-dimensional tensor if the Adam search is batched.
    """
    return state.center


def adam_tell(state: AdamState, *, follow_grad: BatchableVector) -> AdamState:
    """
    Tell the Adam optimizer the current gradient to get its next state.

    Args:
        state: The current state of the Adam optimizer.
        follow_grad: Gradient at the current point of the Adam search.
            Can be a 1-dimensional tensor in the non-batched case,
            or a multi-dimensional tensor in the batched case.
    Returns:
        The updated state of Adam with the given gradient applied.
    """
    new_center, new_m, new_v, new_t = _adam_step(
        follow_grad,
        state.center,
        state.center_learning_rate,
        state.beta1,
        state.beta2,
        state.epsilon,
        state.m,
        state.v,
        state.t,
    )

    return AdamState(
        center=new_center,
        center_learning_rate=state.center_learning_rate,
        beta1=state.beta1,
        beta2=state.beta2,
        epsilon=state.epsilon,
        m=new_m,
        v=new_v,
        t=new_t,
    )
