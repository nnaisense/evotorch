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

from typing import Iterable, NamedTuple, Optional, Union

import torch

from ...decorators import expects_ndim
from ...distributions import SeparableGaussian, make_functional_grad_estimator, make_functional_sampler
from ...tools import BatchableScalar, BatchableVector, modify_vector


class CEMState(NamedTuple):
    center: torch.Tensor
    stdev: torch.Tensor
    stdev_min: torch.Tensor
    stdev_max: torch.Tensor
    stdev_max_change: torch.Tensor
    parenthood_ratio: float
    maximize: bool


def cem(
    *,
    center_init: BatchableVector,
    parenthood_ratio: float,
    objective_sense: str,
    stdev_init: Optional[Union[float, BatchableVector]] = None,
    radius_init: Optional[Union[float, BatchableScalar]] = None,
    stdev_min: Optional[Union[float, BatchableVector]] = None,
    stdev_max: Optional[Union[float, BatchableVector]] = None,
    stdev_max_change: Optional[Union[float, BatchableVector]] = None,
) -> CEMState:
    """
    Get an initial state for the cross entropy method (CEM).

    The received initial state, a named tuple of type `CEMState`, is to be
    passed to the function `cem_ask(...)` to receive the solutions belonging
    to the first generation of the evolutionary search.

    References:

        Rubinstein, R. (1999). The cross-entropy method for combinatorial
        and continuous optimization.
        Methodology and computing in applied probability, 1(2), 127-190.

        Duan, Y., Chen, X., Houthooft, R., Schulman, J., Abbeel, P. (2016).
        Benchmarking deep reinforcement learning for continuous control.
        International conference on machine learning. PMLR, 2016.

    Args:
        center_init: Center (i.e. mean) of the initial search distribution.
            Expected as a PyTorch tensor with at least 1 dimension.
            If the given `center` tensor has more than 1 dimensions, the extra
            leftmost dimensions will be interpreted as batch dimensions.
        stdev_init: Standard deviation of the initial search distribution.
            If this is given as a scalar `s`, the standard deviation for the
            search distribution will be interpreted as `[s, s, ..., s]` whose
            length is the same with the length of `center_init`.
            If this is given as a 1-dimensional tensor, the given tensor will
            be interpreted as the standard deviation vector.
            If this is given as a tensor with at least 2 dimensions, the extra
            leftmost dimension(s) will be interpreted as batch dimensions.
            If you wish to express the coverage area of the initial search
            distribution in terms of "radius" instead, you can leave
            `stdev_init` as None, and provide a value for the argument
            `radius_init`.
        radius_init: Radius for the initial search distribution, representing
            the euclidean norm for the first standard deviation vector.
            Setting this value as `r` means that the standard deviation
            vector will be initialized as a vector `[s, s, ..., s]`
            whose norm will be equal to `r`. In the non-batched case,
            `radius_init` is expected as a scalar value.
            If `radius_init` is given as a tensor with 1 or more
            dimensions, those dimensions will be considered as batch
            dimensions. If you wish to express the coverage are of the initial
            search distribution in terms of the standard deviation values
            instead, you can leave `radius_init` as None, and provide a value
            for the argument `stdev_init`.
        parenthood_ratio: Proportion of the solutions that will be chosen as
            the parents for the next generation. For example, if this is
            given as 0.5, the top 50% of the solutions will be chosen as
            parents.
        objective_sense: Expected as a string, either as 'min' or as 'max'.
            Determines if the goal is to minimize or is to maximize.
        stdev_min: Minimum allowed standard deviation for the search
            distribution. Can be given as a scalar or as a tensor with one or
            more dimensions. When given with at least 2 dimensions, the extra
            leftmost dimensions will be interpreted as batch dimensions.
        stdev_max: Maximum allowed standard deviation for the search
            distribution. Can be given as a scalar or as a tensor with one or
            more dimensions. When given with at least 2 dimensions, the extra
            leftmost dimensions will be interpreted as batch dimensions.
        stdev_max_change: Maximum allowed change for the standard deviation
            vector. If this is given as a scalar, this scalar will serve as a
            limiter for the change of the entire standard deviation vector.
            For example, a scalar value of 0.2 means that the elements of the
            standard deviation vector cannot change more than the 20% of their
            original values. If this is given as a vector (i.e. as a
            1-dimensional tensor), each element of `stdev_max_change` will
            serve as a limiter to its corresponding element within the standard
            deviation vector. If `stdev_max_change` is given as a tensor with
            at least 2 dimensions, the extra leftmost dimension(s) will be
            interpreted as batch dimensions.
            If you do not wish to have such a limiter, you can leave this as
            None.
    Returns:
        A named tuple, of type `CEMState`, storing the hyperparameters and the
        initial state of the cross entropy method.
    """
    from .misc import _get_stdev_init

    center_init = torch.as_tensor(center_init)
    if center_init.ndim < 1:
        raise ValueError(
            "The center of the search distribution for the functional CEM was expected"
            " as a tensor with at least 1 dimension."
            f" However, the encountered `center_init` is {center_init}, of shape {center_init.shape}."
        )

    solution_length = center_init.shape[-1]
    if solution_length == 0:
        raise ValueError("Solution length cannot be 0")

    stdev_init = _get_stdev_init(center_init=center_init, stdev_init=stdev_init, radius_init=radius_init)

    device = center_init.device
    dtype = center_init.dtype

    def as_vector_like_center(x: Iterable, vector_name: str) -> torch.Tensor:
        x = torch.as_tensor(x, dtype=dtype, device=device)
        if x.ndim == 0:
            x = x.repeat(solution_length)
        else:
            if x.shape[-1] != solution_length:
                raise ValueError(
                    f"`{vector_name}` has an incompatible length."
                    f" The length of `{vector_name}`: {x.shape[-1]},"
                    f" but the solution length implied by the provided `center_init` is {solution_length}."
                )
        return x

    if stdev_min is None:
        stdev_min = 0.0
    stdev_min = as_vector_like_center(stdev_min, "stdev_min")

    if stdev_max is None:
        stdev_max = float("inf")
    stdev_max = as_vector_like_center(stdev_max, "stdev_max")

    if stdev_max_change is None:
        stdev_max_change = float("inf")
    stdev_max_change = as_vector_like_center(stdev_max_change, "stdev_max_change")
    parenthood_ratio = float(parenthood_ratio)

    if objective_sense == "min":
        maximize = False
    elif objective_sense == "max":
        maximize = True
    else:
        raise ValueError(
            f"`objective_sense` was expected as 'min' or 'max', but it was received as {repr(objective_sense)}"
        )

    return CEMState(
        center=center_init,
        stdev=stdev_init,
        stdev_min=stdev_min,
        stdev_max=stdev_max,
        stdev_max_change=stdev_max_change,
        parenthood_ratio=parenthood_ratio,
        maximize=maximize,
    )


_required_parameters = ["mu", "sigma", "parenthood_ratio"]
_cem_sample = make_functional_sampler(SeparableGaussian, required_parameters=_required_parameters)
_cem_grad = make_functional_grad_estimator(SeparableGaussian, required_parameters=_required_parameters)


@expects_ndim(1, 1, None, None, randomness="different")
def _cem_ask(center: torch.Tensor, stdev: torch.Tensor, parenthood_ratio: float, popsize: int) -> torch.Tensor:
    return _cem_sample(popsize, mu=center, sigma=stdev, parenthood_ratio=parenthood_ratio)


@expects_ndim(1, 1, 1, None, None, 1, 1, 2, 1, randomness="different")
def _cem_tell(
    stdev_min: torch.Tensor,
    stdev_max: torch.Tensor,
    stdev_max_change: torch.Tensor,
    parenthood_ratio: float,
    maximize: bool,
    org_center: torch.Tensor,
    org_stdev: torch.Tensor,
    values: torch.Tensor,
    evals: torch.Tensor,
) -> tuple:
    grads = _cem_grad(
        values,
        evals,
        mu=org_center,
        sigma=org_stdev,
        objective_sense=("max" if maximize else "min"),
        parenthood_ratio=parenthood_ratio,
    )

    mu_grad = grads["mu"]
    sigma_grad = grads["sigma"]

    center = org_center + mu_grad

    target_stdev = org_stdev + sigma_grad
    stdev = modify_vector(
        org_stdev,
        target_stdev,
        lb=stdev_min,
        ub=stdev_max,
        max_change=stdev_max_change,
    )

    return center, stdev


def cem_ask(state: CEMState, *, popsize: int) -> torch.Tensor:
    """
    Obtain a population from cross entropy method, given the state.

    Args:
        state: The current state of the cross entropy method search.
        popsize: Number of solutions to be generated for the requested
            population.
    Returns:
        Population, as a tensor of at least 2 dimensions.
    """
    return _cem_ask(state.center, state.stdev, state.parenthood_ratio, popsize)


def cem_tell(state: CEMState, values: torch.Tensor, evals: torch.Tensor) -> CEMState:
    """
    Given the old state and the evals (fitnesses), obtain the next state.

    From this state tuple, the center point of the search distribution can be
    obtained via the field `.center`.

    Args:
        state: The old state of the cross entropy method search.
        values: The most recent population, as a PyTorch tensor.
        evals: Evaluation results (i.e. fitnesses) for the solutions expressed
            by `values`. For example, if `values` is shaped `(N, L)`, this means
            that there are `N` solutions (of length `L`). So, `evals` is
            expected as a 1-dimensional tensor of length `N`, where `evals[i]`
            expresses the fitness of the solution `values[i, :]`.
            If `values` is shaped `(B, N, L)`, then there is also a batch
            dimension, so, `evals` is expected as a 2-dimensional tensor of
            shape `(B, N)`.
    Returns:
        The new state of the cross entropy method search.
    """
    new_center, new_stdev = _cem_tell(
        state.stdev_min,
        state.stdev_max,
        state.stdev_max_change,
        state.parenthood_ratio,
        state.maximize,
        state.center,
        state.stdev,
        values,
        evals,
    )
    return CEMState(
        center=new_center,
        stdev=new_stdev,
        stdev_min=state.stdev_min,
        stdev_max=state.stdev_max,
        stdev_max_change=state.stdev_max_change,
        parenthood_ratio=state.parenthood_ratio,
        maximize=state.maximize,
    )
