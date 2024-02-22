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
from ...distributions import (
    SeparableGaussian,
    SymmetricSeparableGaussian,
    make_functional_grad_estimator,
    make_functional_sampler,
)
from ...tools import BatchableScalar, BatchableVector, modify_vector


def _make_sample_and_grad_funcs(symmetric: bool) -> tuple:
    distribution = SymmetricSeparableGaussian if symmetric else SeparableGaussian
    grad_denominator = "num_directions" if symmetric else "num_solutions"

    required_parameters = ["mu", "sigma"]
    fixed_parameters = dict(
        divide_mu_grad_by=grad_denominator,
        divide_sigma_grad_by=grad_denominator,
    )

    sample = make_functional_sampler(
        distribution, required_parameters=required_parameters, fixed_parameters=fixed_parameters
    )

    grad = make_functional_grad_estimator(
        distribution, required_parameters=required_parameters, fixed_parameters=fixed_parameters
    )

    return sample, grad


_nonsymmetric_sample, _nonsymmetric_grad = _make_sample_and_grad_funcs(False)
_symmetic_sample, _symmetric_grad = _make_sample_and_grad_funcs(True)


class PGPEState(NamedTuple):
    optimizer: Union[str, tuple]  # "adam" or (adam, adam_ask, adam_tell)
    optimizer_state: tuple
    stdev: torch.Tensor
    stdev_learning_rate: torch.Tensor
    stdev_min: torch.Tensor
    stdev_max: torch.Tensor
    stdev_max_change: torch.Tensor
    ranking_method: str
    maximize: bool
    symmetric: bool


def pgpe(
    *,
    center_init: BatchableVector,
    center_learning_rate: BatchableScalar,
    stdev_learning_rate: BatchableScalar,
    objective_sense: str,
    ranking_method: str = "centered",
    optimizer: Union[str, tuple] = "clipup",  # or "adam" or "sgd"
    optimizer_config: Optional[dict] = None,
    stdev_init: Optional[Union[float, BatchableVector]] = None,
    radius_init: Optional[Union[float, BatchableScalar]] = None,
    stdev_min: Optional[Union[float, BatchableVector]] = None,
    stdev_max: Optional[Union[float, BatchableVector]] = None,
    stdev_max_change: Optional[Union[float, BatchableVector]] = 0.2,
    symmetric: bool = True,
) -> PGPEState:
    """
    Get an initial state for the PGPE algorithm.

    The received initial state, a named tuple of type `PGPEState`, is to be
    passed to the function `pgpe_ask(...)` to receive the solutions belonging
    to the first generation of the evolutionary search.

    Inspired by the PGPE implementations used in the studies
    of Ha (2017, 2019), and by the evolution strategy variant of
    Salimans et al. (2017), this PGPE implementation uses 0-centered
    ranking by default.
    The default optimizer for this PGPE implementation is ClipUp
    (Toklu et al., 2020).

    References:

        Frank Sehnke, Christian Osendorfer, Thomas Ruckstiess,
        Alex Graves, Jan Peters, Jurgen Schmidhuber (2010).
        Parameter-exploring Policy Gradients.
        Neural Networks 23(4), 551-559.

        David Ha (2017). Evolving Stable Strategies.
        <http://blog.otoro.net/2017/11/12/evolving-stable-strategies/>

        Salimans, T., Ho, J., Chen, X., Sidor, S. and Sutskever, I. (2017).
        Evolution Strategies as a Scalable Alternative to
        Reinforcement Learning.

        David Ha (2019). Reinforcement Learning for Improving Agent Design.
        Artificial life 25 (4), 352-365.

        Toklu, N.E., Liskowski, P., Srivastava, R.K. (2020).
        ClipUp: A Simple and Powerful Optimizer
        for Distribution-based Policy Evolution.
        Parallel Problem Solving from Nature (PPSN 2020).

    Args:
        center_init: Center (i.e. mean) of the initial search distribution.
            Expected as a PyTorch tensor with at least 1 dimension.
            If the given `center` tensor has more than 1 dimensions, the extra
            leftmost dimensions will be interpreted as batch dimensions.
        center_learning_rate: Learning rate for when updating the center of the
            search distribution.
            For normal cases, this is expected as a scalar. If given as an
            n-dimensional tensor (where n>0), the extra dimensions will be
            considered as batch dimensions.
        stdev_learning_rate: Learning rate for when updating the standard
            deviation of the search distribution.
            For normal cases, this is expected as a scalar. If given as an
            n-dimensional tensor (where n>0), the extra dimensions will be
            considered as batch dimensions.
        objective_sense: Expected as a string, either as 'min' or as 'max'.
            Determines if the goal is to minimize or is to maximize.
        ranking_method: Determines how the fitnesses will be ranked before
            computing the gradients. Among the choices are
            "centered" (a linear ranking where the worst solution gets the rank
            -0.5 and the best solution gets the rank +0.5),
            "linear" (a linear ranking where the worst solution gets the rank
            0 and the best solution gets the rank 1),
            "nes" (the ranking method that is used by the natural evolution
            strategies), and
            "raw" (no ranking).
        optimizer: Functional optimizer to use when updating the center of the
            search distribution. The functional optimizer can be expressed via
            a string, or via a tuple.
            If given as string, the valid choices are:
            "clipup" (for the ClipUp optimizer),
            "adam" (for the Adam optimizer),
            "sgd" (for regular gradient ascent/descent).
            If given as a tuple, the tuple should be in the form
            `(optim, optim_ask, optim_tell)`, where the objects
            `optim`, `optim_ask`, and `optim_tell` are the functions for
            initializing the optimizer, asking (for the current search point),
            and telling (the gradient to follow).
            The function `optim` should expect keyword arguments for its
            hyperparameters, and should return a state tuple of the optimizer.
            The function `optim_ask` should expect the state tuple of the
            optimizer, and should return the current search point as a tensor.
            The function `optim_tell` should expect the state tuple of the
            optimizer as a positional argument, and the gradient via the
            keyword argument `follow_grad`.
        optimizer_config: Optionally a dictionary, containing the
            hyperparameters for the optimizer.
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
        symmetric: Whether or not symmetric (i.e. antithetic) sampling will be
            done while generating a new population.
    Returns:
        A named tuple, of type `CEMState`, storing the hyperparameters and the
        initial state of the cross entropy method.
    """
    from .misc import _get_stdev_init, get_functional_optimizer

    center_init = torch.as_tensor(center_init)
    if center_init.ndim < 1:
        raise ValueError(
            "The center of the search distribution for the functional PGPE was expected"
            " as a tensor with at least 1 dimension."
            f" However, the encountered `center` is {center_init}, of shape {center_init.shape}."
        )

    solution_length = center_init.shape[-1]
    if solution_length == 0:
        raise ValueError("Solution length cannot be 0")

    stdev_init = _get_stdev_init(center_init=center_init, stdev_init=stdev_init, radius_init=radius_init)

    device = center_init.device
    dtype = center_init.dtype

    def as_tensor(x) -> torch.Tensor:
        return torch.as_tensor(x, dtype=dtype, device=device)

    def as_vector_like_center(x: Iterable, vector_name: str) -> torch.Tensor:
        x = as_tensor(x)
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

    center_learning_rate = as_tensor(center_learning_rate)
    stdev_learning_rate = as_tensor(stdev_learning_rate)

    if objective_sense == "min":
        maximize = False
    elif objective_sense == "max":
        maximize = True
    else:
        raise ValueError(
            f"`objective_sense` was expected as 'min' or 'max', but it was received as {repr(objective_sense)}"
        )

    ranking_method = str(ranking_method)

    if stdev_min is None:
        stdev_min = 0.0
    stdev_min = as_vector_like_center(stdev_min, "stdev_min")

    if stdev_max is None:
        stdev_max = float("inf")
    stdev_max = as_vector_like_center(stdev_max, "stdev_max")

    if stdev_max_change is None:
        stdev_max_change = float("inf")
    stdev_max_change = as_vector_like_center(stdev_max_change, "stdev_max_change")

    if optimizer_config is None:
        optimizer_config = {}
    optimizer_init_func, _, _ = get_functional_optimizer(optimizer)
    optimizer_state = optimizer_init_func(
        center_init=center_init, center_learning_rate=center_learning_rate, **optimizer_config
    )

    symmetric = bool(symmetric)

    return PGPEState(
        optimizer=optimizer,
        optimizer_state=optimizer_state,
        stdev=stdev_init,
        stdev_learning_rate=stdev_learning_rate,
        stdev_min=stdev_min,
        stdev_max=stdev_max,
        stdev_max_change=stdev_max_change,
        ranking_method=ranking_method,
        maximize=maximize,
        symmetric=symmetric,
    )


def pgpe_ask(state: PGPEState, *, popsize: int) -> torch.Tensor:
    """
    Obtain a population from the PGPE algorithm.

    Args:
        state: The current state of PGPE.
        popsize: Number of solutions to be generated for the requested
            population.
    Returns:
        Population, as a tensor of at least 2 dimensions.
    """
    from .misc import get_functional_optimizer

    _, optimizer_ask, _ = get_functional_optimizer(state.optimizer)
    center = optimizer_ask(state.optimizer_state)
    stdev = state.stdev
    sample_func = _symmetic_sample if state.symmetric else _nonsymmetric_sample
    return sample_func(popsize, mu=center, sigma=stdev)


@expects_ndim(1, 0, 1)
def _follow_stdev_grad(
    original_stdev: torch.Tensor,
    stdev_learning_rate: torch.Tensor,
    stdev_grad: torch.Tensor,
) -> torch.Tensor:
    return original_stdev + (stdev_learning_rate * stdev_grad)


def pgpe_tell(state: PGPEState, values: torch.Tensor, evals: torch.Tensor) -> PGPEState:
    """
    Given the old state and the evals (fitnesses), obtain the next state.

    From this state tuple, the center point of the search distribution can be
    obtained via the field `.optimizer_state.center`.

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
        The new state of PGPE.
    """
    from .misc import get_functional_optimizer

    _, optimizer_ask, optimizer_tell = get_functional_optimizer(state.optimizer)

    grad_func = _symmetric_grad if state.symmetric else _nonsymmetric_grad
    objective_sense = "max" if state.maximize else "min"
    grads = grad_func(
        values,
        evals,
        mu=optimizer_ask(state.optimizer_state),
        sigma=state.stdev,
        objective_sense=objective_sense,
        ranking_method=state.ranking_method,
    )

    new_optimizer_state = optimizer_tell(state.optimizer_state, follow_grad=grads["mu"])

    target_stdev = _follow_stdev_grad(state.stdev, state.stdev_learning_rate, grads["sigma"])
    new_stdev = modify_vector(
        state.stdev, target_stdev, lb=state.stdev_min, ub=state.stdev_max, max_change=state.stdev_max_change
    )

    return PGPEState(
        optimizer=state.optimizer,
        optimizer_state=new_optimizer_state,
        stdev=new_stdev,
        stdev_learning_rate=state.stdev_learning_rate,
        stdev_min=state.stdev_min,
        stdev_max=state.stdev_max,
        stdev_max_change=state.stdev_max_change,
        ranking_method=state.ranking_method,
        maximize=state.maximize,
        symmetric=state.symmetric,
    )
