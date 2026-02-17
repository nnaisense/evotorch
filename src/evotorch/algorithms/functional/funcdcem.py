# Copyright 2026 NNAISENSE SA
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

from numbers import Real
from typing import NamedTuple, Optional, Union

import numpy as np
import torch

from ...decorators import expects_ndim
from ...tools import BatchableScalar, Device, TensorFrame


def _true_div(
    a: torch.Tensor,
    b: Union[Real, torch.Tensor],
    float_dtype: torch.dtype,
) -> torch.Tensor:
    a = torch.as_tensor(a, dtype=float_dtype)
    b = torch.as_tensor(b, dtype=float_dtype, device=a.device)
    return a / b


@expects_ndim(1, 0)
def _apply_lower_bound_for_categorical(categories: torch.Tensor, lb: torch.Tensor) -> torch.Tensor:
    from torch.nn.functional import relu

    dtype = categories.dtype
    device = categories.device
    [num_categories] = categories.shape

    tbl = TensorFrame(
        {
            "CATEGORY_INDEX": torch.arange(0, num_categories, dtype=torch.int64, device=device),
            "PROBABILITY": categories,
        }
    ).sort_values(by="PROBABILITY")

    total_so_far = torch.tensor(0.0, dtype=dtype, device=device)
    for i_tail in range(1, num_categories):
        i_head = i_tail - 1
        head = torch.index_select(
            tbl.PROBABILITY, -1, torch.tensor([i_head], dtype=torch.int64, device=categories.device)
        ).reshape(tuple())
        tail = tbl.PROBABILITY[i_tail:]
        clip_needed = head < lb
        head = torch.where(clip_needed, lb, head)
        total_so_far = total_so_far + head
        rescaler = relu(1 - total_so_far) / torch.sum(tail)
        tail = tail * rescaler
        tbl.pick[i_head:, "PROBABILITY"] = torch.cat([head.reshape(1), tail])

    result = torch.empty_like(categories)
    result[tbl.CATEGORY_INDEX] = tbl.PROBABILITY
    result = result / torch.sum(result)

    return result


@expects_ndim(1, 0)
def _apply_lower_bound(probabilities: torch.Tensor, lower_bound: torch.Tensor) -> torch.Tensor:
    from torch.nn.functional import relu

    [n] = probabilities.shape
    if n == 1:
        upper_bound = relu(1.0 - lower_bound)
        lower_bound = lower_bound.reshape(1)
        upper_bound = upper_bound.reshape(1)
        result = torch.minimum(probabilities, upper_bound)
        result = torch.maximum(result, lower_bound)
        return result
    else:
        implicit_probability = relu(1.0 - torch.sum(probabilities))
        all_probabilities = torch.hstack([probabilities, implicit_probability.reshape(1)])
        all_probabilities = torch.maximum(all_probabilities, lower_bound)
        all_probabilities = _apply_lower_bound_for_categorical(all_probabilities, lower_bound)
        candidate_result = all_probabilities[:n]
        looks_okay = torch.all(torch.isfinite(candidate_result))
        return torch.where(looks_okay, candidate_result, probabilities)


@expects_ndim(1, None, 0, None)
def _probability_for_single_variable(
    samples: torch.Tensor,
    num_choices: Optional[int],
    prob_min: torch.Tensor,
    float_dtype: torch.dtype,
) -> torch.Tensor:
    [num_samples] = samples.shape
    if samples.dtype == torch.bool:
        if num_choices is not None:
            raise ValueError(
                "When the population consists of boolean variables, `num_choices` must be None."
                f" However, its value is {repr(num_choices)}."
            )
        num_trues = torch.sum(samples)
        result = _true_div(num_trues, num_samples, float_dtype).reshape(1)
    elif samples.dtype.is_floating_point or samples.dtype.is_complex:
        raise TypeError(f"Discrete cross entropy method cannot work with a population of this dtype: {samples.dtype}")
    else:
        if num_choices is None:
            raise ValueError(
                "When the population consists of non-boolean integer variables, `num_choices` must be at least 2."
                " However, it is omitted (i.e. left as None)."
            )
        num_choices = int(num_choices)
        if num_choices < 2:
            raise ValueError(
                "When the population consists of non-boolean integer variables, `num_choices` must be at least 2."
                f" However, its value is {num_choices}."
            )
        result = torch.hstack(
            [_true_div(torch.sum(samples == i_choice), num_samples, float_dtype) for i_choice in range(num_choices - 1)]
        )
    return _apply_lower_bound(result, prob_min)


@expects_ndim(2, None, 1, None)
def _probabilities_across_population(
    population: torch.Tensor,
    num_choices: Optional[int],
    prob_min: torch.Tensor,
    float_dtype: torch.dtype,
) -> torch.Tensor:
    return _probability_for_single_variable(population.T, num_choices, prob_min, float_dtype)


@expects_ndim(1, 0)
def _index_of_chosen_item(probabilities: torch.Tensor, number_between_0_1: torch.Tensor) -> torch.Tensor:
    dtype = probabilities.dtype
    device = probabilities.device
    [_n] = probabilities.shape
    n = _n + 1
    lb = torch.cumsum(
        torch.hstack(
            [
                torch.tensor([0.0], dtype=dtype, device=device),
                probabilities,
            ]
        ),
        dim=0,
    )
    ub = torch.hstack(
        [
            torch.cumsum(probabilities, dim=0),
            torch.as_tensor(1.0 + 1e-4, dtype=dtype, device=device),
        ]
    )
    indices = torch.arange(n, dtype=torch.int64, device=device)
    choices = TensorFrame(
        {
            "LB": lb,
            "UB": ub,
            "POSITION": number_between_0_1 * torch.ones(n, dtype=dtype, device=device),
            "INDEX": indices,
        }
    )

    def get_index_if_matches(row: dict) -> torch.Tensor:
        position = row["POSITION"]
        return {
            "CHOSEN_INDEX": torch.where(
                (position >= row["LB"]) & (position < row["UB"]),
                row["INDEX"],
                0,
            )
        }

    chosen_index = choices.each(get_index_if_matches)["CHOSEN_INDEX"].max()
    return chosen_index


@expects_ndim(1, None, randomness="different")
def _sample_for_single_variable(probability: torch.Tensor, num_choices: Optional[int]) -> torch.Tensor:
    dtype = probability.dtype
    device = probability.device
    if num_choices is None:
        if len(probability) != 1:
            raise ValueError(
                "When `num_choices` is None, `probability` was expected as a single-item vector."
                f" However, `probability` has {len(probability)} items in it."
            )
        return torch.rand(tuple(), dtype=dtype, device=device) < probability.reshape(tuple())
    else:
        num_choices = int(num_choices)
        if len(probability) != (num_choices - 1):
            raise ValueError("Number of choices does not match the number of items in the probability vector")
        return _index_of_chosen_item(probability, torch.rand(tuple(), dtype=dtype, device=device))


class DCEMState(NamedTuple):
    center: torch.Tensor
    num_choices: Optional[int]
    prob_min: torch.Tensor
    parenthood_ratio: float
    maximize: bool


def dcem(
    *,
    objective_sense: str,
    center_init: Optional[Union[np.ndarray, torch.Tensor]] = None,
    solution_length: Optional[int] = None,
    num_choices: Optional[int] = None,
    prob_min: Optional[BatchableScalar] = None,
    parenthood_ratio: float,
    device: Optional[Device] = None,
) -> DCEMState:
    """
    Initialize a Discrete Cross Entropy Method.

    This discretized counterpart of cross entropy method can be used for
    heuristically solving optimization problems in which the decision
    variables are binary or categorical.

    **Binary case.**
    If the argument `num_choices` is left as None, then it is interpreted
    that the desired search space consists of binary variables. In this case,
    a solution of `n` decision variables is represented by a tensor
    of dtype `torch.bool`.

    **Categorical case.**
    If the argument `num_choices` is given as an integer `m`, then it is
    interpreted that the search space consists of categorical variables,
    and each categorical value is allowed to take an integer value
    between 0 and m-1. In this case, a solution of `n` decision variables
    is represented by a tensor of dtype `torch.int64`.

    References:

        Rubinstein, R. (1999). The cross-entropy method for combinatorial
        and continuous optimization.
        Methodology and computing in applied probability, 1(2), 127-190.

        Botev, Z. I., Kroese, D. P., Rubinstein, R. Y., & L'ecuyer, P. (2013).
        The cross-entropy method for optimization.
        In Handbook of statistics (Vol. 31, pp. 35-59).

    Args:
        objective_sense: Expected as a string, either as 'min' or as 'max'.
            Determines if the goal is to minimize or is to maximize.
        center_init: Optionally the starting point for the heuristic search,
            in which the values are real numbers between 0 and 1.
            Let us assume that the problem at hand has `n` decision variables.
            If the decision variables are binary (with dtype `torch.bool`),
            then `center_init` can be given as a tensor with at least 1
            dimension, with length `n`. Within this tensor, the i-th item
            represents the initial probability of setting the i-th variable
            as True during the phase of population sampling.
            If the decision variables are categorical with `m` categories,
            then `center_init` can be given as a tensor with at least 2
            dimensions, the shape of these rightmost 2 dimensions being
            `(n, m-1)`. For example, if we have 3 categories, and if the
            item `[..., i, :]` is `[0.2, 0.3]`, then, the i-th variable's
            first category has initial probability of 0.2, its second
            category has initial probability of 0.3, and its third
            category has initial probability of 0.5 (which is 1-(0.2+0.3)),
            during the phase of population sampling.
            Extra leftmost dimensions in the provided `center_init` will
            be interpreted as batch dimensions.
            Alternatively, `center_init` can be omitted altogether, and
            the argument `solution_length` can be provided instead.
        solution_length: Optionally the number of decision variables.
            To be given if `center_init` is omitted.
            If `center_init` is provided, this argument must be left as None,
            because the number of decision variables are then inferred from
            the shape of `center_init`.
        num_choices: Number of categories for each decision variable.
            If left as None, it will be assumed that the problem at hand
            is binary, and solutions will assume the dtype `torch.bool`.
            If given as an integer (at least 2), then the solutions will be
            integer-typed.
        prob_min: Optionally the lower bound for the probability
            of choosing a category belonging to all variables (if given as a
            scalar) or for each variable (if given as a vector whose length
            is equal to the number of decision variables).
            If any categorical choice's sampling probability is lower than
            this value, the discrete cross entropy method will attempt to
            re-adjust those sampling probabilities so that this lower bound
            is respected.
        parenthood_ratio: Proportion of the solutions that will be chosen as
            the parents for the next generation. For example, if this is
            given as 0.5, the top 50% of the solutions will be chosen as
            parents.
        device: If given as a string or as a `torch.device` instance, the
            evolutionary search will be performed on this specified device.
    """
    objective_sense = str(objective_sense)
    if objective_sense == "min":
        maximize = False
    elif objective_sense == "max":
        maximize = True
    else:
        raise ValueError(
            f"`objective_sense` was expected as 'min' or 'max', but it was received as: {repr(objective_sense)}."
        )

    if num_choices is not None:
        num_choices = int(num_choices)
        if num_choices < 2:
            raise ValueError("`num_choices` was encountered as an integer less than 2, which is invalid.")

    if device is None:
        device_kwargs = {}
    else:
        device_kwargs = {"device": device}

    if (center_init is None) and (solution_length is None):
        raise ValueError("Both `center_init` and `solution_length` are avoided. Please provide one of them.")
    elif (center_init is None) and (solution_length is not None):
        solution_length = int(solution_length)
        if solution_length < 1:
            raise ValueError("`solution_length` was given as an integer that is less than 1, which is invalid.")
        if num_choices is None:
            center = torch.ones(solution_length, **device_kwargs) * 0.5
        else:
            center = torch.ones((solution_length, (num_choices - 1)), **device_kwargs) * (1.0 / num_choices)
    elif (center_init is not None) and (solution_length is None):
        center = torch.as_tensor(center_init, **device_kwargs)
        if center.ndim == 0:
            raise ValueError("`center_init` was given as a scalar, which is not supported.")
        if center.numel() == 0:
            raise ValueError("`center_init` was given as an empty tensor, which is not supported.")
        if num_choices is None:
            solution_length = center.shape[-1]
        else:
            if center.ndim < 2:
                raise ValueError(
                    "With `num_choices` given as an integer,"
                    " `center_init` was expected as a tensor with at least 2 dimensions."
                    f" However, its number of dimensions is {center.ndim}."
                )
            if center.shape[-1] != (num_choices - 1):
                raise ValueError(
                    "With `num_choices` given as an integer,"
                    " the rightmost dimension size of `center_init` was expected as `num_choices` - 1."
                    " However, the received `center_init` seems to violate this rule."
                )
            solution_length = center.shape[-2]
    else:
        raise ValueError(
            "Both `center_init` and `solution_length` are provided as values other than None."
            " Please provide only one of them and leave the other one as None."
        )

    if prob_min is None:
        prob_min = torch.zeros(solution_length, dtype=center.dtype, device=center.device)
    else:
        prob_min = torch.as_tensor(prob_min, dtype=center.dtype, device=center.device).clamp(0.0, 1.0) * torch.ones(
            solution_length, dtype=center.dtype, device=center.device
        )

    return DCEMState(
        center=center,
        num_choices=num_choices,
        prob_min=prob_min,
        parenthood_ratio=float(parenthood_ratio),
        maximize=maximize,
    )


def dcem_ask(state: DCEMState, *, popsize: int) -> torch.Tensor:
    """
    Obtain a population from cross entropy method, given the state.

    Args:
        state: The current state of the cross entropy method search.
        popsize: Number of solutions to be generated for the requested
            population.
    Returns:
        Population, as a tensor of at least 2 dimensions.
    """
    center = state.center
    if state.num_choices is None:
        center = center.unsqueeze(dim=-1)

    num_vars = center.shape[-2]
    num_explicit_probs = center.shape[-1]
    batch_dim_sizes = center.shape[:-2]
    num_batch_dims = len(batch_dim_sizes)
    center = center.unsqueeze(num_batch_dims)
    expanded_shape = [*batch_dim_sizes, popsize, num_vars, num_explicit_probs]
    center = center.expand(*expanded_shape)
    return _sample_for_single_variable(center, state.num_choices)


@expects_ndim(2, 1, None, 1, None, None, None)
def _dcem_tell(
    population: torch.Tensor,
    evals: torch.Tensor,
    num_choices: Optional[int],
    prob_min: torch.Tensor,
    parenthood_ratio: float,
    maximize: bool,
    float_dtype: torch.dtype,
) -> torch.Tensor:
    from math import ceil

    num_parents = max(1, ceil(len(population) * float(parenthood_ratio)))
    parent_indices = torch.argsort(evals, descending=maximize)[:num_parents]
    parents = population[parent_indices]
    new_center = _probabilities_across_population(parents, num_choices, prob_min, float_dtype)
    if num_choices is None:
        new_center = new_center.squeeze(dim=-1)
    return new_center


def dcem_tell(state: DCEMState, values: torch.Tensor, evals: torch.Tensor) -> DCEMState:
    """
    Given the old state and the evals (fitnesses), obtain the next state.

    From this state tuple, the converged sampling probabilities can be
    obtained via the `.center` attribute.

    Let us denote the number of decision variables by `n`.
    If the problem at hand is binary, then the `.center` attribute of the
    state object will be a vector of length `n` (excluding the batch
    dimensions, if any). Within this vector, the i-th element represents
    the probability of sampling the i-th variable as True.
    If the problem at hand is categorical and the number of categories is `m`,
    the `.center` attribute will store a tensor of shape `(n, m-1)`.
    Within this tensor, the (i,j)-th item represents the probability of
    deciding that the i-th variable is to pick the category j.
    The probability of picking the last category is implicit, meaning
    that it is not stored, and instead computed via
    `1-sumOfOtherCategoryProbabilities`.
    It is due to this implicitness of the last category that the tensor
    is shaped `(n, m-1)` rather than `(n, m)`.

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
        The new state of the discrete cross entropy method search.
    """
    new_center = _dcem_tell(
        values,
        evals,
        state.num_choices,
        state.prob_min,
        state.parenthood_ratio,
        state.maximize,
        state.center.dtype,
    )
    return DCEMState(
        center=new_center,
        num_choices=state.num_choices,
        prob_min=state.prob_min,
        parenthood_ratio=state.parenthood_ratio,
        maximize=state.maximize,
    )
