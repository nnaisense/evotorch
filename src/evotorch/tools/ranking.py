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

"""
This module contains ranking functions which work with PyTorch tensors.
"""

from typing import Iterable

import torch


def centered(fitnesses: torch.Tensor, *, higher_is_better: bool = True) -> torch.Tensor:
    """
    Apply linearly spaced 0-centered ranking on a PyTorch tensor.
    The lowest weight is -0.5, and the highest weight is 0.5.
    This is the same ranking method that was used in:

        Tim Salimans, Jonathan Ho, Xi Chen, Szymon Sidor, Ilya Sutskever (2017).
        Evolution Strategies as a Scalable Alternative to Reinforcement Learning

    Args:
        fitnesses: A PyTorch tensor which contains real numbers which we want
             to rank.
        higher_is_better: Whether or not the higher values will be assigned
             higher ranks. Changing this to False means that lower values
             are interpreted as better, and therefore lower values will have
             higher ranks.
    Returns:
        The ranks, in the same device, with the same dtype with the original
        tensor.
    """
    device = fitnesses.device
    dtype = fitnesses.dtype
    with torch.no_grad():
        x = fitnesses.reshape(-1)
        n = len(x)
        indices = x.argsort(descending=(not higher_is_better))
        weights = (torch.arange(n, dtype=dtype, device=device) / (n - 1)) - 0.5
        ranks = torch.empty_like(x)
        ranks[indices] = weights
        return ranks.reshape(*(fitnesses.shape))


def linear(fitnesses: torch.Tensor, *, higher_is_better: bool = True) -> torch.Tensor:
    """
    Apply linearly spaced ranking on a PyTorch tensor.
    The lowest weight is 0, and the highest weight is 1.

    Args:
        fitnesses: A PyTorch tensor which contains real numbers which we want
             to rank.
        higher_is_better: Whether or not the higher values will be assigned
             higher ranks. Changing this to False means that lower values
             are interpreted as better, and therefore lower values will have
             higher ranks.
    Returns:
        The ranks, in the same device, with the same dtype with the original
        tensor.
    """
    device = fitnesses.device
    dtype = fitnesses.dtype
    with torch.no_grad():
        x = fitnesses.reshape(-1)
        n = len(x)
        indices = x.argsort(descending=(not higher_is_better))
        weights = torch.arange(n, dtype=dtype, device=device) / (n - 1)
        ranks = torch.empty_like(x)
        ranks[indices] = weights
        return ranks.reshape(*(fitnesses.shape))


def nes(fitnesses: torch.Tensor, *, higher_is_better: bool = True) -> torch.Tensor:
    """
    Apply the ranking mechanism proposed in:

        Wierstra, D., Schaul, T., Glasmachers, T., Sun, Y., Peters, J., & Schmidhuber, J. (2014).
        Natural evolution strategies. The Journal of Machine Learning Research, 15(1), 949-980.

    Args:
        fitnesses: A PyTorch tensor which contains real numbers which we want
             to rank.
        higher_is_better: Whether or not the higher values will be assigned
             higher ranks. Changing this to False means that lower values
             are interpreted as better, and therefore lower values will have
             higher ranks.
    Returns:
        The ranks, in the same device, with the same dtype with the original
        tensor.
    """
    device = fitnesses.device
    dtype = fitnesses.dtype

    with torch.no_grad():
        x = fitnesses.reshape(-1)
        n = len(x)

        incr_indices = torch.arange(n, dtype=dtype, device=device)
        N = torch.tensor(n, dtype=dtype, device=device)

        weights = torch.max(
            torch.tensor(0, dtype=dtype, device=device), torch.log((N / 2.0) + 1.0) - torch.log(N - incr_indices)
        )

        indices = torch.argsort(x, descending=(not higher_is_better))
        ranks = torch.empty(n, dtype=indices.dtype, device=device)
        ranks[indices] = torch.arange(n, dtype=indices.dtype, device=device)

        utils = weights[ranks]
        utils /= torch.sum(utils)
        utils -= 1 / N

        return utils.reshape(*(fitnesses.shape))


def normalized(fitnesses: torch.Tensor, *, higher_is_better: bool = True) -> torch.Tensor:
    """
    Normalize the fitnesses and return the result as ranks.

    The normalization is done in such a way that the mean becomes 0.0 and
    the standard deviation becomes 1.0.

    According to the value of `higher_is_better`, it will be ensured that
    better solutions will have numerically higher rank.
    In more details, if `higher_is_better` is set as False, then the
    fitnesses will be multiplied by -1.0 in addition to being subject
    to normalization.

    Args:
        fitnesses: A PyTorch tensor which contains real numbers which we want
             to rank.
        higher_is_better: Whether or not the higher values will be assigned
             higher ranks. Changing this to False means that lower values
             are interpreted as better, and therefore lower values will have
             higher ranks.
    Returns:
        The ranks, in the same device, with the same dtype with the original
        tensor.
    """
    if not higher_is_better:
        fitnesses = -fitnesses

    fitness_mean = torch.mean(fitnesses)
    fitness_stdev = torch.std(fitnesses)

    fitnesses = fitnesses - fitness_mean
    fitnesses = fitnesses / fitness_stdev

    return fitnesses


def raw(fitnesses: torch.Tensor, *, higher_is_better: bool = True) -> torch.Tensor:
    """
    Return the fitnesses themselves as ranks.

    If `higher_is_better` is given as False, then the fitnesses will first
    be multiplied by -1 and then the result will be returned as ranks.

    Args:
        fitnesses: A PyTorch tensor which contains real numbers which we want
             to rank.
        higher_is_better: Whether or not the higher values will be assigned
             higher ranks. Changing this to False means that lower values
             are interpreted as better, and therefore lower values will have
             higher ranks.
    Returns:
        The ranks, in the same device, with the same dtype with the original
        tensor.
    """
    if not higher_is_better:
        fitnesses = -fitnesses
    return fitnesses


rankers = {"nes": nes, "centered": centered, "linear": linear, "normalized": normalized, "raw": raw}


def rank(fitnesses: Iterable[float], ranking_method: str, *, higher_is_better: bool):
    """
    Get the ranks of the given sequence of numbers.

    Better solutions will have numerically higher ranks.

    Args:
        fitnesses: A sequence of numbers to be ranked.
        ranking_method: The ranking method to be used.
            Can be "centered", which means 0-centered linear ranking
                from -0.5 to 0.5.
            Can be "linear", which means a linear ranking from 0 to 1.
            Can be "nes", which means the ranking method used by
                Natural Evolution Strategies.
            Can be "normalized", which means that the ranks will be
                the normalized counterparts of the fitnesses.
            Can be "raw", which means that the fitnesses themselves
                (or, if `higher_is_better` is False, their inverted
                counterparts, inversion meaning the operation of
                multiplying by -1 in this context) will be the ranks.
        higher_is_better: Whether or not the higher values will be assigned
             higher ranks. Changing this to False means that lower values
             are interpreted as better, and therefore lower values will have
             higher ranks.
    """
    fitnesses = torch.as_tensor(fitnesses)
    rank_func = rankers[ranking_method]
    return rank_func(fitnesses, higher_is_better=higher_is_better)
