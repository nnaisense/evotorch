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

# flake8: noqa: C901

"""
This namespace contains the CMAES class
"""

from typing import Optional, Tuple, Union

import numpy as np
import torch

from ..core import Problem, Solution, SolutionBatch
from ..tools.misc import Real, Vector
from .searchalgorithm import SearchAlgorithm, SinglePopulationAlgorithmMixin


def _h_sig(p_sigma: torch.Tensor, c_sigma: float, iter: int) -> torch.Tensor:
    """Boolean flag for stalling the update to the evolution path for rank-1 updates
    Args:
        p_sigma (torch.Tensor): The evolution path for step-size updates
        c_sigma (float): The learning rate for step-size updates
        iter (int): The current iteration (generation)
    Returns:
        stall (torch.Tensor): Whether to stall the update to p_c, expressed as a single torch float with 0 = continue, 1 = stall
    """
    # infer dimension from p_sigma
    d = p_sigma.shape[-1]
    # Compute the discounted squared sum
    squared_sum = torch.norm(p_sigma).pow(2.0) / (1 - (1 - c_sigma) ** (2 * iter + 1))
    # Check boolean flag and return
    stall = (squared_sum / d) - 1 < 1 + 4.0 / (d + 1)
    return stall.any().to(p_sigma.dtype)


def _limit_stdev(sigma: torch.Tensor, C: torch.Tensor, stdev_min: Optional[float], stdev_max: Optional[float]):
    """Limit the standard deviation of a covariance matrix sigma^2 C
    Args:
        sigma (torch.Tensor): The square root of the scale of the covariance matrix
        C (torch.Tensor): The unscaled shape of the covariance matrix
        stdev_min (Optional[float]): A lower bound on the element-wise standard deviation
        stdev_max (Optional[float]): An upper bound on the element-wise standard deviation
    Returns:
        C (torch.Tensor): The updated shape of the covariance matrix, taking into account the given limits
    """
    if len(C.shape) == 1:
        # Separable case
        # Get element-wise standard deviations, with vector C assumed to be the diagonal
        stdevs = sigma * torch.sqrt(C)
    else:
        # Get element-wise standard deviations by taking variances along diagonal
        stdevs = sigma * torch.sqrt(torch.diag(C))

    # Limit standard deviations
    stdevs = torch.clamp(stdevs, min=stdev_min, max=stdev_max)
    unscaled_stdevs = stdevs / sigma

    # Construct new unscaled covariance matrix
    if len(C.shape) == 1:
        # Separable case
        C = unscaled_stdevs.pow(2.0)
    else:
        # Non-separable case
        C = C.clone()
        torch.diagonal(C)[:] = unscaled_stdevs.pow(2.0)

    return C


def _safe_divide(a: Union[Real, torch.Tensor], b: Union[Real, torch.Tensor]) -> Union[torch.Tensor]:
    tolerance = 1e-8
    if abs(b) < tolerance:
        b = (-tolerance) if b < 0 else tolerance
    return a / b


class CMAES(SearchAlgorithm, SinglePopulationAlgorithmMixin):
    """
    CMAES: Covariance Matrix Adaptation Evolution Strategy.

    This is a GPU-accelerated and vectorized implementation, based on pycma (version r3.2.2)
    and the below references.

    References:

        Nikolaus Hansen, Youhei Akimoto, and Petr Baudis.
        CMA-ES/pycma on Github. Zenodo, DOI:10.5281/zenodo.2559634,
        February 2019.
        <https://github.com/CMA-ES/pycma>

        Nikolaus Hansen, Andreas Ostermeier (2001).
        Completely Derandomized Self-Adaptation in Evolution Strategies.

        Nikolaus Hansen (2016).
        The CMA Evolution Strategy: A Tutorial.

    """

    def __init__(
        self,
        problem: Problem,
        *,
        stdev_init: Real,
        popsize: Optional[int] = None,
        center_init: Optional[Vector] = None,
        c_m: Real = 1.0,
        c_sigma: Optional[Real] = None,
        c_sigma_ratio: Real = 1.0,
        damp_sigma: Optional[Real] = None,
        damp_sigma_ratio: Real = 1.0,
        c_c: Optional[Real] = None,
        c_c_ratio: Real = 1.0,
        c_1: Optional[Real] = None,
        c_1_ratio: Real = 1.0,
        c_mu: Optional[Real] = None,
        c_mu_ratio: Real = 1.0,
        active: bool = True,
        csa_squared: bool = False,
        stdev_min: Optional[Real] = None,
        stdev_max: Optional[Real] = None,
        separable: bool = False,
        limit_C_decomposition: bool = True,
        obj_index: Optional[int] = None,
    ):
        """
        `__init__(...)`: Initialize the CMAES solver.

        Args:
            problem (Problem): The problem object which is being worked on.
            stdev_init (Real): Initial step-size
            popsize: Population size. Can be specified as an int,
                or can be left as None in which case the CMA-ES rule of thumb is applied:
                popsize = 4 + floor(3 log d) where d is the dimension
            center_init: Initial center point of the search distribution.
                Can be given as a Solution or as a 1-D array.
                If left as None, an initial center point is generated
                with the help of the problem object's `generate_values(...)`
                method.
            c_m (Real): Learning rate for updating the mean
                of the search distribution. By default the value is 1.

            c_sigma (Optional[Real]): Learning rate for updating the step size. If None,
                then the CMA-ES rules of thumb will be applied.
            c_sigma_ratio (Real): Multiplier on the learning rate for the step size.
                if c_sigma has been left as None, can be used to rescale the default c_sigma value.

            damp_sigma (Optional[Real]): Damping factor for updating the step size. If None,
                then the CMA-ES rules of thumb will be applied.
            damp_sigma_ratio (Real): Multiplier on the damping factor for the step size.
                if damp_sigma has been left as None, can be used to rescale the default damp_sigma value.

            c_c (Optional[Real]): Learning rate for updating the rank-1 evolution path.
                If None, then the CMA-ES rules of thumb will be applied.
            c_c_ratio (Real): Multiplier on the learning rate for the rank-1 evolution path.
                if c_c has been left as None, can be used to rescale the default c_c value.

            c_1 (Optional[Real]): Learning rate for the rank-1 update to the covariance matrix.
                If None, then the CMA-ES rules of thumb will be applied.
            c_1_ratio (Real): Multiplier on the learning rate for the rank-1 update to the covariance matrix.
                if c_1 has been left as None, can be used to rescale the default c_1 value.

            c_mu (Optional[Real]): Learning rate for the rank-mu update to the covariance matrix.
                If None, then the CMA-ES rules of thumb will be applied.
            c_mu_ratio (Real): Multiplier on the learning rate for the rank-mu update to the covariance matrix.
                if c_mu has been left as None, can be used to rescale the default c_mu value.

            active (bool): Whether to use Active CMA-ES. Defaults to True, consistent with the tutorial paper and pycma.
            csa_squared (bool): Whether to use the squared rule ("CSA_squared" in pycma) for the step-size adapation.
                This effectively corresponds to taking the natural gradient for the evolution path on the step size,
                rather than the default CMA-ES rule of thumb.

            stdev_min (Optional[Real]): Minimum allowed standard deviation of the search
                distribution. Leaving this as None means that no such
                boundary is to be used.
                Can be given as None or as a scalar.
            stdev_max (Optional[Real]): Maximum allowed standard deviation of the search
                distribution. Leaving this as None means that no such
                boundary is to be used.
                Can be given as None or as a scalar.

            separable (bool): Provide this as True if you would like the problem
                to be treated as a separable one. Treating a problem
                as separable means to adapt only the diagonal parts
                of the covariance matrix and to keep the non-diagonal
                parts 0. High dimensional problems result in large
                covariance matrices on which operating is computationally
                expensive. Therefore, for such high dimensional problems,
                setting `separable` as True might be useful.

            limit_C_decomposition (bool): Whether to limit the frequency of decomposition of the shape matrix C
                Setting this to True (default) means that C will not be decomposed every generation
                This degrades the quality of the sampling and updates, but provides a guarantee of O(d^2) time complexity.
                This option can be used with separable=True (e.g. for experimental reasons) but the performance will only degrade
                without time-complexity benefits.


            obj_index (Optional[int]): Objective index according to which evaluation
                of the solution will be done.
        """

        # Initialize the base class
        SearchAlgorithm.__init__(self, problem, center=self._get_center, stepsize=self._get_sigma)

        # Ensure that the problem is numeric
        problem.ensure_numeric()

        # CMAES can't handle problem bounds. Ensure that it is unbounded
        problem.ensure_unbounded()

        # Store the objective index
        self._obj_index = problem.normalize_obj_index(obj_index)

        # Track d = solution length for reference in initialization of hyperparameters
        d = self._problem.solution_length

        # === Initialize population ===
        if not popsize:
            # Default value used in CMA-ES literature 4 + floor(3 log n)
            popsize = 4 + int(np.floor(3 * np.log(d)))
        self.popsize = int(popsize)
        # Half popsize, referred to as mu in CMA-ES literature
        self.mu = int(np.floor(popsize / 2))
        self._population = problem.generate_batch(popsize=popsize)

        # === Initialize search distribution ===

        self.separable = separable

        # If `center_init` is not given, generate an initial solution
        # with the help of the problem object.
        # If it is given as a Solution, then clone the solution's values
        # as a PyTorch tensor.
        # Otherwise, use the given initial solution as the starting
        # point in the search space.
        if center_init is None:
            center_init = self._problem.generate_values(1)
        elif isinstance(center_init, Solution):
            center_init = center_init.values.clone()

        # Store the center
        self.m = self._problem.make_tensor(center_init).squeeze()
        valid_shaped_m = (self.m.ndim == 1) and (len(self.m) == self._problem.solution_length)
        if not valid_shaped_m:
            raise ValueError(
                f"The initial center point was expected as a vector of length {self._problem.solution_length}."
                " However, the provided `center_init` has (or implies) a different shape."
            )

        # Store the initial step size
        self.sigma = self._problem.make_tensor(stdev_init)

        if separable:
            # Initialize C as the diagonal vector. Note that when separable, the eigendecomposition is not needed
            self.C = self._problem.make_ones(d)
            # In this case A is simply the square root of elements of C
            self.A = self._problem.make_ones(d)
        else:
            # Initialize C = AA^T all diagonal.
            self.C = self._problem.make_I(d)
            self.A = self.C.clone()

        # === Initialize raw weights ===
        # Conditioned on popsize

        # w_i = log((lambda + 1) / 2) - log(i) for i = 1 ... lambda
        raw_weights = self.problem.make_tensor(np.log((popsize + 1) / 2) - torch.log(torch.arange(popsize) + 1))
        # positive valued weights are the first mu
        positive_weights = raw_weights[: self.mu]
        negative_weights = raw_weights[self.mu :]

        # Variance effective selection mass of positive weights
        # Not affected by future updates to raw_weights
        self.mu_eff = torch.sum(positive_weights).pow(2.0) / torch.sum(positive_weights.pow(2.0))

        # === Initialize search parameters ===
        # Conditioned on weights

        # Store fixed information
        self.c_m = c_m
        self.active = active
        self.csa_squared = csa_squared
        self.stdev_min = stdev_min
        self.stdev_max = stdev_max

        # Learning rate for step-size adaption
        if c_sigma is None:
            c_sigma = (self.mu_eff + 2.0) / (d + self.mu_eff + 3)
        self.c_sigma = c_sigma_ratio * c_sigma

        # Damping factor for step-size adapation
        if damp_sigma is None:
            damp_sigma = 1 + 2 * max(0, torch.sqrt((self.mu_eff - 1) / (d + 1)) - 1) + self.c_sigma
        self.damp_sigma = damp_sigma_ratio * damp_sigma

        # Learning rate for evolution path for rank-1 update
        if c_c is None:
            # Branches on separability
            if separable:
                c_c = (1 + (1 / d) + (self.mu_eff / d)) / (d**0.5 + (1 / d) + 2 * (self.mu_eff / d))
            else:
                c_c = (4 + self.mu_eff / d) / (d + (4 + 2 * self.mu_eff / d))
        self.c_c = c_c_ratio * c_c

        # Learning rate for rank-1 update to covariance matrix
        if c_1 is None:
            # Branches on separability
            if separable:
                c_1 = 1.0 / (d + 2.0 * np.sqrt(d) + self.mu_eff / d)
            else:
                c_1 = min(1, popsize / 6) * 2 / ((d + 1.3) ** 2.0 + self.mu_eff)
        self.c_1 = c_1_ratio * c_1

        # Learning rate for rank-mu update to covariance matrix
        if c_mu is None:
            # Branches on separability
            if separable:
                c_mu = (0.25 + self.mu_eff + (1.0 / self.mu_eff) - 2) / (d + 4 * np.sqrt(d) + (self.mu_eff / 2.0))
            else:
                c_mu = min(
                    1 - self.c_1, 2 * ((0.25 + self.mu_eff - 2 + (1 / self.mu_eff)) / ((d + 2) ** 2.0 + self.mu_eff))
                )
        self.c_mu = c_mu_ratio * c_mu

        # The 'variance aware' coefficient used for the additive component of the evolution path for sigma
        self.variance_discount_sigma = torch.sqrt(self.c_sigma * (2 - self.c_sigma) * self.mu_eff)
        # The 'variance aware' coefficient used for the additive component of the evolution path for rank-1 updates
        self.variance_discount_c = torch.sqrt(self.c_c * (2 - self.c_c) * self.mu_eff)

        # === Finalize weights ===
        # Conditioned on search parameters and raw weights

        # Positive weights always sum to 1
        positive_weights = positive_weights / torch.sum(positive_weights)

        if self.active:
            # Active CMA-ES: negative weights sum to alpha

            # Get the variance effective selection mass of negative weights
            mu_eff_neg = torch.sum(negative_weights).pow(2.0) / torch.sum(negative_weights.pow(2.0))

            # Alpha is the minimum of the following 3 terms
            alpha_mu = 1 + self.c_1 / self.c_mu
            alpha_mu_eff = 1 + 2 * mu_eff_neg / (self.mu_eff + 2)
            alpha_pos_def = (1 - self.c_mu - self.c_1) / (d * self.c_mu)
            alpha = min([alpha_mu, alpha_mu_eff, alpha_pos_def])

            # Rescale negative weights
            negative_weights = alpha * negative_weights / torch.sum(torch.abs(negative_weights))
        else:
            # Negative weights are simply zero
            negative_weights = torch.zeros_like(negative_weights)

        # Concatenate final weights
        self.weights = torch.cat([positive_weights, negative_weights], dim=-1)

        # === Some final setup ===

        # Initialize the evolution paths
        self.p_sigma = 0.0
        self.p_c = 0.0

        # Hansen's approximation to the expectation of ||x|| x ~ N(0, I_d).
        # Note that we could use the exact formulation with Gamma functions, but we'll retain this form for consistency
        self.unbiased_expectation = np.sqrt(d) * (1 - (1 / (4 * d)) + 1 / (21 * d**2))

        self.last_ex = None

        # How often to decompose C
        if limit_C_decomposition:
            self.decompose_C_freq = max(1, int(np.floor(_safe_divide(1, 10 * d * (self.c_1.cpu() + self.c_mu.cpu())))))
        else:
            self.decompose_C_freq = 1

        # Use the SinglePopulationAlgorithmMixin to enable additional status reports regarding the population.
        SinglePopulationAlgorithmMixin.__init__(self)

    @property
    def population(self) -> SolutionBatch:
        """Population generated by the CMA-ES algorithm"""
        return self._population

    def _get_center(self) -> torch.Tensor:
        """Get the center of search distribution, m"""
        return self.m

    def _get_sigma(self) -> float:
        """Get the step-size of the search distribution, sigma"""
        return float(self.sigma.cpu())

    @property
    def obj_index(self) -> int:
        """Index of the objective being focused on"""
        return self._obj_index

    def sample_distribution(self, num_samples: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample the population. All 3 representations of solutions are returned for easy calculations of updates.
        Note that the computation time of this operation of O(d^2 num_samples) unless separable, in which case O(d num_samples)
        Args:
            num_samples (Optional[int]): The number of samples to draw. If None, then the population size is used
        Returns:
            zs (torch.Tensor): A tensor of shape [num_samples, d] of samples from the local coordinate space e.g. z_i ~ N(0, I_d)
            ys (torch.Tensor): A tensor of shape [num_samples, d] of samples from the shaped coordinate space e.g. y_i ~ N(0, C)
            xs (torch.Tensor): A tensor of shape [num_samples, d] of samples from the search space e.g. x_i ~ N(m, sigma^2 C)
        """
        if num_samples is None:
            num_samples = self.popsize
        # Generate z values
        zs = self._problem.make_gaussian(num_solutions=num_samples)
        # Construct ys = A zs
        if self.separable:
            # In the separable case A is diagonal so is represented as a single vector
            ys = self.A.unsqueeze(0) * zs
        else:
            ys = (self.A @ zs.T).T
        # Construct xs = m + sigma ys
        xs = self.m.unsqueeze(0) + self.sigma * ys
        return zs, ys, xs

    def get_population_weights(self, xs: torch.Tensor) -> torch.Tensor:
        """Get the assigned weights of the population (e.g. evaluate, rank and return)
        Args:
            xs (torch.Tensor): The population samples drawn from N(mu, sigma^2 C)
        Returns:
            assigned_weights (torch.Tensor): A [popsize, ] dimensional tensor of ordered weights
        """
        # Computation is O(popsize * F_time) where F_time is the evalutation time per sample
        # Fill the population
        self._population.set_values(xs)
        # Evaluate the current population
        self.problem.evaluate(self._population)
        # Sort the population
        indices = self._population.argsort(obj_index=self.obj_index)
        # Invert the sorting of the population to obtain the ranks
        # Note that these ranks start at zero, but this is fine as we are just using them for indexing
        ranks = torch.zeros_like(indices)
        ranks[indices] = torch.arange(self.popsize, dtype=indices.dtype, device=indices.device)
        # Get weights corresponding to each rank
        assigned_weights = self.weights[ranks]
        return assigned_weights

    def update_m(self, zs: torch.Tensor, ys: torch.Tensor, assigned_weights: torch.Tensor) -> torch.Tensor:
        """Update the center of the search distribution m
        With zs and ys retained from sampling, this operation is O(popsize d), as it involves summing across popsize d-dimensional vectors.
        Args:
            zs (torch.Tensor): A tensor of shape [popsize, d] of samples from the local coordinate space e.g. z_i ~ N(0, I_d)
            ys (torch.Tensor): A tensor of shape [popsize, d] of samples from the shaped coordinate space e.g. y_i ~ N(0, C)
            assigned_weights (torch.Tensor): A [popsize, ] dimensional tensor of ordered weights
        Returns:
            local_m_displacement (torch.Tensor): A tensor of shape [d], corresponding to the local transformation of m,
                (1/sigma) (C^-1/2) (m' - m) where m' is the updated m
            shaped_m_displacement (torch.Tensor): A tensor of shape [d], corresponding to the shaped transformation of m,
                (1/sigma) (m' - m) where m' is the updated m
        """
        # Get the top-mu weights
        top_mu = torch.topk(assigned_weights, k=self.mu)
        top_mu_weights = top_mu.values
        top_mu_indices = top_mu.indices

        # Compute the weighted recombination in local coordinate space
        local_m_displacement = torch.sum(top_mu_weights.unsqueeze(-1) * zs[top_mu_indices], dim=0)
        # Compute the weighted recombination in shaped coordinate space
        shaped_m_displacement = torch.sum(top_mu_weights.unsqueeze(-1) * ys[top_mu_indices], dim=0)

        # Update m
        self.m = self.m + self.c_m * self.sigma * shaped_m_displacement

        # Return the weighted recombinations
        return local_m_displacement, shaped_m_displacement

    def update_p_sigma(self, local_m_displacement: torch.Tensor) -> None:
        """Update the evolution path for sigma, p_sigma
        This operation is bounded O(d), as is simply the sum of vectors
        Args:
            local_m_displacement (torch.Tensor): The weighted recombination of local samples zs, corresponding to
                (1/sigma) (C^-1/2) (m' - m) where m' is the updated m
        """
        self.p_sigma = (1 - self.c_sigma) * self.p_sigma + self.variance_discount_sigma * local_m_displacement

    def update_sigma(self) -> None:
        """Update the step size sigma according to its evolution path p_sigma
        This operation is bounded O(d), with the most expensive component being the norm of the evolution path, a d-dimensional vector.
        """
        d = self._problem.solution_length
        # Compute the exponential update
        if self.csa_squared:
            # Exponential update based on natural gradient maximizing squared norm of p_sigma
            exponential_update = (torch.norm(self.p_sigma).pow(2.0) / d - 1) / 2
        else:
            # Exponential update increasing likelihood p_sigma having expected norm
            exponential_update = torch.norm(self.p_sigma) / self.unbiased_expectation - 1
        # Rescale exponential update based on learning rate + damping factor
        exponential_update = (self.c_sigma / self.damp_sigma) * exponential_update
        # Multiplicative update to sigma
        self.sigma = self.sigma * torch.exp(exponential_update)

    def update_p_c(self, shaped_m_displacement: torch.Tensor, h_sig: torch.Tensor) -> None:
        """Update the evolution path for rank-1 update, p_c
        This operation is bounded O(d), as is simply the sum of vectors
        Args:
            local_m_displacement (torch.Tensor): The weighted recombination of shaped samples ys, corresponding to
                (1/sigma) (m' - m) where m' is the updated m
            h_sig (torch.Tensor): Whether to stall the update based on the evolution path on sigma, p_sigma, expressed as a torch float
        """
        self.p_c = (1 - self.c_c) * self.p_c + h_sig * self.variance_discount_c * shaped_m_displacement

    def update_C(self, zs: torch.Tensor, ys: torch.Tensor, assigned_weights: torch.Tensor, h_sig: torch.Tensor) -> None:
        """Update the covariance shape matrix C based on rank-1 and rank-mu updates
        This operation is bounded O(d^2 popsize), which is associated with computing the rank-mu update (summing across popsize d*d matrices)
        Args:
            zs (torch.Tensor): A tensor of shape [popsize, d] of samples from the local coordinate space e.g. z_i ~ N(0, I_d)
            ys (torch.Tensor): A tensor of shape [popsize, d] of samples from the shaped coordinate space e.g. y_i ~ N(0, C)
            assigned_weights (torch.Tensor): A [popsize, ] dimensional tensor of ordered weights
            h_sig (torch.Tensor): Whether to stall the update based on the evolution path on sigma, p_sigma, expressed as a torch float
        """
        d = self._problem.solution_length
        # If using Active CMA-ES, reweight negative weights
        if self.active:
            assigned_weights = torch.where(
                assigned_weights > 0, assigned_weights, d * assigned_weights / torch.norm(zs, dim=-1).pow(2.0)
            )
        c1a = self.c_1 * (1 - (1 - h_sig**2) * self.c_c * (2 - self.c_c))  # adjust for variance loss
        weighted_pc = (self.c_1 / (c1a + 1e-23)) ** 0.5
        if self.separable:
            # Rank-1 update
            r1_update = c1a * (self.p_c.pow(2.0) - self.C)
            # Rank-mu update
            rmu_update = self.c_mu * torch.sum(
                assigned_weights.unsqueeze(-1) * (ys.pow(2.0) - self.C.unsqueeze(0)), dim=0
            )
        else:
            # Rank-1 update
            r1_update = c1a * (torch.outer(weighted_pc * self.p_c, weighted_pc * self.p_c) - self.C)
            # Rank-mu update
            rmu_update = self.c_mu * (
                torch.sum(assigned_weights.unsqueeze(-1).unsqueeze(-1) * (ys.unsqueeze(1) * ys.unsqueeze(2)), dim=0)
                - torch.sum(self.weights) * self.C
            )

        # Update C
        self.C = self.C + r1_update + rmu_update

    def decompose_C(self) -> None:
        """Perform the decomposition C = AA^T using a cholesky decomposition
        Note that traditionally CMA-ES uses the eigendecomposition C = BDDB^-1. In our case,
        we keep track of zs, ys and xs when sampling, so we never need C^-1/2.
        Therefore, a cholesky decomposition is all that is necessary. This generally requires
        O(d^3/3) operations, rather than the more costly O(d^3) operations associated with the eigendecomposition.
        """
        if self.separable:
            self.A = self.C.pow(0.5)
        else:
            self.A = torch.linalg.cholesky(self.C)

    def _step(self):
        """Perform a step of the CMA-ES solver"""

        # === Sampling, evaluation and ranking ===

        # Sample the search distribution
        zs, ys, xs = self.sample_distribution()
        # Get the weights assigned to each solution
        assigned_weights = self.get_population_weights(xs)

        # === Center adaption ===

        local_m_displacement, shaped_m_displacement = self.update_m(zs, ys, assigned_weights)

        # === Step size adaption ===

        # Update evolution path p_sigma
        self.update_p_sigma(local_m_displacement)
        # Update sigma
        self.update_sigma()

        # Compute h_sig, a boolean flag for stalling the update to p_c
        h_sig = _h_sig(self.p_sigma, self.c_sigma, self._steps_count)

        # === Unscaled covariance adapation ===

        # Update evolution path p_c
        self.update_p_c(shaped_m_displacement, h_sig)
        # Update the covariance shape C
        self.update_C(zs, ys, assigned_weights, h_sig)

        # === Post-step corrections ===

        # Limit element-wise standard deviation of sigma^2 C
        if self.stdev_min is not None or self.stdev_max is not None:
            self.C = _limit_stdev(self.sigma, self.C, self.stdev_min, self.stdev_max)

        # Decompose C
        if (self._steps_count + 1) % self.decompose_C_freq == 0:
            self.decompose_C()
