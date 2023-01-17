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
This module contains operators defined to work with problems
whose `dtype`s are real numbers (e.g. `torch.float32`).
"""

from copy import deepcopy
from typing import Iterable, Optional, Union

import torch

from ..core import Problem, SolutionBatch
from ..tools.ranking import rank
from .base import CopyingOperator, CrossOver, SingleObjOperator


class GaussianMutation(CopyingOperator):
    """
    Gaussian mutation operator.

    Follows the algorithm description in:

        Sean Luke, 2013, Essentials of Metaheuristics, Lulu, second edition
        available for free at http://cs.gmu.edu/~sean/book/metaheuristics/
    """

    def __init__(self, problem: Problem, *, stdev: float, mutation_probability: Optional[float] = None):
        """
        `__init__(...)`: Initialize the GaussianMutation.

        Args:
            problem: The problem object to work with.
            stdev: The standard deviation of the Gaussian noise to apply on
                each decision variable.
            mutation_probability: The probability of mutation, for each
                decision variable.
                If None, the value of this argument becomes 1.0, which means
                that all of the decision variables will be affected by the
                mutation. Defatuls to None
        """

        super().__init__(problem)
        self._mutation_probability = 1.0 if mutation_probability is None else float(mutation_probability)
        self._stdev = float(stdev)

    @torch.no_grad()
    def _do(self, batch: SolutionBatch) -> SolutionBatch:
        result = deepcopy(batch)
        data = result.access_values()
        mutation_matrix = self.problem.make_uniform_shaped_like(data) <= self._mutation_probability
        data[mutation_matrix] += self._stdev * self.problem.make_gaussian_shaped_like(data[mutation_matrix])
        data[:] = self._respect_bounds(data)
        return result


class MultiPointCrossOver(CrossOver):
    """
    Representation of a multi-point cross-over operator.

    When this operator is applied on a SolutionBatch, a tournament selection
    technique is used for selecting parent solutions from the batch, and then
    those parent solutions are mated via cutting from a random position and
    recombining. The result of these recombination operations is a new
    SolutionBatch, containing the children solutions. The original
    SolutionBatch stays unmodified.

    This operator is a generalization over the standard cross-over operators
    [OnePointCrossOver][evotorch.operators.real.OnePointCrossOver]
    and [TwoPointCrossOver][evotorch.operators.real.TwoPointCrossOver].
    In more details, instead of having one or two cutting points, this operator
    is configurable in terms of how many cutting points is desired.
    This generalized cross-over implementation follows the procedure described
    in:

        Sean Luke, 2013, Essentials of Metaheuristics, Lulu, second edition
        available for free at http://cs.gmu.edu/~sean/book/metaheuristics/
    """

    def __init__(
        self,
        problem: Problem,
        *,
        tournament_size: int,
        obj_index: Optional[int] = None,
        num_points: Optional[int] = None,
        num_children: Optional[int] = None,
        cross_over_rate: Optional[float] = None,
    ):
        """
        `__init__(...)`: Initialize the MultiPointCrossOver.

        Args:
            problem: The problem object to work on.
            tournament_size: What is the size (or length) of a tournament
                when selecting a parent candidate from a population
            obj_index: Objective index according to which the selection
                will be done.
            num_points: Number of cutting points for the cross-over operator.
            num_children: Optionally a number of children to produce by the
                cross-over operation.
                Not to be used together with `cross_over_rate`.
                If `num_children` and `cross_over_rate` are both None,
                then the number of children is equal to the number
                of solutions received.
            cross_over_rate: Optionally expected as a real number between
                0.0 and 1.0. Specifies the number of cross-over operations
                to perform. 1.0 means `1.0 * len(solution_batch)` amount of
                cross overs will be performed, resulting in
                `2.0 * len(solution_batch)` amount of children.
                Not to be used together with `num_children`.
                If `num_children` and `cross_over_rate` are both None,
                then the number of children is equal to the number
                of solutions received.
        """

        super().__init__(
            problem,
            tournament_size=tournament_size,
            obj_index=obj_index,
            num_children=num_children,
            cross_over_rate=cross_over_rate,
        )

        self._num_points = int(num_points)
        if self._num_points < 1:
            raise ValueError(
                f"Invalid `num_points`: {self._num_points}."
                f" Please provide a `num_points` which is greater than or equal to 1"
            )

    @torch.no_grad()
    def _do_cross_over(self, parents1: torch.Tensor, parents2: torch.Tensor) -> SolutionBatch:
        # What we expect here is this:
        #
        #    parents1      parents2
        #    ==========    ==========
        #    parents1[0]   parents2[0]
        #    parents1[1]   parents2[1]
        #    ...           ...
        #    parents1[N]   parents2[N]
        #
        # where parents1 and parents2 are 2D tensors, each containing values of N solutions.
        # For each row i, we will apply cross-over on parents1[i] and parents2[i].
        # From each cross-over, we will obtain 2 children.
        # This means, there are N pairings, and 2N children.

        num_pairings = parents1.shape[0]
        # num_children = num_pairings * 2

        device = parents1[0].device
        solution_length = len(parents1[0])
        num_points = self._num_points

        # For each pairing, generate all gene indices (i.e. [0, 1, 2, ...] for each pairing)
        gene_indices = (
            torch.arange(0, solution_length, device=device).unsqueeze(0).expand(num_pairings, solution_length)
        )

        if num_points == 1:
            # For each pairing, generate a gene index at which the parent solutions will be cut and recombined
            crossover_point = self.problem.make_randint((num_pairings, 1), n=(solution_length - 1), device=device) + 1

            # Make a mask for crossing over
            # (False: take the value from one parent, True: take the value from the other parent).
            # For gene indices less than crossover_point of that pairing, the mask takes the value 0.
            # Otherwise, the mask takes the value 1.
            crossover_mask = gene_indices >= crossover_point
        else:
            # For each pairing, generate gene indices at which the parent solutions will be cut and recombined
            crossover_points = self.problem.make_randint(
                (num_pairings, num_points), n=(solution_length + 1), device=device
            )

            # From `crossover_points`, extract each cutting point for each solution.
            cutting_points = [crossover_points[:, i].reshape(-1, 1) for i in range(num_points)]

            # Initialize `crossover_mask` as a tensor filled with False.
            crossover_mask = torch.zeros((num_pairings, solution_length), dtype=torch.bool, device=device)

            # For each cutting point p, toggle the boolean values of `crossover_mask`
            # for indices bigger than the index pointed to by p
            for p in cutting_points:
                crossover_mask ^= gene_indices >= p

        # Using the mask, generate two children.
        children1 = torch.where(crossover_mask, parents1, parents2)
        children2 = torch.where(crossover_mask, parents2, parents1)

        # Combine the children tensors in one big tensor
        children = torch.cat([children1, children2], dim=0)

        # Write the children solutions into a new SolutionBatch, and return the new batch
        result = self._make_children_batch(children)
        return result


class OnePointCrossOver(MultiPointCrossOver):
    """
    Representation of a one-point cross-over operator.

    When this operator is applied on a SolutionBatch, a tournament selection
    technique is used for selecting parent solutions from the batch, and then
    those parent solutions are mated via cutting from a random position and
    recombining. The result of these recombination operations is a new
    SolutionBatch, containing the children solutions. The original
    SolutionBatch stays unmodified.

    Let us assume that the two of the parent solutions that were selected for
    the cross-over operation are as follows:

    ```
    a: [ a0 , a1 , a2 , a3 , a4 , a5 ]
    b: [ b0 , b1 , b2 , b3 , b4 , b5 ]
    ```

    For recombining parents `a` and `b`, a cutting point is first randomly
    selected. In the case of this example, let us assume that the cutting
    point was chosen as the point between the items with indices 2 and 3:

    ```
    a: [ a0 , a1 , a2 | a3 , a4 , a5 ]
    b: [ b0 , b1 , b2 | b3 , b4 , b5 ]
                      |
                      ^
           Selected cutting point
    ```

    Considering this selected cutting point, the two children `c` and `d`
    will be constructed from `a` and `b` like this:

    ```
    c: [ a0 , a1 , a2 | b3 , b4 , b5 ]
    d: [ b0 , b1 , b2 | a3 , a4 , a5 ]
    ```

    Note that the recombination procedure explained above is be done on all
    of the parents chosen from the given SolutionBatch, in a vectorized manner.
    For each chosen pair of parents, the cutting points will be sampled
    differently.
    """

    def __init__(
        self,
        problem: Problem,
        *,
        tournament_size: int,
        obj_index: Optional[int] = None,
        num_children: Optional[int] = None,
        cross_over_rate: Optional[float] = None,
    ):
        """
        `__init__(...)`: Initialize the OnePointCrossOver.

        Args:
            problem: The problem object to work on.
            tournament_size: What is the size (or length) of a tournament
                when selecting a parent candidate from a population
            obj_index: Objective index according to which the selection
                will be done.
            num_children: Optionally a number of children to produce by the
                cross-over operation.
                Not to be used together with `cross_over_rate`.
                If `num_children` and `cross_over_rate` are both None,
                then the number of children is equal to the number
                of solutions received.
            cross_over_rate: Optionally expected as a real number between
                0.0 and 1.0. Specifies the number of cross-over operations
                to perform. 1.0 means `1.0 * len(solution_batch)` amount of
                cross overs will be performed, resulting in
                `2.0 * len(solution_batch)` amount of children.
                Not to be used together with `num_children`.
                If `num_children` and `cross_over_rate` are both None,
                then the number of children is equal to the number
                of solutions received.
        """
        super().__init__(
            problem,
            tournament_size=tournament_size,
            obj_index=obj_index,
            num_points=1,
            num_children=num_children,
            cross_over_rate=cross_over_rate,
        )


class TwoPointCrossOver(MultiPointCrossOver):
    """
    Representation of a two-point cross-over operator.

    When this operator is applied on a SolutionBatch, a tournament selection
    technique is used for selecting parent solutions from the batch, and then
    those parent solutions are mated via cutting from a random position and
    recombining. The result of these recombination operations is a new
    SolutionBatch, containing the children solutions. The original
    SolutionBatch stays unmodified.

    Let us assume that the two of the parent solutions that were selected for
    the cross-over operation are as follows:

    ```
    a: [ a0 , a1 , a2 , a3 , a4 , a5 ]
    b: [ b0 , b1 , b2 , b3 , b4 , b5 ]
    ```

    For recombining parents `a` and `b`, two cutting points are first randomly
    selected. In the case of this example, let us assume that the cutting
    point were chosen as the point between the items with indices 1 and 2,
    and between 3 and 4:

    ```
    a: [ a0 , a1 | a2 , a3 | a4 , a5 ]
    b: [ b0 , b1 | b2 , b3 | b4 , b5 ]
                 |         |
                 ^         ^
               First     Second
              cutting    cutting
               point     point
    ```

    Given these two cutting points, the two children `c` and `d` will be
    constructed from `a` and `b` like this:

    ```
    c: [ a0 , a1 | b2 , b3 | a4 , a5 ]
    d: [ b0 , b1 | a2 , a3 | b4 , b5 ]
    ```

    Note that the recombination procedure explained above is be done on all
    of the parents chosen from the given SolutionBatch, in a vectorized manner.
    For each chosen pair of parents, the cutting points will be sampled
    differently.
    """

    def __init__(
        self,
        problem: Problem,
        *,
        tournament_size: int,
        obj_index: Optional[int] = None,
        num_children: Optional[int] = None,
        cross_over_rate: Optional[float] = None,
    ):
        """
        `__init__(...)`: Initialize the TwoPointCrossOver.

        Args:
            problem: The problem object to work on.
            tournament_size: What is the size (or length) of a tournament
                when selecting a parent candidate from a population
            obj_index: Objective index according to which the selection
                will be done.
            num_children: Optionally a number of children to produce by the
                cross-over operation.
                Not to be used together with `cross_over_rate`.
                If `num_children` and `cross_over_rate` are both None,
                then the number of children is equal to the number
                of solutions received.
            cross_over_rate: Optionally expected as a real number between
                0.0 and 1.0. Specifies the number of cross-over operations
                to perform. 1.0 means `1.0 * len(solution_batch)` amount of
                cross overs will be performed, resulting in
                `2.0 * len(solution_batch)` amount of children.
                Not to be used together with `num_children`.
                If `num_children` and `cross_over_rate` are both None,
                then the number of children is equal to the number
                of solutions received.
        """
        super().__init__(
            problem,
            tournament_size=tournament_size,
            obj_index=obj_index,
            num_points=2,
            num_children=num_children,
            cross_over_rate=cross_over_rate,
        )


class SimulatedBinaryCrossOver(CrossOver):
    """
    Representation of a simulated binary cross-over (SBX).

    When this operator is applied on a SolutionBatch,
    a tournament selection technique is used for selecting
    parent solutions from the batch, and then those parent
    solutions are mated via SBX. The generated children
    solutions are given in a new SolutionBatch.
    The original SolutionBatch stays unmodified.

    Reference:

        Kalyanmoy Deb, Hans-Georg Beyer (2001).
        Self-Adaptive Genetic Algorithms with Simulated Binary Crossover.
    """

    def __init__(
        self,
        problem: Problem,
        *,
        tournament_size: int,
        eta: float,
        obj_index: Optional[int] = None,
        num_children: Optional[int] = None,
        cross_over_rate: Optional[float] = None,
    ):
        """
        `__init__(...)`: Initialize the SimulatedBinaryCrossOver.

        Args:
            problem: Problem object to work with.
            tournament_size: What is the size (or length) of a tournament
                when selecting a parent candidate from a population.
            eta: The crowding index, expected as a float.
                Bigger eta values result in children closer
                to their parents.
            obj_index: Objective index according to which the selection
                will be done.
            num_children: Optionally a number of children to produce by the
                cross-over operation.
                Not to be used together with `cross_over_rate`.
                If `num_children` and `cross_over_rate` are both None,
                then the number of children is equal to the number
                of solutions received.
            cross_over_rate: Optionally expected as a real number between
                0.0 and 1.0. Specifies the number of cross-over operations
                to perform. 1.0 means `1.0 * len(solution_batch)` amount of
                cross overs will be performed, resulting in
                `2.0 * len(solution_batch)` amount of children.
                Not to be used together with `num_children`.
                If `num_children` and `cross_over_rate` are both None,
                then the number of children is equal to the number
                of solutions received.
        """

        super().__init__(
            problem,
            tournament_size=int(tournament_size),
            obj_index=obj_index,
            num_children=num_children,
            cross_over_rate=cross_over_rate,
        )
        self._eta = float(eta)

    def _do_cross_over(self, parents1: torch.Tensor, parents2: torch.Tensor) -> SolutionBatch:
        # Generate u_i values which determine the spread
        u = self.problem.make_uniform_shaped_like(parents1)

        # Compute beta_i values from u_i values as the actual spread per dimension
        betas = (2 * u).pow(1.0 / (self._eta + 1.0))  # Compute all values for u_i < 0.5 first
        betas[u > 0.5] = (1.0 / (2 * (1.0 - u[u > 0.5]))).pow(
            1.0 / (self._eta + 1.0)
        )  # Replace the values for u_i >= 0.5
        children1 = 0.5 * (
            (1 + betas) * parents1 + (1 - betas) * parents2
        )  # Create the first set of children from the beta values
        children2 = 0.5 * (
            (1 + betas) * parents2 + (1 - betas) * parents1
        )  # Create the second set of children as a mirror of the first set of children

        # Combine the children tensors in one big tensor
        children = torch.cat([children1, children2], dim=0)

        # Respect the lower and upper bounds defined by the problem object
        children = self._respect_bounds(children)

        # Write the children solutions into a new SolutionBatch, and return the new batch
        result = self._make_children_batch(children)

        return result


class PolynomialMutation(CopyingOperator):
    """
    Representation of the polynomial mutation operator.

    Follows the algorithm description in:

        Kalyanmoy Deb, Santosh Tiwari (2008).
        Omni-optimizer: A generic evolutionary algorithm for single
        and multi-objective optimization

    The operator ensures a non-zero probability of generating offspring in
    the entire search space by dividing the space into two regions and using
    independent probability distributions associated with each region.
    In contrast, the original polynomial mutation formulation may render the
    mutation ineffective when the decision variable approaches its boundary.
    """

    def __init__(
        self,
        problem: Problem,
        *,
        eta: Optional[float] = None,
        mutation_probability: Optional[float] = None,
    ):
        """
        `__init__(...)`: Initialize the PolynomialMutation.

        Args:
            problem: The problem object to work with.
            eta: The index for polynomial mutation; a large value gives a higher
                probability for creating near-parent solutions, whereas a small
                value allows distant solutions to be created.
                If not specified, `eta` will be assumed as 20.0.
            mutation_probability: The probability of mutation, for each decision
                variable. If not specified, all variables will be mutated.
        """

        super().__init__(problem)

        if "float" not in str(problem.dtype):
            raise ValueError(
                f"This operator can be used only when `dtype` of the problem is float type"
                f" (like, e.g. torch.float32, torch.float64, etc.)"
                f" The dtype of the problem is {problem.dtype}."
            )

        if (self.problem.lower_bounds is None) or (self.problem.upper_bounds is None):
            raise ValueError(
                "The polynomial mutation operator can be used only when the problem object has"
                " `lower_bounds` and `upper_bounds`."
                " In the given problem object, at least one of them appears to be missing."
            )

        if torch.any(self.problem.lower_bounds > self.problem.upper_bounds):
            raise ValueError("Some of the `lower_bounds` appear greater than their `upper_bounds`")

        self._prob = None if mutation_probability is None else float(mutation_probability)
        self._eta = 20.0 if eta is None else float(eta)
        self._lb = self.problem.lower_bounds
        self._ub = self.problem.upper_bounds

    @torch.no_grad()
    def _do(self, batch: SolutionBatch) -> SolutionBatch:
        # Take a copy of the original batch. Modifications will be done on this copy.
        result = deepcopy(batch)

        # Take the decision values tensor from within the newly made copy of the batch (`result`).
        # Any modification done on this tensor will affect the `result` batch.
        data = result.access_values()

        # Take the population size
        pop_size, solution_length = data.size()

        if self._prob is None:
            # If a probability of mutation is not given, then we prepare our mutation mask (`to_mutate`) as a tensor
            # consisting only of `True`s.
            to_mutate = torch.ones(data.shape, dtype=torch.bool, device=data.device)
        else:
            # If a probability of mutation is given, then we produce a boolean mask that probabilistically marks which
            # decision variables will be affected by this mutation operation.
            to_mutate = self.problem.make_uniform_shaped_like(data) < self._prob

        # Obtain a flattened (1-dimensional) tensor which addresses only the variables that are subject to mutation
        # (i.e. variables that are not subject to mutation are filtered out).
        selected = data[to_mutate]

        # Obtain flattened (1-dimensional) lower and upper bound tensors such that `lb[i]` and `ub[i]` specify the
        # bounds for `selected[i]`.
        lb = self._lb.expand(pop_size, solution_length)[to_mutate]
        ub = self._ub.expand(pop_size, solution_length)[to_mutate]

        # Apply the mutation procedure explained by Deb & Tiwari (2008).
        delta_1 = (selected - lb) / (ub - lb)
        delta_2 = (ub - selected) / (ub - lb)

        r = self.problem.make_uniform(selected.size())
        mask = r < 0.5
        mask_not = torch.logical_not(mask)

        mut_str = 1.0 / (self._eta + 1.0)
        delta_q = torch.zeros_like(selected)

        v = 2.0 * r + (1.0 - 2.0 * r) * (1.0 - delta_1).pow(self._eta + 1.0)
        d = v.pow(mut_str) - 1.0
        delta_q[mask] = d[mask]

        v = 2.0 * (1.0 - r) + 2.0 * (r - 0.5) * (1.0 - delta_2).pow(self._eta + 1.0)
        d = 1.0 - v.pow(mut_str)
        delta_q[mask_not] = d[mask_not]

        mutated = selected + delta_q * (ub - lb)

        # Put the mutated decision values into the decision variables tensor stored within the `result` batch.
        data[to_mutate] = mutated

        # Prevent violations that could happen because of numerical errors.
        data[:] = self._respect_bounds(data)

        # Return the `result` batch.
        return result


class CosynePermutation(CopyingOperator):
    """
    Representation of permutation operation on a SolutionBatch.

    For each decision variable index, a permutation operation across
    all or a subset of solutions, is performed.
    The result is returned on a new SolutionBatch.
    The original SolutionBatch remains unmodified.

    Reference:

        F.Gomez, J.Schmidhuber, R.Miikkulainen (2008).
        Accelerated Neural Evolution through Cooperatively Coevolved Synapses
        Journal of Machine Learning Research 9, 937-965
    """

    def __init__(self, problem: Problem, obj_index: Optional[int] = None, *, permute_all: bool = False):
        """
        `__init__(...)`: Initialize the CosynePermutation.

        Args:
            problem: The problem object to work on.
            obj_index: The index of the objective according to which the
                candidates for permutation will be selected.
                Can be left as None if the problem is single-objective,
                or if `permute_all` is given as True (in which case there
                will be no candidate selection as the entire population will
                be subject to permutation).
            permute_all: Whether or not to apply permutation on the entire
                population, instead of using a selective permutation.
        """

        if permute_all:
            if obj_index is not None:
                raise ValueError(
                    "When `permute_all` is given as True (which seems to be the case)"
                    " `obj_index` is expected as None,"
                    " because the operator is independent of any objective and any fitness in this mode."
                    " However, `permute_all` was found to be something other than None."
                )
            self._obj_index = None
        else:
            self._obj_index = problem.normalize_obj_index(obj_index)

        super().__init__(problem)

        self._permute_all = bool(permute_all)

    @property
    def obj_index(self) -> Optional[int]:
        """Objective index according to which the operator will run.
        If `permute_all` was given as True, objectives are irrelevant, in which case
        `obj_index` is returned as None.
        If `permute_all` was given as False, the relevant `obj_index` is provided
        as an integer.
        """
        return self._obj_index

    @torch.no_grad()
    def _do(self, batch: SolutionBatch) -> SolutionBatch:
        indata = batch._data

        if not self._permute_all:
            n = batch.solution_length
            ranks = batch.utility(self._obj_index, ranking_method="centered")
            # fitnesses = batch.evals[:, self._obj_index].clone().reshape(-1)
            # ranks = rank(
            #    fitnesses, ranking_method="centered", higher_is_better=(self.problem.senses[self.obj_index] == "max")
            # )
            prob_permute = (1 - (ranks + 0.5).pow(1 / float(n))).unsqueeze(1).expand(len(batch), batch.solution_length)
        else:
            prob_permute = torch.ones_like(indata)

        perm_mask = self.problem.make_uniform_shaped_like(prob_permute) <= prob_permute

        perm_mask_sorted = torch.sort(perm_mask.to(torch.long), descending=True, dim=0)[0].to(
            torch.bool
        )  # Sort permutations

        perm_rand = self.problem.make_uniform_shaped_like(prob_permute)
        perm_rand[torch.logical_not(perm_mask)] = 1.0
        permutations = torch.argsort(perm_rand, dim=0)  # Generate permutations

        perm_sort = (
            torch.arange(0, perm_mask.shape[0], device=indata.device).unsqueeze(-1).repeat(1, perm_mask.shape[1])
        )
        perm_sort[torch.logical_not(perm_mask)] += perm_mask.shape[0] + 1
        perm_sort = torch.sort(perm_sort, dim=0)[0]  # Generate the origin of permutations

        _, permutation_columns = torch.nonzero(perm_mask_sorted, as_tuple=True)
        permutation_origin_indices = perm_sort[perm_mask_sorted]
        permutation_target_indices = permutations[perm_mask_sorted]

        newbatch = SolutionBatch(like=batch, empty=True)
        newdata = newbatch._data
        newdata[:] = indata[:]
        newdata[permutation_origin_indices, permutation_columns] = newdata[
            permutation_target_indices, permutation_columns
        ]

        return newbatch
