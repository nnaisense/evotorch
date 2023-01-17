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

"""Base classes for various operators"""

import math
from typing import Optional, Union

import torch

from ..core import Problem, SolutionBatch
from ..tools import DType, clip_tensor, clone
from ..tools.objectarray import ObjectArray


class Operator:
    """Base class for various operations on SolutionBatch objects.

    Some subclasses of Operator may be operating on the batches in-place,
    while some others may generate new batches, leaving the original batches
    untouched.

    One is expected to override the definition of the method `_do(...)`
    in an inheriting subclass to define a custom `Operator`.

    From outside, a subclass of Operator is meant to be called like
    a function. In more details, operators which apply in-place modifications
    are meant to be called like this:

        my_operator_instance(my_batch)

    Operators which return a new batch are meant to be called like this:

        my_new_batch = my_operator_instance(my_batch)
    """

    def __init__(self, problem: Problem):
        """
        `__init__(...)`: Initialize the Operator.

        Args:
            problem: The problem object which is being worked on.
        """
        if not isinstance(problem, Problem):
            raise TypeError(f"Expected a Problem object, but received {repr(problem)}")
        self._problem = problem
        self._lb = clone(self._problem.lower_bounds)
        self._ub = clone(self._problem.upper_bounds)

    @property
    def problem(self) -> Problem:
        """Get the problem to which this cross-over operator is bound"""
        return self._problem

    @property
    def dtype(self) -> DType:
        """Get the dtype of the bound problem.
        If the problem does not work with Solution and
        therefore it does not have a dtype, None is returned.
        """
        return self.problem.dtype

    @torch.no_grad()
    def _respect_bounds(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make sure that a given PyTorch tensor respects the problem's bounds.

        This is a protected method which might be used by the
        inheriting subclasses to ensure that the result of their
        various operations are clipped properly to respect the
        boundaries set by the problem object.

        Note that this function might return the tensor itself
        is the problem is not bounded.

        Args:
            x: The PyTorch tensor to be clipped.
        Returns:
            The clipped tensor.
        """
        if self._lb is not None:
            self._lb = torch.as_tensor(self._lb, dtype=x.dtype, device=x.device)
            x = torch.max(self._lb, x)
        if self._ub is not None:
            self._ub = torch.as_tensor(self._ub, dtype=x.dtype, device=x.device)
            x = torch.min(self._ub, x)
        return x

    def __call__(self, batch: SolutionBatch):
        """
        Apply the operator on the given batch.
        """
        if not isinstance(batch, SolutionBatch):
            raise TypeError(
                f"The operation {self.__class__.__name__} can only work on"
                f" SolutionBatch objects, but it received an object of type"
                f" {repr(type(batch))}."
            )
        self._do(batch)

    def _do(self, batch: SolutionBatch):
        """
        The actual definition of the operation on the batch.
        Expected to be overriden by a subclass.
        """
        raise NotImplementedError


class CopyingOperator(Operator):
    """
    Base class for operators which do not do in-place modifications.

    This class does not add any functionality to the Operator class.
    Instead, the annotations of the `__call__(...)` method is
    updated so that it makes it clear that a new SolutionBatch is
    returned.

    One is expected to override the definition of the method `_do(...)`
    in an inheriting subclass to define a custom `CopyingOperator`.

    From outside, a subclass of `CopyingOperator` is meant to be called like
    a function, as follows:

        my_new_batch = my_copying_operator_instance(my_batch)
    """

    def __init__(self, problem: Problem):
        """
        `__init__(...)`: Initialize the CopyingOperator.

        Args:
            problem: The problem object which is being worked on.
        """
        super().__init__(problem)

    def __call__(self, batch: SolutionBatch) -> SolutionBatch:
        return self._do(batch)

    def _do(self, batch: SolutionBatch) -> SolutionBatch:
        """The actual definition of the operation on the batch.
        Expected to be overriden by a subclass.
        """
        raise NotImplementedError


class CrossOver(CopyingOperator):
    """
    Base class for any CrossOver operator.

    One is expected to override the definition of the method
    `_do_cross_over(...)` in an inheriting subclass to define a
    custom `CrossOver`.

    From outside, a `CrossOver` instance is meant to be called like this:

        child_solution_batch = my_cross_over_instance(population_batch)

    which causes the `CrossOver` instance to select parents from the
    `population_batch`, recombine their values according to what is
    instructed in `_do_cross_over(...)`, and return the newly made solutions
    in a `SolutionBatch`.
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
        `__init__(...)`: Initialize the CrossOver.

        Args:
            problem: The problem object which is being worked on.
            tournament_size: Size of the tournament which will be used for
                doing selection.
            obj_index: Index of the objective according to which the selection
                will be done.
                If `obj_index` is None and the problem is single-objective,
                then the selection will be done according to that single
                objective.
                If `obj_index` is None and the problem is multi-objective,
                then the selection will be done according to pareto-dominance
                and crowding criteria, as done in NSGA-II.
                If `obj_index` is an integer `i`, then the selection will be
                done according to the i-th objective only, even when the
                problem is multi-objective.
            num_children: How many children to generate.
                Expected as an even number.
                Cannot be used together with `cross_over_rate`.
            cross_over_rate: Rate of the cross-over operations in comparison
                with the population size.
                1.0 means that the number of generated children will be equal
                to the original population size.
                Cannot be used together with `num_children`.
        """
        super().__init__(problem)

        self._obj_index = None if obj_index is None else problem.normalize_obj_index(obj_index)
        self._tournament_size = int(tournament_size)

        if num_children is not None and cross_over_rate is not None:
            raise ValueError(
                "Received both `num_children` and `cross_over_rate` as values other than None."
                " It was expected to receive both of them as None, or one of them as None,"
                " but not both of them as values other than None."
            )

        self._num_children = None if num_children is None else int(num_children)
        self._cross_over_rate = None if cross_over_rate is None else float(cross_over_rate)

    def _compute_num_tournaments(self, batch: SolutionBatch) -> int:
        if self._num_children is None and self._cross_over_rate is None:
            # return len(batch) * 2
            result = len(batch)
            if (result % 2) != 0:
                result += 1
            return result
        elif self._num_children is not None:
            if (self._num_children % 2) != 0:
                raise ValueError(
                    f"The initialization argument `num_children` was expected as an even number."
                    f" However, it was found as an odd number: {self._num_children}"
                )
            return self._num_children
        elif self._cross_over_rate is not None:
            f = len(batch) * self._cross_over_rate
            result1 = math.ceil(f)
            result2 = math.floor(f)
            if result1 == result2:
                result = result1
                if (result % 2) != 0:
                    result += 1
            else:
                if (result1 % 2) == 0:
                    result = result1
                else:
                    result = result2

            return result
        else:
            assert False, "Exection should not have reached this point"

    @property
    def obj_index(self) -> Optional[int]:
        """The objective index according to which the selection will be done"""
        return self._obj_index

    @torch.no_grad()
    def _do_tournament(self, batch: SolutionBatch) -> tuple:
        # Compute the required number of tournaments
        num_tournaments = self._compute_num_tournaments(batch)

        if self._problem.is_multi_objective and self._obj_index is None:
            # If the problem is multi-objective, and an objective index is not specified,
            # then we do a multi-objective-specific cross-over

            # At first, pareto-sort the solutions
            ranks, _ = batch.compute_pareto_ranks(crowdsort=False)
            n_fronts = torch.amax(ranks) + 1

            # In NSGA-II-inspired pareto-sorting, smallest rank means the best front.
            # Right now, we want the opposite: we want the solutions in the best front
            # to have rank values which are numerically highest.
            # The following line re-arranges the rank values such that the solutions
            # in the best front have their ranks equal to n_fronts, and the ones
            # in the worst front have their ranks equal to 1.
            ranks = (n_fronts - ranks).to(torch.float)

            # Because the ranks are computed front the fronts indices, we expect many
            # solutions to end up with the same rank values.
            # To ensure that a randomized selection will be made when comparing two
            # solutions with the same rank, we add random noise to the ranks
            # (between 0.0 and 0.1).
            ranks += self._problem.make_uniform(len(batch), dtype=self._problem.eval_dtype, device=batch.device) * 0.1
        else:
            # Rank the solutions. Worst gets -0.5, best gets 0.5
            ranks = batch.utility(self._obj_index, ranking_method="centered")

        # Get the internal values tensor of the solution batch
        indata = batch._data

        # Get a tensor of random integers in the shape (num_tournaments, tournament_size)
        tournament_indices = self.problem.make_randint(
            (num_tournaments, self._tournament_size), n=len(batch), device=indata.device
        )
        tournament_ranks = ranks[tournament_indices]

        # Imagine tournament size is 2, and the solutions are [ worst, bad, best, good ].
        # So, what we have is (0.2s are actually 0.166666...):
        #
        #    ranks = [ -0.5, -0.2, 0.5, 0.2 ]
        #
        #    tournament     tournament
        #     indices         ranks
        #
        #       0, 1        -0.5, -0.2
        #       2, 3         0.5,  0.2
        #       1, 0        -0.2, -0.5
        #       3, 2         0.2,  0.5
        #       1, 2        -0.2,  0.5
        #       0, 3        -0.5,  0.2
        #       2, 0         0.5, -0.5
        #       3, 1         0.2, -0.2
        #
        # According to tournament_indices, there are 8 tournaments.
        #   In tournament 0 (topmost row), parent0 and parent1 compete.
        #   In tournament 1 (next row), parent2 and parent3 compete; and so on.
        # tournament_ranks tells us:
        #   In tournament 0, left-candidate has rank -0.5, and right-candidate has -0.2.
        #   In tournament 1, left-candidate has rank 0.5, and right-candidate has 0.2; and so on.

        tournament_rows = torch.arange(0, num_tournaments, device=indata.device)
        parents = tournament_indices[tournament_rows, torch.argmax(tournament_ranks, dim=-1)]

        # Continuing from the [ worst, bad, best, good ] example, we end up with:
        #
        #            T                                    T
        #  tournament   tournament     tournament   argmax     parents
        #    rows        indices         ranks      dim=-1
        #
        #      0          0, 1        -0.5, -0.2      1           1
        #      1          2, 3         0.5,  0.2      0           2
        #      2          1, 0        -0.2, -0.5      0           1
        #      3          3, 2         0.2,  0.5      1           2
        #      4          1, 2        -0.2,  0.5      1           2
        #      5          0, 3        -0.5,  0.2      1           3
        #      6          2, 0         0.5, -0.5      0           2
        #      7          3, 1         0.2, -0.2      0           3
        #
        # where tournament_rows represents row indices in tournament_indices tensor (from 0 to 7).
        # argmax() tells us who won the competition (0: left-candidate won, 1: right-candidate won).
        #
        # tournament_rows and argmax() together give us the row and column of the winner in tensor
        # tournament_indices, which in turn gives us the index of the winner solution in the batch.

        # We split the parents array from the middle
        split_point = int(len(parents) / 2)
        parents1 = indata[parents][:split_point]
        parents2 = indata[parents][split_point:]
        # We now have:
        #
        #   parents1            parents2
        #   ===============     ===============
        #   values of sln 1     values of sln 2   (solution1 is to generate a child with solution2)
        #   values of sln 2     values of sln 3   (solution2 is to generate a child with solution3)
        #   values of sln 1     values of sln 2   (solution1 is to generate another child with solution2)
        #   values of sln 2     values of sln 3   (solution2 is to generate another child with solution3)
        #
        # With this, the tournament selection phase is over.
        return parents1, parents2

    def _do_cross_over(
        self,
        parents1: Union[torch.Tensor, ObjectArray],
        parents2: Union[torch.Tensor, ObjectArray],
    ) -> SolutionBatch:
        """
        The actual definition of the cross-over operation.

        This is a protected method, meant to be overriden by the inheriting
        subclass.

        The arguments passed to this function are the decision values of the
        first and the second half of the selected parents, both as PyTorch
        tensors or as `ObjectArray`s.

        In the overriding function, for each integer i, one is expected to
        recombine the values of the i-th row of `parents1` with the values of
        the i-th row of `parents2` twice (twice because each pairing is
        expected to generate two children).
        After that, one is expected to generate a SolutionBatch and place
        all the recombination results into the values of that new batch.

        Args:
            parents1: The decision values of the first half of the
                selected parents.
            parents2: The decision values of the second half of the
                selected parents.
        Returns:
            A new SolutionBatch which contains the recombination
            of the parents.
        """
        raise NotImplementedError

    def _make_children_batch(self, child_values: Union[torch.Tensor, ObjectArray]) -> SolutionBatch:
        result = SolutionBatch(self.problem, device=child_values.device, empty=True, popsize=child_values.shape[0])
        result._data = child_values
        return result

    def _do(self, batch: SolutionBatch) -> SolutionBatch:
        parents1, parents2 = self._do_tournament(batch)
        if len(parents1) != len(parents2):
            raise ValueError(
                f"_do_tournament() returned parents1 and parents2 with incompatible sizes. "
                f"len(parents1): {len(parents1)}; len(parents2): {len(parents2)}."
            )
        return self._do_cross_over(parents1, parents2)


class SingleObjOperator(Operator):
    """
    Base class for all the operators which focus on only one objective.

    One is expected to override the definition of the method `_do(...)`
    in an inheriting subclass to define a custom `SingleObjOperator`.
    """

    def __init__(self, problem: Problem, obj_index: Optional[int] = None):
        """
        Initialize the SingleObjOperator.

        Args:
            problem: The problem object which is being worked on.
            obj_index: Index of the objective to focus on.
                Can be given as None if the problem is single-objective.
        """
        super().__init__(problem)
        self._obj_index: int = problem.normalize_obj_index(obj_index)

    @property
    def obj_index(self) -> int:
        """Index of the objective on which this operator is to be applied"""
        return self._obj_index
