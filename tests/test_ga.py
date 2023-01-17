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


import itertools
from collections import namedtuple
from typing import Type

import pytest
import torch

from evotorch import Problem, SolutionBatch
from evotorch.algorithms import GeneticAlgorithm, MAPElites, SteadyStateGA
from evotorch.operators import (
    GaussianMutation,
    MultiPointCrossOver,
    OnePointCrossOver,
    PolynomialMutation,
    SimulatedBinaryCrossOver,
    TwoPointCrossOver,
)
from evotorch.tools import DType, make_uniform, to_torch_dtype

OperatorInfo = namedtuple("OperatorInfo", ["operator_type", "operator_kwargs"])


class DummyProblem(Problem):
    def __init__(self, dtype: DType):
        dtype = to_torch_dtype(dtype)

        if dtype == torch.bool:
            bounds = (False, True)
        else:
            bounds = (-10, 10)

        super().__init__(
            objective_sense="min",
            solution_length=10,
            dtype=dtype,
            bounds=bounds,
            eval_data_length=2,
            store_solution_stats=True,
        )

    def _evaluate_batch(self, batch: SolutionBatch):
        values = torch.as_tensor(batch.access_values(), dtype=torch.float32)
        evals = batch.access_evals()
        evals[:, 0] = torch.linalg.norm(values, dim=-1)
        evals[:, 1] = torch.sin(torch.sum(values, dim=-1))
        evals[:, 2] = torch.cos(torch.sum(values, dim=-1))


class Helpers:
    """Helper functions for our tests"""

    @staticmethod
    def simple_mutation(values: torch.Tensor) -> torch.Tensor:
        """
        A simple mutation function that can be used as an operator.

        Args:
            values: Original decision values
        Returns:
            Mutated decision values
        """
        if values.dtype == torch.bool:
            modified_values = make_uniform(values.shape, lb=False, ub=True, dtype=torch.bool)
        else:
            modified_values = make_uniform(values.shape, lb=-10, ub=10, dtype=values.dtype)
        to_be_modified = make_uniform(values.shape, lb=False, ub=True, dtype=torch.bool)
        return torch.where(to_be_modified, modified_values, values)

    @staticmethod
    def suggest_cross_overs(dtype: torch.dtype) -> list:
        """
        Suggest a list of cross-over configurations for a given dtype.

        We would like to run our tests with various cross-over operators.
        However, the cross-over operators we can use depend on with which dtype
        the target problem was instantiated. Therefore, we define this helper
        function which returns a list of OperatorInfo objects, specifying which
        operator types and which keyword arguments can be used.

        Returns:
            A list of OperatorInfo objects.
        """
        result = [
            OperatorInfo(operator_type=OnePointCrossOver, operator_kwargs={"tournament_size": 4}),
            OperatorInfo(operator_type=TwoPointCrossOver, operator_kwargs={"tournament_size": 4}),
            OperatorInfo(operator_type=MultiPointCrossOver, operator_kwargs={"tournament_size": 4, "num_points": 3}),
        ]
        if dtype == torch.float32:
            result.append(
                OperatorInfo(operator_type=SimulatedBinaryCrossOver, operator_kwargs={"tournament_size": 4, "eta": 20})
            )
        return result

    @classmethod
    def suggest_mutations(cls, dtype: torch.dtype) -> list:
        """
        Suggest a list of mutation configurations for a given dtype.

        We would like to run our tests with various mutation operators.
        However, the mutation operators we can use depend on with which dtype the
        target problem was instantiated. Therefore, we define this function which
        returns a list of operators, where an operator is expressed either via an
        OperatorInfo object or via a reference to a function.

        Returns:
            A list of mutation operator configurations.
        """
        result = [cls.simple_mutation]
        if dtype == torch.float32:
            result.append(OperatorInfo(operator_type=GaussianMutation, operator_kwargs={"stdev": 1.0}))
            result.append(OperatorInfo(operator_type=PolynomialMutation, operator_kwargs={"eta": 20}))
        return result

    @classmethod
    def suggest_operators(cls, dtype: torch.dtype) -> list:
        """
        Suggest a list of operator suggestions for a given dtype.

        We would like to run our tests with various combinations of operators.
        However, the cross-over and mutation operators we can use depend on with
        which dtype the target problem was instantiated. Therefore, we define this
        helper function which returns a list of operator suggestions.
        The returned list looks like this:

        ```
        [
            # first suggestion:
            [
                # cross-over operator:
                OperatorInfo(operator_type=..., operator_kwargs=...),
                # mutation operator:
                OperatorInfo(operator_type=..., operator_kwargs=...),
            ],

            # second suggestion:
            [
                # cross-over operator:
                OperatorInfo(operator_type=..., operator_kwargs=...),
                # mutation operator:
                OperatorInfo(operator_type=..., operator_kwargs=...),
            ],

            # third suggestion (with a direct reference to a function)
            [
                # cross-over operator:
                OperatorInfo(operator_type=..., operator_kwargs=...),
                # mutation operator:
                mutation_function,
            ],

            # ... and so on ...
        ]
        ```

        Each suggestion (i.e. sublist) within this returned list can be
        passed to the function `instantiate_operators(...)`, so that a new
        list is obtained in which each operator is instantiated and ready
        to be used. This new list can then be passed to the `operators`
        keyword argument of a genetic algorithm.

        Returns:
            A list of suggestions
        """
        return list(itertools.product(cls.suggest_cross_overs(dtype), cls.suggest_mutations(dtype)))

    @staticmethod
    def instantiate_operators(problem: Problem, operator_list: list) -> list:
        """
        Instantiate all the operators in an operator suggestion.

        Let us assume that we received a `problem`, and a list like this:

        ```
        [
            OperatorInfo(operator_type=MyOperator, operator_kwargs={"a": 1, "b": 2}),
            OperatorInfo(operator_type=MyOtherOperator, operator_kwargs={}),
        ]
        ```

        then, what is returned is another list in which the operators are
        instantiated and ready to be used:

        ```
        [
            MyOperator(problem, a=1, b=2),
            MyOtherOperator(problem),
        ]
        ```

        If one of the operators is a reference to a function, that function
        itself ends up in the result. For example:

        ```
        [
            OperatorInfo(operator_type=MyOperator, operator_kwargs={"a": 1, "b": 2}),
            my_mutation_function,
        ]
        ```

        becomes:

        ```
        [
            MyOperator(problem, a=1, b=2),
            my_mutation_function,
        ]
        ```

        Args:
            problem: The problem that is being worked on
            operator_list: A list which contains OperatorInfo or function references
        Returns:
            A new list in which operators are instantiated
        """
        result = []
        for op in operator_list:
            if isinstance(op, tuple):
                op_type, op_kwargs = op
                result.append(op_type(problem, **op_kwargs))
            else:
                result.append(op)
        return result


@pytest.mark.parametrize(
    "dtype,ga_type",
    itertools.product(
        [
            torch.float32,
            torch.int64,
            torch.bool,
        ],
        [
            GeneticAlgorithm,
            SteadyStateGA,
            MAPElites,
        ],
    ),
)
def test_ga(dtype: DType, ga_type: Type):
    # Obtain a list of operator suggestions from the problem object
    operator_suggestions = Helpers.suggest_operators(dtype)
    for suggestion in operator_suggestions:
        # Instantiate the dummy problem
        problem = DummyProblem(dtype)

        # For each suggestion within the suggestions list, instantiate the operators
        operators = Helpers.instantiate_operators(problem, suggestion)

        # Put the instantiated operators into the keyword arguments dictionary that we will use
        ga_kwargs = {"operators": operators}

        if issubclass(ga_type, MAPElites):
            # If the algorithm being tested is MAPElites, we make a feature grid
            feature_grid = MAPElites.make_feature_grid(
                lower_bounds=torch.tensor([-1, -1], dtype=torch.float32),
                upper_bounds=torch.tensor([1, 1], dtype=torch.float32),
                num_bins=10,
            )
            ga_kwargs["feature_grid"] = feature_grid
        else:
            # If the algorithm being tested is not MAPElites, we specify a popsize
            ga_kwargs["popsize"] = 100

        # Instantiate the search algorithm
        ga = ga_type(problem, **ga_kwargs)

        # Run the evolution
        ga.run(2)

        assert ga.step_count == 2
        assert "best" in ga.status
        assert "pop_best" in ga.status
