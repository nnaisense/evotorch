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

from typing import Sequence, Union

import pytest
import torch

import evotorch.operators.functional as func_ops
from evotorch import testing
from evotorch.decorators import rowwise
from evotorch.tools import ObjectArray, make_tensor


def test_combine_single_obj():
    tolerance = 1e-8
    solution_length = 5
    batch_size = 3

    popsizeA = 6
    populationA = (
        torch.arange(batch_size * popsizeA * solution_length)
        .reshape(batch_size, popsizeA, solution_length)
        .to(dtype=torch.float32)
    )
    evalsA = populationA.sum(dim=-1)

    popsizeB = 3
    populationB = (
        torch.arange(batch_size * popsizeB * solution_length)
        .reshape(batch_size, popsizeB, solution_length)
        .to(dtype=torch.float32)
        * -1
    )
    evalsB = populationB.sum(dim=-1)

    # test the unbatched case
    unbatched_combined_pop, unbatched_combined_evals = func_ops.combine(
        (populationA[0], evalsA[0]), (populationB[0], evalsB[0])
    )
    unbatched_expected_pop_shape = (popsizeA + popsizeB, solution_length)
    unbatched_expected_evals_shape = popsizeA + popsizeB
    testing.assert_shape_matches(unbatched_combined_pop, unbatched_expected_pop_shape)
    testing.assert_shape_matches(unbatched_combined_evals, unbatched_expected_evals_shape)
    testing.assert_allclose(unbatched_combined_evals, unbatched_combined_pop.sum(dim=-1), atol=tolerance)

    # test the batched case
    expected_pop_shape = (batch_size, popsizeA + popsizeB, solution_length)
    expected_evals_shape = (
        batch_size,
        popsizeA + popsizeB,
    )

    combined_pop, combined_evals = func_ops.combine((populationA, evalsA), (populationB, evalsB))
    testing.assert_shape_matches(combined_pop, expected_pop_shape)
    testing.assert_shape_matches(combined_evals, expected_evals_shape)

    for i_batch in range(batch_size):
        testing.assert_allclose(combined_evals[i_batch], combined_pop[i_batch].sum(dim=-1), atol=tolerance)


def test_combine_multi_obj():
    tolerance = 1e-8
    solution_length = 5
    batch_size = 3

    objective_sense = ["min", "min"]
    num_objectives = len(objective_sense)

    @rowwise
    def f(x: torch.Tensor) -> torch.Tensor:
        return torch.cat([torch.sum(x).reshape(1), torch.min(x).reshape(1)])

    popsizeA = 6
    populationA = (
        torch.arange(batch_size * popsizeA * solution_length)
        .reshape(batch_size, popsizeA, solution_length)
        .to(dtype=torch.float32)
    )
    evalsA = f(populationA)

    popsizeB = 3
    populationB = (
        torch.arange(batch_size * popsizeB * solution_length)
        .reshape(batch_size, popsizeB, solution_length)
        .to(dtype=torch.float32)
        * -1
    )
    evalsB = f(populationB)

    # test the unbatched case
    unbatched_combined_pop, unbatched_combined_evals = func_ops.combine(
        (populationA[0], evalsA[0]), (populationB[0], evalsB[0]), objective_sense=objective_sense
    )
    unbatched_expected_pop_shape = (popsizeA + popsizeB, solution_length)
    unbatched_expected_evals_shape = (popsizeA + popsizeB, num_objectives)
    testing.assert_shape_matches(unbatched_combined_pop, unbatched_expected_pop_shape)
    testing.assert_shape_matches(unbatched_combined_evals, unbatched_expected_evals_shape)
    testing.assert_allclose(unbatched_combined_evals, f(unbatched_combined_pop), atol=tolerance)

    # test the batched case
    expected_pop_shape = (batch_size, popsizeA + popsizeB, solution_length)
    expected_evals_shape = (batch_size, popsizeA + popsizeB, num_objectives)

    combined_pop, combined_evals = func_ops.combine(
        (populationA, evalsA), (populationB, evalsB), objective_sense=objective_sense
    )
    testing.assert_shape_matches(combined_pop, expected_pop_shape)
    testing.assert_shape_matches(combined_evals, expected_evals_shape)

    for i_batch in range(batch_size):
        testing.assert_allclose(combined_evals[i_batch], f(combined_pop[i_batch]), atol=tolerance)


def test_combine_with_objects():
    tolerance = 1e-8

    populationA = make_tensor(
        [
            [1, 2],
            [1, 3, 5],
            [10, 20, 60, 40],
        ],
        dtype=object,
    )

    populationB = make_tensor(
        [
            [-1, -2],
            [-10, -20],
        ],
        dtype=object,
    )

    def f_single(x: ObjectArray) -> torch.Tensor:
        return torch.as_tensor(
            [sum(values) for values in x],
            dtype=torch.float32,
        )

    def f_multi(x: ObjectArray) -> torch.Tensor:
        return torch.as_tensor(
            [[sum(values), min(values)] for values in x],
            dtype=torch.float32,
        )

    for f, objective_sense in ((f_single, None), (f_multi, ["min", "min"])):
        evalsA = f(populationA)
        evalsB = f(populationB)

        combined_pop, combined_evals = func_ops.combine(
            (populationA, evalsA), (populationB, evalsB), objective_sense=objective_sense
        )

        assert isinstance(combined_pop, ObjectArray)
        assert isinstance(combined_evals, torch.Tensor)

        for i_solution in range(len(combined_pop)):
            sln = combined_pop[i_solution : i_solution + 1]
            sln_eval = combined_evals[i_solution : i_solution + 1]
            sln_re_eval = f(sln)
            testing.assert_allclose(sln_eval, sln_re_eval, atol=tolerance)


@pytest.mark.parametrize(
    "population,desired_best_one,desired_best_two,obj_sense",
    [
        # --- argument set ---
        (
            # population
            torch.FloatTensor(
                [
                    [1, 2, 3],
                    [100, 200, 300],
                    [10, 20, 30],
                    [-1, -2, -3],
                ]
            ),
            # desired_best_one
            torch.FloatTensor([100, 200, 300]),
            # desired_best_two
            torch.FloatTensor(
                [
                    [100, 200, 300],
                    [10, 20, 30],
                ]
            ),
            # obj_sense
            "max",
        ),
        # --- argument set ---
        (
            # population
            torch.FloatTensor(
                [
                    [1, 2, 3],
                    [100, 200, 300],
                    [10, 20, 30],
                    [-1, -2, -3],
                ]
            ),
            # desired_best_one
            torch.FloatTensor([-1, -2, -3]),
            # desired_best_two
            torch.FloatTensor(
                [
                    [-1, -2, -3],
                    [1, 2, 3],
                ]
            ),
            # obj_sense
            "min",
        ),
        # --- argument set ---
        (
            # population
            torch.FloatTensor(
                [
                    [
                        [1, 2, 3],
                        [100, 200, 300],
                        [-1, -2, -3],
                    ],
                    [
                        [5, 6, 7],
                        [8, 9, 10],
                        [20, 30, 40],
                    ],
                ],
            ),
            # desired_best_one
            torch.FloatTensor(
                [
                    [100, 200, 300],
                    [20, 30, 40],
                ],
            ),
            # desired_best_two
            torch.FloatTensor(
                [
                    [
                        [100, 200, 300],
                        [1, 2, 3],
                    ],
                    [
                        [20, 30, 40],
                        [8, 9, 10],
                    ],
                ],
            ),
            # obj_sense
            "max",
        ),
        # --- argument set ---
        (
            # population
            make_tensor(
                [
                    [1, 2, 3],
                    [100, 200, 300, 400],
                    [-1, -2],
                ],
                dtype=object,
            ),
            # desired_best_one
            [100, 200, 300, 400],
            # desired_best_two
            [
                [100, 200, 300, 400],
                [1, 2, 3],
            ],
            # obj_sense
            "max",
        ),
    ],
)
def test_take_best(population, desired_best_one, desired_best_two, obj_sense):
    tolerance = 1e-8

    got_objects = isinstance(population, ObjectArray)

    if got_objects:

        def f(x: Union[ObjectArray, Sequence]) -> torch.Tensor:
            if isinstance(x, ObjectArray):
                return torch.as_tensor([sum(solution) for solution in x], dtype=torch.float32)
            else:
                return torch.as_tensor(sum(x), dtype=torch.float32)

    else:

        @rowwise
        def f(x: torch.Tensor) -> torch.Tensor:
            return torch.sum(x)

    evals = f(population)
    best_one, best_one_eval = func_ops.take_best(population, evals, objective_sense=obj_sense)
    testing.assert_allclose(best_one_eval, f(best_one), atol=tolerance)
    if got_objects:
        best_one = torch.as_tensor(list(best_one), dtype=torch.float32)
        desired_best_one = torch.as_tensor(desired_best_one, dtype=torch.float32)
    testing.assert_allclose(best_one, desired_best_one, atol=tolerance)

    best_two, best_two_evals = func_ops.take_best(population, evals, 2, objective_sense=obj_sense)
    testing.assert_allclose(best_two_evals, f(best_two), atol=tolerance)
    if got_objects:
        for i in (0, 1):
            row = torch.as_tensor(list(best_two[i]), dtype=torch.float32)
            desired_row = torch.as_tensor(desired_best_two[i], dtype=torch.float32)
            testing.assert_allclose(row, desired_row, atol=tolerance)
    else:
        testing.assert_allclose(best_two, desired_best_two, atol=tolerance)


@pytest.mark.parametrize(
    "population,desired_best_two,obj_sense",
    [
        # --- argument set ---
        (
            # population
            torch.FloatTensor(
                [
                    [1, 2, 3],
                    [100, 200, 300],
                    [-1, -9, 0],
                    [5, 6, 7],
                    [98, 200, 400],
                    [10, 11, 12],
                ]
            ),
            # desired_best_two
            torch.FloatTensor(
                [
                    [100, 200, 300],
                    [98, 200, 400],
                ],
            ),
            # obj_sense
            ["max", "max"],
        ),
        # --- argument set ---
        (
            # population
            torch.FloatTensor(
                [
                    [22, 4, -11],
                    [10, 5, 7],
                    [11, 6, 3],
                    [23, 4, -10],
                    [3, 6, 7],
                ]
            ),
            # desired_best_two
            torch.FloatTensor(
                [
                    [22, 4, -11],
                    [23, 4, -10],
                ],
            ),
            # obj_sense
            ["max", "min"],
        ),
        # --- argument set ---
        (
            # population
            torch.FloatTensor(
                [
                    [
                        [1, 2, 3],
                        [100, 200, 300],
                        [-1, -9, 0],
                        [5, 6, 7],
                        [98, 200, 400],
                        [10, 11, 12],
                    ],
                    [
                        [1, 2, 3],
                        [-100, -200, -300],
                        [-1, -9, 0],
                        [9, 11, 13],
                        [-98, -200, -400],
                        [10, 11, 12],
                    ],
                ]
            ),
            # desired_best_two
            torch.FloatTensor(
                [
                    [
                        [100, 200, 300],
                        [98, 200, 400],
                    ],
                    [
                        [9, 11, 13],
                        [10, 11, 12],
                    ],
                ],
            ),
            # obj_sense
            ["max", "max"],
        ),
        # --- argument set ---
        (
            # population
            make_tensor(
                [
                    [1, 50, 60, 2],
                    [23, 1, 2, 67],
                    [12, 5, 3, 8, 99],
                    [2, 55, 1],
                    [15, 17, 16, 23, 22],
                ],
                dtype=object,
            ),
            # desired_best_two
            make_tensor(
                [
                    [1, 50, 60, 2],
                    [2, 55, 1],
                ],
                dtype=object,
            ),
            # obj_sense
            ["min", "min"],
        ),
    ],
)
def test_take_best_with_multiobj(population, desired_best_two, obj_sense):
    tolerance = 1e-8

    if isinstance(population, ObjectArray):

        def f(solutions: ObjectArray) -> torch.Tensor:
            num_solutions = len(solutions)
            result = torch.empty(num_solutions, 2, dtype=torch.float32)
            for i in range(num_solutions):
                result[i, 0] = solutions[i][0]
                result[i, 1] = solutions[i][-1]
            return result

        def vertical_sum_of_decision_values(solutions: ObjectArray) -> torch.Tensor:
            length = min([len(solution) for solution in solutions])
            result = torch.zeros(length, dtype=torch.float32)
            for j in range(length):
                result[j] = sum([solution[j] for solution in solutions])
            return result

    else:

        @rowwise
        def f(x: torch.Tensor) -> torch.Tensor:
            return torch.hstack([x[0].reshape(1), x[-1].reshape(1)])

        def vertical_sum_of_decision_values(solutions: torch.Tensor) -> torch.Tensor:
            return torch.sum(solutions, dim=-2)

    evals = f(population)
    best_two, best_two_evals = func_ops.take_best(population, evals, 2, objective_sense=obj_sense)
    best_two_total = vertical_sum_of_decision_values(best_two)
    desired_best_two_total = vertical_sum_of_decision_values(desired_best_two)

    testing.assert_allclose(torch.sum(f(best_two), dim=-2), torch.sum(best_two_evals, dim=-2), atol=tolerance)
    testing.assert_allclose(best_two_total, desired_best_two_total, atol=tolerance)


@pytest.mark.parametrize(
    "population,num_tournaments,obj_sense",
    [
        # --- argument set ---
        (
            # population
            torch.arange(200.0).reshape(20, 10),
            # num_tournaments
            6,
            # obj_sense
            "max",
        ),
        # --- argument set ---
        (
            # population
            torch.arange(200.0).reshape(20, 10),
            # num_tournaments
            6,
            # obj_sense
            ["max", "min"],
        ),
        # --- argument set ---
        (
            # population
            torch.arange(200.0).reshape(2, 10, 10),
            # num_tournaments
            6,
            # obj_sense
            "min",
        ),
        # --- argument set ---
        (
            # population
            torch.arange(200.0).reshape(2, 10, 10),
            # num_tournaments
            6,
            # obj_sense
            ["max", "min"],
        ),
        # --- argument set ---
        (
            # population
            make_tensor(
                [
                    [1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12],
                    [13, 14, 15, 16],
                    [-1, 2, 3, 4],
                    [5, -6, 7, 8],
                    [9, 10, -11, 12],
                    [13, -14, 15, -16],
                    [100, 101, 102, 103],
                    [33, 44, 55, 66],
                ],
                dtype=object,
            ),
            # num_tournaments
            6,
            # obj_sense
            "min",
        ),
        # --- argument set ---
        (
            # population
            make_tensor(
                [
                    [1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12],
                    [13, 14, 15, 16],
                    [-1, 2, 3, 4],
                    [5, -6, 7, 8],
                    [9, 10, -11, 12],
                    [13, -14, 15, -16],
                    [100, 101, 102, 103],
                    [33, 44, 55, 66],
                ],
                dtype=object,
            ),
            # num_tournaments
            6,
            # obj_sense
            ["max", "min"],
        ),
    ],
)
def test_tournament(population, num_tournaments, obj_sense):
    got_objects = isinstance(population, ObjectArray)

    if isinstance(obj_sense, str):
        multi_obj = False
        num_objs = None
    else:
        multi_obj = True
        num_objs = len(obj_sense)

    if multi_obj:
        if got_objects:

            def f(x: ObjectArray) -> torch.Tensor:
                n = len(x)
                result = torch.empty(n, 2, dtype=torch.float32)
                for i, solution in enumerate(x):
                    result[i, 0] = solution[0]
                    result[i, 1] = solution[1]
                return result

        else:

            @rowwise
            def f(x: torch.Tensor) -> torch.Tensor:
                return torch.hstack([x[0].reshape(1), x[1].reshape(1)])

    else:
        if got_objects:

            def f(x: ObjectArray) -> torch.Tensor:
                n = len(x)
                result = torch.empty(n, dtype=torch.float32)
                for i, solution in enumerate(x):
                    result[i] = sum([value**2 for value in solution])
                return result

        else:

            @rowwise
            def f(x: torch.Tensor) -> torch.Tensor:
                return torch.sum(x**2)

    if got_objects:
        solution_length = len(population[0])
        batch_shape = tuple()

        def pop_shape(x: ObjectArray) -> tuple:
            return len(x), len(x[0])

    else:
        solution_length = population.shape[-1]
        batch_shape = population.shape[:-2]

        def pop_shape(x: torch.Tensor) -> tuple:
            return x.shape

    evals = f(population)

    solutions = func_ops.tournament(
        population, evals, num_tournaments=num_tournaments, tournament_size=2, objective_sense=obj_sense
    )
    assert pop_shape(solutions) == tuple([*batch_shape, num_tournaments, solution_length])

    solutions, sln_evals = func_ops.tournament(
        population,
        evals,
        num_tournaments=num_tournaments,
        tournament_size=2,
        objective_sense=obj_sense,
        with_evals=True,
    )
    assert pop_shape(solutions) == tuple([*batch_shape, num_tournaments, solution_length])
    if multi_obj:
        assert sln_evals.shape == tuple([*batch_shape, num_tournaments, num_objs])
    else:
        assert sln_evals.shape == tuple([*batch_shape, num_tournaments])

    parents1, parents2 = func_ops.tournament(
        population,
        evals,
        num_tournaments=num_tournaments,
        tournament_size=2,
        objective_sense=obj_sense,
        split_results=True,
    )
    assert pop_shape(parents1) == tuple([*batch_shape, num_tournaments // 2, solution_length])
    assert pop_shape(parents2) == tuple([*batch_shape, num_tournaments // 2, solution_length])

    parents1, parent_evals1, parents2, parent_evals2 = func_ops.tournament(
        population,
        evals,
        num_tournaments=num_tournaments,
        tournament_size=2,
        objective_sense=obj_sense,
        split_results=True,
        with_evals=True,
    )
    assert pop_shape(parents1) == tuple([*batch_shape, num_tournaments // 2, solution_length])
    assert pop_shape(parents2) == tuple([*batch_shape, num_tournaments // 2, solution_length])
    if multi_obj:
        assert parent_evals1.shape == tuple([*batch_shape, num_tournaments // 2, num_objs])
        assert parent_evals2.shape == tuple([*batch_shape, num_tournaments // 2, num_objs])
    else:
        assert parent_evals1.shape == tuple([*batch_shape, num_tournaments // 2])
        assert parent_evals2.shape == tuple([*batch_shape, num_tournaments // 2])

    indices = func_ops.tournament(
        population,
        evals,
        num_tournaments=num_tournaments,
        tournament_size=2,
        objective_sense=obj_sense,
        return_indices=True,
    )
    assert indices.shape == tuple([*batch_shape, num_tournaments])

    indices1, indices2 = func_ops.tournament(
        population,
        evals,
        num_tournaments=num_tournaments,
        tournament_size=2,
        objective_sense=obj_sense,
        return_indices=True,
        split_results=True,
    )
    assert indices1.shape == tuple([*batch_shape, num_tournaments // 2])
    assert indices2.shape == tuple([*batch_shape, num_tournaments // 2])


@pytest.mark.parametrize(
    "input_shape,num_children,desired_output_shape",
    [
        ((20, 30), 8, (8, 30)),
        ((7, 20, 30), 8, (7, 8, 30)),
        ((5, 7, 20, 30), 8, (5, 7, 8, 30)),
        ((20, 30), None, (20, 30)),
        ((7, 20, 30), None, (7, 20, 30)),
        ((5, 7, 20, 30), None, (5, 7, 20, 30)),
    ],
)
def test_cross_over(input_shape, num_children, desired_output_shape):
    population = torch.randn(input_shape)

    @rowwise
    def single_objective_f(x: torch.Tensor) -> torch.Tensor:
        return torch.sum(x**2)

    @rowwise
    def multi_objective_f(x: torch.Tensor) -> torch.Tensor:
        return torch.cat([torch.sum(x**2).reshape(1), torch.sum(x).reshape(1)])

    single_objective_evals = single_objective_f(population)
    multi_objective_evals = multi_objective_f(population)

    for objective_sense, evals in [("min", single_objective_evals), (["min", "min"], multi_objective_evals)]:
        cross_over_functions = [
            ("multi_point_cross_over", {"num_points": 1}),
            ("one_point_cross_over", {}),
            ("multi_point_cross_over", {"num_points": 2}),
            ("two_point_cross_over", {}),
            ("multi_point_cross_over", {"num_points": 3}),
            ("simulated_binary_cross_over", {"eta": 10.0}),
            ("simulated_binary_cross_over", {"eta": 20.0}),
        ]

        for tournament_size in (2, 3, 4):
            for cross_over_fn, cross_over_cfg in cross_over_functions:
                output = getattr(func_ops, cross_over_fn)(
                    population,
                    evals,
                    tournament_size=tournament_size,
                    num_children=num_children,
                    objective_sense=objective_sense,
                    **cross_over_cfg,
                )
                assert output.shape == desired_output_shape
