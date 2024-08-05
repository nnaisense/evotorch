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


"""
Functional implementations of the genetic algorithm operators.

Instead of the object-oriented genetic algorithm API
([GeneticAlgorithm][evotorch.algorithms.ga.GeneticAlgorithm]), one might
wish to adopt a style that is more compatible with the functional programming
paradigm. For such cases, the functional operators within this namespace can
be used.

The operators within this namespace are designed to be called directly,
allowing one to implement a genetic algorithm according to which, how, and
when these operators are used.

**Reasons for using the functional operators.**

- **Flexibility.** This API provides various genetic-algorithm-related
    operators and gets out of the picture. The user has the complete control
    over what happens between this operator calls, and in what order these
    operators are used.
- **Batched search.** These functional operators are designed in such a way
    that, if they receive a batched population instead of a single population
    (i.e. if they receive a 3-or-more-dimensional tensor instead of a
    2-dimensional tensor), they will broadcast their operations across the
    extra leftmost dimensions. This allows one to implement a genetic
    algorithm that works across many populations at once, in a vectorized
    manner.
- **Nested optimization.** It could be the case that the optimization problem
    at hand has an inner optimization problem within its fitness function.
    This inner optimization problem could be tackled with the help of a
    genetic algorithm built upon these functional operators.
    Such an approach would allow the user to run a search for each inner
    optimization problem across the entire population of the outer
    problem, in a vectorized manner (see the previous point titled
    "batched search").

**Example usage.**
Let us assume that we have the following cost function that to be minimized:

```python
import torch


def f(x: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(x - 1, dim=-1)
```

A genetic algorithm could be designed with the help of these functional
operators as follows:

```python
import torch
from evotorch.operators.functional import two_point_cross_over, combine, take_best


def f(x: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(x - 1, dim=-1)


popsize = 100  # population size
solution_length = 20  # length of a solution
num_generations = 200  # number of generations
mutation_stdev = 0.01  # standard deviation for mutation
tournament_size = 4  # size for the tournament selection

# Randomly initialize a population, and compute the solution costs
population = torch.randn(popsize, solution_length)
costs = f(population)

# Initialize the variables that will store the decision values and the cost
# for the last population's best solution.
pop_best_values = None
pop_best_cost = None

# main loop of the optimization
for generation in range(1, 1 + num_generations):
    # Given the population and the solution costs, pick parents and apply
    # cross-over on them.
    candidates = two_point_cross_over(
        population,
        costs,
        tournament_size=tournament_size,
        objective_sense="min",
    )

    # Apply Gaussian mutation on the candidates
    candidates = candidates + (torch.randn(popsize, solution_length) * mutation_stdev)

    # Compute the solution costs of the candidates
    candidate_costs = f(candidates)

    # Combine the parents and the candidates into an extended population
    extended_values, extended_costs = combine(
        (population, costs),
        (candidates, candidate_costs),
    )

    # Take the best `popsize` number of solutions from the extended population
    population, costs = take_best(
        extended_values,
        extended_costs,
        popsize,
        objective_sense="min",
    )

    # Take the best solution and its cost
    pop_best_values, pop_best_cost = take_best(population, costs, objective_sense="min")

    # Print the status
    print("Generation:", generation, "  Best cost within population:", best_cost)

# Print the result
print()
print("Best solution of the last population:")
print(pop_best_values)
print("Cost of the best solution of the last population:")
print(pop_best_cost)
```
"""


from typing import Iterable, Optional, Union

import torch

from evotorch.decorators import expects_ndim


def _index_comparison_matrices(n: int, *, device: Union[str, torch.dtype]) -> tuple:
    """
    Return index tensors that are meant for pairwise comparisons.

    In more details, suppose that the argument `n` is given as 4.
    What is returned by this function is a 3-element tuple of the form
    `(indices_matrix1, indices_matrix2, index_row)`. In this returned
    tuple, `indices_matrix1` is:

    ```
    0 0 0 0
    1 1 1 1
    2 2 2 2
    3 3 3 3
    ```

    `indices_matrix2` is:

    ```
    0 1 2 3
    0 1 2 3
    0 1 2 3
    0 1 2 3
    ```

    `index_row` is:

    ```
    0 1 2 3
    ```

    Note: `indices_matrix1` and `indices_matrix2` are expanded views to the
    tensor `index_row`. Do not mutate any of these returned tensors, because
    such mutations might probably reflect on all of them in unexpected ways.

    Args:
        n: Size for the index row and matrices
        device: The device in which the index tensors will be generated
    Returns:
        A tuple of the form `(indices_matrix1, indices_matrix2, index_row)`
        where each item is a PyTorch tensor.
    """
    increasing_indices = torch.arange(n, device=device)
    indices1 = increasing_indices.reshape(n, 1).expand(n, n)
    indices2 = increasing_indices.reshape(1, n).expand(n, n)
    return indices1, indices2, increasing_indices


@expects_ndim(1, 1, None)
def _dominates(
    evals1: torch.Tensor,
    evals2: torch.Tensor,
    objective_sense: list,
) -> torch.Tensor:
    [num_objs] = evals1.shape
    [n2] = evals2.shape
    if num_objs != n2:
        raise ValueError("The lengths of the evaluation results vectors do not match.")
    if num_objs != len(objective_sense):
        raise ValueError("The lengths of the evaluation results vectors do not match the number of objectives")

    # For easier internal representation, we generate a sign adjustment tensor.
    # The motivation is to be able to multiply the evaluation tensors with this adjustment tensor,
    # resulting in new evaluation tensors that guarantee that better results are higher values.
    dtype = evals1.dtype
    device = evals1.device
    sign_adjustment = torch.empty(num_objs, dtype=dtype, device=device)
    for i_obj, obj in enumerate(objective_sense):
        if obj == "min":
            sign_adjustment[i_obj] = -1
        elif obj == "max":
            sign_adjustment[i_obj] = 1
        else:
            raise ValueError(
                "`objective_sense` was expected as a list that consists only of the strings 'min' or 'max'."
                f" However, one of the items encountered within `objective_sense` is: {repr(obj)}."
            )

    # Adjust the signs of the evaluation tensors
    evals1 = sign_adjustment * evals1
    evals2 = sign_adjustment * evals2

    # Count the number of victories for each solution
    num_victories_of_first = (evals1 > evals2).to(dtype=torch.int64).sum()
    num_victories_of_second = (evals2 > evals1).to(dtype=torch.int64).sum()

    # If the first solution has won at least 1 time, and the second solution never won, we can say that the
    # first solution pareto-dominates the second one.
    return (num_victories_of_first >= 1) & (num_victories_of_second == 0)


def dominates(
    evals1: torch.Tensor,
    evals2: torch.Tensor,
    *,
    objective_sense: list,
) -> torch.Tensor:
    """
    Return whether or not the first solution pareto-dominates the second one.

    Args:
        evals1: Evaluation results of the first solution. Expected as an
            at-least-1-dimensional tensor, the length of which must be
            equal to the number of objectives. Extra leftmost dimensions
            will be considered as batch dimensions.
        evals2: Evaluation results of the second solution. Expected as an
            at-least-1-dimensional tensor, the length of which must be
            equal to the number of objectives. Extra leftmost dimensions
            will be considered as batch dimensions.
        objective_sense: Expected as a list of strings, where each
            string is either 'min' or 'max', expressing the direction of
            optimization regarding each objective.
    Returns:
        A tensor of boolean(s), indicating whether or not the first
        solution(s) dominate(s) the second solution(s).
    """
    if isinstance(objective_sense, str):
        raise ValueError(
            "`objective_sense` was received as a string, implying that the problem at hand has a single objective."
            " However, this `dominates(...)` function does not support single-objective cases."
        )
    elif isinstance(objective_sense, Iterable):
        return _dominates(evals1, evals2, objective_sense)
    else:
        raise TypeError(f"Unrecognized `objective_sense`: {repr(objective_sense)}")


@expects_ndim(2, 0, 0, None)
def _domination_check_via_indices(
    population_evals: torch.Tensor,
    solution1_index: torch.Tensor,
    solution2_index: torch.Tensor,
    objective_sense: list,
) -> torch.Tensor:
    evals1 = torch.index_select(population_evals, 0, solution1_index.reshape(1))[0]
    evals2 = torch.index_select(population_evals, 0, solution2_index.reshape(1))[0]
    return _dominates(evals1, evals2, objective_sense)


@expects_ndim(2, None)
def _domination_matrix(
    evals: torch.Tensor,
    objective_sense: list,
) -> torch.Tensor:
    num_solutions, _ = evals.shape
    indices1, indices2, _ = _index_comparison_matrices(num_solutions, device=evals.device)
    return _domination_check_via_indices(evals, indices2, indices1, objective_sense)


def domination_matrix(evals: torch.Tensor, *, objective_sense: list) -> torch.Tensor:
    """
    Compute and return a pareto-domination matrix.

    In this pareto-domination matrix `P`, the item `P[i,j]` is True if the
    `i`-th solution is dominated by the `j`-th solution.

    Args:
        evals: Evaluation results of the solutions, expected as a tensor
            with at least 2 dimensions. In a 2-dimensional `evals` tensor,
            the item `i,j` represents the evaluation result of the
            `i`-th solution according to the `j`-th objective.
            Extra leftmost dimensions are interpreted as batch dimensions.
        objective_sense: A list of strings, where each string is either
            'min' or 'max', expressing the direction of optimization regarding
            each objective.
    Returns:
        A boolean tensor of size `(n,n)`, where `n` is the number of solutions.
    """
    return _domination_matrix(evals, objective_sense)


@expects_ndim(2, None)
def _domination_counts(evals: torch.Tensor, objective_sense: list) -> torch.Tensor:
    return _domination_matrix(evals, objective_sense).to(dtype=torch.int64).sum(dim=-1)


def domination_counts(evals: torch.Tensor, *, objective_sense: list) -> torch.Tensor:
    """
    Return a tensor expressing how many times each solution gets dominated

    In this returned tensor, the `i`-th item is an integer which specifies how
    many times the `i`-th solution is dominated.

    Args:
        evals: Expected as an at-least-2-dimensional tensor. In such a
            2-dimensional evaluation tensor, the item `i,j` represents the
            evaluation result of the `i`-th solution according to the `j`-th
            objective. Extra leftmost dimensions are interpreted as batch
            dimensions.
        objective_sense: A list of strings, where each string is either
            'min' or 'max', expressing the direction of optimization regarding
            each objective.
    Returns:
        An integer tensor of length `n`, where `n` is the number of solutions.
    """
    return _domination_counts(evals, objective_sense)


@expects_ndim(2, 1, 0, 0)
def _crowding_distance_of_solution_considering_objective(
    population_evals: torch.Tensor,
    domination_counts: torch.Tensor,
    solution_index: torch.Tensor,
    objective_index: torch.Tensor,
) -> torch.Tensor:
    num_solutions, _ = population_evals.shape

    [num_domination_counts] = domination_counts.shape
    if num_domination_counts != num_solutions:
        raise ValueError(
            "The number of solutions stored within `evals` does not match the length of `domination_counts`."
        )

    # Get the evaluation results vector for the considered objective
    eval_vector = torch.index_select(population_evals, -1, objective_index.reshape(1)).reshape(num_solutions)

    # Get the evaluation result and the domination count for the considered solution
    solution_eval = torch.index_select(eval_vector, 0, solution_index.reshape(1))[0]
    solution_domination_count = torch.index_select(domination_counts, 0, solution_index.reshape(1))[0]

    # Prepare the masks `got_lower_eval` and `got_higher_eval`. These masks store True for any solution in the
    # same pareto-front with lower evaluation result, and with higher evaluation result, respectively.
    within_same_front = domination_counts == solution_domination_count
    got_lower_eval = within_same_front & (eval_vector < solution_eval)
    got_higher_eval = within_same_front & (eval_vector > solution_eval)

    # Compute a large-enough constant that will be the crowding distance for when the considered solution is
    # pareto-extreme
    large_constant = 2 * (eval_vector.max() - eval_vector.min())

    # For each solution within the same pareto-front with lower evaluation result, compute the fitness distance
    distances_from_below = torch.where(got_lower_eval, solution_eval - eval_vector, large_constant)
    # For each solution within the same pareto-front with higher evaluation result, compute the fitness distance
    distances_from_above = torch.where(got_higher_eval, eval_vector - solution_eval, large_constant)

    # Sum of the nearest (min) distance from below and the nearest (min) distance from above is the crowding distance
    # for the considered objective.
    return distances_from_below.min() + distances_from_above.min()


@expects_ndim(2, 1, 0)
def _crowding_distance_of_solution(
    population_evals: torch.Tensor,
    domination_counts: torch.Tensor,
    solution_index: torch.Tensor,
) -> torch.Tensor:
    _, num_objectives = population_evals.shape
    objective_indices = torch.arange(num_objectives, dtype=torch.int64, device=population_evals.device)

    # Compute the crowding distances for all objectives, then sum those distances, then return the result.
    return _crowding_distance_of_solution_considering_objective(
        population_evals, domination_counts, solution_index, objective_indices
    ).sum()


@expects_ndim(2, 1)
def _crowding_distances(population_evals: torch.Tensor, domination_counts: torch.Tensor) -> torch.Tensor:
    num_solutions, _ = population_evals.shape
    all_solution_indices = torch.arange(num_solutions, dtype=torch.int64, device=population_evals.device)
    return _crowding_distance_of_solution(population_evals, domination_counts, all_solution_indices)


@expects_ndim(2, None, None)
def _pareto_utility(evals: torch.Tensor, objective_sense: list, crowdsort: bool) -> torch.Tensor:
    num_solutions, _ = evals.shape
    domination_counts = _domination_counts(evals, objective_sense)

    # Compute utility values such that a solution that has less domination counts (i.e. a solution that has been
    # dominated less) will have a higher utility value.
    result = torch.as_tensor(num_solutions - domination_counts, dtype=evals.dtype)

    if crowdsort:
        # Compute the crowding distances
        distances = _crowding_distances(evals, domination_counts)
        # Rescale the crowding distances so that they are between 0 and 0.99
        min_distance = distances.min()
        max_distance = distances.max()
        distance_range = (max_distance - min_distance) + 1e-8
        rescaled_distances = 0.99 * ((distances - min_distance) / distance_range)
        # Add the rescaled distances to the resulting utility values
        result = result + rescaled_distances

    return result


def pareto_utility(evals: torch.Tensor, *, objective_sense: list, crowdsort: bool = True) -> torch.Tensor:
    """
    Compute utility values for the solutions of a multi-objective problem.

    A solution on a better pareto-front is assigned a higher utility value.
    Additionally, if `crowdsort` is given as True crowding distances will also
    be taken into account. In more details, in the same pareto-front,
    solutions with higher crowding distances will have increased utility
    values.

    Args:
        evals: Evaluation results, expected as a tensor with at least two
            dimensions. A 2-dimensional `evals` tensor is expected to be
            shaped as (numberOfSolutions, numberOfObjectives). Extra
            leftmost dimensions will be interpreted as batch dimensions.
        objective_sense: Expected as a list of strings, where each string
            is either 'min' or 'max'. The i-th item within this list
            represents the direction of the optimization for the i-th
            objective.
    Returns:
        A utility tensor. Considering the non-batched case (i.e. considering
        that `evals` was given as a 2-dimensional tensor), the i-th item
        within the returned utility tensor represents the utility value
        assigned to the i-th solution.
    """
    return _pareto_utility(evals, objective_sense, crowdsort)


@expects_ndim(2, 1, 1, None, randomness="different")
def _pick_solution_via_tournament(
    solutions: torch.Tensor,
    evals: torch.Tensor,
    indices: torch.Tensor,
    objective_sense: str,
) -> tuple:
    """
    Run a single tournament among multiple solutions to pick the best.

    Args:
        solutions: Decision values of the solutions, as a tensor of at least
            2 dimensions. Extra leftmost dimensions will be considered as
            batch dimensions.
        evals: Evaluation results (i.e. fitnesses) of the solutions, as a
            tensor with at least 1 dimension. Extra leftmost dimensions will
            be considered as batch dimensions.
        indices: Indices of solutions that participate into the tournament,
            as a tensor of integers with at least 1 dimension. Extra leftmost
            dimensions will be considered as batch dimensions.
        objective_sense: A string with value 'min' or 'max', representing the
            goal of the optimization.
    Returns:
        A tuple of the form `(decision_values, eval_result)` where
        `decision_values` is the tensor that contains the decision values
        of the winning solution(s), and `eval_result` is a tensor that
        contains the evaluation result(s) (i.e. fitness(es)) of the
        winning solution(s).
    """
    # Get the evaluation results of the solutions that participate into the tournament
    competing_evals = torch.index_select(evals, 0, indices)

    if objective_sense == "max":
        # If the objective sense is 'max', we are looking for the solution with the highest evaluation result
        argbest = torch.argmax
    elif objective_sense == "min":
        # If the objective sense is 'min', we are looking for the solution with the lowest evaluation result
        argbest = torch.argmin
    else:
        raise ValueError(
            "`objective_sense` was expected either as 'min' or as 'max'."
            f" However, it was received as {repr(objective_sense)}."
        )

    # Among the competing solutions, which one is the best?
    winner_competing_eval_index = argbest(competing_evals)

    # Get the index (within the original `solutions`) of the winning solution
    winner_solution_index = torch.index_select(indices, 0, winner_competing_eval_index.reshape(1))

    # Get the decision values and the evaluation result of the winning solution
    winner_solution = torch.squeeze(torch.index_select(solutions, 0, winner_solution_index), dim=0)
    winner_eval = torch.squeeze(torch.index_select(evals, 0, winner_solution_index), dim=0)

    # Return the winning solution's decision values and evaluation results
    return winner_solution, winner_eval


@expects_ndim(2, 1, None, None, None, randomness="different")
def _single_objective_tournament(
    solutions: torch.Tensor,
    evals: torch.Tensor,
    num_tournaments: int,
    tournament_size: int,
    objective_sense: str,
) -> tuple:
    if tournament_size < 1:
        raise ValueError(
            "The argument `tournament_size` was expected to be greater than or equal to 1."
            f" However, it was encountered as {tournament_size}."
        )
    popsize, _ = solutions.shape
    indices_for_tournament = torch.randint_like(
        solutions[:1, :1].expand(num_tournaments, tournament_size), 0, popsize, dtype=torch.int64
    )
    return _pick_solution_via_tournament(solutions, evals, indices_for_tournament, objective_sense)


def _tournament(
    solutions: torch.Tensor,
    evals: torch.Tensor,
    num_tournaments: int,
    tournament_size: int,
    objective_sense: Union[str, list],
) -> tuple:
    """
    Randomly pick solutions, put them into a tournament, pick the winners.

    Args:
        solutions: Decision values of the solutions.
        evals: Evaluation results of the solutions.
            In the single-objective case, this is expected as an
            at-least-1-dimensional tensor, the `i`-th item expressing
            the evaluation result of the `i`-th solution.
            In the multi-objective case, this is expected as an
            at-least-2-dimensional tensor, the `(i,j)`-th item
            expressing the evaluation result of the `i`-th solution
            according to the `j`-th objective.
            Extra leftmost dimensions are interpreted as batch dimensions.
        num_tournaments: Number of tournaments that will be applied.
            In other words, number of winners that will be picked.
        tournament_size: Number of solutions to be picked for the tournament
        objective_sense: A string or a list of strings, where (each) string
            has either the value 'min' for minimization or 'max' for
            maximization.
    Returns:
        A tuple of the form `(decision_values, eval_results)` where
        `decision_values` is the tensor that contains the decision values
        of the winning solutions, and `eval_result` is a tensor that
        contains the evaluation results (i.e. fitnesses) of the
        winning solutions.
    """
    if isinstance(objective_sense, str):
        pass  # nothing to do
    elif isinstance(objective_sense, Iterable):
        objective_sense = list(objective_sense)
        evals = pareto_utility(evals, objective_sense=objective_sense, crowdsort=False)
        objective_sense = "max"
    else:
        raise TypeError(
            "The argument `objective_sense` was expected as a string for the single-objective case,"
            " or as a list of strings for the multi-objective case."
            f" However, the encountered `objective_sense` is {repr(objective_sense)}."
        )
    return _single_objective_tournament(solutions, evals, num_tournaments, tournament_size, objective_sense)


@expects_ndim(2, randomness="different")
def _pair_solutions_for_cross_over(solutions: torch.Tensor) -> tuple:
    """
    Split the solutions to make its 1st and 2nd halves the 1st and 2nd parents.

    Args:
        solutions: A tensor of decision values that are subject to pairing.
            Must be at least 2-dimensional. Extra leftmost dimensions will
            be interpreted as batch dimensions.
        num_children: Number of children, as an integer. Assuming that each
            cross-over operation will generate two children, the number of
            pairs to be picked from within `solutions` will be the half
            of `num_children`.
    Returns:
        A tuple of the form `(parents1, parents2)` where both parent items
        are (at least) 2-dimensional tensors. In the non-batched case, this
        resulting tuple indicates that `parents1[i, :]` is paired with
        `parents2[i, :]`.
    """
    popsize, _ = solutions.shape

    # Ensure that the number of solutions is divisible by 2.
    if (popsize % 2) != 0:
        raise ValueError(f"The number of `solutions` was expected as an even number. However, it is {popsize}.")

    # Compute the number of pairs to be generated as the half of `num_children`.
    num_pairings = popsize // 2

    return solutions[:num_pairings, :], solutions[num_pairings:, :]


@expects_ndim(1, 1, None, randomness="different")
def _do_cross_over_between_two_solutions(solution1: torch.Tensor, solution2: torch.Tensor, num_points: int) -> tuple:
    """
    Do cross-over between two solutions (or between batches of solutions).

    Args:
        solution1: A tensor, with at least 1 dimension, representing the
            decision values of the first parent(s).
        solution1: A tensor, with at least 1 dimension, representing the
            decision values of the second parent(s).
        num_points: Number of cutting points for the cross-over operation.
    Returns:
        A tuple of the form `(child1, child2)`, representing the decision
        values of the generated child solution(s).
    """
    device = solution1.device
    [solution_length] = solution1.shape

    # Randomly generate the tensor `cut_points` that represents the indices at which the decision values of the
    # parent solutions will be cut.
    like_what = (solution1[:1] + solution2[:1]).reshape(tuple()).expand(num_points)
    cut_points = torch.randint_like(like_what, 1, solution_length - 1, dtype=torch.int64)

    item_indices = torch.arange(solution_length, dtype=torch.int64, device=device)

    # Initialize the tensor `switch_parent` as a tensor filled with False.
    switch_parent = torch.zeros(solution_length, dtype=torch.bool, device=device)

    # For each cutting point, flip the booleans within `switch_parent` whose indices are greater than or equal to
    # the encountered cutting point.
    for i_num_point in range(num_points):
        cut_point_index = torch.as_tensor([i_num_point], dtype=torch.int64, device=device)
        cut_point = torch.index_select(cut_points, 0, cut_point_index).reshape(tuple())
        switch_parent = (item_indices >= cut_point) ^ switch_parent

    dont_switch_parent = ~switch_parent

    # If `switch_parent` is False, child1 takes its value from solution1.
    # If `switch_parent` is True, child1 takes its value from solution2.
    child1 = (dont_switch_parent * solution1) + (switch_parent * solution2)

    # If `switch_parent` is False, child2 takes its value from solution2.
    # If `switch_parent` is True, child2 takes its value from solution1.
    child2 = (dont_switch_parent * solution2) + (switch_parent * solution1)

    # Return the generated child solutions
    return child1, child2


@expects_ndim(2, None, None, randomness="different")
def _do_cross_over(solutions: torch.Tensor, num_points: int) -> torch.Tensor:
    """
    Apply cross-over on multiple solutions.

    Args:
        solutions: Decision values of the parent solutions, as a tensor with
            at least 2 dimensions. Extra leftmost dimensions will be considered
            batch dimensions.
        num_points: Number of cutting points for when applying cross-over.
    Returns:
        A tensor with at least 2 dimensions, representing the decision values
        of the child solutions.
    """
    parents1, parents2 = _pair_solutions_for_cross_over(solutions)
    children1, children2 = _do_cross_over_between_two_solutions(parents1, parents2, num_points)
    return torch.vstack([children1, children2])


def multi_point_cross_over(
    parents: torch.Tensor,
    evals: Optional[torch.Tensor] = None,
    *,
    num_points: int,
    tournament_size: Optional[int] = None,
    num_children: Optional[int] = None,
    objective_sense: Optional[Union[str, list]] = None,
) -> torch.Tensor:
    """
    Apply multi-point cross-over on the given `parents`.

    If `tournament_size` is given, parents for the cross-over operation will
    be picked with the help of a tournament. Otherwise, the first half of the
    given `parents` will be the first set of parents, and the second half
    of the given `parents` will be the second set of parents.

    The return value of this function is a new tensor containing the decision
    values of the child solutions.

    Args:
        parents: A tensor with at least 2 dimensions, representing the decision
            values of the parent solutions. If this tensor has more than 2
            dimensions, the extra leftmost dimension(s) will be considered as
            batch dimensions.
        evals: A tensor with at least 1 dimension, representing the evaluation
            results (i.e. fitnesses) of the parent solutions. If this tensor
            has more than 1 dimension, the extra leftmost dimension(s) will be
            considered as batch dimensions. If `tournament_size` is not given,
            `evals` can be left as None.
        num_points: Number of points at which the decision values of the
            parent solutions will be cut and recombined to form the child
            solutions.
        tournament_size: If given as an integer that is greater than or equal
            to 1, the parents for the cross-over operation will be picked
            with the help of a tournament. In more details, each parent will
            be picked as the result of comparing multiple competing solutions,
            the number of these competing solutions being equal to this
            `tournament_size`. Please note that, if `tournament_size` is given
            as an integer, the arguments `evals` and `objective_sense` are
            also required. If `tournament_size` is left as None, the first half
            of `parents` will be the first set of parents, and the second half
            of `parents` will be the second set of parents.
        num_children: Optionally the number of children to produce as the
            result of tournament selection and cross-over, as an even integer.
            If tournament selection is enabled (i.e. if `tournament_size` is
            an integer) but `num_children` is omitted, the number of children
            will be equal to the number of `parents`.
            If there is no tournament selection (i.e. if `tournament_size` is
            None), `num_children` is expected to be None.
        objective_sense: Mandatory if `tournament_size` is not None.
            For when there is only one objective, `objective_sense` is
            expected as 'max' for when the goal of the optimization is to
            maximize `evals`, 'min' for when the goal of the optimization is
            to minimize `evals`. For when there are multiple objectives,
            `objective_sense` is expected as a list of strings, where each
            string is either 'min' or 'max'.
            If `tournament_size` is None, `objective_sense` can also be left
            as None.
    Returns:
        Decision values of the child solutions, as a new tensor.
    """

    if tournament_size is None:
        if num_children is not None:
            raise ValueError(
                "`num_children` was received as something other than None."
                " However, `num_children` is expected only when a `tournament_size` is given,"
                " which seems to be omitted (i.e. which is None)."
            )
    else:
        # This is the case where the tournament selection feature is enabled.
        # We first ensure that the required arguments `evals` and `objective_sense` are available.
        if evals is None:
            raise ValueError(
                "When a `tournament_size` is given, the argument `evals` is also required."
                " However, it was received as None."
            )
        if num_children is None:
            # If `num_children` is not given, we make it equal to the number of `parents`.
            num_children = parents.shape[-2]
        if objective_sense is None:
            raise ValueError(
                "When a `tournament_size` is given, the argument `objective_sense` is also required."
                " However, it was received as None."
            )

        # Apply tournament selection on the original `parents`
        parents, _ = _tournament(parents, evals, num_children, tournament_size, objective_sense)

    # Apply the cross-over operation on `parents`, and return the recombined decision values tensor.
    return _do_cross_over(parents, num_points)


def one_point_cross_over(
    parents: torch.Tensor,
    evals: Optional[torch.Tensor] = None,
    *,
    tournament_size: Optional[int] = None,
    num_children: Optional[int] = None,
    objective_sense: Optional[str] = None,
) -> torch.Tensor:
    """
    Apply one-point cross-over on the given `parents`.

    Let us assume that we have the following two parent solutions:

    ```text
             ________________________
    parentA | a1  a2  a3  a4  a5  a6 |
    parentB | b1  b2  b3  b4  b5  b6 |
            |________________________|
    ```

    This cross-over operation will first randomly decide a cutting point:

    ```text
             ________|________________
    parentA | a1  a2 | a3  a4  a5  a6 |
    parentB | b1  b2 | b3  b4  b5  b6 |
            |________|________________|
                     |
    ```

    ...and then form the following child solutions by recombining the decision
    values of the parents:

    ```text
             ________|________________
    child1  | a1  a2 | b3  b4  b5  b6 |
    child2  | b1  b2 | a3  a4  a5  a6 |
            |________|________________|
                     |
    ```

    If `tournament_size` is given, parents for the cross-over operation will
    be picked with the help of a tournament. Otherwise, the first half of the
    given `parents` will be the first set of parents, and the second half
    of the given `parents` will be the second set of parents.

    The return value of this function is a new tensor containing the decision
    values of the child solutions.

    Args:
        parents: A tensor with at least 2 dimensions, representing the decision
            values of the parent solutions. If this tensor has more than 2
            dimensions, the extra leftmost dimension(s) will be considered as
            batch dimensions.
        evals: A tensor with at least 1 dimension, representing the evaluation
            results (i.e. fitnesses) of the parent solutions. If this tensor
            has more than 1 dimension, the extra leftmost dimension(s) will be
            considered as batch dimensions. If `tournament_size` is not given,
            `evals` can be left as None.
        tournament_size: If given as an integer that is greater than or equal
            to 1, the parents for the cross-over operation will be picked
            with the help of a tournament. In more details, each parent will
            be picked as the result of comparing multiple competing solutions,
            the number of these competing solutions being equal to this
            `tournament_size`. Please note that, if `tournament_size` is given
            as an integer, the arguments `evals` and `objective_sense` are
            also required. If `tournament_size` is left as None, the first half
            of `parents` will be the first set of parents, and the second half
            of `parents` will be the second set of parents.
        num_children: Optionally the number of children to produce as the
            result of tournament selection and cross-over, as an even integer.
            If tournament selection is enabled (i.e. if `tournament_size` is
            an integer) but `num_children` is omitted, the number of children
            will be equal to the number of `parents`.
            If there is no tournament selection (i.e. if `tournament_size` is
            None), `num_children` is expected to be None.
        objective_sense: Mandatory if `tournament_size` is not None.
            For when there is only one objective, `objective_sense` is
            expected as 'max' for when the goal of the optimization is to
            maximize `evals`, 'min' for when the goal of the optimization is
            to minimize `evals`. For when there are multiple objectives,
            `objective_sense` is expected as a list of strings, where each
            string is either 'min' or 'max'.
            If `tournament_size` is None, `objective_sense` can also be left
            as None.
    Returns:
        Decision values of the child solutions, as a new tensor.
    """
    return multi_point_cross_over(
        parents,
        evals,
        num_points=1,
        num_children=num_children,
        tournament_size=tournament_size,
        objective_sense=objective_sense,
    )


def two_point_cross_over(
    parents: torch.Tensor,
    evals: Optional[torch.Tensor] = None,
    *,
    tournament_size: Optional[int] = None,
    num_children: Optional[int] = None,
    objective_sense: Optional[str] = None,
) -> torch.Tensor:
    """
    Apply two-point cross-over on the given `parents`.

    Let us assume that we have the following two parent solutions:

    ```text
             ________________________
    parentA | a1  a2  a3  a4  a5  a6 |
    parentB | b1  b2  b3  b4  b5  b6 |
            |________________________|
    ```

    This cross-over operation will first randomly decide two cutting points:

    ```text
             ________|____________|____
    parentA | a1  a2 | a3  a4  a5 | a6 |
    parentB | b1  b2 | b3  b4  b5 | b6 |
            |________|____________|____|
                     |            |
    ```

    ...and then form the following child solutions by recombining the decision
    values of the parents:

    ```text
             ________|____________|____
    child1  | a1  a2 | b3  b4  b5 | a6 |
    child2  | b1  b2 | a3  a4  a5 | b6 |
            |________|____________|____|
                     |            |
    ```

    If `tournament_size` is given, parents for the cross-over operation will
    be picked with the help of a tournament. Otherwise, the first half of the
    given `parents` will be the first set of parents, and the second half
    of the given `parents` will be the second set of parents.

    The return value of this function is a new tensor containing the decision
    values of the child solutions.

    Args:
        parents: A tensor with at least 2 dimensions, representing the decision
            values of the parent solutions. If this tensor has more than 2
            dimensions, the extra leftmost dimension(s) will be considered as
            batch dimensions.
        evals: A tensor with at least 1 dimension, representing the evaluation
            results (i.e. fitnesses) of the parent solutions. If this tensor
            has more than 1 dimension, the extra leftmost dimension(s) will be
            considered as batch dimensions. If `tournament_size` is not given,
            `evals` can be left as None.
        tournament_size: If given as an integer that is greater than or equal
            to 1, the parents for the cross-over operation will be picked
            with the help of a tournament. In more details, each parent will
            be picked as the result of comparing multiple competing solutions,
            the number of these competing solutions being equal to this
            `tournament_size`. Please note that, if `tournament_size` is given
            as an integer, the arguments `evals` and `objective_sense` are
            also required. If `tournament_size` is left as None, the first half
            of `parents` will be the first set of parents, and the second half
            of `parents` will be the second set of parents.
        num_children: Optionally the number of children to produce as the
            result of tournament selection and cross-over, as an even integer.
            If tournament selection is enabled (i.e. if `tournament_size` is
            an integer) but `num_children` is omitted, the number of children
            will be equal to the number of `parents`.
            If there is no tournament selection (i.e. if `tournament_size` is
            None), `num_children` is expected to be None.
        objective_sense: Mandatory if `tournament_size` is not None.
            For when there is only one objective, `objective_sense` is
            expected as 'max' for when the goal of the optimization is to
            maximize `evals`, 'min' for when the goal of the optimization is
            to minimize `evals`. For when there are multiple objectives,
            `objective_sense` is expected as a list of strings, where each
            string is either 'min' or 'max'.
            If `tournament_size` is None, `objective_sense` can also be left
            as None.
    Returns:
        Decision values of the child solutions, as a new tensor.
    """
    return multi_point_cross_over(
        parents,
        evals,
        num_points=2,
        num_children=num_children,
        tournament_size=tournament_size,
        objective_sense=objective_sense,
    )


@expects_ndim(1, 1, 0, randomness="different")
def _do_sbx_between_two_solutions(parent1: torch.Tensor, parent2: torch.Tensor, eta: torch.Tensor) -> tuple:
    u = torch.rand_like(parent1)

    beta = torch.where(
        u <= 0.5,
        (2 * u) ** (1.0 / (eta + 1.0)),
        (1 / (2 * (1.0 - u))) ** (1.0 / (eta + 1.0)),
    )

    child1 = 0.5 * (((1 + beta) * parent1) + ((1 - beta) * parent2))
    child2 = 0.5 * (((1 - beta) * parent1) + ((1 + beta) * parent2))

    return child1, child2


@expects_ndim(2, 0, randomness="different")
def _do_sbx(solutions: torch.Tensor, eta: Union[float, torch.Tensor]) -> torch.Tensor:
    parents1, parents2 = _pair_solutions_for_cross_over(solutions)
    children1, children2 = _do_sbx_between_two_solutions(parents1, parents2, eta)
    return torch.vstack([children1, children2])


def simulated_binary_cross_over(
    parents: torch.Tensor,
    evals: Optional[torch.Tensor] = None,
    *,
    eta: Union[float, torch.Tensor],
    tournament_size: Optional[int] = None,
    num_children: Optional[int] = None,
    objective_sense: Optional[str] = None,
) -> torch.Tensor:
    """
    Apply simulated binary cross-over (SBX) on the given `parents`.

    If `tournament_size` is given, parents for the cross-over operation will
    be picked with the help of a tournament. Otherwise, the first half of the
    given `parents` will be the first set of parents, and the second half
    of the given `parents` will be the second set of parents.

    The return value of this function is a new tensor containing the decision
    values of the child solutions.

    Args:
        parents: A tensor with at least 2 dimensions, representing the decision
            values of the parent solutions. If this tensor has more than 2
            dimensions, the extra leftmost dimension(s) will be considered as
            batch dimensions.
        evals: A tensor with at least 1 dimension, representing the evaluation
            results (i.e. fitnesses) of the parent solutions. If this tensor
            has more than 1 dimension, the extra leftmost dimension(s) will be
            considered as batch dimensions. If `tournament_size` is not given,
            `evals` can be left as None.
        eta: The crowding index, expected as a real number. Bigger eta values
            result in children closer to their parents. If `eta` is given as
            an `n`-dimensional tensor instead of a scalar, those extra
            dimensions will be considered as batch dimensions.
        tournament_size: If given as an integer that is greater than or equal
            to 1, the parents for the cross-over operation will be picked
            with the help of a tournament. In more details, each parent will
            be picked as the result of comparing multiple competing solutions,
            the number of these competing solutions being equal to this
            `tournament_size`. Please note that, if `tournament_size` is given
            as an integer, the arguments `evals` and `objective_sense` are
            also required. If `tournament_size` is left as None, the first half
            of `parents` will be the first set of parents, and the second half
            of `parents` will be the second set of parents.
        num_children: Optionally the number of children to produce as the
            result of tournament selection and cross-over, as an even integer.
            If tournament selection is enabled (i.e. if `tournament_size` is
            an integer) but `num_children` is omitted, the number of children
            will be equal to the number of `parents`.
            If there is no tournament selection (i.e. if `tournament_size` is
            None), `num_children` is expected to be None.
        objective_sense: Mandatory if `tournament_size` is not None.
            For when there is only one objective, `objective_sense` is
            expected as 'max' for when the goal of the optimization is to
            maximize `evals`, 'min' for when the goal of the optimization is
            to minimize `evals`. For when there are multiple objectives,
            `objective_sense` is expected as a list of strings, where each
            string is either 'min' or 'max'.
            If `tournament_size` is None, `objective_sense` can also be left
            as None.
    Returns:
        Decision values of the child solutions, as a new tensor.
    """
    if tournament_size is None:
        if num_children is not None:
            raise ValueError(
                "`num_children` was received as something other than None."
                " However, `num_children` is expected only when a `tournament_size` is given,"
                " which seems to be omitted (i.e. which is None)."
            )
    else:
        # This is the case where the tournament selection feature is enabled.
        # We first ensure that the required arguments `evals` and `objective_sense` are available.
        if evals is None:
            raise ValueError(
                "When a `tournament_size` is given, the argument `evals` is also required."
                " However, it was received as None."
            )
        if num_children is None:
            # If `num_children` is not given, we make it equal to the number of `parents`.
            num_children = parents.shape[-2]
        if objective_sense is None:
            raise ValueError(
                "When a `tournament_size` is given, the argument `objective_sense` is also required."
                " However, it was received as None."
            )

        # Apply tournament selection on the original `parents`
        parents, _ = _tournament(parents, evals, num_children, tournament_size, objective_sense)

    return _do_sbx(parents, eta)


@expects_ndim(1, None, None)
def _utility(evals: torch.Tensor, objective_sense: str, ranking_method: Optional[str] = "centered") -> torch.Tensor:
    """
    Return utility values representing how good the evaluation results are.

    Args:
        evals: An at least 1-dimensional tensor that stores evaluation results
            (i.e. fitness values). Extra leftmost dimensions will be taken as
            batch dimensions.
        objective_sense: A string whose value is either 'min' or 'max', which
            represents the goal of the optimization (minimization or
            maximization).
        ranking_method: Ranking method according to which the utilities will
            be computed. Currently, this function supports:
            'centered' (worst one gets -0.5, best one gets 0.5);
            'linear' (worst one gets 0.0, best one gets 1.0);
            'raw' (evaluation results themselves are returned, with the
            additional behavior of flipping the signs if `objective_sense`
            is 'min', ensuring that the worst evaluation result gets the
            lowest value, and the best evaluation result gets the highest
            value). None also means 'raw'.
    Returns:
        Utility values, as a tensor whose shape is the same with the shape of
        `evals`.
    """
    if objective_sense == "min":
        # If the objective sense is 'min', we set `descending=True`, so that the order will be inverted, and the
        # highest number in `evals` will end up at index 0 (and therefore with the lowest rank).
        descending = True
    elif objective_sense == "max":
        # If the objective sense is 'max', we set `descending=False`, so that the order of sorting will be from
        # lowest to highest, and therefore, the highest number in `evals` will end up at the highest index
        # (and therefore with the highest rank).
        descending = False
    else:
        raise ValueError(f"Expected `objective_sense` as 'min' or 'max', but received it as {repr(objective_sense)}")

    if (ranking_method is None) or (ranking_method == "raw"):
        # This is the case where `ranking_method` is "raw" (or is None), which means that we do not even need to
        # do sorting. We can just use `evals` itself.
        if descending:
            # If `descending` is True, we are in the case that the objective sense is 'min'.
            # In this case, the highest number within `evals` should have the lowest utility, and the lowest number
            # within `evals` should have the highest utility. To ensure this, we flip the signs and return the result.
            return -evals
        else:
            # If `descending` is False, there is nothing to do. We can just return `evals` as it is.
            return evals

    [n] = evals.shape
    increasing_indices = torch.arange(n, device=evals.device)

    # Compute the ranks, initially in the form of indices (i.e. worst one gets 0, best one gets n-1)
    indices_for_sorting = torch.argsort(evals, descending=descending)
    ranks = torch.empty_like(indices_for_sorting)
    ranks[indices_for_sorting] = increasing_indices

    if ranking_method == "linear":
        # Rescale the ranks so that the worst one gets 0.0, and the best one gets 1.0.
        ranks = ranks / (n - 1)
    elif ranking_method == "centered":
        # Rescale and shift the ranks so that the worst one gets -0.5, and the best one gets +0.5.
        ranks = (ranks / (n - 1)) - 0.5
    else:
        raise ValueError(f"Unrecognized ranking method: {repr(ranking_method)}")

    return ranks


def utility(
    evals: torch.Tensor,
    *,
    objective_sense: str,
    ranking_method: Optional[str] = "centered",
) -> torch.Tensor:
    """
    Return utility values representing how good the evaluation results are.

    A utility number is different from `evals` in the sense that, worst
    solution has the lowest utility, and the best solution has the highest
    utility, regardless of the objective sense ('min' or 'max').
    On the other hand, the lowest number within `evals` could represent
    the fitness of the best solution or of the worst solution, depending
    on the objective sense.

    The "centered" ranking is the same ranking method that was used within:

    ```
    Tim Salimans, Jonathan Ho, Xi Chen, Szymon Sidor, Ilya Sutskever (2017).
    Evolution Strategies as a Scalable Alternative to Reinforcement Learning
    ```

    Args:
        evals: An at least 1-dimensional tensor that stores evaluation results
            (i.e. fitness values). Extra leftmost dimensions will be taken as
            batch dimensions.
        objective_sense: A string whose value is either 'min' or 'max', which
            represents the goal of the optimization (minimization or
            maximization).
        ranking_method: Ranking method according to which the utilities will
            be computed. Currently, this function supports:
            'centered' (worst one gets -0.5, best one gets 0.5);
            'linear' (worst one gets 0.0, best one gets 1.0);
            'raw' (evaluation results themselves are returned, with the
            additional behavior of flipping the signs if `objective_sense`
            is 'min', ensuring that the worst evaluation result gets the
            lowest value, and the best evaluation result gets the highest
            value). None also means 'raw'.
    Returns:
        Utility values, as a tensor whose shape is the same with the shape of
        `evals`.
    """
    if isinstance(objective_sense, str):
        return _utility(evals, objective_sense, ranking_method)
    elif isinstance(objective_sense, Iterable):
        raise ValueError(
            "The argument `objective_sense` was received as an iterable other than string,"
            " implying that the problem at hand has multiple objectives."
            " However, this `utility(...)` function does not support multiple objectives."
            " Consider using `pareto_utility(...)`."
        )
    else:
        raise TypeError(f"Unrecognized `objective_sense`: {repr(objective_sense)}")


@expects_ndim(1, randomness="different")
def _cosyne_permutation_for_entire_subpopulation(subpopulation: torch.Tensor) -> torch.Tensor:
    """
    Return the permuted (i.e. shuffled) version of the given subpopulation.

    In the context of the Cosyne algorithm, a "subpopulation" is the population
    of decision values for a single decision variable. Therefore, subpopulation
    represents a column of an entire population.

    Args:
        subpopulation: Population of decision values for a single decision
            variable. Expected as an at least 1-dimensional tensor. Extra
            leftmost dimensions will be considered as batch dimensions.
    Returns:
        Shuffled version of the given subpopulation.
    """
    return subpopulation[torch.argsort(torch.rand_like(subpopulation))]


@expects_ndim(1, 1, None, randomness="different")
def _partial_cosyne_permutation_for_subpopulation(
    subpopulation: torch.Tensor, evals: torch.Tensor, objective_sense: Optional[str]
) -> torch.Tensor:
    """
    Return the permuted (i.e. shuffled) version of the given subpopulation.

    In the context of the Cosyne algorithm, a "subpopulation" is the population
    of decision values for a single decision variable. Therefore, subpopulation
    represents a column of an entire population.

    Probabilistically, some items within the given subpopulation stay the same.
    In more details, if an item belongs to a solution that has better fitness,
    that item has lower probability to change.

    Args:
        subpopulation: Population of decision values for a single decision
            variable. Expected as an at least 1-dimensional tensor. Extra
            leftmost dimensions will be considered as batch dimensions.
        evals: Evaluation results (i.e. fitnesses).
        objective_sense: A string whose value is either 'min' or 'max',
            representing the goal of the optimization.
    Returns:
        Shuffled version of the given subpopulation.
    """
    permuted = _cosyne_permutation_for_entire_subpopulation(subpopulation)

    [n] = subpopulation.shape
    [num_evals] = evals.shape

    if n != num_evals:
        raise ValueError(f"The population size is {n}, but the number of evaluations is different ({num_evals})")

    ranks = utility(evals, objective_sense=objective_sense, ranking_method="linear")
    permutation_probs = 1 - ranks.pow(1 / float(n))
    to_permute = torch.rand_like(subpopulation) < permutation_probs
    return torch.where(to_permute, permuted, subpopulation)


@expects_ndim(2)
def _cosyne_permutation_for_entire_population(population: torch.Tensor) -> torch.Tensor:
    """
    Return the permuted (i.e. shuffled) version of a population.

    Shuffling of the values is done columnwise.

    Args:
        population: A tensor with at least 2 dimensions, representing the
            decision values of the solutions.
    Returns:
        The shuffled counterpart of the given population, as a new tensor.
    """
    return _cosyne_permutation_for_entire_subpopulation(population.T).T


@expects_ndim(2, 1, None, randomness="different")
def _partial_cosyne_permutation_for_population(
    population: torch.Tensor, evals: torch.Tensor, objective_sense: str
) -> torch.Tensor:
    """
    Return the permuted (i.e. shuffled) version of a population.

    Shuffling of the values is done columnwise. For each column, while doing
    the shuffling, each item is given a probability of staying the same.
    This probability is higher for items that belong to solutions with better
    fitnesses.

    Args:
        population: A tensor with at least 2 dimensions, representing the
            decision values of the solutions. Extra leftmost dimensions will
            be considered as batch dimensions.
        evals: Evaluation results (i.e. fitnesses), as a tensor with at least
            one dimension. Extra leftmost dimensions will be considered as
            batch dimensions.
        objective_sense: A string whose value is either 'min' or 'max',
            representing the goal of the optimization.
    Returns:
        The shuffled counterpart of the given population, as a new tensor.
    """
    return _partial_cosyne_permutation_for_subpopulation(population.T, evals, objective_sense).T


def cosyne_permutation(
    values: torch.Tensor,
    evals: Optional[torch.Tensor] = None,
    *,
    permute_all: bool = True,
    objective_sense: Optional[str] = None,
) -> torch.Tensor:
    """
    Return the permuted (i.e. shuffled) version of the given decision values.

    Shuffling of the decision values is done columnwise.

    If `permute_all` is given as True, each item within each column will be
    subject to permutation. In this mode, the arguments `evals` and
    `objective_sense` can be omitted (i.e. can be left as None).

    If `permute_all` is given as False, each item within each column is given
    a probability of staying the same. This probability is higher for items
    that belong to solutions with better fitnesses. In this mode, the
    arguments `evals` and `objective_sense` are mandatory.

    Reference:

    ```
    Gomez, F., Schmidhuber, J., Miikkulainen, R., & Mitchell, M. (2008).
    Accelerated Neural Evolution through Cooperatively Coevolved Synapses.
    Journal of Machine Learning Research, 9(5).
    ```

    Args:
        population: A tensor with at least 2 dimensions, representing the
            decision values of the solutions. Extra leftmost dimensions will
            be considered as batch dimensions.
        evals: Evaluation results (i.e. fitnesses), as a tensor with at least
            one dimension. Extra leftmost dimensions will be considered as
            batch dimensions. If `permute_all` is True, this argument can be
            left as None.
        permute_all: Whether or not each item within each column will be
            subject to permutation operation. If given as False, items
            with better fitnesses have greater probabilities of staying the
            same. The default is True.
        objective_sense: A string whose value is either 'min' or 'max',
            representing the goal of the optimization. If `permute_all` is
            True, this argument can be left as None.
    Returns:
        The shuffled counterpart of the given population, as a new tensor.
    """
    if permute_all:
        return _cosyne_permutation_for_entire_population(values)
    else:
        if evals is None:
            raise ValueError("When `permute_all` is False, `evals` is required")
        if objective_sense is None:
            raise ValueError("When `permute_all` is False, `objective_sense` is required")
        return _partial_cosyne_permutation_for_population(values, evals, objective_sense)


@expects_ndim(2, 2)
def _combine_values(values1: torch.Tensor, values2: torch.Tensor) -> torch.Tensor:
    return torch.vstack([values1, values2])


@expects_ndim(2, 1, 2, 1)
def _combine_values_and_evals(
    values1: torch.Tensor, evals1: torch.Tensor, values2: torch.Tensor, evals2: torch.Tensor
) -> tuple:
    return torch.vstack([values1, values2]), torch.hstack([evals1, evals2])


@expects_ndim(2, 2, 2, 2)
def _combine_values_and_multiobjective_evals(
    values1: torch.Tensor, evals1: torch.Tensor, values2: torch.Tensor, evals2: torch.Tensor
) -> tuple:
    return torch.vstack([values1, values2]), torch.vstack([evals1, evals2])


def combine(
    a: Union[torch.Tensor, tuple],
    b: Union[torch.Tensor, tuple],
    *,
    objective_sense: Optional[Union[str, Iterable]] = None,
) -> Union[torch.Tensor, tuple]:
    """
    Combine two populations into one.

    This function can be used in two forms.

    **Usage 1: without evaluation results.**
    Let us assume that we have two decision values matrices, `values1`
    `values2`. The shapes of these matrices are (n1, L) and (n2, L)
    respectively, where L represents the length of a solution.
    Let us assume that the solutions that these decision values
    represent are not evaluated yet. Therefore, we do not have evaluation
    results (i.e. we do not have fitnesses). To combine these two
    unevaluated populations, we use this function as follows:

    ```python
    combined_population = combine(values1, values2)

    # We now have a combined decision values matrix, shaped (n1+n2, L).
    ```

    **Usage 2: with evaluation results, single-objective.**
    We again assume that we have two decision values matrices, `values1`
    and `values2`. Like in our previous example, these matrices are shaped
    (n1, L) and (n2, L), respectively. Additionally, let us assume that we
    know the evaluation results for the solutions represented by `values1`
    and `values2`. These evaluation results are represented by the tensors
    `evals1` and `evals2`, shaped (n1,) and (n2,), respectively. To combine
    these two evaluated populations, we use this function as follows:

    ```python
    c_values, c_evals = combine((values1, evals1), (values2, evals2))

    # We now have a combined decision values matrix and a combined evaluations
    # vector.
    # `c_values` is shaped (n1+n2, L), and `c_evals` is shaped (n1+n2,).
    ```

    **Usage 3: with evaluation results, multi-objective.**
    We again assume that we have two decision values matrices, `values1`
    and `values2`. Like in our previous example, these matrices are shaped
    (n1, L) and (n2, L), respectively. Additionally, we assume that we know
    the evaluation results for these solutions. The evaluation results are
    stored within the tensors `evals1` and `evals2`, whose shapes are
    (n1, M) and (n2, M), where M is the number of objectives. To combine
    these two evaluated populations, we use this function as follows:

    ```python
    c_values, c_evals = combine(
        (values1, evals1),
        (values2, evals2),
        objective_sense=["min", "min"],  # Assuming we have 2 min objectives
    )

    # We now have a combined decision values matrix and a combined evaluations
    # vector.
    # `c_values` is shaped (n1+n2, L), and `c_evals` is shaped (n1+n2,).
    ```

    Args:
        a: A decision values tensor with at least 2 dimensions, or a tuple
            of the form `(values, evals)`, where `values` is an at least
            2-dimensional decision values tensor, and `evals` is an at least
            1-dimensional evaluation results tensor.
            Extra leftmost dimensions are taken as batch dimensions.
            If this positional argument is a tensor, the second positional
            argument must also be a tensor. If this positional argument is a
            tuple, the second positional argument must also be a tuple.
        b: A decision values tensor with at least 2 dimensions, or a tuple
            of the form `(values, evals)`, where `values` is an at least
            2-dimensional decision values tensor, and `evals` is an at least
            1-dimensional evaluation results tensor.
            Extra leftmost dimensions are taken as batch dimensions.
            If this positional argument is a tensor, the first positional
            argument must also be a tensor. If this positional argument is a
            tuple, the first positional argument must also be a tuple.
        objective_sense: In the case of single-objective optimization,
            `objective_sense` can be left as None, or can be 'min' or 'max',
            representing the direction of the optimization.
            In the case of multi-objective optimization, `objective_sense`
            is expected as a list of strings, each string being 'min' or
            'max', representing the direction for each objective.
            Please also note that, if this combination operation is done
            without evaluation results (i.e. if the first two positional
            arguments are given as tensors, not tuples), `objective_sense`
            is not needed, and can be omitted, regardless of whether or
            not the problem at hand is single-objective.
    Returns:
        The combined decision values tensor, or a tuple of the form
        `(values, evals)` where `values` is the combined decision values
        tensor, and `evals` is the combined evaluation results tensor.
    """
    if isinstance(a, tuple):
        values1, evals1 = a
        if not isinstance(b, tuple):
            raise TypeError(
                "The first positional argument was received as a tuple."
                " Therefore, the second positional argument was also expected as a tuple."
                f" However, the second argument is {repr(b)} (of type {type(b)})."
            )
        values2, evals2 = b
        if (objective_sense is None) or isinstance(objective_sense, str):
            return _combine_values_and_evals(values1, evals1, values2, evals2)
        elif isinstance(objective_sense, Iterable):
            return _combine_values_and_multiobjective_evals(values1, evals1, values2, evals2)
        else:
            raise TypeError(f"Unrecognized `objective_sense`: {repr(objective_sense)}")
    elif isinstance(a, torch.Tensor):
        if not isinstance(b, torch.Tensor):
            raise TypeError(
                "The first positional argument was received as a tensor."
                " Therefore, the second positional argument was also expected as a tensor."
                f" However, the second argument is {repr(b)} (of type {type(b)})."
            )
        return _combine_values(a, b)
    else:
        raise TypeError(
            "Expected both positional arguments as tensors, or as tuples."
            f" However, the first positional argument is {repr(a)} (of type {type(a)})."
        )


@expects_ndim(2, 1, None)
def _take_single_best(values: torch.Tensor, evals: torch.Tensor, objective_sense: str) -> tuple:
    if objective_sense == "min":
        argfn = torch.argmin
    elif objective_sense == "max":
        argfn = torch.argmax
    else:
        raise ValueError(
            f"`objective_sense` was expected as 'min' or 'max', but was received as {repr(objective_sense)}"
        )

    _, solution_length = values.shape
    index_of_best = argfn(evals).reshape(1)
    best_row = torch.index_select(values, 0, index_of_best).reshape(solution_length)
    best_eval = torch.index_select(evals, 0, index_of_best).reshape(tuple())
    return best_row, best_eval


@expects_ndim(2, 1, None, None)
def _take_multiple_best(values: torch.Tensor, evals: torch.Tensor, n: int, objective_sense: str) -> tuple:
    if objective_sense == "min":
        descending = False
    elif objective_sense == "max":
        descending = True
    else:
        raise ValueError(
            f"`objective_sense` was expected as 'min' or 'max', but was received as {repr(objective_sense)}"
        )

    indices_of_best = torch.argsort(evals, descending=descending)[:n]
    best_rows = torch.index_select(values, 0, indices_of_best)
    best_evals = torch.index_select(evals, 0, indices_of_best)
    return best_rows, best_evals


@expects_ndim(2, 2, None, None, None)
def _take_multiple_best_with_multiobjective(
    values: torch.Tensor,
    evals: torch.Tensor,
    n: int,
    objective_sense: str,
    crowdsort: bool,
) -> tuple:
    utils = pareto_utility(evals, objective_sense=objective_sense, crowdsort=crowdsort)
    indices_of_best = torch.argsort(utils, descending=True)[:n]
    best_rows = torch.index_select(values, 0, indices_of_best)
    best_evals = torch.index_select(evals, 0, indices_of_best)
    return best_rows, best_evals


def take_best(
    values: torch.Tensor,
    evals: torch.Tensor,
    n: Optional[int] = None,
    *,
    objective_sense: Union[str, list],
    crowdsort: bool = True,
) -> tuple:
    """
    Take the best solution, or the best `n` number of solutions.

    **Single-objective case.**
    If the positional argument `n` is omitted (i.e. is left as None), the
    decision values and the evaluation result of the single best solution
    will be returned.
    If the positional argument `n` is provided, top-`n` solutions, together
    with their evaluation results, will be returned.

    **Multi-objective case.**
    In the multi-objective case, the positional argument `n` is mandatory.
    With a valid value for `n` given, `n` number of solutions will be taken
    from the best pareto-fronts. If `crowdsort` is given as True (which is
    the default), crowding distances of the solutions within the same
    pareto-fronts will be an additional criterion when deciding which
    solutions to take. Like in the single-objective case, the decision values
    and the evaluation results of the taken solutions will be returned.

    Args:
        values: Decision values tensor, with at least 2 dimensions.
            Extra leftmost dimensions will be taken as batch dimensions.
        evals: Evaluation results tensor, with at least 1 dimension.
            Extra leftmost dimensions will be taken as batch dimensions.
        n: If left as None, the single best solution will be taken.
            If given as an integer, this number of best solutions will be
            taken. Please note that, if the problem at hand has multiple
            objectives, this argument cannot be omitted.
        objective_sense: In the single-objective case, `objective_sense` is
            expected as a string 'min' or 'max', representing the direction
            of the optimization. In the multi-objective case,
            `objective_sense` is expected as a list of strings, each string
            being 'min' or 'max', representing the goal of optimization for
            each objective.
        crowdsort: Relevant only when there are multiple objectives.
            If `crowdsort` is True, the crowding distances of the solutions
            within the given population will be an additional criterion
            when choosing the best `n` solution. If `crowdsort` is False,
            how many times a solution was dominated will be the only factor
            when deciding whether or not it is among the best `n` solutions.
    Returns:
        A tuple of the form `(decision_values, evaluation_results)`, where
        `decision_values` is the decision values tensor for the taken
        solution(s), and `evaluation_results` is the evaluation results tensor
        for the taken solution(s).
    """
    if isinstance(objective_sense, str):
        multi_objective = False
    elif isinstance(objective_sense, Iterable):
        multi_objective = True
    else:
        raise TypeError("Unrecognized `objective_sense`: {repr(objective_sense)}")

    if n is None:
        if multi_objective:
            raise ValueError(
                "`objective_sense` not given as a string, implying that there are multiple objectives."
                " When there are multiple objectives, the argument `n` (i.e. number of solutions to take)"
                " must not be omitted. However, `n` was encountered as None."
            )
        return _take_single_best(values, evals, objective_sense)
    else:
        if multi_objective:
            return _take_multiple_best_with_multiobjective(values, evals, n, objective_sense, crowdsort)
        else:
            return _take_multiple_best(values, evals, n, objective_sense)
