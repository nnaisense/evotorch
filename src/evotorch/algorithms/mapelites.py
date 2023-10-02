import itertools
from typing import Callable, Iterable, Optional, Union

import torch

try:
    from torch.func import vmap
except ImportError:
    from functorch import vmap

from ..core import Problem, SolutionBatch
from ..operators import CosynePermutation, CrossOver, GaussianMutation, OnePointCrossOver, SimulatedBinaryCrossOver
from ..tools import Device, DType, to_torch_dtype
from .ga import ExtendedPopulationMixin
from .searchalgorithm import SearchAlgorithm, SinglePopulationAlgorithmMixin


def _all_across_rightmost_dim(x: torch.Tensor) -> torch.Tensor:
    # This is a vmap-friendly function which is equivalent to torch.all(..., dim=-1)
    rightmost_size = x.shape[-1]
    return x.to(dtype=torch.int64).sum(dim=-1) == rightmost_size


def _best_solution_considering_feature(
    objective_sense: str,
    decision_values: torch.Tensor,
    evals: torch.Tensor,
    feature_grid: torch.Tensor,
) -> tuple:
    feature_lb = feature_grid[:, 0]
    feature_ub = feature_grid[:, 1]

    if objective_sense == "min":
        penalty = float("inf")
        argbest = torch.argmin
    elif objective_sense == "max":
        penalty = float("-inf")
        argbest = torch.argmax
    else:
        raise ValueError(f"Unexpected objective_sense: {repr(objective_sense)}")

    fitnesses = evals[:, 0]
    features = evals[:, 1:]
    # suitable = torch.all(features >= feature_lb, dim=-1) & torch.all(features <= feature_ub, dim=-1)
    suitable = _all_across_rightmost_dim(features >= feature_lb) & _all_across_rightmost_dim(features <= feature_ub)
    processed_fitnesses = torch.where(suitable, fitnesses, penalty)
    index = argbest(processed_fitnesses)

    selected_dec_values = torch.index_select(decision_values, 0, index)[0]
    selected_evals = torch.index_select(evals, 0, index)[0]
    selected_suitable = torch.index_select(suitable, 0, index)[0]

    return selected_dec_values, selected_evals, selected_suitable


def _best_solution_considering_all_features(
    objective_sense: str,
    decision_values: torch.Tensor,
    evals: torch.Tensor,
    feature_grid: torch.Tensor,
) -> tuple:
    return vmap(_best_solution_considering_feature, in_dims=(None, None, None, 0))(
        objective_sense,
        decision_values,
        evals,
        feature_grid,
    )


class MAPElites(SearchAlgorithm, SinglePopulationAlgorithmMixin, ExtendedPopulationMixin):
    """
    Implementation of the MAPElites algorithm.

    In MAPElites, we deal with optimization problems where, in addition
    to the fitness, there are additional evaluation data ("features") that
    are computed during the phase of evaluation. To ensure a diversity
    of the solutions, the population is organized into cells of features.

    Reference:

        Jean-Baptiste Mouret and Jeff Clune (2015).
        Illuminating search spaces by mapping elites.
        arXiv preprint arXiv:1504.04909 (2015).

    As an example, let us imagine that our problem has two features.
    Let us call these features `feat0` and `feat1`.
    Let us also imagine that we wish to organize `feat0` according to
    the boundaries `[(-inf, 0), (0, 10), (10, 20), (20, +inf)]` and `feat1`
    according to the boundaries `[(-inf, 0), (0, 50), (50, +inf)]`.
    Our population gets organized into:

    ```text

         +inf
              ^
              |
      f       |           |        |         |
      e       |    pop[0] | pop[1] | pop[ 2] | pop[ 3]
      a   50 -|-  --------+--------+---------+---------
      t       |    pop[4] | pop[5] | pop[ 6] | pop[ 7]
      1    0 -|-  --------+--------|---------+---------
              |    pop[8] | pop[9] | pop[10] | pop[11]
              |           |        |         |
        <-----------------|--------|---------|----------->
     -inf     |           0       10        20            +inf
              |                  feat0
              |
              v
          -inf
    ```

    where `pop[i]` is short for `population[i]`, that is, the i-th solution
    of the population.

    **Which problems can be solved by MAPElites?**
    The problems that can be addressed by MAPElites are the problems with
    one objective, and with its `eval_data_length` (additional evaluation
    data length) set as an integer greater than or equal to 1.
    For example, let us imagine an optimization problem where we handle
    2 features. The evaluation function for such a problem could look like:

    ```python
    def f(x: torch.Tensor) -> torch.Tensor:
        # Somehow compute the fitness
        fitness = ...

        # Somehow compute the value for the first feature
        feat0 = ...

        # Somehow compute the value for the second feature
        feat1 = ...

        # Prepare an evaluation result tensor for the solution
        eval_result = torch.tensor([fitness, feat0, feat1], device=x.device)

        # Here, we return the eval_result.
        # Notice that `eval_result` is a 1-dimensional tensor of length 3,
        # where the item with index 0 is the fitness, and the items with
        # indices 1 and 2 represent the two features of the solution.
        # Please also note that, in vectorized mode, we would receive `n`
        # solutions, and the evaluation result tensor would have to be of shape
        # (n, 3).
        return eval_result
    ```

    The problem definition then would look like this:

    ```python
    from evotorch import Problem

    problem = Problem(
        "min",
        f,
        initial_bounds=(..., ...),
        solution_length=...,
        eval_data_length=2,  # we have 2 features
    )
    ```

    **Using MAPElites.**
    Let us continue using the example `problem` shown above, where we have
    two features.
    The first step towards configuring MAPElites is to come up with a
    hypergrid tensor, from in the lower and upper bound for each
    feature on each cell will be expressed. The hypergrid tensor is structured
    like this:

    ```python
    hypergrid = torch.tensor(
        [
            [
                [
                    feat0_lower_bound_for_cell0,
                    feat0_upper_bound_for_cell0,
                ],
                [
                    feat1_lower_bound_for_cell0,
                    feat1_upper_bound_for_cell0,
                ],
            ],
            [
                [
                    feat0_lower_bound_for_cell1,
                    feat0_upper_bound_for_cell1,
                ],
                [
                    feat1_lower_bound_for_cell1,
                    feat1_upper_bound_for_cell1,
                ],
            ],
            [
                [
                    feat0_lower_bound_for_cell2,
                    feat0_upper_bound_for_cell2,
                ],
                [
                    feat1_lower_bound_for_cell2,
                    feat1_upper_bound_for_cell2,
                ],
            ],
            ...,
        ],
        dtype=problem.eval_dtype,
        device=problem.device,
    )
    ```

    that is, the item with index `i,j,0` represents the lower bound for the
    j-th feature in i-th cell, and the item with index `i,j,1` represents the
    upper bound for the j-th feature in i-th cell.

    Specifying lower and upper bounds for each feature and for each cell can
    be tedious. MAPElites provides a static helper function named
    [make_feature_grid][evotorch.algorithms.mapelites.MAPElites.make_feature_grid]
    which asks for how many bins are desired for each feature, and then
    produces a hypergrid tensor. For example, if we want 10 bins for feature
    `feat0` and 5 bins for feature `feat1`, then, we could do:

    ```python
    hypergrid = MAPElites.make_feature_grid(
        lower_bounds=[
            global_lower_bound_for_feat0,
            global_lower_bound_for_feat1,
        ],
        upper_bounds=[
            global_upper_bound_for_feat0,
            global_upper_bound_for_feat1,
        ],
        num_bins=[10, 5],
        dtype=problem.eval_dtype,
        device=problem.device,
    )
    ```

    Now that `hypergrid` is prepared, one can instantiate `MAPElites` like
    this:

    ```python
    searcher = MAPElites(
        problem,
        operators=[...],  # list of operators like in GeneticAlgorithm
        feature_grid=hypergrid,
    )
    ```

    where the keyword argument `operators` is a list that contains functions
    or instances of [Operator][evotorch.operators.base.Operator], like expected
    by [GeneticAlgorithm][evotorch.algorithms.ga.GeneticAlgorithm].

    Once `MAPElites` is instantiated, it can be run like most of the search
    algorithm implementations provided by EvoTorch, as shown below:

    ```python
    from evotorch.logging import StdOutLogger

    _ = StdOutLogger(ga)  # Report the evolution's progress to standard output
    searcher.run(100)  # Run MAPElites for 100 generations
    print(dict(searcher.status))  # Show the final status dictionary
    ```

    **Vectorization capabilities.**
    According to the basic definition of the MAPElites algorithm, a cell is
    first chosen, then mutated, and then the mutated solution is placed back
    into the most suitable cell (if the cell is not filled yet or if the
    fitness of the newly mutated solution is better than the existing solution
    in that cell). When vectorization, and especially GPU-based parallelization
    is available, picking and mutating solutions one by one can be wasteful in
    terms of performance. Therefore, this MAPElites implementation mutates the
    entire population, evaluates all of the mutated solutions, and places all
    of them back into the most suitable cells, all in such a way that the
    vectorization and/or GPU-based parallelization can be exploited.
    """

    def __init__(
        self,
        problem: Problem,
        *,
        operators: Iterable,
        feature_grid: Iterable,
        re_evaluate: bool = True,
        re_evaluate_parents_first: Optional[bool] = None,
    ):
        """
        `__init__(...)`: Initialize the MAPElites algorithm.

        Args:
            problem: The problem object to work on. This problem object
                is expected to have one objective, and also have its
                `eval_data_length` set as an integer that is greater than
                or equal to 1.
            operators: Operators to be used by the MAPElites algorithm.
                Expected as an iterable, such as a list or a tuple.
                Each item within this iterable object is expected either
                as an instance of [Operator][evotorch.operators.base.Operator],
                or as a function which receives the decision values of
                multiple solutions in a PyTorch tensor and returns a modified
                copy.
            re_evaluate: Whether or not to evaluate the solutions
                that were already evaluated in the previous generations.
                By default, this is set as True.
                The reason behind this default setting is that,
                in problems where the evaluation procedure is noisy,
                by re-evaluating the already-evaluated solutions,
                we prevent the bad solutions that were luckily evaluated
                from hanging onto the population.
                Instead, at every generation, each solution must go through
                the evaluation procedure again and prove their worth.
                For problems whose evaluation procedures are NOT noisy,
                the user might consider turning re_evaluate to False
                for saving computational cycles.
            re_evaluate_parents_first: This is to be specified only when
                `re_evaluate` is True (otherwise to be left as None).
                If this is given as True, then it will be assumed that the
                provided operators require the parents to be evaluated.
                If this is given as False, then it will be assumed that the
                provided operators work without looking at the parents'
                fitnesses (in which case both parents and children can be
                evaluated in a single vectorized computation cycle).
                If this is left as None, then whether or not the operators
                need to know the parent evaluations will be determined
                automatically as follows:
                if the operators contain at least one cross-over operator
                then `re_evaluate_parents_first` will be internally set as
                True; otherwise `re_evaluate_parents_first` will be internally
                set as False.
        """
        problem.ensure_single_objective()
        problem.ensure_numeric()

        SearchAlgorithm.__init__(self, problem)

        self._feature_grid = problem.as_tensor(feature_grid, use_eval_dtype=True)
        self._sense = self._problem.senses[0]
        self._popsize = self._feature_grid.shape[0]

        self._population = problem.generate_batch(self._popsize)
        self._filled = torch.zeros(self._popsize, dtype=torch.bool, device=self._population.device)

        ExtendedPopulationMixin.__init__(
            self,
            re_evaluate=re_evaluate,
            re_evaluate_parents_first=re_evaluate_parents_first,
            operators=operators,
            allow_empty_operators_list=False,
        )

        SinglePopulationAlgorithmMixin.__init__(self)

    @property
    def population(self) -> SolutionBatch:
        """
        Get the population as a SolutionBatch object

        In this MAP-Elites implementation, i-th solution corresponds to the
        solution belonging to the i-th cell of the MAP-Elites hypergrid.
        If `filled[i]` is True, then this means that the i-th cell is filled,
        and therefore `population[i]` will get the solution belonging to the
        i-th cell.
        """
        return self._population

    @property
    def filled(self) -> torch.Tensor:
        """
        Get a boolean tensor specifying whether or not the i-th cell is filled.

        In this MAP-Elites implementation, i-th solution within the population
        corresponds to the solution belonging to the i-th cell of the MAP-Elites
        hypergrid. If `filled[i]` is True, then the solution stored in the i-th
        cell satisfies the feature boundaries imposed by that cell.
        If `filled[i]` is False, then the solution stored in the i-th cell
        does not satisfy those boundaries, and therefore does not really belong
        in that cell.
        """
        from ..tools.readonlytensor import as_read_only_tensor

        with torch.no_grad():
            return as_read_only_tensor(self._filled)

    def _step(self):
        # Form an extended population from the parents and from the children
        extended_population = self._make_extended_population(split=False)

        # Get the most suitable solutions for each cell of the hypergrid.
        # values[i, :] stores the decision values most suitable for the i-th cell.
        # evals[i, :] stores the evaluation results most suitable for the i-th cell.
        # if the suggested i-th solution completely satisfies the boundaries imposed by the i-th cell,
        # then suitable_mask[i] will be True.
        values, evals, suitable = _best_solution_considering_all_features(
            self._sense,
            extended_population.values.as_subclass(torch.Tensor),
            extended_population.evals.as_subclass(torch.Tensor),
            self._feature_grid,
        )

        # Place the most suitable decision values and evaluation results into the current population.
        self._population.access_values(keep_evals=True)[:] = values
        self._population.access_evals()[:] = evals

        # If there was a suitable solution for the i-th cell, fill[i] is to be set as True.
        self._filled[:] = suitable

    @staticmethod
    def make_feature_grid(
        lower_bounds: Iterable,
        upper_bounds: Iterable,
        num_bins: Union[int, torch.Tensor],
        *,
        device: Optional[Device] = None,
        dtype: Optional[DType] = None,
    ) -> torch.Tensor:
        """
        Make a hypergrid for the MAPElites algorithm.

        The [MAPElites][evotorch.algorithms.mapelites.MAPElites] organizes its
        population not only according to the fitness, but also according to the
        additional evaluation data which are interpreted as the additional features
        of the solutions. To organize the current population according to these
        [MAPElites][evotorch.algorithms.mapelites.MAPElites] requires "cells",
        each cell having a lower and an upper bound for each feature.
        `make_map_elites_grid(...)` is a helper function which generates the
        required hypergrid of features in such a way that each cell, for each
        feature, has the same interval.

        The result of this function is a PyTorch tensor, which can be passed to
        the `feature_grid` keyword argument of
        [MAPElites][evotorch.algorithms.mapelites.MAPElites].

        Args:
            lower_bounds: The lower bounds, as a 1-dimensional sequence of numbers.
                The length of this sequence must be equal to the number of
                features, and the i-th element must express the lower bound
                of the i-th feature.
            upper_bounds: The upper bounds, as a 1-dimensional sequence of numbers.
                The length of this sequence must be equal to the number of
                features, and the i-th element must express the upper bound
                of the i-th feature.
            num_bins: Can be given as an integer or as a sequence of integers.
                If given as an integer `n`, then there will be `n` bins for each
                feature on the hypergrid. If given as a sequence of integers,
                then the i-th element of the sequence will express the number of
                bins for the i-th feature.
        Returns:
            The hypergrid, as a PyTorch tensor.
        """

        cast_args = {}
        if device is not None:
            cast_args["device"] = torch.device(device)
        if dtype is not None:
            cast_args["dtype"] = to_torch_dtype(dtype)

        has_casting = len(cast_args) > 0

        if has_casting:
            lower_bounds = torch.as_tensor(lower_bounds, **cast_args)
            upper_bounds = torch.as_tensor(upper_bounds, **cast_args)

        if (not isinstance(lower_bounds, torch.Tensor)) or (not isinstance(upper_bounds, torch.Tensor)):
            raise TypeError(
                f"While preparing the map elites hypergrid with device={device} and dtype={dtype},"
                f"`lower_bounds` and `upper_bounds` were expected as tensors, but their types are different."
                f" The type of `lower_bounds` is {type(lower_bounds)}."
                f" The type of `upper_bounds` is {type(upper_bounds)}."
            )

        if lower_bounds.device != upper_bounds.device:
            raise ValueError(
                f"Cannot determine on which device to place the map elites grid,"
                f" because `lower_bounds` and `upper_bounds` are on different devices."
                f" The device of `lower_bounds` is {lower_bounds.device}."
                f" The device of `upper_bounds` is {upper_bounds.device}."
            )

        if lower_bounds.dtype != upper_bounds.dtype:
            raise ValueError(
                f"Cannot determine the dtype of the map elites grid,"
                f" because `lower_bounds` and `upper_bounds` have different dtypes."
                f" The dtype of `lower_bounds` is {lower_bounds.dtype}."
                f" The dtype of `upper_bounds` is {upper_bounds.dtype}."
            )

        if lower_bounds.size() != upper_bounds.size():
            raise ValueError("`lower_bounds` and `upper_bounds` have incompatible shapes")

        if lower_bounds.dim() != 1:
            raise ValueError("Only 1D tensors are supported for `lower_bounds` and for `upper_bounds`")

        dtype = lower_bounds.dtype
        device = lower_bounds.device

        num_bins = torch.as_tensor(num_bins, dtype=torch.int64, device=device)
        if num_bins.dim() == 0:
            num_bins = num_bins.expand(lower_bounds.size())

        p_inf = torch.tensor([float("inf")], dtype=dtype, device=device)
        n_inf = torch.tensor([float("-inf")], dtype=dtype, device=device)

        def _make_feature_grid(lb, ub, num_bins):
            sp = torch.linspace(lb, ub, num_bins - 1, device=device)
            sp = torch.cat((n_inf, sp, p_inf))
            return sp.unfold(dimension=0, size=2, step=1).unsqueeze(1)

        f_grids = [_make_feature_grid(*bounds) for bounds in zip(lower_bounds, upper_bounds, num_bins)]
        return torch.stack([torch.cat(c) for c in itertools.product(*f_grids)])
