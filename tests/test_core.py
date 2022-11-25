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

import pickle
from copy import copy, deepcopy
from itertools import product
from typing import Callable, Iterable, Optional, Union

import pytest
import torch

import evotorch as et
import evotorch.tools as ett
from evotorch import testing
from evotorch.tools.objectarray import ObjectArray


class CloningMethods:
    @staticmethod
    def clone_via_method(x):
        return x.clone()

    @staticmethod
    def clone_via_copy(x):
        return copy(x)

    @staticmethod
    def clone_via_deepcopy(x):
        return deepcopy(x)

    @staticmethod
    def clone_via_clone_func(x):
        return ett.clone(x)

    @staticmethod
    def deep_clone(x):
        return ett.cloning.deep_clone(x, otherwise_deepcopy=True)

    @staticmethod
    def pickle_and_unpickle(x):
        return pickle.loads(pickle.dumps(x))


class DummyProblems:
    @staticmethod
    def _sum(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            return torch.sum(x)
        elif x.ndim == 2:
            return torch.sum(x, dim=-1)
        else:
            raise ValueError(f"Unexpected shape: {x.shape}")

    @staticmethod
    def _make_simple_problem(
        *,
        vectorized: bool,
        objective_sense: Union[str, Iterable[str]] = "min",
        solution_length: int = 5,
        dtype: ett.DType = "float32",
        num_actors: Union[int, str] = 0,
    ):
        return et.Problem(
            objective_sense,
            DummyProblems._sum,
            solution_length=solution_length,
            dtype=dtype,
            num_actors=num_actors,
            initial_bounds=(-10.0, 10.0),
            vectorized=vectorized,
        )

    @staticmethod
    def make_simple_non_batched_problem(
        *,
        objective_sense: Union[str, Iterable[str]] = "min",
        solution_length: int = 5,
        dtype: ett.DType = "float32",
        num_actors: Union[int, str] = 0,
    ):
        return DummyProblems._make_simple_problem(
            objective_sense=objective_sense,
            solution_length=solution_length,
            dtype=dtype,
            num_actors=num_actors,
            vectorized=False,
        )

    @staticmethod
    def make_simple_batched_problem(
        *,
        objective_sense: Union[str, Iterable[str]] = "min",
        solution_length: int = 5,
        dtype: ett.DType = "float32",
        num_actors: Union[int, str] = 0,
    ):
        return DummyProblems._make_simple_problem(
            objective_sense=objective_sense,
            solution_length=solution_length,
            dtype=dtype,
            num_actors=num_actors,
            vectorized=True,
        )

    class Batched(et.Problem):
        def __init__(
            self,
            *,
            objective_sense: Union[str, Iterable[str]] = "min",
            solution_length: int = 5,
            dtype: ett.DType = "float32",
            num_actors: Union[int, str] = 0,
            eval_data_length: Optional[int] = None,
        ):
            super().__init__(
                objective_sense=objective_sense,
                solution_length=5,
                num_actors=num_actors,
                dtype=dtype,
                initial_bounds=(-10.0, 10.0),
                eval_data_length=eval_data_length,
            )

        def _evaluate_batch(self, batch: et.SolutionBatch):
            batch.forget_evals()
            batch.access_evals(0)[:] = torch.sum(batch.access_values(), dim=-1)

    class NonBatched(et.Problem):
        def __init__(
            self,
            *,
            objective_sense: Union[str, Iterable[str]] = "min",
            solution_length: int = 5,
            dtype: ett.DType = "float32",
            num_actors: Union[int, str] = 0,
        ):
            super().__init__(
                objective_sense=objective_sense,
                solution_length=5,
                num_actors=num_actors,
                dtype=dtype,
                initial_bounds=(-10.0, 10.0),
            )

        def _evaluate(self, x: et.Solution):
            x.set_evaluation(torch.sum(x.access_values()))

    class MultiObj(et.Problem):
        def __init__(self, *, eval_data_length: Optional[int] = None):
            super().__init__(
                objective_sense=["min", "max"],
                solution_length=2,
                dtype="float32",
                initial_bounds=(-10.0, 10.0),
                eval_data_length=eval_data_length,
            )

        def _evaluate_batch(self, batch: et.SolutionBatch):
            values = batch.access_values()
            x = values[:, 0]
            y = values[:, 1]
            batch.forget_evals()
            batch.access_evals(0)[:] = x
            batch.access_evals(1)[:] = y

    class OptimizeStrings(et.Problem):
        def __init__(self):
            super().__init__(objective_sense="min", dtype=object)

        def _evaluate(self, x: et.Solution):
            s = x.access_values()
            assert isinstance(s, str)
            x.set_evaluation(float(s))

        def _fill(self, x: ObjectArray):
            for i in range(len(x)):
                x[i] = str(int(self.make_randint(tuple(), n=100)))


@pytest.mark.parametrize(
    "prob_class,num_actors",
    [
        (DummyProblems.make_simple_batched_problem, 0),
        (DummyProblems.make_simple_non_batched_problem, 0),
        (DummyProblems.Batched, 0),
        (DummyProblems.NonBatched, 0),
        (DummyProblems.make_simple_batched_problem, 2),
        (DummyProblems.make_simple_non_batched_problem, 2),
        (DummyProblems.Batched, 2),
        (DummyProblems.NonBatched, 2),
    ],
)
def test_batch_and_solutions(prob_class, num_actors):
    # Instantiate the problem
    prob = prob_class(num_actors=num_actors)

    # Declare a population size
    popsize = 10

    # Make a batch of the specified population size
    batch: et.SolutionBatch = prob.generate_batch(popsize)

    # Length and shape of the batch should be consistent
    assert len(batch) == popsize
    assert batch.values_shape[0] == popsize
    assert batch.eval_shape[0] == popsize

    # Evaluate the batch
    prob.evaluate(batch)

    # The example problems in this test return the sum of decision values as fitnesses.
    # We manually compare the values here and assert that the fitnesses are correct.
    evals = batch.access_evals(0)
    for i in range(popsize):
        f = evals[i]
        s = torch.sum(batch.access_values(keep_evals=True)[i, :])
        testing.assert_allclose(s, f, atol=0.00001)

    # Here, we call `access_values()`, which should clear the attached fitnesses.
    batch.access_values()

    # Assert that the fitnesses are erased (they now should be nan).
    evals = batch.access_evals(0)
    assert torch.all(torch.isnan(evals))


def test_batch_slicing():
    prob = DummyProblems.Batched()
    popsize = 10

    # Make a new SolutionBatch
    batch = prob.generate_batch(10)

    # Evaluate the batch
    prob.evaluate(batch)

    # We will slice this batch from its middle row
    # (which we call the cutpoint in this test)
    cutpoint = popsize // 2

    # Take the bottom half of the population
    batchview = batch[cutpoint:]

    # Accessing the values of this batch should affect the original batch too.
    # Because `batchview` is just a view of the original batch.
    batchview.access_values()[:] = 1.0

    # Let's see if the related fitnesses in the original batch are indeed
    # erased.
    org_fitnesses = batch.access_evals(0)[cutpoint:]
    assert torch.all(torch.isnan(org_fitnesses))

    # Let's see if the decision values in the original batch got updated too.
    testing.assert_allclose(
        batch.access_values(keep_evals=True)[cutpoint:], batchview.access_values(keep_evals=True), atol=0.00001
    )

    # Now, let us fill the original batch with ones
    batch.access_values()[:] = 1.0

    # Let us also make sure that the original batch is evaluated again.
    prob.evaluate(batch)

    # Let's make another slice, but this time using an indices list.
    # Like in PyTorch, an indices list is considered an advanced slicing,
    # and it should lead to a copy, not a view.
    batch2 = batch[[1, 4, 2]]

    # Let's make all the decision values in the new batch 0.0
    batch2.access_values()[:] = 0.0

    # Did we cause the original batch's fitnesses to be erased?
    # Hopefully, we did not.
    assert not torch.any(torch.isnan(batch.access_evals(0)))

    # We now evaluate the second batch
    prob.evaluate(batch2)

    # Between the original batch (batch) and the newly made batch (batch2),
    # both the decision values and the fitnesses should be different at this point.
    bvalues = batch.access_values(keep_evals=True)
    b2values = batch2.access_values(keep_evals=True)

    bfitnesses = batch.access_evals(0)
    b2fitnesses = batch2.access_evals(0)

    testing.assert_allclose(bvalues, torch.ones_like(bvalues), atol=0.00001)
    testing.assert_allclose(b2values, torch.zeros_like(b2values), atol=0.00001)

    testing.assert_allclose(bfitnesses, torch.ones_like(bfitnesses) * batch.solution_length, atol=0.00001)
    testing.assert_allclose(b2fitnesses, torch.zeros_like(b2fitnesses), atol=0.00001)


@pytest.mark.parametrize(
    "clone_func",
    [
        CloningMethods.clone_via_clone_func,
        CloningMethods.clone_via_copy,
        CloningMethods.clone_via_deepcopy,
        CloningMethods.clone_via_method,
        CloningMethods.deep_clone,
        CloningMethods.pickle_and_unpickle,
    ],
)
def test_batch_cloning(clone_func):
    prob = DummyProblems.Batched()

    # Create a new SolutionBatch
    batch = prob.generate_batch(10)

    # Create a new clone of the original SolutionBatch
    batch2 = clone_func(batch)

    # We fill the original batch with ones, and the clone with zeroes.
    batch.access_values()[:] = 1.0
    batch2.access_values()[:] = 0.0

    # While batch2 gets filled with zeroes, the original batch should stay
    # the same (filled with ones). Below, we do the assertion for this.
    bvalues = batch.access_values(keep_evals=True)
    b2values = batch2.access_values(keep_evals=True)
    testing.assert_allclose(bvalues, torch.ones_like(bvalues), atol=0.00001)
    testing.assert_allclose(b2values, torch.zeros_like(b2values), atol=0.00001)


def _storage_address(solution_or_batch: Union[et.SolutionBatch, et.Solution]) -> int:
    return solution_or_batch.access_values(keep_evals=True).storage().data_ptr()


def _share_memory(a: Union[et.SolutionBatch, et.Solution], b: Union[et.SolutionBatch, et.Solution]) -> bool:
    return _storage_address(a) == _storage_address(b)


def test_solutions_referring_to_batch():
    prob = DummyProblems.Batched()

    # Create a new SolutionBatch, consisting of two solutions.
    batch = prob.generate_batch(2)

    # Fill the batch with 10s.
    batch.access_values()[:] = 10.0

    # Evaluate the new batch
    prob.evaluate(batch)

    # Get the solution length
    slnlen = prob.solution_length

    # Get the two solutions from the batch
    sln1 = batch[0]
    sln2 = batch[1]

    # Since we obtained these solutions from a SolutionBatch, these Solution
    # objects are references to the batch (i.e. they share their memories
    # with the batch).
    assert _share_memory(sln1, batch)
    assert _share_memory(sln2, batch)

    # Are the lengths correct?
    assert len(sln1) == slnlen
    assert len(sln2) == slnlen
    assert sln1.shape == (slnlen,)
    assert sln2.shape == (slnlen,)

    # Fill the first solution with zeroes
    sln1.access_values()[:] = 0.0

    # Accessing the values of the first solution should have caused the
    # fitness of the related row in the batch to be erased.
    # Also, the decision values of that first solution in the batch should
    # reflect the changes.
    assert torch.isnan(batch.access_evals(0)[0])
    # assert sln1.evaluation is None
    assert not sln1.is_evaluated
    testing.assert_eachclose(batch.access_values(keep_evals=True)[0, :], 0.0, atol=0.00001)

    # The other solution in the batch should stay the same.
    testing.assert_allclose(batch.access_evals(0)[1], 10 * slnlen, atol=0.00001)
    testing.assert_allclose(sln2.evaluation, 10 * slnlen, atol=0.00001)
    assert sln2.is_evaluated
    testing.assert_eachclose(batch.access_values(keep_evals=True)[1, :], 10.0, atol=0.00001)

    # Fill the second solution with 5s.
    sln2.access_values()[:] = 5.0

    # Now, the fitness of the second solution should be gone as well.
    assert torch.isnan(batch.access_evals(0)[1])
    # assert sln2.evaluation is None
    assert not sln2.is_evaluated
    testing.assert_eachclose(batch.access_values(keep_evals=True)[1, :], 5.0, atol=0.00001)


@pytest.mark.parametrize(
    "clone_func",
    [
        CloningMethods.clone_via_clone_func,
        CloningMethods.clone_via_copy,
        CloningMethods.clone_via_deepcopy,
        CloningMethods.clone_via_method,
        CloningMethods.deep_clone,
    ],
)
def test_solution_cloning(clone_func):
    prob = DummyProblems.Batched()

    # Create a new SolutionBatch, consisting of two solutions.
    batch = prob.generate_batch(2)

    # Fill the batch with 10s.
    batch.access_values()[:] = 10.0

    # Evaluate the new batch
    prob.evaluate(batch)

    # Get a new solution
    sln = batch[0]
    assert _share_memory(batch, sln)

    # Clone the solution
    sln2: et.Solution = clone_func(sln)

    # Since the new solution is a clone, it should NOT be a reference anymore
    assert not _share_memory(sln, sln2)

    sln2.access_values()[:] = 0.0

    testing.assert_eachclose(sln2, 0.0, atol=0.00001)
    testing.assert_eachclose(batch.access_values(keep_evals=True)[:], 10.0, atol=0.00001)


@pytest.mark.parametrize(
    "sense,ranking_method", [("min", None), ("max", None), ("min", "centered"), ("max", "centered")]
)
def test_utilities(sense, ranking_method):
    prob = DummyProblems.Batched(objective_sense=sense)
    batch = prob.generate_batch(2)

    values = batch.access_values()
    numel = values.numel()
    values.view(-1)[:] = torch.arange(numel, dtype=values.dtype)

    prob.evaluate(batch)

    utils = batch.utility()

    if sense == "min":
        assert utils[0] > utils[1]
    else:
        assert utils[1] > utils[0]


def test_multiobj():
    prob = DummyProblems.MultiObj()
    # Objectives: min, max

    batch = et.SolutionBatch(prob, popsize=5, empty=True)
    batch.access_values()[:] = torch.tensor(
        [
            [10, 2],
            [2, 10],
            [1, 9],
            [3, 7],
            [4, 8],
        ],
        dtype=batch.values_dtype,
    )

    prob.evaluate(batch)

    pareto = batch.arg_pareto_sort()

    assert torch.all(pareto.ranks == torch.LongTensor([2, 0, 0, 1, 1]))

    ranks, crowdsort_ranks = batch.compute_pareto_ranks(crowdsort=True)

    assert torch.all(ranks == torch.LongTensor([2, 0, 0, 1, 1]))
    assert torch.all(crowdsort_ranks == torch.LongTensor([0, 0, 1, 0, 1]))

    for ranking_method in (None, "centered"):
        utils = batch.utils(ranking_method=ranking_method)

        assert utils.shape == (5, 2)

        first_obj_utils = utils[:, 0]
        first_obj_sorted = first_obj_utils.argsort(descending=True)
        assert torch.all(first_obj_sorted == torch.LongTensor([2, 1, 3, 4, 0]))

        second_obj_utils = utils[:, 1]
        second_obj_sorted = second_obj_utils.argsort(descending=True)
        assert torch.all(second_obj_sorted == torch.LongTensor([1, 2, 4, 3, 0]))


def test_prob_with_dtype_object():
    prob = DummyProblems.OptimizeStrings()

    batch = prob.generate_batch(5)
    prob.evaluate(batch)

    fitnesses = [float(solution.access_values(keep_evals=True)) for solution in batch]

    testing.assert_allclose(batch.access_evals(0), fitnesses, atol=0.00001)


def test_api():
    def f(x: torch.Tensor):
        return torch.sum(x)

    def fmulti(x: torch.Tensor):
        return torch.sum(x), torch.linalg.norm(x)

    with pytest.raises(ValueError):
        et.Problem("min", f)  # Missing solution_length

    prob_with_bounds_missing = et.Problem("min", f, solution_length=5)  # Missing (initial_)bounds
    with pytest.raises(RuntimeError):
        prob_with_bounds_missing.generate_batch(10)  # fails because `fill(...)` does not know the bounds

    with pytest.raises(RuntimeError):
        prob_with_bounds_missing.generate_values(10)  # fails because `fill(...)` does not know the bounds

    with pytest.raises(ValueError):
        # This should fail, because et.Problem cannot work with non-numeric dtypes
        # when solution_length and bounds are given
        et.Problem("min", f, solution_length=5, bounds=(-1.0, 1.0), dtype=object)

    prob = et.Problem("min", f, solution_length=5, bounds=(-1.0, 1.0))
    prob.ensure_numeric()
    prob.ensure_single_objective()

    with pytest.raises(ValueError):
        prob.ensure_unbounded()

    assert prob.objective_sense == "min"
    assert prob.senses == ["min"]

    assert prob.initial_lower_bounds is not None
    assert prob.initial_upper_bounds is not None
    testing.assert_allclose(prob.lower_bounds, prob.initial_lower_bounds, atol=0.00001)
    testing.assert_allclose(prob.upper_bounds, prob.initial_upper_bounds, atol=0.00001)

    prob2 = et.Problem("min", f, solution_length=5, initial_bounds=(-1.0, 1.0), bounds=(-5.0, 5.0))
    prob2.ensure_numeric()
    prob2.ensure_single_objective()

    with pytest.raises(ValueError):
        prob2.ensure_unbounded()

    assert prob2.objective_sense == "min"
    assert prob2.senses == ["min"]

    prob3 = et.Problem("max", f, solution_length=5, initial_bounds=(-1.0, 1.0))
    prob3.ensure_unbounded()
    prob3.ensure_numeric()
    prob3.ensure_single_objective()

    assert prob3.objective_sense == "max"
    assert prob3.senses == ["max"]

    prob4 = et.Problem(["min", "min"], fmulti, solution_length=5, initial_bounds=(-1.0, 1.0))
    prob4.ensure_unbounded()
    prob4.ensure_numeric()

    with pytest.raises(ValueError):
        prob4.ensure_single_objective()

    assert prob4.objective_sense == ["min", "min"]
    assert prob4.senses == ["min", "min"]


def test_manual_fill():
    class ProblemWithManualFill(et.Problem):
        def __init__(self):
            super().__init__(objective_sense="min", solution_length=5, dtype="int64")

        def _evaluate_batch(self, batch: et.SolutionBatch):
            batch.set_evals(torch.linalg.norm(batch.values, dim=-1))

        def _fill(self, x: torch.Tensor):
            x.zero_()

    problem = ProblemWithManualFill()

    new_batch = problem.generate_batch(10)
    assert len(new_batch) == 10
    assert torch.all(new_batch.values == 0)

    new_values = problem.generate_values(10)
    assert new_values.shape == (10, 5)
    assert torch.all(new_values == 0)


def test_problem_hooks():
    class FuncWasCalled:
        f1 = False
        f2 = False

    prob = DummyProblems.Batched()

    def f1(b):
        assert isinstance(b, et.SolutionBatch)
        FuncWasCalled.f1 = True

    def f2(b):
        assert isinstance(b, et.SolutionBatch)
        FuncWasCalled.f2 = True
        return {"dummy_status": 1}

    prob.before_eval_hook.append(f1)
    prob.after_eval_hook.append(f2)

    batch = prob.generate_batch(5)
    prob.evaluate(batch)

    assert FuncWasCalled.f1
    assert FuncWasCalled.f2

    assert "dummy_status" in prob.status
    assert prob.status["dummy_status"] == 1


def _must_be_regular_tensor(x: torch.Tensor):
    assert isinstance(x, torch.Tensor) and (not isinstance(x, ett.ReadOnlyTensor))


def _must_be_read_only_tensor(x: torch.Tensor):
    assert isinstance(x, torch.Tensor) and isinstance(x, ett.ReadOnlyTensor)


def test_accessing():
    prob = DummyProblems.NonBatched()
    length = prob.solution_length
    population = prob.generate_batch(10)
    n = len(population)

    _must_be_regular_tensor(population.access_values(keep_evals=True))
    _must_be_regular_tensor(population.access_values())
    _must_be_read_only_tensor(population.values)
    _must_be_read_only_tensor(population.evals)
    testing.assert_shape_matches(population.access_values(keep_evals=True), (n, length))
    testing.assert_shape_matches(population.access_evals(), (n, 1))
    testing.assert_shape_matches(population.values, (n, length))
    testing.assert_shape_matches(population.evals, (n, 1))

    sln = population[0]
    _must_be_regular_tensor(sln.access_values(keep_evals=True))
    _must_be_regular_tensor(sln.access_values())
    _must_be_read_only_tensor(sln.values)
    _must_be_read_only_tensor(sln.evals)
    testing.assert_shape_matches(sln.access_values(keep_evals=True), length)
    testing.assert_shape_matches(sln.access_evals(), 1)
    testing.assert_shape_matches(sln.values, length)
    testing.assert_shape_matches(sln.evals, 1)


@pytest.mark.parametrize(
    "problem_cls,eval_data_length",
    product(
        [DummyProblems.Batched, DummyProblems.MultiObj],
        [None, 0, 2, 10],
    ),
)
def test_set_evals(problem_cls: Callable, eval_data_length: Optional[int]):
    problem: et.Problem = problem_cls(eval_data_length=eval_data_length)
    num_objs = len(problem.senses)
    popsize = 5
    dtype = problem.eval_dtype
    tolerance = 1e-4

    # =============================================

    if num_objs == 1:
        # Make a new batch
        batch = problem.generate_batch(popsize)

        # Produce some arbitrary evaluation results
        f = torch.arange(popsize, dtype=dtype)

        # Generate the expected contents of the evals tensor
        e = torch.empty(batch.evals.shape, dtype=dtype)
        e[:] = float("nan")
        e[:, 0] = f

        # Set the evaluation results in the batch
        batch.set_evals(f)

        # Is the evals tensor looking as expected?
        testing.assert_allclose(batch.evals, e, atol=tolerance)

    # =============================================
    # (Re-)make the batch
    batch = problem.generate_batch(popsize)

    # Produce some arbitrary evaluation results
    f = torch.randn((popsize, num_objs), dtype=dtype)

    # Generate the expected contents of the evals tensor
    e = torch.empty(batch.evals.shape, dtype=dtype)
    e[:] = float("nan")
    e[:, :num_objs] = f

    # Set the evaluation results in the batch
    batch.set_evals(f)

    # Is the evals tensor looking as expected?
    testing.assert_allclose(batch.evals, e, atol=tolerance)

    # =============================================
    if (eval_data_length is not None) and (eval_data_length > 0):
        # Re-make the batch
        batch = problem.generate_batch(popsize)

        # Produce some arbitrary evaluation results
        f = torch.randn((popsize, num_objs), dtype=dtype)

        # Produce some arbitrary evaluation data
        d = torch.randn((popsize, eval_data_length), dtype=dtype)

        # Generate the expected contents of the evals tensor
        e = torch.empty(batch.evals.shape, dtype=dtype)
        e[:, :num_objs] = f
        e[:, num_objs:] = d

        # Set the evaluation results in the batch
        batch.set_evals(f, d)

        # Is the evals tensor looking as expected?
        testing.assert_allclose(batch.evals, e, atol=tolerance)

        # =============================================
        # Re-make the batch
        batch = problem.generate_batch(popsize)

        # Produce some arbitrary evaluation results and evaluation data
        f = torch.randn((popsize, num_objs + eval_data_length), dtype=dtype)

        # Set the evaluation results in the batch
        batch.set_evals(f)

        # Is the evals tensor looking as expected?
        testing.assert_allclose(batch.evals, f, atol=tolerance)

    # =============================================
    if num_objs >= 2:
        for two_dim in False, True:
            # Re-make the batch
            batch = problem.generate_batch(popsize)

            # Produce some evaluation results, deliberately in a wrong shape
            f = torch.arange(popsize, dtype=dtype)
            if two_dim:
                f = f.reshape(popsize, 1)

            # This must fail
            with pytest.raises(ValueError):
                batch.set_evals(f)

    # =============================================
    if (eval_data_length is not None) and eval_data_length >= 2:
        # Re-make the batch
        batch = problem.generate_batch(popsize)

        # Produce some evaluation results, deliberately in a wrong shape
        f = torch.randn((popsize, num_objs + 1), dtype=dtype)

        # This must fail
        with pytest.raises(ValueError):
            batch.set_evals(f)
