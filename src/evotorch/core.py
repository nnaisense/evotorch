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
Definitions of the core classes:
Problem, Solution, and SolutionBatch.
"""

import io
import logging
import math
import os
import random
from collections.abc import Sequence
from contextlib import nullcontext
from copy import deepcopy
from typing import Any, Callable, Iterable, List, NamedTuple, Optional, Tuple, Union

import numpy as np
import ray
import torch

_evolog = logging.getLogger(__name__)

try:
    from numba import njit

    class NumbaLib:
        is_found = True

except ImportError:
    numba = None

    def njit(f):
        return f

    class NumbaLib:
        is_found = False
        already_warned = set()
        general_opr = "_general"

        @classmethod
        def warn(cls, operation_name: Optional[str] = None):
            if operation_name is None:
                operation_name = cls.general_opr

            if operation_name not in cls.already_warned:
                cls.already_warned.add(operation_name)

                if operation_name == cls.general_opr:
                    msg = "The library 'numba' is missing. Installing 'numba' can make certain operations faster."
                else:
                    msg = (
                        f"Regarding the operation {repr(operation_name)}:"
                        f" Install numba to make this procedure run faster."
                    )

                _evolog.warning(msg)


from ray.util import ActorPool

from .tools import (
    Device,
    DType,
    ErroneousResult,
    RealOrVector,
    cast_tensors_in_container,
    clone,
    empty_tensor_like,
    ensure_ray,
    expect_none,
    is_dtype_object,
    is_real,
    is_sequence,
    make_batched_false_for_vmap,
    message_from,
    modify_tensor,
    multiply_rows_by_scalars,
    rank,
    rowwise_sum,
    split_workload,
    storage_ptr,
    to_torch_dtype,
)
from .tools.cloning import Serializable, deep_clone
from .tools.hook import Hook
from .tools.objectarray import ObjectArray
from .tools.tensormaker import TensorMakerMixin

ObjectiveSense = Union[str, Iterable[str]]
Bounds = RealOrVector
BoundsPair = NamedTuple("BoundsPair", lb=RealOrVector, ub=RealOrVector)
BoundsPairLike = Union[Iterable[Bounds], BoundsPair]

IndicesOrSlice = Union[int, Iterable[int], slice]
MaybeIndicesOrSlice = Optional[IndicesOrSlice]

ActorSeeds = NamedTuple("ActorSeeds", py_global=int, np_global=int, torch_global=int, problem=int)


@ray.remote
class EvaluationActor:
    """An actor class for remotely evaluating solutions"""

    def __init__(self, problem: "Problem", index: int, seeds: Union[ActorSeeds, tuple], state: dict):
        """
        `__init__(...)`: Initialize the actor.

        Args:
            problem: The problem object to be stored by the actor.
            index: Index of this actor
            seed: An integer which is bigger than or equal to 0
                and less than 2**32, to be used as the seed
                of the internal random number generator of the
                stored problem object.
            state: The state dictionary to be loaded by the stored
                problem object.
        """
        self._problem = problem
        self._problem._actor_index = index
        py_global, np_global, torch_global, probseed = seeds
        random.seed(py_global)
        np.random.seed(np_global)
        torch.manual_seed(torch_global)
        if self._problem.has_own_generator:
            self._problem.seed(probseed)
        self._problem._use_pickle_data_from_main(state)
        self._problem.remote_hook(self._problem)
        if isinstance(self._problem._num_gpus_per_actor, str) and (self._problem._num_gpus_per_actor == "all"):
            if "CUDA_VISIBLE_DEVICES" in os.environ:
                del os.environ["CUDA_VISIBLE_DEVICES"]

    def evaluate_batch(self, solution_batch: "SolutionBatch") -> torch.Tensor:
        """Evaluate a solution batch.

        Args:
            solution_batch (SolutionBatch): Batch to be evaluated
        Returns:
            Evaluation results
        """
        self._problem.evaluate(solution_batch)
        return solution_batch.access_evals()

    def evaluate_batch_piece(self, piece_index: int, batch_piece: "SolutionBatch") -> tuple:
        """Evaluate a solution batch, which is considered to be a piece
        (most probably a slice) of a bigger batch.
        Call this function for unordered parallelization, as this function
        returns the `piece_index` it receives, allowing one to reconstruct
        the true order of results.

        Args:
            piece_index: Index of the piece in the bigger batch.
            batch_piece: The SolutionBatch that is the piece.
        Returns:
            The piece index and the evaluation results in a tuple.
        """
        return piece_index, self.evaluate_batch(batch_piece)

    def call(self, method_name: str, args: list, kwargs: dict) -> object:
        """Call a method of the contained Problem object.

        Args:
            method_name: Name of the method belonging to the
                stored Problem object.
            args: A list containing the arguments to be passed
                to the method of the Problem object.
            kwargs: A dictionary containing the keyword arguments
                to be passed to the method of the Problem object.

        Returns:
            The return value of the method.
        """
        return getattr(self._problem, method_name)(*args, **kwargs)

    def get(self, field: Union[str, Iterable[str]]) -> Any:
        """
        Get the value of a field of the problem object stored by the actor.

        For example, the following code would get the value of the field
        named `a` from the problem object.

            my_actor.get.remote("a")

        Also, the following codes would get the value of the field `b.c`
        (`c` of `b` of the problem object):

            my_actor.get.remote("b.c")
            my_actor.get.remote(["b", "c"])

        Args:
            field: Field to read. Can be a string, or a list of strings.
        Returns:
            Value of the field.
        """
        if isinstance(field, str):
            field = field.split(".")

        obj = self._problem
        for field_name in field:
            obj = getattr(obj, field_name)

        return obj

    def set(self, field: str, x: Any):
        """
        Set the value of a field of the problem object stored by the actor.

        For example, the following code would set the value of the field
        named `a` of the problem object to 10.

            my_actor.set.remote("a", 10)

        Also, the following codes would set the value of the field `b.c`
        (`c` of `b` of the problem object) to 20:

            my_actor.set.remote("b.c", 20)
            my_actor.set.remote(["b", "c"], 20)

        Args:
            field: Field to set. Can be a string, or a list of strings.
            x: Value to be written into the field.
        """
        if isinstance(field, str):
            field = field.split(".")
        else:
            field = list(field)

        n = len(field)
        last = n - 1

        obj = self._problem
        for i, field_name in enumerate(field):
            if i == last:
                setattr(obj, field_name, x)
            else:
                obj = getattr(obj, field_name)

    def call_on_env(self, method_name: str, args: list, kwargs: dict) -> Any:
        """
        Call a method on the Problem object's contained reinforcement
        learning environment.
        It is assumed that the Problem object is a GymProblem, or
        a Problem subclass with a similar interface.

        Args:
            method_name: Name of the method belonging to the
                stored reinforcement learning environment.
            args: A list containing the arguments to be passed
                to the method of the Problem object.
            kwargs: A dictionary containing the keyword arguments
                to be passed to the method of the Problem object.

        Returns:
            The return value of the method.
        """
        return getattr(self._problem.get_env(), method_name)(*args, **kwargs)


class AllRemoteProblems:
    """
    Representation of all remote problem instances stored by the ray actors.

    An instance of this class is to be obtained from a main
    (i.e. non-remote) Problem object, as follows:

        remote_probs = my_problem.all_remote_problems()

    A remote method f() on all remote Problem instances can then
    be executed as follows:

        results = remote_probs.f()

    Given that there are `n` actors, `results` contains `n` objects,
    the i-th object being the method's result from the i-th actor.
    An alternative to the example above is like this:

        results = my_problem.all_remote_problems().f()
    """

    def __init__(self, actors: list):
        self._actors = actors

    def __getattr__(self, attr_name: str) -> Any:
        return RemoteMethod(attr_name, self._actors)


class AllRemoteEnvs:
    """
    Representation of all remote reinforcement learning instances
    stored by the ray actors.

    An instance of this class is to be obtained from a main
    (i.e. non-remote) Problem object, as follows:

        remote_envs = my_problem.all_remote_envs()

    A remote method f() on all remote environments can then
    be executed as follows:

        results = remote_envs.f()

    Given that there are `n` actors, `results` contains `n` objects,
    the i-th object being the method's result from the i-th actor.
    An alternative to the example above is like this:

        results = my_problem.all_remote_envs().f()
    """

    def __init__(self, actors: list):
        self._actors = actors

    def __getattr__(self, attr_name: str) -> Any:
        return RemoteMethod(attr_name, self._actors, on_env=True)


class RemoteMethod:
    """
    Representation of a method on a remote actor's contained Problem
    or reinforcement learning environment
    """

    def __init__(self, method_name: str, actors: list, on_env: bool = False):
        self._method_name = str(method_name)
        self._actors = actors
        self._on_env = bool(on_env)

    def __call__(self, *args, **kwargs) -> Any:
        def invoke(actor):
            if self._on_env:
                return actor.call_on_env.remote(self._method_name, args, kwargs)
            else:
                return actor.call.remote(self._method_name, args, kwargs)

        return ray.get([invoke(actor) for actor in self._actors])

    def __repr__(self) -> str:
        if self._on_env:
            further = ", on_env=True"
        else:
            further = ""
        return f"<{type(self).__name__} {repr(self._method_name)}{further}>"


def _no_grad_if_basic_dtype(dtype: DType):
    if is_dtype_object(dtype):
        return nullcontext()
    else:
        return torch.no_grad()


class Problem(TensorMakerMixin, Serializable):
    """
    Representation of a problem to be optimized.

    The simplest way to use this class is to instantiate it with an
    external fitness function.

    Let us imagine that we have the following fitness function:

    ```
    import torch

    def f(solution: torch.Tensor) -> torch.Tensor:
        return torch.linalg.norm(solution)
    ```

    A problem definition can be made around this fitness function as follows:

    ```
    from evotorch import Problem

    problem = Problem(
        "min", f,  # Goal is to minimize f (would be "max" for maximization)
        solution_length=10,  # Length of a solution is 10
        initial_bounds=(-5.0, 5.0),  # Bounds for sampling a new solution
        dtype=torch.float32,  # dtype of a solution
    )
    ```

    **Vectorized problem definitions.**
    To boost the runtime performance, one might want to define a vectorized
    fitness function where the fitnesses of multiple solutions are computed
    in a batched manner using the vectorization capabilities of PyTorch.
    A vectorized problem definition can be made as follows:

    ```
    from evotorch.decorators import vectorized

    @vectorized
    def vf(solutions: torch.Tensor) -> torch.Tensor:
        return torch.linalg.norm(solutions ** 2, dim=-1)

    problem = Problem(
        "min", vf,  # Goal is to minimize vf (would be "max" for maximization)
        solution_length=10,  # Length of a solution is 10
        initial_bounds=(-5.0, 5.0),  # Bounds for sampling a new solution
        dtype=torch.float32,  # dtype of a solution
    )
    ```

    **Parallelization across multiple CPUs.**
    An optimization problem can be configured to parallelize its evaluation
    operations across multiple CPUs as follows:

    ```python
    problem = Problem("min", f, ..., num_actors=4)  # will use 4 actors
    ```

    **Exploiting hardware accelerators.**
    As an alternative to CPU-based parallelization, one might prefer to use
    the parallelized computation capabilities of a hardware accelerator such
    as CUDA. To load the problem onto a cuda device (for example, onto
    "cuda:0"), one can do:

    ```python
    from evotorch.decorators import vectorized


    @vectorized
    def vf(solutions: torch.Tensor) -> torch.Tensor:
        return ...


    problem = Problem("min", vf, ..., device="cuda:0")
    ```

    **Exploiting multiple GPUs in parallel.**
    One can also keep the entire population on the CPU, and split and distribute
    it to multiple GPUs for GPU-accelerated and parallelized fitness evaluation.
    For this, the main device of the problem is set as CPU, but the fitness
    function is decorated with `evotorch.decorators.on_cuda`.

    ```python
    from evotorch.decorators import on_cuda, vectorized


    @on_cuda
    @vectorized
    def vf(solutions: torch.Tensor) -> torch.Tensor:
        return ...


    problem = Problem(
        "min",
        vf,
        ...,
        num_actors=N,  # where N>1 and equal to the number of GPUs
        # Note: if you are on a computer or on a ray cluster with multiple
        # GPUs, you might prefer to use the string "num_gpus" instead of an
        # integer N, which will cause the number of available GPUs to be
        # counted, and the number of actors to be configured as that count.
        #
        num_gpus_per_actor=1,  # each GPU is assigned to an actor
        device="cpu",
    )
    ```

    **Defining problems via inheritance.**
    A problem can also be defined via inheritance.
    Using inheritance, one can define problems which carry their own additional
    data, and/or update their states as more solutions are evaluated,
    and/or have custom procedures for sampling new solutions, etc.

    As a first example, let us define a parameterized problem. In this example
    problem, the goal is to minimize `x^(2q)`, `q` being a parameter of the
    problem. The definition of such a problem can be as follows:

    ```python
    from evotorch import Problem, Solution


    class MyProblem(Problem):
        def __init__(self, q: float):
            self.q = float(q)

            super().__init__(
                objective_sense="min",  # the goal is to minimize
                solution_length=10,  # a solution has the length 10
                initial_bounds=(-5.0, 5.0),  # sample new solutions from within [-5, 5]
                dtype=torch.float32,  # the dtype of a solution is float32
                # num_actors=...,  # if parallelization via multiple actors is desired
            )

        def _evaluate(self, solution: Solution):
            # This is where we declare the procedure of evaluating a solution

            # Get the decision values of the solution as a PyTorch tensor
            x = solution.values

            # Compute the fitness
            fitness = torch.sum(x ** (2 * self.q))

            # Register the fitness into the Solution object
            solution.set_evaluation(fitness)
    ```

    This parameterized problem can be instantiated as follows (let's say with q=3):

    ```python
    problem = MyProblem(q=3)
    ```

    **Defining vectorized problems via inheritance.**
    Vectorization can be used with inheritance-based problem definitions as well.
    Please see the following example where the method `_evaluate_batch`
    is used instead of `_evaluate` for vectorization:

    ```python
    from evotorch import Problem, SolutionBatch


    class MyVectorizedProblem(Problem):
        def __init__(self, q: float):
            self.q = float(q)

            super().__init__(
                objective_sense="min",  # the goal is to minimize
                solution_length=10,  # a solution has the length 10
                initial_bounds=(-5.0, 5.0),  # sample new solutions from within [-5, 5]
                dtype=torch.float32,  # the dtype of a solution is float32
                # num_actors=...,  # if parallelization via multiple actors is desired
                # device="cuda:0",  # if hardware acceleration is desired
            )

        def _evaluate_batch(self, solutions: SolutionBatch):
            # Get the decision values of all the solutions in a 2D PyTorch tensor:
            xs = solutions.values

            # Compute the fitnesses
            fitnesses = torch.sum(x ** (2 * self.q), dim=-1)

            # Register the fitness into the Solution object
            solutions.set_evals(fitnesses)
    ```

    **Using multiple GPUs from a problem defined via inheritance.**
    The previous example demonstrating the use of multiple GPUs showed how
    an independent fitness function can be decorated via
    `evotorch.decorators.on_cuda`. Instead of using an independent fitness
    function, if one wishes to define a problem by subclassing `Problem`,
    the overriden method `_evaluate_batch(...)` has to be decorated by
    `evotorch.decorators.on_cuda`. Like in the previous multi-GPU example,
    let us assume that we want to parallelize the fitness evaluation
    across N GPUs (where N>1). The inheritance-based code to achieve this
    can look like this:

    ```python
    from evotorch import Problem, SolutionBatch
    from evotorch.decorators import on_cuda


    class MyMultiGPUProblem(Problem):
        def __init__(self):
            ...
            super().__init__(
                objective_sense="min",  # the goal is to minimize
                solution_length=10,  # a solution has the length 10
                initial_bounds=(-5.0, 5.0),  # sample new solutions from within [-5, 5]
                dtype=torch.float32,  # the dtype of a solution is float32
                num_actors=N,  # allocate N actors
                # Note: if you are on a computer or on a ray cluster with multiple
                # GPUs, you might prefer to use the string "num_gpus" instead of an
                # integer N, which will cause the number of available GPUs to be
                # counted, and the number of actors to be configured as that count.
                #
                num_gpus_per_actor=1,  # for each actor, assign a cuda device
                device="cpu",  # keep the main population on the CPU
            )

        @on_cuda
        def _evaluate_batch(self, solutions: SolutionBatch):
            # Get the decision values of all the solutions in a 2D PyTorch tensor:
            xs = solutions.values

            # Compute the fitnesses
            fitnesses = ...

            # Register the fitness into the Solution object
            solutions.set_evals(fitnesses)
    ```

    **Customizing how initial solutions are sampled.**
    Instead of sampling solutions from within an interval, one might wish to
    define a special procedure for generating new solutions. This can be
    achieved by overriding the `_fill(...)` method of the Problem class.
    Please see the example below.

    ```python
    class MyProblemWithCustomizedFilling(Problem):
        def __init__(self):
            super().__init__(
                objective_sense="min",
                solution_length=10,
                dtype=torch.float32,
                # we do not set initial_bounds because we have a manual procedure
                # for initializing solutions
            )

        def _evaluate_batch(
            self, solutions: SolutionBatch
        ): ...  # code to compute and fill the fitnesses goes here

        def _fill(self, values: torch.Tensor):
            # `values` is an empty tensor of shape (n, m) where n is the number
            # of solutions and m is the solution length.
            # The responsibility of this method is to fill this tensor.
            # In the case of this example, let us say that we wish the new
            # solutions to have values sampled from a standard normal distribution.
            values.normal_()
    ```

    **Defining manually-structured optimization problems.**
    The `dtype` of an optimization problem can be set as `object`.
    When the `dtype` is set as an `object`, it means that a solution's
    value can be a PyTorch tensor, or a numpy array, or a Python list,
    or a Python dictionary, or a string, or a scalar, or `None`.
    This gives the user enough flexibility to express non-numeric
    optimization problems and/or problems where each solution has its
    own length, or even its own structure.

    In the example below, we define an optimization problem where a
    solution is represented by a Python list and where each solution can
    have its own length. For simplicity, we define the fitness function
    as the sum of the values of a solution.

    ```python
    from evotorch import Problem, SolutionBatch
    from evotorch.tools import ObjectArray
    import random
    import torch


    class MyCustomStructuredProblem(Problem):
        def __init__(self):
            super().__init__(
                objective_sense="min",
                dtype=object,
            )

        def _evaluate_batch(self, solutions: SolutionBatch):
            # Get the number of solutions
            n = len(solutions)

            # Allocate a PyTorch tensor that will store the fitnesses
            fitnesses = torch.empty(n, dtype=torch.float32)

            # Fitness is computed as the sum of numeric values stored
            # by a solution.
            for i in range(n):
                # Get the values stored by a solution (which, in the case of
                # this example, is a Python list, because we initialize them
                # so in the _fill method).
                sln_values = solutions[i].values
                fitnesses[i] = sum(sln_values)

            # Set the fitnesses
            solutions.set_evals(fitnesses)

        def _fill(self, values: ObjectArray):
            # At this point, we have an ObjectArray of length `n`.
            # This means, we need to fill the values of `n` solutions.
            # `values[i]` represents the values of the i-th solution.
            # Initially, `values[i]` is None.
            # It is up to us how `values[i]` will be filled.
            # Let us make each solution be initialized as a list of
            # random length, containing random real numbers.

            for i in range(len(values)):
                ith_solution_length = random.randint(1, 10)
                ith_solution_values = [random.random() for _ in range(ith_solution_length)]
                values[i] = ith_solution_values
    ```

    **Multi-objective optimization.**
    A multi-objective optimization problem can be expressed by using multiple
    objective senses. As an example, let us consider an optimization problem
    where the first objective sense is minimization and the second objective
    sense is maximization. When working with an external fitness function,
    the code to express such an optimization problem would look like this:

    ```python
    from evotorch import Problem
    from evotorch.decorators import vectorized
    import torch


    @vectorized
    def f(x: torch.Tensor) -> torch.Tensor:
        # (Note that if dtype is object, x will be of type ObjectArray,
        # and not a PyTorch tensor)

        # Code to compute the fitnesses goes here.
        # Our resulting tensor `fitnesses` is expected to have a shape (n, m)
        # where n is the number of solutions and m is the number of objectives
        # (which is 2 in the case of this example).
        # `fitnesses[i, k]` is expected to store the fitness value belonging
        # to the i-th solution according to the k-th objective.
        fitnesses: torch.Tensor = ...
        return fitnesses


    problem = Problem(["min", "max"], f, ...)
    ```

    A multi-objective problem defined via class inheritance would look like this:

    ```python
    from evotorch import Problem, SolutionBatch


    class MyMultiObjectiveProblem(Problem):
        def __init__(self):
            super().__init__(objective_sense=["min", "max"], ...)

        def _evaluate_batch(self, solutions: SolutionBatch):
            # Code to compute the fitnesses goes here.
            # `fitnesses[i, k]` is expected to store the fitness value belonging
            # to the i-th solution according to the k-th objective.
            fitnesses: torch.Tensor = ...

            # Set the fitnesses
            solutions.set_evals(fitnesses)
    ```

    **How to solve a problem.**
    If the optimization problem is single-objective and its dtype is a float
    (e.g. torch.float32, torch.float64, etc.), then it can be solved using
    any search algorithm implemented in EvoTorch. Let us assume that we have
    such an optimization problem stored by the variable `prob`. We could use
    the [cross entropy method][evotorch.algorithms.distributed.gaussian.CEM])
    to solve it:

    ```python
    from evotorch import Problem
    from evotorch.algorithms import CEM
    from evotorch.logging import StdOutLogger


    def f(x: torch.Tensor) -> torch.Tensor: ...


    prob = Problem("min", f, solution_length=..., dtype=torch.float32)

    searcher = CEM(
        problem,
        # The keyword arguments below refer to hyperparameters specific to the
        # cross entropy method algorithm. It is recommended to tune these
        # hyperparameters according to the problem at hand.
        popsize=100,  # population size
        parenthood_ratio=0.5,  # 0.5 means better half of solutions become parents
        stdev_init=10.0,  # initial standard deviation of the search distribution
    )

    _ = StdOutLogger(searcher)  # to report the progress onto the screen
    searcher.run(50)  # run for 50 generations

    print("Center of the search distribution:", searcher.status["center"])
    print("Solution with best fitness ever:", searcher.status["best"])
    ```

    See the namespace [evotorch.algorithms][evotorch.algorithms] to see the
    algorithms implemented within EvoTorch.

    If the optimization problem at hand has an integer dtype (e.g. torch.int64),
    or has the `object` dtype, or has multiple objectives, then distribution-based
    search algorithms such as CEM cannot be used (since those algorithms were
    implemented with continuous decision variables and with single-objective
    problems in mind). In such cases, one can use the algorithm named
    [SteadyState][evotorch.algorithms.ga.SteadyStateGA].
    Please also note that, while using
    [SteadyStateGA][evotorch.algorithms.ga.SteadyStateGA] on a problem with an
    integer dtype or with the `object` dtype, one will have to define manual
    cross-over and mutation operators specialized to the solution structure
    of the problem at hand. Please see the documentation of
    [SteadyStateGA][evotorch.algorithms.ga.SteadyStateGA] for details.
    """

    def __init__(
        self,
        objective_sense: ObjectiveSense,
        objective_func: Optional[Callable] = None,
        *,
        initial_bounds: Optional[BoundsPairLike] = None,
        bounds: Optional[BoundsPairLike] = None,
        solution_length: Optional[int] = None,
        dtype: Optional[DType] = None,
        eval_dtype: Optional[DType] = None,
        device: Optional[Device] = None,
        eval_data_length: Optional[int] = None,
        seed: Optional[int] = None,
        num_actors: Optional[Union[int, str]] = None,
        actor_config: Optional[dict] = None,
        num_gpus_per_actor: Optional[Union[int, float, str]] = None,
        num_subbatches: Optional[int] = None,
        subbatch_size: Optional[int] = None,
        store_solution_stats: Optional[bool] = None,
        vectorized: Optional[bool] = None,
    ):
        """
        `__init__(...)`: Initialize the Problem object.

        Args:
            objective_sense: A string, or a sequence of strings.
                For a single-objective problem, a single string
                ("min" or "max", for minimization or maximization)
                is enough.
                For a problem with `n` objectives, a sequence
                of strings (e.g. a list of strings) of length `n` is
                required, each string in the sequence being "min" or
                "max". This argument specifies the goal of the
                optimization.
            initial_bounds: In which interval will the values of a
                new solution will be initialized.
                Expected as a tuple, each element being either a
                scalar, or a vector of length `n`, `n` being the
                length of a solution.
                If a manual solution initialization is preferred
                (instead of an interval-based initialization),
                one can leave `initial_bounds` as None, and override
                the `_fill(...)` method in the inheriting subclass.
            bounds: Interval in which all the solutions must always
                reside.
                Expected as a tuple, each element being either a
                scalar, or a vector of length `n`, `n` being the
                length of a solution.
                This argument is optional, and can be left as None
                if one does not wish to declare hard bounds on the
                decision values of the problem.
                If `bounds` is specified, `initial_bounds` is missing,
                and `_fill(...)` is not overriden, then `bounds` will
                also serve as the `initial_bounds`.
            solution_length: Length of a solution.
                Required for all fixed-length numeric optimization
                problems.
                For variable-length problems (which might or might not
                be numeric), one is expected to leave `solution_length`
                as None, and declare `dtype` as `object`.
            dtype: dtype (data type) of the data stored by a solution.
                Can be given as a string (e.g. "float32"),
                or as a numpy dtype (e.g. `numpy.dtype("float32")`),
                or as a PyTorch dtype (e.g. `torch.float32`).
                Alternatively, if the problem is variable-length
                and/or non-numeric, one is expected to declare `dtype`
                as `object`.
            eval_dtype: dtype to be used for storing the evaluations
                (or fitnesses, or scores, or costs, or losses)
                of the solutions.
                Can be given as a string (e.g. "float32"),
                or as a numpy dtype (e.g. `numpy.dtype("float32")`),
                or as a PyTorch dtype (e.g. `torch.float32`).
                `eval_dtype` must always refer to a "float" data type,
                therefore, `object` is not accepted as a valid `eval_dtype`.
                If `eval_dtype` is not specified (i.e. left as None),
                then the following actions are taken to determine the
                `eval_dtype`:
                if `dtype` is "float16", `eval_dtype` becomes "float16";
                if `dtype` is "bfloat16", `eval_dtype` becomes "bfloat16";
                if `dtype` is "float32", `eval_dtype` becomes "float32";
                if `dtype` is "float64", `eval_dtype` becomes "float64";
                and for any other `dtype`, `eval_dtype` becomes "float32".
            device: Default device in which a new population will be
                generated. For non-numeric problems, this must be "cpu".
                For numeric problems, this can be any device supported
                by PyTorch (e.g. "cuda").
                Note that, if the number of actors of the problem is configured
                to be more than 1, `device` has to be "cpu" (or, equivalently,
                left as None).
            eval_data_length: In addition to evaluation results
                (which are (un)fitnesses, or scores, or costs, or losses),
                each solution can store extra evaluation data.
                If storage of such extra evaluation data is required,
                one can set this argument to an integer bigger than 0.
            seed: Random seed to be used by the random number generator
                attached to the problem object.
                If left as None, no random number generator will be
                attached, and the global random number generator of
                PyTorch will be used instead.
            num_actors: Number of actors to create for parallelized
                evaluation of the solutions.
                Certain string values are also accepted.
                When given as "max" or as "num_cpus", the number of actors
                will be equal to the number of all available CPUs in the ray
                cluster.
                When given as "num_gpus", the number of actors will be
                equal to the number of all available GPUs in the ray
                cluster, and each actor will be assigned a GPU.
                There is also an option, "num_devices", which means that
                both the numbers of CPUs and GPUs will be analyzed, and
                new actors and GPUs for them will be allocated,
                in a one-to-one mapping manner, if possible.
                In more details, with `num_actors="num_devices"`, if
                `device` is given as a GPU device, then it will be inferred
                that the user wishes to put everything (including the
                population) on a single GPU, and therefore there won't be
                any allocation of actors nor GPUs.
                With `num_actors="num_devices"` and with `device` set as
                "cpu" (or as left as None), if there are multiple CPUs
                and multiple GPUs, then `n` actors will be allocated
                where `n` is the minimum among the number of CPUs
                and the number of GPUs, so that there can be one-to-one
                mapping between CPUs and GPUs (i.e. such that each actor
                can be assigned an entire GPU).
                If `num_actors` is given as "num_gpus" or "num_devices",
                the argument `num_gpus_per_actor` must not be used,
                and the `actor_config` dictionary must not contain the
                key "num_gpus".
                If `num_actors` is given as something other than "num_gpus"
                or "num_devices", and if you wish to assign GPUs to each
                actor, then please see the argument `num_gpus_per_actor`.
            actor_config: A dictionary, representing the keyword arguments
                to be passed to the options(...) used when creating the
                ray actor objects. To be used for explicitly allocating
                resources per each actor.
                For example, for declaring that each actor is to use a GPU,
                one can pass `actor_config=dict(num_gpus=1)`.
                Can also be given as None (which is the default),
                if no such options are to be passed.
            num_gpus_per_actor: Number of GPUs to be allocated by each
                remote actor.
                The default behavior is to NOT allocate any GPU at all
                (which is the default behavior of the ray library as well).
                When given as a number `n`, each actor will be given
                `n` GPUs (where `n` can be an integer, or can be a `float`
                for fractional allocation).
                When given as a string "max", then the available GPUs
                across the entire ray cluster (or within the local computer
                in the simplest cases) will be equally distributed among
                the actors.
                When given as a string "all", then each actor will have
                access to all the GPUs (this will be achieved by suppressing
                the environment variable `CUDA_VISIBLE_DEVICES` for each
                actor).
                When the problem is not distributed (i.e. when there are
                no actors), this argument is expected to be left as None.
            num_subbatches: If `num_subbatches` is None (assuming that
                `subbatch_size` is also None), then, when evaluating a
                population, the population will be split into n pieces, `n`
                being the number of actors, and each actor will evaluate
                its assigned piece. If `num_subbatches` is an integer `m`,
                then the population will be split into `m` pieces,
                and actors will continually accept the next unevaluated
                piece as they finish their current tasks.
                The arguments `num_subbatches` and `subbatch_size` cannot
                be given values other than None at the same time.
                While using a distributed algorithm, this argument determines
                how many sub-batches will be generated, and therefore,
                how many gradients will be computed by the remote actors.
            subbatch_size: If `subbatch_size` is None (assuming that
                `num_subbatches` is also None), then, when evaluating a
                population, the population will be split into `n` pieces, `n`
                being the number of actors, and each actor will evaluate its
                assigned piece. If `subbatch_size` is an integer `m`,
                then the population will be split into pieces of size `m`,
                and actors will continually accept the next unevaluated
                piece as they finish their current tasks.
                When there can be significant difference across the solutions
                in terms of computational requirements, specifying a
                `subbatch_size` can be beneficial, because, while one
                actor is busy with a subbatch containing computationally
                challenging solutions, other actors can accept more
                tasks and save time.
                The arguments `num_subbatches` and `subbatch_size` cannot
                be given values other than None at the same time.
                While using a distributed algorithm, this argument determines
                the size of a sub-batch (or sub-population) sampled by a
                remote actor for computing a gradient.
                In distributed mode, it is expected that the population size
                is divisible by `subbatch_size`.
            store_solution_stats: Whether or not the problem object should
                keep track of the best and worst solutions.
                Can also be left as None (which is the default behavior),
                in which case, it will store the best and worst solutions
                only when the first solution batch it encounters is on the
                cpu. This default behavior is to ensure that there is no
                transfer between the cpu and a foreign computation device
                (like the gpu) just for the sake of keeping the best and
                the worst solutions.
            vectorized: Set this to True if the provided fitness function
                is vectorized but is not decorated via `@vectorized`.
        """

        # Set the dtype for the decision variables of the Problem
        if dtype is None:
            self._dtype = torch.float32
        elif is_dtype_object(dtype):
            self._dtype = object
        else:
            self._dtype = to_torch_dtype(dtype)

        _evolog.info(message_from(self, f"The `dtype` for the problem's decision variables is set as {self._dtype}"))

        # Set the dtype for the solution evaluations (i.e. fitnesses and evaluation data)
        if eval_dtype is not None:
            # If an `eval_dtype` is explicitly stated, then accept it as the `_eval_dtype` of the Problem
            self._eval_dtype = to_torch_dtype(eval_dtype)
        else:
            # This is the case where an `eval_dtype` is not explicitly stated by the user.
            # We need to choose a default.
            if self._dtype in (torch.float16, torch.bfloat16, torch.float64):
                # If the `dtype` of the problem is a non-32-bit float type (i.e. float16, bfloat16, float64)
                # then we use that as our `_eval_dtype` as well.
                self._eval_dtype = self._dtype
            else:
                # For any other `dtype`, we use float32 as our `_eval_dtype`.
                self._eval_dtype = torch.float32

            _evolog.info(
                message_from(
                    self, f"`eval_dtype` (the dtype of the fitnesses and evaluation data) is set as {self._eval_dtype}"
                )
            )

        # Set the main device of the Problem object
        self._device = torch.device("cpu") if device is None else torch.device(device)
        _evolog.info(message_from(self, f"The `device` of the problem is set as {self._device}"))

        # Declare the internal variable that might store the random number generator
        self._generator: Optional[torch.Generator] = None

        # Set the seed of the Problem object, if a seed is provided
        self.manual_seed(seed)

        # Declare the internal variables that will store the bounds and the solution length
        self._initial_lower_bounds: Optional[torch.Tensor] = None
        self._initial_upper_bounds: Optional[torch.Tensor] = None
        self._lower_bounds: Optional[torch.Tensor] = None
        self._upper_bounds: Optional[torch.Tensor] = None
        self._solution_length: Optional[int] = None

        if self._dtype is object:
            # If dtype is given as `object`, then there are some runtime sanity checks to perform
            if bounds is not None or initial_bounds is not None:
                # With dtype as object, if bounds are given then we raise an error.
                # This is because the `object` dtype implies that the decision values are not necessarily numeric,
                # and therefore, we cannot have the guarantee of satisfying numeric bounds.
                raise ValueError(
                    f"With dtype as {repr(dtype)}, expected to receive `initial_bounds` and/or `bounds` as None."
                    f" However, one or both of them is/are set as value(s) other than None."
                )
            if solution_length is not None:
                # With dtype as object, if `solution_length` is provided, then we raise an error.
                # This is because the `object` dtype implies that the solutions can be expressed via various
                # containers, each with its own length, and therefore, a fixed solution length cannot be guaranteed.
                raise ValueError(
                    f"With dtype as {repr(dtype)}, expected to receive `solution_length` as None."
                    f" However, received `solution_length` as {repr(solution_length)}."
                )
            if str(self._device) != "cpu":
                # With dtype as object, if `device` is something other than "cpu", then we raise an error.
                # This is because the `object` dtype implies that the decision values are stored by an ObjectArray,
                # whose device is always "cpu".
                raise ValueError(
                    f"With dtype as {repr(dtype)}, expected to receive `device` as 'cpu'."
                    f" However, received `device` as {repr(device)}."
                )
        else:
            # If dtype is something other than `object`, then we need to make sure that we have a valid length for
            # solutions, and also properly store the given numeric bounds.

            if solution_length is None:
                # With a numeric dtype, if solution length is missing, then we raise an error.
                raise ValueError(
                    f"Together with a numeric dtype ({repr(dtype)}),"
                    f" expected to receive `solution_length` as an integer."
                    f" However, `solution_length` is None."
                )
            else:
                # With a numeric dtype, if a solution length is provided, we make sure that it is integer.
                solution_length = int(solution_length)

            # Store the solution length
            self._solution_length = solution_length

            if (bounds is not None) or (initial_bounds is not None):
                # This is the case where we have a dtype other than `object`, and either `bounds` or `initial_bounds`
                # was provided.
                initbnd_tuple_name = "initial_bounds"
                bnd_tuple_name = "bounds"

                if (bounds is not None) and (initial_bounds is None):
                    # With a numeric dtype, if strict bounds are given but initial bounds are not given, then we assume
                    # that the strict bounds also serve as the initial bounds.
                    # Therefore, we take clones of the strict bounds and use this clones as the initial bounds.
                    initial_bounds = clone(bounds)
                    initbnd_tuple_name = "bounds"

                # Below is an internal helper function for some common operations for the (strict) bounds
                # and for the initial bounds.
                def process_bounds(bounds_tuple: BoundsPairLike, tuple_name: str) -> BoundsPair:
                    # This function receives the bounds_tuple (a tuple containing lower and upper bounds),
                    # and the string name of the bounds argument ("bounds" or "initial_bounds").
                    # What is returned is the bounds expressed as PyTorch tensors in the correct dtype and device.

                    nonlocal solution_length

                    # Extract the lower and upper bounds from the received bounds tuple.
                    lb, ub = bounds_tuple

                    # Make sure that the lower and upper bounds are expressed as tensors of correct dtype and device.
                    lb = self.make_tensor(lb)
                    ub = self.make_tensor(ub)

                    for bound_array in (lb, ub):  # For each boundary tensor (lb and ub)
                        if bound_array.ndim not in (0, 1):
                            # If the boundary tensor is not as scalar and is not a 1-dimensional vector, then raise an
                            # error.
                            raise ValueError(
                                f"Lower and upper bounds are expected as scalars or as 1-dimensional vectors."
                                f" However, these given boundaries have incompatible shape:"
                                f" {bound_array} (of shape {bound_array.shape})."
                            )
                        if bound_array.ndim == 1:
                            if len(bound_array) != solution_length:
                                # In the case where the boundary tensor is a 1-dimensional vector, if this vector's length
                                # is not equal to the solution length, then we raise an error.
                                raise ValueError(
                                    f"When boundaries are expressed as 1-dimensional vectors, their length are"
                                    f" expected as the solution length of the Problem object."
                                    f" However, while the problem's solution length is {solution_length},"
                                    f" these given boundaries have incompatible length:"
                                    f" {bound_array} (of length {len(bound_array)})."
                                )

                    # Return the processed forms of the lower and upper boundary tensors.
                    return lb, ub

                # Process the initial bounds with the help of the internal function `process_bounds(...)`
                init_lb, init_ub = process_bounds(initial_bounds, initbnd_tuple_name)

                # Store the processed initial bounds
                self._initial_lower_bounds = init_lb
                self._initial_upper_bounds = init_ub

                if bounds is not None:
                    # If there are strict bounds, then process those bounds with the help of `process_bounds(...)`.
                    lb, ub = process_bounds(bounds, bnd_tuple_name)
                    # Store the processed bounds
                    self._lower_bounds = lb
                    self._upper_bounds = ub

        # Annotate the variable that will store the objective sense(s) of the problem
        self._objective_sense: ObjectiveSense

        # Below is an internal function which makes sure that a provided objective sense has a valid value
        # (where valid values are "min" or "max")
        def validate_sense(s: str):
            if s not in ("min", "max"):
                raise ValueError(
                    f"Invalid objective sense: {repr(s)}."
                    f"Instead, please provide the objective sense as 'min' or 'max'."
                )

        if not is_sequence(objective_sense):
            # If the provided objective sense is not a sequence, then convert it to a single-element list
            senses = [objective_sense]
            num_senses = 1
        else:
            # If the provided objective sense is a sequence, then take a list copy of it
            senses = list(objective_sense)
            num_senses = len(objective_sense)

            # Ensure that each provided objective sense is valid
            for sense in senses:
                validate_sense(sense)

            if num_senses == 0:
                # If the given sequence of objective senses is empty, then we raise an error.
                raise ValueError(
                    "Encountered an empty sequence via `objective_sense`."
                    " For a single-objective problem, please set `objective_sense` as 'min' or 'max'."
                    " For a multi-objective problem, please set `objective_sense` as a sequence,"
                    " each element being 'min' or 'max'."
                )

        # Store the objective senses
        self._senses: Iterable[str] = senses

        # Store the provided objective function (which can be None)
        self._objective_func: Optional[Callable] = objective_func

        # Declare the instance variable that will store whether or not the external fitness function is
        # vectorized, if such an external fitness function is given.
        self._vectorized: Optional[bool]

        # Store the information which indicates whether or not the given objective function is vectorized
        if self._objective_func is None:
            # This is the case where an external fitness function is not given.
            # In this case, we expect the keyword argument `vectorized` to be left as None.

            if vectorized is not None:
                # If the keyword argument `vectorized` is something other than None, then we raise an error
                # to let the user know.
                raise ValueError(
                    f"This problem object received no external fitness function."
                    f" When not using an external fitness function, the keyword argument `vectorized`"
                    f" is expected to be left as None."
                    f" However, the value of the keyword argument `vectorized` is {vectorized}."
                )

            # At this point, we know that we do not have an external fitness function.
            # The variable which is supposed to tell us whether or not the external fitness function is vectorized
            # is therefore irrelevant. We just set it as None.
            self._vectorized = None
        else:
            # This is the case where an external fitness function is given.

            if (
                hasattr(self._objective_func, "__evotorch_vectorized__")
                and self._objective_func.__evotorch_vectorized__
            ):
                # If the external fitness function has an attribute `__evotorch_vectorized__`, and the value of this
                # attribute evaluates to True, then this is an indication that the fitness function was decorated
                # with `@vectorized`.

                if vectorized is not None:
                    # At this point, we know (or at least have the assumption) that the fitness function was decorated
                    # with `@vectorized`. Any boolean value given via the keyword argument `vectorized` would therefore
                    # be either redundant or conflicting.
                    # Therefore, in this case, if the keyword argument `vectorized` is anything other than None, we
                    # raise an error to inform the user.

                    raise ValueError(
                        f"Received a fitness function that was decorated via @vectorized."
                        f" When using such a fitness function, the keyword argument `vectorized`"
                        f" is expected to be left as None."
                        f" However, the value of the keyword argument `vectorized` is {vectorized}."
                    )

                # Since we know that our fitness function declares itself as vectorized, we set the instance variable
                # _vectorized as True.
                self._vectorized = True
            else:
                # This is the case in which the fitness function does not appear to be decorated via `@vectorized`.
                # In this case, if the keyword argument `vectorized` has a value that is equivalent to True,
                # then the value of `_vectorized` becomes True. On the other hand, if the keyword argument `vectorized`
                # was left as None or if it has a value that is equivalent to False, `_vectorized` becomes False.
                self._vectorized = bool(vectorized)

        # If the evaluation data length is explicitly stated, then convert it to an integer and store it.
        # Otherwise, store the evaluation data length as 0.
        self._eval_data_length = 0 if eval_data_length is None else int(eval_data_length)

        # Initialize the actor index.
        # If the problem is configured to be parallelized and the parallelization is triggered, then each remote
        # copy will have a different integer value for `_actor_index`.
        self._actor_index: Optional[int] = None

        # Initialize the variable that might store the list of actors as None.
        # If the problem is configured to be parallelized and the parallelization is triggered, then this variable
        # will store references to the remote actors (each remote actor storing its own copy of this Problem
        # instance).
        self._actors: Optional[list] = None

        # Initialize the variable that might store the ray ActorPool.
        # If the problem is configured to be parallelized and the parallelization is triggered, then this variable
        # will store the ray ActorPool that is generated out of the remote actors.
        self._actor_pool: Optional[ActorPool] = None

        # Store the ray actor configuration dictionary provided by the user (if any).
        # When (or if) the parallelization is triggered, each actor will be created with this given configuration.
        self._actor_config: Optional[dict] = None if actor_config is None else deepcopy(dict(actor_config))

        # If given, store the sub-batch size or number of sub-batches.
        # When the problem is parallelized, a sub-batch size determines the maximum size for a SolutionBatch
        # that will be sent to a remote actor for parallel solution evaluation.
        # Alternatively, num_subbatches determines into how many pieces will a SolutionBatch be split
        # for parallelization.
        # If both are None, then the main SolutionBatch will be split among the actors.
        if (num_subbatches is not None) and (subbatch_size is not None):
            raise ValueError(
                f"Encountered both `num_subbatches` and `subbatch_size` as values other than None."
                f" num_subbatches={num_subbatches}, subbatch_size={subbatch_size}."
                f" Having both of them as values other than None cannot be accepted."
            )
        self._num_subbatches: Optional[int] = None if num_subbatches is None else int(num_subbatches)
        self._subbatch_size: Optional[int] = None if subbatch_size is None else int(subbatch_size)

        # Initialize the additional states to be loaded by the remote actor as None.
        # If there are such additional states for remote actors, the inheriting class can fill this as a list
        # of dictionaries.
        self._remote_states: Optional[Iterable[dict]] = None

        # Initialize a temporary internal variable which stores the resources available in the ray cluster.
        # Most probably, we are interested in the resources "CPU" and "GPU".
        ray_resources: Optional[dict] = None

        # The following is an internal helper function which returns the amount of availability for a given
        # resource in the ray cluster.
        # If the requested resource is not available at all, None will be returned.
        def get_ray_resource(resource_name: str) -> Any:
            # Ensure that the ray cluster is initialized
            ensure_ray()
            nonlocal ray_resources
            if ray_resources is None:
                # If the ray resource information was not fetched, then fetch them and store them.
                ray_resources = ray.available_resources()
            # Return the information regarding the requested resource from the fetched resource information.
            # If it turns out that the requested resource is not available at all, the result will be None.
            return ray_resources.get(resource_name, None)

        # Annotate the variable that will store the number of actors (to be created when the parallelization
        # is triggered).
        self._num_actors: int

        if num_actors is None:
            # If the argument `num_actors` is left as None, then we set `_num_actors` as 0, which means that
            # there will be no parallelization.
            self._num_actors = 0
        elif isinstance(num_actors, str):
            # This is the case where `num_actors` has a string value
            if num_actors in ("max", "num_cpus"):
                # If the `num_actors` argument was given as "max" or as "num_cpus", then we first read how many CPUs
                # are available in the ray cluster, then convert it to integer (via computing its ceil value), and
                # finally set `_num_actors` as this integer.
                self._num_actors = math.ceil(get_ray_resource("CPU"))
            elif num_actors == "num_gpus":
                # If the `num_actors` argument was given as "num_gpus", then we first read how many GPUs are
                # available in the ray cluster.
                num_gpus = get_ray_resource("GPU")
                if num_gpus is None:
                    # If there are no GPUs at all, then we raise an error
                    raise ValueError(
                        "The argument `num_actors` was encountered as 'num_gpus'."
                        " However, there does not seem to be any GPU available."
                    )
                if num_gpus < 1e-4:
                    # If the number of available GPUs are 0 or close to 0, then we raise an error
                    raise ValueError(
                        f"The argument `num_actors` was encountered as 'num_gpus'."
                        f" However, the number of available GPUs are either 0 or close to 0 (= {num_gpus})."
                    )
                if (actor_config is not None) and ("num_gpus" in actor_config):
                    # With `num_actors` argument given as "num_gpus", we will also allocate each GPU to an actor.
                    # If `actor_config` contains an item with key "num_gpus", then that configuration item would
                    # conflict with the GPU allocation we are about to do here.
                    # So, we raise an error.
                    raise ValueError(
                        "The argument `num_actors` was encountered as 'num_gpus'."
                        " With this configuration, the number of GPUs assigned to an actor is automatically determined."
                        " However, at the same time, the `actor_config` argument was received with the key 'num_gpus',"
                        " which causes a conflict."
                    )
                if num_gpus_per_actor is not None:
                    # With `num_actors` argument given as "num_gpus", we will also allocate each GPU to an actor.
                    # If the argument `num_gpus_per_actor` is also stated, then such a configuration item would
                    # conflict with the GPU allocation we are about to do here.
                    # So, we raise an error.
                    raise ValueError(
                        f"The argument `num_actors` was encountered as 'num_gpus'."
                        f" With this configuration, the number of GPUs assigned to an actor is automatically determined."
                        f" However, at the same time, the `num_gpus_per_actor` argument was received with a value other"
                        f" than None ({repr(num_gpus_per_actor)}), which causes a conflict."
                    )
                # Set the number of actors as the ceiled integer counterpart of the number of available GPUs
                self._num_actors = math.ceil(num_gpus)
                # We assign a GPU for each actor (by overriding the value for the argument `num_gpus_per_actor`).
                num_gpus_per_actor = num_gpus / self._num_actors
            elif num_actors == "num_devices":
                # This is the case where `num_actors` has the string value "num_devices".

                # With `num_actors` set as "num_devices", if there are any GPUs, the behavior is to assign a GPU
                # to each actor. If there are conflicting configurations regarding how many GPUs are to be assigned
                # to each actor, then we raise an error.
                if (actor_config is not None) and ("num_gpus" in actor_config):
                    raise ValueError(
                        "The argument `num_actors` was encountered as 'num_devices'."
                        " With this configuration, the number of GPUs assigned to an actor is automatically determined."
                        " However, at the same time, the `actor_config` argument was received with the key 'num_gpus',"
                        " which causes a conflict."
                    )
                if num_gpus_per_actor is not None:
                    raise ValueError(
                        f"The argument `num_actors` was encountered as 'num_devices'."
                        f" With this configuration, the number of GPUs assigned to an actor is automatically determined."
                        f" However, at the same time, the `num_gpus_per_actor` argument was received with a value other"
                        f" than None ({repr(num_gpus_per_actor)}), which causes a conflict."
                    )

                if self._device != torch.device("cpu"):
                    # If the main device is not CPU, then the user most probably wishes to put all the
                    # computations (both evaluations and the population) on the GPU, without allocating
                    # any actor.
                    # So, we set `_num_actors` as None, and overwrite `num_gpus_per_actor` with None.
                    self._num_actors = None
                    num_gpus_per_actor = None
                else:
                    # If the device argument is "cpu" or left as None, then we assume that actor allocations
                    # might be desired.

                    # Read how many CPUs and GPUs are available in the ray cluster.
                    num_cpus = get_ray_resource("CPU")
                    num_gpus = get_ray_resource("GPU")

                    # If we have multiple CPUs, then we continue with the actor allocation procedures.
                    if (num_gpus is None) or (num_gpus < 1e-4):
                        # If there are no GPUs, then we set the number of actors as the number of CPUs, and we
                        # set the number of GPUs per actor as None (which means that there will be no GPU
                        # assignment)
                        self._num_actors = math.ceil(num_cpus)
                        num_gpus_per_actor = None
                    else:
                        # If there are GPUs available, then we compute the minimum among the number of CPUs and
                        # GPUs, and this minimum value becomes the number of actors (so that there can be
                        # one-to-one mapping between actors and GPUs).
                        self._num_actors = math.ceil(min(num_cpus, num_gpus))

                        # We assign a GPU for each actor (by overriding the value for the argument
                        # `num_gpus_per_actor`).
                        if self._num_actors <= num_gpus:
                            num_gpus_per_actor = 1
                        else:
                            num_gpus_per_actor = num_gpus / self._num_actors
            else:
                # This is the case where `num_actors` is given as an unexpected string. We raise an error here.
                raise ValueError(
                    f"Invalid string value for `num_actors`: {repr(num_actors)}."
                    f" The acceptable string values for `num_actors` are 'max', 'num_cpus', 'num_gpus', 'num_devices'."
                )
        else:
            # This is the case where `num_actors` has a value which is not a string.
            # In this case, we make sure that the given value is an integer, and then use this integer as our
            # number of actors.
            self._num_actors = int(num_actors)

        if self._num_actors == 1:
            _evolog.info(
                message_from(
                    self,
                    (
                        "The number of actors that will be allocated for parallelized evaluation was encountered as 1."
                        " This number is automatically dropped to 0,"
                        " because having only 1 actor does not bring any benefit in terms of parallelization."
                    ),
                )
            )
            # Creating a single actor does not bring any benefit of parallelization.
            # Therefore, at the end of all the computations above regarding the number of actors, if it turns out
            # that the target number of actors is 1, we reduce it to 0 (meaning that no actor will be initialized).
            self._num_actors = 0

            # Since we are to allocate no actor, the value of the argument `num_gpus_per_actor` is meaningless.
            # We therefore overwrite the value of that argument with None.
            num_gpus_per_actor = None

        _evolog.info(
            message_from(
                self, f"The number of actors that will be allocated for parallelized evaluation is {self._num_actors}"
            )
        )

        if (self._num_actors >= 2) and (self._device != torch.device("cpu")):
            detailed_error_msg = (
                f"The number of actors that will be allocated for parallelized evaluation is {self._num_actors}."
                " When the number of actors is at least 2,"
                ' the only supported value for the `device` argument is "cpu".'
                f" However, `device` was received as {self._device}."
                "\n\n---- Possible ways to fix the error: ----"
                "\n\n"
                "(1)"
                " If both the population and the fitness evaluation operations can fit into the same device,"
                f" try setting `device={self._device}` and `num_actors=0`."
                "\n\n"
                "(2)"
                " If you would like to use N number of GPUs in parallel for fitness evaluation (where N>1),"
                ' set `device="cpu"` (so that the main process will keep the population on the cpu), set'
                " `num_actors=N` and `num_gpus_per_actor=1` (to allocate an actor for each of the `N` GPUs),"
                " and then, decorate your fitness function using `evotorch.decorators.on_cuda`"
                " so that the fitness evaluation will be performed on the cuda device assigned to the actor."
                " The code for achieving this can look like this:"
                "\n\n"
                "    from evotorch import Problem\n"
                "    from evotorch.decorators import on_cuda, vectorized\n"
                "    import torch\n"
                "\n"
                "    @on_cuda\n"
                "    @vectorized\n"
                "    def f(x: torch.Tensor) -> torch.Tensor:\n"
                "        ...\n"
                "\n"
                '    problem = Problem("min", f, device="cpu", num_actors=N, num_gpus_per_actor=1)\n'
                "\n"
                "Or, it can look like this:\n"
                "\n"
                "    from evotorch import Problem, SolutionBatch\n"
                "    from evotorch.decorators import on_cuda\n"
                "    import torch\n"
                "\n"
                "    class MyProblem(Problem):\n"
                "        def __init__(self, ...):\n"
                "            super().__init__(\n"
                '                objective_sense="min", device="cpu", num_actors=N, num_gpus_per_actor=1, ...\n'
                "            )\n"
                "\n"
                "        @on_cuda\n"
                "        def _evaluate_batch(self, batch: SolutionBatch):\n"
                "            ...\n"
                "\n"
                "    problem = MyProblem(...)\n"
                "\n"
                "\n"
                "(3)"
                " Similarly to option (2), for when you wish to use N number of GPUs for fitness evaluation,"
                ' set `device="cpu"`, set `num_actors=N` and `num_gpus_per_actor=1`, then, within the evaluation'
                ' function, manually use the device `"cuda"` to accelerate the computation.'
                "\n\n"
                "--------------\n"
                "Note for cases (2) and (3): if you are on a computer or on a ray cluster with multiple GPUs, you"
                ' might prefer to set `num_actors` as the string "num_gpus" instead of an integer N,'
                " which will cause the number of available GPUs to be counted, and the number of actors to be"
                " configured as that count."
            )

            raise ValueError(detailed_error_msg)

        # Annotate the variable which will determine how many GPUs are to be assigned to each actor.
        self._num_gpus_per_actor: Optional[Union[str, int, float]]

        if (actor_config is not None) and ("num_gpus" in actor_config) and (num_gpus_per_actor is not None):
            # If `actor_config` dictionary has the item "num_gpus" and also `num_gpus_per_actor` is not None,
            # then there is a conflicting (or redundant) configuration. We raise an error here.
            raise ValueError(
                'The `actor_config` dictionary contains the key "num_gpus".'
                " At the same time, `num_gpus_per_actor` has a value other than None."
                " These two configurations are conflicting."
                " Please specify the number of GPUs per actor either via the `actor_config` dictionary,"
                " or via the `num_gpus_per_actor` argument, but not via both."
            )

        if num_gpus_per_actor is None:
            # If the argument `num_gpus_per_actor` is not specified, then we set the attribute
            # `_num_gpus_per_actor` as None, which means that no GPUs will be assigned to the actors.
            self._num_gpus_per_actor = None
        elif isinstance(num_gpus_per_actor, str):
            # This is the case where `num_gpus_per_actor` is given as a string.
            if num_gpus_per_actor == "max":
                # This is the case where `num_gpus_per_actor` is given as "max".
                num_gpus = get_ray_resource("GPU")
                if num_gpus is None:
                    # With `num_gpus_per_actor` as "max", if there is no GPU available, then we set the attribute
                    # `_num_gpus_per_actor` as None, which means there will be no GPU assignment to the actors.
                    self._num_gpus_per_actor = None
                else:
                    # With `num_gpus_per_actor` as "max", if there are GPUs available, then the available GPUs will
                    # be shared among the actors.
                    self._num_gpus_per_actor = num_gpus / self._num_actors
            elif num_gpus_per_actor == "all":
                # When `num_gpus_per_actor` is "all", we also set the attribute `_num_gpus_per_actor` as "all".
                # When a remote actor is initialized, the remote actor will see that the Problem instance has its
                # `_num_gpus_per_actor` set as "all", and it will remove the environment variable named
                # "CUDA_VISIBLE_DEVICES" in its own environment.
                # With "CUDA_VISIBLE_DEVICES" removed, an actor will see all the GPUs available in its own
                # environment.
                self._num_gpus_per_actor = "all"
            else:
                # This is the case where `num_gpus_per_actor` argument has an unexpected string value.
                # We raise an error.
                raise ValueError(
                    f"Invalid string value for `num_gpus_per_actor`: {repr(num_gpus_per_actor)}."
                    f' Acceptable string values for `num_gpus_per_actor` are: "max", "all".'
                )
        elif isinstance(num_gpus_per_actor, int):
            # When the argument `num_gpus_per_actor` is set as an integer we just set the attribute
            # `_num_gpus_per_actor` as this integer.
            self._num_gpus_per_actor = num_gpus_per_actor
        else:
            # For anything else, we assume that `num_gpus_per_actor` is an object that is convertible to float.
            # Therefore, we convert it to float and store it in the attribute `_num_gpus_per_actor`.
            # Also, remember that, when `num_actors` is given as "num_gpus" or as "num_devices",
            # the code above overrides the value for the argument `num_gpus_per_actor`, which means,
            # this is the case that is activated when `num_actors` is "num_gpus" or "num_devices".
            self._num_gpus_per_actor = float(num_gpus_per_actor)

        if self._num_actors > 0:
            _evolog.info(
                message_from(self, f"Number of GPUs that will be allocated per actor is {self._num_gpus_per_actor}")
            )

        # Initialize the Hook instances (and the related status dictionary for the `_after_eval_hook`)
        self._before_eval_hook: Hook = Hook()
        self._after_eval_hook: Hook = Hook()
        self._after_eval_status: dict = {}
        self._remote_hook: Hook = Hook()
        self._before_grad_hook: Hook = Hook()
        self._after_grad_hook: Hook = Hook()

        # Initialize various stats regarding the solutions encountered by this Problem instance.
        self._store_solution_stats = None if store_solution_stats is None else bool(store_solution_stats)
        self._best: Optional[list] = None
        self._worst: Optional[list] = None
        self._best_evals: Optional[torch.Tensor] = None
        self._worst_evals: Optional[torch.Tensor] = None

        # Initialize the boolean attribute which indicates whether or not this Problem instance (which can be
        # the main instance or a remote instance on an actor) is "prepared" via the `_prepare` method.
        self._prepared: bool = False

    def manual_seed(self, seed: Optional[int] = None):
        """
        Provide a manual seed for the Problem object.

        If the given seed is None, then the Problem object will remove
        its own stored generator, and start using the global generator
        of PyTorch instead.
        If the given seed is an integer, then the Problem object will
        instantiate its own generator with the given seed.

        Args:
            seed: None for using the global PyTorch generator; an integer
                for instantiating a new PyTorch generator with this given
                integer seed, specific to this Problem object.
        """
        if seed is None:
            self._generator = None
        else:
            if self._generator is None:
                self._generator = torch.Generator(device=self.device)
            self._generator.manual_seed(seed)

    @property
    def dtype(self) -> DType:
        """
        dtype of the Problem object.

        The decision variables of the optimization problem are of this dtype.
        """
        return self._dtype

    @property
    def device(self) -> Device:
        """
        device of the Problem object.

        New solutions and populations will be generated in this device.
        """
        return self._device

    @property
    def aux_device(self) -> Device:
        """
        Auxiliary device to help with the computations, most commonly for
        speeding up the solution evaluations.

        An auxiliary device is different than the main device of the Problem
        object (the main device being expressed by the `device` property).
        While the main device of the Problem object determines where the
        solutions and the populations are stored (and also using which device
        should a SearchAlgorithm instance communicate with the problem),
        an auxiliary device is a device that might be used by the Problem
        instance itself for its own computations (e.g. computations defined
        within the methods `_evaluate(...)` or `_evaluate_batch(...)`).

        If the problem's main device is something other than "cpu", that main
        device is also seen as the auxiliary device, and therefore returned
        by this property.

        If the problem's main device is "cpu", then the auxiliary device
        is decided as follows. If `num_gpus_per_actor` of the Problem object
        was set as "all" and if this instance is a remote instance, then the
        auxiliary device is guessed as "cuda:N" where N is the actor index.
        In all other cases, the auxiliary device is "cuda" if cuda is
        available, and "cpu" otherwise.
        """
        cpu_device = torch.device("cpu")
        if torch.device(self.device) == cpu_device:
            if torch.cuda.is_available():
                if isinstance(self._num_gpus_per_actor, str) and (self._num_gpus_per_actor == "all") and self.is_remote:
                    return torch.device("cuda", self.actor_index)
                else:
                    return torch.device("cuda")
            else:
                return cpu_device
        else:
            return self.device

    @property
    def eval_dtype(self) -> DType:
        """
        evaluation dtype of the Problem object.

        The evaluation results of the solutions are stored according to this
        dtype.
        """
        return self._eval_dtype

    @property
    def generator(self) -> Optional[torch.Generator]:
        """
        Random generator used by this Problem object.

        Can also be None, which means that the Problem object will use the
        global random generator of PyTorch.
        """
        return self._generator

    @property
    def has_own_generator(self) -> bool:
        """
        Whether or not the Problem object has its own random generator.

        If this is True, then the Problem object will use its own
        random generator when creating random values or tensors.
        If this is False, then the Problem object will use the global
        random generator when creating random values or tensors.
        """
        return self.generator is not None

    @property
    def objective_sense(self) -> ObjectiveSense:
        """
        Get the objective sense.

        If the problem is single-objective, then a single string is returned.
        If the problem is multi-objective, then the objective senses will be
        returned in a list.

        The returned string in the single-objective case, or each returned
        string in the multi-objective case, is "min" or "max".
        """
        if len(self.senses) == 1:
            return self.senses[0]
        else:
            return self.senses

    @property
    def senses(self) -> Iterable[str]:
        """
        Get the objective senses.

        The return value is a list of strings, each string being
        "min" or "max".
        """
        return self._senses

    @property
    def is_single_objective(self) -> bool:
        """Whether or not the problem is single-objective"""
        return len(self.senses) == 1

    @property
    def is_multi_objective(self) -> bool:
        """Whether or not the problem is multi-objective"""
        return len(self.senses) > 1

    def get_obj_order_descending(self) -> Iterable[bool]:
        """When sorting the solutions from best to worst according to each objective i, is the ordering descending?"""
        result = []
        for s in self.senses:
            if s == "min":
                result.append(False)
            elif s == "max":
                result.append(True)
            else:
                raise ValueError(f"Invalid sense: {repr(s)}")
        return result

    @property
    def solution_length(self) -> Optional[int]:
        """
        Get the solution length.

        Problems with `dtype=None` do not have solution lengths.
        For such problems, this property returns None.
        """
        return self._solution_length

    @property
    def eval_data_length(self) -> int:
        """
        Length of the extra evaluation data vector for each solution.
        """
        return self._eval_data_length

    @property
    def initial_lower_bounds(self) -> Optional[torch.Tensor]:
        """
        Initial lower bounds, for when initializing a new solution.

        If such a bound was declared during the initialization phase,
        the returned value is a torch tensor (in the form of a vector
        or in the form of a scalar).
        If no such bound was declared, the returned value is None.
        """
        return self._initial_lower_bounds

    @property
    def initial_upper_bounds(self) -> Optional[torch.Tensor]:
        """
        Initial upper bounds, for when initializing a new solution.

        If such a bound was declared during the initialization phase,
        the returned value is a torch tensor (in the form of a vector
        or in the form of a scalar).
        If no such bound was declared, the returned value is None.
        """
        return self._initial_upper_bounds

    @property
    def lower_bounds(self) -> Optional[torch.Tensor]:
        """
        Lower bounds for the allowed values of a solution.

        If such a bound was declared during the initialization phase,
        the returned value is a torch tensor (in the form of a vector
        or in the form of a scalar).
        If no such bound was declared, the returned value is None.
        """
        return self._lower_bounds

    @property
    def upper_bounds(self) -> Optional[torch.Tensor]:
        """
        Upper bounds for the allowed values of a solution.

        If such a bound was declared during the initialization phase,
        the returned value is a torch tensor (in the form of a vector
        or in the form of a scalar).
        If no such bound was declared, the returned value is None.
        """
        return self._upper_bounds

    def generate_values(self, num_solutions: int) -> Union[torch.Tensor, ObjectArray]:
        """
        Generate decision values.

        This function returns a tensor containing the decision values
        for `n` new solutions, `n` being the integer passed as the `num_rows`
        argument.

        For numeric problems, this function generates the decision values
        which respect `initial_bounds` (or `bounds`, if `initial_bounds`
        was not provided).
        If this type of initialization is not desired, one can override
        this function and define a manual initialization scheme in the
        inheriting subclass.

        For non-numeric problems, it is expected that the inheriting subclass
        will override the method `_fill(...)`.

        Args:
            num_solutions: For how many solutions will new decision values be
                generated.
        Returns:
            A PyTorch tensor for numeric problems, an ObjectArray for
            non-numeric problems.
        """
        if self.dtype is object:
            result = self.make_empty(num_solutions=num_solutions)
        else:
            result = torch.empty(tuple(), dtype=self.dtype, device=self.device)
            result = result.expand(num_solutions, self.solution_length)
            result = result + make_batched_false_for_vmap(result.device)
        self._fill(result)
        return result

    def _fill(self, values: Iterable):
        """
        Fill the provided `values` tensor with new decision values.

        Inheriting subclasses can override this method to specialize how
        new solutions are generated.

        For numeric problems, this method already has an implementation
        which samples the initial decision values uniformly from the
        interval expressed by `initial_bounds` attribute.
        For non-numeric problems, overriding this method is mandatory.

        Args:
            values: The tensor which is to be filled with the new decision
                values.
        """
        if self.dtype is object:
            raise NotImplementedError(
                "The dtype of this problem is object, therefore a manual implementation of the"
                " method `_fill(...)` needs to be provided by the inheriting class."
            )
        else:
            if (self.initial_lower_bounds is None) or (self.initial_upper_bounds is None):
                raise RuntimeError(
                    "The default implementation of the method `_fill(...)` does not know how to initialize solutions"
                    " because it appears that this Problem object was not given neither `initial_bounds` nor `bounds`"
                    " during the moment of initialization."
                    " Please either instantiate this Problem object with `initial_bounds` and/or `bounds`, or override"
                    " the method `_fill(...)` to specify how solutions should be initialized."
                )
            else:
                return self.make_uniform(
                    out=values,
                    lb=self.initial_lower_bounds,
                    ub=self.initial_upper_bounds,
                )

    def generate_batch(
        self,
        popsize: Optional[int] = None,
        *,
        empty: bool = False,
        center: Optional[RealOrVector] = None,
        stdev: Optional[RealOrVector] = None,
        symmetric: bool = False,
    ) -> "SolutionBatch":
        """
        Generate a new SolutionBatch.

        Args:
            popsize: Number of solutions that will be contained in the new
                batch.
            empty: Set this as True if you would like to receive the solutions
                un-initialized.
            center: Center point of the Gaussian distribution from which
                the decision values will be sampled, as a scalar or as a
                1-dimensional vector.
                Can also be left as None.
                If `center` is None and `stdev` is None, all the decision
                values will be sampled from the interval specified by
                `initial_bounds` (or by `bounds` if `initial_bounds` was not
                specified).
                If `center` is None and `stdev` is not None, a center point
                will be sampled from within the interval specified by
                `initial_bounds` or `bounds`, and the decision values will be
                sampled from a Gaussian distribution around this center point.
            stdev: Can be None (default) if the SolutionBatch is to contain
                decision values sampled from the interval specified by
                `initial_bounds` (or by `bounds` if `initial_bounds` was not
                provided during the initialization phase).
                Alternatively, a scalar or a 1-dimensional vector specifying
                the standard deviation of the Gaussian distribution from which
                the decision values will be sampled.
            symmetric: To be used only when `stdev` is not None.
                If `symmetric` is True, decision values will be sampled from
                the Gaussian distribution in a symmetric (i.e. antithetic)
                manner.
                Otherwise, the decision values will be sampled in the
                non-antithetic manner.
        """
        if (center is None) and (stdev is None):
            if symmetric:
                raise ValueError(
                    f"The argument `symmetric` can be set as True only when `center` and `stdev` are provided."
                    f" Although `center` and `stdev` are None, `symmetric` was received as {symmetric}."
                )
            return SolutionBatch(self, popsize, empty=empty, device=self.device)
        elif (center is not None) and (stdev is not None):
            if empty:
                raise ValueError(
                    f"When `center` and `stdev` are provided, the argument `empty` must be False."
                    f" However, the received value for `empty` is {empty}."
                )
            result = SolutionBatch(self, popsize, device=self.device, empty=True)
            self.make_gaussian(out=result.access_values(), center=center, stdev=stdev, symmetric=symmetric)
            return result
        else:
            raise ValueError(
                f"The arguments `center` and `stdev` were expected to be None or non-None at the same time."
                f" Received `center`: {center}."
                f" Received `stdev`: {stdev}."
            )

    def _parallelize(self):
        """Create ray actors for parallelizing the solution evaluations."""

        # If the problem was explicitly configured for
        # NOT having parallelization, leave this function.
        if (not isinstance(self._num_actors, str)) and (self._num_actors <= 0):
            return

        # If this problem object is a remote one,
        # leave this function
        # (because we do not want the remote worker
        # to parallelize itself further)
        if self._actor_index is not None:
            return

        # If the actors list is not None, then this means
        # that the initialization of the parallelization mechanism
        # was already completed. So, leave this function.
        if self._actors is not None:
            return

        # Make sure that ray is initialized
        ensure_ray()
        number_of_actors = self._num_actors

        # numpy's RandomState uses 32-bit unsigned integers
        # for random seeds.
        # So, the following value is the exclusive upper bound
        # for a random seed.
        supremum_seed = 2**32

        # Generate an integer from the main problem object's
        # random_state. From this integer, further seed integers
        # will be computed, and these generated seeds will be
        # used by the remote actors.
        base_seed = int(self.make_randint(tuple(), n=supremum_seed))

        # The following function returns a seed number for the actor
        # number i.
        def generate_actor_seed(i):
            nonlocal base_seed, supremum_seed
            return (base_seed + (i + 1)) % supremum_seed

        all_seeds = []
        j = 0
        for i in range(number_of_actors):
            actor_seeds = []
            for _ in range(4):
                actor_seeds.append(generate_actor_seed(j))
                j += 1
            all_seeds.append(tuple(actor_seeds))

        if self._remote_states is None:
            remote_states = [{} for _ in range(number_of_actors)]
        else:
            remote_states = self._remote_states

        # Prepare the necessary actor config
        config_per_actor = {}
        if self._actor_config is not None:
            config_per_actor.update(self._actor_config)
        if isinstance(self._num_gpus_per_actor, (int, float)):
            config_per_actor["num_gpus"] = self._num_gpus_per_actor

        # Generate the actors, each with a unique seed.
        if config_per_actor is None:
            actors = [EvaluationActor.remote(self, i, all_seeds[i], remote_states[i]) for i in range(number_of_actors)]
        else:
            actors = [
                EvaluationActor.options(**config_per_actor).remote(self, i, all_seeds[i], remote_states[i])
                for i in range(number_of_actors)
            ]

        self._actors = actors
        self._actor_pool = ActorPool(self._actors)
        self._remote_states = None

    def all_remote_problems(self) -> AllRemoteProblems:
        """
        Get an accessor which is used for running a method
        on all remote clones of this Problem object.

        For example, given a Problem object named `my_problem`,
        also assuming that this Problem object is parallelized,
        and therefore has `n` remote actors, a method `f()`
        can be executed on all the remote instances as follows:

            results = my_problem.all_remote_problems().f()

        The variable `results` is a list of length `n`, the i-th
        item of the list belonging to the method f's result
        from the i-th actor.

        Returns:
            A method accessor for all the remote Problem objects.
        """
        self._parallelize()
        if self.is_remote:
            raise RuntimeError(
                "The method `all_remote_problems()` can only be used on the main (i.e. non-remote)"
                " Problem instance."
                " However, this Problem instance is on a remote actor."
            )
        return AllRemoteProblems(self._actors)

    def all_remote_envs(self) -> AllRemoteEnvs:
        """
        Get an accessor which is used for running a method
        on all remote reinforcement learning environments.

        This method can only be used on parallelized Problem
        objects which have their `get_env()` methods defined.
        For example, one can use this feature on a parallelized
        GymProblem.

        As an example, let us consider a parallelized GymProblem
        object named `my_problem`. Given that `my_problem` has
        `n` remote actors, a method `f()` can be executed
        on all remote reinforcement learning environments as
        follows:

            results = my_problem.all_remote_envs().f()

        The variable `results` is a list of length `n`, the i-th
        item of the list belonging to the method f's result
        from the i-th actor.

        Returns:
            A method accessor for all the remote reinforcement
            learning environments.
        """
        self._parallelize()
        if self.is_remote:
            raise RuntimeError(
                "The method `all_remote_envs()` can only be used on the main (i.e. non-remote)"
                " Problem instance."
                " However, this Problem instance is on a remote actor."
            )
        return AllRemoteEnvs(self._actors)

    def kill_actors(self):
        """
        Kill all the remote actors used by the Problem instance.

        One might use this method to release the resources used by the
        remote actors.
        """
        if not self.is_main:
            raise RuntimeError(
                "The method `kill_actors()` can only be used on the main (i.e. non-remote)"
                " Problem instance."
                " However, this Problem instance is on a remote actor."
            )
        for actor in self._actors:
            ray.kill(actor)
        self._actors = None
        self._actor_pool = None

    @property
    def num_actors(self) -> int:
        """
        Number of actors (to be) used for parallelization.
        If the problem is configured for no parallelization,
        the result will be 0.
        """
        return self._num_actors

    @property
    def actors(self) -> Optional[list]:
        """
        Get the ray actors, if the Problem object is distributed.
        If the Problem object is not distributed and therefore
        has no actors, then, the result will be None.
        """
        return self._actors

    @property
    def actor_index(self) -> Optional[int]:
        """Return the actor index if this is a remote worker.
        If this is not a remote worker, return None.
        """
        return self._actor_index

    @property
    def is_remote(self) -> bool:
        """Returns True if this problem object lives in a remote ray actor.
        Otherwise, returns False.
        """
        return self._actor_index is not None

    @property
    def is_main(self) -> bool:
        """Returns True if this problem object lives in the main process
        and not in a remote actor.
        Otherwise, returns False.
        """
        return self._actor_index is None

    @property
    def before_eval_hook(self) -> Hook:
        """
        Get the Hook which stores the functions to call just before
        evaluating a SolutionBatch.

        The functions to be stored in this hook are expected to
        accept one positional argument, that one argument being the
        SolutionBatch which is about to be evaluated.
        """
        return self._before_eval_hook

    @property
    def after_eval_hook(self) -> Hook:
        """
        Get the Hook which stores the functions to call just after
        evaluating a SolutionBatch.

        The functions to be stored in this hook are expected to
        accept one argument, that one argument being the SolutionBatch
        whose evaluation has just been completed.

        The dictionaries returned by the functions in this hook
        are accumulated, and reported in the status dictionary of this
        problem object.
        """
        return self._after_eval_hook

    @property
    def before_grad_hook(self) -> Hook:
        """
        Get the Hook which stores the functions to call just before
        its `sample_and_compute_gradients(...)` operation.
        """
        return self._before_grad_hook

    @property
    def after_grad_hook(self) -> Hook:
        """
        Get the Hook which stores the functions to call just after
        its `sample_and_compute_gradients(...)` operation.

        The functions to be stored in this hook are expected to
        accept one argument, that one argument being the gradients
        dictionary (which was produced by the Problem object,
        but not yet followed by the search algorithm).

        The dictionaries returned by the functions in this hook
        are accumulated, and reported in the status dictionary of this
        problem object.
        """
        return self._after_grad_hook

    @property
    def remote_hook(self) -> Hook:
        """
        Get the Hook which stores the functions to call when this
        Problem object is (re)created on a remote actor.

        The functions in this hook should expect one positional
        argument, that is the Problem object itself.
        """
        return self._remote_hook

    def _make_sync_data_for_actors(self) -> Any:
        """
        Override this function for providing synchronization between
        the main process and the remote actors.

        The responsibility of this function is to prepare and return the
        data to be sent to the remote actors for synchronization.

        If this function returns NotImplemented, then there will be no
        syncing.
        If this function returns None, there will be no data sent to the
        actors for syncing, however, syncing will still be enabled, and
        the main actor will ask for sync data from the remote actors
        after their jobs are finished.
        """
        return NotImplemented

    def _use_sync_data_from_main(self, received: Any):
        """
        Override this function for providing synchronization between
        the main process and the remote actors.

        The responsibility of this function is to update the state
        of the remote Problem object according to the synchronization
        data received by the main process.
        """
        pass

    def _make_sync_data_for_main(self) -> Any:
        """
        Override this function for providing synchronization between
        the main process and the remote actors.

        The responsibility of this function is to prepare and return the
        data to be sent to the main Problem object by a remote actor.
        """
        return NotImplemented

    def _use_sync_data_from_actors(self, received: list):
        """
        Override this function for providing synchronization between
        the main process and the remote actors.

        The responsibility of this function is to update the state
        of the main Problem object according to the synchronization
        data received by the remote actors.
        """
        pass

    def _make_pickle_data_for_main(self) -> dict:
        """
        Override this function for preserving the state of a remote
        actor in the main state dictionary when pickling a parallelized
        problem.

        The responsibility of this function is to return the state
        of a problem object which lives in a remote actor.

        If the remote clones of this problem do not need to be stateful
        then you probably do not need to override this method.
        """
        return {}

    def _use_pickle_data_from_main(self, state: dict):
        """
        Override this function for re-creating the internal state of
        a problem instance living in a remote actor, by using the
        given state dictionary.

        If the remote clones of this problem do not need to be stateful
        then you probably do not need to override this method.
        """
        pass

    def _sync_before(self) -> bool:
        if self._actors is None:
            return False

        to_send = self._make_sync_data_for_actors()
        if to_send is NotImplemented:
            return False

        if to_send is not None:
            ray.get([actor.call.remote("_use_sync_data_from_main", [to_send], {}) for actor in self._actors])

        return True

    def _sync_after(self):
        if self._actors is None:
            return

        received = ray.get([actor.call.remote("_make_sync_data_for_main", [], {}) for actor in self._actors])

        self._use_sync_data_from_actors(received)

    @torch.no_grad()
    def _get_best_and_worst(self, batch: "SolutionBatch") -> Optional[dict]:
        if self._store_solution_stats is None:
            self._store_solution_stats = str(batch.device) == "cpu"

        if not self._store_solution_stats:
            return {}

        senses = self.senses
        nobjs = len(senses)

        if self._best is None:
            self._best_evals = self.make_empty(nobjs, device=batch.device, use_eval_dtype=True)
            self._worst_evals = self.make_empty(nobjs, device=batch.device, use_eval_dtype=True)
            for i_obj in range(nobjs):
                if senses[i_obj] == "min":
                    self._best_evals[i_obj] = float("inf")
                    self._worst_evals[i_obj] = float("-inf")
                elif senses[i_obj] == "max":
                    self._best_evals[i_obj] = float("-inf")
                    self._worst_evals[i_obj] = float("inf")
                else:
                    raise ValueError(f"Invalid sense: {senses[i_obj]}")
            self._best = [None] * nobjs
            self._worst = [None] * nobjs

        def first_is_better(a, b, i_obj):
            if senses[i_obj] == "min":
                return a < b
            elif senses[i_obj] == "max":
                return a > b
            else:
                raise ValueError(f"Invalid sense: {senses[i_obj]}")

        def first_is_worse(a, b, i_obj):
            if senses[i_obj] == "min":
                return a > b
            elif senses[i_obj] == "max":
                return a < b
            else:
                raise ValueError(f"Invalid sense: {senses[i_obj]}")

        best_sln_indices = [batch.argbest(i) for i in range(nobjs)]
        worst_sln_indices = [batch.argworst(i) for i in range(nobjs)]

        for i_obj in range(nobjs):
            best_sln_index = best_sln_indices[i_obj]
            worst_sln_index = worst_sln_indices[i_obj]
            scores = batch.access_evals(i_obj)
            best_score = scores[best_sln_index]
            worst_score = scores[worst_sln_index]
            if first_is_better(best_score, self._best_evals[i_obj], i_obj):
                self._best_evals[i_obj] = best_score
                self._best[i_obj] = batch[best_sln_index].clone()
            if first_is_worse(worst_score, self._worst_evals[i_obj], i_obj):
                self._worst_evals[i_obj] = worst_score
                self._worst[i_obj] = batch[worst_sln_index].clone()

        if len(senses) == 1:
            return dict(
                best=self._best[0],
                worst=self._worst[0],
                best_eval=float(self._best[0].evals[0]),
                worst_eval=float(self._worst[0].evals[0]),
            )
        else:
            return {"best": self._best, "worst": self._worst}

    def compare_solutions(self, a: "Solution", b: "Solution", obj_index: Optional[int] = None) -> float:
        """
        Compare two solutions.
        It is assumed that both solutions are already evaluated.

        Args:
            a: The first solution.
            b: The second solution.
            obj_index: The objective index according to which the comparison
                will be made.
                Can be left as None if the problem is single-objective.
        Returns:
            A positive number if `a` is better;
            a negative number if `b` is better;
            0 if there is a tie.
        """
        senses = self.senses
        obj_index = self.normalize_obj_index(obj_index)
        sense = senses[obj_index]

        def score(s: Solution):
            return s.evals[obj_index]

        if sense == "max":
            return score(a) - score(b)
        elif sense == "min":
            return score(b) - score(a)
        else:
            raise ValueError("Unrecognized sense: " + repr(sense))

    def is_better(self, a: "Solution", b: "Solution", obj_index: Optional[int] = None) -> bool:
        """
        Check whether or not the first solution is better.
        It is assumed that both solutions are already evaluated.

        Args:
            a: The first solution.
            b: The second solution.
            obj_index: The objective index according to which the comparison
                will be made.
                Can be left as None if the problem is single-objective.
        Returns:
            True if `a` is better; False otherwise.
        """
        return self.compare_solutions(a, b, obj_index) > 0

    def is_worse(self, a: "Solution", b: "Solution", obj_index: Optional[int] = None) -> bool:
        """
        Check whether or not the first solution is worse.
        It is assumed that both solutions are already evaluated.

        Args:
            a: The first solution.
            b: The second solution.
            obj_index: The objective index according to which the comparison
                will be made.
                Can be left as None if the problem is single-objective.
        Returns:
            True if `a` is worse; False otherwise.
        """
        return self.compare_solutions(a, b, obj_index) < 0

    def _prepare(self) -> None:
        """Prepare a worker instance of the problem for evaluation. To be overridden by the user"""
        pass

    def _prepare_main(self) -> None:
        """Prepare the main instance of the problem for evaluation."""
        self._share_attributes()

    def _start_preparations(self) -> None:
        """Prepare the problem for evaluation. Calls self._prepare() if the self._prepared flag is not True."""
        if not self._prepared:
            if self.actors is None or self._num_actors == 0:
                # Call prepare method for any problem class that is expected to do work
                self._prepare()
            if self.is_main:
                # Call share method to distribute shared attributes to actors
                self._prepare_main()

            self._prepared = True

    @property
    def _nonserialized_attribs(self) -> List[str]:
        return []

    def _share_attributes(self) -> None:
        if (self._actors is not None) and (len(self._actors) > 0):
            for attrib_name in self._shared_attribs:
                obj_ref = ray.put(getattr(self, attrib_name))
                for actor in self.actors:
                    actor.call.remote("put_ray_object", [], {"obj_ref": obj_ref, "attrib_name": attrib_name})

    def put_ray_object(self, obj_ref: ray.ObjectRef, attrib_name: str) -> None:
        setattr(self, attrib_name, ray.get(obj_ref))

    @property
    def _shared_attribs(self) -> List[str]:
        return []

    def _device_of_fitness_function(self) -> Optional[Device]:
        def device_of_fn(fn: Optional[Callable]) -> Optional[Device]:
            if fn is None:
                return None
            else:
                if hasattr(fn, "__evotorch_on_aux_device__") and fn.__evotorch_on_aux_device__:
                    return self.aux_device
                elif hasattr(fn, "device"):
                    return fn.device
                else:
                    return None

        for candidate_fn in (self._objective_func, self._evaluate_all, self._evaluate_batch, self._evaluate):
            device = device_of_fn(candidate_fn)
            if device is not None:
                if candidate_fn is self._evaluate_all:
                    raise RuntimeError(
                        "It seems that the `_evaluate_all(...)` method of this Problem object is either decorated"
                        " by @on_aux_device or by @on_device, or it is specifying a target device via a `device`"
                        " attribute. However, these decorators (or the `device` attribute) are not supported in the"
                        " case of `_evaluate_all(...)`."
                        " The reason is that the checking of the target device and the operations of moving the batch"
                        " onto the target device are handled by the default implementation of `_evaluate_all` itself."
                        " To specify a target device, consider decorating `_evaluate_batch(...)` or `_evaluate(...)`"
                        " instead."
                    )
                return device

        return None

    def evaluate(self, x: Union["SolutionBatch", "Solution"]):
        """
        Evaluate the given Solution or SolutionBatch.

        Args:
            x: The SolutionBatch to be evaluated.
        """

        if isinstance(x, Solution):
            batch = x.to_batch()
        elif isinstance(x, SolutionBatch):
            batch = x
        else:
            raise TypeError(
                f"The method `evaluate(...)` expected a Solution or a SolutionBatch as its argument."
                f" However, the received object is {repr(x)}, which is of type {repr(type(x))}."
            )

        self._parallelize()

        if self.is_main:
            self.before_eval_hook(batch)

        must_sync_after = self._sync_before()

        self._start_preparations()

        self._evaluate_all(batch)

        if must_sync_after:
            self._sync_after()

        if self.is_main:
            self._after_eval_status = {}

            best_and_worst = self._get_best_and_worst(batch)
            if best_and_worst is not None:
                self._after_eval_status.update(best_and_worst)

            self._after_eval_status.update(self.after_eval_hook.accumulate_dict(batch))

    def _evaluate_all(self, batch: "SolutionBatch"):
        if self._actors is None:
            fitness_device = self._device_of_fitness_function()
            if fitness_device is None:
                self._evaluate_batch(batch)
            else:
                original_device = batch.device
                moved_batch = batch.to(fitness_device)
                self._evaluate_batch(moved_batch)
                batch.set_evals(moved_batch.evals.to(original_device))
        else:
            if self._num_subbatches is not None:
                pieces = batch.split(self._num_subbatches)
            elif self._subbatch_size is not None:
                pieces = batch.split(max_size=self._subbatch_size)
            else:
                pieces = batch.split(len(self._actors))
            # mapresult = self._actor_pool.map(lambda a, v: a.evaluate_batch.remote(v), list(pieces))
            # for i, evals in enumerate(mapresult):
            #    row_begin, row_end = pieces.indices_of(i)
            #    batch._evdata[row_begin:row_end, :] = evals

            mapresult = self._actor_pool.map_unordered(
                lambda a, v: a.evaluate_batch_piece.remote(v[0], v[1]), list(enumerate(pieces))
            )
            for i, evals in mapresult:
                row_begin, row_end = pieces.indices_of(i)
                batch._evdata[row_begin:row_end, :] = evals

    def _evaluate_batch(self, batch: "SolutionBatch"):
        if self._vectorized and (self._objective_func is not None):
            result = self._objective_func(batch.values)
            if isinstance(result, tuple):
                batch.set_evals(*result)
            else:
                batch.set_evals(result)
        else:
            for sln in batch:
                self._evaluate(sln)

    def _evaluate(self, solution: "Solution"):
        if self._objective_func is not None:
            result = self._objective_func(solution.values)
            if isinstance(result, tuple):
                solution.set_evals(*result)
            else:
                solution.set_evals(result)
        else:
            raise NotImplementedError

    @property
    def stores_solution_stats(self) -> Optional[bool]:
        """
        Whether or not the best and worst solutions are kept.
        """
        return self._store_solution_stats

    @property
    def status(self) -> dict:
        """
        Status dictionary of the problem object, updated after the last
        evaluation operation.

        The dictionaries returned by the functions in `after_eval_hook`
        are accumulated, and reported in this status dictionary.
        """
        return self._after_eval_status

    def ensure_numeric(self):
        """
        Ensure that the problem has a numeric dtype.

        Raises:
            ValueError: if the problem has a non-numeric dtype.
        """
        if is_dtype_object(self.dtype):
            raise ValueError("Expected a problem with numeric dtype, but the dtype is object.")

    def ensure_unbounded(self):
        """
        Ensure that the problem has no strict lower and upper bounds.

        Raises:
            ValueError: if the problem has strict lower and upper bounds.
        """
        if not (self.lower_bounds is None and self.upper_bounds is None):
            raise ValueError("Expected an unbounded problem, but this problem has lower and/or upper bounds.")

    def ensure_single_objective(self):
        """
        Ensure that the problem has only one objective.

        Raises:
            ValueError: if the problem is multi-objective.
        """
        n = len(self.senses)
        if n > 1:
            raise ValueError(f"Expected a single-objective problem, but this problem has {n} objectives.")

    def normalize_obj_index(self, obj_index: Optional[int] = None) -> int:
        """
        Normalize the objective index.

        If the provided index is non-negative, it is ensured that the index
        is valid.

        If the provided index is negative, the objectives are counted in the
        reverse order, and the corresponding non-negative index is returned.
        For example, -1 is converted to a non-negative integer corresponding to
        the last objective.

        If the provided index is None and if the problem is single-objective,
        the returned value is 0, which represents the only objective.

        If the provided index is None and if the problem is multi-objective,
        an error is raised.

        Args:
            obj_index: The non-normalized objective index.
        Returns:
            The normalized objective index, as a non-negative integer.
        """
        if obj_index is None:
            if len(self.senses) == 1:
                return 0
            else:
                raise ValueError(
                    "This problem is multi-objective, therefore, an explicit objective index was expected."
                    " However, `obj_index` was found to be None."
                )
        else:
            obj_index = int(obj_index)
            if obj_index < 0:
                obj_index = len(self.senses) + obj_index
            if obj_index < 0 or obj_index >= len(self.senses):
                raise IndexError("Objective index out of range.")
            return obj_index

    def _get_cloned_state(self, *, memo: dict) -> dict:
        # Collect the inner states of the remote Problem clones
        if self._actors is not None:
            self._remote_states = ray.get(
                [actor.call.remote("_make_pickle_data_for_main", [], {}) for actor in self._actors]
            )

        # Prepare the main state dictionary
        result = {}
        for k, v in self.__dict__.items():
            if k in ("_actors", "_actor_pool") or k in self._nonserialized_attribs:
                result[k] = None
            else:
                v_id = id(v)
                if v_id in memo:
                    result[k] = memo[v_id]
                else:
                    with _no_grad_if_basic_dtype(self.dtype):
                        result[k] = deep_clone(
                            v,
                            otherwise_deepcopy=True,
                            memo=memo,
                        )
        return result

    def _get_local_interaction_count(self) -> int:
        """
        Get the number of simulator interactions this Problem encountered.

        For problems focused on reinforcement learning, it is expected
        that the subclass overrides this method to describe its own way
        of getting the local interaction count.

        When working on parallelized problems, what is returned here is
        not necessarily synchronized with the other parallelized instance.
        """
        raise NotImplementedError

    def _get_local_episode_count(self) -> int:
        """
        Get the number of episodes this Problem encountered.

        For problems focused on reinforcement learning, it is expected
        that the subclass overrides this method to describe its own way
        of getting the local episode count.

        When working on parallelized problems, what is returned here is
        not necessarily synchronized with the other parallelized instance.
        """
        raise NotImplementedError

    def sample_and_compute_gradients(
        self,
        distribution,
        popsize: int,
        *,
        num_interactions: Optional[int] = None,
        popsize_max: Optional[int] = None,
        obj_index: Optional[int] = None,
        ranking_method: Optional[str] = None,
        with_stats: bool = True,
        ensure_even_popsize: bool = False,
    ) -> Union[list, dict]:
        """
        Sample new solutions from the distribution and compute gradients.

        The distribution can then be updated according to the computed
        gradients.

        If the problem is not parallelized, and `with_stats` is False,
        then the result will be a single dictionary of gradients.
        For example, in the case of a Gaussian distribution, the returned
        gradients dictionary would look like this:

            {
                "mu": ...,     # the gradient for the mean
                "sigma": ...,  # the gradient for the standard deviation
            }

        If the problem is not parallelized, and `with_stats` is True,
        then the result will be a dictionary which contains in itself
        the gradients dictionary, and additional elements for providing
        further information. In the case of a Gaussian distribution,
        the returned dictionary with additional stats would look like
        this:

            {
                "gradients": {
                    "mu": ...,     # the gradient for the mean
                    "sigma": ...,  # the gradient for the standard deviation
                },
                "num_solutions": ...,  # how many solutions were sampled
                "mean_eval": ...,      # Mean of all evaluations
            }

        If the problem is parallelized, then the gradient computation will
        be distributed among the remote actors. In more details, each actor
        will sample its own solutions (such that the total population size
        across all remote actors will be near the provided `popsize`)
        and will compute its own gradients, and will produce its own
        additional stats (if `with_stats` is given as True).
        These remote results will then be collected by the main process,
        and the final result of this method will be a list of dictionaries,
        each dictionary being the result of a remote gradient computation.

        The sampled solutions are temporary, and will not be kept
        (and will not be returned).

        To customize how solutions are sampled and how gradients are
        computed, one is encouraged to override
        `_sample_and_compute_gradients(...)` (instead of overriding this
        method directly.

        Args:
            distribution: The search distribution from which the solutions
                will be sampled, and according to which the gradients will
                be computed.
            popsize: The number of solutions which will be sampled.
            num_interactions: Number of simulator interactions that must
                be completed (more solutions will be sampled until this
                threshold is reached). This argument is to be used when
                the problem has characteristics similar to reinforcement
                learning, and an adaptive population size, depending on
                the interactions made, is desired.
                Otherwise, one can leave this argument as None, in which
                case, there will not be any threshold based on number
                of interactions.
            popsize_max: To be used when `num_interactions` is provided,
                as an additional criterion for ending the solution sampling
                phase. This argument can be used to prevent the population
                size from growing too much while trying to satisfy the
                `num_interactions`. If not needed, `popsize_max` can be left
                as None.
            obj_index: Index of the objective according to which the gradients
                will be computed. Can be left as None if the problem has only
                one objective.
            ranking_method: The solution ranking method to be used when
                computing the gradients.
                If not specified, the raw fitnesses will be used.
            with_stats: If given as False, then the results dictionary will
                only contain the gradients information. If given as True,
                then the results dictionary will contain within itself
                the gradients dictionary, and also additional elements for
                providing further information.
                The default is True.
            ensure_even_popsize: If `ensure_even_popsize` is True and the
                problem is not parallelized, then a `popsize` given as an odd
                number will cause an error. If `ensure_even_popsize` is True
                and the problem is parallelized, then the remote actors will
                sample their own sub-populations in such a way that their
                sizes are even.
                If `ensure_even_popsize` is False, whether or not the
                `popsize` is even will not be checked.
                When the provided `distribution` is a symmetric (or
                "mirrored", or "antithetic"), then this argument must be
                given as True.
        Returns:
            A results dictionary when the problem is not parallelized,
            or list of results dictionaries when the problem is parallelized.
        """

        # For problems which are configured for parallelization, make sure that the actors are created.
        self._parallelize()

        # Below we check if there is an inconsistency in arguments.
        if (num_interactions is None) and (popsize_max is not None):
            # If `num_interactions` is None, then we assume that the user does not wish an adaptive population size.
            # However, at the same time, if `popsize_max` is not None, then there is an inconsistency,
            # because, `popsize_max` without `num_interactions` (therefore without adaptive population size)
            # does not make sense.
            # This is probably a configuration error, so, we inform the user by raising an error.
            raise ValueError(
                f"`popsize_max` was expected as None, because `num_interactions` is None."
                f" However, `popsize_max` was found as {popsize_max}."
            )

        # The problem instance in the main process should trigger the `before_grad_hook`.
        if self.is_main:
            self._before_grad_hook()

        if self.is_main and (self._actors is not None) and (len(self._actors) > 0):
            # If this is the main process and the problem is parallelized, then we need to split the request
            # into multiple tasks, and then execute those tasks in parallel using the problem's actor pool.

            if self._subbatch_size is not None:
                # If `subbatch_size` is provided, then we first make sure that `popsize` is divisible by
                # `subbatch_size`
                if (popsize % self._subbatch_size) != 0:
                    raise ValueError(
                        f"This Problem was created with `subbatch_size` as {self._subbatch_size}."
                        f" When doing remote gradient computation, the requested population size must be divisible by"
                        f" the `subbatch_size`."
                        f" However, the requested population size is {popsize}, and the remainder after dividing it"
                        f" by `subbatch_size` is not 0 (it is {popsize % self._subbatch_size})."
                    )
                # After making sure that `popsize` and `subbatch_size` configurations are compatible, we declare that
                # we are going to have n tasks, each task imposing a sample size of `subbatch_size`.
                n = int(popsize // self._subbatch_size)
                popsize_per_task = [self._subbatch_size for _ in range(n)]
            elif self._num_subbatches is not None:
                # If `num_subbatches` is provided, then we are going to have n tasks where n is equal to the given
                # `num_subbatches`.
                popsize_per_task = split_workload(popsize, self._num_subbatches)
            else:
                # If neither `subbatch_size` nor `num_subbatches` is given, then we will split the workload in such
                # a way that each actor will have its share.
                popsize_per_task = split_workload(popsize, len(self._actors))

            if ensure_even_popsize:
                # If `ensure_even_popsize` argument is True, then we need to make sure that each tasks's popsize is
                # an even number.
                for i in range(len(popsize_per_task)):
                    if (popsize_per_task[i] % 2) != 0:
                        # If the i-th actor's assigned popsize is not even, increase its assigned popsize by 1.
                        popsize_per_task[i] += 1

            # The number of tasks is finally determined by the length of `popsize_per_task` list we created above.
            num_tasks = len(popsize_per_task)

            if num_interactions is None:
                # If the argument `num_interactions` is not given, then, for each task, we declare that
                # `num_interactions` is None.
                num_inter_per_task = [None for _ in range(num_tasks)]
            else:
                # If the argument `num_interactions` is given, then we compute each task's target number of
                # interactions from its sample size.
                num_inter_per_task = [
                    math.ceil((popsize_per_task[i] / popsize) * num_interactions) for i in range(num_tasks)
                ]

            if popsize_max is None:
                # If the argument `popsize_max` is not given, then, for each task, we declare that
                # `popsize_max` is None.
                popsize_max_per_task = [None for _ in range(num_tasks)]
            else:
                # If the argument `popsize_max` is given, then we compute each task's target maximum population size
                # from its sample size.
                popsize_max_per_task = [
                    math.ceil((popsize_per_task[i] / popsize) * popsize_max) for i in range(num_tasks)
                ]

            # We trigger the synchronization between the main process and the remote actors.
            # If this problem instance has nothing to synchronize, then `must_sync_after` will be False.
            must_sync_after = self._sync_before()

            # Because we want to send the distribution to remote actors, we first copy the distribution to cpu
            # (unless it is already on cpu)
            dist_on_cpu = distribution.to("cpu")

            # Here, we use our actor pool to execute our tasks in parallel.
            result = list(
                self._actor_pool.map_unordered(
                    (
                        lambda a, v: a.call.remote(
                            "_sample_and_compute_gradients",
                            [dist_on_cpu, v[0]],
                            {
                                "obj_index": obj_index,
                                "num_interactions": v[1],
                                "popsize_max": v[2],
                                "ranking_method": ranking_method,
                            },
                        )
                    ),
                    list(zip(popsize_per_task, num_inter_per_task, popsize_max_per_task)),
                )
            )

            # At this point, all the tensors within our collected results are on the CPU.

            if torch.device(self.device) != torch.device("cpu"):
                # If the main device of this problem instance is not CPU, then we move the tensors to the main device.
                result = cast_tensors_in_container(result, device=self.device)

            if must_sync_after:
                # If a post-gradient synchronization is required, we trigger the synchronization operations.
                self._sync_after()

            # ####################################################
            # # If this is the main process and the problem is parallelized, then we need to split the workload among
            # # the remote actors, and then request each of them to compute their gradients.
            #
            # # We begin by getting the number of actors, and computing the `popsize` for each actor.
            # num_actors = len(self._actors)
            # popsize_per_actor = split_workload(popsize, num_actors)
            #
            # if ensure_even_popsize:
            #     # If `ensure_even_popsize` argument is True, then we need to make sure that each actor's popsize is
            #     # an even number.
            #     for i in range(len(popsize_per_actor)):
            #         if (popsize_per_actor[i] % 2) != 0:
            #             # If the i-th actor's assigned popsize is not even, increase its assigned popsize by 1.
            #             popsize_per_actor[i] += 1
            #
            # if num_interactions is None:
            #     # If `num_interactions` is None, then the `num_interactions` argument for each actor must also be
            #     # passed as None.
            #     num_int_per_actor = [None] * num_actors
            # else:
            #     # If `num_interactions` is not None, then we split the `num_interactions` workload among the actors.
            #     num_int_per_actor = split_workload(num_interactions, num_actors)
            #
            # if popsize_max is None:
            #     # If `popsize_max` is None, then the `popsize_max` argument for each actor must also be None.
            #     popsize_max_per_actor = [None] * num_actors
            # else:
            #     # If `popsize_max` is not None, then we split the `popsize_max` workload among the actors.
            #     popsize_max_per_actor = split_workload(popsize_max, num_actors)
            #
            # # We trigger the synchronization between the main process and the remote actors.
            # # If this problem instance has nothing to synchronize, then `must_sync_after` will be False.
            # must_sync_after = self._sync_before()
            #
            # # Because we want to send the distribution to remote actors, we first copy the distribution to cpu
            # # (unless it is already on cpu)
            # dist_on_cpu = distribution.to("cpu")
            #
            # # To each actor, we send the request of computing the gradients, and then collect the results
            # result = ray.get(
            #     [
            #         self._actors[i].call.remote(
            #             "_gradient_computation_helper",
            #             [dist_on_cpu, popsize_per_actor[i]],
            #             dict(
            #                 num_interactions=num_int_per_actor[i],
            #                 popsize_max=popsize_max_per_actor[i],
            #                 obj_index=obj_index,
            #                 ranking_method=ranking_method,
            #                 with_stats=with_stats,
            #                 move_results_to_device="cpu",
            #             ),
            #         )
            #         for i in range(num_actors)
            #     ]
            # )
            #
            # # At this point, all the tensors within our collected results are on the CPU.
            #
            # if torch.device(self.device) != torch.device("cpu"):
            #     # If the main device of this problem instance is not CPU, then we move the tensors to the main device.
            #     result = cast_tensors_in_container(result, device=device)
            #
            # if must_sync_after:
            #     # If a post-gradient synchronization is required, we trigger the synchronization operations.
            #     self._sync_after()
        else:
            # If the problem is not parallelized, then we request this instance itself to compute the gradients.
            result = self._gradient_computation_helper(
                distribution,
                popsize,
                popsize_max=popsize_max,
                obj_index=obj_index,
                ranking_method=ranking_method,
                num_interactions=num_interactions,
                with_stats=with_stats,
            )

        # The problem instance in the main process should trigger the `after_grad_hook`.
        if self.is_main:
            self._after_eval_status = self._after_grad_hook.accumulate_dict(result)

        # We finally return the results
        return result

    def _gradient_computation_helper(
        self,
        distribution,
        popsize: int,
        *,
        num_interactions: Optional[int] = None,
        popsize_max: Optional[int] = None,
        obj_index: Optional[int] = None,
        ranking_method: Optional[str] = None,
        with_stats: bool = True,
        move_results_to_device: Optional[Device] = None,
    ) -> dict:
        # This is a helper method which makes sure that the provided distribution is in the correct dtype and device.
        # This method also makes sure that the results are moved to the desired device.

        # At first, we make sure that the objective index is normalized
        # (for example, the objective -1 is converted to the index of the last objective).
        obj_index = self.normalize_obj_index(obj_index)

        if (distribution.dtype != self.dtype) or (distribution.device != self.device):
            # Make sure that the distribution is in the correct dtype and device
            distribution = distribution.modified_copy(dtype=self.dtype, device=self.device)

        # Call the protected method responsible for sampling solutions and computing the gradients
        result = self._sample_and_compute_gradients(
            distribution,
            popsize,
            popsize_max=popsize_max,
            obj_index=obj_index,
            num_interactions=num_interactions,
            ranking_method=ranking_method,
        )

        if move_results_to_device is not None:
            # If `move_results_to_device` is provided, move the results to the desired device
            result = cast_tensors_in_container(result, device=move_results_to_device)

        # Finally, return the result
        if with_stats:
            return result
        else:
            return result["gradients"]

    @property
    def _grad_device(self) -> Device:
        """
        Get the device in which new solutions will be made in distributed mode.

        In more details, in distributed mode, each actor creates its own
        sub-populations, evaluates them, and computes its own gradient
        (all such actor gradients eventually being collected by the
        distribution-based search algorithm in the main process).
        For some problem types, it can make sense for the remote actors to
        create their temporary sub-populations on another device
        (e.g. on the GPU that is allocated specifically for them).
        For such situations, one is encouraged to override this property
        and make it return whatever device is to be used.

        Note that this property is used by the default implementation of the
        method named `_sample_and_compute_grad(...)`. If the method named
        `_sample_and_compute_grad(...)` is overriden, this property might not
        be called at all.

        This is the default (i.e. not-yet-overriden) implementation in the
        Problem class, and performs the following operations to decide the
        device:
        (i) if the Problem object was given an external fitness function that
        is decorated by @[on_device][evotorch.decorators.on_device],
        or by @[on_aux_device][evotorch.decorators.on_aux_device],
        or has a `device` attribute, then return the device requested by that
        function; otherwise
        (ii) if either one of the methods `_evaluate_batch`, and `_evaluate`
        was decorated by @[on_device][evotorch.decorators.on_device]
        or by @[on_aux_device][evotorch.decorators.on_aux_device],
        or has a `device` attribute, then return the device requested by that
        method; otherwise
        (iii) return the main device of the Problem object.
        """
        fitness_device = self._device_of_fitness_function()
        return self.device if fitness_device is None else fitness_device

    def _sample_and_compute_gradients(
        self,
        distribution,
        popsize: int,
        *,
        obj_index: int,
        num_interactions: Optional[int] = None,
        popsize_max: Optional[int] = None,
        ranking_method: Optional[str] = None,
    ) -> dict:
        """
        This method contains the description of how the solutions are sampled
        and the gradients are computed according to the given distribution.

        One might override this method for customizing the procedure of
        sampling solutions and the gradient computation, but this method does
        have a default implementation.

        This returns a dictionary which contains the gradients for the given
        distribution, and also further information. For example, considering
        a Gaussian distribution with parameters 'mu' and 'sigma', the result
        is expected to look like this:

            {
                "gradients": {
                    "mu": ...,     # the gradient for the mean (tensor)
                    "sigma": ...,  # the gradient for the std.dev. (tensor)
                },
                "num_solutions": ...,  # how many solutions were sampled (int)
                "mean_eval": ...,      # Mean of all evaluations (float)
            }

        A customized version of this method can add more items to the outer
        dictionary.

        Args:
            distribution: The search distribution from which the solutions
                will be sampled and according to which the gradients will
                be computed. This method assumes that `distribution` is
                given with this problem instance's dtype, and in this problem
                instance's device.
            popsize: Number of solutions to sample.
            obj_index: Objective index, expected as an integer.
            num_interactions: Number of simulator interactions that must be
                reached before computing the gradients.
                Having this argument as an integer implies that adaptive
                population is requested: more solutions are to be sampled
                until this number of simulator interactions are made.
                Can also be None if this threshold is not needed.
            popsize_max: Maximum population size for when the population
                size is adaptive (where the adaptiveness is enabled when
                `num_interactions` is not None).
                Can be left as None if a maximum population size limit
                is not needed.
            ranking_method: Ranking method to be used when computing the
                gradients. Can be left as None, in which case the raw
                fitnesses will be used.
        Returns:
            A dictionary which contains the gradients, number of solutions,
            mean of all the evaluation results, and optionally further
            items (if customized to do so).
        """

        # Annotate the variable which will store the temporary SolutionBatch for computing the local gradient.
        resulting_batch: SolutionBatch

        # Get the device in which the new solutions will be made.
        grad_device = torch.device(self._grad_device)
        distribution = distribution.to(grad_device)

        # Below we define an inner utility function which samples and evaluates a new SolutionBatch.
        # This newly evaluated SolutionBatch is returned.
        def sample_evaluated_batch() -> SolutionBatch:
            batch = SolutionBatch(self, popsize, device=grad_device)
            distribution.sample(out=batch.access_values(), generator=self.generator)
            self.evaluate(batch)
            return batch

        if num_interactions is None:
            # If a `num_interactions` threshold is not given (i.e. is left as None), then we assume that an adaptive
            # population is not desired.
            # We therefore simply sample and evaluate a single SolutionBatch, and declare it as our main batch.
            resulting_batch = sample_evaluated_batch()
        else:
            # If we have a `num_interactions` threshold, then we might have to sample more than one SolutionBatch
            # (until `num_interactions` is reached).
            # We start by defining a list (`batches`) which is to store all the batches we will sample.
            batches = []

            # We will have to count the number of all simulator interactions that we have encountered during the
            # execution of this method. So, to count it correctly, we first get the interaction count that we already
            # have before sampling and evaluating our new solutions.
            interaction_count_at_first = self._get_local_interaction_count()

            # Below is an inner function which returns how many simulator interactions we have done so far.
            # It makes use of the variable `interaction_count_at_first` defined above.
            def current_num_interactions() -> int:
                return self._get_local_interaction_count() - interaction_count_at_first

            # We also keep track of the total number of solutions.
            # We might need this if there is a `popsize_max` threshold.
            current_popsize = 0

            # The main loop of the adaptive sampling.
            while True:
                # Sample and evaluate a new SolutionBatch, and add it to our batches list.
                batches.append(sample_evaluated_batch())

                # Increase our total population size by the size of the most recent batch.
                current_popsize += popsize

                if current_num_interactions() > num_interactions:
                    # If the number of interactions has reached or exceeded the `num_interactions` threshold,
                    # we exit the loop.
                    break
                if (popsize_max is not None) and (current_popsize >= popsize_max):
                    # If we have `popsize_max` threshold and our total population size have reached or exceeded
                    # the `popsize_max` threshold, we exit the loop.
                    break

            if len(batches) == 1:
                # If we have only one batch in our batches list, that batch can be declared as our main batch.
                resulting_batch = batches[0]
            else:
                # If we have multiple batches in our batches list, we concatenate all those batches and
                # declare the result of the concatenation as our main batch.
                resulting_batch = SolutionBatch.cat(batches)

        # We take the solutions (`samples`) and the fitnesses from our main batch.
        samples = resulting_batch.access_values(keep_evals=True)
        fitnesses = resulting_batch.access_evals(obj_index)

        # With the help of `samples` and `fitnesses`, we now compute our gradients.
        grads = distribution.compute_gradients(
            samples, fitnesses, objective_sense=self.senses[obj_index], ranking_method=ranking_method
        )

        if grad_device != self.device:
            grads = cast_tensors_in_container(grads, device=self.device)

        # Finally, we return the result, which is a dictionary containing the gradients and further information.
        return {
            "gradients": grads,
            "num_solutions": len(resulting_batch),
            "mean_eval": float(torch.mean(resulting_batch.access_evals(obj_index))),
        }

    def is_on_cpu(self) -> bool:
        """
        Whether or not the Problem object has its device set as "cpu".
        """
        return str(self.device) == "cpu"

    def make_callable_evaluator(self, *, obj_index: Optional[int] = None) -> "ProblemBoundEvaluator":
        """
        Get a callable evaluator for evaluating the given solutions.

        Let us assume that we have a [Problem][evotorch.core.Problem]
        declared like this:

        ```python
        from evotorch import Problem

        my_problem = Problem(
            "min",
            fitness_function_goes_here,
            ...,
        )
        ```

        Using the regular API of EvoTorch, one has to generate solutions for this
        problem as follows:

        ```python
        population_size = ...
        my_solutions = my_problem.generate_batch(population_size)
        ```

        For editing the decision values within the
        [SolutionBatch][evotorch.core.SolutionBatch] `my_solutions`, one has to
        do the following:

        ```python
        new_decision_values = ...
        my_solutions.set_values(new_decision_values)
        ```

        Finally, to evaluate `my_solutions`, one would have to do these:

        ```python
        my_problem.evaluate(my_solutions)
        fitnesses = my_problem.evals
        ```

        One could desire a different interface which is more compatible with
        functional programming paradigm, especially when planning to use the
        functional algorithms (such as, for example, the functional
        [cem][evotorch.algorithms.functional.funccem.cem]).
        To achieve this, one can do the following:

        ```python
        f = my_problem.make_callable_evaluator()
        ```

        Now, we have a new object `f`, which behaves like a function.
        This function-like object expects a tensor of decision values, and
        returns fitnesses, as shown below:

        ```python
        random_decision_values = torch.randn(
            population_size,
            my_problem.solution_length,
            dtype=my_problem.dtype,
        )

        fitnesses = f(random_decision_values)
        ```

        **Parallelized fitness evaluation.**
        If a `Problem` object is condifured to use parallelized evaluation with
        the help of multiple actors, a callable evaluator made out of that
        `Problem` object will also make use of those multiple actors.

        **Additional batch dimensions.**
        If a callable evaluator receives a tensor with 3 or more dimensions,
        those extra leftmost dimensions will be considered as batch
        dimensions. The returned fitness tensor will also preserve those batch
        dimensions.

        **Notes on vmap.**
        `ProblemBoundEvaluator` is a shallow wrapper around a `Problem` object.
        It does NOT transform the underlying problem object to its stateless
        counterpart, and therefore it does NOT conform to pure functional
        programming paradigm. Being stateful, it will NOT work correctly with
        `vmap`. For batched evaluations, it is recommended to use extra batch
        dimensions, instead of using `vmap`.

        Args:
            obj_index: The index of the objective according to which the
                evaluations will be done. If the problem is single-objective,
                this is not required. If the problem is multi-objective, this
                needs to be given as an integer.
        Returns:
            A callable fitness evaluator, bound to this problem object.
        """
        return ProblemBoundEvaluator(self, obj_index=obj_index)


SolutionBatchSliceInfo = NamedTuple("SolutionBatchSliceInfo", source="SolutionBatch", slice=IndicesOrSlice)


def _opt_bool(x: Optional[bool], default: bool) -> bool:
    result = default
    if x is not None:
        result = bool(x)
    return result


_near_zero_float_tolerance = {
    torch.float16: 1e-4,
}


if hasattr(torch, "bfloat16"):
    _near_zero_float_tolerance[torch.bfloat16] = 1e-4


def _crowding_distance_assignment(pareto_set_utilities: torch.Tensor) -> torch.Tensor:
    """Compute the crowding distance metric as described in:
        Deb, Kalyanmoy, et al.
        "A fast and elitist multiobjective genetic algorithm: NSGA-II."
        IEEE transactions on evolutionary computation 6.2 (2002): 182-197.
    Args:
        pareto_set_utilities (torch.Tensor): The utilities values (or fitnesses) of shape [num_samples, num_objectives] from the pareto set.
    Returns:
        crowding_distances (torch.Tensor): The computed crowding distances of the pareto_set_utilities.
    """
    Inf = float("inf")
    near_zero_tolerance = _near_zero_float_tolerance.get(pareto_set_utilities.dtype, 1e-8)

    # Arg sort each objective
    argsorted_utilities = torch.argsort(pareto_set_utilities, dim=0)
    # Initialize distances to zero
    crowding_distances = torch.zeros_like(pareto_set_utilities[:, 0])

    # Solutions at the limits are assigned infinite distance
    crowding_distances[argsorted_utilities[0]] = Inf
    crowding_distances[argsorted_utilities[-1]] = Inf

    # Enumerate objectives (TODO can this be vectorized also?)
    for obj_index in range(pareto_set_utilities.shape[-1]):
        # Get the sorting and utility values for this objective
        obj_utilities = pareto_set_utilities[:, obj_index]
        obj_argsorted_utilities = argsorted_utilities[:, obj_index]
        obj_sorted_utilities = obj_utilities[obj_argsorted_utilities]

        # Compute the denominator (f_max - f_min)
        denominator = torch.amax(obj_utilities) - torch.amin(obj_utilities)

        # Ensure that the denominator is not very close to 0
        denominator.clamp_(near_zero_tolerance, Inf)

        # Get the solutions 0 ... num_samples -2
        obj_sorted_utilities_low = obj_sorted_utilities[:-2]
        # Get the solutions 2 ... num_samples
        obj_sorted_utilities_high = obj_sorted_utilities[2:]

        # Add the distance, for sorted solution i, (obj[i + 1] - obj[i - 1]) / denominator
        crowding_distances[obj_argsorted_utilities[1:-1]] += (
            obj_sorted_utilities_high - obj_sorted_utilities_low
        ) / denominator

    return crowding_distances


def _compute_pareto_ranks(utils: torch.Tensor, crowdsort: bool) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """GPU-friendly + Vectorized pareto ranking based on:
        Deb, Kalyanmoy, et al.
        "A fast and elitist multiobjective genetic algorithm: NSGA-II."
        IEEE transactions on evolutionary computation 6.2 (2002): 182-197.
    Args:
        utils (torch.Tensor): The utilities (or fitnesses) to rank of shape [num_samples, num_objectives]
        crowdsort (bool): Whether to apply crowd sorting, see: the above paper.
                        If crowd sorting is applied, then an additional tensor is returned containing crowd-sort ranks.
    Returns:
        ranks (torch.Tensor): The computed pareto ranks, of shape [num_samples,]
                              Ranks are encoded in the form 'lowest is best'. So solutions in the pareto front will be assigned rank 0,
                              solutions in the next non-dominated set (excluding the pareto front) will be assigned rank 1, and so on.
        crowdsort_ranks (Optional[torch.Tensor]): The computed crowd sort ranks, only returned if crowdsort=True. Otherwise, None is returned.
                              Solutions within a front are assigned a crowding score, as described in the above paper.
                              Then, the solution with the best crowding score within a front of size K is assigned rank 0, and the solution with the
                              worst crowding score within a front is assigned rank K-1.
    """
    # TODO, this is only needed while there are issues with ReadOnlyTensor
    utils = utils.clone()

    # Construct dense matrix of domination. Assumes maximization for all objectives.
    # For element i,j we have True iff solution j dominates solution i, False otherwise.
    # Solution `a` dominates solution `b` if the two conditions below are satisfied:
    # - `a` is never worse than `b` on any objective,
    # - `a` is better than `b` on at least one objective.
    utils_a = utils.unsqueeze(0)
    utils_b = utils.unsqueeze(1)
    never_worse_matrix = utils_a >= utils_b
    strictly_better_matrix = utils_a > utils_b
    dominated_matrix = torch.all(never_worse_matrix, dim=-1) & torch.any(strictly_better_matrix, dim=-1)

    # Calculate how many samples are dominated
    n_dominations = torch.sum(dominated_matrix, dim=-1)

    # Initialize all ranks to zero
    ranks = torch.zeros_like(utils[:, 0], dtype=torch.long)
    # Boolean flag as to whether each sample is currently unranked
    unranked = torch.ones_like(utils[:, 0], dtype=torch.bool)
    # Current rank that we will next assign to solutions
    current_rank = 0

    # Only initialize crowdsort_ranks if crowdsort is True
    if crowdsort:
        crowdsort_ranks = torch.zeros_like(ranks)
    else:
        crowdsort_ranks = None

    # Keep going while any solution is not ranked
    while unranked.any():
        # Get non-dominated solutions where the number of dominations is 0
        non_dominated = n_dominations == 0
        # Filter out previously non-dominated solutions
        new_non_dominated = non_dominated & unranked

        # Store the rank of the new non-dominated solutions
        ranks[new_non_dominated] = current_rank
        # Keep track of which solutions have been sorted
        unranked[new_non_dominated] = False

        # If crowd sorting, get the crowdsort distances, convert them to ranks and fill in the crowdsort_ranks elements
        if crowdsort:
            crowdsort_distances = _crowding_distance_assignment(utils[new_non_dominated])
            # Note the descending sort -- we want to maximize distances
            crowdsort_ranks[new_non_dominated] = torch.argsort(crowdsort_distances, descending=True)

        # Update the number of dominations for remaining solutions by removing any domination counts introduce by the solutions we just sorted
        n_dominations += -dominated_matrix[:, new_non_dominated].sum(dim=-1)
        # Increase the rank for the next iteration
        current_rank += 1

    return ranks, crowdsort_ranks


def _pareto_sort(utils: torch.Tensor, crowdsort: bool) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """Pareto sort a given set of utilities, in a GPU-friendly + Vectorized manner
    Args:
        utils (torch.Tensor): The utilities (or fitnesses) to rank of shape [num_samples, num_objectives]
        crowdsort (bool): Whether to apply crowd sorting
                        If crowd sorting is applied, then an additional tensor is returned containing crowd-sort ranks.
    Returns:
        fronts (torch.Tensor): The computed pareto fronts, a list of tensors. Each tensor is the indices of that front,
                              with fronts[0] being the best-ranked front (with rank 0) and fronts[-1] being the worst-ranked front
                              (with rank as the highest). If crowdsort == True, then each front is internally sorted by the crowding distance metric.
        ranks (torch.Tensor): The computed pareto ranks, of shape [num_samples,]
                              Ranks are encoded in the form 'lowest is best'. So solutions in the pareto front will be assigned rank 0,
                              solutions in the next non-dominated set (excluding the pareto front) will be assigned rank 1, and so on.
    """
    # Compute ranks (and crowdsort_ranks if crowdsort = True)
    ranks, crowdsort_ranks = _compute_pareto_ranks(utils, crowdsort)

    # Get the number of fronts
    n_fronts = torch.amax(ranks) + 1
    fronts = []
    # Enumerate fronts
    for front_idx in range(n_fronts):
        # Get the referenced front
        front = torch.argwhere(ranks == front_idx).flatten()
        # Sort according to crowdsort_ranks if crowdsort
        if crowdsort:
            front_crowdsort_ranks = crowdsort_ranks[front]
            front = front[front_crowdsort_ranks]

        fronts.append(front)
    return fronts, ranks


ParetoInfo = NamedTuple("ParetoInfo", fronts=list, ranks=torch.Tensor)


class SolutionBatch(Serializable):
    """
    Representation of a batch of solutions.

    A SolutionBatch stores the decision values of multiple solutions
    in a single contiguous tensor. For numeric and fixed-length
    problems, this contiguous tensor is a PyTorch tensor.
    For not-necessarily-numeric and not-necessarily-fixed-length
    problems, this contiguous tensor is an ObjectArray.

    The evalution results and extra evaluation data of the solutions
    are also stored in an additional contiguous tensor.

    Interface-wise, a SolutionBatch behaves like a sequence of
    Solution objects. One can get single Solution from a SolutionBatch
    via the indexing operator (`[]`).
    Additionally, one can iterate over each solution using
    the `for ... in ...` statement.

    One can also get a slice of a SolutionBatch.
    The slicing of a SolutionBatch results in a new SolutionBatch.
    With simple slicing, the obtained SolutionBatch shares its
    memory with the original SolutionBatch.
    With advanced slicing (i.e. the kind of slicing where the
    solution indices are specified one by one, like:
    `mybatch[[0, 4, 2, 5]]`), the obtained SolutionBatch is a copy,
    and does not share any memory with its original.

    The decision values of all the stored solutions in the batch
    can be obtained in a read-only tensor via:

        values = batch.values

    If one has modified decision values and wishes to put them
    into the batch, the `set_values(...)` method can be used
    as follows:

        batch.set_values(modified_values)

    The evaluation results of the solutions can be obtained
    in a read-only tensor via:

        evals = batch.evals

    If one has newly computed evaluation results, and wishes
    to put them into the batch, the `set_evals(...)` method
    can be used as follows:

        batch.set_evals(newly_computed_evals)
    """

    def __init__(
        self,
        problem: Optional[Problem] = None,
        popsize: Optional[int] = None,
        *,
        device: Optional[Device] = None,
        slice_of: Optional[Union[tuple, SolutionBatchSliceInfo]] = None,
        like: Optional["SolutionBatch"] = None,
        merging_of: Iterable["SolutionBatch"] = None,
        empty: Optional[bool] = None,
    ):
        self._num_objs: int
        self._data: Union[torch.Tensor, ObjectArray]
        self._descending: Iterable[bool]
        self._slice: Optional[IndicesOrSlice] = None

        if slice_of is not None:
            expect_none(
                "While making a new SolutionBatch via slicing",
                problem=problem,
                popsize=popsize,
                device=device,
                merging_of=merging_of,
                like=like,
                empty=empty,
            )

            source: "SolutionBatch"
            slice_info: IndicesOrSlice
            source, slice_info = slice_of

            def safe_slice(t: torch.Tensor, slice_info):
                d0 = t.ndim
                t = t[slice_info]
                d1 = t.ndim
                if d0 != d1:
                    raise ValueError(
                        "Encountered an illegal slicing operation which would"
                        " change the shape of the stored tensor(s) of the"
                        " SolutionBatch."
                    )
                return t

            with torch.no_grad():
                # self._data = source._data[slice_info]
                # self._evdata = source._evdata[slice_info]
                self._data = safe_slice(source._data, slice_info)
                self._evdata = safe_slice(source._evdata, slice_info)
            self._slice = slice_info
            self._descending = source._descending

            shares_storage = storage_ptr(self._data) == storage_ptr(source._data)

            if not shares_storage:
                self._descending = deepcopy(self._descending)

            self._num_objs = source._num_objs
        elif like is not None:
            expect_none(
                "While making a new SolutionBatch via the like=... argument",
                merging_of=merging_of,
                slice_of=slice_of,
            )
            self._data = empty_tensor_like(like._data, length=popsize, device=device)
            self._evdata = empty_tensor_like(like._evdata, length=popsize, device=device)
            self._evdata[:] = float("nan")

            self._descending = like._descending
            self._num_objs = like._num_objs

            if not _opt_bool(empty, default=False):
                self._fill_via_problem(problem)
        elif merging_of is not None:
            expect_none(
                "While making a new SolutionBatch via merging",
                problem=problem,
                popsize=popsize,
                device=device,
                slice_of=slice_of,
                like=like,
                empty=empty,
            )

            # Convert `merging_of` into a list.
            # While doing that, also count the total number of rows
            batches = []
            total_rows = 0
            for batch in merging_of:
                total_rows += len(batch)
                batches.append(batch)

            # Get essential attributes from the first batch
            self._descending = deepcopy(batches[0]._descending)
            self._num_objs = batches[0]._num_objs

            if isinstance(batches[0]._data, ObjectArray):

                def process_data(x):
                    return deepcopy(x)

                self._data = ObjectArray(total_rows)
            else:

                def process_data(x):
                    return x

                self._data = empty_tensor_like(batches[0]._data, length=total_rows)

            self._evdata = empty_tensor_like(batches[0]._evdata, length=total_rows)

            row_begin = 0
            for batch in batches:
                row_end = row_begin + len(batch)
                self._data[row_begin:row_end] = process_data(batch._data)
                self._evdata[row_begin:row_end] = batch._evdata
                row_begin = row_end

        elif problem is not None:
            expect_none(
                "While making a new SolutionBatch with a given problem",
                slice_of=slice_of,
                like=like,
                merging_of=merging_of,
            )

            if device is None:
                device = problem.device

            self._num_objs = len(problem.senses)

            if problem.dtype is object:
                if str(device) != "cpu":
                    raise ValueError("Cannot create a batch containing arbitrary objects on a device other than cpu")
                self._data = ObjectArray(popsize)
            else:
                self._data = torch.empty((popsize, problem.solution_length), device=device, dtype=problem.dtype)

            if not _opt_bool(empty, default=False):
                self._data[:] = problem.generate_values(len(self._data))

            self._evdata = problem.make_nan(
                popsize, self._num_objs + problem.eval_data_length, device=device, use_eval_dtype=True
            )
            self._descending = problem.get_obj_order_descending()
        else:
            raise ValueError("Invalid call to the __init__(...) of SolutionBatch")

    def _normalize_row_index(self, i: int) -> int:
        i = int(i)
        org_i = i

        if i < 0:
            i = int(self._data.shape[0]) + i

        if (i < 0) or (i > (self._data.shape[0] - 1)):
            raise IndexError(f"Invalid row: {org_i}")

        return i

    def _normalize_obj_index(self, i: int) -> int:
        i = int(i)
        org_i = i

        if i < 0:
            i = self._num_objs + i

        if (i < 0) or (i > (self._num_objs)):
            raise IndexError(f"Invalid objective index: {org_i}")

        return i

    def _optionally_get_obj_index(self, i: Optional[int]) -> int:
        if i is None:
            if self._num_objs != 1:
                raise ValueError(
                    f"The objective index was given as None."
                    f" However, the number of objectives is not 1,"
                    f" it is {self._num_objs}."
                    f" Therefore, the objective index is not optional,"
                    f" and must be provided as an integer, not as None."
                )
            return 0
        else:
            return self._normalize_obj_index(i)

    @torch.no_grad()
    def argsort(self, obj_index: Optional[int] = None) -> torch.Tensor:
        """Return the indices of solutions, sorted from best to worst.

        Args:
            obj_index: The objective index. Can be passed as None
                if the problem is single-objective. Otherwise,
                expected as an int.
        Returns:
            A PyTorch tensor, containing the solution indices,
            sorted from the best solution to the worst.
        """
        obj_index = self._optionally_get_obj_index(obj_index)

        descending = self._descending[obj_index]
        ev_col = self._evdata[:, obj_index]

        return torch.argsort(ev_col, descending=descending)

    @torch.no_grad()
    def compute_pareto_ranks(self, crowdsort: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the pareto-ranks of the solutions in the batch.
        Args:
            crowdsort: If given as True, each front in itself
                will be sorted from the least crowding solution
                to the most crowding solution.
                If given as False, there will be no crowd-sorting.
        Returns:
            ranks (torch.Tensor): The computed pareto ranks, of shape [num_samples,]
                                Ranks are encoded in the form 'lowest is best'. So solutions in the pareto front will be assigned rank 0,
                                solutions in the next non-dominated set (excluding the pareto front) will be assigned rank 1, and so on.
            crowdsort_ranks (Optional[torch.Tensor]): The computed crowd sort ranks, only returned if crowdsort=True. Otherwise, None is returned.
                                Solutions within a front are assigned a crowding score, as described in the above paper.
                                Then, the solution with the best crowding score within a front of size K is assigned rank 0, and the solution with the
                                worst crowding score within a front is assigned rank K-1.
        """
        utils = self.utils()

        ranks, crowdsort_ranks = _compute_pareto_ranks(utils, crowdsort)

        return ranks, crowdsort_ranks

    @torch.no_grad()
    def arg_pareto_sort(self, crowdsort: bool = True) -> ParetoInfo:
        """
        Pareto-sort the solutions in the batch.

        The result is a namedtuple consisting of two elements:
        `fronts` and `ranks`.
        Let us assume that we have 5 solutions, and after a
        pareto-sorting they ended up in this order:

            front 0 (best front) : solution 1, solution 2
            front 1              : solution 0, solution 4
            front 2 (worst front): solution 3

        Considering the example ordering above, the returned
        ParetoInfo instance looks like this:

            ParetoInfo(
                fronts=[[1, 2], [0, 4], [3]],
                ranks=tensor([1, 0, 0, 2, 1])
            )

        where `fronts` stores the solution indices grouped by
        pareto fronts; and `ranks` stores, as a tensor of int64,
        the pareto rank for each solution (where 0 means best
        rank).

        Args:
            crowdsort: If given as True, each front in itself
                will be sorted from the least crowding solution
                to the most crowding solution.
                If given as False, there will be no crowd-sorting.
        Returns:
            A ParetoInfo instance
        """

        utils = self.utils()

        fronts, ranks = _pareto_sort(utils, crowdsort)

        return ParetoInfo(fronts=fronts, ranks=ranks)

    @torch.no_grad()
    def argbest(self, obj_index: Optional[int] = None) -> torch.Tensor:
        """Return the best solution's index

        Args:
            obj_index: The objective index. Can be passed as None
                if the problem is single-objective. Otherwise,
                expected as an int.
        Returns:
            The index of the best solution.
        """
        obj_index = self._optionally_get_obj_index(obj_index)
        descending = self._descending[obj_index]
        argf = torch.argmax if descending else torch.argmin
        return argf(self._evdata[:, obj_index])

    @torch.no_grad()
    def argworst(self, obj_index: Optional[int] = None) -> torch.Tensor:
        """Return the worst solution's index

        Args:
            obj_index: The objective index. Can be passed as None
                if the problem is single-objective. Otherwise,
                expected as an int.
        Returns:
            The index of the worst solution.
        """
        obj_index = self._optionally_get_obj_index(obj_index)
        descending = self._descending[obj_index]
        argf = torch.argmin if descending else torch.argmax
        return argf(self._evdata[:, obj_index])

    def _get_objective_sign(self, i_obj: int) -> float:
        if self._descending[i_obj]:
            return 1.0
        else:
            return -1.0

    @torch.no_grad()
    def set_values(self, values: Any, *, solutions: MaybeIndicesOrSlice = None):
        """
        Set the decision values of the solutions.

        Args:
            values: New decision values.
            solutions: Optionally a list of integer indices or an instance
                of `slice(...)`, to be used if one wishes to set the
                decision values of only some of the solutions.
        """
        if solutions is None:
            solutions = slice(None, None, None)
        self._data[solutions] = values
        self._evdata[solutions] = float("nan")

    @torch.no_grad()
    def set_evals(
        self,
        evals: torch.Tensor,
        eval_data: Optional[torch.Tensor] = None,
        *,
        solutions: MaybeIndicesOrSlice = None,
    ):
        """
        Set the evaluations of the solutions.

        Args:
            evals: A numeric tensor which contains the evaluation results.
                Acceptable shapes are as follows:
                `(n,)` only to be used for single-objective problems, sets
                the evaluation results of the target `n` solutions, and clears
                (where clearing means to fill with NaN values)
                extra evaluation data (if the problem has allocations for such
                extra evaluation data);
                `(n,m)` where `m` is the number of objectives, sets the
                evaluation results of the target `n` solutions, and clears
                their extra evaluation data;
                `(n,m+q)` where `m` is the number of objectives and `q` is the
                length of extra evaluation data, sets the evaluation result
                and extra data of the target `n` solutions.
            eval_data: To be used only when the problem has extra evaluation
                data. Optionally, one can pass the extra evaluation data
                separately via this argument (instead of jointly through
                a single tensor via `evals`).
                The expected shape of this tensor is `(n,q)` where `n`
                is the number of solutions and `q` is the length of the
                extra evaluation data.
            solutions: Optionally a list of integer indices or an instance
                of `slice(...)`, to be used if one wishes to set the
                evaluations of only some of the solutions.
        Raises:
            ValueError: if the given tensor has an incompatible shape.
        """
        if solutions is None:
            solutions = slice(None, None, None)
            num_solutions = self._evdata.shape[0]
        elif isinstance(solutions, slice):
            num_solutions = self._evdata[solutions].shape[0]
        elif is_sequence(solutions):
            num_solutions = len(solutions)
        total_eval_width = self._evdata.shape[1]

        num_objs = self._num_objs
        num_data = total_eval_width - num_objs

        if evals.ndim == 1:
            if num_objs != 1:
                raise ValueError(
                    f"The method `set_evals(...)` was given a 1-dimensional tensor."
                    f" However, the number of objectives of the problem at hand is {num_objs}, not 1."
                    f" 1-dimensional evaluation tensors can only be accepted if the problem"
                    f" has one objective."
                )
            evals = evals.reshape(-1, 1)
        elif evals.ndim == 2:
            pass  # nothing to do here
        else:
            if num_objs == 1:
                raise ValueError(
                    f"The method `set_evals(...)` received a tensor with {evals.ndim} dimensions."
                    f" Since the problem at hand has only one objective,"
                    f" 1-dimensional or 2-dimensional tensors are acceptable, but not {evals.ndim}-dimensional ones."
                )
            else:
                raise ValueError(
                    f"The method `set_evals(...)` received a tensor with {evals.ndim} dimensions."
                    f" Since the problem at hand has more than one objective (there are {num_objs} objectives),"
                    f" only 2-dimensional tensors are acceptable, not {evals.ndim}-dimensional ones."
                )

        [nrows, ncols] = evals.shape

        if nrows != num_solutions:
            raise ValueError(
                f"Trying to set the evaluations of {num_solutions} solutions, but the given tensor has {nrows} rows."
            )

        if eval_data is not None:
            if eval_data.ndim != 2:
                raise ValueError(
                    f"The `eval_data` argument was expected as a 2-dimensional tensor."
                    f" However, the shape of the given `eval_data` is {eval_data.shape}."
                )
            if eval_data.shape[1] != num_data:
                raise ValueError(
                    f"The `eval_data` argument was expected to have {num_data} columns."
                    f" However, the received `eval_data` has the shape: {eval_data.shape}."
                )
            if ncols != num_objs:
                raise ValueError(
                    f"The method `set_evals(...)` was used with `evals` and `eval_data` arguments."
                    f" When both of these arguments are provided, `evals` is expected either as a 1-dimensional tensor"
                    f" (for single-objective cases only), or as a tensor of shape (n, m) where n is the number of"
                    f" solutions, and m is the number of objectives."
                    f" However, while the problem at hand has {num_objs} objectives,"
                    f" the `evals` tensor has {ncols} columns."
                )
            if evals.shape[0] != eval_data.shape[0]:
                raise ValueError(
                    f"The provided `evals` and `eval_data` tensors have incompatible shapes."
                    f" Shape of `evals`: {evals.shape},"
                    f" shape of `eval_data`: {eval_data.shape}."
                )
            self._evdata[solutions, :] = torch.hstack([evals, eval_data])
        else:
            if ncols == num_objs:
                self._evdata[solutions, :num_objs] = evals
                self._evdata[solutions, num_objs:] = float("nan")
            elif ncols == total_eval_width:
                self._evdata[solutions, :] = evals
            else:
                raise ValueError(
                    f"The method `set_evals(...)` received a tensor with {ncols} columns, which is incompatible."
                    f" Acceptable number of columns are: {num_objs}"
                    f" (for setting only the objective-associated evaluations and leave extra evaluation data as NaN), or"
                    f" {total_eval_width} (for setting both objective-associated evaluations and extra evaluation data)."
                )

    @property
    def evals(self) -> torch.Tensor:
        """
        Evaluation results of the solutions, in a ReadOnlyTensor
        """
        from .tools.readonlytensor import as_read_only_tensor

        with torch.no_grad():
            return as_read_only_tensor(self._evdata)

    @property
    def values(self) -> Union[torch.Tensor, Iterable]:
        """
        Decision values of the solutions, in a read-only tensor-like object
        """
        from .tools.readonlytensor import as_read_only_tensor

        with torch.no_grad():
            return as_read_only_tensor(self._data)

    # @property
    # def unsafe_evals(self) -> torch.Tensor:
    #    """
    #    It is not recommended to use this property.
    #
    #    Grants mutable access to the evaluations of the solutions.
    #    """
    #    return self._evdata
    #
    # @property
    # def unsafe_values(self) -> Union[torch.Tensor, Iterable]:
    #    """
    #    It is not recommended to use this property.
    #
    #    Grants mutable access to the decision values of the solutions.
    #    """
    #    return self._data

    @torch.no_grad()
    def access_evals(self, obj_index: Optional[int] = None) -> torch.Tensor:
        """
        Get the internal mutable tensor storing the evaluations.

        IMPORTANT: This method exposes the evaluation tensor of the batch
        as it is, in mutable mode. It is therefore considered unsafe to rely
        on this method. Before using this method, please consider using the
        `evals` property for reading the evaluation results, and using the
        `set_evals(...)` method which allows one to update the evaluations
        without exposing any internal tensor.

        When this method is used without any argument, the returned tensor
        will be of shape `(n, m)`, where `n` is the number of solutions,
        and `m` is the number of objectives plus the length of extra
        evaluation data.

        When this method is used with an integer argument specifying an
        objective index, the returned tensor will be 1-dimensional
        having a length of `n`, where `n` is the number of solutions.
        In this case, the returned 1-dimensional tensor will be a view
        upon the evaluation results of the specified objective.

        The value `nan` (not-a-number) means not evaluated yet.

        Args:
            obj_index: None for getting the entire 2-dimensional evaluation
                tensor; an objective index (as integer) for getting a
                1-dimensional mutable slice of the evaluation tensor,
                the slice being a view upon the evaluation results
                regarding the specified objective.
        Returns:
            The mutable tensor storing the evaluation information.
        """
        if obj_index is None:
            return self._evdata
        else:
            return self._evdata[:, self._normalize_obj_index(obj_index)]

    @torch.no_grad()
    def access_values(self, *, keep_evals: bool = False) -> Union[torch.Tensor, ObjectArray]:
        """
        Get the internal mutable tensor storing the decision values.

        IMPORTANT: This method exposes the internal decision values tensor of
        the batch as it is, in mutable mode. It is therefore considered unsafe
        to rely on this method. Before using this method, please consider
        using the `values` property for reading the decision values, and using
        the `set_values(...)` method which allows one to update the decision
        values without exposing any internal tensor.

        IMPORTANT: The default assumption of this method is that the tensor
        is requested for modification purposes. Therefore, by default, as soon
        as this method is called, the evaluation results of the solutions will
        be cleared (where clearing means that the evaluation results will be
        filled with `NaN`s).
        The reasoning behind this default behavior is to prevent the modified
        solutions from having outdated evaluation results.

        Args:
            keep_evals: If set as False, the evaluation data of the solutions
                will be cleared (i.e. will be filled with `NaN`s).
                If set as True, the existing evaluation data will be kept.
        Returns:
            The mutable tensor storing the decision values.
        """
        if not keep_evals:
            self.forget_evals()
        return self._data

    @torch.no_grad()
    def forget_evals(self, *, solutions: MaybeIndicesOrSlice = None):
        """
        Forget the evaluations of the solutions.
        The evaluation results will be cleared, which means that they will
        be filled with `NaN`s.
        """
        if solutions is None:
            solutions = slice(None, None, None)
        self._evdata[solutions, :] = float("nan")

    @torch.no_grad()
    def utility(
        self,
        obj_index: Optional[int] = None,
        *,
        ranking_method: Optional[str] = None,
        check_nans: bool = True,
        using_values_dtype: bool = False,
    ) -> torch.Tensor:
        """
        Return numeric scores for each solution.

        Utility scores are different from evaluation results,
        in the sense that utilities monotonically increase from
        bad solutions to good solutions, regardless of the
        objective sense.

        **If ranking method is passed as None:**
        if the objective sense is 'max', the evaluation results are returned
        as the utility scores; otherwise, if the objective sense is 'min',
        the evaluation results multiplied by -1 are returned as the
        utility scores.

        **If the name of a ranking method is given** (e.g. 'centered'):
        then the solutions are ranked (best solutions having the
        highest rank), and those ranks are returned as the utility
        scores.

        **If an objective index is not provided:** (i.e. passed as None)
        if the problem is multi-objective, the utility scores
        for each objective is given, in a tensor shaped (n, m),
        n being the number of solutions and m being the number
        of objectives; otherwise, if the problem is single-objective,
        the utility scores for each objective is given in a 1-dimensional
        tensor of length n, n being the number of solutions.

        **If an objective index is provided as an int:**
        the utility scores are returned in a 1-dimensional tensor
        of length n, n being the number of solutions.

        Args:
            obj_index: Expected as None, or as an integer.
                In the single-objective case, None is equivalent to 0.
                In the multi-objective case, None means "for each
                objective".
            ranking_method: If the utility scores are to be generated
                according to a certain ranking method, pass here the name
                of that ranking method as a str (e.g. 'centered').
            check_nans: Check for nan (not-a-number) values in the
                evaluation results, which is an indication of
                unevaluated solutions.
            using_values_dtype: If True, the utility values will be returned
                using the dtype of the decision values.
                If False, the utility values will be returned using the dtype
                of the evaluation data.
                The default is False.
        Returns:
            Utility scores, in a PyTorch tensor.
        """
        if obj_index is not None:
            obj_index = self._normalize_obj_index(obj_index)

            evdata = self._evdata[:, obj_index]
            if check_nans:
                if torch.any(torch.isnan(evdata)):
                    raise ValueError(
                        "Cannot compute the utility values, because there are solutions which are not evaluated yet."
                    )

            if ranking_method is None:
                result = evdata * self._get_objective_sign(obj_index)
            else:
                result = rank(evdata, ranking_method=ranking_method, higher_is_better=self._descending[obj_index])

            if using_values_dtype:
                result = torch.as_tensor(result, dtype=self._data.dtype, device=self._data.device)

            return result
        else:
            if self._num_objs == 1:
                return self.utility(
                    0, ranking_method=ranking_method, check_nans=check_nans, using_values_dtype=using_values_dtype
                )
            else:
                return torch.stack(
                    [
                        self.utility(
                            j,
                            ranking_method=ranking_method,
                            check_nans=check_nans,
                            using_values_dtype=using_values_dtype,
                        )
                        for j in range(self._num_objs)
                    ],
                ).T

    @torch.no_grad()
    def utils(
        self,
        *,
        ranking_method: Optional[str] = None,
        check_nans: bool = True,
        using_values_dtype: bool = False,
    ) -> torch.Tensor:
        """
        Return numeric scores for each solution, and for each objective.
        Utility scores are different from evaluation results,
        in the sense that utilities monotonically increase from
        bad solutions to good solutions, regardless of the
        objective sense.

        Unlike the method called `utility(...)`, this function returns
        a 2-dimensional tensor even when the problem is single-objective.

        The result of this method is always a 2-dimensional tensor of
        shape `(n, m)`, `n` being the number of solutions, `m` being the
        number of objectives.

        Args:
            ranking_method: If the utility scores are to be generated
                according to a certain ranking method, pass here the name
                of that ranking method as a str (e.g. 'centered').
            check_nans: Check for nan (not-a-number) values in the
                evaluation results, which is an indication of
                unevaluated solutions.
            using_values_dtype: If True, the utility values will be returned
                using the dtype of the decision values.
                If False, the utility values will be returned using the dtype
                of the evaluation data.
                The default is False.
        Returns:
            Utility scores, in a 2-dimensional PyTorch tensor.

        """
        result = self.utility(
            ranking_method=ranking_method, check_nans=check_nans, using_values_dtype=using_values_dtype
        )
        if result.ndim == 1:
            result = result.view(len(result), 1)
        return result

    def split(self, num_pieces: Optional[int] = None, *, max_size: Optional[int] = None) -> "SolutionBatchPieces":
        """Split this SolutionBatch into a specified number of pieces,
        or into an unspecified number of pieces where the maximum
        size of each piece is specified.

        Args:
            num_pieces: Can be provided as an integer n, which means
                that the this SolutionBatch will be split to n pieces.
                Alternatively, can be left as None if the user intends
                to set max_size as an integer instead.
            max_size: Can be provided as an integer n, which means
                that this SolutionBatch will be split to multiple
                pieces, each piece containing n solutions at most.
                Alternatively, can be left as None if the user intends
                to set num_pieces as an integer instead.
        Returns:
            A SolutionBatchPieces object, which behaves like a list of
            SolutionBatch objects, each object in the list being a
            slice view of this SolutionBatch object.
        """
        return SolutionBatchPieces(self, num_pieces=num_pieces, max_size=max_size)

    @torch.no_grad()
    def concat(self, other: Union["SolutionBatch", Iterable]) -> "SolutionBatch":
        """Concatenate this SolutionBatch with the other(s).

        In this context, concatenation means that the solutions of
        this SolutionBatch and of the others are collected in one big
        SolutionBatch object.

        Args:
            other: A SolutionBatch, or a sequence of SolutionBatch objects.
        Returns:
            A new SolutionBatch object which is the result of the
            concatenation.
        """
        if isinstance(other, SolutionBatch):
            lst = [self, other]
        else:
            lst = [self]
            lst.extend(list(other))
        return SolutionBatch(merging_of=lst)

    def take(self, indices: Iterable) -> "SolutionBatch":
        """Make a new SolutionBatch containing the specified solutions.

        Args:
            indices: A sequence of solution indices. These specified
                solutions will make it to the newly made SolutionBatch.
        Returns:
            The new SolutionBatch.
        """
        if is_sequence(indices):
            return type(self)(slice_of=(self, indices))
        else:
            raise TypeError("Expected a sequence of solution indices, but got a `{type(indices)}`")

    def take_best(self, n: int, *, obj_index: Optional[int] = None) -> "SolutionBatch":
        """Make a new SolutionBatch containing the best `n` solutions.

        Args:
            n: Number of solutions which will be taken.
            obj_index: Objective index according to which the best ones
                will be taken.
                If `obj_index` is left as None and the problem is multi-
                objective, then the solutions will be ranked according to
                their fronts, and according how crowding they are, and then
                the topmost `n` solutions will be taken.
                If `obj_index` is left as None and the problem is single-
                objective, then that single objective will be taken as the
                ranking criterion.
        Returns:
            The new SolutionBatch.
        """
        if obj_index is None and self._num_objs >= 2:
            ranks, crowdsort_ranks = self.compute_pareto_ranks(crowdsort=True)
            # Combine the ranks, such that solutions with a better crowdsort rank are weighted above solutions with the same pareto rank **only**
            combined_ranks = ranks.to(torch.float) + 0.1 * crowdsort_ranks.to(torch.float) / len(self)
            indices = torch.argsort(combined_ranks)[:n]
        else:
            indices = self.argsort(obj_index)[:n]
        return type(self)(slice_of=(self, indices))

    def __getitem__(self, i):
        if isinstance(i, slice) or is_sequence(i) or isinstance(i, type(...)):
            return type(self)(slice_of=(self, i))
        else:
            return Solution(parent=self, index=i)

    def __len__(self):
        return int(self._data.shape[0])

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def _get_cloned_state(self, *, memo: dict) -> dict:
        with _no_grad_if_basic_dtype(self.dtype):
            return deep_clone(
                self.__dict__,
                otherwise_deepcopy=True,
                memo=memo,
            )

    def to(self, device: Device) -> "SolutionBatch":
        """
        Get the counterpart of this SolutionBatch on the new device.

        If the specified device is the device of this SolutionBatch,
        then this SolutionBatch itself is returned.
        If the specified device is a different device, then a clone
        of this SolutionBatch on this different device is first
        created, and then this new clone is returned.

        Please note that the `to(...)` method is not supported when
        the dtype is `object`.

        Args:
            device: The device on which the resulting SolutionBatch
                will be stored.
        Returns:
            The SolutionBatch on the specified device.
        """
        if isinstance(self._data, ObjectArray):
            raise ValueError("The `to(...)` method is not supported when the dtype is `object`.")

        device = torch.device(device)
        if device == self.device:
            return self
        else:
            new_batch = SolutionBatch(like=self, device=device, empty=True)
            with torch.no_grad():
                new_batch._data[:] = self._data.to(device)
                new_batch._evdata[:] = self._evdata.to(device)
            return new_batch

    @property
    def device(self) -> Device:
        """
        The device in which the solutions are stored.
        """
        return self._data.device

    @property
    def dtype(self) -> DType:
        """
        The dtype of the decision values of the solutions.

        This property exists as an alias for the property
        `.values_dtype`.
        """
        return self._data.dtype

    @property
    def values_dtype(self) -> DType:
        """
        The dtype of the decision values of the solutions.
        """
        return self._data.dtype

    @property
    def eval_dtype(self) -> DType:
        """
        The dtype of the evaluation results and extra evaluation data
        of the solutions.
        """
        return self._evdata.dtype

    @property
    def values_shape(self) -> torch.Size:
        """
        The shape of the batch's decision values tensor, as a tuple (n, l),
        where `n` is the number of solutions, and `l` is the length
        of a single solution.

        If `dtype=None`, then there is no fixed length.
        Therefore, the shape is returned as (n,).
        """
        return self._data.shape

    @property
    def eval_shape(self) -> torch.Size:
        """
        The shape of the batch's evaluation tensor, as a tuple (n, l),
        where `n` is the number of solutions, and `l` is an integer
        which is equal to number of objectives plus the length of the
        extra evaluation data, if any.
        """
        return self._evdata.shape

    @property
    def solution_length(self) -> Optional[int]:
        """
        Get the length of a solution, if this batch is numeric.
        For non-numeric batches (i.e. batches with dtype=object),
        `solution_length` is given as None.
        """
        if self._data.ndim == 2:
            return self._data.shape[1]
        else:
            return None

    @property
    def objective_sense(self) -> ObjectiveSense:
        """
        Get the objective sense(s) of this batch's associated Problem.

        If the problem is single-objective, then a single string is returned.
        If the problem is multi-objective, then the objective senses will be
        returned in a list.

        The returned string in the single-objective case, or each returned
        string in the multi-objective case, is "min" or "max".
        """
        if len(self.senses) == 1:
            return self.senses[0]
        else:
            return self.senses

    @property
    def senses(self) -> Iterable[str]:
        """
        Objective sense(s) of this batch's associated Problem.

        This is a list of strings, each string being "min" or "max".
        """

        def desc_to_sense(desc: bool) -> str:
            return "max" if desc else "min"

        return [desc_to_sense(desc) for desc in self._descending]

    @staticmethod
    def cat(solution_batches: Iterable) -> "SolutionBatch":
        """
        Concatenate multiple SolutionBatch instances into one.

        Args:
            solution_batches: An Iterable of SolutionBatch objects to
                concatenate.
        Returns:
            The result of the concatenation, as a new SolutionBatch.
        """
        first = None
        rest = []
        for i, batch in enumerate(solution_batches):
            if not isinstance(batch, SolutionBatch):
                raise TypeError(f"Expected a SolutionBatch but got {repr(batch)}")
            if i == 0:
                first = batch
            else:
                rest.append(batch)
        return first.concat(rest)


class SolutionBatchPieces(Sequence):
    """A collection of SolutionBatch slice views.

    An instance of this class behaves like a read-only collection of
    SolutionBatch objects (each being a sliced view of a bigger
    SolutionBatch).
    """

    @torch.no_grad()
    def __init__(self, batch: SolutionBatch, *, num_pieces: Optional[int] = None, max_size: Optional[int] = None):
        """
        `__init__(...)`: Initialize the SolutionBatchPieces.

        Args:
            batch: The SolutionBatch which will be split into
                multiple SolutionBatch views.
                Each view itself is a SolutionBatch object,
                but not independent, meaning that any modification
                done to a SolutionBatch view will reflect on this
                main batch.
            num_pieces: Can be provided as an integer n, which means
                that the main SolutionBatch will be split to n pieces.
                Alternatively, can be left as None if the user intends
                to set max_size as an integer instead.
            max_size: Can be provided as an integer n, which means
                that the main SolutionBatch will be split to multiple
                pieces, each piece containing n solutions at most.
                Alternatively, can be left as None if the user intends
                to set num_pieces as an integer instead.
        """

        self._batch = batch

        self._pieces: List[SolutionBatch] = []
        self._piece_sizes: List[int] = []
        self._piece_slices: List[Tuple[int, int]] = []

        total_size = len(self._batch)

        if max_size is None and num_pieces is not None:
            num_pieces = int(num_pieces)
            # divide to pieces
            base_size = total_size // num_pieces
            rest = total_size - (base_size * num_pieces)
            self._piece_sizes = [base_size] * num_pieces
            for i in range(rest):
                self._piece_sizes[i] += 1
        elif max_size is not None and num_pieces is None:
            max_size = int(max_size)
            # divide to pieces
            num_pieces = math.ceil(total_size / max_size)
            current_total = 0
            for i in range(num_pieces):
                if current_total + max_size > total_size:
                    self._piece_sizes.append(total_size - current_total)
                else:
                    self._piece_sizes.append(max_size)
                current_total += max_size
        elif max_size is not None and num_pieces is not None:
            raise ValueError("Expected either max_size or num_pieces, received both.")
        elif max_size is None and num_pieces is None:
            raise ValueError("Expected either max_size or num_pieces, received none.")

        current_begin = 0
        for size in self._piece_sizes:
            current_end = current_begin + size
            self._piece_slices.append((current_begin, current_end))
            current_begin = current_end

        for slice_begin, slice_end in self._piece_slices:
            self._pieces.append(self._batch[slice_begin:slice_end])

    def __len__(self) -> int:
        return len(self._pieces)

    def __getitem__(self, i: Union[int, slice]) -> SolutionBatch:
        return self._pieces[i]

    def iter_with_indices(self):
        """Iterate over each `(piece, (i_begin, i_end))`
        where `piece` is a SolutionBatch view, `i_begin` is the beginning
        index of the SolutionBatch view in the main batch, `j_begin` is the
        ending index (exclusive) of the SolutionBatch view in the main batch.
        """
        for i in range(len(self._pieces)):
            yield self._pieces[i], self._piece_slices[i]

    def indices_of(self, n) -> tuple:
        """Get `(i_begin, i_end)` for the n-th piece
        (i.e. the n-th sliced view of the main SolutionBatch)
        where `i_begin` is the beginning index of the n-th piece,
        `i_end` is the (exclusive) ending index of the n-th piece.

        Args:
            n: Specifies the index of the queried SolutionBatch view.
        Returns:
            Beginning and ending indices of the SolutionBatch view,
            in a tuple.
        """
        return self._piece_slices[n]

    @property
    def batch(self) -> SolutionBatch:
        """Get the main SolutionBatch object, in its non-split form"""
        return self._batch

    def _to_string(self) -> str:
        f = io.StringIO()
        print(f"<{type(self).__name__}", file=f)
        n = len(self._pieces)
        for i, piece in enumerate(self._pieces):
            print(f"    {piece}", end="", file=f)
            if (i + 1) == n:
                print(file=f)
            else:
                print(",", file=f)
        print(">", file=f)
        f.seek(0)
        return f.read()

    def __str__(self) -> str:
        return self._to_string()

    def __repr__(self) -> str:
        return self._to_string()


def _all_none(x):
    if x is None:
        return True
    elif isinstance(x, Iterable):
        for element in x:
            if element is not None:
                return False
        return True
    else:
        return False


class Solution(Serializable):
    """
    Representation of a single Solution.

    A Solution can be a reference to a row of a SolutionBatch
    (in which case it shares its storage with the SolutionBatch),
    or can be an independent solution.
    When the Solution shares its storage with a SolutionBatch,
    any modifications to its decision values and/or evaluation
    results will affect its parent SolutionBatch as well.

    When a Solution object is cloned (via its `clone()` method,
    or via the functions `copy.copy(...)` and `copy.deepcopy(...)`,
    a new independent Solution object will be created.
    This new independent copy will NOT share its storage with
    its original SolutionBatch anymore.
    """

    def __init__(self, parent: SolutionBatch, index: int):
        """
        `__init__(...)`: Initialize the Solution object.

        Args:
            parent: The parent SolutionBatch which stores the Solution.
            index: Index of the solution in SolutionBatch.
        """
        if not isinstance(parent, SolutionBatch):
            raise TypeError(
                f"Expected a SolutionBatch as a parent, but encountered {repr(parent)},"
                f" which is of type {repr(type(parent))}."
            )
        index = int(index)
        if index < 0:
            index = len(parent) + index
        if not ((index >= 0) and (index <= len(parent))):
            raise IndexError(f"Invalid index: {index}")
        self._batch: SolutionBatch = parent[index : index + 1]

    def access_values(self, *, keep_evals: bool = False) -> torch.Tensor:
        """
        Access the decision values tensor of the solution.
        The received tensor will be mutable.

        By default, it will be assumed that the user wishes to
        obtain this tensor to change the decision values, and therefore,
        the evaluation results associated with this solution will be
        cleared (i.e. will be NaN).

        Args:
            keep_evals: When this is set to True, the evaluation results
                associated with this solution will be kept (i.e. will NOT
                be cleared).
        Returns:
            The decision values tensor of the solution.
        """
        return self._batch.access_values(keep_evals=keep_evals)[0]

    def access_evals(self) -> torch.Tensor:
        """
        Access the evaluation results of the solution.
        The received tensor will be mutable.

        Returns:
            The evaluation results tensor of the solution.
        """
        return self._batch.access_evals()[0]

    @property
    def values(self) -> Any:
        """
        Decision values of the solution
        """
        return self._batch.values[0]

    @property
    def evals(self) -> torch.Tensor:
        """
        Evaluation results of the solution in a 1-dimensional tensor.
        """
        return self._batch.evals[0]

    @property
    def evaluation(self) -> torch.Tensor:
        """
        Get the evaluation result.

        If the problem is single-objective and the problem does not
        allocate any space for extra evaluation data, then a scalar
        is returned.
        Otherwise, this property becomes equivalent to the `evals`
        property, and a 1-dimensional tensor is returned.
        """
        result = self.evals
        if len(result) == 1:
            result = result[0]
        return result

    def set_values(self, values: Any):
        """
        Set the decision values of the Solution.

        Note that modifying the decision values will result in the
        evaluation results being getting cleared (in more details,
        the evaluation results tensor will be filled with NaN values).

        Args:
            values: New decision values for this Solution.
        """
        if is_dtype_object(self.dtype):
            value_tensor = ObjectArray(1)
            value_tensor[0] = values
        else:
            value_tensor = torch.as_tensor(values, dtype=self.dtype).reshape(1, -1)
        self._batch.set_values(value_tensor)

    def set_evals(self, evals: torch.Tensor, eval_data: Optional[Iterable] = None):
        """
        Set the evaluation results of the Solution.

        Args:
            evals: New evaluation result(s) for the Solution.
                For single-objective problems, this argument can be given
                as a scalar.
                When this argument is given as a scalar (for single-objective
                cases) or as a tensor which is long enough to cover for
                all the objectives but not for the extra evaluation data,
                then the extra evaluation data will be cleared
                (in more details, extra evaluation data will be filled with
                NaN values).
            eval_data: Optionally, the argument `eval_data` can be used to
                specify extra evaluation data separately.
                `eval_data` is expected as a 1-dimensional sequence.
        """
        evals = torch.as_tensor(evals, dtype=self.eval_dtype, device=self.device)
        if evals.ndim in (0, 1):
            evals = evals.reshape(1, -1)
        else:
            raise ValueError(
                f"`set_evals(...)` method of a Solution expects a 1-dimensional or a 2-dimensional"
                f" evaluation tensor. However, the received evaluation tensor has {evals.ndim} dimensions"
                f" (having a shape of {evals.shape})."
            )

        if eval_data is not None:
            eval_data = torch.as_tensor(eval_data, dtype=self.eval_dtype, device=self.device)
            if eval_data.ndim != 1:
                raise ValueError(
                    f"The argument `eval_data` was expected as a 1-dimensional sequence."
                    f" However, the shape of `eval_data` is {eval_data.shape}."
                )
            eval_data = eval_data.reshape(1, -1)

        self._batch.set_evals(evals, eval_data)

    def set_evaluation(self, evaluation: RealOrVector, eval_data: Optional[Iterable] = None):
        """
        Set the evaluation results of the Solution.

        This method is an alias for `set_evals(...)`, added for having
        a setter counterpart for the `evaluation` property of the Solution
        class.

        Args:
            evaluation: New evaluation result(s) for the Solution.
                For single-objective problems, this argument can be given
                as a scalar.
                When this argument is given as a scalar (for single-objective
                cases) or as a tensor which is long enough to cover for
                all the objectives but not for the extra evaluation data,
                then the extra evaluation data will be cleared
                (in more details, extra evaluation data will be filled with
                NaN values).
            eval_data: Optionally, the argument `eval_data` can be used to
                specify extra evaluation data separately.
                `eval_data` is expected as a 1-dimensional sequence.
        """
        self.set_evals(evaluation, eval_data)

    def objective_sense(self) -> ObjectiveSense:
        """
        Get the objective sense(s) of this Solution's associated Problem.

        If the problem is single-objective, then a single string is returned.
        If the problem is multi-objective, then the objective senses will be
        returned in a list.

        The returned string in the single-objective case, or each returned
        string in the multi-objective case, is "min" or "max".
        """
        return self._batch.objective_sense

    @property
    def senses(self) -> Iterable[str]:
        """
        Objective sense(s) of this Solution's associated Problem.

        This is a list of strings, each string being "min" or "max".
        """
        return self._batch.senses

    @property
    def is_evaluated(self) -> bool:
        """
        Whether or not the Solution is fully evaluated.

        This property returns True only when all of the evaluation results
        for all objectives have numeric values other than NaN.

        This property assumes that the extra evaluation data is optional,
        and therefore does not take into consideration whether or not the
        extra evaluation data contains NaN values.
        In other words, while determining whether or not a solution is fully
        evaluated, only the evaluation results corresponding to the
        objectives are taken into account.
        """
        num_objs = len(self.senses)
        with torch.no_grad():
            return not bool(torch.any(torch.isnan(self._batch.evals[0, :num_objs])))

    @property
    def dtype(self) -> DType:
        """
        dtype of the decision values
        """
        return self._batch.dtype

    @property
    def device(self) -> Device:
        """
        The device storing the Solution
        """
        return self._batch.device

    @property
    def eval_dtype(self) -> DType:
        """
        dtype of the evaluation results
        """
        return self._batch.eval_dtype

    @staticmethod
    def _rightmost_shape(shape: Iterable) -> torch.Size:
        if len(shape) >= 2:
            return torch.Size([int(shape[-1])])
        else:
            return torch.Size([])

    @property
    def shape(self) -> torch.Size:
        """
        Shape of the decision values of the Solution
        """
        return self._rightmost_shape(self._batch.values_shape)

    def size(self) -> torch.Size:
        """
        Shape of the decision values of the Solution
        """
        return self.shape

    @property
    def eval_shape(self) -> torch.Size:
        """
        Shape of the evaluation results
        """
        return self._rightmost_shape(self._batch.eval_shape)

    @property
    def ndim(self) -> int:
        """
        Number of dimensions of the decision values.

        For numeric solutions (e.g. of dtype `torch.float32`), this returns
        1, since such numeric solutions are kepts as 1-dimensional vectors.

        When dtype is `object`, `ndim` is reported as whatever the contained
        object reports as its `ndim`, or 0 if the contained object does not
        have an `ndim` attribute.
        """
        values = self.values
        if hasattr(values, "ndim"):
            return values.ndim
        else:
            return 0

    def dim(self) -> int:
        """
        This method returns the `ndim` attribute of this Solution.
        """
        return self.ndim

    def __len__(self) -> int:
        return len(self.values)

    def __iter__(self):
        return self.values.__iter__()

    def __reversed__(self):
        return self.values.__reversed__()

    def __getitem__(self, i):
        return self.values.__getitem__(i)

    def _to_string(self) -> str:
        clsname = type(self).__name__
        result = []

        values = self._batch.access_values(keep_evals=True)[0]
        evals = self._batch.access_evals()[0]

        def write(*args):
            for arg in args:
                result.append(str(arg))

        write("<", clsname, " values=", values)
        if not torch.all(torch.isnan(evals)):
            write(", evals=", evals)
        write(">")
        return "".join(result)

    def __repr__(self) -> str:
        return self._to_string()

    def __str__(self) -> str:
        return self._to_string()

    def _get_cloned_state(self, *, memo: dict) -> dict:
        with _no_grad_if_basic_dtype(self.dtype):
            return deep_clone(
                self.__dict__,
                otherwise_deepcopy=True,
                memo=memo,
            )

    def to(self, device: Device) -> "Solution":
        """
        Get the counterpart of this Solution on the new device.

        If the specified device is the device of this Solution,
        then this Solution itself is returned.
        If the specified device is a different device, then a clone
        of this Solution on this different device is first
        created, and then this new clone is returned.

        Please note that the `to(...)` method is not supported when
        the dtype is `object`.

        Args:
            device: The device on which the resulting Solution
                will be stored.
        Returns:
            The Solution on the specified device.
        """
        return Solution(self._batch.to(device), 0)

    def to_batch(self) -> SolutionBatch:
        """
        Get the single-row SolutionBatch counterpart of the Solution.
        The returned SolutionBatch and the Solution have shared
        storage, meaning that modifying one of them affects the other.

        Returns:
            The SolutionBatch counterpart of the Solution.
        """
        return self._batch


class ProblemBoundEvaluator:
    """
    A callable fitness evaluator, bound to the given `Problem`.

    A callable evaluator returned by the method
    `Problem.make_callable_evaluator` is an instance of this class.
    For details, please see the documentation of
    [Problem][evotorch.core.Problem], and of its method
    `make_callable_evaluator`.
    """

    def __init__(self, problem: Problem, *, obj_index: Optional[int] = None):
        """
        `__init__(...)`: Initialize the `ProblemBoundEvaluator`.

        Args:
            problem: The problem object to be wrapped.
            obj_index: The objective index. Optional if the problem being
                wrapped is single-objective. If the problem being wrapped
                is multi-objective, this is expected as an integer.
        """
        self._problem = problem
        if not isinstance(self._problem, Problem):
            clsname = type(self).__name__
            raise TypeError(
                f"In its initialization phase, {clsname} expected a `Problem` object,"
                f" but found: {repr(self._problem)} (of type {repr(type(self._problem))})"
            )
        self._obj_index = self._problem.normalize_obj_index(obj_index)
        self._problem.ensure_numeric()
        if problem.dtype != problem.eval_dtype:
            raise TypeError(
                "The dtype of the decision values is not the same with the dtype of the evaluations."
                " Currently, it is not supported to make callable evaluators out of problems whose"
                " decision value dtypes are different than their evaluation dtypes."
            )

    def _make_empty_solution_batch(self, popsize: int) -> SolutionBatch:
        return SolutionBatch(self._problem, popsize=popsize, empty=True, device="meta")

    def _prepare_evaluated_solution_batch(self, values_2d: torch.Tensor) -> SolutionBatch:
        num_solutions, solution_length = values_2d.shape
        batch = self._make_empty_solution_batch(num_solutions)
        batch._data = values_2d
        batch._evdata = torch.empty_like(batch._evdata, device=values_2d.device)
        self._problem.evaluate(batch)
        return batch

    def __call__(self, values: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the solutions expressed by the given `values` tensor.

        Args:
            values: Decision values. Expected as a tensor with at least
                2 dimensions. If the number of dimensions is more than 2,
                the extra leftmost dimensions will be considered as batch
                dimensions.
        Returns:
            The fitnesses, as a tensor.
        """
        ndim = values.ndim
        if ndim == 0:
            clsname = type(self).__name__
            raise ValueError(
                f"{clsname} was expecting a tensor with at least 1 dimension for solution evaluation."
                f" However, it received a scalar: {values}"
            )

        solution_length = values.shape[-1]
        original_batch_shape = values.shape[:-1]

        values = values.reshape(-1, solution_length)
        evaluated_batch = self._prepare_evaluated_solution_batch(values)
        evals = evaluated_batch.evals[:, self._obj_index]

        return evals.reshape(original_batch_shape).as_subclass(torch.Tensor)
