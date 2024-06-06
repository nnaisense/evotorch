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
Genetic algorithm variants: GeneticAlgorithm, Cosyne.
"""

from typing import Callable, Iterable, Optional, Union

from ..core import Problem, SolutionBatch
from ..operators import CosynePermutation, CrossOver, GaussianMutation, OnePointCrossOver, SimulatedBinaryCrossOver
from .searchalgorithm import SearchAlgorithm, SinglePopulationAlgorithmMixin


def _use_operator(batch: SolutionBatch, operator: Callable) -> SolutionBatch:
    from ..operators import CopyingOperator, Operator
    from ..tools import is_dtype_object

    if isinstance(operator, CopyingOperator):
        result = operator(batch)
    elif isinstance(operator, Operator):
        result = batch.clone()
        operator(result)
    else:
        cloned_batch = batch.clone()
        original_values = cloned_batch.access_values(keep_evals=True)
        new_values = operator(original_values)
        if not is_dtype_object(new_values.dtype):
            if new_values.ndim != 2:
                raise ValueError(
                    "The tensor returned by the given operator was expected to have 2 dimensions."
                    f" However, it has {new_values.ndim} dimensions, with shape {new_values.shape}."
                )
        n = len(new_values)
        if n == len(cloned_batch):
            cloned_batch.set_values(new_values)
            result = cloned_batch
        else:
            result = SolutionBatch(popsize=n, like=batch, empty=True)
            result.set_values(new_values)

    return result


def _use_operators(batch: SolutionBatch, operators: Iterable) -> SolutionBatch:
    for operator in operators:
        batch = _use_operator(batch, operator)
    return batch


class ExtendedPopulationMixin:
    """
    A mixin class that provides the method `_make_extended_population(...)`.

    This mixin class assumes that the inheriting class has the properties
    `problem` (of type [Problem][evotorch.core.Problem]), which provide
    and `population` (of type [SolutionBatch][evotorch.core.SolutionBatch]),
    which provide the associated problem object and the current population,
    respectively.

    The class which inherits this mixin class gains the method
    `_make_extended_population(...)`. This new method applies the operators
    specified during the initialization phase of this mixin class on the
    current population, produces children, and then returns an extended
    population.
    """

    def __init__(
        self,
        *,
        re_evaluate: bool,
        re_evaluate_parents_first: Optional[bool] = None,
        operators: Optional[Iterable] = None,
        allow_empty_operators_list: bool = False,
    ):
        """
        `__init__(...)`: Initialize the ExtendedPopulationMixin.

        Args:
            re_evaluate: Whether or not to re-evaluate the parent population
                at every generation. When dealing with problems where the
                fitness and/or feature evaluations are stochastic, one might
                want to set this as True. On the other hand, for when the
                fitness and/or feature evaluations are deterministic, one
                might prefer to set this as False for efficiency.
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
            operators: List of operators to apply on the current population
                for generating a new extended population.
            allow_empty_operators_list: Whether or not to allow the operator
                list to be empty. The default and the recommended value
                is False. For cases where the inheriting class wants to
                decide the operators later (via the attribute `_operators`)
                this can be set as True.
        """
        self._operators = [] if operators is None else list(operators)
        if (not allow_empty_operators_list) and (len(self._operators) == 0):
            raise ValueError("Received `operators` as an empty sequence. Please provide at least one operator.")

        self._using_cross_over: bool = False
        for operator in self._operators:
            if isinstance(operator, CrossOver):
                self._using_cross_over = True
                break

        self._re_evaluate: bool = bool(re_evaluate)

        if re_evaluate_parents_first is None:
            self._re_evaluate_parents_first = self._using_cross_over
        else:
            if not self._re_evaluate:
                raise ValueError(
                    "Encountered the argument `re_evaluate_parents_first` as something other than None."
                    " However, `re_evaluate` is given as False."
                    " Please use `re_evaluate_parents_first` only when `re_evaluate` is True."
                )
            self._re_evaluate_parents_first = bool(re_evaluate_parents_first)

        self._first_iter: bool = True

    def _make_extended_population(self, split: bool = False) -> Union[SolutionBatch, tuple]:
        """
        Make and return a new extended population that is evaluated.

        Args:
            split: If False, then the extended population will be returned
                as a single SolutionBatch object which contains both the
                parents and the children.
                If True, then the extended population will be returned
                as a pair of SolutionBatch objects, the first one being
                the parents and the second one being the children.
        Returns:
            The extended population.
        """

        # Get the problem object and the population
        problem: Problem = self.problem
        population: SolutionBatch = self.population

        if self._re_evaluate:
            # This is the case where our mixin is configured to re-evaluate the parents at every generation.

            # Set the first iteration indicator to False
            self._first_iter = False

            if self._re_evaluate_parents_first:
                # This is the case where our mixin is configured to evaluate the parents separately first.
                # This is a sub-case of `_re_evaluate=True`.

                # Evaluate the population, which stores the parents
                problem.evaluate(population)

                # Now that our parents are evaluated, we use the operators on them and get the children.
                children = _use_operators(population, self._operators)

                # Evaluate the children
                problem.evaluate(children)

                if split:
                    # If our mixin is configured to return the population and the children, then we return a tuple
                    # containing them as separate items.
                    return population, children
                else:
                    # If our mixin is configured to return the population and the children in a single batch,
                    # then we concatenate the population and the children and return the resulting combined batch.
                    return SolutionBatch.cat([population, children])
            else:
                # This is the case where our mixin is configured to evaluate the parents and the children in one go.
                # This is a sub-case of `_re_evaluate=True`.

                # Use the operators on the parent solutions. It does not matter whether or not the parents are evaluated.
                children = _use_operators(population, self._operators)

                # Form an extended population by concatenating the population and the children.
                extended_population = SolutionBatch.cat([population, children])

                # Evaluate the extended population in one go.
                problem.evaluate(extended_population)

                if split:
                    # The method was configured to return the parents and the children separately.
                    # Because we combined them earlier for evaluating them in one go, we will split them now.

                    # Get the number of parents
                    num_parents = len(population)

                    # Get the newly evaluated copies of the parents from the extended population
                    parents = extended_population[:num_parents]

                    # Get the children from the extended population
                    children = extended_population[num_parents:]

                    # Return the newly evaluated copies of the parents and the children separately.
                    return parents, children
                else:
                    # The method was configured to return the parents and the children in a single SolutionBatch.
                    # Here, we just return the extended population that we already have produced.
                    return extended_population
        else:
            # This is the case where our mixin is configured NOT to re-evaluate the parents at every generation.
            if self._first_iter:
                # The first iteration indicator (`_first_iter`) is True. So, this is the first iteration.
                # We set `_first_iter` to False for future generations.
                self._first_iter = False
                # We not evaluate the parent population (because the parents are expected to be non-evaluated at the
                # beginning).
                problem.evaluate(population)

            # Here, we assume that the parents are already evaluated. We apply our operators on the parents.
            children = _use_operators(population, self._operators)

            # Now, we evaluate the children.
            problem.evaluate(children)

            if split:
                # Return the population and the children separately if `split=True`.
                return population, children
            else:
                # Return the population and the children in a single SolutionBatch if `split=False`.
                return SolutionBatch.cat([population, children])

    @property
    def re_evaluate(self) -> bool:
        """
        Whether or not this search algorithm re-evaluates the parents
        """
        return self._re_evaluate

    @property
    def re_evaluate_parents_first(self) -> Optional[bool]:
        """
        Whether or not this search algorithm re-evaluates the parents separately.
        This property is relevant only when `re_evaluate` is True.
        If `re_evaluate` is False, then this property will return None.
        """
        if self._re_evaluate:
            return self._re_evaluate_parents_first
        else:
            return None


class GeneticAlgorithm(SearchAlgorithm, SinglePopulationAlgorithmMixin, ExtendedPopulationMixin):
    """
    A genetic algorithm implementation.

    **Basic usage.**
    Let us consider a single-objective optimization problem where the goal is to
    minimize the L2 norm of a continuous tensor of length 10:

    ```python
    from evotorch import Problem
    from evotorch.algorithms import GeneticAlgorithm
    from evotorch.operators import OnePointCrossOver, GaussianMutation

    import torch


    def f(x: torch.Tensor) -> torch.Tensor:
        return torch.linalg.norm(x)


    problem = Problem(
        "min",
        f,
        initial_bounds=(-10.0, 10.0),
        solution_length=10,
    )
    ```

    For solving this problem, a genetic algorithm could be instantiated as
    follows:

    ```python
    ga = GeneticAlgorithm(
        problem,
        popsize=100,
        operators=[
            OnePointCrossOver(problem, tournament_size=4),
            GaussianMutation(problem, stdev=0.1),
        ],
    )
    ```

    The genetic algorithm instantiated above is configured to have a population
    size of 100, and is configured to perform the following operations on the
    population at each generation:
    (i) select solutions with a tournament of size 4, and produce children from
    the selected solutions by applying one-point cross-over;
    (ii) apply a gaussian mutation on the values of the solutions produced by
    the previous step, the amount of the mutation being sampled according to a
    standard deviation of 0.1.
    Once instantiated, this GeneticAlgorithm instance can be used with an API
    compatible with other search algorithms, as shown below:

    ```python
    from evotorch.logging import StdOutLogger

    _ = StdOutLogger(ga)  # Report the evolution's progress to standard output
    ga.run(100)  # Run the algorithm for 100 generations
    print("Solution with best fitness ever:", ga.status["best"])
    print("Current population's best:", ga.status["pop_best"])
    ```

    Please also note:

    - The operators are always executed according to the order specified within
      the `operators` argument.
    - There are more operators available in the namespace
      [evotorch.operators][evotorch.operators].
    - By default, GeneticAlgorithm is elitist. In the elitist mode, an extended
      population is formed from parent solutions and child solutions, and the
      best n solutions of this extended population are accepted as the next
      generation. If you wish to switch to a non-elitist mode (where children
      unconditionally replace the worst-performing parents), you can use the
      initialization argument `elitist=False`.
    - It is not mandatory to specify a cross-over operator. When a cross-over
      operator is missing, the GeneticAlgorithm will work like a simple
      evolution strategy implementation which produces children by mutating
      the parents, and then replaces the parents (where the criteria for
      replacing the parents depend on whether or not elitism is enabled).
    - To be able to deal with stochastic fitness functions correctly,
      GeneticAlgorithm re-evaluates previously evaluated parents as well.
      When you are sure that the fitness function is deterministic,
      you can pass the initialization argument `re_evaluate=False` to prevent
      unnecessary computations.

    **Integer decision variables.**
    GeneticAlgorithm can be used on problems with `dtype` declared as integer
    (e.g. `torch.int32`, `torch.int64`, etc.).
    Within the field of discrete optimization, it is common to encounter
    one or more of these scenarios:

    - The search space of the problem has a special structure that one will
      wish to exploit (within the cross-over and/or mutation operators) to
      be able to reach the (near-)optimum within a reasonable amount of time.
    - The problem is partially or fully combinatorial.
    - The problem is constrained in such a way that arbitrarily sampling
      discrete values for its decision variables might cause infeasibility.

    Considering all these scenarios, it is difficult to come up with general
    cross-over and mutation operators that will work across various discrete
    optimization problems, and it is common to design problem-specific
    operators. In EvoTorch, it is possible to define custom operators and
    use them with GeneticAlgorithm, which is required when using
    GeneticAlgorithm on a problem with a non-float dtype.

    As an example, let us consider the following discrete optimization
    problem:

    ```python
    def f(x: torch.Tensor) -> torch.Tensor:
        return torch.sum(x)


    problem = Problem(
        "min",
        f,
        bounds=(-10, 10),
        solution_length=10,
        dtype=torch.int64,
    )
    ```

    Although EvoTorch does provide a very simple and generic (usable with float
    and int dtypes) cross-over named
    [OnePointCrossOver][evotorch.operators.real.OnePointCrossOver]
    (a cross-over which randomly decides a cutting point for each pair of
    parents, cuts them from those points and recombines them), it can be
    desirable and necessary to implement a custom cross-over operator.
    One can inherit from [CrossOver][evotorch.operators.base.CrossOver] to
    define a custom cross-over operator, as shown below:

    ```python
    from evotorch import SolutionBatch
    from evotorch.operators import CrossOver


    class CustomCrossOver(CrossOver):
        def _do_cross_over(
            self,
            parents1: torch.Tensor,
            parents2: torch.Tensor,
        ) -> SolutionBatch:
            # parents1 is a tensor storing the decision values of the first
            # half of the chosen parents.
            # parents2 is a tensor storing the decision values of the second
            # half of the chosen parents.

            # We expect that the lengths of parents1 and parents2 are equal.
            assert len(parents1) == len(parents2)

            # Allocate an empty SolutionBatch that will store the children
            childpop = SolutionBatch(self.problem, popsize=num_parents, empty=True)

            # Gain access to the decision values tensor of the newly allocated
            # childpop
            childpop_values = childpop.access_values()

            # Here we somehow fill `childpop_values` by recombining the parents.
            # The most common thing to do is to produce two children by
            # combining parents1[0] and parents2[0], to produce the next two
            # children parents1[1] and parents2[1], and so on.
            childpop_values[:] = ...

            # Return the child population
            return childpop
    ```

    One can define a custom mutation operator by inheriting from
    [Operator][evotorch.operators.base.Operator], as shown below:

    ```python
    class CustomMutation(Operator):
        def _do(self, solutions: SolutionBatch):
            # Get the decision values tensor of the solutions
            sln_values = solutions.access_values()

            # do in-place modifications to the decision values
            sln_values[:] = ...
    ```

    Alternatively, you could define the mutation operator as a function:

    ```python
    def my_mutation_function(original_values: torch.Tensor) -> torch.Tensor:
        # Somehow produce mutated copies of the original values
        mutated_values = ...

        # Return the mutated values
        return mutated_values
    ```

    With these defined operators, we are now ready to instantiate our
    GeneticAlgorithm:

    ```python
    ga = GeneticAlgorithm(
        problem,
        popsize=100,
        operators=[
            CustomCrossOver(problem, tournament_size=4),
            CustomMutation(problem),
            # -- or, if you chose to define the mutation as a function: --
            # my_mutation_function,
        ],
    )
    ```

    **Non-numeric or variable-length solutions.**
    GeneticAlgorithm can also work on problems whose `dtype` is declared
    as `object`, where `dtype=object` means that a solution's value(s) can be
    expressed via a tensor, a numpy array, a scalar, a tuple, a list, a
    dictionary.

    Like in the previously discussed case (where dtype is an integer type),
    one has to define custom operators when working on problems with
    `dtype=object`. A custom cross-over definition specialized for
    `dtype=object` looks like this:

    ```python
    from evotorch.tools import ObjectArray


    class CrossOverForObjectDType(CrossOver):
        def _do_cross_over(
            self,
            parents1: ObjectArray,
            parents2: ObjectArray,
        ) -> SolutionBatch:
            # Allocate an empty SolutionBatch that will store the children
            childpop = SolutionBatch(self.problem, popsize=num_parents, empty=True)

            # Gain access to the decision values ObjectArray of the newly allocated
            # childpop
            childpop_values = childpop.access_values()

            # Here we somehow fill `childpop_values` by recombining the parents.
            # The most common thing to do is to produce two children by
            # combining parents1[0] and parents2[0], to produce the next two
            # children parents1[1] and parents2[1], and so on.
            childpop_values[:] = ...

            # Return the child population
            return childpop
    ```

    A custom mutation operator specialized for `dtype=object` looks like this:

    ```python
    class MutationForObjectDType(Operator):
        def _do(self, solutions: SolutionBatch):
            # Get the decision values ObjectArray of the solutions
            sln_values = solutions.access_values()

            # do in-place modifications to the decision values
            sln_values[:] = ...
    ```

    A custom mutation function specialized for `dtype=object` looks like this:

    ```python
    def mutation_for_object_dtype(original_values: ObjectArray) -> ObjectArray:
        # Somehow produce mutated copies of the original values
        mutated_values = ...

        # Return the mutated values
        return mutated_values
    ```

    With these operators defined, one can instantiate the GeneticAlgorithm:

    ```python
    ga = GeneticAlgorithm(
        problem_with_object_dtype,
        popsize=100,
        operators=[
            CrossOverForObjectDType(problem_with_object_dtype, tournament_size=4),
            MutationForObjectDType(problem_with_object_dtype),
            # -- or, if you chose to define the mutation as a function: --
            # mutation_for_object_dtype,
        ],
    )
    ```

    **Multiple objectives.**
    GeneticAlgorithm can work on problems with multiple objectives.
    When there are multiple objectives, GeneticAlgorithm will compare the
    solutions according to their pareto-ranks and their crowding distances,
    like done by the NSGA-II algorithm (Deb, 2002).

    References:

        Sean Luke, 2013, Essentials of Metaheuristics, Lulu, second edition
        available for free at http://cs.gmu.edu/~sean/book/metaheuristics/

        Kalyanmoy Deb, Amrit Pratap, Sameer Agarwal, T. Meyarivan (2002).
        A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II.
    """

    def __init__(
        self,
        problem: Problem,
        *,
        operators: Iterable,
        popsize: int,
        elitist: bool = True,
        re_evaluate: bool = True,
        re_evaluate_parents_first: Optional[bool] = None,
        _allow_empty_operator_list: bool = False,
    ):
        """
        `__init__(...)`: Initialize the GeneticAlgorithm.

        Args:
            problem: The problem to optimize.
            operators: Operators to be used by the genetic algorithm.
                Expected as an iterable, such as a list or a tuple.
                Each item within this iterable object is expected either
                as an instance of [Operator][evotorch.operators.base.Operator],
                or as a function which receives the decision values of
                multiple solutions in a PyTorch tensor (or in an
                [ObjectArray][evotorch.tools.objectarray.ObjectArray]
                for when dtype is `object`) and returns a modified copy.
            popsize: Population size.
            elitist: Whether or not this genetic algorithm will behave in an
                elitist manner. This argument controls how the genetic
                algorithm will form the next generation from the parents
                and the children. In elitist mode (i.e. with `elitist=True`),
                the procedure to be followed by this genetic algorithm is:
                (i) form an extended population which consists of
                both the parents and the children,
                (ii) sort the extended population from best to worst,
                (iii) select the best `n` solutions as the new generation where
                `n` is `popsize`.
                In non-elitist mode (i.e. with `elitist=False`), the worst `m`
                solutions within the parent population are replaced with
                the children, `m` being the number of produced children.
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
        SearchAlgorithm.__init__(self, problem)

        self._popsize = int(popsize)
        self._elitist: bool = bool(elitist)
        self._population = problem.generate_batch(self._popsize)

        ExtendedPopulationMixin.__init__(
            self,
            re_evaluate=re_evaluate,
            re_evaluate_parents_first=re_evaluate_parents_first,
            operators=operators,
            allow_empty_operators_list=_allow_empty_operator_list,
        )

        SinglePopulationAlgorithmMixin.__init__(self)

    @property
    def population(self) -> SolutionBatch:
        """Get the population"""
        return self._population

    def _step(self):
        # Get the population size
        popsize = self._popsize

        if self._elitist:
            # This is where we handle the elitist mode.

            # Produce and get an extended population in a single SolutionBatch
            extended_population = self._make_extended_population(split=False)

            # From the extended population, take the best n solutions, n being the popsize.
            self._population = extended_population.take_best(popsize)
        else:
            # This is where we handle the non-elitist mode.

            # Take the parent solutions (ensured to be evaluated) and the children separately.
            parents, children = self._make_extended_population(split=True)

            # Get the number of children
            num_children = len(children)

            if num_children < popsize:
                # If the number of children is less than the population size, then we keep the best m solutions from
                # the parents, m being `popsize - num_children`.
                chosen_parents = self._population.take_best(popsize - num_children)

                # Combine the children with the chosen parents, and declare them as the new population.
                self._population = SolutionBatch.cat([chosen_parents, children])
            elif num_children == popsize:
                # If the number of children is the same with the population size, then these children are declared as
                # the new population.
                self._population = children
            else:
                # If the number of children is more than the population size, then we take the best n solutions from
                # these children, n being the population size. These chosen children are then declared as the new
                # population.
                self._population = children.take_best(self._popsize)


class SteadyStateGA(GeneticAlgorithm):
    """
    Thin wrapper around GeneticAlgorithm for compatibility with old code.

    This `SteadyStateGA` class is equivalent to
    [GeneticAlgorithm][evotorch.algorithms.ga.GeneticAlgorithm] except that
    `SteadyStateGA` provides an additional method named `use(...)` for
    specifying a cross-over and/or a mutation operator.
    The method `use(...)` exists only for API compatibility with the previous
    versions of EvoTorch. It is recommended to specify the operators via
    the keyword argument `operators` instead.
    """

    def __init__(
        self,
        problem: Problem,
        *,
        popsize: int,
        operators: Optional[Iterable] = None,
        elitist: bool = True,
        re_evaluate: bool = True,
        re_evaluate_parents_first: Optional[bool] = None,
    ):
        """
        `__init__(...)`: Initialize the SteadyStateGA.

        Args:
            problem: The problem to optimize.
            operators: Optionally, an iterable of operators to be used by the
                genetic algorithm. Each item within this iterable object is
                expected either as an instance of
                [Operator][evotorch.operators.base.Operator],
                or as a function which receives the decision values of
                multiple solutions in a PyTorch tensor (or in an
                [ObjectArray][evotorch.tools.objectarray.ObjectArray]
                for when dtype is `object`) and returns a modified copy.
                If this is omitted, then it will be required to specify the
                operators via the `use(...)` method.
            popsize: Population size.
            elitist: Whether or not this genetic algorithm will behave in an
                elitist manner. This argument controls how the genetic
                algorithm will form the next generation from the parents
                and the children. In elitist mode (i.e. with `elitist=True`),
                the procedure to be followed by this genetic algorithm is:
                (i) form an extended population which consists of
                both the parents and the children,
                (ii) sort the extended population from best to worst,
                (iii) select the best `n` solutions as the new generation where
                `n` is `popsize`.
                In non-elitist mode (i.e. with `elitist=False`), the worst `m`
                solutions within the parent population are replaced with
                the children, `m` being the number of produced children.
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
                Additional note specific to `SteadyStateGA`: if the argument
                `operators` is not given (or is given as an empty list), and
                also `re_evaluate_parents_first` is left as None, then
                `SteadyStateGA` will assume that the operators will be later
                given via the `use(...)` method, and that these operators will
                require the parents to be evaluated first (equivalent to
                setting `re_evaluate_parents_first` as True).
        """
        if operators is None:
            operators = []

        self._cross_over_op: Optional[Callable] = None
        self._mutation_op: Optional[Callable] = None
        self._forbid_use_method: bool = False
        self._prepare_ops: bool = False

        if (len(operators) == 0) and re_evaluate and (re_evaluate_parents_first is None):
            re_evaluate_parents_first = True

        super().__init__(
            problem,
            operators=operators,
            popsize=popsize,
            elitist=elitist,
            re_evaluate=re_evaluate,
            re_evaluate_parents_first=re_evaluate_parents_first,
            _allow_empty_operator_list=True,
        )

    def use(self, operator: Callable):
        """
        Specify the cross-over or the mutation operator to use.

        This method exists for compatibility with previous EvoTorch code.
        Instead of using this method, it is recommended to specify the
        operators via the `operators` keyword argument while initializing
        this class.

        Using this method, one can specify one cross-over operator and one
        mutation operator that will be used during the evolutionary search.
        Specifying multiple cross-over operators or multiple mutation operators
        is not allowed. When the cross-over and mutation operators are
        specified via `use(...)`, the order of execution will always be
        arranged such that the cross-over comes first and the mutation comes
        comes second. If desired, one can specify only the cross-over operator
        or only the mutation operator.

        Please note that the `operators` keyword argument works differently,
        and offers more flexibility for defining the procedure to follow at
        each generation. In more details, the `operators` keyword argument
        allows one to specify multiple cross-over and/or multiple mutation
        operators, and those operators will be executed in the specified
        order.

        Args:
            operator: The operator to be registered to SteadyStateGA.
                If the specified operator is cross-over (i.e. an instance
                of [CrossOver][evotorch.operators.base.CrossOver]),
                then this operator will be registered for the cross-over
                phase. If the specified operator is an operator that is
                not of the cross-over type (i.e. any instance of
                [Operator][evotorch.operators.base.Operator] that is not
                [CrossOver][evotorch.operators.base.CrossOver]) or if it is
                just a function which receives the decision values as a PyTorch
                tensor (or, in the case where `dtype` of the problem is
                `object` as an instance of
                [ObjectArray][evotorch.tools.objectarray.ObjectArray]) and
                returns a modified copy, then that operator will be registered
                for the mutation phase of the genetic algorithm.
        """
        if self._forbid_use_method:
            raise RuntimeError(
                "The method `use(...)` cannot be called anymore, because the evolutionary search has started."
            )

        if len(self._operators) > 0:
            raise RuntimeError(
                f"The method `use(...)` cannot be called"
                f" because an operator list was provided while initializing this {type(self).__name__} instance."
            )

        if isinstance(operator, CrossOver):
            if self._cross_over_op is not None:
                raise ValueError(
                    f"The method `use(...)` received this cross-over operator as its argument:"
                    f" {operator} (of type {type(operator)})."
                    f" However, a cross-over operator was already set:"
                    f" {self._cross_over_op} (of type {type(self._cross_over_op)})."
                )
            self._cross_over_op = operator
            self._prepare_ops = True
        else:
            if self._mutation_op is not None:
                raise ValueError(
                    f"The method `use(...)` received this mutation operator as its argument:"
                    f" {operator} (of type {type(operator)})."
                    f" However, a mutation operator was already set:"
                    f" {self._mutation_op} (of type {type(self._mutation_op)})."
                )
            self._mutation_op = operator
            self._prepare_ops = True

    def _step(self):
        self._forbid_use_method = True

        if self._prepare_ops:
            self._prepare_ops = False
            if self._cross_over_op is not None:
                self._operators.append(self._cross_over_op)
            if self._mutation_op is not None:
                self._operators.append(self._mutation_op)
        else:
            if len(self._operators) == 0:
                raise RuntimeError(
                    f"This {type(self).__name__} instance does not know how to proceed, "
                    f" because neither the `operators` keyword argument was used during initialization"
                    f" nor was the `use(...)` method called later."
                )

        super()._step()


class Cosyne(SearchAlgorithm, SinglePopulationAlgorithmMixin):
    """
    Implementation of the CoSyNE algorithm.

    References:

        F.Gomez, J.Schmidhuber, R.Miikkulainen, M.Mitchell (2008).
        Accelerated Neural Evolution through Cooperatively Coevolved Synapses.
        Journal of Machine Learning Research 9 (5).
    """

    def __init__(
        self,
        problem: Problem,
        *,
        popsize: int,
        tournament_size: int,
        mutation_stdev: Optional[float],
        mutation_probability: Optional[float] = None,
        permute_all: bool = False,
        num_elites: Optional[int] = None,
        elitism_ratio: Optional[float] = None,
        eta: Optional[float] = None,
        num_children: Optional[int] = None,
    ):
        """
        `__init__(...)`: Initialize the Cosyne instance.

        Args:
            problem: The problem object to work on.
            popsize: Population size, as an integer.
            tournament_size: Tournament size, for tournament selection.
            mutation_stdev: Standard deviation of the Gaussian mutation.
                See [GaussianMutation][evotorch.operators.real.GaussianMutation] for more information.
            mutation_probability: Elementwise Gaussian mutation probability.
                Defaults to None.
                See [GaussianMutation][evotorch.operators.real.GaussianMutation] for more information.
            permute_all: If given as True, all solutions are subject to
                permutation. If given as False (which is the default),
                there will be a selection procedure for each decision
                variable.
            num_elites: Optionally expected as an integer, specifying the
                number of elites to pass to the next generation.
                Cannot be used together with the argument `elitism_ratio`.
            elitism_ratio: Optionally expected as a real number between
                0 and 1, specifying the amount of elites to pass to the
                next generation. For example, 0.1 means that the best 10%
                of the population are accepted as elites and passed onto
                the next generation.
                Cannot be used together with the argument `num_elites`.
            eta: Optionally expected as an integer, specifying the eta
                hyperparameter for the simulated binary cross-over (SBX).
                If left as None, one-point cross-over will be used instead.
            num_children: Number of children to generate at each iteration.
                If left as None, then this number is half of the population
                size.
        """

        problem.ensure_numeric()

        SearchAlgorithm.__init__(self, problem)

        if mutation_stdev is None:
            if mutation_probability is not None:
                raise ValueError(
                    f"`mutation_probability` was set to {mutation_probability}, but `mutation_stdev` is None, "
                    "which means, mutation is disabled. If you want to enable the mutation, be sure to provide "
                    "`mutation_stdev` as well."
                )
            self.mutation_op = None
        else:
            self.mutation_op = GaussianMutation(
                self._problem,
                stdev=mutation_stdev,
                mutation_probability=mutation_probability,
            )

        cross_over_kwargs = {"tournament_size": tournament_size}
        if num_children is None:
            cross_over_kwargs["cross_over_rate"] = 2.0
        else:
            cross_over_kwargs["num_children"] = num_children

        if eta is None:
            self._cross_over_op = OnePointCrossOver(self._problem, **cross_over_kwargs)
        else:
            self._cross_over_op = SimulatedBinaryCrossOver(self._problem, eta=eta, **cross_over_kwargs)

        self._permutation_op = CosynePermutation(self._problem, permute_all=permute_all)

        self._popsize = int(popsize)

        if num_elites is not None and elitism_ratio is None:
            self._num_elites = int(num_elites)
        elif num_elites is None and elitism_ratio is not None:
            self._num_elites = int(self._popsize * elitism_ratio)
        elif num_elites is None and elitism_ratio is None:
            self._num_elites = None
        else:
            raise ValueError(
                "Received both `num_elites` and `elitism_ratio`. Please provide only one of them, or none of them."
            )

        self._population = SolutionBatch(problem, device=problem.device, popsize=self._popsize)
        self._first_generation: bool = True

        # GAStatusMixin.__init__(self)
        SinglePopulationAlgorithmMixin.__init__(self)

    @property
    def population(self) -> SolutionBatch:
        return self._population

    def _step(self):
        if self._first_generation:
            self._first_generation = False
            self._problem.evaluate(self._population)

        to_merge = []

        num_elites = self._num_elites
        num_parents = int(self._popsize / 4)
        num_relevant = max((0 if num_elites is None else num_elites), num_parents)

        sorted_relevant = self._population.take_best(num_relevant)

        if self._num_elites is not None and self._num_elites >= 1:
            to_merge.append(sorted_relevant[:num_elites].clone())

        parents = sorted_relevant[:num_parents]
        children = self._cross_over_op(parents)
        if self.mutation_op is not None:
            children = self.mutation_op(children)

        permuted = self._permutation_op(self._population)

        to_merge.extend([children, permuted])

        extended_population = SolutionBatch(merging_of=to_merge)
        self._problem.evaluate(extended_population)
        self._population = extended_population.take_best(self._popsize)
