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
This namespace contains `SearchAlgorithm`, the base class for all
evolutionary algorithms.
"""

import io
from collections.abc import Mapping
from datetime import datetime
from typing import Any, Iterable, Optional

import torch

from ..core import Problem, SolutionBatch
from ..tools import clone
from ..tools.hook import Hook
from ..tools.objectarray import ObjectArray
from ..tools.readonlytensor import as_read_only_tensor


class LazyReporter:
    """
    This class provides an interface of storing and reporting status.
    This class is designed to be inherited by other classes.

    Let us assume that we have the following class inheriting from
    LazyReporter:

    ```python
    class Example(LazyReporter):
        def __init__(self):
            LazyReporter.__init__(self, a=self._get_a, b=self._get_b)

        def _get_a(self):
            return ...  # return the status 'a'

        def _get_b(self):
            return ...  # return the status 'b'
    ```

    At its initialization phase, this Example class registers its methods
    ``_get_a`` and ``_get_b`` as its status providers.
    Having the LazyReporter interface, the Example class gains a ``status``
    property:

    ```python
    ex = Example()
    print(ex.status["a"])  # Get the status 'a'
    print(ex.status["b"])  # Get the status 'b'
    ```

    Once a status is queried, its computation result is stored to be re-used
    later. After running the code above, if we query the status 'a' again:

    ```python
    print(ex.status["a"])  # Getting the status 'a' again
    ```

    then the status 'a' is not computed again (i.e. ``_get_a`` is not
    called again). Instead, the stored status value of 'a' is re-used.

    To force re-computation of the status values, one can execute:

    ```python
    ex.clear_status()
    ```

    Or the Example instance can clear its status from within one of its
    methods:

    ```python
    class Example(LazyReporter):
        ...

        def some_method(self):
            ...
            self.clear_status()
    ```
    """

    @staticmethod
    def _missing_status_producer():
        return None

    def __init__(self, **kwargs):
        """
        `__init__(...)`: Initialize the LazyReporter instance.

        Args:
            kwargs: Keyword arguments, mapping the status keys to the
                methods or functions providing the status values.
        """
        self.__getters = kwargs
        self.__computed = {}

    def get_status_value(self, key: Any) -> Any:
        """
        Get the specified status value.

        Args:
            key: The key (i.e. the name) of the status variable.
        """
        if key not in self.__computed:
            self.__computed[key] = self.__getters[key]()
        return self.__computed[key]

    def has_status_key(self, key: Any) -> bool:
        """
        Return True if there is a status variable with the specified key.
        Otherwise, return False.

        Args:
            key: The key (i.e. the name) of the status variable whose
                existence is to be checked.
        Returns:
            True if there is such a key; False otherwise.
        """
        return key in self.__getters

    def iter_status_keys(self):
        """Iterate over the status keys."""
        return self.__getters.keys()

    def clear_status(self):
        """Clear all the stored values of the status variables."""
        self.__computed.clear()

    def is_status_computed(self, key) -> bool:
        """
        Return True if the specified status is computed yet.
        Return False otherwise.

        Args:
            key: The key (i.e. the name) of the status variable.
        Returns:
            True if the status of the given key is computed; False otherwise.
        """
        return key in self.__computed

    def update_status(self, additional_status: Mapping):
        """
        Update the stored status with an external dict-like object.
        The given dict-like object can override existing status keys
        with new values, and also bring new keys to the status.

        Args:
            additional_status: A dict-like object storing the status update.
        """
        for k, v in additional_status.items():
            if k not in self.__getters:
                self.__getters[k] = LazyReporter._missing_status_producer
            self.__computed[k] = v

    def add_status_getters(self, getters: Mapping):
        """
        Register additional status-getting functions.

        Args:
            getters: A dictionary-like object where the keys are the
                additional status variable names, and values are functions
                which are expected to compute/retrieve the values for those
                status variables.
        """
        self.__getters.update(getters)

    @property
    def status(self) -> "LazyStatusDict":
        """Get a LazyStatusDict which is bound to this LazyReporter."""
        return LazyStatusDict(self)


class LazyStatusDict(Mapping):
    """
    A Mapping subclass used by the `status` property of a `LazyReporter`.

    The interface of this object is similar to a read-only dictionary.
    """

    def __init__(self, lazy_reporter: LazyReporter):
        """
        `__init__(...)`: Initialize the LazyStatusDict object.

        Args:
            lazy_reporter: The LazyReporter object whose status is to be
                accessed.
        """
        super().__init__()
        self.__lazy_reporter = lazy_reporter

    def __getitem__(self, key: Any) -> Any:
        result = self.__lazy_reporter.get_status_value(key)
        if isinstance(result, (torch.Tensor, ObjectArray)):
            result = as_read_only_tensor(result)
        return result

    def __len__(self) -> int:
        return len(list(self.__lazy_reporter.iter_status_keys()))

    def __iter__(self):
        for k in self.__lazy_reporter.iter_status_keys():
            yield k

    def __contains__(self, key: Any) -> bool:
        return self.__lazy_reporter.has_status_key(key)

    def _to_string(self) -> str:
        with io.StringIO() as f:
            print("<" + type(self).__name__, file=f)
            for k in self.__lazy_reporter.iter_status_keys():
                if self.__lazy_reporter.is_status_computed(k):
                    r = repr(self.__lazy_reporter.get_status_value(k))
                else:
                    r = "<not yet computed>"
                print("   ", k, "=", r, file=f)
            print(">", end="", file=f)
            f.seek(0)
            entire_str = f.read()
        return entire_str

    def __str__(self) -> str:
        return self._to_string()

    def __repr__(self) -> str:
        return self._to_string()


class SearchAlgorithm(LazyReporter):
    """
    Base class for all evolutionary search algorithms.

    An algorithm developer is expected to inherit from this base class,
    and override the method named `_step()` to define how a single
    step of this new algorithm is performed.

    For each core status dictionary element, a new method is expected
    to exist within the inheriting class. These status reporting
    methods are then registered via the keyword arguments of the
    `__init__(...)` method of `SearchAlgorithm`.

    To sum up, a newly developed algorithm inheriting from this base
    class is expected in this structure:

    ```python
    from evotorch import Problem


    class MyNewAlgorithm(SearchAlgorithm):
        def __init__(self, problem: Problem):
            SearchAlgorithm.__init__(
                self, problem, status1=self._get_status1, status2=self._get_status2, ...
            )

        def _step(self):
            # Code that defines how a step of this algorithm
            # should work goes here.
            ...

        def _get_status1(self):
            # The value returned by this function will be shown
            # in the status dictionary, associated with the key
            # 'status1'.
            return ...

        def _get_status2(self):
            # The value returned by this function will be shown
            # in the status dictionary, associated with the key
            # 'status2'.
            return ...
    ```
    """

    def __init__(self, problem: Problem, **kwargs):
        """
        Initialize the SearchAlgorithm instance.

        Args:
            problem: Problem to work with.
            kwargs: Any additional keyword argument, in the form of `k=f`,
                is accepted in this manner: for each pair of `k` and `f`,
                `k` is accepted as the status key (i.e. a status variable
                name), and `f` is accepted as a function (probably a method
                of the inheriting class) that will generate the value of that
                status variable.
        """
        super().__init__(**kwargs)
        self._problem = problem
        self._before_step_hook = Hook()
        self._after_step_hook = Hook()
        self._log_hook = Hook()
        self._end_of_run_hook = Hook()
        self._steps_count: int = 0
        self._first_step_datetime: Optional[datetime] = None

    @property
    def problem(self) -> Problem:
        """
        The problem object which is being worked on.
        """
        return self._problem

    @property
    def before_step_hook(self) -> Hook:
        """
        Use this Hook to add more behavior to the search algorithm
        to be performed just before executing a step.
        """
        return self._before_step_hook

    @property
    def after_step_hook(self) -> Hook:
        """
        Use this Hook to add more behavior to the search algorithm
        to be performed just after executing a step.

        The dictionaries returned by the functions registered into
        this Hook will be accumulated and added into the status
        dictionary of the search algorithm.
        """
        return self._after_step_hook

    @property
    def log_hook(self) -> Hook:
        """
        Use this Hook to add more behavior to the search algorithm
        at the moment of logging the constructed status dictionary.

        This Hook is executed after the execution of `after_step_hook`
        is complete.

        The functions in this Hook are assumed to expect a single
        argument, that is the status dictionary of the search algorithm.
        """
        return self._log_hook

    @property
    def end_of_run_hook(self) -> Hook:
        """
        Use this Hook to add more behavior to the search algorithm
        at the end of a run.

        This Hook is executed after all the generations of a run
        are done.

        The functions in this Hook are assumed to expect a single
        argument, that is the status dictionary of the search algorithm.
        """
        return self._end_of_run_hook

    @property
    def step_count(self) -> int:
        """
        Number of search steps performed.

        This is equivalent to the number of generations, or to the
        number of iterations.
        """
        return self._steps_count

    @property
    def steps_count(self) -> int:
        """
        Deprecated alias for the `step_count` property.
        It is recommended to use the `step_count` property instead.
        """
        return self._steps_count

    def step(self):
        """
        Perform a step of the search algorithm.
        """
        self._before_step_hook()
        self.clear_status()

        if self._first_step_datetime is None:
            self._first_step_datetime = datetime.now()

        self._step()
        self._steps_count += 1
        self.update_status({"iter": self._steps_count})
        self.update_status(self._problem.status)
        extra_status = self._after_step_hook.accumulate_dict()
        self.update_status(extra_status)
        if len(self._log_hook) >= 1:
            self._log_hook(dict(self.status))

    def _step(self):
        """
        Algorithm developers are expected to override this method
        in an inheriting subclass.

        The code which defines how a step of the evolutionary algorithm
        is performed goes here.
        """
        raise NotImplementedError

    def run(self, num_generations: int, *, reset_first_step_datetime: bool = True):
        """
        Run the algorithm for the given number of generations
        (i.e. iterations).

        Args:
            num_generations: Number of generations.
            reset_first_step_datetime: If this argument is given as True,
                then, the datetime of the first search step will be forgotten.
                Forgetting the first step's datetime means that the first step
                taken by this new run will be the new first step datetime.
        """
        if reset_first_step_datetime:
            self.reset_first_step_datetime()

        for _ in range(int(num_generations)):
            self.step()

        if len(self._end_of_run_hook) >= 1:
            self._end_of_run_hook(dict(self.status))

    @property
    def first_step_datetime(self) -> Optional[datetime]:
        """
        Get the datetime when the algorithm took the first search step.
        If a step is not taken at all, then the result will be None.
        """
        return self._first_step_datetime

    def reset_first_step_datetime(self):
        """
        Reset (or forget) the first step's datetime.
        """
        self._first_step_datetime = None

    @property
    def is_terminated(self) -> bool:
        """Whether the algorithm is in a terminal state"""
        return False


class SinglePopulationAlgorithmMixin:
    """
    A mixin class that can be inherited by a SearchAlgorithm subclass.

    This mixin class assumes that the inheriting class has the following
    members:

    - `problem`: The problem object that is associated with the search
      algorithm. This attribute is already provided by the SearchAlgorithm
      base class.
    - `population`: An attribute or a (possibly read-only) property which
      stores the population of the search algorithm as a `SolutionBatch`
      instance.

    This mixin class also assumes that the inheriting class _might_
    contain an attribute (or a property) named `obj_index`.
    If there is such an attribute and its value is not None, then this
    mixin class assumes that `obj_index` represents the index of the
    objective that is being focused on.

    Upon initialization, this mixin class first determines whether or not
    the algorithm is a single-objective one.
    In more details, if there is an attribute named `obj_index` (and its
    value is not None), or if the associated problem has only one objective,
    then this mixin class assumes that the inheriting SearchAlgorithm is a
    single objective algorithm.
    Otherwise, it is assumed that the underlying algorithm works (or might
    work) on multiple objectives.

    In the single-objective case, this mixin class brings the inheriting
    SearchAlgorithm the ability to report the following:
    `pop_best` (best solution of the population),
    `pop_best_eval` (evaluation result of the population's best solution),
    `mean_eval` (mean evaluation result of the population),
    `median_eval` (median evaluation result of the population).

    In the multi-objective case, for each objective `i`, this mixin class
    brings the inheriting SearchAlgorithm the ability to report the following:
    `obj<i>_pop_best` (best solution of the population according),
    `obj<i>_pop_best_eval` (evaluation result of the population's best
    solution),
    `obj<i>_mean_eval` (mean evaluation result of the population)
    `obj<iP_median_eval` (median evaluation result of the population).
    """

    class ObjectiveStatusReporter:
        REPORTABLES = {"pop_best", "pop_best_eval", "mean_eval", "median_eval"}

        def __init__(
            self,
            algorithm: SearchAlgorithm,
            *,
            obj_index: int,
            to_report: str,
        ):
            self.__algorithm = algorithm
            self.__obj_index = int(obj_index)
            if to_report not in self.REPORTABLES:
                raise ValueError(f"Unrecognized report request: {to_report}")
            self.__to_report = to_report

        @property
        def population(self) -> SolutionBatch:
            return self.__algorithm.population

        @property
        def obj_index(self) -> int:
            return self.__obj_index

        def get_status_value(self, status_key: str) -> Any:
            return self.__algorithm.get_status_value(status_key)

        def has_status_key(self, status_key: str) -> bool:
            return self.__algorithm.has_status_key(status_key)

        def _get_pop_best(self):
            i = self.population.argbest(self.obj_index)
            return clone(self.population[i])

        def _get_pop_best_eval(self):
            pop_best = None
            pop_best_keys = ("pop_best", f"obj{self.obj_index}_pop_best")

            for pop_best_key in pop_best_keys:
                if self.has_status_key(pop_best_key):
                    pop_best = self.get_status_value(pop_best_key)
                    break

            if (pop_best is not None) and pop_best.is_evaluated:
                return float(pop_best.evals[self.obj_index])
            else:
                return None

        @torch.no_grad()
        def _get_mean_eval(self):
            return float(torch.mean(self.population.access_evals(self.obj_index)))

        @torch.no_grad()
        def _get_median_eval(self):
            return float(torch.median(self.population.access_evals(self.obj_index)))

        def __call__(self):
            return getattr(self, "_get_" + self.__to_report)()

    def __init__(self, *, exclude: Optional[Iterable] = None, enable: bool = True):
        if not enable:
            return

        ObjectiveStatusReporter = self.ObjectiveStatusReporter
        reportables = ObjectiveStatusReporter.REPORTABLES
        single_obj: Optional[int] = None
        self.__exclude = set() if exclude is None else set(exclude)

        if hasattr(self, "obj_index") and (self.obj_index is not None):
            single_obj = self.obj_index
        elif len(self.problem.senses) == 1:
            single_obj = 0

        if single_obj is not None:
            for reportable in reportables:
                if reportable not in self.__exclude:
                    self.add_status_getters(
                        {reportable: ObjectiveStatusReporter(self, obj_index=single_obj, to_report=reportable)}
                    )
        else:
            for i_obj in range(len(self.problem.senses)):
                for reportable in reportables:
                    if reportable not in self.__exclude:
                        self.add_status_getters(
                            {
                                f"obj{i_obj}_{reportable}": ObjectiveStatusReporter(
                                    self, obj_index=i_obj, to_report=reportable
                                ),
                            }
                        )
