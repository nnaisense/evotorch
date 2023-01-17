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
This namespace contains the PyCMAES class, which is a wrapper
for the CMA-ES implementation of the `cma` package.
"""

import math
from copy import copy, deepcopy
from typing import Any, Callable, Iterable, List, Optional, Union

import numpy as np
import torch

from ..core import Problem, SolutionBatch
from ..tools.misc import Device, DType, RealOrVector, Vector, is_sequence, numpy_copy
from .searchalgorithm import SearchAlgorithm, SinglePopulationAlgorithmMixin

try:
    import cma
except ImportError:
    cma = None


class PyCMAES(SearchAlgorithm, SinglePopulationAlgorithmMixin):
    """
    PyCMAES: Covariance Matrix Adaptation Evolution Strategy.

    This is an interface class between the CMAES implementation
    within the `cma` package developed within the GitHub repository
    CMA-ES/pycma.

    References:

        Nikolaus Hansen, Youhei Akimoto, and Petr Baudis.
        CMA-ES/pycma on Github. Zenodo, DOI:10.5281/zenodo.2559634,
        February 2019.
        <https://github.com/CMA-ES/pycma>

        Nikolaus Hansen, Andreas Ostermeier (2001).
        Completely Derandomized Self-Adaptation in Evolution Strategies.

    """

    def __init__(
        self,
        problem: Problem,
        *,
        stdev_init: RealOrVector,  # sigma0
        popsize: Optional[int] = None,  # popsize
        center_init: Optional[Vector] = None,  # x0
        center_learning_rate: Optional[float] = None,  # CMA_cmean
        cov_learning_rate: Optional[float] = None,  # CMA_on
        rankmu_learning_rate: Optional[float] = None,  # CMA_rankmu
        rankone_learning_rate: Optional[float] = None,  # CMA_rankone
        stdev_min: Optional[Union[float, np.ndarray]] = None,  # minstd
        stdev_max: Optional[Union[float, np.ndarray]] = None,  # maxstd
        separable: bool = False,  # CMA_diagonal
        obj_index: Optional[int] = None,
        cma_options: dict = {},
    ):
        """
        `__init__(...)`: Initialize the PyCMAES solver.

        Args:
            problem: The problem object which is being worked on.
            stdev_init: Initial standard deviation as a scalar or
                as a 1-dimensional array.
            popsize: Population size. Can be specified as an int,
                or can be left as None to let the CMAES solver
                decide the population size according to the length
                of a solution.
            center_init: Initial center point of the search distribution.
                Can be given as a Solution or as a 1-D array.
                If left as None, an initial center point is generated
                with the help of the problem object's `generate_values(...)`
                method.
            center_learning_rate: Learning rate for updating the mean
                of the search distribution. Leaving this as None
                means that the CMAES solver is to use its own default,
                which is documented as 1.0.
            cov_learning_rate: Learning rate for updating the covariance
                matrix of the search distribution. This hyperparameter
                acts as a common multiplier for rank_one update and rank_mu
                update of the covariance matrix. Leaving this as None
                means that the CMAES solver is to use its own default,
                which is documented as 1.0.
            rankmu_learning_rate: Learning rate for the rank_mu update
                of the covariance matrix of the search distribution.
                Leaving this as None means that the CMAES solver is to use
                its own default, which is documented as 1.0.
            rankone_learning_rate: Learning rate for the rank_one update
                of the covariance matrix of the search distribution.
                Leaving this as None means that the CMAES solver is to use
                its own default, which is documented as 1.0.
            stdev_min: Minimum allowed standard deviation of the search
                distribution. Leaving this as None means that no such
                boundary is to be used.
                Can be given as None, as a scalar, or as a 1-dimensional
                array.
            stdev_max: Maximum allowed standard deviation of the search
                distribution. Leaving this as None means that no such
                boundary is to be used.
                Can be given as None, as a scalar, or as a 1-dimensional
                array.
            separable: Provide this as True if you would like the problem
                to be treated as a separable one. Treating a problem
                as separable means to adapt only the diagonal parts
                of the covariance matrix and to keep the non-diagonal
                parts 0. High dimensional problems result in large
                covariance matrices on which operating is computationally
                expensive. Therefore, for such high dimensional problems,
                setting `separable` as True might be useful.
                If, instead, you would like to configure on which
                iterations the diagonal parts of the covariance matrix
                are to be adapted, then it is recommended to leave
                `separable` as False and set a new value for the key
                "CMA_diagonal" via `cma_options` (see the official
                documentation of pycma for details regarding the
                "CMA_diagonal" setting).
            obj_index: Objective index according to which evaluation
                of the solution will be done.
            cma_options: Any other configuration for the CMAES solver
                can be passed via the cma_options dictionary.
        """

        # Make sure that the cma module is installed
        if cma is None:
            raise ImportError(f"The class {type(self).__name__} is only available if the package `cma` is installed.")

        # Initialize the base class
        SearchAlgorithm.__init__(self, problem, center=self._get_center)

        # Ensure that the problem is numeric
        problem.ensure_numeric()

        # Store the objective index
        self._obj_index = problem.normalize_obj_index(obj_index)

        # If `center_init` is not given, generate an initial solution
        # with the help of the problem object.
        # Otherwise, use the given initial solution as the starting
        # point in the search space.
        if center_init is None:
            x0 = self._problem.generate_values(1).to("cpu").view(-1).numpy().astype(dtype=float)
        else:
            x0 = numpy_copy(center_init, dtype=float)

        # Store the initial standard deviations
        sigma0 = numpy_copy(stdev_init, dtype=float)

        # Generate an options dictionary to pass to the cma solver.
        inopts = {}
        for k, v in cma_options.items():
            if isinstance(v, torch.Tensor):
                v = numpy_copy(v, dtype=float)
            inopts[k] = v

        # Remove the number of iterations boundary
        if "maxiter" not in inopts:
            inopts["maxiter"] = np.inf

        # Below is a temporary helper function for safely storing the configuration items.
        # This inner function updates the `inopts` variable.
        def store_opt(key: str, long_name: str, value: Any, converter: Callable):
            # Here, `key` represents the configuration key used by pycma
            # `long_name` represents the configuration's long name used by this class
            # `value` is the configuration value associated with `key`.

            # Declare that this inner function accesses the `inopts` variable.
            nonlocal inopts

            if value is None:
                # If the provided `value` is None, then there is no configuration to store.
                # So, we just leave this inner function.
                return

            if key in inopts:
                # If the given `key` already exists within `inopts`, this means that the configuration was specified
                # twice: via the keyword argument `cma_options` AND via a keyword argument.
                # We raise an error and inform the user about this redundancy.
                raise ValueError(
                    f"The configuration {repr(key)} was redundantly provided"
                    f" both via the initialization argument {long_name}"
                    f" and via the cma_options dictionary."
                    f" {long_name}={repr(value)};"
                    f" cma_options[{repr(key)}]={repr(inopts[key])}."
                )

            inopts[key] = converter(value)

        # Temporary helper function which makes sure that `x` is a numpy array or a float.
        def array_or_float(x):
            if is_sequence(x):
                return numpy_copy(x)
            else:
                return float(x)

        # Store the cma configuration received through the initialization arguments (and raise error if there is
        # redundancy with the cma_options dictionary).
        store_opt("popsize", "popsize", popsize, int)
        store_opt("CMA_cmean", "center_learning_rate", center_learning_rate, float)
        store_opt("CMA_on", "cov_learning_rate", cov_learning_rate, float)
        store_opt("CMA_rankmu", "rankmu_learning_rate", rankmu_learning_rate, float)
        store_opt("CMA_rankone", "rankone_learning_rate", rankone_learning_rate, float)
        store_opt("minstd", "stdev_min", stdev_min, array_or_float)
        store_opt("maxstd", "stdev_max", stdev_max, array_or_float)
        if separable:
            store_opt("CMA_diagonal", "separable", separable, bool)

        # If the problem defines lower and upper bounds, pass these into the options dict.
        def process_bounds(bounds: RealOrVector) -> np.ndarray:
            if bounds is None:
                return None
            else:
                if is_sequence(bounds):
                    bounds = numpy_copy(bounds)
                else:
                    bounds = np.array(float(bounds)).repeat(self._problem.solution_length)
                return bounds

        lb = process_bounds(self._problem.lower_bounds)
        ub = process_bounds(self._problem.upper_bounds)

        register_bounds = False
        if lb is not None and ub is None:
            ub = np.array(np.inf).repeat(self._problem.solution_length)
            register_bounds = True
        elif lb is None and ub is not None:
            lb = np.array(-(np.inf)).repeat(self._problem.solution_length)
            register_bounds = True
        elif lb is not None and ub is not None:
            register_bounds = True

        if register_bounds:
            inopts["bounds"] = [lb, ub]

        # Generate a random seed using the problem object for the sake of reproducibility.
        if "seed" not in inopts:
            inopts["seed"] = int(self._problem.make_randint(tuple(), n=(2**32) - 100) + 100)

        # Instantiate the CMAEvolutionStrategy with the prepared configuration items.
        self._es = cma.CMAEvolutionStrategy(x0, sigma0, inopts)

        # Initialize the population.
        self._population: SolutionBatch = self._problem.generate_batch(self._es.popsize, empty=True)

        # Use the SinglePopulationAlgorithmMixin to enable additional status reports regarding the population.
        SinglePopulationAlgorithmMixin.__init__(self)

    @property
    def population(self) -> SolutionBatch:
        """Population generated by the CMA-ES algorithm"""
        return self._population

    def _step(self):
        """Perform a step of the CMA-ES solver"""
        asked = self._es.ask()
        self._population.access_values()[:] = torch.as_tensor(
            np.asarray(asked), dtype=self._problem.dtype, device=self._population.device
        )
        self._problem.evaluate(self._population)
        scores = numpy_copy(self._population.utility(self._obj_index), dtype=float)
        self._es.tell(asked, -1.0 * scores)

    def _get_center(self) -> torch.Tensor:
        return torch.as_tensor(self._es.result[5], dtype=self._population.dtype, device=self._population.device)

    @property
    def obj_index(self) -> int:
        """Index of the objective being focused on"""
        return self._obj_index
