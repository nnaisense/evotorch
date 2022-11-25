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

from copy import deepcopy
from typing import Type

from .restart import Restart
from evotorch import Problem
from evotorch.algorithms import SearchAlgorithm


class ModifyingRestart(Restart):
    def _modify_algorithm_args(self) -> None:
        """Modify the algorithm arguments on restart"""
        pass

    def _restart(self) -> None:
        """Restart the search algorithm"""
        self._modify_algorithm_args()
        return super()._restart()


class IPOP(ModifyingRestart):
    def __init__(
        self,
        problem: Problem,
        algorithm_class: Type[SearchAlgorithm],
        algorithm_args: dict = {},
        min_fitness_stdev: float = 1e-9,
        popsize_multiplier: float = 2,
    ):
        """IPOP restart, terminates algorithm when minimum standard deviation in fitness values is hit, multiplies the population size in that case
        References:
            Glasmachers, Tobias, and Oswin Krause.
            "The hessian estimation evolution strategy."
            PPSN 2020
        Args:
            problem (Problem): A Problem to solve
            algorithm_class (Type[SearchAlgorithm]): The class of the search algorithm to restart
            algorithm_args (dict): Arguments to pass to the search algorithm on restart
            min_fitness_stdev (float): The minimum standard deviation in fitnesses; going below this will trigger a restart
            popsize_multiplier (float): A multiplier on the population size within algorithm_args
        """
        super().__init__(problem, algorithm_class, algorithm_args)

        self.min_fitness_stdev = min_fitness_stdev
        self.popsize_multiplier = popsize_multiplier

    def _search_algorithm_terminated(self) -> bool:
        # Additional check on standard deviation of fitnesses of population
        if self.search_algorithm.population.evals.std() < self.min_fitness_stdev:
            return True
        return super()._search_algorithm_terminated()

    def _modify_algorithm_args(self) -> None:
        # Only modify arguments if this isn't the first restart
        if self.num_restarts >= 1:
            new_args = deepcopy(self._algorithm_args)
            # Multiply population size
            new_args["popsize"] = int(self.popsize_multiplier * len(self.search_algorithm.population))
            self._algorithm_args = new_args
