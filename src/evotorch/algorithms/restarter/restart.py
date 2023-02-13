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

from typing import Any, Type

from evotorch import Problem
from evotorch.algorithms.searchalgorithm import SearchAlgorithm


class Restart(SearchAlgorithm):
    def __init__(
        self,
        problem: Problem,
        algorithm_class: Type[SearchAlgorithm],
        algorithm_args: dict = {},
        **kwargs: Any,
    ):
        """Base class for independent restarts methods
        Args:
            problem (Problem): A Problem to solve
            algorithm_class (Type[SearchAlgorithm]): The class of the search algorithm to restart
            algorithm_args (dict): Arguments to pass to the search algorithm on restart
        """

        SearchAlgorithm.__init__(
            self,
            problem,
            search_algorithm=self._get_sa_status,
            num_restarts=self._get_num_restarts,
            algorithm_terminated=self._search_algorithm_terminated,
            **kwargs,
        )

        self._algorithm_class = algorithm_class
        self._algorithm_args = algorithm_args

        self.num_restarts = 0
        self._restart()

    def _get_sa_status(self) -> dict:
        """Status dictionary of search algorithm"""
        return self.search_algorithm.status

    def _get_num_restarts(self) -> int:
        """Number of restarts (including the first start) so far"""
        return self.num_restarts

    def _restart(self) -> None:
        """Restart the search algorithm"""
        self.search_algorithm = self._algorithm_class(self._problem, **self._algorithm_args)
        self.num_restarts += 1

    def _search_algorithm_terminated(self) -> bool:
        """Boolean flag for search algorithm terminated"""
        return self.search_algorithm.is_terminated

    def _step(self):
        # Step the search algorithm
        self.search_algorithm.step()

        # If stepping the search algorithm has reached a terminal state, restart the search algorithm
        if self._search_algorithm_terminated():
            self._restart()
