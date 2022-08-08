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
This module contains operators for problems whose solutions contain
variable-length sequences (list-like objects).
"""

from ..core import SolutionBatch
from ..tools.objectarray import ObjectArray
from .base import CrossOver


class CutAndSplice(CrossOver):
    """Cut & Splice operator for variable-length solutions.

    This class serves as a cross-over operator to be used on problems
    with their `dtype`s set as `object`, and with their solutions
    initialized to contain variable-length sequences (list-like objects).

    Reference:

        David E. Goldberg, Bradley Korb, Kalyanmoy Deb (1989).
        Messy Genetic Algorithms: Motivation, Analysis, and First Results.
        Complex Systems 3, 493-530.
    """

    def _cut_and_splice(
        self,
        parents1: ObjectArray,
        parents2: ObjectArray,
        children1: SolutionBatch,
        children2: SolutionBatch,
        row_index: int,
    ):
        parvals1 = parents1[row_index]
        parvals2 = parents2[row_index]

        length1 = len(parvals1)
        length2 = len(parvals2)

        cutpoint1 = int(self.problem.make_randint(tuple(), n=length1))
        cutpoint2 = int(self.problem.make_randint(tuple(), n=length2))

        childvals1 = parvals1[:cutpoint1]
        childvals1.extend(parvals2[cutpoint2:])

        childvals2 = parvals2[:cutpoint2]
        childvals2.extend(parvals1[cutpoint1:])

        children1.access_values(keep_evals=True)[row_index] = childvals1
        children2.access_values(keep_evals=True)[row_index] = childvals2

    def _do_cross_over(self, parents1: ObjectArray, parents2: ObjectArray) -> SolutionBatch:
        n = len(parents1)

        children1 = SolutionBatch(self.problem, popsize=n, empty=True)
        children2 = SolutionBatch(self.problem, popsize=n, empty=True)

        for i in range(n):
            self._cut_and_splice(parents1, parents2, children1, children2, i)

        return children1.concat(children2)
