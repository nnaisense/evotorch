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
This module provides various common operators
to be used within evolutionary algorithms.

Each operator is provided as a separate class,
which is to be instantiated in this form:

    op = OperatorName(
        problem,  # where problem is a Problem instance
        hyperparameter1=...,
        hyperparameter2=...,
        ...
    )

Each operator has its `__call__(...)` method overriden
so that it can be used like a function.
For example, if the operator `op` instantiated above
were a mutation operator, it would be used like this:

    # Apply mutation on a SolutionBatch
    mutated_solution = op(my_solution_batch)

Please see the documentations of the provided operator
classes for details about how to instantiate them,
and how to call them.

A common usage for the operators provided here is to
use them with a genetic algorithm.
More specifically, the SteadyStateGA algorithm provided
within the namespace `evotorch.algorithms` needs
to be configured so that it knows which cross-over operator
and which mutation operator it should apply on the
solutions. The way this is done is as follows:

    import evotorch.algorithms as dra
    import evotorch.operators as dro

    problem = ...   # initialize the Problem

    ga = dra.SteadyStateGA(problem, popsize=...)

    # Configure the genetic algorithm to use
    # simulated binary cross-over
    ga.use(
        dro.SimulatedBinaryCrossOver(
            problem,
            tournament_size=...,
            cross_over_rate=...,
            eta=...
        )
    )

    # Configure the genetic algorithm to use
    # Gaussian mutation
    ga.use(
        dro.GaussianMutation(
            problem,
            stdev=...
        )
    )
"""

__all__ = (
    "CopyingOperator",
    "CrossOver",
    "Operator",
    "SingleObjOperator",
    "CosynePermutation",
    "GaussianMutation",
    "OnePointCrossOver",
    "SimulatedBinaryCrossOver",
    "CutAndSplice",
)


from . import base, real, sequence
from .base import CopyingOperator, CrossOver, Operator, SingleObjOperator
from .real import CosynePermutation, GaussianMutation, OnePointCrossOver, SimulatedBinaryCrossOver
from .sequence import CutAndSplice
