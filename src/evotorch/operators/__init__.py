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
This module provides various common operators to be used within evolutionary
algorithms.

Each operator is provided as a separate class, which is to be instantiated in
this form:

```python
op = OperatorName(
    problem,  # where problem is a Problem instance
    hyperparameter1=...,
    hyperparameter2=...,
    # ...
)
```

Each operator has its `__call__(...)` method overriden so that it can be used
like a function. For example, if the operator `op` instantiated above were a
mutation operator, it would be used like this:

```python
# Apply mutation on a SolutionBatch
mutated_solution = op(my_solution_batch)
```

Please see the documentations of the provided operator classes for details
about how to instantiate them, and how to call them.

A common usage for the operators provided here is to use them with
[GeneticAlgorithm][evotorch.algorithms.ga.GeneticAlgorithm], as shown below:

```python
from evotorch.algorithms import GeneticAlgorithm
from evotorch.operators import SimulatedBinaryCrossOver, GaussianMutation

problem = ...  # initialize the Problem

ga = GeneticAlgorithm(
    problem,
    operators=[
        SimulatedBinaryCrossOver(
            problem,
            tournament_size=...,
            cross_over_rate=...,
            eta=...,
        ),
        GaussianMutation(
            problem,
            stdev=...,
        ),
    ],
    popsize=...,
)
```
"""

__all__ = (
    "CopyingOperator",
    "CosynePermutation",
    "CrossOver",
    "CutAndSplice",
    "GaussianMutation",
    "MultiPointCrossOver",
    "OnePointCrossOver",
    "Operator",
    "PolynomialMutation",
    "SimulatedBinaryCrossOver",
    "SingleObjOperator",
    "TwoPointCrossOver",
)


from . import base, real, sequence
from .base import CopyingOperator, CrossOver, Operator, SingleObjOperator
from .real import (
    CosynePermutation,
    GaussianMutation,
    MultiPointCrossOver,
    OnePointCrossOver,
    PolynomialMutation,
    SimulatedBinaryCrossOver,
    TwoPointCrossOver,
)
from .sequence import CutAndSplice
