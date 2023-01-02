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

import torch

from evotorch import Problem
from evotorch.algorithms import GeneticAlgorithm
from evotorch.logging import StdOutLogger
from evotorch.operators import GaussianMutation, SimulatedBinaryCrossOver


# Kursawe function with two conflicting objectives
# Non-vectorized (evaluates a single solution vector) for illustrative purposes
def kursawe(x: torch.Tensor) -> torch.Tensor:
    f1 = torch.sum(-10 * torch.exp(-0.2 * torch.sqrt(x[0:2] ** 2.0 + x[1:3] ** 2.0)), dim=-1)
    f2 = torch.sum((torch.abs(x) ** 0.8) + (5 * torch.sin(x**3)), dim=-1)
    fitnesses = torch.stack([f1, f2], dim=-1)
    return fitnesses


problem = Problem(
    # Two objectives, both minimization
    ["min", "min"],
    kursawe,
    initial_bounds=(-5.0, 5.0),
    solution_length=3,
    vectorized=False,
    # Create as many parallel Ray actors as there are available CPUs
    num_actors="max",
)

# Run a GA similar to NSGA-II for 100 steps and log results to standard output every 10 steps
searcher = GeneticAlgorithm(
    problem,
    popsize=200,
    operators=[
        SimulatedBinaryCrossOver(problem, tournament_size=4, cross_over_rate=1.0, eta=8),
        GaussianMutation(problem, stdev=0.03),
    ],
)
logger = StdOutLogger(searcher, interval=10)

searcher.run(100)
print("Final status:\n", searcher.status)
