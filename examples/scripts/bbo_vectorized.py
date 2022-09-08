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
from evotorch.algorithms import SNES
from evotorch.logging import StdOutLogger


# Define the cost function that receives a batch of solution vectors (2D Tensor) and returns the costs (1D Tensor)
def cost(x: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(x, dim=-1)


# The problem is to minimize the cost, for solution vectors of default dtype (float32) and length 10, on the first GPU
problem = Problem(
    "min",
    cost,
    solution_length=10,
    initial_bounds=(-10, 10),
    vectorized=True,
    device="cuda:0",  # Change to "cpu" if a GPU is not available
)

# Run the distribution-based search algorithm SNES for 500 steps and log results to standard output every 100 steps
searcher = SNES(problem, popsize=100, stdev_init=0.01)
logger = StdOutLogger(searcher, interval=100)
searcher.run(500)
print("Final status:\n", searcher.status)
