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

from typing import Any, Dict

import pytest
import torch

from evotorch import Problem
from evotorch.algorithms import CEM, CMAES, PGPE, SNES, XNES
from evotorch.logging import PandasLogger, StdOutLogger
from evotorch.neuroevolution import GymNE, NEProblem
from evotorch.tools import device_of, dtype_of


def sign_prediction_score(network: torch.nn.Module):
    # Generate 32 random gaussian vectors
    samples = torch.randn((32, 3), dtype=dtype_of(network), device=device_of(network))
    # Apply the network to the gaussian vectors
    network_out = network(samples)
    # Get the sign of the single output
    sign_out = torch.sign(network_out[:, 0])
    # Get the sign of the sum of the inputs
    sign_sum = torch.sign(samples.sum(dim=-1))
    # Number of times the network was correct
    reward_gained = (sign_sum == sign_out).to(torch.float).sum()
    # Number of times the network was incorrect
    reward_lost = (sign_sum != sign_out).to(torch.float).sum()
    return (reward_gained - reward_lost) / 32


@pytest.mark.parametrize(
    "algorithm, kwargs",
    [
        (SNES, {"stdev_init": 5}),
        (XNES, {"stdev_init": 5}),
        (PGPE, {"popsize": 10, "center_learning_rate": 0.01, "stdev_learning_rate": 0.1, "radius_init": 0.27}),
        (CEM, {"popsize": 10, "parenthood_ratio": 0.1, "radius_init": 0.27}),
        (CMAES, {"popsize": 10, "stdev_init": 5}),
    ],
)
def test_quickstart_example(algorithm: Any, kwargs: Dict[str, Any]) -> None:
    # Define a function to minimize
    def sphere(x: torch.Tensor) -> torch.Tensor:
        return torch.sum(x.pow(2.0))

    # Define a Problem instance wrapping the function
    # Solutions have length 10
    problem = Problem("min", sphere, solution_length=10, initial_bounds=(-1, 1))

    # Instantiate a searcher
    searcher = algorithm(problem, **kwargs)

    # Create a logger
    _ = StdOutLogger(searcher)

    # Evolve!
    searcher.run(2)


def test_neuroevolution_example():
    sign_prediction_problem = NEProblem(
        # The objective sense -- we wish to maximize the sign_prediction_score
        objective_sense="max",
        # The network is a Linear layer mapping 3 inputs to 1 output
        network=torch.nn.Linear(3, 1),
        # Networks will be evaluated according to sign_prediction_score
        network_eval_func=sign_prediction_score,
    )

    searcher = PGPE(
        sign_prediction_problem,
        popsize=10,
        radius_init=2.25,
        center_learning_rate=0.2,
        stdev_learning_rate=0.1,
    )
    _ = PandasLogger(searcher)
    searcher.run(2)


def test_neuroevolution_for_gym():
    problem = GymNE(
        # Name of the environment
        env="CartPole-v1",
        # Linear policy mapping observations to actions
        network="Linear(obs_length, act_length)",
        # Use 4 available CPUs. Note that you can modify this value,
        # or use 'max' to exploit all available GPUs
        num_actors=4,
    )

    assert problem is not None
