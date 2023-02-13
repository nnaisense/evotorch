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
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from evotorch import Problem
from evotorch.algorithms import CEM, CMAES, PGPE, SNES, XNES, Cosyne
from evotorch.logging import PandasLogger, StdOutLogger
from evotorch.neuroevolution import GymNE, NEProblem, SupervisedNE, VecGymNE
from evotorch.tools import device_of, dtype_of


@pytest.mark.parametrize(
    "algorithm, kwargs",
    [
        (SNES, {"stdev_init": 5}),
        (XNES, {"stdev_init": 5}),
        (PGPE, {"popsize": 10, "center_learning_rate": 0.01, "stdev_learning_rate": 0.1, "radius_init": 0.27}),
        (CEM, {"popsize": 10, "parenthood_ratio": 0.1, "radius_init": 0.27}),
        (SNES, {"stdev_init": 5, "distributed": True}),
        (
            PGPE,
            {
                "popsize": 10,
                "center_learning_rate": 0.01,
                "stdev_learning_rate": 0.1,
                "radius_init": 0.27,
                "distributed": True,
            },
        ),
        (CEM, {"popsize": 10, "parenthood_ratio": 0.1, "radius_init": 0.27, "distributed": True}),
        (CMAES, {"popsize": 10, "stdev_init": 5}),
        (Cosyne, {"popsize": 10, "tournament_size": 3, "mutation_stdev": 0.1}),
        (Cosyne, {"popsize": 10, "tournament_size": 3, "mutation_stdev": None}),
    ],
)
def test_quickstart_example(algorithm: Any, kwargs: Dict[str, Any]) -> None:
    # Define a function to minimize
    def sphere(x: torch.Tensor) -> torch.Tensor:
        return torch.sum(x.pow(2.0))

    distributed = kwargs.get("distributed", False)
    # Define a Problem instance wrapping the function
    # Solutions have length 10
    problem = Problem(
        "min",
        sphere,
        solution_length=10,
        initial_bounds=(-1, 1),
        num_actors=2 if distributed else None,
    )

    # Instantiate a searcher
    searcher = algorithm(problem, **kwargs)

    # Evolve!
    searcher.run(2)

    if distributed:
        assert "center" in searcher.status
    else:
        assert "best" in searcher.status
    assert searcher.step_count == 2


def test_neuroevolution_example():
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

    problem = NEProblem(
        # The objective sense -- we wish to maximize the sign_prediction_score
        objective_sense="max",
        # The network is a Linear layer mapping 3 inputs to 1 output
        network=torch.nn.Linear(3, 1),
        # Networks will be evaluated according to sign_prediction_score
        network_eval_func=sign_prediction_score,
    )

    searcher = PGPE(
        problem,
        popsize=10,
        radius_init=2.25,
        center_learning_rate=0.2,
        stdev_learning_rate=0.1,
    )
    searcher.run(2)

    assert "best" in searcher.status
    assert searcher.step_count == 2


@pytest.mark.parametrize(
    "network",
    [
        "Linear(obs_length, act_length)",
        torch.nn.Linear(4, 2),
        lambda: torch.nn.Linear(4, 2),
    ],
)
def test_neuroevolution_for_gymne(network):
    problem = GymNE(
        # Name of the environment
        env="CartPole-v1",
        # Linear policy mapping observations to actions
        network=network,
        # Use 4 available CPUs. Note that you can modify this value,
        # or use 'max' to exploit all available GPUs
        num_actors=4,
    )

    searcher = PGPE(
        problem,
        popsize=10,
        radius_init=2.25,
        center_learning_rate=0.2,
        stdev_learning_rate=0.1,
    )
    searcher.run(2)

    assert "best" in searcher.status
    assert searcher.step_count == 2


@pytest.mark.parametrize(
    "network",
    [
        "Linear(obs_length, act_length)",
        torch.nn.Linear(4, 2),
        lambda: torch.nn.Linear(4, 2),
    ],
)
def test_neuroevolution_for_vecgymne(network):
    problem = VecGymNE(
        # Name of the environment
        env="CartPole-v1",
        # Linear policy mapping observations to actions
        network=network,
        # Use 4 available CPUs. Note that you can modify this value,
        # or use 'max' to exploit all available GPUs
        num_actors=4,
        observation_normalization=True,
    )

    searcher = PGPE(
        problem,
        popsize=10,
        radius_init=2.25,
        center_learning_rate=0.2,
        stdev_learning_rate=0.1,
    )
    searcher.run(2)

    assert "best" in searcher.status
    assert searcher.step_count == 2


@pytest.mark.parametrize(
    "network",
    [
        nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 1)),
        lambda: nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 1)),
    ],
)
def test_supervised_neuroevolution(network):
    N = 100
    X = torch.randn((N, 2))
    Y = X.sum(dim=-1, keepdim=True)

    train_dataset = TensorDataset(X, Y)

    sum_of_problem = SupervisedNE(
        dataset=train_dataset,
        network=network,
        minibatch_size=32,
        loss_func=nn.MSELoss(),
    )

    searcher = SNES(sum_of_problem, popsize=10, radius_init=2.25)
    searcher.run(2)

    assert "best" in searcher.status
    assert searcher.step_count == 2
