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
from evotorch.algorithms import PGPE
from evotorch.neuroevolution import VecGymNE
from evotorch.tools import device_of, dtype_of


@pytest.mark.parametrize(
    "network",
    [
        "Linear(obs_length, act_length)",
        torch.nn.Linear(4, 2),
        lambda: torch.nn.Linear(4, 2),
    ],
)
def test_to_policy(network):
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

    # Test Tensor object
    solution = problem.generate_values(1)[0]

    policy = problem.to_policy(solution)

    assert policy is not None
    assert isinstance(policy, torch.nn.Linear)
    assert policy.in_features == 4
    assert policy.out_features == 2

    # Test Solution object
    solution = problem.generate_batch(1)[0]

    policy = problem.to_policy(solution)

    assert policy is not None
    assert isinstance(policy, torch.nn.Linear)
    assert policy.in_features == 4
    assert policy.out_features == 2

    # Test ReadOnlyTensor object
    solution = problem.generate_batch(1)[0].values

    policy = problem.to_policy(solution)

    assert policy is not None
    assert isinstance(policy, torch.nn.Linear)
    assert policy.in_features == 4
    assert policy.out_features == 2


@pytest.mark.parametrize(
    "network",
    [
        "Linear(obs_length, act_length)",
        torch.nn.Linear(4, 2),
    ],
)
def test_save_solution(network, tmp_path):
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

    # Test Tensor object
    solution = problem.generate_values(1)[0]

    fname = tmp_path / "solution-tensor.pt"
    problem.save_solution(solution, fname)

    assert fname.exists()
    assert fname.stat().st_size > 0

    # Test Solution object
    solution = problem.generate_batch(1)[0]

    fname = tmp_path / "solution-solution.pt"
    problem.save_solution(solution, fname)

    assert fname.exists()
    assert fname.stat().st_size > 0

    # Test ReadOnlyTensor object
    solution = problem.generate_batch(1)[0].values

    fname = tmp_path / "solution-readonly.pt"
    problem.save_solution(solution, fname)

    assert fname.exists()
    assert fname.stat().st_size > 0
