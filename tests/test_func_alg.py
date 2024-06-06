# Copyright 2024 NNAISENSE SA
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


import pytest
import torch
from torch.func import grad

from evotorch.algorithms import functional as funcalg
from evotorch.decorators import expects_ndim, rowwise


@rowwise
def f(x: torch.Tensor) -> torch.Tensor:
    return torch.sum(x**2)


def test_cem():
    batch_size = 3
    solution_length = 5
    popsize = 20

    x = torch.randn(batch_size, solution_length)

    state = funcalg.cem(
        center_init=x, stdev_init=1.0, parenthood_ratio=0.5, objective_sense="min", stdev_max_change=0.2
    )
    assert state.center.shape == (batch_size, solution_length)

    population = funcalg.cem_ask(state, popsize=popsize)
    assert population.shape == (batch_size, popsize, solution_length)

    fitnesses = f(population)
    assert fitnesses.shape == (batch_size, popsize)

    state = funcalg.cem_tell(state, population, fitnesses)
    assert state.center.shape == (batch_size, solution_length)


def test_pgpe():
    batch_size = 3
    solution_length = 5
    popsize = 20

    x = torch.randn(batch_size, solution_length)

    state = funcalg.pgpe(
        center_init=x,
        stdev_init=1.0,
        center_learning_rate=0.1,
        stdev_learning_rate=0.1,
        optimizer="clipup",
        optimizer_config={"max_speed": 0.2},
        ranking_method="centered",
        objective_sense="min",
        stdev_max_change=0.2,
    )
    assert state.optimizer_state.center.shape == (batch_size, solution_length)

    population = funcalg.pgpe_ask(state, popsize=popsize)
    assert population.shape == (batch_size, popsize, solution_length)

    fitnesses = f(population)
    assert fitnesses.shape == (batch_size, popsize)

    state = funcalg.pgpe_tell(state, population, fitnesses)
    assert state.optimizer_state.center.shape == (batch_size, solution_length)


@pytest.mark.parametrize("optimizer_name", ["adam", "clipup", "sgd"])
def test_optimizer(optimizer_name: str):
    batch_size = 3
    solution_length = 5
    x = torch.randn(batch_size, solution_length)

    optimizer_start, optimizer_ask, optimizer_tell = funcalg.misc.get_functional_optimizer(optimizer_name)
    state = optimizer_start(center_init=x, center_learning_rate=0.1)
    assert state.center.shape == (batch_size, solution_length)

    center = optimizer_ask(state)
    assert center.shape == (batch_size, solution_length)

    g = expects_ndim(grad(f), (1,))(center)
    assert g.shape == (batch_size, solution_length)

    state = optimizer_tell(state, follow_grad=-g)
    assert state.center.shape == (batch_size, solution_length)
