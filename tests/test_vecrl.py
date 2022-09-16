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
from torch import nn

from evotorch.neuroevolution.net import fill_parameters, parameter_vector
from evotorch.neuroevolution.net.layers import LSTM, RNN
from evotorch.neuroevolution.net.vecrl import Policy, reset_tensors
from evotorch.testing import assert_allclose
from evotorch.tools import clone


def test_reset_tensors_simple():
    tolerance = 1e-4

    # Make a two-dimensional tensor
    x = torch.tensor(
        [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9, 10, 11],
            [12, 13, 14, 15],
        ],
        dtype=torch.float32,
    )

    # Use `reset_tensors(...)` to reset the rows with indices 0 and 2
    reset_tensors(x, [0, 2])

    # Assert that the rows with indices 0 and 2 are indeed zeroed out
    assert_allclose(
        x,
        torch.tensor(
            [
                [0, 0, 0, 0],
                [4, 5, 6, 7],
                [0, 0, 0, 0],
                [12, 13, 14, 15],
            ],
            dtype=torch.float32,
        ),
        atol=tolerance,
    )


def test_reset_tensors_with_containers():
    tolerance = 1e-4

    # Generate tensors a, b, c, and d
    a = torch.tensor(
        [
            [0, 1],
            [2, 3],
            [4, 5],
        ],
        dtype=torch.float32,
    )

    b = torch.tensor(
        [
            [0, 10, 20],
            [30, 40, 50],
            [60, 70, 80],
        ],
        dtype=torch.float32,
    )

    c = torch.tensor(
        [
            [100],
            [200],
            [300],
        ],
        dtype=torch.float32,
    )

    d = torch.tensor([-1, -2, -3], dtype=torch.float32)

    # Put these tensors into a container
    my_tensors = [a, {"1": b, "2": (c, d)}]

    # Use `reset_tensors(...)` to clear the regions with indices 1 and 2 of each tensor in the container
    reset_tensors(my_tensors, [1, 2])

    # Assert that the reset operation was successful for each tensor in the container
    assert_allclose(
        a,
        torch.tensor(
            [
                [0, 1],
                [0, 0],
                [0, 0],
            ],
            dtype=torch.float32,
        ),
        atol=tolerance,
    )

    assert_allclose(
        b,
        torch.tensor(
            [
                [0, 10, 20],
                [0, 0, 0],
                [0, 0, 0],
            ],
            dtype=torch.float32,
        ),
        atol=tolerance,
    )

    assert_allclose(
        c,
        torch.tensor(
            [
                [100],
                [0],
                [0],
            ],
            dtype=torch.float32,
        ),
        atol=tolerance,
    )

    assert_allclose(d, torch.tensor([-1, 0, 0], dtype=torch.float32), atol=tolerance)


@torch.no_grad()
def test_policy_simple():
    tolerance = 1e-4

    # Make a torch module
    net = nn.Linear(5, 8)

    # Take its parameters
    parameters = parameter_vector(net)
    parameter_length = len(parameters)

    # Instantiate a policy from a clone of this torch module, and fill it with the same parameters
    policy = Policy(clone(net))
    policy.set_parameters(parameters)
    assert policy.parameter_length == parameter_length

    # Generate a random input
    x = torch.randn(5)

    # Pass the input through the original torch module and through the policy
    y1 = net(x)
    y2 = policy(x)

    # Assert that the output is the same
    assert_allclose(y2, y1, atol=tolerance)


@torch.no_grad()
def test_policy_batched():
    tolerance = 1e-4

    # Make a new torch module
    net = nn.Linear(5, 8)

    # Take its parameters
    parameters = parameter_vector(net)
    parameter_length = len(parameters)

    # Make a new Policy from a clone of the torch module, and fill it with the same parameters
    policy = Policy(clone(net))
    policy.set_parameters(parameters)
    assert policy.parameter_length == parameter_length

    # Declare a batch size, generate a batch of network parameters, and a batch of inputs
    batch_size = 10
    batch_of_parameters = torch.randn(batch_size, parameter_length)
    batch_of_x = torch.randn(batch_size, 5)

    # For each row in the parameter batch, fill the torch module's parameters with that row, pass the corresponding
    # input through the network, get its output
    y1 = []
    for i_row, row_of_x in enumerate(batch_of_x):
        row_of_parameters = batch_of_parameters[i_row]
        fill_parameters(net, row_of_parameters)
        y1.append(net(row_of_x))

    # Stack all the outputs generated by the torch module
    y1 = torch.stack(y1)

    # Fill the policy with the batch of parameters, and pass the batch of inputs through the policy
    policy.set_parameters(batch_of_parameters)
    y2 = policy(batch_of_x)

    # Assert that the policy's output matches the collected outputs of the torch module
    assert_allclose(y2, y1, atol=tolerance)


@torch.no_grad()
def test_policy_recurrent():
    tolerance = 1e-4

    net = LSTM(5, 8)

    parameters = parameter_vector(net)
    parameter_length = len(parameters)

    policy = Policy(clone(net))
    policy.set_parameters(parameters)
    assert policy.parameter_length == parameter_length

    x = torch.randn(3, 5)

    h1 = None
    y1 = None
    y2 = None
    for single_x in x:
        y1, h1 = net(single_x, h1)
        y2 = policy(single_x)

    assert_allclose(y2, y1, atol=tolerance)
    assert_allclose(policy.h[0], h1[0], atol=tolerance)
    assert_allclose(policy.h[1], h1[1], atol=tolerance)


@torch.no_grad()
def test_policy_recurrent_batched():
    tolerance = 1e-4
    batch_size = 10

    nets = [RNN(5, 8) for _ in range(batch_size)]

    policy = Policy(clone(nets[0]))

    parameters = torch.stack([parameter_vector(net) for net in nets])

    policy.set_parameters(parameters)
    assert policy.parameter_length == parameters.shape[1]

    num_steps = 20
    x = torch.randn(num_steps, batch_size, 5)
    dones_per_t = {
        5: [3, 7, 8],
        9: [True, True, False, True, False, False, True, True, False, False],
        14: [0, 1],
    }

    h1 = torch.zeros(batch_size, 8)
    y1 = torch.zeros(batch_size, 8)
    y2 = None
    for t in range(num_steps):
        single_x = x[t]

        for i_row, net in enumerate(nets):
            y1[i_row], h1[i_row] = net(single_x[i_row], h1[i_row])

        y2 = policy(single_x)

        if t in dones_per_t:
            mask = dones_per_t[t]
            h1[mask] = 0
            policy.reset(mask)

    assert_allclose(y2, y1, atol=tolerance)
    assert_allclose(policy.h, h1, atol=tolerance)
