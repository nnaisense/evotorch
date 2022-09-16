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

from typing import Optional

import pytest
import torch
from torch import nn

from evotorch.neuroevolution.net import (
    MultiLayered,
    fill_parameters,
    layers,
    make_functional_module,
    parameter_vector,
    str_to_net,
)
from evotorch.testing import assert_allclose


class Unbatched:
    # These are wrappers around `torch.nn.LSTM` and `torch.nn.RNN`.
    # The interface is simplified so that the inputs, outputs, and the hidden states have neither a batch dimension
    # nor a timestep dimension.

    class LSTM(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.lstm = nn.LSTM(*args, **kwargs)

        def forward(self, x: torch.Tensor, h: Optional[tuple] = None) -> tuple:
            if h is not None:
                a, b = h
                a = a.reshape(1, 1, -1)
                b = b.reshape(1, 1, -1)
                h = a, b

            x = x.reshape(1, 1, -1)
            x, h = self.lstm(x, h)
            x = x.reshape(-1)

            a, b = h
            a = a.reshape(-1)
            b = b.reshape(-1)
            h = a, b

            return x, h

    class RNN(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.rnn = nn.RNN(*args, **kwargs)

        def forward(self, x: torch.Tensor, h: Optional[torch.Tensor] = None) -> tuple:
            if h is not None:
                h = h.reshape(1, 1, -1)
            x = x.reshape(1, 1, -1)
            x, h = self.rnn(x, h)
            x = x.reshape(-1)
            h = h.reshape(-1)
            return x, h


@pytest.mark.parametrize("class_name", ["LSTM", "RNN"])
def test_recurrent_network(class_name: str):
    tolerance = 1e-2

    # Randomly generate some inputs
    inputs = torch.randn(4, 3)

    # Get the unbatched version of the recurrent network from `torch.nn`
    net_from_unbatched = getattr(Unbatched, class_name)(3, 5)

    # Get the EvoTorch counterpart of the recurrent network.
    net_from_layers = getattr(layers, class_name)(3, 5)

    # Get the parameter vector from the first network and fill the parameters of the second network using its values.
    nn_params = parameter_vector(net_from_unbatched)
    fill_parameters(net_from_layers, nn_params)

    # Now that both networks have the same parameters, they should generate the same outputs.
    # We now feed both networks the same inputs.
    with torch.no_grad():
        h_of_unbatched_net = None
        h_of_layers_net = None
        for x in inputs:
            y_of_unbatched_net, h_of_unbatched_net = net_from_unbatched(x, h_of_unbatched_net)
            y_of_layers_net, h_of_layers_net = net_from_layers(x, h_of_layers_net)

    # Check if both networks report the same output.
    assert_allclose(y_of_layers_net, y_of_unbatched_net, atol=tolerance)


def test_multilayered():
    # When recurrent networks are NOT involved, a MultiLayered module should behave like a `nn.Sequential instance`.
    # Here, we instantiate an `nn.Sequential` and a MultiLayered, and fill them with the same layers.
    tolerance = 1e-2

    layer1 = nn.Linear(3, 5)
    layer2 = nn.Linear(5, 2)

    net1 = nn.Sequential(layer1, nn.Tanh(), layer2)
    net2 = MultiLayered(layer1, nn.Tanh(), layer2)

    x = torch.rand(3)

    with torch.no_grad():
        y1 = net1(x)
        y2 = net2(x)

    # Given the same inputs, these two networks should generate the same output.
    assert_allclose(y2, y1, atol=tolerance)


class DummyRecurrentNet(nn.Module):
    def __init__(self, first_value: int = 1):
        super().__init__()
        self.first_value = int(first_value)

    def forward(self, x: torch.Tensor, h: Optional[torch.Tensor] = None) -> tuple:
        if h is None:
            h = torch.tensor(self.first_value, dtype=torch.int64, device=x.device)
        return x * torch.as_tensor(h, dtype=x.dtype, device=x.device), h + 1


@torch.no_grad()
def test_multilayered_with_recurrence():
    # When it contains recurrent layers, a MultiLayered stores the hidden states of its layers in a dictionary,
    # and then returns that dictionary as its own hidden state.
    # In this test, we instantiate a MultiLayered with two recurrent layers, and then check if its generates
    # its outputs correctly after receiving two inputs.
    tolerance = 1e-3

    net = MultiLayered(DummyRecurrentNet(first_value=1), nn.ReLU(), DummyRecurrentNet(first_value=2))

    x = torch.tensor([-1.0, 2.0, 3.0], dtype=torch.float32)

    x, h = net(x)
    assert_allclose(x, [0.0, 4.0, 6.0], atol=tolerance)
    assert isinstance(h, dict)
    assert set(h.keys()) == {0, 2}
    assert int(h[0]) == 2
    assert int(h[2]) == 3

    x, h = net(x, h)
    assert_allclose(x, [0.0, 24.0, 36.0], atol=tolerance)
    assert isinstance(h, dict)
    assert set(h.keys()) == {0, 2}
    assert int(h[0]) == 3
    assert int(h[2]) == 4


@torch.no_grad()
def test_str_to_net():
    tolerance = 1e-2

    # Make an `nn.Sequential` instance, and an equivalent network using `str_to_net(...)`.
    net = nn.Sequential(nn.Linear(3, 5), nn.Tanh(), nn.Linear(5, 2))
    snet = str_to_net("Linear(3, 5) >> Tanh() >> Linear(5, 2)")

    # Make sure that both networks are using the same parameters.
    params = parameter_vector(net)
    fill_parameters(snet, params)

    # Generate random inputs
    x = torch.randn(3)

    # Feed both networks the same inputs
    y = net(x)
    sy = snet(x)

    # Assert that the outputs match
    assert_allclose(sy, y, rtol=tolerance)


class DummyComposedRecurrent(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([Unbatched.RNN(3, 5), nn.Linear(5, 8), Unbatched.LSTM(8, 2)])

    def forward(self, x: torch.Tensor, h: Optional[dict] = None) -> tuple:
        if h is None:
            h = {0: None, 2: None}

        x, h[0] = self.layers[0](x, h[0])
        x = self.layers[1](x)
        x = torch.tanh(x)
        x, h[2] = self.layers[2](x, h[2])

        return x, h


@torch.no_grad()
def test_str_to_net_with_recurrence():
    tolerance = 1e-2

    net = DummyComposedRecurrent()
    snet = str_to_net("RNN(3, 5) >> Linear(5, 8) >> Tanh() >> LSTM(8, 2)")

    params = parameter_vector(net)
    fill_parameters(snet, params)

    inputs = torch.randn(10, 3)

    h = None
    sh = None
    for x in inputs:
        y, h = net(x, h)
        sy, sh = snet(x, sh)

    assert_allclose(sy, y, atol=tolerance)


@torch.no_grad()
def test_make_functional_module():
    tolerance = 1e-2

    net = str_to_net("Linear(3, 8) >> Tanh() >> Linear(8, 2)")
    fnet = make_functional_module(net)

    params = parameter_vector(net)

    x = torch.randn(3)
    y1 = net(x)
    y2 = fnet(params, x)

    assert_allclose(y2, y1, atol=tolerance)


@torch.no_grad()
def test_make_functional_module_with_lstm():
    tolerance = 1e-2

    net = str_to_net("LSTM(3, 8) >> Linear(8, 2)")
    fnet = make_functional_module(net)

    params = parameter_vector(net)

    inputs = torch.randn(7, 3)

    h1 = None
    h2 = None
    for x in inputs:
        y1, h1 = net(x, h1)
        y2, h2 = fnet(params, x, h2)

    assert_allclose(y2, y1, atol=tolerance)
