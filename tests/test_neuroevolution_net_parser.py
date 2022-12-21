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

import pytest
import torch

from evotorch.neuroevolution.net import MultiLayered, NetParsingError, str_to_net


def test_str_to_net():
    net = str_to_net("Linear(4, 2)")
    assert isinstance(net, torch.nn.Linear)
    assert net.in_features == 4
    assert net.out_features == 2


def test_str_to_net_with_kwargs():
    net = str_to_net("Linear(in_features=4, out_features=2)")
    assert isinstance(net, torch.nn.Linear)
    assert net.in_features == 4
    assert net.out_features == 2


def test_str_to_net_with_kwargs_and_spaces():
    net = str_to_net("Linear( in_features = 4 , out_features = 2 )")
    assert isinstance(net, torch.nn.Linear)
    assert net.in_features == 4
    assert net.out_features == 2


def test_str_to_net_with_kwargs_and_spaces_and_comments():
    net = str_to_net("Linear( in_features = 4 , out_features = 2 ) # comment")
    assert isinstance(net, torch.nn.Linear)
    assert net.in_features == 4
    assert net.out_features == 2


def test_str_to_net_with_kwargs_and_spaces_and_comments_and_newlines():
    net = str_to_net(
        """Linear( in_features = 4 , out_features = 2 ) # comment

    """
    )
    assert isinstance(net, torch.nn.Linear)
    assert net.in_features == 4
    assert net.out_features == 2


def test_str_to_net_more_complex():
    net = str_to_net("Linear(8, 16) >> Tanh() >> Linear(16, 4, bias=False) >> ReLU()")

    assert isinstance(net, MultiLayered)
    assert len(net) == 4
    assert isinstance(net[0], torch.nn.Linear)
    assert net[0].in_features == 8
    assert net[0].out_features == 16
    assert isinstance(net[1], torch.nn.Tanh)
    assert isinstance(net[2], torch.nn.Linear)
    assert net[2].in_features == 16
    assert net[2].out_features == 4
    assert net[2].bias is None
    assert isinstance(net[3], torch.nn.ReLU)


def test_str_to_net_more_complex_multiline():
    net = str_to_net(
        """
        Linear(8, 16)
        >> Tanh()
        >> Linear(16, 4, bias=False)
        >> ReLU()
        """
    )

    assert isinstance(net, MultiLayered)
    assert len(net) == 4
    assert isinstance(net[0], torch.nn.Linear)
    assert net[0].in_features == 8
    assert net[0].out_features == 16
    assert isinstance(net[1], torch.nn.Tanh)
    assert isinstance(net[2], torch.nn.Linear)
    assert net[2].in_features == 16
    assert net[2].out_features == 4
    assert net[2].bias is None
    assert isinstance(net[3], torch.nn.ReLU)


def test_str_to_net_more_complex_with_constants():
    net = str_to_net(
        """
        Linear(input_size, hidden_size)
        >> Tanh()
        >> Linear(hidden_size, output_size[0], bias=False)
        >> ReLU()
        """,
        input_size=8,
        hidden_size=16,
        output_size=(4,),
    )

    assert isinstance(net, MultiLayered)
    assert len(net) == 4
    assert isinstance(net[0], torch.nn.Linear)
    assert net[0].in_features == 8
    assert net[0].out_features == 16
    assert isinstance(net[1], torch.nn.Tanh)
    assert isinstance(net[2], torch.nn.Linear)
    assert net[2].in_features == 16
    assert net[2].out_features == 4
    assert net[2].bias is None
    assert isinstance(net[3], torch.nn.ReLU)


@pytest.mark.parametrize(
    "net_str, error_msg",
    [
        (
            "NonExistentModule(4, 2)",
            "NetParsingError: Unrecognized module class: 'NonExistentModule'",
        ),
        (
            "Linear(4, 2) + Linear(4, 2)",
            r"NetParsingError at line\(1\) at column\(1\): Binary operators other than '>>' are not recognized.",
        ),
        (
            "Linear(input_size, 2)",
            r"NetParsingError at line\(1\) at column\(8\): Unknown constant: input_size. Available constants: \['hidden_size'\]",
        ),
        (
            "Linear(1, 1 + 2)",
            r"NetParsingError at line\(1\) at column\(11\): When trying to parse expression, encountered: ValueError\(.+\)",
        ),
        (
            "Linear(4,2) if True else None",
            r"NetParsingError at line\(1\) at column\(1\): Unrecognized expression of type .+",
        ),
        (
            "Linear(i[1 + 2],2)",
            r"NetParsingError at line\(1\) at column\(10\): Expected a simple indexing operation, but got a .+",
        ),
    ],
)
def test_str_to_net_raises_unrecognized_module(net_str, error_msg):
    with pytest.raises(NetParsingError, match=error_msg):
        str_to_net(net_str, hidden_size=16)
