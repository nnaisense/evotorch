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

"""Utilities for reading and for writing neural network parameters"""

from typing import Optional, Union

import torch
from torch import nn

from ...tools.misc import Device


@torch.no_grad()
def fill_parameters(net: nn.Module, vector: torch.Tensor):
    """Fill the parameters of a torch module (net) from a vector.

    No gradient information is kept.

    The vector's length must be exactly the same with the number
    of parameters of the PyTorch module.

    Args:
        net: The torch module whose parameter values will be filled.
        vector: A 1-D torch tensor which stores the parameter values.
    """
    address = 0
    for p in net.parameters():
        d = p.data.view(-1)
        n = len(d)
        d[:] = torch.as_tensor(vector[address : address + n], device=d.device)
        address += n

    if address != len(vector):
        raise IndexError("The parameter vector is larger than expected")


@torch.no_grad()
def parameter_vector(net: nn.Module, *, device: Optional[Device] = None) -> torch.Tensor:
    """Get all the parameters of a torch module (net) into a vector

    No gradient information is kept.

    Args:
        net: The torch module whose parameters will be extracted.
        device: The device in which the parameter vector will be constructed.
            If the network has parameter across multiple devices,
            you can specify this argument so that concatenation of all the
            parameters will be successful.
    Returns:
        The parameters of the module in a 1-D tensor.
    """
    dev_kwarg = {} if device is None else {"device": device}

    all_vectors = []
    for p in net.parameters():
        all_vectors.append(torch.as_tensor(p.data.view(-1), **dev_kwarg))

    return torch.cat(all_vectors)


def count_parameters(net: nn.Module) -> int:
    """
    Get the number of parameters the network.

    Args:
        net: The torch module whose parameters will be counted.
    Returns:
        The number of parameters, as an integer.
    """

    count = 0

    for p in net.parameters():
        count += p.numel()

    return count


def device_of_module(m: nn.Module, default: Optional[Union[str, torch.device]] = None) -> torch.device:
    """
    Get the device in which the module exists.

    This function looks at the first parameter of the module, and returns
    its device. This function is not meant to be used on modules whose
    parameters exist on different devices.

    Args:
        m: The module whose device is being queried.
        default: The fallback device to return if the module has no
            parameters. If this is left as None, the fallback device
            is assumed to be "cpu".
    Returns:
        The device of the module, determined from its first parameter.
    """
    if default is None:
        default = torch.device("cpu")

    device = default

    for p in m.parameters():
        device = p.device
        break

    return device
