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

from copy import deepcopy
from typing import Any

import torch
from torch import nn

try:
    from torch.func import functional_call
except ImportError:
    from torch.nn.utils.stateless import functional_call

from contextlib import nullcontext


def _shape_length(shape: tuple) -> int:
    """
    Return the number of elements implied by a shape tuple.

    Args:
        shape: A tuple or a torch.Size instance whose implied number of
            elements is being queried.
    Returns:
        An integer which represents the number of elements implied by the
        given shape tuple.
    """
    result = 1
    for item in shape:
        result *= int(item)
    return result


class ModuleExpectingFlatParameters:
    """
    A wrapper which brings a functional interface around a torch module.

    Similar to `functorch.FunctionalModule`, `ModuleExpectingFlatParameters`
    turns a `torch.nn.Module` instance to a function which expects a new
    leftmost argument representing the parameters of the network.
    Unlike `functorch.FunctionalModule`, a `ModuleExpectingFlatParameters`
    instance, as its name suggests, expects the network parameters to be
    given as a 1-dimensional (i.e. flattened) tensor.
    Also, unlike `functorch.FunctionalModule`, an instance of
    `ModuleExpectingFlatParameters` is NOT an instance of `torch.nn.Module`.

    PyTorch modules with buffers can be wrapped by this class, but it is
    assumed that those buffers are constant. If the wrapped module changes
    the value(s) of its buffer(s) during its forward passes, most probably
    things will NOT work right.

    As an example, let us consider the following linear layer.

    ```python
    import torch
    from torch import nn

    net = nn.Linear(3, 8)
    ```

    The functional counterpart of `net` can be obtained via:

    ```python
    from evotorch.neuroevolution.net import ModuleExpectingFlatParameters

    fnet = ModuleExpectingFlatParameters(net)
    ```

    Now, `fnet` is a callable object which expects network parameters
    and network inputs. Let us call `fnet` with randomly generated network
    parameters and with a randomly generated input tensor.

    ```python
    param_length = fnet.parameter_length
    random_parameters = torch.randn(param_length)
    random_input = torch.randn(3)

    result = fnet(random_parameters, random_input)
    ```
    """

    @torch.no_grad()
    def __init__(self, net: nn.Module, *, disable_autograd_tracking: bool = False):
        """
        `__init__(...)`: Initialize the `ModuleExpectingFlatParameters` instance.

        Args:
            net: The module that is to be wrapped by a functional interface.
            disable_autograd_tracking: If given as True, all operations
                regarding the wrapped module will be performed in the context
                `torch.no_grad()`, forcefully disabling the autograd.
                If given as False, autograd will not be affected.
                The default is False.
        """

        # Declare the variables which will store information regarding the parameters of the module.
        self.__param_names = []
        self.__param_shapes = []
        self.__param_length = 0
        self.__param_slices = []
        self.__num_params = 0

        # Iterate over the parameters of the module and fill the related information.
        i = 0
        j = 0
        for pname, p in net.named_parameters():
            self.__param_names.append(pname)

            shape = p.shape
            self.__param_shapes.append(shape)

            length = _shape_length(shape)
            self.__param_length += length

            j = i + length
            self.__param_slices.append(slice(i, j))
            i = j

            self.__num_params += 1

        self.__buffer_dict = {bname: b.clone() for bname, b in net.named_buffers()}

        self.__net = deepcopy(net)
        self.__net.to("meta")
        self.__disable_autograd_tracking = bool(disable_autograd_tracking)

    def __transfer_buffers(self, x: torch.Tensor):
        """
        Transfer the buffer tensors to the device of the given tensor.

        Args:
            x: The tensor whose device will also store the buffer tensors.
        """
        for bname in self.__buffer_dict.keys():
            self.__buffer_dict[bname] = torch.as_tensor(self.__buffer_dict[bname], device=x.device)

    @property
    def buffers(self) -> tuple:
        """Get the stored buffers"""
        return tuple(self.__buffer_dict)

    @property
    def parameter_length(self) -> int:
        return self.__param_length

    def __call__(self, parameter_vector: torch.Tensor, x: torch.Tensor, h: Any = None) -> Any:
        """
        Call the wrapped module's forward pass procedure.

        Args:
            parameter_vector: A 1-dimensional tensor which represents the
                parameters of the tensor.
            x: The inputs.
            h: Hidden state(s), in case this is a recurrent network.
        Returns:
            The result of the forward pass.
        """
        if parameter_vector.ndim != 1:
            raise ValueError(
                f"Expected the parameters as 1 dimensional,"
                f" but the received parameter vector has {parameter_vector.ndim} dimensions"
            )
        if len(parameter_vector) != self.__param_length:
            raise ValueError(
                f"Expected a parameter vector of length {self.__param_length},"
                f" but the received parameter vector's length is {len(parameter_vector)}."
            )
        state_args = [] if h is None else [h]

        params_and_buffers = {}
        for i, pname in enumerate(self.__param_names):
            param_slice = self.__param_slices[i]
            param_shape = self.__param_shapes[i]
            param = parameter_vector[param_slice].reshape(param_shape)
            params_and_buffers[pname] = param

        # Make sure that the buffer tensors are in the same device with x
        self.__transfer_buffers(x)

        # Add the buffer tensors to the dictionary `params_and_buffers`
        params_and_buffers.update(self.__buffer_dict)

        # Prepare the no-gradient context if gradient tracking is disabled
        context = torch.no_grad() if self.__disable_autograd_tracking else nullcontext()

        # Run the module and return the results
        with context:
            return functional_call(self.__net, params_and_buffers, tuple([x, *state_args]))


def make_functional_module(net: nn.Module, *, disable_autograd_tracking: bool = False) -> ModuleExpectingFlatParameters:
    """
    Wrap a torch module so that it has a functional interface.

    Similar to `functorch.make_functional(...)`, this function turns a
    `torch.nn.Module` instance to a function which expects a new leftmost
    argument representing the parameters of the network.
    Unlike with `functorch.make_functional(...)`, the parameters of the
    network are expected in a 1-dimensional (i.e. flattened) tensor.

    PyTorch modules with buffers can be wrapped by this class, but it is
    assumed that those buffers are constant. If the wrapped module changes
    the value(s) of its buffer(s) during its forward passes, most probably
    things will NOT work right.

    As an example, let us consider the following linear layer.

    ```python
    import torch
    from torch import nn

    net = nn.Linear(3, 8)
    ```

    The functional counterpart of `net` can be obtained via:

    ```python
    from evotorch.neuroevolution.net import make_functional_module

    fnet = make_functional_module(net)
    ```

    Now, `fnet` is a callable object which expects network parameters
    and network inputs. Let us call `fnet` with randomly generated network
    parameters and with a randomly generated input tensor.

    ```python
    param_length = fnet.parameter_length
    random_parameters = torch.randn(param_length)
    random_input = torch.randn(3)

    result = fnet(random_parameters, random_input)
    ```

    Args:
        net: The `torch.nn.Module` instance to be wrapped by a functional
            interface.
        disable_autograd_tracking: If given as True, all operations
            regarding the wrapped module will be performed in the context
            `torch.no_grad()`, forcefully disabling the autograd.
            If given as False, autograd will not be affected.
            The default is False.
    Returns:
        The functional wrapper, as an instance of
        `evotorch.neuroevolution.net.ModuleExpectingFlatParameters`.
    """
    return ModuleExpectingFlatParameters(net, disable_autograd_tracking=disable_autograd_tracking)
