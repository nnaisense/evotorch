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

from typing import Any

import torch
from functorch import make_functional_with_buffers
from torch import nn


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

    For obtaining the functional interface, this class internally uses
    the `functorch` library.

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

    def __init__(self, module: nn.Module, disable_autograd_tracking: bool = False):
        # Declare the variables which will store information regarding the parameters of the module.
        self.__param_shapes = []
        self.__param_length = 0
        self.__param_slices = []
        self.__num_params = 0
        self.__buffers = []

        # Iterate over the parameters of the module and fill the related information.
        i = 0
        j = 0
        for p in module.parameters():
            shape = p.shape
            self.__param_shapes.append(shape)

            length = _shape_length(shape)
            self.__param_length += length

            j = i + length
            self.__param_slices.append(slice(i, j))
            i = j

            self.__num_params += 1

        self.__fmodel, _, self.__buffers = make_functional_with_buffers(
            module, disable_autograd_tracking=bool(disable_autograd_tracking)
        )

        self.__buffers = list(self.__buffers)

    def __transfer_buffers(self, x: torch.Tensor):
        """
        Transfer the buffer tensors to the device of the given tensor.

        Args:
            x: The tensor whose device will also store the buffer tensors.
        """
        n = len(self.__buffers)
        for i in range(n):
            self.__buffers[i] = torch.as_tensor(self.__buffers[i], device=x.device)

    @property
    def buffers(self) -> tuple:
        """Get the stored buffers"""
        return self.__buffers

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

        params = []
        for i in range(self.__num_params):
            param_slice = self.__param_slices[i]
            param_shape = self.__param_shapes[i]
            param = parameter_vector[param_slice].reshape(param_shape)
            params.append(param)

        # Make sure that the tensors are in the same device with x
        self.__transfer_buffers(x)

        # Run the functional module and return the results
        return self.__fmodel(params, self.__buffers, x, *state_args)


def make_functional_module(net: nn.Module) -> ModuleExpectingFlatParameters:
    """
    Wrap a torch module so that it has a functional interface.

    For obtaining a functional interface, this function internally uses the
    `functorch` library.

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
    Returns:
        The functional wrapper, as an instance of
        `evotorch.neuroevolution.net.ModuleExpectingFlatParameters`.
    """
    return ModuleExpectingFlatParameters(net)
