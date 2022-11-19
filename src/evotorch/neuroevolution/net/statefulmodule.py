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
from torch import nn


class StatefulModule(nn.Module):
    """
    A wrapper that provides a stateful interface for recurrent torch modules.

    If the torch module to be wrapped is non-recurrent and its forward method
    has a single input (the input tensor) and a single output (the output
    tensor), then this wrapper module acts as a no-op wrapper.

    If the torch module to be wrapped is recurrent and its forward method has
    two inputs (the input tensor and an optional second argument for the hidden
    state) and two outputs (the output tensor and the new hidden state), then
    this wrapper brings a new forward-passing interface. In this new interface,
    the forward method has a single input (the input tensor) and a single
    output (the output tensor). The hidden states, instead of being
    explicitly requested via a second argument and returned as a second
    result, are stored and used by the wrapper.
    When a new series of inputs is to be used, one has to call the `reset()`
    method of this wrapper.
    """

    def __init__(self, wrapped_module: nn.Module):
        """
        `__init__(...)`: Initialize the StatefulModule.

        Args:
            wrapped_module: The `torch.nn.Module` instance to wrap.
        """
        super().__init__()

        # Declare the variable that will store the hidden state of wrapped_module, if any.
        self._hidden: Any = None

        # Store the module that is wrapped.
        self.wrapped_module = wrapped_module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._hidden is None:
            # If there is no stored hidden state, then only pass the input tensor to the wrapped module.
            out = self.wrapped_module(x)
        else:
            # If there is a hidden state saved from the previous call to this `forward(...)` method, then pass the
            # input tensor and this stored hidden state.
            out = self.wrapped_module(x, self._hidden)

        if isinstance(out, tuple):
            # If the result of the wrapped module is a tuple, then we assume that the wrapped module returned an
            # output tensor and a hidden state. We assume the first element of this tuple as the output tensor,
            # and the second element as the new hidden state.
            # We set the variable y to the output tensor, and we store the new hidden state via the attribute
            # `_hidden`.
            y, self._hidden = out
        else:
            # If the result of the wrapped module is not a tuple, then we assume that the wrapped module returned
            # only the output tensor. We set the variable y to the output tensor, and set the attribute `_hidden`
            # as None to indicate that there was no hidden state received.
            y = out
            self._hidden = None

        # We return y, which stores the output received by the wrapped module.
        return y

    def reset(self):
        """
        Reset the hidden state, if any.
        """
        self._hidden = None


def ensure_stateful(net: nn.Module) -> StatefulModule:
    """
    Ensure that a module is wrapped by StatefulModule.

    If the given module is already wrapped by StatefulModule, then the
    module itself is returned.
    If the given module is not wrapped by StatefulModule, then this function
    first wraps the module via a new StatefulModule instance, and then this
    new wrapper is returned.

    Args:
        net: The `torch.nn.Module` to be wrapped by StatefulModule (if it is
            not already wrapped by it).
    Returns:
        The module `net`, wrapped by StatefulModule.
    """
    if not isinstance(net, StatefulModule):
        return StatefulModule(net)
    return net
