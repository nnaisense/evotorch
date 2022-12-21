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

"""
Utilities for parsing string representations of neural net policies
"""

import ast
from pprint import pprint
from typing import Any, Callable, Optional, Type

import torch
from torch import nn

from . import layers
from .multilayered import MultiLayered


class NetParsingError(Exception):
    """
    Representation of a parsing error
    """

    def __init__(
        self,
        message: str,
        lineno: Optional[int] = None,
        col_offset: Optional[int] = None,
        original_error: Optional[Exception] = None,
    ):
        """
        `__init__(...)`: Initialize the NetParsingError.

        Args:
            message: Error message, as string.
            lineno: Erroneous line number in the string representation of the
                neural network structure.
            col_offset: Erroneous column number in the string representation
                of the neural network structure.
            original_error: If another error caused this parsing error,
                that original error can be attached to this `NetParsingError`
                instance via this argument.
        """
        super().__init__()
        self.message = message
        self.lineno = lineno
        self.col_offset = col_offset
        self.original_error = original_error

    def _to_string(self) -> str:
        parts = []

        parts.append(type(self).__name__)

        if self.lineno is not None:
            parts.append(" at line(")
            parts.append(str(self.lineno - 1))
            parts.append(")")

        if self.col_offset is not None:
            parts.append(" at column(")
            parts.append(str(self.col_offset + 1))
            parts.append(")")

        parts.append(": ")
        parts.append(self.message)

        return "".join(parts)

    def __str__(self) -> str:
        return self._to_string()

    def __repr__(self) -> str:
        return self._to_string()


def submodules(a: nn.Module) -> list:
    if isinstance(a, (nn.Sequential, MultiLayered)):
        return [module for module in a]
    else:
        return [a]


def concat_modules(a: nn.Module, b: nn.Module) -> nn.Module:
    all_modules = submodules(a) + submodules(b)
    return MultiLayered(*all_modules)


def _get_nn_module(name: str) -> Type:
    if hasattr(layers, name):
        return getattr(layers, name)
    elif hasattr(nn, name):
        return getattr(nn, name)
    else:
        raise NetParsingError(f"Unrecognized module class: {repr(name)}")


def _eval_with_constants(node: ast.expr, constants: dict) -> Any:
    def fail(msg, erroneous_node=None, original_error=None):
        nonlocal node

        if (
            erroneous_node is None
            or (not hasattr(erroneous_node, "lineno"))
            or (not hasattr(erroneous_node, "col_offset"))
        ):
            erroneous_node = node

        raise NetParsingError(msg, erroneous_node.lineno, erroneous_node.col_offset, original_error=original_error)

    def literal_eval(subnode: ast.expr):
        err = None
        result = None

        try:
            result = ast.literal_eval(subnode)
        except Exception as ex:
            err = ex

        if err is not None:
            fail(f"When trying to parse expression, encountered: {repr(err)}", subnode, original_error=err)

        return result

    def get_constant(name):
        nonlocal constants

        if name not in constants.keys():
            fail(f"Unknown constant: {name}. Available constants: {repr(list(constants.keys()))}")

        return constants[name]

    def get_from_constant(name, index):
        cnst = get_constant(name)

        err = None
        result = None

        try:
            result = cnst[index]
        except Exception as ex:
            err = ex

        if err is not None:
            fail(
                f"When applying the indexing operation on the constant {name}, encountered: {repr(err)}",
                original_error=err,
            )

        return result

    if isinstance(node, ast.Name):
        return get_constant(node.id)
    elif isinstance(node, ast.Subscript):
        name = node.value
        if not isinstance(name, ast.Name):
            fail(
                "Expression which was expected to express a simple indexing over a constant"
                " is either too complex, or is unrecognized.",
                name,
            )
        name = name.id

        index = node.slice
        if isinstance(index, ast.Index):
            index = index.value

        # Note: ast.Num is a subclass of ast.Constant in Python 3.8 and later.
        #       To support Python 3.7 and earlier, we need to check for both.
        if not isinstance(index, (ast.Constant, ast.Num)):
            fail(f"Expected a simple indexing operation, but got a {type(index).__name__}.", index)

        index = literal_eval(index)

        return get_from_constant(name, index)
    else:
        return literal_eval(node)


def _process_call_expr(node: ast.Call, constants: dict) -> nn.Module:
    args = [_eval_with_constants(arg, constants=constants) for arg in node.args]

    kwargs = {}
    for kw in node.keywords:
        kwname = kw.arg
        kwvalue = _eval_with_constants(kw.value, constants=constants)
        kwargs[kwname] = kwvalue

    name = node.func.id

    return _get_nn_module(name)(*args, **kwargs)


def _process_rshift_expr(node: ast.BinOp, constants: dict) -> nn.Module:
    if not isinstance(node.op, ast.RShift):
        raise NetParsingError("Binary operators other than '>>' are not recognized.", node.lineno, node.col_offset)
    left_module = _process_expr(node.left, constants=constants)
    right_module = _process_expr(node.right, constants=constants)
    return concat_modules(left_module, right_module)


def _process_expr(node: ast.expr, constants: dict) -> nn.Module:
    if isinstance(node, ast.Call):
        return _process_call_expr(node, constants=constants)
    elif isinstance(node, ast.BinOp):
        return _process_rshift_expr(node, constants=constants)
    else:
        raise NetParsingError(f"Unrecognized expression of type {type(node)}", node.lineno, node.col_offset)


def str_to_net(s: str, **constants) -> nn.Module:
    """
    Read a string representation of a neural net structure,
    and return a `torch.nn.Module` instance out of it.

    Let us imagine that one wants to describe the following
    neural network structure:

    ```python
    from torch import nn
    from evotorch.neuroevolution.net import MultiLayered

    net = MultiLayered(nn.Linear(8, 16), nn.Tanh(), nn.Linear(16, 4, bias=False), nn.ReLU())
    ```

    By using `str_to_net(...)` one can construct an equivalent
    module via:

    ```python
    from evotorch.neuroevolution.net import str_to_net

    net = str_to_net("Linear(8, 16) >> Tanh() >> Linear(16, 4, bias=False) >> ReLU()")
    ```

    The string can also be multi-line:

    ```python
    net = str_to_net(
        '''
        Linear(8, 16)
        >> Tanh()
        >> Linear(16, 4, bias=False)
        >> ReLU()
        '''
    )
    ```

    One can also define constants for using them in strings:

    ```python
    net = str_to_net(
        '''
        Linear(input_size, hidden_size)
        >> Tanh()
        >> Linear(hidden_size, output_size, bias=False)
        >> ReLU()
        ''',
        input_size=8,
        hidden_size=16,
        output_size=4,
    )
    ```

    In the neural net structure string, when one refers to a module type,
    say, `Linear`, first the name `Linear` is searched for in the namespace
    `evotorch.neuroevolution.net.layers`, and then in the namespace `torch.nn`.
    In the case of `Linear`, the searched name exists in `torch.nn`,
    and therefore, the layer type to be instantiated is accepted as
    `torch.nn.Linear`.
    Instead of `Linear`, if one had used the name, say,
    `StructuredControlNet`, then, the layer type to be instantiated
    would be `evotorch.neuroevolution.net.layers.StructuredControlNet`.

    The namespace `evotorch.neuroevolution.net.layers` contains its own
    implementations for RNN and LSTM. These recurrent layer implementations
    work similarly to their counterparts `torch.nn.RNN` and `torch.nn.LSTM`,
    except that EvoTorch's implementations do not expect the data with extra
    leftmost dimensions for batching and for timesteps. Instead, they expect
    to receive a single input and a single current hidden state, and produce
    a single output and a single new hidden state. These recurrent layer
    implementations of EvoTorch can be used within a neural net structure
    string. Therefore, the following examples are valid:

    ```python
    rnn1 = str_to_net("RNN(4, 8) >> Linear(8, 2)")

    rnn2 = str_to_net(
        '''
        Linear(4, 10)
        >> Tanh()
        >> RNN(input_size=10, hidden_size=24, nonlinearity='tanh'
        >> Linear(24, 2)
        '''
    )

    lstm1 = str_to_net("LSTM(4, 32) >> Linear(32, 2)")

    lstm2 = str_to_net("LSTM(input_size=4, hidden_size=32) >> Linear(32, 2)")
    ```

    **Notes regarding usage with `evotorch.neuroevolution.GymNE`
    or with `evotorch.neuroevolution.VecGymNE`:**

    While instantiating a `GymNE` or a `VecGymNE`, one can specify a neural
    net structure string as the policy. Therefore, while filling the policy
    string for a `GymNE`, all these rules mentioned above apply. Additionally,
    while using `str_to_net(...)` internally, `GymNE` and `VecGymNE` define
    these extra constants:
    `obs_length` (length of the observation vector),
    `act_length` (length of the action vector for continuous-action
    environments, or number of actions for discrete-action
    environments), and
    `obs_shape` (shape of the observation as a tuple, assuming that the
    observation space is of type `gym.spaces.Box`, usable within the string
    like `obs_shape[0]`, `obs_shape[1]`, etc., or simply `obs_shape` to refer
    to the entire tuple).

    Therefore, while instantiating a `GymNE` or a `VecGymNE`, one can define a
    single-hidden-layered policy via this string:

    ```
    "Linear(obs_length, 16) >> Tanh() >> Linear(16, act_length) >> Tanh()"
    ```

    In the policy string above, one might choose to omit the last `Tanh()`, as
    `GymNE` and `VecGymNE` will clip the final output of the policy to conform
    to the action boundaries defined by the target reinforcement learning
    environment, and such a clipping operation might be seen as using an
    activation function similar to hard-tanh anyway.

    Args:
        s: The string which expresses the neural net structure.
    Returns:
        The PyTorch module of the specified structure.
    """
    s = f"(\n{s}\n)"
    return _process_expr(ast.parse(s, mode="eval").body, constants=constants)
