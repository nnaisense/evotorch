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

"""Various neural network layer types"""

from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn.functional as nnf
from torch import nn


class Clip(nn.Module):
    """A small torch module for clipping the values of tensors"""

    def __init__(self, lb: float, ub: float):
        """`__init__(...)`: Initialize the Clip operator.

        Args:
            lb: Lower bound. Values less than this will be clipped.
            ub: Upper bound. Values greater than this will be clipped.
        """
        nn.Module.__init__(self)
        self._lb = float(lb)
        self._ub = float(ub)

    def forward(self, x: torch.Tensor):
        return x.clamp(self._lb, self._ub)

    def extra_repr(self):
        return "lb={}, ub={}".format(self._lb, self._ub)


class Bin(nn.Module):
    """A small torch module for binning the values of tensors.

    In more details, considering a lower bound value lb,
    an upper bound value ub, and an input tensor x,
    each value within x closer to lb will be converted to lb
    and each value within x closer to ub will be converted to ub.
    """

    def __init__(self, lb: float, ub: float):
        """`__init__(...)`: Initialize the Clip operator.

        Args:
            lb: Lower bound
            ub: Upper bound
        """
        nn.Module.__init__(self)
        self._lb = float(lb)
        self._ub = float(ub)
        self._interval_size = self._ub - self._lb
        self._shrink_amount = self._interval_size / 2.0
        self._shift_amount = (self._ub + self._lb) / 2.0

    def forward(self, x: torch.Tensor):
        x = x - self._shift_amount
        x = x / self._shrink_amount
        x = torch.sign(x)
        x = x * self._shrink_amount
        x = x + self._shift_amount
        return x

    def extra_repr(self):
        return "lb={}, ub={}".format(self._lb, self._ub)


class Slice(nn.Module):
    """A small torch module for getting the slice of an input tensor"""

    def __init__(self, from_index: int, to_index: int):
        """`__init__(...)`: Initialize the Slice operator.

        Args:
            from_index: The index from which the slice begins.
            to_index: The exclusive index at which the slice ends.
        """
        nn.Module.__init__(self)
        self._from_index = from_index
        self._to_index = to_index

    def forward(self, x):
        return x[self._from_index : self._to_index]

    def extra_repr(self):
        return "from_index={}, to_index={}".format(self._from_index, self._to_index)


class Round(nn.Module):
    """A small torch module for rounding the values of an input tensor"""

    def __init__(self, ndigits: int = 0):
        nn.Module.__init__(self)
        self._ndigits = int(ndigits)
        self._q = 10.0**self._ndigits

    def forward(self, x):
        x = x * self._q
        x = torch.round(x)
        x = x / self._q
        return x

    def extra_repr(self):
        return "ndigits=" + str(self._ndigits)


class Apply(nn.Module):
    """A torch module for applying an arithmetic operator on an input tensor"""

    def __init__(self, operator: str, argument: float):
        """`__init__(...)`: Initialize the Apply module.

        Args:
            operator: Must be '+', '-', '*', '/', or '**'.
                Indicates which operation will be done
                on the input tensor.
            argument: Expected as a float, represents
                the right-argument of the operation
                (the left-argument being the input
                tensor).
        """
        nn.Module.__init__(self)

        self._operator = str(operator)
        assert self._operator in ("+", "-", "*", "/", "**")

        self._argument = float(argument)

    def forward(self, x):
        op = self._operator
        arg = self._argument
        if op == "+":
            return x + arg
        elif op == "-":
            return x - arg
        elif op == "*":
            return x * arg
        elif op == "/":
            return x / arg
        elif op == "**":
            return x**arg
        else:
            raise ValueError("Unknown operator:" + repr(op))

    def extra_repr(self):
        return "operator={}, argument={}".format(repr(self._operator), self._argument)


class RNN(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        nonlinearity: str = "tanh",
        *,
        dtype: torch.dtype = torch.float32,
        device: Union[str, torch.device] = "cpu",
    ):
        super().__init__()

        input_size = int(input_size)
        hidden_size = int(hidden_size)
        nonlinearity = str(nonlinearity)

        self.W1 = nn.Parameter(torch.randn(hidden_size, input_size, dtype=dtype, device=device))
        self.W2 = nn.Parameter(torch.randn(hidden_size, hidden_size, dtype=dtype, device=device))
        self.b1 = nn.Parameter(torch.zeros(hidden_size, dtype=dtype, device=device))
        self.b2 = nn.Parameter(torch.zeros(hidden_size, dtype=dtype, device=device))

        if nonlinearity == "tanh":
            self.actfunc = torch.tanh
        else:
            self.actfunc = getattr(nnf, nonlinearity)

        self.nonlinearity = nonlinearity
        self.input_size = input_size
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor, h: Optional[torch.Tensor] = None) -> tuple:
        if h is None:
            h = torch.zeros(self.hidden_size, dtype=x.dtype, device=x.device)
        act = self.actfunc
        W1 = self.W1
        W2 = self.W2
        b1 = self.b1.unsqueeze(-1)
        b2 = self.b2.unsqueeze(-1)
        x = x.unsqueeze(-1)
        h = h.unsqueeze(-1)
        y = act(((W1 @ x) + b1) + ((W2 @ h) + b2))
        y = y.squeeze(-1)
        return y, y

    def __repr__(self) -> str:
        clsname = type(self).__name__
        return f"{clsname}(input_size={self.input_size}, hidden_size={self.hidden_size}, nonlinearity={repr(self.nonlinearity)})"


class LSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        *,
        dtype: torch.dtype = torch.float32,
        device: Union[str, torch.device] = "cpu",
    ):
        super().__init__()
        input_size = int(input_size)
        hidden_size = int(hidden_size)

        self.input_size = input_size
        self.hidden_size = hidden_size

        def input_weight():
            return nn.Parameter(torch.randn(self.hidden_size, self.input_size, dtype=dtype, device=device))

        def weight():
            return nn.Parameter(torch.randn(self.hidden_size, self.hidden_size, dtype=dtype, device=device))

        def bias():
            return nn.Parameter(torch.zeros(self.hidden_size, dtype=dtype, device=device))

        self.W_ii = input_weight()
        self.W_if = input_weight()
        self.W_ig = input_weight()
        self.W_io = input_weight()

        self.W_hi = weight()
        self.W_hf = weight()
        self.W_hg = weight()
        self.W_ho = weight()

        self.b_ii = bias()
        self.b_if = bias()
        self.b_ig = bias()
        self.b_io = bias()

        self.b_hi = bias()
        self.b_hf = bias()
        self.b_hg = bias()
        self.b_ho = bias()

    def forward(self, x: torch.Tensor, hidden=None) -> tuple:
        sigm = torch.sigmoid
        tanh = torch.tanh

        if hidden is None:
            h_prev = torch.zeros(self.hidden_size, dtype=x.dtype, device=x.device)
            c_prev = torch.zeros(self.hidden_size, dtype=x.dtype, device=x.device)
        else:
            h_prev, c_prev = hidden

        i_t = sigm(self.W_ii @ x + self.b_ii + self.W_hi @ h_prev + self.b_hi)
        f_t = sigm(self.W_if @ x + self.b_if + self.W_hf @ h_prev + self.b_hf)
        g_t = tanh(self.W_ig @ x + self.b_ig + self.W_hg @ h_prev + self.b_hg)
        o_t = sigm(self.W_io @ x + self.b_io + self.W_ho @ h_prev + self.b_ho)
        c_t = f_t * c_prev + i_t * g_t
        h_t = o_t * tanh(c_t)

        return h_t, (h_t, c_t)

    def __repr__(self) -> str:
        clsname = type(self).__name__
        return f"{clsname}(input_size={self.input_size}, hidden_size={self.hidden_size})"


RecurrentNet = RNN
LSTMNet = LSTM


class FeedForwardNet(nn.Module):
    """
    Representation of a feed forward neural network as a torch Module.

    An example initialization of a FeedForwardNet is as follows:

        net = drt.FeedForwardNet(4, [(8, 'tanh'), (6, 'tanh')])

    which means that we would like to have a network which expects an input
    vector of length 4 and passes its input through 2 tanh-activated hidden
    layers (with neurons count 8 and 6, respectively).
    The output of the last hidden layer (of length 6) is the final
    output vector.

    The string representation of the module obtained via the example above
    is:

        FeedForwardNet(
          (layer_0): Linear(in_features=4, out_features=8, bias=True)
          (actfunc_0): Tanh()
          (layer_1): Linear(in_features=8, out_features=6, bias=True)
          (actfunc_1): Tanh()
        )
    """

    LengthActTuple = Tuple[int, Union[str, Callable]]
    LengthActBiasTuple = Tuple[int, Union[str, Callable], Union[bool]]

    def __init__(self, input_size: int, layers: List[Union[LengthActTuple, LengthActBiasTuple]]):
        """`__init__(...)`: Initialize the FeedForward network.

        Args:
            input_size: Input size of the network, expected as an int.
            layers: Expected as a list of tuples,
                where each tuple is either of the form
                `(layer_size, activation_function)`
                or of the form
                `(layer_size, activation_function, bias)`
                in which
                (i) `layer_size` is an int, specifying the number of neurons;
                (ii) `activation_function` is None, or a callable object,
                or a string containing the name of the activation function
                ('relu', 'selu', 'elu', 'tanh', 'hardtanh', or 'sigmoid');
                (iii) `bias` is a boolean, specifying whether the layer
                is to have a bias or not.
                When omitted, bias is set to True.
        """

        nn.Module.__init__(self)

        for i, layer in enumerate(layers):
            if len(layer) == 2:
                size, actfunc = layer
                bias = True
            elif len(layer) == 3:
                size, actfunc, bias = layer
            else:
                assert False, "A layer tuple of invalid size is encountered"

            setattr(self, "layer_" + str(i), nn.Linear(input_size, size, bias=bias))

            if isinstance(actfunc, str):
                if actfunc == "relu":
                    actfunc = nn.ReLU()
                elif actfunc == "selu":
                    actfunc = nn.SELU()
                elif actfunc == "elu":
                    actfunc = nn.ELU()
                elif actfunc == "tanh":
                    actfunc = nn.Tanh()
                elif actfunc == "hardtanh":
                    actfunc = nn.Hardtanh()
                elif actfunc == "sigmoid":
                    actfunc = nn.Sigmoid()
                elif actfunc == "round":
                    actfunc = Round()
                else:
                    raise ValueError("Unknown activation function: " + repr(actfunc))

            setattr(self, "actfunc_" + str(i), actfunc)

            input_size = size

    def forward(self, x):
        i = 0
        while hasattr(self, "layer_" + str(i)):
            x = getattr(self, "layer_" + str(i))(x)
            f = getattr(self, "actfunc_" + str(i))
            if f is not None:
                x = f(x)
            i += 1
        return x


class StructuredControlNet(nn.Module):
    """Structured Control Net.

    This is a control network consisting of two components:
    (i) a non-linear component, which is a feed-forward network; and
    (ii) a linear component, which is a linear layer.
    Both components take the input vector provided to the
    structured control network.
    The final output is the sum of the outputs of both components.

    Reference:
        Mario Srouji, Jian Zhang, Ruslan Salakhutdinov (2018).
        Structured Control Nets for Deep Reinforcement Learning.
    """

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        num_layers: int,
        hidden_size: int,
        bias: bool = True,
        nonlinearity: Union[str, Callable] = "tanh",
    ):
        """`__init__(...)`: Initialize the structured control net.

        Args:
            in_features: Length of the input vector
            out_features: Length of the output vector
            num_layers: Number of hidden layers for the non-linear component
            hidden_size: Number of neurons in a hidden layer of the
                non-linear component
            bias: Whether or not the linear component is to have bias
            nonlinearity: Activation function
        """

        nn.Module.__init__(self)

        self._in_features = in_features
        self._out_features = out_features
        self._num_layers = num_layers
        self._hidden_size = hidden_size
        self._bias = bias
        self._nonlinearity = nonlinearity

        self._linear_component = nn.Linear(
            in_features=self._in_features, out_features=self._out_features, bias=self._bias
        )

        self._nonlinear_component = FeedForwardNet(
            input_size=self._in_features,
            layers=(
                list((self._hidden_size, self._nonlinearity) for _ in range(self._num_layers))
                + [(self._out_features, self._nonlinearity)]
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """TODO: documentation"""
        return self._linear_component(x) + self._nonlinear_component(x)

    @property
    def in_features(self):
        """TODO: documentation"""
        return self._in_features

    @property
    def out_features(self):
        """TODO: documentation"""
        return self._out_features

    @property
    def num_layers(self):
        """TODO: documentation"""
        return self._num_layers

    @property
    def hidden_size(self):
        """TODO: documentation"""
        return self._hidden_size

    @property
    def bias(self):
        """TODO: documentation"""
        return self._bias

    @property
    def nonlinearity(self):
        """TODO: documentation"""
        return self._nonlinearity


class LocomotorNet(nn.Module):
    """LocomotorNet: A locomotion-specific structured control net.

    This is a control network which consists of two components:
    one linear, and one non-linear. The non-linear component
    is an input-independent set of sinusoidals waves whose
    amplitudes, frequencies and phases are trainable.
    Upon execution of a forward pass, the output of the non-linear
    component is the sum of all these sinusoidal waves.
    The linear component is a linear layer (optionally with bias)
    whose weights (and biases) are trainable.
    The final output of the LocomotorNet at the end of a forward pass
    is the sum of the linear and the non-linear components.

    Note that this is a stateful network, where the only state
    is the timestep t, which starts from 0 and gets incremented by 1
    at the end of each forward pass. The `reset()` method resets
    t back to 0.

    Reference:
        Mario Srouji, Jian Zhang, Ruslan Salakhutdinov (2018).
        Structured Control Nets for Deep Reinforcement Learning.
    """

    def __init__(self, *, in_features: int, out_features: int, bias: bool = True, num_sinusoids=16):
        """`__init__(...)`: Initialize the LocomotorNet.

        Args:
            in_features: Length of the input vector
            out_features: Length of the output vector
            bias: Whether or not the linear component is to have a bias
            num_sinusoids: Number of sinusoidal waves
        """

        nn.Module.__init__(self)

        self._in_features = in_features
        self._out_features = out_features
        self._bias = bias
        self._num_sinusoids = num_sinusoids

        self._linear_component = nn.Linear(
            in_features=self._in_features, out_features=self._out_features, bias=self._bias
        )

        self._amplitudes = nn.ParameterList()
        self._frequencies = nn.ParameterList()
        self._phases = nn.ParameterList()

        for _ in range(self._num_sinusoids):
            for paramlist in (self._amplitudes, self._frequencies, self._phases):
                paramlist.append(nn.Parameter(torch.randn(self._out_features, dtype=torch.float32)))

        self.reset()

    def reset(self):
        """Set the timestep t to 0"""
        self._t = 0

    @property
    def t(self) -> int:
        """The current timestep t"""
        return self._t

    @property
    def in_features(self) -> int:
        """Get the length of the input vector"""
        return self._in_features

    @property
    def out_features(self) -> int:
        """Get the length of the output vector"""
        return self._out_features

    @property
    def num_sinusoids(self) -> int:
        """Get the number of sinusoidal waves of the non-linear component"""
        return self._num_sinusoids

    @property
    def bias(self) -> bool:
        """Get whether or not the linear component has bias"""
        return self._bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute a forward pass"""
        u_linear = self._linear_component(x)

        t = self._t
        u_nonlinear = torch.zeros(self._out_features)
        for i in range(self._num_sinusoids):
            A = self._amplitudes[i]
            w = self._frequencies[i]
            phi = self._phases[i]
            u_nonlinear = u_nonlinear + (A * torch.sin(w * t + phi))

        self._t += 1

        return u_linear + u_nonlinear
