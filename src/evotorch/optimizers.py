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
Optimizers (like Adam or ClipUp) to be used with distribution-based
search algorithms.
"""

from typing import Callable, Optional, Type

import torch

from ._main_logger import logger as _evolog
from .tools.misc import Device, DType, RealOrVector, ensure_tensor_length_and_dtype, message_from, to_torch_dtype


class TorchOptimizer:
    """
    Base class for using a PyTorch optimizer
    """

    def __init__(
        self,
        torch_optimizer: Type,
        *,
        config: dict,
        solution_length: int,
        dtype: DType,
        device: Device = "cpu",
    ):
        """
        `__init__(...)`: Initialize the TorchOptimizer.

        Args:
            torch_optimizer: The class which represents a PyTorch optimizer.
            config: The configuration dictionary to be passed to the optimizer
                as keyword arguments.
            solution_length: Length of a solution of the problem on which the
                optimizer will work.
            dtype: The dtype of the problem.
            device: The device on which the solutions are kept.
        """
        self._data = torch.empty(int(solution_length), dtype=to_torch_dtype(dtype), device=device)
        self._optim = torch_optimizer([self._data], **config)

    @torch.no_grad()
    def ascent(self, globalg: RealOrVector, *, cloned_result: bool = True) -> torch.Tensor:
        """
        Compute the ascent, i.e. the step to follow.

        Args:
            globalg: The estimated gradient.
            cloned_result: If `cloned_result` is True, then the result is a
                copy, guaranteed not to be the view of any other tensor
                internal to the TorchOptimizer class.
                If `cloned_result` is False, then the result is not a copy.
                Use `cloned_result=False` only when you are sure that your
                algorithm will never do direct modification on the ascent
                vector it receives.
        Returns:
            The ascent vector, representing the step to follow.
        """

        globalg = ensure_tensor_length_and_dtype(
            globalg,
            len(self._data),
            dtype=self._data.dtype,
            device=self._data.device,
            about=f"{type(self).__name__}.ascent",
        )

        self._data.zero_()
        self._data.grad = globalg
        self._optim.step()
        result = -1.0 * self._data

        return result


class Adam(TorchOptimizer):
    """
    The Adam optimizer.

    Reference:

        Kingma, D. P. and J. Ba (2015).
        Adam: A method for stochastic optimization.
        In Proceedings of 3rd International Conference on Learning Representations.
    """

    def __init__(
        self,
        *,
        solution_length: int,
        dtype: DType,
        device: Device = "cpu",
        stepsize: Optional[float] = None,
        beta1: Optional[float] = None,
        beta2: Optional[float] = None,
        epsilon: Optional[float] = None,
        amsgrad: Optional[bool] = None,
    ):
        """
        `__init__(...)`: Initialize the Adam optimizer.

        Args:
            solution_length: Length of a solution of the problem which is
                being worked on.
            dtype: The dtype of the problem which is being worked on.
            device: The device on which the solutions are kept.
            stepsize: The step size (i.e. the learning rate) employed
                by the optimizer.
            beta1: The beta1 hyperparameter. None means the default.
            beta2: The beta2 hyperparameter. None means the default.
            epsilon: The epsilon hyperparameters. None means the default.
            amsgrad: Whether or not to use the amsgrad behavior.
                None means the default behavior.
                See `torch.optim.Adam` for details.
        """

        config = {}

        if stepsize is not None:
            config["lr"] = float(stepsize)

        if beta1 is None and beta2 is None:
            pass  # nothing to do
        elif beta1 is not None and beta2 is not None:
            config["betas"] = (float(beta1), float(beta2))
        else:
            raise ValueError(
                "The arguments beta1 and beta2 were expected"
                " as both None, or as both real numbers."
                " However, one of them was encountered as None and"
                " the other was encountered as something other than None."
            )

        if epsilon is not None:
            config["eps"] = float(epsilon)

        if amsgrad is not None:
            config["amsgrad"] = bool(amsgrad)

        super().__init__(torch.optim.Adam, solution_length=solution_length, dtype=dtype, device=device, config=config)


class SGD(TorchOptimizer):
    """
    The SGD optimizer.

    Reference regarding the momentum behavior:

        Polyak, B. T. (1964).
        Some methods of speeding up the convergence of iteration methods.
        USSR Computational Mathematics and Mathematical Physics, 4(5):1–17.

    Reference regarding the Nesterov behavior:

        Yurii Nesterov (1983).
        A method for unconstrained convex minimization problem with the rate ofconvergence o(1/k2).
        Doklady ANSSSR (translated as Soviet.Math.Docl.), 269:543–547.
    """

    def __init__(
        self,
        *,
        solution_length: int,
        dtype: DType,
        stepsize: float,
        device: Device = "cpu",
        momentum: Optional[float] = None,
        dampening: Optional[bool] = None,
        nesterov: Optional[bool] = None,
    ):
        """
        `__init__(...)`: Initialize the SGD optimizer.

        Args:
            solution_length: Length of a solution of the problem which is
                being worked on.
            dtype: The dtype of the problem which is being worked on.
            stepsize: The step size (i.e. the learning rate) employed
                by the optimizer.
            device: The device on which the solutions are kept.
            momentum: The momentum coefficient. None means the default.
            dampening: Whether or not to activate the dampening behavior.
                None means the default.
                See `torch.optim.SGD` for details.
            nesterov: Whether or not to activate the nesterov behavior.
                None means the default.
                See `torch.optim.SGD` for details.
        """

        config = {}

        config["lr"] = float(stepsize)

        if momentum is not None:
            config["momentum"] = float(momentum)

        if dampening is not None:
            config["dampening"] = float(dampening)

        if nesterov is not None:
            config["nesterov"] = bool(nesterov)

        super().__init__(torch.optim.SGD, solution_length=solution_length, dtype=dtype, device=device, config=config)


class ClipUp:
    """
    The ClipUp optimizer.

    Although this optimizer has the very same interface with SGD and Adam,
    it is not a PyTorch optimizer. Therefore, it does not inherit from
    TorchOptimizer.

    Reference:

        Toklu, N. E., Liskowski, P., & Srivastava, R. K. (2020, September).
        ClipUp: A Simple and Powerful Optimizer for Distribution-Based Policy Evolution.
        In International Conference on Parallel Problem Solving from Nature (pp. 515-527).
        Springer, Cham.
    """

    def __init__(
        self,
        *,
        solution_length: int,
        dtype: DType,
        stepsize: float,
        momentum: float = 0.9,
        max_speed: Optional[float] = None,
        device: Device = "cpu",
    ):
        """
        `__init__(...)`: Initialize the ClipUp optimizer.

        Args:
            solution_length: Length of a solution of the problem which is
                being worked on.
            dtype: The dtype of the problem which is being worked on.
            stepsize: The step size (i.e. the learning rate) employed
                by the optimizer.
            momentum: The momentum coefficient. None means the default.
            max_speed: The maximum speed. If given as None, the
                `max_speed` will be taken as two times the stepsize.
            device: The device on which the solutions are kept.
        """

        stepsize = float(stepsize)
        momentum = float(momentum)
        if max_speed is None:
            max_speed = stepsize * 2.0
            _evolog.info(
                message_from(
                    self,
                    (
                        f"The maximum speed for the ClipUp optimizer is set as {max_speed}"
                        f" which is two times the given step size."
                    ),
                )
            )
        else:
            max_speed = float(max_speed)
        solution_length = int(solution_length)

        if stepsize < 0.0:
            raise ValueError(f"Invalid stepsize: {stepsize}")
        if momentum < 0.0 or momentum > 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if max_speed < 0.0:
            raise ValueError(f"Invalid max_speed: {max_speed}")

        self._stepsize = stepsize
        self._momentum = momentum
        self._max_speed = max_speed

        self._velocity: Optional[torch.Tensor] = torch.zeros(
            solution_length, dtype=to_torch_dtype(dtype), device=device
        )

        self._dtype = to_torch_dtype(dtype)
        self._device = device

    @staticmethod
    def _clip(x: torch.Tensor, limit: float) -> torch.Tensor:
        with torch.no_grad():
            normx = torch.norm(x)
            if normx > limit:
                ratio = limit / normx
                return x * ratio
            else:
                return x

    @torch.no_grad()
    def ascent(self, globalg: RealOrVector, *, cloned_result: bool = True) -> torch.Tensor:
        """
        Compute the ascent, i.e. the step to follow.

        Args:
            globalg: The estimated gradient.
            cloned_result: If `cloned_result` is True, then the result is a
                copy, guaranteed not to be the view of any other tensor
                internal to the TorchOptimizer class.
                If `cloned_result` is False, then the result is not a copy.
                Use `cloned_result=False` only when you are sure that your
                algorithm will never do direct modification on the ascent
                vector it receives.
                Important: if you set `cloned_result=False`, and do in-place
                modifications on the returned result of `ascent(...)`, then
                the internal velocity of ClipUp will be corrupted!
        Returns:
            The ascent vector, representing the step to follow.
        """

        globalg = ensure_tensor_length_and_dtype(
            globalg,
            len(self._velocity),
            dtype=self._dtype,
            device=self._device,
            about=f"{type(self).__name__}.ascent",
        )

        grad = (globalg / torch.norm(globalg)) * self._stepsize

        self._velocity = self._clip((self._momentum * self._velocity) + grad, self._max_speed)

        result = self._velocity

        if cloned_result:
            result = result.clone()

        return result


def get_optimizer_class(s: str, optimizer_config: Optional[dict] = None) -> Callable:
    """
    Get the optimizer class from the given string.

    Args:
        s: A string, referring to the optimizer class.
            "clipsgd", "clipsga", "clipup" refers to ClipUp.
            "adam" refers to Adam.
            "sgd" or "sga" refers to SGD.
        optimizer_config: A dictionary containing the configurations to be
            passed to the optimizer. If this argument is not None,
            then, instead of the class being referred to, a dynamically
            generated factory function will be returned, which will pass
            these configurations to the actual class upon being called.
    Returns:
        The class, or a factory function instantiating that class.
    """
    if s in ("clipsgd", "clipsga", "clipup"):
        cls = ClipUp
    elif s == "adam":
        cls = Adam
    elif s in ("sgd", "sga"):
        cls = SGD
    else:
        raise ValueError(f"Unknown optimizer: {repr(s)}")

    if optimizer_config is None:
        return cls
    else:

        def f(*args, **kwargs):
            nonlocal cls, optimizer_config
            conf = {}
            conf.update(optimizer_config)
            conf.update(kwargs)
            return cls(*args, **conf)

        return f
