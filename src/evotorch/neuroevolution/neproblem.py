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

"""This namespace contains the `NeuroevolutionProblem` class."""

import math
from copy import deepcopy
from typing import Any, Callable, Iterable, List, Optional, Union

import ray
import torch
from torch import nn

from ..core import BoundsPairLike, DType, ObjectiveSense, Solution
from ..tools.misc import Device, is_sequence, pass_info_if_needed
from .baseneproblem import BaseNEProblem
from .net.misc import count_parameters, fill_parameters
from .net.parser import str_to_net
from .net.statefulmodule import ensure_stateful


class NEProblem(BaseNEProblem):
    """
    Base class for neuro-evolution problems where the goal is to optimize the
    parameters of a neural network represented as a PyTorch module.

    Any problem inheriting from this class is expected to override the method
    `_evaluate_network(self, net: torch.nn.Module) -> Union[torch.Tensor, float]`
    where `net` is the neural network to be evaluated, and the return value
    is a scalar or a vector (for multi-objective cases) expressing the
    fitness value(s).

    Alternatively, this class can be directly instantiated in the following
    way:

    ```python
    def f(module: MyTorchModuleClass) -> Union[float, torch.Tensor, tuple]:
        # Evaluate the given PyTorch module here
        fitness = ...
        return fitness


    problem = NEProblem("min", MyTorchModuleClass, f, ...)
    ```

    which specifies that the problem's goal is to minimize the return of the
    function `f`.
    For multi-objective cases, the fitness returned by `f` is expected as a
    1-dimensional tensor. For when the problem has additional evaluation data,
    a two-element tuple can be returned by `f` instead, where the first
    element is the fitness value(s) and the second element is a 1-dimensional
    tensor storing the additional data.
    """

    def __init__(
        self,
        objective_sense: ObjectiveSense,
        network: Union[str, nn.Module, Callable[[], nn.Module]],
        network_eval_func: Optional[Callable] = None,
        *,
        network_args: Optional[dict] = None,
        initial_bounds: Optional[BoundsPairLike] = (-0.00001, 0.00001),
        eval_dtype: Optional[DType] = None,
        eval_data_length: int = 0,
        seed: Optional[int] = None,
        num_actors: Optional[Union[int, str]] = None,
        actor_config: Optional[dict] = None,
        num_gpus_per_actor: Optional[Union[int, float, str]] = None,
        num_subbatches: Optional[int] = None,
        subbatch_size: Optional[int] = None,
        device: Optional[Device] = None,
    ):
        """
        `__init__(...)`: Initialize the NEProblem.

        Args:
            objective_sense: The objective sense, expected as "min" or "max"
                for single-objective cases, or as a sequence of strings
                (each string being "min" or "max") for multi-objective cases.
            network: A network structure string, or a Callable (which can be
                a class inheriting from `torch.nn.Module`, or a function
                which returns a `torch.nn.Module` instance), or an instance
                of `torch.nn.Module`.
                The object provided here determines the structure of the
                neural network whose parameters will be evolved.
                A network structure string is a string which can be processed
                by `evotorch.neuroevolution.net.str_to_net(...)`.
                Please see the documentation of the function
                `evotorch.neuroevolution.net.str_to_net(...)` to see how such
                a neural network structure string looks like.
            network_eval_func: Optionally a function (or any Callable object)
                which receives a PyTorch module as its argument, and returns
                either a fitness, or a two-element tuple containing the fitness
                and the additional evaluation data. The fitness can be a scalar
                (for single-objective cases) or a 1-dimensional tensor (for
                multi-objective cases). The additional evaluation data is
                expected as a 1-dimensional tensor.
                If this argument is left as None, it will be expected that
                the method `_evaluate_network(...)` is overriden by the
                inheriting class.
            network_args: Optionally a dict-like object, storing keyword
                arguments to be passed to the network while instantiating it.
            initial_bounds: Specifies an interval from which the values of the
                initial neural network parameters will be drawn.
            eval_dtype: dtype to be used for fitnesses. If not specified, then
                `eval_dtype` will be inferred from the dtype of the parameters
                of the neural network.
                In more details, if the neural network's parameters have a
                float dtype, `eval_dtype` will be a compatible float.
                Otherwise, it will be "float32".
            eval_data_length: Length of the extra evaluation data.
            seed: Random number seed. If left as None, this NEProblem instance
                will not have its own random generator, and the global random
                generator of PyTorch will be used instead.
            num_actors: Number of actors to create for parallelized
                evaluation of the solutions.
                Certain string values are also accepted.
                When given as "max" or as "num_cpus", the number of actors
                will be equal to the number of all available CPUs in the ray
                cluster.
                When given as "num_gpus", the number of actors will be
                equal to the number of all available GPUs in the ray
                cluster, and each actor will be assigned a GPU.
                When given as "num_devices", the number of actors will be
                equal to the minimum among the number of CPUs and the number
                of GPUs available in the cluster (or will be equal to the
                number of CPUs if there is no GPU), and each actor will be
                assigned a GPU (if available).
                If `num_actors` is given as "num_gpus" or "num_devices",
                the argument `num_gpus_per_actor` must not be used,
                and the `actor_config` dictionary must not contain the
                key "num_gpus".
                If `num_actors` is given as something other than "num_gpus"
                or "num_devices", and if you wish to assign GPUs to each
                actor, then please see the argument `num_gpus_per_actor`.
            actor_config: A dictionary, representing the keyword arguments
                to be passed to the options(...) used when creating the
                ray actor objects. To be used for explicitly allocating
                resources per each actor.
                For example, for declaring that each actor is to use a GPU,
                one can pass `actor_config=dict(num_gpus=1)`.
                Can also be given as None (which is the default),
                if no such options are to be passed.
            num_gpus_per_actor: Number of GPUs to be allocated by each
                remote actor.
                The default behavior is to NOT allocate any GPU at all
                (which is the default behavior of the ray library as well).
                When given as a number `n`, each actor will be given
                `n` GPUs (where `n` can be an integer, or can be a `float`
                for fractional allocation).
                When given as a string "max", then the available GPUs
                across the entire ray cluster (or within the local computer
                in the simplest cases) will be equally distributed among
                the actors.
                When given as a string "all", then each actor will have
                access to all the GPUs (this will be achieved by suppressing
                the environment variable `CUDA_VISIBLE_DEVICES` for each
                actor).
                When the problem is not distributed (i.e. when there are
                no actors), this argument is expected to be left as None.
            num_subbatches: If `num_subbatches` is None (assuming that
                `subbatch_size` is also None), then, when evaluating a
                population, the population will be split into n pieces, `n`
                being the number of actors, and each actor will evaluate
                its assigned piece. If `num_subbatches` is an integer `m`,
                then the population will be split into `m` pieces,
                and actors will continually accept the next unevaluated
                piece as they finish their current tasks.
                The arguments `num_subbatches` and `subbatch_size` cannot
                be given values other than None at the same time.
                While using a distributed algorithm, this argument determines
                how many sub-batches will be generated, and therefore,
                how many gradients will be computed by the remote actors.
            subbatch_size: If `subbatch_size` is None (assuming that
                `num_subbatches` is also None), then, when evaluating a
                population, the population will be split into `n` pieces, `n`
                being the number of actors, and each actor will evaluate its
                assigned piece. If `subbatch_size` is an integer `m`,
                then the population will be split into pieces of size `m`,
                and actors will continually accept the next unevaluated
                piece as they finish their current tasks.
                When there can be significant difference across the solutions
                in terms of computational requirements, specifying a
                `subbatch_size` can be beneficial, because, while one
                actor is busy with a subbatch containing computationally
                challenging solutions, other actors can accept more
                tasks and save time.
                The arguments `num_subbatches` and `subbatch_size` cannot
                be given values other than None at the same time.
                While using a distributed algorithm, this argument determines
                the size of a sub-batch (or sub-population) sampled by a
                remote actor for computing a gradient.
                In distributed mode, it is expected that the population size
                is divisible by `subbatch_size`.
            device: Default device in which a new population will be generated
                and the neural networks will operate.
                If not specified, "cpu" will be used.
        """
        # Set the main device of the problem
        # Although the operation of setting the main device is done by the main Problem class,
        # here we need this at an earlier stage.
        if device is None:
            device = "cpu"
        self._device = torch.device(device)

        # Set the network
        self._original_network = network
        self._network_args = {} if network_args is None else deepcopy(network_args)
        if isinstance(self._original_network, nn.Module):
            self._original_network = self._original_network.cpu()

        # Store the function that will evaluate the network, if available
        self._network_eval_func: Optional[Callable] = network_eval_func

        self.instantiated_network: nn.Module = None

        # Create temporary network
        temp_network = self._instantiate_net(self._original_network, device="cpu")

        super().__init__(
            objective_sense=objective_sense,
            initial_bounds=initial_bounds,
            bounds=None,  # Neuroevolution is an unbounded problem
            solution_length=count_parameters(temp_network),  # The solution length is inherited from the network passed
            dtype=next(temp_network.parameters()).dtype,  # The datatype is inherited from the network passed
            eval_dtype=eval_dtype,
            device=device,
            eval_data_length=eval_data_length,
            seed=seed,
            num_actors=num_actors,
            num_gpus_per_actor=num_gpus_per_actor,
            actor_config=actor_config,
            num_subbatches=num_subbatches,
            subbatch_size=subbatch_size,
            store_solution_stats=None,
        )

    @property
    def network_device(self) -> Device:
        """The device on which the problem should place data e.g. the network"""
        cpu_device = torch.device("cpu")
        if self.is_main:
            # This is the case where this is the main process (not a remote actor)
            if self.device == cpu_device:
                # If the main device of the problem is "cpu", then we assume that the network is going to be on the cpu as well
                return cpu_device
            else:
                # If the main device of the problem is some other device, then it is that device into which the network will be put
                return self.device
        else:
            # If this is a remote actor, then the network will be put into the auxiliary device allocated for that actor
            return self.aux_device

    @property
    def _str_network_constants(self) -> dict:
        """
        Named constants which will be passed to `str_to_net`.
        To be overridden by the user for custom fixed constants for a problem.
        """
        return {}

    @property
    def _network_constants(self) -> dict:
        """
        Named constants which will be passed to the network instantiation.
        To be overridden by the user for custom fixed constants for a problem.
        """
        return {}

    def network_constants(self) -> dict:
        """Named constants which can be passed to the network instantiation"""
        constants = {}
        constants.update(self._network_constants)
        constants.update(self._network_args)
        return constants

    @property
    def _nonserialized_attribs(self) -> List[str]:
        return ["instantiated_network"]

    def _instantiate_net(self, network: Union[str, nn.Module, dict], device: Optional[Device] = None) -> nn.Module:
        """Instantiate the network on the target device, to be overridden by the user for custom behaviour
        Returns:
            instantiated_network (nn.Module): The network instantiated on the target device
        """
        # Branching point determines instantiation of network
        if isinstance(network, str):
            # Passed argument was a string representation of a torch module
            net_consts = {}
            net_consts.update(self.network_constants())
            net_consts.update(self._str_network_constants)
            instantiated_network = str_to_net(network, **net_consts)
        elif isinstance(network, nn.Module):
            # Passed argument was directly a torch module
            instantiated_network = network
        else:
            # Passed argument was callable yielding network
            instantiated_network = pass_info_if_needed(network, self._network_constants)(**self._network_args)

        # Map to device
        device = self.network_device if device is None else device
        instantiated_network = instantiated_network.to(device)

        return instantiated_network

    def _prepare(self) -> None:
        """Instantiate the network on the target device, if not already done"""
        self.instantiated_network = self._instantiate_net(self._original_network)
        # Clear reference to original network
        self._original_network = None

    def make_net(self, parameters: Iterable) -> nn.Module:
        """
        Make a new network filled with the provided parameters.

        Args:
            parameters: Parameters to be used as weights within the network.
                Can be a Solution, or any 1-dimensional Iterable that can be
                converted to a PyTorch tensor.
        Returns:
            A new network, as a `torch.Module` instance.
        """
        if isinstance(parameters, Solution):
            parameters = parameters.access_values(keep_evals=True)
        else:
            parameters = self.as_tensor(parameters)
        with torch.no_grad():
            net = deepcopy(self.parameterize_net(parameters))
        return net

    def parameterize_net(self, parameters: torch.Tensor) -> nn.Module:
        """Parameterize the network with a given set of parameters.
        Args:
            parameters (torch.Tensor): The parameters with which to instantiate the network
        Returns:
            instantiated_network (nn.Module): The network instantiated with the parameters
        """
        # Check if network exists
        if self.instantiated_network is None:
            self.instantiated_network = self._instantiate_net(self._original_network)

        network = self.instantiated_network

        # Move the parameters if needed
        if parameters.device != self.network_device:
            parameters = parameters.to(self.network_device)

        # Fill the network with the parameters
        fill_parameters(network, parameters)

        # Return the network
        return network

    @property
    def _grad_device(self) -> Device:
        """
        Get the device in which new solutions will be made in distributed mode.

        In more details, in distributed mode, each actor creates its own
        sub-populations, evaluates them, and computes its own gradient
        (all such actor gradients eventually being collected by the
        distribution-based search algorithm in the main process).
        For some problem types, it can make sense for the remote actors to
        create their temporary sub-populations on another device
        (e.g. on the GPU that is allocated specifically for them).
        For such situations, one is encouraged to override this property
        and make it return whatever device is to be used.

        In the case of NEProblem, this property returns whatever device
        is specified by the property `network_device`.
        """
        return self.network_device

    def _evaluate_network(self, network: nn.Module) -> Union[float, torch.Tensor, tuple]:
        """
        Evaluate a network and return the evaluation result(s).

        In the case where the `__init__` of `NEProblem` was not given
        a network evaluator function (via the argument `network_eval_func`),
        it will be expected that the inheriting class overrides this
        method and defines how a network should be evaluated.

        Args:
            network (nn.Module): The network to evaluate
        Returns:
            fitness: The networks' fitness value(s), as a scalar for
                single-objective cases, or as a 1-dimensional tensor
                for multi-objective cases. The returned value can also
                be a two-element tuple where the first element is the
                fitness (as a scalar or as a vector) and the second
                element is a 1-dimensional vector storing the extra
                evaluation data.
        """
        raise NotImplementedError

    def _evaluate(self, solution: Solution):
        """
        Evaluate a single solution.
        This is achieved by parameterising the problem's attribute
        named `instantiated_network`, and then evaluating the network
        with the method `_evaluate_network(...)`.

        Args:
            solution (Solution): The solution to evaluate.
        """
        parameters = solution.values

        if self._network_eval_func is None:
            evaluator = self._evaluate_network
        else:
            evaluator = self._network_eval_func

        fitnesses = evaluator(self.parameterize_net(parameters))

        if isinstance(fitnesses, tuple):
            solution.set_evals(*fitnesses)
        else:
            solution.set_evals(fitnesses)
