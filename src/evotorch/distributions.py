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

import math
from copy import copy
from typing import Any, Callable, Iterable, NamedTuple, Optional, Type, Union

import torch
from torch.func import vmap

from .tools import (
    Device,
    DType,
    cast_tensors_in_container,
    device_of_container,
    dtype_of_container,
    make_batched_false_for_vmap,
)
from .tools import multiply_rows_by_scalars as dot
from .tools import rowwise_sum as total
from .tools import to_torch_dtype
from .tools.cloning import Serializable, deep_clone
from .tools.ranking import rank
from .tools.tensormaker import TensorMakerMixin


class Distribution(TensorMakerMixin, Serializable):
    """
    Base class for any search distribution.
    """

    MANDATORY_PARAMETERS = set()
    OPTIONAL_PARAMETERS = set()
    PARAMETER_NDIMS = {}

    functional_sample = NotImplemented

    def __init__(
        self, *, solution_length: int, parameters: dict, dtype: Optional[DType] = None, device: Optional[Device] = None
    ):
        """
        `__init__(...)`: Initialize the Distribution.

        It is expected that one of these two conditions is met:
        (i) the inheriting search distribution class does not implement its
        own `__init__(...)` method; or
        (ii) the inheriting search distribution class has its own
        `__init__(...)` method, and calls `Distribution.__init__(...)`
        from there, during its initialization phase.

        Args:
            solution_length: Expected as an integer, this argument represents
                the solution length.
            parameters: Expected as a dictionary, this argument stores
                the parameters of the search distribution.
                For example, for a Gaussian distribution where `mu`
                represents the mean, and `sigma` represents the coverage
                area, this dictionary would have the keys "mu" and "sigma",
                and each of these keys would map to a PyTorch tensor.
            dtype: The dtype of the search distribution (e.g. torch.float32).
            device: The device of the search distribution (e.g. "cpu").
        """
        self.__solution_length: int = int(solution_length)

        self.__parameters: dict
        self.__dtype: torch.dtype
        self.__device: torch.device

        self.__check_correctness(parameters)

        cast_kwargs = {}
        if dtype is not None:
            cast_kwargs["dtype"] = to_torch_dtype(dtype)
        if device is not None:
            cast_kwargs["device"] = torch.device(device)

        if len(cast_kwargs) == 0:
            self.__parameters = copy(parameters)
        else:
            self.__parameters = cast_tensors_in_container(parameters, **cast_kwargs)

        self.__dtype = cast_kwargs.get("dtype", dtype_of_container(parameters))
        self.__device = cast_kwargs.get("device", device_of_container(parameters))

    def __check_correctness(self, parameters: dict):
        found_mandatory = 0
        for param_name in parameters.keys():
            if param_name in self.MANDATORY_PARAMETERS:
                found_mandatory += 1
            elif param_name in self.OPTIONAL_PARAMETERS:
                pass  # nothing to do
            else:
                raise ValueError(f"Unrecognized parameter: {repr(param_name)}")
        if found_mandatory < len(self.MANDATORY_PARAMETERS):
            raise ValueError(
                f"Not all mandatory parameters of this Distribution were specified."
                f" Mandatory parameters of this distribution: {self.MANDATORY_PARAMETERS};"
                f" optional parameters of this distribution: {self.OPTIONAL_PARAMETERS};"
                f" encountered parameters: {set(parameters.keys())}."
            )

    def to(self, device: Device) -> "Distribution":
        """
        Bring the Distribution onto a computational device.

        If the given device is already the device of this Distribution,
        then the Distribution itself will be returned.
        If the given device is different than the device of this
        Distribution, a copy of this Distribution on the given device
        will be created and returned.

        Args:
            device: The computation device onto which the Distribution
                will be brought.
        Returns:
            The Distribution on the target device.
        """
        if torch.device(self.device) == torch.device(device):
            return self
        else:
            cls = self.__class__
            return cls(solution_length=self.solution_length, parameters=self.parameters, device=device)

    def _fill(self, out: torch.Tensor, *, generator: Optional[torch.Generator] = None):
        """
        Fill the given tensor with samples from this search distribution.

        It is expected that the inheriting search distribution class
        has its own implementation for this method.

        Args:
            out: The PyTorch tensor that will be filled with the samples.
                This tensor is expected as 2-dimensional with its number
                of columns equal to the solution length declared by this
                distribution.
            generator: Optionally a PyTorch generator, to be used for
                sampling. None means that the global generator of PyTorch
                is to be used.
        """
        raise NotImplementedError

    def sample(
        self,
        num_solutions: Optional[int] = None,
        *,
        out: Optional[torch.Tensor] = None,
        generator: Any = None,
    ) -> torch.Tensor:
        """
        Sample solutions from this search distribution.

        Args:
            num_solutions: How many solutions will be sampled.
                If this argument is given as an integer and the argument
                `out` is left as None, then a new PyTorch tensor, filled
                with the samples from this distribution, will be generated
                and returned. The number of rows of this new tensor will
                be equal to the given `num_solutions`.
                If the argument `num_solutions` is provided as an integer,
                then the argument `out` is expected as None.
            out: The PyTorch tensor that will be filled with the samples
                of this distribution. This tensor is expected as a
                2-dimensional tensor with its number of columns equal to
                the solution length declared by this distribution.
                If the argument `out` is provided as a tensor, then the
                argument `num_solutions` is expected as None.
            generator: Optionally a PyTorch generator or any object which
                has a `generator` attribute (e.g. a Problem instance).
                If left as None, the global generator of PyTorch will be
                used.
        Returns:
            A 2-dimensional PyTorch tensor which stores the sampled solutions.
        """
        if (num_solutions is not None) and (out is not None):
            raise ValueError(
                "Received both `num_solutions` and `out` with values other than None."
                "Please provide only one of these arguments with a value other than None, not both of them."
            )
        elif (num_solutions is not None) and (out is None):
            num_solutions = int(num_solutions)
            out = self.make_empty(num_solutions=num_solutions)
            out = out + make_batched_false_for_vmap(out.device)
        elif (num_solutions is None) and (out is not None):
            if out.ndim != 2:
                raise ValueError(
                    f"The `sample(...)` method can fill only 2-dimensional tensors."
                    f" However, the provided `out` tensor has {out.ndim} dimensions, its shape being {out.shape}."
                )
            _, num_cols = out.shape
            if num_cols != self.solution_length:
                raise ValueError(
                    f"The solution length declared by this distribution is {self.solution_length}."
                    f" However, the provided `out` tensor has {num_cols} columns."
                    f" The `sample(...)` method can only work with tensors whose number of columns are equal"
                    f" to the declared solution length."
                )
        else:
            raise ValueError(
                "Received both `num_solutions` and `out` as None."
                "Please provide one of these arguments with a value other than None."
            )
        self._fill(out, generator=generator)
        return out

    def _compute_gradients(self, samples: torch.Tensor, weights: torch.Tensor, ranking_used: Optional[str]) -> dict:
        """
        Compute the gradients out of the samples (sampled solutions)
        and weights (i.e. weights or ranks of the solutions, better
        solutions having numerically higher weights).

        It is expected that the inheriting class implements this method.

        Args:
            samples: The sampled solutions, as a 2-dimensional tensor.
            weights: Solution weights, as a 1-dimensional tensor of length
                `n`, `n` being the number of sampled solutions.
            ranking_used: Ranking that was used to obtain the weights.
        Returns:
            The gradient(s) in a dictionary.
        """
        raise NotImplementedError

    def compute_gradients(
        self,
        samples: torch.Tensor,
        fitnesses: torch.Tensor,
        *,
        objective_sense: str,
        ranking_method: Optional[str] = None,
    ) -> dict:
        """
        Compute and return gradients.

        Args:
            samples: The solutions that were sampled from this Distribution.
                The tensor passed via this argument is expected to have
                the same dtype and device with this Distribution.
            fitnesses: The evaluation results of the sampled solutions.
                If fitnesses are given with a different dtype (maybe because
                the eval_dtype of the Problem object is different than its
                decision variable dtype), then this method will first
                create an internal copy of the fitnesses with the correct
                dtype, and then will use those copied fitnesses for
                computing the gradients.
            objective_sense: The objective sense, expected as "min" or "max".
                In the case of "min", lower fitness values will be regarded
                as better (therefore, in this case, one can alternatively
                refer to fitnesses as 'unfitnesses' or 'solution costs').
                In the case of "max", higher fitness values will be regarded
                as better.
            ranking_method: The ranking method to be used.
                Can be: "linear" (where ranks linearly go from 0 to 1);
                "centered" (where ranks linearly go from -0.5 to +0.5);
                "normalized" (where the standard-normalized fitnesses
                serve as ranks); or "raw" (where the fitnesses themselves
                serve as ranks).
                The default is "raw".
        Returns:
            A dictionary which contains the gradient for each parameter of the
            distribution.
        """
        if objective_sense == "max":
            higher_is_better = True
        elif objective_sense == "min":
            higher_is_better = False
        else:
            raise ValueError(
                f'`objective_sense` was expected as "min" or as "max".'
                f" However, it was encountered as {repr(objective_sense)}."
            )

        if ranking_method is None:
            ranking_method = "raw"

        # Make sure that the fitnesses are in the correct dtype
        fitnesses = torch.as_tensor(fitnesses, dtype=self.dtype)

        [num_samples, _] = samples.shape
        [num_fitnesses] = fitnesses.shape
        if num_samples != num_fitnesses:
            raise ValueError(
                f"The number of samples and the number of fitnesses do not match:" f" {num_samples} != {num_fitnesses}."
            )

        weights = rank(fitnesses, ranking_method=ranking_method, higher_is_better=higher_is_better)
        return self._compute_gradients(samples, weights, ranking_method)

    def update_parameters(
        self,
        gradients: dict,
        *,
        learning_rates: Optional[dict] = None,
        optimizers: Optional[dict] = None,
    ) -> "Distribution":
        """
        Do an update on the distribution by following the given gradients.

        It is expected that the inheriting class has its own implementation
        for this method.

        Args:
            gradients: Gradients, as a dictionary, which will be used for
                computing the necessary updates.
            learning_rates: A dictionary which contains learning rates
                for parameters that will be updated using a learning rate
                coefficient.
            optimizers: A dictionary which contains optimizer objects
                for parameters that will be updated using an adaptive
                optimizer.
        Returns:
            The updated copy of the distribution.
        """
        raise NotImplementedError

    def modified_copy(
        self, *, dtype: Optional[DType] = None, device: Optional[Device] = None, **parameters
    ) -> "Distribution":
        """
        Return a modified copy of this distribution.

        Args:
            dtype: The new dtype of the distribution.
            device: The new device of the distribution.
            parameters: Expected in the form of extra keyword arguments.
                Each of these keyword arguments will cause the new distribution
                to have a modified value for the specified parameter.
        Returns:
            The modified copy of the distribution.
        """
        cls = self.__class__
        if device is None:
            device = self.device
        if dtype is None:
            dtype = self.dtype

        new_parameters = copy(self.parameters)
        new_parameters.update(parameters)
        return cls(parameters=new_parameters, dtype=dtype, device=device)

    def relative_entropy(dist_0: "Distribution", dist_1: "Distribution") -> float:
        raise NotImplementedError

    @property
    def solution_length(self) -> int:
        return self.__solution_length

    @property
    def device(self) -> torch.device:
        return self.__device

    @property
    def dtype(self) -> torch.dtype:
        return self.__dtype

    @property
    def parameters(self) -> dict:
        return self.__parameters

    def _follow_gradient(
        self,
        param_name: str,
        x: torch.Tensor,
        *,
        learning_rates: Optional[dict] = None,
        optimizers: Optional[dict] = None,
    ) -> torch.Tensor:
        x = torch.as_tensor(x, dtype=self.dtype, device=self.device)
        learning_rate, optimizer = self._get_learning_rate_and_optimizer(param_name, learning_rates, optimizers)
        if (learning_rate is None) and (optimizer is None):
            return x
        elif (learning_rate is not None) and (optimizer is None):
            return learning_rate * x
        elif (learning_rate is None) and (optimizer is not None):
            return optimizer.ascent(x)
        else:
            raise ValueError(
                "Encountered both `learning_rate` and `optimizer` as values other than None."
                " This method can only work if both of them are None or only one of them is not None."
            )

    @staticmethod
    def _get_learning_rate_and_optimizer(
        param_name: str, learning_rates: Optional[dict], optimizers: Optional[dict]
    ) -> tuple:
        if learning_rates is None:
            learning_rates = {}
        if optimizers is None:
            optimizers = {}
        return learning_rates.get(param_name, None), optimizers.get(param_name, None)

    @torch.no_grad()
    def _get_cloned_state(self, *, memo: dict) -> dict:
        return deep_clone(
            self.__dict__,
            otherwise_deepcopy=True,
            memo=memo,
        )


class SeparableGaussian(Distribution):
    """Separable Multivariate Gaussian, as used by PGPE"""

    MANDATORY_PARAMETERS = {"mu", "sigma"}
    OPTIONAL_PARAMETERS = {"divide_mu_grad_by", "divide_sigma_grad_by", "parenthood_ratio"}
    PARAMETER_NDIMS = {"mu": 1, "sigma": 1}

    @classmethod
    def _unbatched_functional_sample(cls, num_solutions: int, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        [L] = mu.shape
        [sigma_L] = sigma.shape
        if L != sigma_L:
            raise ValueError(f"The lengths of `mu` ({L}) and `sigma` ({sigma_L}) do not match.")
        mu = mu.expand(int(num_solutions), L)
        return torch.normal(mu, sigma)

    @classmethod
    def functional_sample(cls, num_solutions: int, parameters: dict) -> torch.Tensor:
        """
        Sample and return separable Gaussian noise

        This is a static utility method, which allows one to sample separable
        Gaussian noise, without having to instantiate the distribution class
        `SeparableGaussian`.

        Args:
            num_solutions: Number of solutions (or 1-dimensional tensors)
                that will be sampled.
            parameters: A parameter dictionary. Within this parameter
                dictionary, the item `mu` is expected to store the mean, and
                the item `sigma` is expected to store the standard deviation,
                each in the form of a 1-dimensional tensor.
        Returns:
            Sampled separable Gaussian noise, as a PyTorch tensor.
            If `mu` and/or `sigma` was given as tensors with 2 or more
            dimensions (instead of only 1 dimension), the extra leftmost
            dimensions will be interpreted as batch dimensions, and therefore,
            this returned tensor will also have batch dimensions.
        """
        from .decorators import expects_ndim

        for k in parameters.keys():
            if (k not in cls.MANDATORY_PARAMETERS) and (k not in cls.OPTIONAL_PARAMETERS):
                raise ValueError(f"{cls.__name__} encountered an unrecognized parameter: {repr(k)}")
        mu = parameters["mu"]
        sigma = parameters["sigma"]
        return expects_ndim(cls._unbatched_functional_sample, (None, 1, 1), randomness="different")(
            num_solutions, mu, sigma
        )

    def __init__(
        self,
        parameters: dict,
        *,
        solution_length: Optional[int] = None,
        device: Optional[Device] = None,
        dtype: Optional[DType] = None,
    ):
        [mu_length] = parameters["mu"].shape
        [sigma_length] = parameters["sigma"].shape

        if solution_length is None:
            solution_length = mu_length
        else:
            if solution_length != mu_length:
                raise ValueError(
                    f"The argument `solution_length` does not match the length of `mu` provided in `parameters`."
                    f" solution_length={solution_length},"
                    f' parameters["mu"]={mu_length}.'
                )

        if mu_length != sigma_length:
            raise ValueError(
                f"The tensors `mu` and `sigma` provided within `parameters` have mismatching lengths."
                f' parameters["mu"]={mu_length},'
                f' parameters["sigma"]={sigma_length}.'
            )

        super().__init__(
            solution_length=solution_length,
            parameters=parameters,
            device=device,
            dtype=dtype,
        )

    @property
    def mu(self) -> torch.Tensor:
        return self.parameters["mu"]

    @mu.setter
    def mu(self, new_mu: Iterable):
        self.parameters["mu"] = torch.as_tensor(new_mu, dtype=self.dtype, device=self.device)

    @property
    def sigma(self) -> torch.Tensor:
        return self.parameters["sigma"]

    @sigma.setter
    def sigma(self, new_sigma: Iterable):
        self.parameters["sigma"] = torch.as_tensor(new_sigma, dtype=self.dtype, device=self.device)

    def _fill(self, out: torch.Tensor, *, generator: Optional[torch.Generator] = None):
        self.make_gaussian(out=out, center=self.mu, stdev=self.sigma, generator=generator)

    def _divide_grad(self, param_name: str, grad: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        option = f"divide_{param_name}_grad_by"
        if option in self.parameters:
            div_by_what = self.parameters[option]
            if div_by_what == "num_solutions":
                [num_solutions] = weights.shape
                grad = grad / num_solutions
            elif div_by_what == "num_directions":
                [num_solutions] = weights.shape
                num_directions = num_solutions // 2
                grad = grad / num_directions
            elif div_by_what == "total_weight":
                total_weight = torch.sum(torch.abs(weights))
                grad = grad / total_weight
            elif div_by_what == "weight_stdev":
                weight_stdev = torch.std(weights)
                grad = grad / weight_stdev
            else:
                raise ValueError(f"The parameter {option} has an unrecognized value: {div_by_what}")
        return grad

    def _compute_gradients_via_parenthood_ratio(self, samples: torch.Tensor, weights: torch.Tensor) -> dict:
        [num_samples, _] = samples.shape
        num_elites = math.floor(num_samples * self.parameters["parenthood_ratio"])
        elite_indices = weights.argsort(descending=True)[:num_elites]
        elites = samples[elite_indices, :]
        return {
            "mu": torch.mean(elites, dim=0) - self.parameters["mu"],
            "sigma": torch.std(elites, dim=0) - self.parameters["sigma"],
        }

    def _compute_gradients(self, samples: torch.Tensor, weights: torch.Tensor, ranking_used: Optional[str]) -> dict:
        if "parenthood_ratio" in self.parameters:
            return self._compute_gradients_via_parenthood_ratio(samples, weights)
        else:
            mu = self.mu
            sigma = self.sigma

            # Compute the scaled noises, that is, the noise vectors which
            # were used for generating the solutions
            # (solution = scaled_noise + center)
            scaled_noises = samples - mu

            # Make sure that the weights (utilities) are 0-centered
            # (Otherwise the formulations would have to consider a bias term)
            if ranking_used not in ("centered", "normalized"):
                weights = weights - torch.mean(weights)

            mu_grad = self._divide_grad(
                "mu",
                total(dot(weights, scaled_noises)),
                weights,
            )
            sigma_grad = self._divide_grad(
                "sigma",
                total(dot(weights, ((scaled_noises**2) - (sigma**2)) / sigma)),
                weights,
            )

            return {
                "mu": mu_grad,
                "sigma": sigma_grad,
            }

    def update_parameters(
        self,
        gradients: dict,
        *,
        learning_rates: Optional[dict] = None,
        optimizers: Optional[dict] = None,
    ) -> "SeparableGaussian":
        mu_grad = gradients["mu"]
        sigma_grad = gradients["sigma"]

        new_mu = self.mu + self._follow_gradient("mu", mu_grad, learning_rates=learning_rates, optimizers=optimizers)
        new_sigma = self.sigma + self._follow_gradient(
            "sigma", sigma_grad, learning_rates=learning_rates, optimizers=optimizers
        )

        return self.modified_copy(mu=new_mu, sigma=new_sigma)

    def relative_entropy(dist_0: "SeparableGaussian", dist_1: "SeparableGaussian") -> float:
        mu_0 = dist_0.parameters["mu"]
        mu_1 = dist_1.parameters["mu"]
        sigma_0 = dist_0.parameters["sigma"]
        sigma_1 = dist_1.parameters["sigma"]
        cov_0 = sigma_0.pow(2.0)
        cov_1 = sigma_1.pow(2.0)

        mu_delta = mu_1 - mu_0

        trace_cov = torch.sum(cov_0 / cov_1)
        k = dist_0.solution_length
        scaled_mu = torch.sum(mu_delta.pow(2.0) / cov_1)
        log_det = torch.sum(torch.log(cov_1)) - torch.sum(torch.log(cov_0))

        return 0.5 * (trace_cov - k + scaled_mu + log_det)


class SymmetricSeparableGaussian(SeparableGaussian):
    r"""
    Symmetric (antithetic) separable Gaussian distribution as used by PGPE.

    For example, if the desired number of samples (or number of solutions,
    provided via the argument `num_solutions`) is 6, 3 "directions" will
    be sampled. Each direction is a pair of solutions, where one of the
    solutions is the center vector plus perturbation, and the other
    solution is the center vector minus the same perturbation. Therefore,
    such a symmetric population of size 6 looks like this:

    ```
                                                   ___
    solution[0]: center + sampled_perturbation[0]     \
                                                       >  direction0
    solution[1]: center - sampled_perturbation[1]  ___/

                                                   ___
    solution[2]: center + sampled_perturbation[2]     \
                                                       >  direction1
    solution[3]: center - sampled_perturbation[3]  ___/

                                                   ___
    solution[4]: center + sampled_perturbation[4]     \
                                                       >  direction2
    solution[5]: center - sampled_perturbation[5]  ___/
    ```
    """

    MANDATORY_PARAMETERS = {"mu", "sigma"}
    OPTIONAL_PARAMETERS = {"divide_mu_grad_by", "divide_sigma_grad_by", "parenthood_ratio"}
    PARAMETER_NDIMS = {"mu": 1, "sigma": 1}

    @classmethod
    def _unbatched_functional_sample(cls, num_solutions: int, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        zeros = torch.zeros_like(mu)
        num_solutions = int(num_solutions)
        if (num_solutions % 2) != 0:
            raise ValueError(
                f"Number of solutions to be sampled from {cls.__name__} must be an even number."
                f" However, the encountered `num_solutions` is {num_solutions}."
            )
        num_directions = num_solutions // 2

        positive_ends = SeparableGaussian._unbatched_functional_sample(num_directions, zeros, sigma)
        negative_ends = -positive_ends

        positive_ends += mu
        negative_ends += mu

        combined_samples = vmap(torch.stack)([positive_ends, negative_ends]).reshape(-1, positive_ends.shape[-1])

        return combined_samples

    @classmethod
    def functional_sample(cls, num_solutions: int, parameters: dict) -> torch.Tensor:
        """
        Sample and return symmetric separable Gaussian noise

        This is a static utility method, which allows one to sample symmetric
        separable Gaussian noise, without having to instantiate the
        distribution class `SymmetricSeparableGaussian`.

        Args:
            num_solutions: Number of solutions (or 1-dimensional tensors)
                that will be sampled. Note that, since this distribution is
                symmetric, `num_solutions` must be even.
            parameters: A parameter dictionary. Within this parameter
                dictionary, the item `mu` is expected to store the mean, and
                the item `sigma` is expected to store the standard deviation,
                each in the form of a 1-dimensional tensor.
        Returns:
            Sampled symmetric separable Gaussian noise, as a PyTorch tensor.
            If `mu` and/or `sigma` was given as tensors with 2 or more
            dimensions (instead of only 1 dimension), the extra leftmost
            dimensions will be interpreted as batch dimensions, and therefore,
            this returned tensor will also have batch dimensions.
        """
        from .decorators import expects_ndim

        for k in parameters.keys():
            if (k not in cls.MANDATORY_PARAMETERS) and (k not in cls.OPTIONAL_PARAMETERS):
                raise ValueError(f"{cls.__name__} encountered an unrecognized parameter: {repr(k)}")
        mu = parameters["mu"]
        sigma = parameters["sigma"]
        return expects_ndim(cls._unbatched_functional_sample, (None, 1, 1), randomness="different")(
            num_solutions, mu, sigma
        )

    def _fill(self, out: torch.Tensor, *, generator: Optional[torch.Generator] = None):
        self.make_gaussian(out=out, center=self.mu, stdev=self.sigma, symmetric=True, generator=generator)

    def _compute_gradients(
        self,
        samples: torch.Tensor,
        weights: torch.Tensor,
        ranking_used: Optional[str],
    ) -> dict:
        if "parenthood_ratio" in self.parameters:
            return self._compute_gradients_via_parenthood_ratio(samples, weights)
        else:
            mu = self.mu
            sigma = self.sigma

            # Make sure that the weights (utilities) are 0-centered
            # (Otherwise the formulations would have to consider a bias term)
            if ranking_used not in ("centered", "normalized"):
                weights = weights - torch.mean(weights)

            [nslns] = weights.shape
            # ndirs = nslns // 2

            # Compute the scaled noises, that is, the noise vectors which
            # were used for generating the solutions
            # (solution = scaled_noise + center)
            scaled_noises = samples[0::2] - mu

            # Separate the plus and the minus ends of the directions
            fdplus = weights[0::2]
            fdminus = weights[1::2]

            # Considering that the population is stored like this:
            #                                     _
            #   solution0: center + scaled_noise0  \
            #                                       > direction0
            #   solution1: center - scaled_noise0 _/
            #                                     _
            #   solution2: center + scaled_noise1  \
            #                                       > direction1
            #   solution3: center - scaled_noise1 _/
            #
            #   ...

            # fdplus[0] becomes the utility of the plus end of direction0
            #                   (i.e. utility of solution0)

            # fdminus[0] becomes the utility of the minus end of direction0
            #                   (i.e. utility of solution1)

            # fdplus[1] becomes the utility of the plus end of direction1
            #                   (i.e. utility of solution2)

            # fdminus[1] becomes the utility of the minus end of direction1
            #                   (i.e. utility of solution3)

            # ... and so on...

            grad_mu = self._divide_grad("mu", total(dot((fdplus - fdminus) / 2, scaled_noises)), weights)
            grad_sigma = self._divide_grad(
                "sigma",
                total(dot(((fdplus + fdminus) / 2), ((scaled_noises**2) - (sigma**2)) / sigma)),
                weights,
            )

            return {
                "mu": grad_mu,
                "sigma": grad_sigma,
            }


class ExpSeparableGaussian(SeparableGaussian):
    """Exponential Separable Multivariate Gaussian, as used by SNES"""

    MANDATORY_PARAMETERS = {"mu", "sigma"}
    OPTIONAL_PARAMETERS = set()
    PARAMETER_NDIMS = {"mu": 1, "sigma": 1}

    def _compute_gradients(self, samples: torch.Tensor, weights: torch.Tensor, ranking_used: Optional[str]) -> dict:
        if ranking_used != "nes":
            weights = weights / torch.sum(torch.abs(weights))

        scaled_noises = samples - self.mu
        raw_noises = scaled_noises / self.sigma

        mu_grad = total(dot(weights, scaled_noises))
        sigma_grad = total(dot(weights, (raw_noises**2) - 1))

        return {"mu": mu_grad, "sigma": sigma_grad}

    def update_parameters(
        self,
        gradients: dict,
        *,
        learning_rates: Optional[dict] = None,
        optimizers: Optional[dict] = None,
    ) -> "ExpSeparableGaussian":
        mu_grad = gradients["mu"]
        sigma_grad = gradients["sigma"]

        new_mu = self.mu + self._follow_gradient("mu", mu_grad, learning_rates=learning_rates, optimizers=optimizers)
        new_sigma = self.sigma * torch.exp(
            0.5 * self._follow_gradient("sigma", sigma_grad, learning_rates=learning_rates, optimizers=optimizers)
        )

        return self.modified_copy(mu=new_mu, sigma=new_sigma)


class ExpGaussian(Distribution):
    """Exponential Multivariate Gaussian, as used by XNES"""

    # Corresponding to mu and A in symbols used in xNES paper
    MANDATORY_PARAMETERS = {"mu", "sigma"}

    # Inverse of sigma, numerically more stable to track this independently to sigma
    OPTIONAL_PARAMETERS = {"sigma_inv"}

    PARAMETER_NDIMS = {"mu": 1, "sigma": 2, "sigma_inv": 2}

    def __init__(
        self,
        parameters: dict,
        *,
        solution_length: Optional[int] = None,
        device: Optional[Device] = None,
        dtype: Optional[DType] = None,
    ):
        [mu_length] = parameters["mu"].shape

        # Make sigma 2D
        if len(parameters["sigma"].shape) == 1:
            parameters["sigma"] = torch.diag(parameters["sigma"])

        # Automatically generate sigma_inv if not provided
        if "sigma_inv" not in parameters:
            parameters["sigma_inv"] = torch.inverse(parameters["sigma"])

        [sigma_length, _] = parameters["sigma"].shape

        if solution_length is None:
            solution_length = mu_length
        else:
            if solution_length != mu_length:
                raise ValueError(
                    f"The argument `solution_length` does not match the length of `mu` provided in `parameters`."
                    f" solution_length={solution_length},"
                    f' parameters["mu"]={mu_length}.'
                )

        if mu_length != sigma_length:
            raise ValueError(
                f"The tensors `mu` and `sigma` provided within `parameters` have mismatching lengths."
                f' parameters["mu"]={mu_length},'
                f' parameters["sigma"]={sigma_length}.'
            )

        super().__init__(
            solution_length=solution_length,
            parameters=parameters,
            device=device,
            dtype=dtype,
        )
        # Make identity matrix as this is used throughout in gradient computation
        self.eye = self.make_I(solution_length)

    @property
    def mu(self) -> torch.Tensor:
        """Getter for mu
        Returns:
            mu (torch.Tensor): The center of the search distribution
        """
        return self.parameters["mu"]

    @mu.setter
    def mu(self, new_mu: Iterable):
        """Setter for mu
        Args:
            new_mu (torch.Tensor): The new value of mu
        """
        self.parameters["mu"] = torch.as_tensor(new_mu, dtype=self.dtype, device=self.device)

    @property
    def cov(self) -> torch.Tensor:
        """The covariance matrix A^T A"""
        return self.sigma.transpose(0, 1) @ self.sigma

    @property
    def sigma(self) -> torch.Tensor:
        """Getter for sigma
        Returns:
            sigma (torch.Tensor): The square root of the covariance matrix
        """
        return self.parameters["sigma"]

    @property
    def sigma_inv(self) -> torch.Tensor:
        """Getter for sigma_inv
        Returns:
            sigma_inv (torch.Tensor): The inverse square root of the covariance matrix
        """
        if "sigma_inv" in self.parameters:
            return self.parameters["sigma_inv"]
        else:
            return torch.inverse(self.parameters["sigma"])

    @property
    def A(self) -> torch.Tensor:
        """Alias for self.sigma, for notational consistency with paper"""
        return self.sigma

    @property
    def A_inv(self) -> torch.Tensor:
        """Alias for self.sigma_inv, for notational consistency with paper"""
        return self.sigma_inv

    @sigma.setter
    def sigma(self, new_sigma: Iterable):
        """Setter for sigma
        Args:
            new_sigma (torch.Tensor): The new value of sigma, the square root of the covariance matrix
        """
        self.parameters["sigma"] = torch.as_tensor(new_sigma, dtype=self.dtype, device=self.device)

    def to_global_coordinates(self, local_coordinates: torch.Tensor) -> torch.Tensor:
        """Map samples from local coordinate space N(0, I_d) to global coordinate space N(mu, A^T A)
        This function is the inverse of to_local_coordinates
        Args:
            local_coordinates (torch.Tensor): The local coordinates sampled from N(0, I_d)
        Returns:
            global_coordinates (torch.Tensor): The global coordinates sampled from N(mu, A^T A)
        """
        # Global samples are constructed as x = mu + A z where z is local coordinate
        # We use transpose here to simplify the batched application of A
        return self.mu.unsqueeze(0) + (self.A @ local_coordinates.T).T

    def to_local_coordinates(self, global_coordinates: torch.Tensor) -> torch.Tensor:
        """Map samples from global coordinate space N(mu, A^T A) to local coordinate space N(0, I_d)
        This function is the inverse of to_global_coordinates
        Args:
            global_coordinates (torch.Tensor): The global coordinates sampled from N(mu, A^T A)
        Returns:
            local_coordinates (torch.Tensor): The local coordinates sampled from N(0, I_d)
        """
        # Global samples are constructed as x = mu + A z where z is local coordinate
        # Therefore, we can recover z according to z = A_inv (x - mu)
        return (self.A_inv @ (global_coordinates - self.mu.unsqueeze(0)).T).T

    def _fill(self, out: torch.Tensor, *, generator: Optional[torch.Generator] = None):
        """Fill a tensor with samples from N(mu, A^T A)
        Args:
            out (torch.Tensor): The tensor to fill
            generator (Optional[torch.Generator]): A generator to use to generate random values
        """
        # Fill with local coordinates from N(0, I_d)
        self.make_gaussian(out=out, generator=generator)
        # Map local coordinates to global coordinate system
        out[:] = self.to_global_coordinates(out)

    def _compute_gradients(self, samples: torch.Tensor, weights: torch.Tensor, ranking_used: Optional[str]) -> dict:
        """Compute the gradients with respect to a given set of samples and weights
        Args:
            samples (torch.Tensor): Samples drawn from N(mu, A^T A), ideally using self._fill
            weights (torch.Tensor): Weights e.g. fitnesses or utilities assigned to samples
            ranking_used (optional[str]): The ranking method used to compute weights
        Returns:
            grads (dict): A dictionary containing the approximated natural gradient on d and M
        """
        # Compute the local coordinates
        local_coordinates = self.to_local_coordinates(samples)

        # Make sure that the weights (utilities) are 0-centered
        # (Otherwise the formulations would have to consider a bias term)
        if ranking_used not in ("centered", "normalized"):
            weights = weights - torch.mean(weights)

        d_grad = total(dot(weights, local_coordinates))
        local_coordinates_outer = local_coordinates.unsqueeze(1) * local_coordinates.unsqueeze(2)
        M_grad = torch.sum(
            weights.unsqueeze(-1).unsqueeze(-1) * (local_coordinates_outer - self.eye.unsqueeze(0)), dim=0
        )

        return {
            "d": d_grad,
            "M": M_grad,
        }

    def update_parameters(
        self,
        gradients: dict,
        *,
        learning_rates: Optional[dict] = None,
        optimizers: Optional[dict] = None,
    ) -> "ExpGaussian":
        d_grad = gradients["d"]
        M_grad = gradients["M"]

        if "d" not in learning_rates:
            learning_rates["d"] = learning_rates["mu"]
        if "M" not in learning_rates:
            learning_rates["M"] = learning_rates["sigma"]

        # Follow gradients for d, and M
        update_d = self._follow_gradient("d", d_grad, learning_rates=learning_rates, optimizers=optimizers)
        update_M = self._follow_gradient("M", M_grad, learning_rates=learning_rates, optimizers=optimizers)

        # Fold into parameters mu, A and A inv
        new_mu = self.mu + torch.mv(self.A, update_d)
        new_A = self.A @ torch.matrix_exp(0.5 * update_M)
        new_A_inv = torch.matrix_exp(-0.5 * update_M) @ self.A_inv

        # Return modified distribution
        return self.modified_copy(mu=new_mu, sigma=new_A, sigma_inv=new_A_inv)


def _get_param_ndim(distribution_class: Type, param_name: str) -> int:
    return distribution_class.PARAMETER_NDIMS.get(param_name, None)


class FunctionalSampler:
    """
    Represents a sampler returned by `make_functional_sampler`.

    Please see the documentation of
    [make_functional_sampler][evotorch.distributions.make_functional_sampler].
    """

    def __init__(
        self, distribution_class: Type, *, required_parameters: Iterable, fixed_parameters: Optional[dict] = None
    ):
        from .decorators import expects_ndim

        self.__distribution_class = distribution_class
        self.__required_parameters = [str(element) for element in required_parameters]
        if len(self.__required_parameters) == 0:
            raise TypeError("`required_parameters` cannot be empty")
        self.__required_param_pos = {
            required_parameter: i_parameter for i_parameter, required_parameter in enumerate(self.__required_parameters)
        }
        self.__num_required_parameters = len(self.__required_parameters)

        self.__fixed_parameters = {} if fixed_parameters is None else fixed_parameters
        self.__sample_batch = expects_ndim(
            self.__sample,
            (None,) + tuple(_get_param_ndim(distribution_class, p) for p in self.__required_parameters),
            randomness="different",
        )

    def __sample(self, num_solutions: int, *parameters) -> torch.Tensor:
        parameters = {**dict(zip(self.__required_parameters, parameters)), **(self.__fixed_parameters)}
        if self.__distribution_class.functional_sample is NotImplemented:
            distribution = self.__distribution_class(parameters)
            return distribution.sample(num_solutions)
        else:
            return self.__distribution_class.functional_sample(num_solutions, parameters)

    def __call__(self, num_samples: int, *parameter_args, **parameter_kwargs) -> torch.Tensor:
        num_parameter_args = len(parameter_args)
        num_parameter_kwargs = len(parameter_kwargs)
        if (num_parameter_args == 0) and (num_parameter_kwargs == self.__num_required_parameters):
            parameters = [None] * self.__num_required_parameters
            for parameter_name, parameter_value in parameter_kwargs.items():
                parameter_pos = self.__required_param_pos[parameter_name]
                parameters[parameter_pos] = parameter_value
        elif (num_parameter_args == self.__num_required_parameters) and (num_parameter_kwargs == 0):
            parameters = parameter_args
        elif (num_parameter_args == 0) and (num_parameter_kwargs == 0):
            raise TypeError("Missing parameter arguments")
        elif (num_parameter_args > 0) and (num_parameter_kwargs > 0):
            raise TypeError(
                "Specifying some of the distribution parameters via positional arguments and some others"
                " via keyword arguments is not supported."
                " Please provide the distribution parameters only via positional arguments"
                " or only via keyword arguments."
            )
        else:
            raise TypeError("Invalid number of arguments")
        return self.__sample_batch(num_samples, *parameters)


def make_functional_sampler(
    distribution_class, *, required_parameters: Iterable, fixed_parameters: Optional[dict] = None
) -> Callable:
    """
    Make a stateless function that samples from a distribution.

    This function is meant to be used when one wants to follow the functional
    programming paradigm.

    As an example, let us imagine that we are interested in sampling from the
    distribution `SymmetricSeparableGaussian`. A sampler function out of this
    distribution can be created like this:

    ```python
    from evotorch.distributions import SymmetricSeparableGaussian, make_functional_sampler

    get_samples = make_functional_sampler(
        SymmetricSeparableGaussian,
        required_parameters=["mu", "sigma"],
    )
    ```

    Now we have a function `get_samples()`, which can be used like this:

    ```python
    number_of_samples = ...  # an integer representing the desired number of samples
    mu = ...  # a one-dimensional tensor
    sigma = ...  # a one-dimensional tensor

    my_samples = get_samples(number_of_samples, mu, sigma)
    ```

    Alternatively, the parameters of the distribution can be specified via
    keyword arguments, like this:

    ```python
    my_samples = get_samples(number_of_samples, mu=mu, sigma=sigma)
    ```

    **Batched sampling.**
    A functional sampler can be further transformed via `torch.func.vmap(...)`
    for creating batched samples.

    As an alternative to `vmap`, a functional sampler has built-in support for
    batched sampling (which actually still uses `vmap` internally).
    Let us again consider our example sampler `get_samples`.
    In this example, the parameters `mu` and `sigma` would be 1-dimensional
    tensors in the non-batched case (because the distribution
    `SymmetricSeparableGaussian` expects the parameters `mu` and `sigma` as
    1-dimensional tensors, as can be observed from the class attribute
    `SymmetricSeparableGaussian.PARAMETER_NDIMS`).
    If `get_samples` is given `mu` and/or `sigma` with more than 1 dimensions,
    those extra leftmost dimensions will be considered as batch dimensions,
    and therefore the resulting sample tensor will have extra leftmost
    dimensions too.

    **Declaring fixed parameters.**
    A functional sampler can be created in such a way that some of the
    parameters are pre-defined and only a subset of the mandatory parameters
    are expected via arguments. For example, a sampler that samples from
    `SymmetricSeparableGaussian` with a fixed sigma can be defined like this:

    ```python
    predefined_sigma = ...  # The constant sigma goes here

    get_samples2 = make_functional_sampler(
        SymmetricSeparableGaussian,
        required_parameters=["mu"],
        fixed_parameters={"sigma": predefined_sigma},
    )
    ```

    The function `get_samples2` can be called like this:

    ```python
    number_of_samples = ...  # an integer, representing the desired number of samples
    mu = ...  # a one-dimensional tensor

    # or like this:
    # my_samples2 = get_samples2(number_of_samples, mu=mu)
    ```

    **How the functional sampler uses its wrapped distribution.**
    If the wrapped distribution class has a static method with the signature
    `functional_sample(num_solutions: int, parameters: dict) -> torch.Tensor`
    (which expects `num_solutions` as the number of solutions/samples
    and `parameters` as the parameter dictionary), this functional sampler
    will use that static method to obtain and return the samples.
    On the other hand, if the wrapped distribution class declares its class
    attribute `functional_sample = NotImplemented`, then, the wrapped
    distribution class will be temporarily instantiated, and then, the
    `sample()` method of this instance will be used to generate and return
    the samples.

    Args:
        distribution_class: A class that inherits from the base class
            [Distribution][evotorch.distributions.Distribution].
        required_parameters: A list of strings, each string being the name
            of a distribution parameter. The order of this list determines
            the order of parameter-related positional arguments in the
            returned callable object.
        fixed_parameters: A dictionary where the keys are parameter names
            (as strings), and the values are pre-defined parameter values.
    Returns:
        A callable object whose function is to return samples from the
        specified distribution.
    """
    return FunctionalSampler(
        distribution_class, required_parameters=required_parameters, fixed_parameters=fixed_parameters
    )


class GradsWithSamplesAndFitnesses(NamedTuple):
    grads: torch.Tensor
    samples: torch.Tensor
    fitnesses: torch.Tensor


class GradsWithSamples(NamedTuple):
    grads: torch.Tensor
    samples: torch.Tensor


class GradsWithFitnesses(NamedTuple):
    grads: torch.Tensor
    fitnesses: torch.Tensor


class FunctionalGradEstimator:
    """
    Represents the callable object returned by `make_functional_grad_estimator`.

    Please see the documentation of
    [make_functional_grad_estimator][evotorch.distributions.make_functional_grad_estimator]
    """

    def __init__(
        self,
        distribution_class: Type,
        *,
        function: Optional[Callable] = None,
        objective_sense: str,
        required_parameters: Iterable,
        fixed_parameters: Optional[dict] = None,
        ranking_method: Optional[str] = None,
        return_samples: bool = False,
        return_fitnesses: bool = False,
    ):
        from .decorators import expects_ndim

        self.__function = function
        self.__objective_sense = None if objective_sense is None else str(objective_sense)
        self.__distribution_class = distribution_class
        self.__required_parameters = [str(element) for element in required_parameters]
        if len(self.__required_parameters) == 0:
            raise TypeError("`required_parameters` cannot be empty")
        self.__required_param_pos = {
            required_parameter: i_parameter for i_parameter, required_parameter in enumerate(self.__required_parameters)
        }
        self.__num_required_parameters = len(self.__required_parameters)

        self.__fixed_parameters = {} if fixed_parameters is None else fixed_parameters
        self.__return_samples = bool(return_samples)
        self.__return_fitnesses = bool(return_fitnesses)
        self.__ranking_method = None if ranking_method is None else str(ranking_method)

        if self.__function is None:
            leftmost_ndims = (None, None, 2, 1)
        else:
            leftmost_ndims = (
                None,
                None,
                None,
            )

        self.__grad_batch = expects_ndim(
            self.__grad,
            leftmost_ndims + tuple(_get_param_ndim(distribution_class, p) for p in self.__required_parameters),
            randomness="different",
        )

    def __grad(self, *args) -> torch.Tensor:
        objective_sense = args[0]
        ranking_method = args[1]

        if self.__function is None:
            samples = args[2]
            fitnesses = args[3]
            vectors = args[4:]
            [num_solutions, _] = samples.shape
            [num_fitnesses] = fitnesses.shape
            if num_solutions != num_fitnesses:
                raise ValueError("The length of the fitness vector does not match the number of samples")
        else:
            num_solutions = args[2]
            vectors = args[3:]
            samples = None
            fitnesses = None

        parameters = {**dict(zip(self.__required_parameters, vectors)), **(self.__fixed_parameters)}
        distribution = self.__distribution_class(parameters)

        if samples is None:
            samples = distribution.sample(num_solutions)
            fitnesses = self.__function(samples)

        grads = distribution.compute_gradients(
            samples, fitnesses, objective_sense=objective_sense, ranking_method=ranking_method
        )

        if self.__return_samples and self.__return_fitnesses:
            return GradsWithSamplesAndFitnesses(grads=grads, samples=samples, fitnesses=fitnesses)
        elif self.__return_samples:
            return GradsWithSamples(grads=grads, samples=samples)
        elif self.__return_fitnesses:
            return GradsWithFitnesses(grads=grads, fitnesses=fitnesses)
        else:
            return grads

    def __call__(self, *args, **parameter_kwargs) -> torch.Tensor:
        parameters_need_filtering = False
        if "objective_sense" in parameter_kwargs:
            objective_sense = parameter_kwargs["objective_sense"]
            parameters_need_filtering = True
        else:
            objective_sense = self.__objective_sense
            if self.__objective_sense is None:
                raise ValueError(
                    "The gradient estimator was not given an `objective_sense`, neither at its phase of initialization,"
                    " nor when it got called."
                )

        if "ranking_method" in parameter_kwargs:
            ranking_method = parameter_kwargs["ranking_method"]
            parameters_need_filtering = True
        else:
            ranking_method = self.__ranking_method

        if parameters_need_filtering:
            parameter_kwargs = {
                k: v for k, v in parameter_kwargs.items() if k not in ("objective_sense", "ranking_method")
            }

        if self.__function is None:
            samples = args[0]
            fitnesses = args[1]
            num_solutions = None
            parameter_args = args[2:]
        else:
            samples = None
            fitnesses = None
            num_solutions = args[0]
            parameter_args = args[1:]

        num_parameter_args = len(parameter_args)
        num_parameter_kwargs = len(parameter_kwargs)
        if (num_parameter_args == 0) and (num_parameter_kwargs == self.__num_required_parameters):
            parameters = [None] * self.__num_required_parameters
            for parameter_name, parameter_value in parameter_kwargs.items():
                parameter_pos = self.__required_param_pos[parameter_name]
                parameters[parameter_pos] = parameter_value
        elif (num_parameter_args == self.__num_required_parameters) and (num_parameter_kwargs == 0):
            parameters = parameter_args
        elif (num_parameter_args == 0) and (num_parameter_kwargs == 0):
            raise TypeError("Missing parameter arguments")
        elif (num_parameter_args > 0) and (num_parameter_kwargs > 0):
            raise TypeError(
                "Specifying some of the distribution parameters via positional arguments and some others"
                " via keyword arguments is not supported."
                " Please provide the distribution parameters only via positional arguments"
                " or only via keyword arguments."
            )
        else:
            raise TypeError("Invalid number of arguments")

        if self.__function is None:
            return self.__grad_batch(objective_sense, ranking_method, samples, fitnesses, *parameters)
        else:
            return self.__grad_batch(objective_sense, ranking_method, num_solutions, *parameters)


def make_functional_grad_estimator(
    distribution_class: Type,
    *,
    required_parameters: Iterable,
    function: Optional[Callable] = None,
    objective_sense: Optional[str] = None,
    fixed_parameters: Optional[dict] = None,
    ranking_method: Optional[str] = None,
    return_samples: bool = False,
    return_fitnesses: bool = False,
) -> Callable:
    """
    Make a stateless gradient estimator function.

    The returned function estimates gradients for the parameters of the
    specified distribution, either with the help of a fitness function,
    or with the help of a pair of tensors representing the samples
    (or solutions) and their associated fitnesses.

    **Usage 1: with the help of a fitness function.**
    Let us assume that we have a fitness function `f`, which receives a
    matrix (i.e. 2-dimensional tensor) and returns a vector (i.e. a
    1-dimensional tensor), where each the i-th row of the matrix represents
    the i-th solution, and i-th element of the returned vector represents
    the fitness of the i-th solution.
    A functional gradient estimator for this function can be created like
    this:

    ```python
    from evotorch.distributions import (
        SymmetricSeparableGaussian,
        make_functional_grad_estimator,
    )


    def f(x: torch.Tensor) -> torch.Tensor: ...


    fgrad = make_functional_grad_estimator(
        # The gradient estimator will use this distribution:
        SymmetricSeparableGaussian,
        # The gradient estimator will be bound to this fitness function:
        function=f,
        # We want to maximize the fitnesses returned by `f`
        # (use "min" for minimizing them)
        objective_sense="max",
        # The distribution parameters "mu" and "sigma" are to be passed
        # as arguments every time we call it as a function.
        required_parameters=["mu", "sigma"],
        # The fitnesses will be ranked according to this method:
        ranking_method="centered",  # the default is "raw"
    )
    ```

    Now that we have our gradient estimator `fgrad`, we can use it as a
    function:

    ```python
    current_mu = ...  # mu parameter vector
    current_sigma = ...  # sigma parameter vector
    num_samples = ...  # number of samples (temporary population size)

    gradients = fgrad(num_samples, current_mu, current_sigma)
    # or, alternatively:
    # gradients = fgrad(num_samples, mu=current_mu, sigma=current_sigma)

    # At this point, we have our `gradients`, which is in the form of a
    # dictionary. Gradients for the parameters mu and sigma can be obtained
    # from this dictionary like this:
    grad_for_mu = gradients["mu"]
    grad_for_sigma = gradients["sigma"]
    ```

    **Usage 2: without an explicit fitness function.**
    Let us imagine a scenario where the procedure of computing the fitnesses
    is not so straightforward and therefore it is not possible to wrap it
    within a single fitness function. For such cases, we can create and use a
    gradient estimator that is not bound to any such fitness function:

    ```python
    from evotorch.distributions import (
        SymmetricSeparableGaussian,
        make_functional_sampler,
        make_functional_grad_estimator,
    )

    estimate_grads = make_functional_grad_estimator(
        # The gradient estimator will use this distribution:
        SymmetricSeparableGaussian,
        # We want to maximize the fitnesses (use "min" for minimizing them)
        objective_sense="max",
        # The distribution parameters "mu" and "sigma" are to be passed
        # as arguments every time we call it as a function.
        required_parameters=["mu", "sigma"],
        # The fitnesses will be ranked according to this method:
        ranking_method="centered",  # the default is "raw"
    )
    ```

    Note that without being bound to any fitness function, `estimate_grad`
    will ask us to provide the samples and the fitnesses. A practical way
    of obtaining such samples is to have a functional sampler:

    ```python
    get_samples = make_functional_sampler(
        SymmetricSeparableGaussian,
        required_parameters=["mu", "sigma"],
    )
    ```

    Now we are ready to use our sampler and our estimator:

    ```python
    current_mu = ...  # mu parameter vector
    current_sigma = ...  # sigma parameter vector
    num_samples = ...  # number of samples (temporary population size)

    samples = get_samples(num_samples, current_mu, current_sigma)
    # or, alternatively:
    # samples = get_samples(num_samples, mu=current_mu, sigma=current_sigma)

    fitnesses = ...  # code to compute fitnesses from the samples goes here

    gradients = estimate_grads(samples, fitnesses, current_mu, current_sigma)
    # or, alternatively:
    # gradients = estimate_grads(
    #     samples, fitnesses, mu=current_mu, sigma=current_sigma
    # )

    # At this point, we have our `gradients`, which is in the form of a
    # dictionary. Gradients for the parameters mu and sigma can be obtained
    # from this dictionary like this:
    grad_for_mu = gradients["mu"]
    grad_for_sigma = gradients["sigma"]
    ```

    **Batched gradient estimation.**
    The function returned by `make_functional_grad_estimator` is compatible
    with `vmap`. If the estimator is bound to a specific fitness function,
    that fitness function should also be compatible with `vmap`. If the
    fitness function is not `vmap`-compatible, or if its behavior is
    unexpected in the presence of `vmap`, then, consider instantiating
    the gradient estimator without binding it to a fitness function.

    As an alternative to `vmap`, a functional sampler has built-in support for
    batched sampling (which actually still uses `vmap` internally).
    Let us again consider our example estimator, `estimate_grads`.
    In this example, the parameters `current_mu` and `current_sigma` would be
    1-dimensional tensors in the non-batched case (because the distribution
    `SymmetricSeparableGaussian` expects the parameters `mu` and `sigma` as
    1-dimensional tensors, as can be observed from the class attribute
    `SymmetricSeparableGaussian.PARAMETER_NDIMS`).
    If `estimate_grads` is given `mu` and/or `sigma` with more than 1
    dimensions, those extra leftmost dimensions will be considered as batch
    dimensions, and therefore the resulting sample tensor will have extra
    leftmost dimensions too.

    **Declaring fixed parameters.**
    A functional gradient estimator can be created in such a way that some of
    the parameters are pre-defined and only a subset of the mandatory parameters
    are expected via arguments. For example, a gradient estimator that samples
    from `SymmetricSeparableGaussian` with a fixed sigma can be defined like
    this:

    ```python
    predefined_sigma = ...  # The constant sigma goes here

    fgrad2 = make_functional_sampler(
        SymmetricSeparableGaussian,
        function=f,
        objective_sense="max",
        required_parameters=["mu"],
        fixed_parameters={"sigma": predefined_sigma},
        ranking_method="centered",
    )
    ```

    The function `fgrad2` can be called like this:

    ```python
    gradients = fgrad2(num_samples, current_mu)
    # or, alternatively:
    # gradients = fgrad2(num_samples, mu=current_mu)
    ```

    **Specifying `objective_sense` and/or `ranking_method` later.**
    One can omit `objective_sense` and `ranking_method` while making the
    functional gradient estimator, and specify them later at the moment of
    estimation. For example:

    ```python
    fgrad3 = make_functional_sampler(
        SymmetricSeparableGaussian,
        function=f,
        required_parameters=["mu", "sigma"],
        # Notice: `objective_sense` and `ranking_method` are omitted
    )

    ...

    mu = ...
    sigma = ...

    gradients = fgrad3(
        num_samples,
        mu=mu,
        sigma=sigma,
        objective_sense="max",
        ranking_method="centered",
    )
    ```

    Args:
        function: The fitness function that will be called for estimating
            the gradients. If provided, the first positional argument of the
            returned gradient estimator will be the number of solutions.
            If omitted, the first and second positional arguments of the
            returned gradient estimator will be the samples (solutions)
            and fitnesses.
            Please note that this `function` is expected to receive a
            2-dimensional tensor (representing the population, where each
            row of the 2-dimensional tensor is a solution) and is expected
            to return a 1-dimensional tensor (where each element is a
            scalar fitness).
            For batching and/or `vmap` to work, this `function` itself should
            be `vmap`-compatible.
        objective_sense: Specify this as "max" if a higher fitness value means
            better solution. Specify this as "min" if a higher fitness value
            means worse solution. Please note that, if `objective_sense` is not
            provided at the moment of its making, one will have to specify it
            later every time the estimator is called.
        required_parameters: A list of strings, each string being the name
            of a distribution parameter. The order of this list determines
            the order of parameter-related positional arguments in the
            returned callable object.
        fixed_parameters: A dictionary where the keys are parameter names
            (as strings), and the values are pre-defined parameter values.
        ranking_method: Give a string here if you would like the fitnesses
            to be ranked first. Possible values are "centered", "linear",
            "raw".
        return_samples: Set this as True if you would like the gradient
            estimator to return not just the gradients, but also the samples
            (solutions) that were used for estimating the gradients.
        return_fitnesses: Set this as True if you would like the gradient
            estimator to return not just the gradients, but also the fitnesses
            that were used for estimating the gradients.
    Returns:
        A callable object whose function is to estimate gradients.
    """
    return FunctionalGradEstimator(
        distribution_class,
        function=function,
        objective_sense=objective_sense,
        required_parameters=required_parameters,
        fixed_parameters=fixed_parameters,
        ranking_method=ranking_method,
        return_samples=return_samples,
        return_fitnesses=return_fitnesses,
    )
