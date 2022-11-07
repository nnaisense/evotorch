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


from typing import Optional

import numpy as np
import torch

from evotorch.bbo.bbob_utilities import lambda_alpha, random_binary_vector, random_orthogonal_matrix, standardized_range
from evotorch.core import BoundsPairLike, Device, DType, Problem, SolutionBatch


class BBOBProblem(Problem):
    def __init__(
        self,
        solution_length: int,
        initial_bounds: Optional[BoundsPairLike] = (-5, 5),
        bounds: Optional[BoundsPairLike] = None,
        dtype: Optional[DType] = torch.float64,
        eval_dtype: Optional[DType] = torch.float64,
        device: Optional[Device] = None,
        seed: Optional[int] = None,
    ):

        super().__init__(
            objective_sense="min",
            objective_func=None,
            initial_bounds=initial_bounds,
            bounds=bounds,
            solution_length=solution_length,
            dtype=dtype,
            eval_dtype=eval_dtype,
            device=device,
            eval_data_length=None,
            seed=seed,
            num_actors=None,
            actor_config=None,
            num_gpus_per_actor=None,
            num_subbatches=None,
            subbatch_size=None,
            store_solution_stats=None,
            vectorized=True,
        )

        # Initialize meta variables
        self.initialize_meta_variables()
        self._log_closest = 1e6

    """ Extra BBOB-specific generator functions that ensure compliant dtype, device and generator
    """

    def make_f_opt(self) -> torch.Tensor:
        """Generate fitness shift f_opt. f_opt is sampled from the Cauchy distribution with location 0 and scale 100.
        This means that the median distance from 0 is 100. The value is then clamped to the range [-1000, 1000] and rounded to 2 decimal places.
        Cauchy generation is done on the same device, dtype and generator by doing uniform sampling followed by inverse transform sampling.
        """
        # Generate uniformly from [0,1] to sample in the CDF
        cdf_f_opt = self.make_uniform(1)
        # CDF(x) is 1/pi arctan((x - loc) / scale) + 1/2
        # Thus x is loc + scale tan(1/2 (2 CDF(x) - 1) pi)
        f_opt_sampled_from_cauchy = 100 * torch.tan(0.5 * (2 * cdf_f_opt - 1) * np.pi)
        # Clamp to range [-1000, 1000]
        f_opt_clamped = torch.clamp(f_opt_sampled_from_cauchy, -1000, 1000)
        # Round to 2 decimal places
        f_opt_rounded = torch.round(f_opt_clamped, decimals=2)
        return f_opt_rounded

    def make_x_opt(self) -> torch.Tensor:
        """Make the optimal point x_opt. By default, this is drawn from the uniform distribution U[-4, 4]^d"""
        return 8 * (self.make_uniform(self.solution_length) - 0.5)

    def make_lambda_alpha(self, alpha: float, diagonal_only: bool = True) -> torch.Tensor:
        """Make the Lambda^alpha variable for a given alpha
        Args:
            alpha (float): The alpha parameter to the matrix.
            diagonal_only (bool): Whether to only return the diagonal elements.
        """
        return lambda_alpha(
            alpha=alpha,
            dimension=self.solution_length,
            diagonal_only=diagonal_only,
            dtype=self.dtype,
            device=self.device,
        )

    def make_random_binary_vector(self) -> torch.Tensor:
        """Make random binary vector 1^+_- where each element is either -1 or 1 with probability 0.5"""
        return random_binary_vector(
            dimension=self.solution_length, dtype=self.dtype, device=self.device, generator=self.generator
        )

    def make_standardized_range(self) -> torch.Tensor:
        """Make the standardized range (i-1)/(D-1) for i = 1 ... D"""
        return standardized_range(dimension=self.solution_length, dtype=self.dtype, device=self.device)

    def make_random_orthogonal_matrix(self) -> torch.Tensor:
        """Make a random orthogonal matrix ("R" and "Q") using the Gram-Schmidt orthonormalization."""
        return random_orthogonal_matrix(
            dimension=self.solution_length,
            dtype=self.dtype,
            device=self.device,
            generator=self.generator,
        )

    """ Functionality for specifying which of the above maker functions need to be called to specify the problem
    """

    def _initialize_meta_variables(self):
        """Initialise meta variables. Override to define problem-specific meta-variables."""
        pass

    def initialize_meta_variables(self):
        """Initialize meta variables. Problem-specific meta variables should be instantiated by overriding _initialize_meta_variables"""
        # All function definitions require f_opt and x_opt
        self._x_opt = self.make_x_opt()
        self._f_opt = self.make_f_opt()
        # Create problem-specific meta variables
        self._initialize_meta_variables()

    """ Implementation of the objective function in vectorized form
    """

    def map_x_to_z(self, x: torch.Tensor) -> torch.Tensor:
        """The x to z mapping used in all function definitions. By default, this simply maps z = x - x_opt, but it should be overridden for other use-cases
        Args:
            x (torch.Tensor): The values x to map, of shape [num_samples, dimension]
        Returns:
            z (torch.Tensor): The mapped values z, of shape [num_samples, dimension]
        """
        # Subtract x_opt from x and return
        z = x - self._x_opt.unsqueeze(0)
        return z

    def _apply_function(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Apply the function to the given values z. Override to define problem-specific function
        Args:
            z (torch.Tensor): The values to apply the function to, of shape [num_samples, dimension]
            x (torch.Tensor): The original, untransformed, values, of shape [num_samples, dimension]
        Returns:
            f_z (torch.Tensor): The output of applying the function to z, of shape [num_samples,0]
        """
        raise NotImplementedError("Function must be defined for BBOBProblem")

    def apply_function(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Apply the function to the given values z. Note that self._apply_function should be overridden for problem-specific function
        Args:
            z (torch.Tensor): The values to apply the function to, of shape [num_samples, dimension]
            x (torch.Tensor): The original, untransformed, values, of shape [num_samples, dimension]
        Returns:
            f_z (torch.Tensor): The output of applying the function to z, of shape [num_samples,0]
        """
        # Shift result by f_opt and return
        f_z = self._apply_function(z, x) + self._f_opt
        return f_z

    @property
    def log_closest(self) -> torch.Tensor:
        """The logarithm of the best discovered solution so far"""
        return self._log_closest

    @log_closest.setter
    def log_closest(self, new_log_closest):
        self._log_closest = new_log_closest

    def _evaluate_batch(self, batch: SolutionBatch) -> None:
        # Get x from batch
        x = batch.values.clone()
        # Map x to z
        z = self.map_x_to_z(x)
        # Get f(x) from function application to z
        f_x = self.apply_function(z, x)
        # Compute log distance from f_opt
        log_f_x = torch.log(f_x - self._f_opt)
        if torch.amin(log_f_x) < self.log_closest:
            self.log_closest = torch.amin(log_f_x)
        # Assign fitnesses to batch
        batch.set_evals(f_x)
