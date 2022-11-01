""" Implementation of the Real-Parameter Black-Box Optimization Benchmarking 2009 functions
"""

import numpy as np
import torch

from evotorch.bbo import bbob_utilities
from evotorch.bbo.bbob_problem import BBOBProblem


class Sphere(BBOBProblem):
    def _apply_function(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return torch.norm(z, dim=-1).pow(2)


class SeparableEllipsoidal(BBOBProblem):
    def _initialize_meta_variables(self):
        # Initialize 10^ (6 * standardized range)
        self.power_range = torch.pow(10, 6 * self.make_standardized_range()).unsqueeze(0)

    def map_x_to_z(self, x: torch.Tensor) -> torch.Tensor:
        # T_osz(x - x_opt)
        return bbob_utilities.T_osz(x - self._x_opt.unsqueeze(0))

    def _apply_function(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return torch.sum(
            self.power_range * z.pow(2.0),
            dim=-1,
        )


class SeparableRastrigin(BBOBProblem):
    def _initialize_meta_variables(self):
        self.lambda_10 = self.make_lambda_alpha(10, diagonal_only=True).unsqueeze(0)

    def map_x_to_z(self, x: torch.Tensor) -> torch.Tensor:
        # Lambda^10 T^0.2_asy ( T_osz (x - x_opt) )
        return self.lambda_10 * bbob_utilities.T_beta_asy(
            values=bbob_utilities.T_osz(values=x - self._x_opt.unsqueeze(0)),
            beta=0.2,
        )

    def _apply_function(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return 10 * (self.solution_length - torch.sum(torch.cos(2 * np.pi * z), dim=-1)) + torch.norm(z, dim=-1).pow(
            2.0
        )


class BucheRastrigin(BBOBProblem):
    def _initialize_meta_variables(self):
        standardized_range = self.make_standardized_range()
        # Values when T_osz(x_i - x_opt_i) is positive and i is odd
        self.s_positive_odd = 10 * torch.pow(10, 0.5 * standardized_range).unsqueeze(0)
        # Values when T_osz(x_i - x_opt_i) is negative or i is even
        self.s_negative_even = torch.pow(10, 0.5 * standardized_range).unsqueeze(0)
        # Mask for even values. Note that this is for i = 1 ... D, so we actually offset by one so that the first value is counted as odd
        self.even_mask = torch.as_tensor(
            [i + 1 % 2 == 0 for i in range(self.solution_length)], dtype=torch.bool, device=self.device
        )

    def map_x_to_z(self, x: torch.Tensor) -> torch.Tensor:
        # s T_osz (x - x_opt) where s_i = s_positive_odd when T_osz(x_i - x_opt_i) is positive and i is odd, and s_i = s_negative_even otherwise
        pre_z = bbob_utilities.T_osz(x - self._x_opt.unsqueeze(0))
        # Branching on whether pre_z > 0
        s_values = torch.where(pre_z > 0, self.s_positive_odd, self.s_negative_even)
        # Always s_negative_even when the value has an even index
        s_values[:, self.even_mask] = self.s_negative_even[:, self.even_mask]
        # s * pre_z
        return s_values * pre_z

    def _apply_function(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return (
            10 * (self.solution_length - torch.sum(torch.cos(2 * np.pi * z), dim=-1))
            + torch.norm(z, dim=-1).pow(2.0)
            + 100 * bbob_utilities.f_pen(x)
        )


class LinearSlope(BBOBProblem):
    def make_x_opt(self) -> torch.Tensor:
        # Linear slope has special global optimum at 5 * +/- 1
        return 5 * self.make_random_binary_vector()

    def _initialize_meta_variables(self):
        # Initialize 10^ (standardized range)
        power_range = torch.pow(10, self.make_standardized_range())
        self.s = (torch.sign(self._x_opt) * power_range).unsqueeze(0)

    def map_x_to_z(self, x: torch.Tensor) -> torch.Tensor:
        # x unless x_i x_opt_i > 5^2, in which case x_opt_i
        return torch.where(
            x * self._x_opt.unsqueeze(0) < 25,
            x,
            self._x_opt.unsqueeze(0),
        )

    def _apply_function(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return torch.sum(
            5 * torch.abs(self.s) - self.s * z,
            dim=-1,
        )


class AttractiveSector(BBOBProblem):
    def _initialize_meta_variables(self):
        self.Q = self.make_random_orthogonal_matrix()
        self.R = self.make_random_orthogonal_matrix()
        self.lambda_10 = self.make_lambda_alpha(10.0, diagonal_only=True).unsqueeze(0)

    def map_x_to_z(self, x: torch.Tensor) -> torch.Tensor:
        # Q lambda^10 R (x - x_opt)
        return bbob_utilities.apply_orthogonal_matrix(
            self.lambda_10 * bbob_utilities.apply_orthogonal_matrix(x - self._x_opt.unsqueeze(0), self.R),
            self.Q,
        )

    def _apply_function(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        s = torch.where(z * self._x_opt.unsqueeze(0) > 0, 100, 1)
        return bbob_utilities.T_osz(
            torch.sum(
                (s * z).pow(2.0),
                dim=-1,
            )
        ).pow(0.9)


class StepEllipsoidal(BBOBProblem):
    def _initialize_meta_variables(self):
        self.Q = self.make_random_orthogonal_matrix()
        self.R = self.make_random_orthogonal_matrix()
        self.lambda_10 = self.make_lambda_alpha(10.0, diagonal_only=True).unsqueeze(0)
        standardized_range = self.make_standardized_range()
        self.weighted_norm_coeffs = torch.pow(10, 2 * standardized_range).unsqueeze(0)

    def map_x_to_z(self, x: torch.Tensor) -> torch.Tensor:
        # R lambda^10 (x - x_opt)
        z_hat = self.lambda_10 * bbob_utilities.apply_orthogonal_matrix(x - self._x_opt.unsqueeze(0), self.R)
        z_bar = torch.where(
            torch.abs(z_hat) > 0.5,
            bbob_utilities.nearest_integer(0.5 + z_hat),
            bbob_utilities.nearest_integer(0.5 + 10 * z_hat) / 10,
        )
        # Q z bar
        return bbob_utilities.apply_orthogonal_matrix(z_bar, self.Q)

    def _apply_function(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # Absolute value of zhat_1 / 10^4. Note that here zhat_1 refers to element at index 0
        zhat_1_div_104 = torch.abs(z[:, 0]) / (10**4)
        # Sum of weighted norm
        weighted_norm = torch.sum(self.weighted_norm_coeffs * z.pow(2.0), dim=-1)
        # Branching value gives max
        f_base = 0.1 * torch.where(weighted_norm > zhat_1_div_104, weighted_norm, zhat_1_div_104)
        return f_base + bbob_utilities.f_pen(x)


class RosenbrockOriginal(BBOBProblem):
    def make_x_opt(self) -> torch.Tensor:
        # Linear slope has special global optimum from U(-3, 3)
        return 6 * (self.make_uniform(self.solution_length) - 0.5)

    def _initialize_meta_variables(self):
        self.z_coeff = max(1, np.sqrt(self.solution_length) / 8)

    def map_x_to_z(self, x: torch.Tensor) -> torch.Tensor:
        # max(1, sqrt(d)/8) (x - x_opt) + 1
        return self.z_coeff * (x - self._x_opt.unsqueeze(0)) + 1

    def _apply_function(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        z_starts_at_0 = z[:, : self.solution_length - 1]
        z_starts_at_1 = z[:, 1:]
        return torch.sum(100 * (z_starts_at_0.pow(2.0) - z_starts_at_1).pow(2.0) + (z_starts_at_0 - 1).pow(2.0), dim=-1)


class RosenbrockRotated(BBOBProblem):
    def make_x_opt(self) -> torch.Tensor:
        # Linear slope has special global optimum R^T (ones) / (2 z_coeff)
        return (
            bbob_utilities.apply_orthogonal_matrix(self.make_ones(self.solution_length).unsqueeze(0), self.R.T)
            / (2 * self.z_coeff)
        )[0]

    def initialize_meta_variables(self):
        # x_opt must set manually for this task (note that this is hidden in the source code of COCO)
        # see: https://github.com/numbbo/coco/blob/master/code-experiments/src/f_rosenbrock.c#L157
        # so we actually override initialize_meta_variables, rather than _initialize_meta_variables, so that x_opt can be set after R is initialized
        self.z_coeff = max(1, np.sqrt(self.solution_length) / 8)
        self.R = self.make_random_orthogonal_matrix()
        self._x_opt = self.make_x_opt()
        self._f_opt = self.make_f_opt()

    def map_x_to_z(self, x: torch.Tensor) -> torch.Tensor:
        # max(1, sqrt(d)/8) R x + 1/2
        return 1 / 2 + self.z_coeff * bbob_utilities.apply_orthogonal_matrix(x, self.R)

    def _apply_function(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        z_starts_at_0 = z[:, : self.solution_length - 1]
        z_starts_at_1 = z[:, 1:]
        return torch.sum(100 * (z_starts_at_0.pow(2.0) - z_starts_at_1).pow(2.0) + (z_starts_at_0 - 1).pow(2.0), dim=-1)


class HighConditioningEllipsoidal(BBOBProblem):
    def _initialize_meta_variables(self):
        # Initialize 10^ (6 * standardized range)
        self.power_range = torch.pow(10, 6 * self.make_standardized_range()).unsqueeze(0)
        self.R = self.make_random_orthogonal_matrix()

    def map_x_to_z(self, x: torch.Tensor) -> torch.Tensor:
        # T_osz(R(x - x_opt))
        return bbob_utilities.T_osz(bbob_utilities.apply_orthogonal_matrix(x - self._x_opt.unsqueeze(0), self.R))

    def _apply_function(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return torch.sum(
            self.power_range * z.pow(2.0),
            dim=-1,
        )


class Discus(BBOBProblem):
    def _initialize_meta_variables(self):
        self.R = self.make_random_orthogonal_matrix()

    def map_x_to_z(self, x: torch.Tensor) -> torch.Tensor:
        # T_osz(R(x - x_opt))
        return bbob_utilities.T_osz(bbob_utilities.apply_orthogonal_matrix(x - self._x_opt.unsqueeze(0), self.R))

    def _apply_function(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        z_starts_at_1 = z[:, 1:]
        return 1e6 * z[:, 0].pow(2.0) + torch.sum(z_starts_at_1.pow(2.0), dim=-1)


class BentCigar(BBOBProblem):
    def _initialize_meta_variables(self):
        self.R = self.make_random_orthogonal_matrix()

    def map_x_to_z(self, x: torch.Tensor) -> torch.Tensor:
        # R T_asy^0.5 (R(x - x_opt))
        return bbob_utilities.apply_orthogonal_matrix(
            bbob_utilities.T_beta_asy(
                bbob_utilities.apply_orthogonal_matrix(x - self._x_opt.unsqueeze(0), self.R), beta=0.5
            ),
            self.R,
        )

    def _apply_function(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        z_starts_at_1 = z[:, 1:]
        return z[:, 0].pow(2.0) + 1e6 * torch.sum(z_starts_at_1.pow(2.0), dim=-1)


class SharpRidge(BBOBProblem):
    def _initialize_meta_variables(self):
        self.R = self.make_random_orthogonal_matrix()
        self.Q = self.make_random_orthogonal_matrix()
        self.lambda_10 = self.make_lambda_alpha(10.0, diagonal_only=True).unsqueeze(0)

    def map_x_to_z(self, x: torch.Tensor) -> torch.Tensor:
        # Q Lambda^10 R (x - x_opt)
        return bbob_utilities.apply_orthogonal_matrix(
            self.lambda_10 * bbob_utilities.apply_orthogonal_matrix(x - self._x_opt.unsqueeze(0), self.R),
            self.Q,
        )

    def _apply_function(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        z_starts_at_1 = z[:, 1:]
        return z[:, 0].pow(2.0) + 100 * torch.sum(z_starts_at_1.pow(2.0), dim=-1).pow(0.5)


class DifferentPowers(BBOBProblem):
    def _initialize_meta_variables(self):
        self.R = self.make_random_orthogonal_matrix()
        self.power_range = (2 + 4 * self.make_standardized_range()).unsqueeze(0)

    def map_x_to_z(self, x: torch.Tensor) -> torch.Tensor:
        # R (x - x_opt)
        return bbob_utilities.apply_orthogonal_matrix(x - self._x_opt.unsqueeze(0), self.R)

    def _apply_function(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(torch.sum(torch.abs(z).pow(self.power_range), dim=-1))


class NonSeparableRastrigin(BBOBProblem):
    def _initialize_meta_variables(self):
        self.lambda_10 = self.make_lambda_alpha(10, diagonal_only=True).unsqueeze(0)
        self.Q = self.make_random_orthogonal_matrix()
        self.R = self.make_random_orthogonal_matrix()

    def map_x_to_z(self, x: torch.Tensor) -> torch.Tensor:
        # R Lambda^10 Q T^0.2_asy ( T_osz (R(x - x_opt)) )
        return bbob_utilities.apply_orthogonal_matrix(
            self.lambda_10
            * bbob_utilities.apply_orthogonal_matrix(
                bbob_utilities.T_beta_asy(
                    values=bbob_utilities.T_osz(
                        values=bbob_utilities.apply_orthogonal_matrix(
                            x - self._x_opt.unsqueeze(0),
                            self.R,
                        )
                    ),
                    beta=0.2,
                ),
                self.Q,
            ),
            self.R,
        )

    def _apply_function(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return 10 * (self.solution_length - torch.sum(torch.cos(2 * np.pi * z), dim=-1)) + torch.norm(z, dim=-1).pow(
            2.0
        )


# Array of functions in ordered form e.g. so that they can be accessed like 'F1' rather than by name
_functions = [
    Sphere,
    SeparableEllipsoidal,
    SeparableRastrigin,
    BucheRastrigin,
    LinearSlope,
    AttractiveSector,
    StepEllipsoidal,
    RosenbrockOriginal,
    RosenbrockRotated,
    HighConditioningEllipsoidal,
    Discus,
    BentCigar,
    SharpRidge,
    DifferentPowers,
    NonSeparableRastrigin,
]


def get_function_i(i: int) -> BBOBProblem:
    """Get the ith function, for i in 1 ... 24
    Args:
        i (int): The index of the function to obtain, between 1 and 24
    Returns:
        function_i (BBOBProblem): The ith function Fi
    """
    if i < 1 or i > 24:
        raise ValueError("The BBOB Noiseless suite defines only functions F1 ... F24")
    function_i = _functions[i - 1]
    return function_i


if __name__ == "__main__":

    for i in range(len(_functions)):
        func = get_function_i(i + 1)
        print("Function", func)
        obj = func(10)
        batch = obj.generate_batch(5)
        batch[0].set_values(obj._x_opt)
        print(batch)
        obj.evaluate(batch)
        print(batch.evals - obj._f_opt)
