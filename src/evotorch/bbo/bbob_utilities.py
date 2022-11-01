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

import torch

from evotorch.tools import Device, DType

""" Various utility functionality for implementing BBOB functions. See section 0.2 of
    Hansen, Nikolaus, et al. "Real-Parameter Black-Box Optimization Benchmarking 2009: Noiseless Functions Definitions."
"""


def nearest_integer(values: torch.Tensor) -> torch.Tensor:
    """Alias for rounding to nearest whole integer. Included for unambiguous implementation of BBOB benchmark functions
    Args:
        values (torch.Tensor): The values to round to the nearest integer.
    Returns:
        torch.Tensor: The values rounded to the nearest integer.
    """
    return torch.round(values)


def standardized_range(
    dimension: int, dtype: DType = torch.float32, device: Device = torch.device("cpu")
) -> torch.Tensor:
    """The commonly used standardized range (i-1)/(D-1) for i = 1 ... D
    Args:
        dimension (int): The dimension D of the range.
        dtype (Dtype): The datatype of the generated vector.
        device (Device): The device of the generated vector.
    Returns:
        i_values (torch.Tensor): The generated standardized range.
    """
    # Note that the paper states (i - 1) for i = 1 ... D but as arange starts at zero, we can drop the constant
    i_values = (torch.arange(dimension, dtype=dtype, device=device)) / (dimension - 1)
    return i_values


def lambda_alpha(
    alpha: float,
    dimension: int,
    diagonal_only: bool = True,
    dtype: DType = torch.float32,
    device: Device = torch.device("cpu"),
) -> torch.Tensor:
    """The Lambda^alpha matrix, which is diagonal only with Lamba^alpha_i,i = alpha^(1/2 (i-1)/(dimension - 1))
    Args:
        alpha (float): The alpha parameter to the matrix.
        dimension (int): The dimension of the matrix.
        diagonal_only (bool): Whether to only return the diagonal elements.
        dtype (Dtype): The datatype of the generated matrix.
        device (Device): The device of the generated matrix.
    Returns:
        alpha_matrix: The parameterised matrix Lambda^alpha.
            If diagonal_only, then it is the diagonal elements of shape [dimension,]
            Otherwise, then it is the matrix of shape [dimension, dimension,]
    """
    # Place alpha in a tensor so it can be used within torch.pow
    alpha_tensor = torch.as_tensor(alpha, dtype=dtype, device=device)
    # Exponents of diagonal terms of Lambda^alpha.
    exponents = 0.5 * standardized_range(dimension, dtype=dtype, device=device)
    # Diagonal elements of Lambda^alpha
    alpha_diagonal = torch.pow(alpha_tensor, exponents)
    # Branching on whether diagonal_only
    if diagonal_only:
        alpha_matrix = alpha_diagonal
    else:
        alpha_matrix = torch.diag(alpha_diagonal)
    return alpha_matrix


def f_pen(values: torch.Tensor) -> torch.Tensor:
    """The f_pen magnitude penalty function, in vectorized form.
    For a given sample x, the penalty is the sum of the element-wise penalty.
    The element-wise penalty for element x_i is max(0, |x_i| - 5)^2
    Args:
        values (torch.Tensor): The values to apply f_pen to, of shape [num_samples, dimension,]
    Returns:
        penalties (torch.Tensor): The penalised values, of shape [num_samples,]
    """
    # Compute element-wise penalty
    elementwise_penalty = torch.clamp(
        torch.abs(values) - 5,  # Penalty applies whenever the absolute of the value is less than 5.
        min=0.0,  # Minimum penalty is zero, cannot be based on a negative value (implies value within [-5,5])
        max=None,  # No upper bound on penalty
    ).pow(2.0)

    # Sum across the individual samples
    penalties = torch.sum(elementwise_penalty, dim=-1)
    return penalties


def random_binary_vector(
    dimension: int,
    dtype: DType = torch.float32,
    device: Device = torch.device("cpu"),
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """The random binary vector 1^+_- where each element is either -1 or 1 with probability 0.5
    Args:
        dimension (int): The dimension of the random binary vector.
        dtype (Dtype): The datatype of the generated vector.
        device (Device): The device of the generated vector.
        generator (Optional[torch.Generator]): An optional generator for the randomised values.
    Returns:
        random_vec (torch.Tensor): The generated random binary vector
    """
    # Sample the uniform distribution [0,1] in the given dimension
    uniform_noise = torch.rand(dimension, dtype=dtype, device=device, generator=generator)
    # Round the noise to give a uniform distribution over 0/1 values, and rescale to -1/1
    random_vec = 2 * (torch.round(uniform_noise) - 0.5)
    return random_vec


def _gram_schmidt_projection(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """The Gram-Scmidt projection operator u <u, v> / <u u>
    Args:
        u, v (torch.Tensor): The vector arguments to the projection, of shapes [dimension,]
    Returns:
        projection (torch.Tensor): The projected vector of shape [dimension,]
    """
    # Compute dot products
    u_dot_v = torch.dot(u, v)
    u_dot_u = torch.dot(u, u)
    # Construct projection
    projection = u * u_dot_v / u_dot_u
    return projection


def random_orthogonal_matrix(
    dimension: int,
    dtype: DType = torch.float32,
    device: Device = torch.device("cpu"),
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Generate a random orthogonal matrix ("R" and "Q") using the Gram-Schmidt orthonormalization.
    Note that this process uses the notation found on Wikipedia https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process
    Args:
        dimension (int): The dimension of the random orthogonal matrix.
        dtype (Dtype): The datatype of the generated orthogonal matrix.
        device (Device): The device of the generated orthogonal matrix.
        generator (Optional[torch.Generator]): An optional generator for the randomised values.
    Returns:
        orthogonal_matrix (torch.Tensor): The generated random orthogonal matrix.
    """
    # Generate normally distributed vectors
    vs = torch.randn(
        (
            dimension,
            dimension,
        ),
        dtype=dtype,
        device=device,
        generator=generator,
    )

    # Compute u vectors -- first u is simply first v
    us = [vs[0]]
    for u_index in range(1, dimension):
        v = vs[u_index]
        projection_sum = sum([_gram_schmidt_projection(u, v) for u in us])
        u = v - projection_sum
        us.append(u)

    # Convert u vectors to e vectors
    es = [u / torch.norm(u) for u in us]

    # Convert back into full tensor form
    orthogonal_matrix = torch.stack(es, dim=0)

    return orthogonal_matrix


def T_beta_asy(values: torch.Tensor, beta: float) -> torch.Tensor:
    """The T^beta_asy function
    Args:
        values (torch.Tensor): The values to apply the T^beta_asy function to, of shape [num_samples, dimension,]
        beta (float): The beta parameter of the function
    Returns:
        transformed_values (torch.Tensor): The transformed values of shape [num_samples, dimension,]
    """
    # Get the dimension
    dimension = values.shape[-1]
    # Exponents of values when values are positive.
    exponents = 1 + beta * torch.sqrt(torch.abs(values)) * standardized_range(
        dimension=dimension, dtype=values.dtype, device=values.device
    ).unsqueeze(0)

    # Branching on whether the values are positive... when <= 0, values are instead left as passed
    transformed_values = torch.where(values > 0, torch.pow(values, exponents), values)

    return transformed_values


def T_osz(values: torch.Tensor, epsilon: float = 1e-7) -> torch.Tensor:
    """The T_osz function
    Args:
        values (torch.Tensor): The vaues to apply the T_osz function to, of shape [num_samples, dimension,]
        epsilon (float): Error threshold for assuming a value is zero. The paper states that xhat and sign(x) have specific behavior at x = 0
                Here, we assume that when |x| < epsilon, that rule should apply.
    Returns:
        transformed_values (torch.Tensor): The transformed values of shape [num_samples, dimension,]
    """
    # Identify the values to treat as zero
    values_close_to_zero = torch.abs(values) < epsilon
    # Precompute x hat, sign x, c1 and c2

    # xhat is log(|x|) unless x is 0, in which case xhat is also 0
    xhat = torch.log(torch.abs(values))
    xhat[values_close_to_zero] = 0.0

    # sign(x) is -1 if x < 0, 1 if x > 0 and 0 if x = 0
    sign_x = torch.sign(values)
    sign_x[values_close_to_zero] = 0

    # c1 is 10 if x > 0, 5.5 otherwise
    c1 = torch.where(values > 0, 10, 5.5)

    # c2 is 7.9 if x > 0, 3.1 otherwise
    c2 = torch.where(values > 0, 7.9, 3.1)

    # Construct the transformed values
    transformed_values = sign_x * torch.exp(xhat + 0.049 * (torch.sin(c1 * xhat) + torch.sin(c2 * xhat)))

    return transformed_values


def apply_orthogonal_matrix(values: torch.Tensor, orthogonal_matrix: torch.Tensor) -> torch.Tensor:
    """Apply a given orthogonal matrix R to a given batch of vectors x e.g. Rx
    Args:
        values (torch.Tensor): The batch of values to apply the orthogonal matrix to, of shape [num_solutions, dimension]
        orthogonal_matrix (torch.Tensor): The orthogonal matrix to apply to the values, of shape [dimension, dimension]
    Returns:
        transformed_values (torch.Tensor):
    """
    return torch.matmul(orthogonal_matrix, values.T).T


if __name__ == "__main__":

    x1 = torch.tensor([[0.0, 0.0], [0.0, 7.0]])
    print(f_pen(x1))

    for x in x1:
        penalty = 0.0
        for v in x:
            c1 = v - 5
            c2 = -5 - v
            if c1 > 0.0:
                penalty += c1 * c1
            elif c2 > 0:
                penalty += c2 * c2
        print(x, "->", penalty)
    print(torch.clamp(torch.abs(x1) - 5, min=0.0, max=None))
    # import matplotlib.pyplot as plt
    # import numpy as np

    # d = 10
    # v = torch.zeros((2, d))
    # betas = [0.1, 0.2, 0.5]

    # lims = [6, 1, 0.1, 0.01]

    # print(random_binary_vector(d))

    # for lim in lims:
    #     lim_low = -lim
    #     lim_high = lim
    #     xs = np.linspace(lim_low, lim_high, 1000)
    #     ys_osy = []
    #     ys_beta = {str(beta): [] for beta in betas}
    #     for x in xs:
    #         v[0, -1] = x
    #         ozy = T_osz(v)[0,-1]
    #         ys_osy.append(ozy)
    #         for beta in betas:
    #             asy = T_beta_asy(v, beta)[0,-1]
    #             ys_beta[str(beta)].append(asy)
    #     plt.plot(xs, ys_osy)
    #     for beta in betas:
    #         plt.plot(xs, ys_beta[str(beta)])
    #     plt.show()
