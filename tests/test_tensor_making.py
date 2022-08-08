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

from itertools import product
from typing import Iterable, Optional, Union

import numpy as np
import pytest
import torch

from evotorch import Problem, testing, tools
from evotorch.tools import DType

NON_EMPTY_SHAPES = [(40,), (50, 24), (30, 70, 50)]
SHAPES = [tuple()] + NON_EMPTY_SHAPES
FLOAT_DTYPES = ["float16", "float32", "float64"]
INT_DTYPES = ["int32", "int64"]
DTYPES = FLOAT_DTYPES + INT_DTYPES


FLOAT_TOLERANCE = 1e-4


def _all_must_be_close(x: torch.Tensor, y: torch.Tensor):
    is_float = "float" in str(x.dtype)
    if x.dtype != y.dtype:
        raise testing.TestingError(f"`x` and `y` have different dtypes: {x.dtype}, {y.dtype}")
    if is_float:
        testing.assert_allclose(x, y, atol=FLOAT_TOLERANCE)
    else:
        assert bool(torch.all(x == y))


def _some_must_be_different(x: torch.Tensor, y: torch.Tensor):
    is_float = "float" in str(x.dtype)
    if x.dtype != y.dtype:
        raise testing.TestingError(f"`x` and `y` have different dtypes: {x.dtype}, {y.dtype}")
    if is_float:
        assert torch.any(
            torch.abs(x - y) > FLOAT_TOLERANCE
        ), f"These tensors were expected to be different, but they are almost the same: {x}; {y}"
    else:
        assert torch.any(x != y), f"These tensors were expected to be different, but they are the same: {x}; {y}"


def _all_must_be_between(x: torch.Tensor, lb: Union[float, int, torch.Tensor], ub: Union[float, int, torch.Tensor]):
    is_float = "float" in str(x.dtype)
    if is_float:
        testing.assert_almost_between(x, lb, ub, atol=FLOAT_TOLERANCE)
    else:
        assert bool(torch.all((x >= lb) & (x <= ub)))


def _each_must_be_equal(x: torch.Tensor, value: Union[int, float]):
    is_float = "float" in str(x.dtype)
    if is_float:
        value = float(value)
        if x.ndim == 0:
            assert abs(x - value) <= FLOAT_TOLERANCE
        else:
            testing.assert_eachclose(x, value, atol=FLOAT_TOLERANCE)
    else:
        value = int(value)
        if x.ndim == 0:
            assert x == value
        else:
            assert torch.all(x == value)


def _prepare_dtype(dtype: Union[torch.dtype, str]) -> tuple:
    dtype = tools.to_torch_dtype(dtype)
    is_float = "float" in str(dtype)
    return dtype, is_float


@pytest.mark.parametrize("shape,dtype", product(SHAPES, DTYPES))
def test_basic_makers(shape: tuple, dtype: DType):
    dtype, is_float = _prepare_dtype(dtype)

    empty = tools.make_empty(shape, dtype=dtype)
    assert empty.shape == shape
    assert empty.dtype == dtype

    ones = tools.make_ones(shape, dtype=dtype)
    assert ones.shape == shape
    assert ones.dtype == dtype
    _each_must_be_equal(ones, 1)

    zeros = tools.make_zeros(shape, dtype=dtype)
    assert zeros.shape == shape
    assert zeros.dtype == dtype
    _each_must_be_equal(zeros, 0)

    if is_float:
        nans = tools.make_nan(shape, dtype=dtype)
        assert nans.shape == shape
        assert nans.dtype == dtype
        assert torch.all(torch.isnan(nans))


@pytest.mark.parametrize("shape,dtype", product(NON_EMPTY_SHAPES, DTYPES))
def test_targeted_makers(shape: tuple, dtype: DType):
    dtype, is_float = _prepare_dtype(dtype)

    ones = torch.empty(shape, dtype=dtype)
    tools.make_ones(out=ones)
    _each_must_be_equal(ones, 1)

    zeros = torch.empty(shape, dtype=dtype)
    tools.make_zeros(out=zeros)
    _each_must_be_equal(zeros, 0)

    if is_float:
        nans = torch.empty(shape, dtype=dtype)
        tools.make_nan(out=nans)
        assert torch.all(torch.isnan(nans))


@pytest.mark.parametrize("size,dtype,using_out", product([3, 5, 10], DTYPES, [False, True]))
def test_identity_maker(size: int, dtype: DType, using_out: bool):
    dtype, _ = _prepare_dtype(dtype)

    if using_out:
        I_matrix = torch.empty((size, size), dtype=dtype)
        tools.make_I(out=I_matrix)
    else:
        I_matrix = tools.make_I(size, dtype=dtype)

    desired = torch.eye(size, dtype=dtype)
    _all_must_be_close(I_matrix, desired)


@pytest.mark.parametrize("shape,dtype,using_out", product(SHAPES, FLOAT_DTYPES, [False, True]))
def test_random_maker(shape: tuple, dtype: DType, using_out: bool):
    if using_out and (shape == tuple()):
        return

    dtype, is_float = _prepare_dtype(dtype)

    # The following are the random-sampling functions and some arbitrary configurations for them.
    funcs = [
        (tools.make_uniform, {"lb": -10, "ub": 10}),
        (tools.make_gaussian, {"center": 0, "stdev": 10}),
        (tools.make_randint, {"n": 5}),
    ]

    for func, kwargs in funcs:  # for each random-making function and its keyword arguments

        # The following is a temporary function which samples new tensors using `func`
        def sample(g: Optional[torch.Generator] = None) -> torch.Tensor:
            # Prepare the keyword arguments we will use on the function
            all_kwargs = {}
            all_kwargs.update(kwargs)

            if g is not None:
                # If a generator is provided, then we add the generator to our keyword arguments
                all_kwargs["generator"] = g

            if using_out:
                # We test the `out` keyword argument of the function
                result = torch.empty(shape, dtype=dtype)
                func(out=result, **all_kwargs)
            else:
                # We test the function's capabilities without using `out`.
                result = func(shape, dtype=dtype, **all_kwargs)
                assert result.dtype == dtype  # Make sure that dtype is correct
                assert result.shape == shape  # Make sure that shape is correct

            if ("lb" in all_kwargs) and ("ub" in all_kwargs):
                # If the "lb" and "ub" keyword arguments exist in all_kwargs, then the function we just tested was
                # `make_uniform`. Knowing this, we perform an additional test and ensure that the sample does not
                # violate the boundaries.
                _all_must_be_between(result, all_kwargs["lb"], all_kwargs["ub"])

            return result

        if (len(shape) == 0) and ((not is_float) or ("n" in kwargs)):
            # When we have an empty shape (i.e. when we are sampling a scalar) and when dtype is not float
            # (or when we are using make_randint, which we understand from the existence of "n" in kwargs),
            # then the separate samples can still end up with same values by coincidence, due to limited amount of
            # choices.
            # Therefore, the only test we do here is to just take a single sample and let the inner tests work.
            sample()
        else:
            # Make a random generator
            g = torch.Generator()

            # Set a manual seed, and take two samples
            g.manual_seed(1)
            data1 = sample(g)
            data2 = sample(g)

            # Use the same seed again, and take a sample
            g.manual_seed(1)
            data3 = sample(g)

            # data1 and data2 must be different
            _some_must_be_different(data2, data1)

            # data3 and data1 were sampled from equivalent states of a random generator
            # so, they must be equal
            _all_must_be_close(data3, data1)


def _make_problem(*, dtype: DType, eval_dtype: DType, solution_length: int):
    return Problem(
        "min",
        torch.linalg.norm,
        dtype=dtype,
        eval_dtype=eval_dtype,
        initial_bounds=(-10, 10),
        solution_length=solution_length,
    )


@pytest.mark.parametrize("shape,dtype,eval_dtype", product(SHAPES, DTYPES, FLOAT_DTYPES))
def test_tensor_makers_of_problem(shape: Optional[tuple], dtype: DType, eval_dtype: DType):
    dtype, is_float = _prepare_dtype(dtype)
    eval_dtype, _ = _prepare_dtype(eval_dtype)

    # some arbitrary configurations:
    numsln = 20
    slnlen = 7

    # Make a new Problem instance
    problem = _make_problem(dtype=dtype, eval_dtype=eval_dtype, solution_length=slnlen)

    # some more arbitrary configurations:
    center = 5.0
    stdev = 8.0
    lb = 5
    ub = 10

    # In this test, we consider the following methods of the problem instance.
    # Each tuple is in the form (method, filling_value, further_kwargs).
    # For example, problem.make_ones has its filling_value as 1, which means that it is a method which fills its
    # target with 1 values.
    funcs = [
        (problem.make_empty, None, {}),
        (problem.make_ones, 1, {}),
        (problem.make_zeros, 0, {}),
        (problem.make_nan, float("nan"), {}),
        (problem.make_gaussian, None, {"center": center, "stdev": stdev}),
        (problem.make_uniform, None, {"lb": lb, "ub": ub}),
    ]

    for func, filling_value, further_kwargs in funcs:  # For each function and its filling value
        for use_eval_dtype in (False, True):  # Do without and with use_eval_dtype

            if ("center" in further_kwargs) and ("stdev" in further_kwargs) and (not use_eval_dtype) and (not is_float):
                # If "center" and "stdev" are in further_kwargs, then our function is make_gaussian(...).
                # If our current dtype is an integer type (i.e. not is_float), then make_gaussian(...) cannot work.
                # So, we skip this step of the loop.
                continue

            # Determine the desired dtype of the new tensor
            target_dtype = problem.eval_dtype if use_eval_dtype else problem.dtype

            # Are we trying to use make_gaussian(...) on an integer type (which would fail)?
            gaussian_on_int = (
                ("center" in further_kwargs) and ("stdev" in further_kwargs) and ("int" in str(target_dtype))
            )

            if not gaussian_on_int:
                if (filling_value is not None) and np.isnan(filling_value) and ("int" in str(target_dtype)):
                    # If the new tensor's dtype is expected as an integer, and the function's filling value is NaN, then
                    # we skip this step of the loop.
                    continue

                if shape is None:
                    # If the shape argument is None, then we create a new tensor with the help of the `num_solutions`
                    # keyword argument of the tensor creation method.
                    new_tensor = func(num_solutions=numsln, use_eval_dtype=use_eval_dtype, **further_kwargs)
                    assert new_tensor.shape == (numsln,)  # the new tensor's shape must be correct
                else:
                    # If a shape is given, then we create a new tensor of the desired shape
                    new_tensor = func(shape, use_eval_dtype=use_eval_dtype, **further_kwargs)
                    assert new_tensor.shape == shape  # the new tensor's shape must be correct
                assert new_tensor.dtype == target_dtype  # the new tensor's dtype must be correct

                if filling_value is not None:
                    # If there is a filling value for this function (e.g. like 1 for make_ones, or 0 for make_zeros)
                    # then we need to ensure that the function did the filling correctly.

                    if np.isnan(filling_value):
                        # If the filling value is NaN, then we ensure that all values within the new tensor are NaN
                        assert torch.all(torch.isnan(new_tensor))
                    else:
                        # If the filling value is a real number, then we check if the tensor is filled correctly
                        _each_must_be_equal(new_tensor, filling_value)

                if ("lb" in further_kwargs) and ("ub" in further_kwargs):
                    # If "lb" and "ub" are in the args, then the function we are working with is make_uniform.
                    # With this knowledge, we now ensure that the new tensor has its values within the desired boundaries.
                    _all_must_be_between(new_tensor, lb, ub)
