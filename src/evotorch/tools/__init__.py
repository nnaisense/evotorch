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
This namespace contains various utility functions, classes, and type aliases.
"""


__all__ = (
    "cloning",
    "versionchecking",
    "Hook",
    "as_immutable",
    "mutable_copy",
    "recursiveprintable",
    "Device",
    "DType",
    "DTypeAndDevice",
    "ErroneousResult",
    "RealOrVector",
    "Size",
    "SuppressSacredExperiment",
    "Vector",
    "as_tensor",
    "cast_tensors_in_container",
    "clip_tensor",
    "clone",
    "device_of",
    "device_of_container",
    "dtype_of",
    "dtype_of_container",
    "empty_tensor_like",
    "ensure_ray",
    "ensure_tensor_length_and_dtype",
    "expect_none",
    "inject",
    "is_bool",
    "is_bool_vector",
    "is_dtype_bool",
    "is_dtype_float",
    "is_dtype_integer",
    "is_dtype_object",
    "is_dtype_real",
    "is_integer",
    "is_integer_vector",
    "is_real",
    "is_real_vector",
    "is_sequence",
    "is_tensor_on_cpu",
    "make_empty",
    "make_gaussian",
    "make_I",
    "make_nan",
    "make_ones",
    "make_randint",
    "make_tensor",
    "make_uniform",
    "make_zeros",
    "modify_tensor",
    "multiply_rows_by_scalars",
    "numpy_copy",
    "rowwise_sum",
    "split_workload",
    "stdev_from_radius",
    "to_numpy_dtype",
    "to_stdev_init",
    "to_torch_dtype",
    "ObjectArray",
    "rank",
    "ReadOnlyTensor",
    "as_read_only_tensor",
    "read_only_tensor",
)


from . import (
    cloning,
    hook,
    immutable,
    objectarray,
    ranking,
    readonlytensor,
    recursiveprintable,
    tensormaker,
    versionchecking,
)
from .hook import Hook
from .immutable import as_immutable, mutable_copy
from .misc import (
    Device,
    DType,
    DTypeAndDevice,
    ErroneousResult,
    RealOrVector,
    Size,
    SuppressSacredExperiment,
    Vector,
    as_tensor,
    cast_tensors_in_container,
    clip_tensor,
    clone,
    device_of,
    device_of_container,
    dtype_of,
    dtype_of_container,
    empty_tensor_like,
    ensure_ray,
    ensure_tensor_length_and_dtype,
    expect_none,
    inject,
    is_bool,
    is_bool_vector,
    is_dtype_bool,
    is_dtype_float,
    is_dtype_integer,
    is_dtype_object,
    is_dtype_real,
    is_integer,
    is_integer_vector,
    is_real,
    is_real_vector,
    is_sequence,
    is_tensor_on_cpu,
    make_empty,
    make_gaussian,
    make_I,
    make_nan,
    make_ones,
    make_randint,
    make_tensor,
    make_uniform,
    make_zeros,
    modify_tensor,
    multiply_rows_by_scalars,
    numpy_copy,
    rowwise_sum,
    split_workload,
    stdev_from_radius,
    to_numpy_dtype,
    to_stdev_init,
    to_torch_dtype,
)
from .objectarray import ObjectArray
from .ranking import rank
from .readonlytensor import ReadOnlyTensor, as_read_only_tensor, read_only_tensor