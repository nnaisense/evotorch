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
Base classes with various utilities for creating tensors.
"""

from typing import Any, Iterable, Optional, Union

import torch

from . import misc
from .misc import Device, DType, DTypeAndDevice, RealOrVector, Size, is_dtype_object, to_torch_dtype


class TensorMakerMixin:
    def __get_dtype_and_device_kwargs(
        self,
        *,
        dtype: Optional[DType],
        device: Optional[Device],
        use_eval_dtype: bool,
        out: Optional[Iterable],
    ) -> dict:
        result = {}
        if out is None:
            if dtype is None:
                if use_eval_dtype:
                    if hasattr(self, "eval_dtype"):
                        result["dtype"] = self.eval_dtype
                    else:
                        raise AttributeError(
                            f"Received `use_eval_dtype` as {repr(use_eval_dtype)}, which represents boolean truth."
                            f" However, evaluation dtype cannot be determined, because this object does not have"
                            f" an attribute named `eval_dtype`."
                        )
                else:
                    result["dtype"] = self.dtype
            else:
                if use_eval_dtype:
                    raise ValueError(
                        f"Received both a `dtype` argument ({repr(dtype)}) and `use_eval_dtype` as True."
                        f" These arguments are conflicting."
                        f" Please either provide a `dtype`, or leave `dtype` as None and pass `use_eval_dtype=True`."
                    )
                else:
                    result["dtype"] = dtype

            if device is None:
                result["device"] = self.device
            else:
                result["device"] = device

        return result

    def __get_size_args(self, *size: Size, num_solutions: Optional[int], out: Optional[Iterable]) -> tuple:
        if out is None:
            nsize = len(size)
            if (nsize == 0) and (num_solutions is None):
                return tuple()
            elif (nsize >= 1) and (num_solutions is None):
                return size
            elif (nsize == 0) and (num_solutions is not None):
                if hasattr(self, "solution_length"):
                    num_solutions = int(num_solutions)
                    if self.solution_length is None:
                        return (num_solutions,)
                    else:
                        return (num_solutions, self.solution_length)
                else:
                    raise AttributeError(
                        f"Received `num_solutions` as {repr(num_solutions)}."
                        f" However, to determine the target tensor's size via `num_solutions`, this object"
                        f" needs to have an attribute named `solution_length`, which seems to be missing."
                    )
            else:
                raise ValueError(
                    f"Encountered both `size` arguments ({repr(size)})"
                    f" and `num_solutions` keyword argument (num_solutions={repr(num_solutions)})."
                    f" Specifying both `size` and `num_solutions` is not valid."
                )
        else:
            return tuple()

    def __get_generator_kwargs(self, *, generator: Any) -> dict:
        result = {}
        if generator is None:
            if hasattr(self, "generator"):
                result["generator"] = self.generator
        else:
            result["generator"] = generator
        return result

    def __get_all_args_for_maker(
        self,
        *size: Size,
        num_solutions: Optional[int],
        out: Optional[Iterable],
        dtype: Optional[DType],
        device: Optional[Device],
        use_eval_dtype: bool,
    ) -> tuple:
        args = self.__get_size_args(*size, num_solutions=num_solutions, out=out)
        kwargs = self.__get_dtype_and_device_kwargs(dtype=dtype, device=device, use_eval_dtype=use_eval_dtype, out=out)
        if out is not None:
            kwargs["out"] = out
        return args, kwargs

    def __get_all_args_for_random_maker(
        self,
        *size: Size,
        num_solutions: Optional[int],
        out: Optional[Iterable],
        dtype: Optional[DType],
        device: Optional[Device],
        use_eval_dtype: bool,
        generator: Any,
    ):
        args = self.__get_size_args(*size, num_solutions=num_solutions, out=out)

        kwargs = {}
        kwargs.update(
            self.__get_dtype_and_device_kwargs(dtype=dtype, device=device, use_eval_dtype=use_eval_dtype, out=out)
        )
        kwargs.update(self.__get_generator_kwargs(generator=generator))
        if out is not None:
            kwargs["out"] = out

        return args, kwargs

    def make_tensor(
        self,
        data: Any,
        *,
        dtype: Optional[DType] = None,
        device: Optional[Device] = None,
        use_eval_dtype: bool = False,
        read_only: bool = False,
    ) -> Iterable:
        """
        Make a new tensor.

        When not explicitly specified via arguments, the dtype and the device
        of the resulting tensor is determined by this method's parent object.

        Args:
            data: The data to be converted to a tensor.
                If one wishes to create a PyTorch tensor, this can be anything
                that can be stored by a PyTorch tensor.
                If one wishes to create an `ObjectArray` and therefore passes
                `dtype=object`, then the provided `data` is expected as an
                `Iterable`.
            dtype: Optionally a string (e.g. "float32"), or a PyTorch dtype
                (e.g. torch.float32), or `object` or "object" (as a string)
                or `Any` if one wishes to create an `ObjectArray`.
                If `dtype` is not specified it will be assumed that the user
                wishes to create a tensor using the dtype of this method's
                parent object.
            device: The device in which the tensor will be stored.
                If `device` is not specified, it will be assumed that the user
                wishes to create a tensor on the device of this method's
                parent object.
            use_eval_dtype: If this is given as True and a `dtype` is not
                specified, then the `dtype` of the result will be taken
                from the `eval_dtype` attribute of this method's parent
                object.
            read_only: Whether or not the created tensor will be read-only.
                By default, this is False.
        Returns:
            A PyTorch tensor or an ObjectArray.
        """
        kwargs = self.__get_dtype_and_device_kwargs(dtype=dtype, device=device, use_eval_dtype=use_eval_dtype, out=None)
        return misc.make_tensor(data, read_only=read_only, **kwargs)

    def make_empty(
        self,
        *size: Size,
        num_solutions: Optional[int] = None,
        out: Optional[Iterable] = None,
        dtype: Optional[DType] = None,
        device: Optional[Device] = None,
        use_eval_dtype: bool = False,
    ) -> Iterable:
        """
        Make an empty tensor.

        When not explicitly specified via arguments, the dtype and the device
        of the resulting tensor is determined by this method's parent object.

        Args:
            size: Shape of the empty tensor to be created.
                expected as multiple positional arguments of integers,
                or as a single positional argument containing a tuple of
                integers.
                Note that when the user wishes to create an `ObjectArray`
                (i.e. when `dtype` is given as `object`), then the size
                is expected as a single integer, or as a single-element
                tuple containing an integer (because `ObjectArray` can only
                be one-dimensional).
            num_solutions: This can be used instead of the `size` arguments
                for specifying the shape of the target tensor.
                Expected as an integer, when `num_solutions` is specified
                as `n`, the shape of the resulting tensor will be
                `(n, m)` where `m` is the solution length reported by this
                method's parent object's `solution_length` attribute.
            dtype: Optionally a string (e.g. "float32") or a PyTorch dtype
                (e.g. torch.float32) or, for creating an `ObjectArray`,
                "object" (as string) or `object` or `Any`.
                If `dtype` is not specified (and also `out` is None),
                it will be assumed that the user wishes to create a tensor
                using the dtype of this method's parent object.
            device: The device in which the new empty tensor will be stored.
                If not specified (and also `out` is None), it will be
                assumed that the user wishes to create a tensor on the
                same device with this method's parent object.
            use_eval_dtype: If this is given as True and a `dtype` is not
                specified, then the `dtype` of the result will be taken
                from the `eval_dtype` attribute of this method's parent
                object.
        Returns:
            The new empty tensor, which can be a PyTorch tensor or an
            `ObjectArray`.
        """
        args, kwargs = self.__get_all_args_for_maker(
            *size,
            num_solutions=num_solutions,
            out=out,
            dtype=dtype,
            device=device,
            use_eval_dtype=use_eval_dtype,
        )
        return misc.make_empty(*args, **kwargs)

    def make_zeros(
        self,
        *size: Size,
        num_solutions: Optional[int] = None,
        out: Optional[torch.Tensor] = None,
        dtype: Optional[DType] = None,
        device: Optional[Device] = None,
        use_eval_dtype: bool = False,
    ) -> torch.Tensor:
        """
        Make a new tensor filled with 0, or fill an existing tensor with 0.

        When not explicitly specified via arguments, the dtype and the device
        of the resulting tensor is determined by this method's parent object.

        Args:
            size: Size of the new tensor to be filled with 0.
                This can be given as multiple positional arguments, each such
                positional argument being an integer, or as a single positional
                argument of a tuple, the tuple containing multiple integers.
                Note that, if the user wishes to fill an existing tensor with
                0 values, then no positional argument is expected.
            num_solutions: This can be used instead of the `size` arguments
                for specifying the shape of the target tensor.
                Expected as an integer, when `num_solutions` is specified
                as `n`, the shape of the resulting tensor will be
                `(n, m)` where `m` is the solution length reported by this
                method's parent object's `solution_length` attribute.
            out: Optionally, the tensor to be filled by 0 values.
                If an `out` tensor is given, then no `size` argument is expected.
            dtype: Optionally a string (e.g. "float32") or a PyTorch dtype
                (e.g. torch.float32).
                If `dtype` is not specified (and also `out` is None),
                it will be assumed that the user wishes to create a tensor
                using the dtype of this method's parent object.
                If an `out` tensor is specified, then `dtype` is expected
                as None.
            device: The device in which the new empty tensor will be stored.
                If not specified (and also `out` is None), it will be
                assumed that the user wishes to create a tensor on the
                same device with this method's parent object.
                If an `out` tensor is specified, then `device` is expected
                as None.
            use_eval_dtype: If this is given as True and a `dtype` is not
                specified, then the `dtype` of the result will be taken
                from the `eval_dtype` attribute of this method's parent
                object.
        Returns:
            The created or modified tensor after placing 0 values.
        """

        args, kwargs = self.__get_all_args_for_maker(
            *size,
            num_solutions=num_solutions,
            out=out,
            dtype=dtype,
            device=device,
            use_eval_dtype=use_eval_dtype,
        )
        return misc.make_zeros(*args, **kwargs)

    def make_ones(
        self,
        *size: Size,
        num_solutions: Optional[int] = None,
        out: Optional[torch.Tensor] = None,
        dtype: Optional[DType] = None,
        device: Optional[Device] = None,
        use_eval_dtype: bool = False,
    ) -> torch.Tensor:
        """
        Make a new tensor filled with 1, or fill an existing tensor with 1.

        When not explicitly specified via arguments, the dtype and the device
        of the resulting tensor is determined by this method's parent object.

        Args:
            size: Size of the new tensor to be filled with 1.
                This can be given as multiple positional arguments, each such
                positional argument being an integer, or as a single positional
                argument of a tuple, the tuple containing multiple integers.
                Note that, if the user wishes to fill an existing tensor with
                1 values, then no positional argument is expected.
            num_solutions: This can be used instead of the `size` arguments
                for specifying the shape of the target tensor.
                Expected as an integer, when `num_solutions` is specified
                as `n`, the shape of the resulting tensor will be
                `(n, m)` where `m` is the solution length reported by this
                method's parent object's `solution_length` attribute.
            out: Optionally, the tensor to be filled by 1 values.
                If an `out` tensor is given, then no `size` argument is expected.
            dtype: Optionally a string (e.g. "float32") or a PyTorch dtype
                (e.g. torch.float32).
                If `dtype` is not specified (and also `out` is None),
                it will be assumed that the user wishes to create a tensor
                using the dtype of this method's parent object.
                If an `out` tensor is specified, then `dtype` is expected
                as None.
            device: The device in which the new empty tensor will be stored.
                If not specified (and also `out` is None), it will be
                assumed that the user wishes to create a tensor on the
                same device with this method's parent object.
                If an `out` tensor is specified, then `device` is expected
                as None.
            use_eval_dtype: If this is given as True and a `dtype` is not
                specified, then the `dtype` of the result will be taken
                from the `eval_dtype` attribute of this method's parent
                object.
        Returns:
            The created or modified tensor after placing 1 values.
        """
        args, kwargs = self.__get_all_args_for_maker(
            *size,
            num_solutions=num_solutions,
            out=out,
            dtype=dtype,
            device=device,
            use_eval_dtype=use_eval_dtype,
        )
        return misc.make_ones(*args, **kwargs)

    def make_nan(
        self,
        *size: Size,
        num_solutions: Optional[int] = None,
        out: Optional[torch.Tensor] = None,
        dtype: Optional[DType] = None,
        device: Optional[Device] = None,
        use_eval_dtype: bool = False,
    ) -> torch.Tensor:
        """
        Make a new tensor filled with NaN values, or fill an existing tensor
        with NaN values.

        When not explicitly specified via arguments, the dtype and the device
        of the resulting tensor is determined by this method's parent object.

        Args:
            size: Size of the new tensor to be filled with NaN.
                This can be given as multiple positional arguments, each such
                positional argument being an integer, or as a single positional
                argument of a tuple, the tuple containing multiple integers.
                Note that, if the user wishes to fill an existing tensor with
                NaN values, then no positional argument is expected.
            num_solutions: This can be used instead of the `size` arguments
                for specifying the shape of the target tensor.
                Expected as an integer, when `num_solutions` is specified
                as `n`, the shape of the resulting tensor will be
                `(n, m)` where `m` is the solution length reported by this
                method's parent object's `solution_length` attribute.
            out: Optionally, the tensor to be filled by NaN values.
                If an `out` tensor is given, then no `size` argument is expected.
            dtype: Optionally a string (e.g. "float32") or a PyTorch dtype
                (e.g. torch.float32).
                If `dtype` is not specified (and also `out` is None),
                it will be assumed that the user wishes to create a tensor
                using the dtype of this method's parent object.
                If an `out` tensor is specified, then `dtype` is expected
                as None.
            device: The device in which the new empty tensor will be stored.
                If not specified (and also `out` is None), it will be
                assumed that the user wishes to create a tensor on the
                same device with this method's parent object.
                If an `out` tensor is specified, then `device` is expected
                as None.
            use_eval_dtype: If this is given as True and a `dtype` is not
                specified, then the `dtype` of the result will be taken
                from the `eval_dtype` attribute of this method's parent
                object.
        Returns:
            The created or modified tensor after placing NaN values.
        """
        args, kwargs = self.__get_all_args_for_maker(
            *size,
            num_solutions=num_solutions,
            out=out,
            dtype=dtype,
            device=device,
            use_eval_dtype=use_eval_dtype,
        )
        return misc.make_nan(*args, **kwargs)

    def make_I(
        self,
        size: Optional[Union[int, tuple]] = None,
        *,
        out: Optional[torch.Tensor] = None,
        dtype: Optional[DType] = None,
        device: Optional[Device] = None,
        use_eval_dtype: bool = False,
    ) -> torch.Tensor:
        """
        Make a new identity matrix (I), or change an existing tensor so that
        it expresses the identity matrix.

        When not explicitly specified via arguments, the dtype and the device
        of the resulting tensor is determined by this method's parent object.

        Args:
            size: A single integer or a tuple containing a single integer,
                where the integer specifies the length of the target square
                matrix. In this context, "length" means both rowwise length
                and columnwise length, since the target is a square matrix.
                Note that, if the user wishes to fill an existing tensor with
                identity values, then `size` is expected to be left as None.
            out: Optionally, the existing tensor whose values will be changed
                so that they represent an identity matrix.
                If an `out` tensor is given, then `size` is expected as None.
            dtype: Optionally a string (e.g. "float32") or a PyTorch dtype
                (e.g. torch.float32).
                If `dtype` is not specified (and also `out` is None),
                it will be assumed that the user wishes to create a tensor
                using the dtype of this method's parent object.
                If an `out` tensor is specified, then `dtype` is expected
                as None.
            device: The device in which the new empty tensor will be stored.
                If not specified (and also `out` is None), it will be
                assumed that the user wishes to create a tensor on the
                same device with this method's parent object.
                If an `out` tensor is specified, then `device` is expected
                as None.
            use_eval_dtype: If this is given as True and a `dtype` is not
                specified, then the `dtype` of the result will be taken
                from the `eval_dtype` attribute of this method's parent
                object.
        Returns:
            The created or modified tensor after placing the I matrix values
        """

        if size is None:
            if out is None:
                if hasattr(self, "solution_length"):
                    size_args = (self.solution_length,)
                else:
                    raise AttributeError(
                        "The method `.make_I(...)` was used without any `size`"
                        " arguments."
                        " When the `size` argument is missing, the default"
                        " behavior of this method is to create an identity matrix"
                        " of size (n, n), n being the length of a solution."
                        " However, the parent object of this method does not have"
                        " an attribute name `solution_length`."
                    )
            else:
                size_args = tuple()
        elif isinstance(size, tuple):
            if len(size) != 1:
                raise ValueError(
                    f"When the size argument is given as a tuple, the method `make_I(...)` expects the tuple to have"
                    f" only one element. The given tuple is {size}."
                )
            size_args = size
        else:
            size_args = (int(size),)

        args, kwargs = self.__get_all_args_for_maker(
            *size_args,
            num_solutions=None,
            out=out,
            dtype=dtype,
            device=device,
            use_eval_dtype=use_eval_dtype,
        )
        return misc.make_I(*args, **kwargs)

    def make_uniform(
        self,
        *size: Size,
        num_solutions: Optional[int] = None,
        lb: Optional[RealOrVector] = None,
        ub: Optional[RealOrVector] = None,
        out: Optional[torch.Tensor] = None,
        dtype: Optional[DType] = None,
        device: Optional[Device] = None,
        use_eval_dtype: bool = False,
        generator: Any = None,
    ) -> torch.Tensor:
        """
        Make a new or existing tensor filled by uniformly distributed values.
        Both lower and upper bounds are inclusive.
        This function can work with both float and int dtypes.

        When not explicitly specified via arguments, the dtype and the device
        of the resulting tensor is determined by this method's parent object.

        Args:
            size: Size of the new tensor to be filled with uniformly distributed
                values. This can be given as multiple positional arguments, each
                such positional argument being an integer, or as a single
                positional argument of a tuple, the tuple containing multiple
                integers. Note that, if the user wishes to fill an existing
                tensor instead, then no positional argument is expected.
            num_solutions: This can be used instead of the `size` arguments
                for specifying the shape of the target tensor.
                Expected as an integer, when `num_solutions` is specified
                as `n`, the shape of the resulting tensor will be
                `(n, m)` where `m` is the solution length reported by this
                method's parent object's `solution_length` attribute.
            lb: Lower bound for the uniformly distributed values.
                Can be a scalar, or a tensor.
                If not specified, the lower bound will be taken as 0.
                Note that, if one specifies `lb`, then `ub` is also expected to
                be explicitly specified.
            ub: Upper bound for the uniformly distributed values.
                Can be a scalar, or a tensor.
                If not specified, the upper bound will be taken as 1.
                Note that, if one specifies `ub`, then `lb` is also expected to
                be explicitly specified.
            out: Optionally, the tensor to be filled by uniformly distributed
                values. If an `out` tensor is given, then no `size` argument is
                expected.
            dtype: Optionally a string (e.g. "float32") or a PyTorch dtype
                (e.g. torch.float32).
                If `dtype` is not specified (and also `out` is None),
                it will be assumed that the user wishes to create a tensor
                using the dtype of this method's parent object.
                If an `out` tensor is specified, then `dtype` is expected
                as None.
            device: The device in which the new empty tensor will be stored.
                If not specified (and also `out` is None), it will be
                assumed that the user wishes to create a tensor on the
                same device with this method's parent object.
                If an `out` tensor is specified, then `device` is expected
                as None.
            use_eval_dtype: If this is given as True and a `dtype` is not
                specified, then the `dtype` of the result will be taken
                from the `eval_dtype` attribute of this method's parent
                object.
            generator: Pseudo-random generator to be used when sampling
                the values. Can be a `torch.Generator` or any object with
                a `generator` attribute (e.g. a Problem object).
                If not given, then this method's parent object will be
                analyzed whether or not it has its own generator.
                If it does, that generator will be used.
                If not, the global generator of PyTorch will be used.
        Returns:
            The created or modified tensor after placing the uniformly
            distributed values.
        """
        args, kwargs = self.__get_all_args_for_random_maker(
            *size,
            num_solutions=num_solutions,
            out=out,
            dtype=dtype,
            device=device,
            use_eval_dtype=use_eval_dtype,
            generator=generator,
        )
        return misc.make_uniform(*args, lb=lb, ub=ub, **kwargs)

    def make_gaussian(
        self,
        *size: Size,
        num_solutions: Optional[int] = None,
        center: Optional[RealOrVector] = None,
        stdev: Optional[RealOrVector] = None,
        symmetric: bool = False,
        out: Optional[torch.Tensor] = None,
        dtype: Optional[DType] = None,
        device: Optional[Device] = None,
        use_eval_dtype: bool = False,
        generator: Any = None,
    ) -> torch.Tensor:
        """
        Make a new or existing tensor filled by Gaussian distributed values.
        This function can work only with float dtypes.

        Args:
            size: Size of the new tensor to be filled with Gaussian distributed
                values. This can be given as multiple positional arguments, each
                such positional argument being an integer, or as a single
                positional argument of a tuple, the tuple containing multiple
                integers. Note that, if the user wishes to fill an existing
                tensor instead, then no positional argument is expected.
            num_solutions: This can be used instead of the `size` arguments
                for specifying the shape of the target tensor.
                Expected as an integer, when `num_solutions` is specified
                as `n`, the shape of the resulting tensor will be
                `(n, m)` where `m` is the solution length reported by this
                method's parent object's `solution_length` attribute.
            center: Center point (i.e. mean) of the Gaussian distribution.
                Can be a scalar, or a tensor.
                If not specified, the center point will be taken as 0.
                Note that, if one specifies `center`, then `stdev` is also
                expected to be explicitly specified.
            stdev: Standard deviation for the Gaussian distributed values.
                Can be a scalar, or a tensor.
                If not specified, the standard deviation will be taken as 1.
                Note that, if one specifies `stdev`, then `center` is also
                expected to be explicitly specified.
            symmetric: Whether or not the values should be sampled in a
                symmetric (i.e. antithetic) manner.
                The default is False.
            out: Optionally, the tensor to be filled by Gaussian distributed
                values. If an `out` tensor is given, then no `size` argument is
                expected.
            dtype: Optionally a string (e.g. "float32") or a PyTorch dtype
                (e.g. torch.float32).
                If `dtype` is not specified (and also `out` is None),
                it will be assumed that the user wishes to create a tensor
                using the dtype of this method's parent object.
                If an `out` tensor is specified, then `dtype` is expected
                as None.
            device: The device in which the new empty tensor will be stored.
                If not specified (and also `out` is None), it will be
                assumed that the user wishes to create a tensor on the
                same device with this method's parent object.
                If an `out` tensor is specified, then `device` is expected
                as None.
            use_eval_dtype: If this is given as True and a `dtype` is not
                specified, then the `dtype` of the result will be taken
                from the `eval_dtype` attribute of this method's parent
                object.
            generator: Pseudo-random generator to be used when sampling
                the values. Can be a `torch.Generator` or any object with
                a `generator` attribute (e.g. a Problem object).
                If not given, then this method's parent object will be
                analyzed whether or not it has its own generator.
                If it does, that generator will be used.
                If not, the global generator of PyTorch will be used.
        Returns:
            The created or modified tensor after placing the Gaussian
            distributed values.
        """

        args, kwargs = self.__get_all_args_for_random_maker(
            *size,
            num_solutions=num_solutions,
            out=out,
            dtype=dtype,
            device=device,
            use_eval_dtype=use_eval_dtype,
            generator=generator,
        )
        return misc.make_gaussian(*args, center=center, stdev=stdev, symmetric=symmetric, **kwargs)

    def make_randint(
        self,
        *size: Size,
        n: Union[int, float, torch.Tensor],
        num_solutions: Optional[int] = None,
        out: Optional[torch.Tensor] = None,
        dtype: Optional[DType] = None,
        device: Optional[Device] = None,
        use_eval_dtype: bool = False,
        generator: Any = None,
    ) -> torch.Tensor:
        """
        Make a new or existing tensor filled by random integers.
        The integers are uniformly distributed within `[0 ... n-1]`.
        This function can be used with integer or float dtypes.

        Args:
            size: Size of the new tensor to be filled with uniformly distributed
                values. This can be given as multiple positional arguments, each
                such positional argument being an integer, or as a single
                positional argument of a tuple, the tuple containing multiple
                integers. Note that, if the user wishes to fill an existing
                tensor instead, then no positional argument is expected.
            n: Number of choice(s) for integer sampling.
                The lowest possible value will be 0, and the highest possible
                value will be n - 1.
                `n` can be a scalar, or a tensor.
            out: Optionally, the tensor to be filled by the random integers.
                If an `out` tensor is given, then no `size` argument is
                expected.
            dtype: Optionally a string (e.g. "int64") or a PyTorch dtype
                (e.g. torch.int64).
                If `dtype` is not specified (and also `out` is None),
                `torch.int64` will be used.
                If an `out` tensor is specified, then `dtype` is expected
                as None.
            device: The device in which the new empty tensor will be stored.
                If not specified (and also `out` is None), it will be
                assumed that the user wishes to create a tensor on the
                same device with this method's parent object.
                If an `out` tensor is specified, then `device` is expected
                as None.
            use_eval_dtype: If this is given as True and a `dtype` is not
                specified, then the `dtype` of the result will be taken
                from the `eval_dtype` attribute of this method's parent
                object.
            generator: Pseudo-random generator to be used when sampling
                the values. Can be a `torch.Generator` or any object with
                a `generator` attribute (e.g. a Problem object).
                If not given, then this method's parent object will be
                analyzed whether or not it has its own generator.
                If it does, that generator will be used.
                If not, the global generator of PyTorch will be used.
        Returns:
            The created or modified tensor after placing the uniformly
            distributed values.
        """

        if (dtype is None) and (out is None):
            dtype = torch.int64

        args, kwargs = self.__get_all_args_for_random_maker(
            *size,
            num_solutions=num_solutions,
            out=out,
            dtype=dtype,
            device=device,
            use_eval_dtype=use_eval_dtype,
            generator=generator,
        )
        return misc.make_randint(*args, n=n, **kwargs)

    def as_tensor(
        self,
        x: Any,
        dtype: Optional[DType] = None,
        device: Optional[Device] = None,
        use_eval_dtype: bool = False,
    ) -> torch.Tensor:
        """
        Get the tensor counterpart of the given object `x`.

        Args:
            x: Any object to be converted to a tensor.
            dtype: Optionally a string (e.g. "float32") or a PyTorch dtype
                (e.g. torch.float32) or, for creating an `ObjectArray`,
                "object" (as string) or `object` or `Any`.
                If `dtype` is not specified, the dtype of this method's
                parent object will be used.
            device: The device in which the resulting tensor will be stored.
                If `device` is not specified, the device of this method's
                parent object will be used.
            use_eval_dtype: If this is given as True and a `dtype` is not
                specified, then the `dtype` of the result will be taken
                from the `eval_dtype` attribute of this method's parent
                object.
        Returns:
            The tensor counterpart of the given object `x`.
        """
        kwargs = self.__get_dtype_and_device_kwargs(dtype=dtype, device=device, use_eval_dtype=use_eval_dtype, out=None)
        return misc.as_tensor(x, **kwargs)

    def ensure_tensor_length_and_dtype(
        self,
        t: Any,
        length: Optional[int] = None,
        dtype: Optional[DType] = None,
        about: Optional[str] = None,
        *,
        allow_scalar: bool = False,
        device: Optional[Device] = None,
        use_eval_dtype: bool = False,
    ) -> Iterable:
        """
        Return the given sequence as a tensor while also confirming its
        length, dtype, and device.

        Default length, dtype, device are taken from this method's
        parent object.
        In more details, these attributes belonging to this method's parent
        object will be used for determining the the defaults:
        `solution_length`, `dtype`, and `device`.

        Args:
            t: The tensor, or a sequence which is convertible to a tensor.
            length: The length to which the tensor is expected to conform.
                If missing, the `solution_length` attribute of this method's
                parent object will be used as the default value.
            dtype: The dtype to which the tensor is expected to conform.
                If `dtype` argument is missing and `use_eval_dtype` is False,
                then the default dtype will be determined by the `dtype`
                attribute of this method's parent object.
                If `dtype` argument is missing and `use_eval_dtype` is True,
                then the default dtype will be determined by the `eval_dtype`
                attribute of this method's parent object.
            about: The prefix for the error message. Can be left as None.
            allow_scalar: Whether or not to accept scalars in addition
                to vector of the desired length.
                If `allow_scalar` is False, then scalars will be converted
                to sequences of the desired length. The sequence will contain
                the same scalar, repeated.
                If `allow_scalar` is True, then the scalar itself will be
                converted to a PyTorch scalar, and then will be returned.
            device: The device in which the sequence is to be stored.
                If the given sequence is on a different device than the
                desired device, a copy on the correct device will be made.
                If device is None, the default behavior of `torch.tensor(...)`
                will be used, that is: if `t` is already a tensor, the result
                will be on the same device, otherwise, the result will be on
                the cpu.
            use_eval_dtype: Whether or not to use the evaluation dtype
                (instead of the dtype of decision values).
                If this is given as True, the `dtype` argument is expected
                as None.
                If `dtype` argument is missing and `use_eval_dtype` is False,
                then the default dtype will be determined by the `dtype`
                attribute of this method's parent object.
                If `dtype` argument is missing and `use_eval_dtype` is True,
                then the default dtype will be determined by the `eval_dtype`
                attribute of this method's parent object.
        Returns:
            The sequence whose correctness in terms of length, dtype, and
            device is ensured.
        Raises:
            ValueError: if there is a length mismatch.
        """

        if length is None:
            if hasattr(self, "solution_length"):
                length = self.solution_length
            else:
                raise AttributeError(
                    f"{about}: The argument `length` was found to be None."
                    f" When the `length` argument is None, the default behavior is to use the `solution_length`"
                    f" attribute of this method's parent object."
                    f" However, this method's parent object does NOT have a `solution_length` attribute."
                )
        dtype_and_device = self.__get_dtype_and_device_kwargs(
            dtype=dtype, device=device, use_eval_dtype=use_eval_dtype, out=None
        )

        return misc.ensure_tensor_length_and_dtype(
            t, length=length, about=about, allow_scalar=allow_scalar, **dtype_and_device
        )

    def make_uniform_shaped_like(
        self,
        t: torch.Tensor,
        *,
        lb: Optional[RealOrVector] = None,
        ub: Optional[RealOrVector] = None,
    ) -> torch.Tensor:
        """
        Make a new uniformly-filled tensor, shaped like the given tensor.
        The `dtype` and `device` will be determined by the parent of this
        method (not by the given tensor).
        If the parent of this method has its own random generator, then that
        generator will be used.

        Args:
            t: The tensor according to which the result will be shaped.
            lb: The inclusive lower bounds for the uniform distribution.
                Can be a scalar or a tensor.
                If left as None, 0.0 will be used as the upper bound.
            ub: The inclusive upper bounds for the uniform distribution.
                Can be a scalar or a tensor.
                If left as None, 1.0 will be used as the upper bound.
        Returns:
            A new tensor whose shape is the same with the given tensor.
        """
        return self.make_uniform(t.shape, lb=lb, ub=ub)

    def make_gaussian_shaped_like(
        self,
        t: torch.Tensor,
        *,
        center: Optional[RealOrVector] = None,
        stdev: Optional[RealOrVector] = None,
    ) -> torch.Tensor:
        """
        Make a new tensor, shaped like the given tensor, with its values
        filled by the Gaussian distribution.

        The `dtype` and `device` will be determined by the parent of this
        method (not by the given tensor).
        If the parent of this method has its own random generator, then that
        generator will be used.

        Args:
            t: The tensor according to which the result will be shaped.
            center: Center point for the Gaussian distribution.
                Can be a scalar or a tensor.
                If left as None, 0.0 will be used as the center point.
            stdev: The standard deviation for the Gaussian distribution.
                Can be a scalar or a tensor.
                If left as None, 1.0 will be used as the standard deviation.
        Returns:
            A new tensor whose shape is the same with the given tensor.
        """
        return self.make_gaussian(t.shape, center=center, stdev=stdev)
