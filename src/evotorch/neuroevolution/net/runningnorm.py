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

from collections import namedtuple
from copy import deepcopy
from typing import Iterable, Optional, Union

import torch
from torch import nn

from evotorch.tools import Device, DType, to_torch_dtype

CollectedStats = namedtuple("CollectedStats", ["mean", "stdev"])


def _clamp(x: torch.Tensor, min: Optional[float], max: Optional[float]) -> torch.Tensor:
    """
    Clamp the tensor x according to the given min and max values.
    Unlike PyTorch's clamp, this function allows both min and max
    to be None, in which case no clamping will be done.

    Args:
        x: The tensor subject to the clamp operation.
        min: The minimum value.
        max: The maximum value.
    Returns:
        The result of the clamp operation, as a tensor.
        If both min and max were None, the returned object is x itself.
    """
    if (min is None) and (max is None):
        return x
    else:
        return torch.clamp(x, min, max)


class RunningNorm:
    """
    An online observation normalization tool
    """

    def __init__(
        self,
        *,
        shape: Union[tuple, int],
        dtype: DType,
        device: Optional[Device] = None,
        min_variance: float = 1e-2,
        clip: Optional[tuple] = None,
    ) -> None:
        """
        `__init__(...)`: Initialize the RunningNorm

        Args:
            shape: Observation shape. Can be an integer or a tuple.
            dtype: The dtype of the observations.
            device: The device in which the observation stats are held.
                If left as None, the device is assumed to be "cpu".
            min_variance: A lower bound for the variance to be used in
                the normalization computations.
                In other words, if the computed variance according to the
                collected observations ends up lower than `min_variance`,
                this `min_variance` will be used instead (in an elementwise
                manner) while computing the normalized observations.
                As in Salimans et al. (2017), the default is 1e-2.
            clip: Can be left as None (which is the default), or can be
                given as a pair of real numbers.
                This is used for clipping the observations after the
                normalization operation.
                In Salimans et al. (2017), (-5.0, +5.0) was used.
        """

        # Make sure that the shape is stored as a torch.Size object.
        if isinstance(shape, Iterable):
            self._shape = torch.Size(shape)
        else:
            self._shape = torch.Size([int(shape)])

        # Store the number of dimensions
        self._ndim = len(self._shape)

        # Store the dtype and the device
        self._dtype = to_torch_dtype(dtype)
        self._device = "cpu" if device is None else device

        # Initialize the internally stored data as empty
        self._sum: Optional[torch.Tensor] = None
        self._sum_of_squares: Optional[torch.Tensor] = None
        self._count: int = 0

        # Store the minimum variance
        self._min_variance = float(min_variance)

        if clip is not None:
            # If a clip tuple was provided, store the specified lower and upper bounds
            lb, ub = clip
            self._lb = float(lb)
            self._ub = float(ub)
        else:
            # If a clip tuple was not provided the bounds are stored as None
            self._lb = None
            self._ub = None

    def to(self, device: Device) -> "RunningNorm":
        """
        If the target device is a different device, then make a copy of this
        RunningNorm instance on the target device.
        If the target device is the same with this RunningNorm's device, then
        return this RunningNorm itself.

        Args:
            device: The target device.
        Returns:
            The RunningNorm on the target device. This can be a copy, or the
            original RunningNorm instance itself.
        """
        if torch.device(device) == torch.device(self.device):
            return self
        else:
            new_running_norm = object.__new__(type(self))

            already_handled = {"_sum", "_sum_of_squares", "_device"}
            new_running_norm._sum = self._sum.to(device)
            new_running_norm._sum_of_squares = self._sum_of_squares.to(device)
            new_running_norm._device = device

            for k, v in self.__dict__.items():
                if k not in already_handled:
                    setattr(new_running_norm, k, deepcopy(v))

            return new_running_norm

    @property
    def device(self) -> Device:
        """
        The device in which the observation stats are held
        """
        return self._device

    @property
    def dtype(self) -> DType:
        """
        The dtype of the stored observation stats
        """
        return self._dtype

    @property
    def shape(self) -> tuple:
        """
        Observation shape
        """
        return self._shape

    @property
    def min_variance(self) -> float:
        """
        Minimum variance
        """
        return self._min_variance

    @property
    def low(self) -> Optional[float]:
        """
        The lower component of the bounds given in the `clip` tuple.
        If `clip` was initialized as None, this is also None.
        """
        return self._lb

    @property
    def high(self) -> Optional[float]:
        """
        The higher (upper) component of the bounds given in the `clip` tuple.
        If `clip` was initialized as None, this is also None.
        """
        return self._ub

    def _like_its_own(self, x: Iterable) -> torch.Tensor:
        return torch.as_tensor(x, dtype=self._dtype, device=self._device)

    def _verify(self, x: Iterable) -> torch.Tensor:
        x = self._like_its_own(x)
        if x.ndim == self._ndim:
            if x.shape != self._shape:
                raise ValueError(
                    f"This RunningNorm instance was initialized with shape: {self._shape}."
                    f" However, the provided tensor has an incompatible shape: {x._shape}."
                )
        elif x.ndim == (self._ndim + 1):
            if x.shape[1:] != self._shape:
                raise ValueError(
                    f"This RunningNorm instance was initialized with shape: {self._shape}."
                    f" The provided tensor is shaped {x.shape}."
                    f" Accepting the tensor's leftmost dimension as the batch size,"
                    f" the remaining shape is incompatible: {x.shape[1:]}"
                )
        else:
            raise ValueError(
                f"This RunningNorm instance was initialized with shape: {self._shape}."
                f" The provided tensor is shaped {x.shape}."
                f" The number of dimensions of the given tensor is incompatible."
            )
        return x

    def _has_no_data(self) -> bool:
        return (self._sum is None) and (self._sum_of_squares is None) and (self._count == 0)

    def _has_data(self) -> bool:
        return (self._sum is not None) and (self._sum_of_squares is not None) and (self._count > 0)

    def reset(self):
        """
        Remove all the collected observation data.
        """
        self._sum = None
        self._sum_of_squares = None
        self._count = 0

    @torch.no_grad()
    def update(self, x: Union[Iterable, "RunningNorm"], mask: Optional[Iterable] = None, *, verify: bool = True):
        """
        Update the stored stats with new observation data.

        Args:
            x: The new observation(s), as a PyTorch tensor, or any Iterable
                that can be converted to a PyTorch tensor, or another
                RunningNorm instance.
                If given as a tensor or as an Iterable, the shape of `x` can
                be the same with observation shape, or it can be augmented
                with an extra leftmost dimension.
                In the case of augmented dimension, `x` is interpreted not as
                a single observation, but as a batch of observations.
                If `x` is another RunningNorm instance, the stats stored by
                this RunningNorm instance will be updated with all the data
                stored by `x`.
            mask: Can be given as a 1-dimensional Iterable of booleans ONLY
                if `x` represents a batch of observations.
                If a `mask` is provided, the i-th observation within the
                observation batch `x` will be taken into account only if
                the i-th item of the `mask` is True.
            verify: Whether or not to verify the shape of the given Iterable
                objects. The default is True.
        """
        if isinstance(x, RunningNorm):
            # If we are to update our stats according to another RunningNorm instance

            if x._count > 0:
                # We bother only if x is non-empty

                if mask is not None:
                    # We were given another RunningNorm, not a batch of observations.
                    # So, we do not expect to receive a mask tensor.
                    # If a mask was provided, then this is an unexpected way of calling this function.
                    # We therefore raise an error.
                    raise ValueError(
                        "The `mask` argument is expected as None if the first argument is a RunningNorm."
                        " However, `mask` is found as something other than None."
                    )

                if self._shape != x._shape:
                    # If the shapes of this RunningNorm and of the other RunningNorm
                    # do not match, then we cannot use `x` for updating our stats.
                    # It might be the case that `x` was initialized for another
                    # task, with differently sized observations.
                    # We therefore raise an error.
                    raise ValueError(
                        f"The RunningNorm to be updated has the shape {self._shape}"
                        f" The other RunningNorm has the shape {self._shape}"
                        f" These shapes are incompatible."
                    )

                if self._has_no_data():
                    # If this RunningNorm has no data at all, then we clone the
                    # data of x.
                    self._sum = self._like_its_own(x._sum.clone())
                    self._sum_of_squares = self._like_its_own(x._sum_of_squares.clone())
                    self._count = x._count
                elif self._has_data():
                    # If this RunningNorm has its own data, then we update the
                    # stored data with the data stored by x.
                    self._sum += self._like_its_own(x._sum)
                    self._sum_of_squares += self._like_its_own(x._sum_of_squares)
                    self._count += x._count
                else:
                    assert False, "RunningNorm is in an invalid state! This might be a bug."
        else:
            # This is the case where the received argument x is not a
            # RunningNorm object, but an Iterable.

            if verify:
                # If we have the `verify` flag, then we make sure that
                # x is a tensor of the correct shape
                x = self._verify(x)

            if x.ndim == self._ndim:
                # If the shape of x is exactly the same with the observation shape
                # then we assume that x represents a single observation, and not a
                # batch of observations.

                if mask is not None:
                    # Since we are dealing with a single observation,
                    # we do not expect to receive a mask argument.
                    # If the mask argument was provided, then this is an unexpected
                    # usage of this function.
                    # We therefore raise an error.
                    raise ValueError(
                        "The `mask` argument is expected as None if the first argument is a single observation"
                        " (i.e. not a batch of observations, with an extra leftmost dimension)."
                        " However, `mask` is found as something other than None."
                    )

                # Since x is a single observation,
                # the sum of observations extracted from x is x itself,
                # and the sum of squared observations extracted from x is
                # the square of x itself.
                sum_of_x = x
                sum_of_x_squared = x.square()
                # We extracted a single observation from x
                n = 1
            elif x.ndim == (self._ndim + 1):
                # If the number of dimensions of x is one more than the number
                # of dimensions of this RunningNorm, then we assume that x is a batch
                # of observations.

                if mask is not None:
                    # If a mask is provided, then we first make sure that it is a tensor
                    # of dtype bool in the correct device.
                    mask = torch.as_tensor(mask, dtype=torch.bool, device=self._device)

                    if mask.ndim != 1:
                        # We expect the mask to be 1-dimensional.
                        # If not, we raise an error.
                        raise ValueError(
                            f"The `mask` tensor was expected as a 1-dimensional tensor."
                            f" However, its shape is {mask.shape}."
                        )

                    if len(mask) != x.shape[0]:
                        # If the length of the mask is not the batch size of x,
                        # then there is a mismatch.
                        # We therefore raise an error.
                        raise ValueError(
                            f"The shape of the given tensor is {x.shape}."
                            f" Therefore, the batch size of observations is {x.shape[0]}."
                            f" However, the given `mask` tensor does not has an incompatible length: {len(mask)}."
                        )

                    # We compute how many True items we have in the mask.
                    # This integer gives us how many observations we extract from x.
                    n = int(torch.sum(torch.as_tensor(mask, dtype=torch.int64, device=self._device)))

                    # We now re-cast the mask as the observation dtype (so that True items turn to 1.0
                    # and False items turn to 0.0), and then increase its number of dimensions so that
                    # it can operate directly with x.
                    mask = self._like_its_own(mask).reshape(torch.Size([x.shape[0]] + ([1] * (x.ndim - 1))))

                    # Finally, we multiply x with the mask. This means that the observations with corresponding
                    # mask values as False are zeroed out.
                    x = x * mask
                else:
                    # This is the case where we did not receive a mask.
                    # We can simply say that the number of observations to extract from x
                    # is the size of its leftmost dimension, i.e. the batch size.
                    n = x.shape[0]

                # With or without a mask, we are now ready to extract the sum and sum of squares
                # from x.
                sum_of_x = torch.sum(x, dim=0)
                sum_of_x_squared = torch.sum(x.square(), dim=0)
            else:
                # This is the case where the number of dimensions of x is unrecognized.
                # This case is actually already checked by the _verify(...) method earlier.
                # This defensive fallback case is only for when verify=False and it turned out
                # that the ndim is invalid.
                raise ValueError(f"Invalid shape: {x.shape}")

            # At this point, we handled all the valid cases regarding the Iterable x,
            # and we have our sum_of_x (sum of all observations), sum_of_squares
            # (sum of all squared observations), and n (number of observations extracted
            # from x).

            if self._has_no_data():
                # If our RunningNorm is empty, the observation data we extracted from x
                # become our RunningNorm's new data.
                self._sum = sum_of_x
                self._sum_of_squares = sum_of_x_squared
                self._count = n
            elif self._has_data():
                # If our RunningNorm is not empty, the stored data is updated with the
                # data extracted from x.
                self._sum += sum_of_x
                self._sum_of_squares += sum_of_x_squared
                self._count += n
            else:
                # This is an erroneous state where the internal data looks neither
                # existent nor completely empty.
                # This might be the result of a bug, or maybe this instance's
                # protected variables were tempered with from the outside.
                assert False, "RunningNorm is in an invalid state! This might be a bug."

    @property
    @torch.no_grad()
    def stats(self) -> CollectedStats:
        """
        The collected data's mean and standard deviation (stdev) in a tuple
        """

        # Using the internally stored sum, sum_of_squares, and count,
        # compute E[x] and E[x^2]
        E_x = self._sum / self._count
        E_x2 = self._sum_of_squares / self._count

        # The mean is E[x]
        mean = E_x

        # The variance is E[x^2] - (E[x])^2, elementwise clipped such that
        # it cannot go below min_variance
        variance = _clamp(E_x2 - E_x.square(), self._min_variance, None)

        # Standard deviation is finally computed as the square root of the variance
        stdev = torch.sqrt(variance)

        # Return the stats in a named tuple
        return CollectedStats(mean=mean, stdev=stdev)

    @property
    def mean(self) -> torch.Tensor:
        """
        The collected data's mean
        """
        return self._sum / self._count

    @property
    def stdev(self) -> torch.Tensor:
        """
        The collected data's standard deviation
        """
        return self.stats.stdev

    @property
    def sum(self) -> torch.Tensor:
        """
        The collected data's sum
        """
        return self._sum

    @property
    def sum_of_squares(self) -> torch.Tensor:
        """
        Sum of squares of the collected data
        """
        return self._sum_of_squares

    @property
    def count(self) -> int:
        """
        Number of observations encountered
        """
        return self._count

    @torch.no_grad()
    def normalize(self, x: Iterable, *, result_as_numpy: Optional[bool] = None, verify: bool = True) -> Iterable:
        """
        Normalize the given observation x.

        Args:
            x: The observation(s), as a PyTorch tensor, or any Iterable
                that is convertable to a PyTorch tensor.
                `x` can be a single observation, or it can be a batch
                of observations (with an extra leftmost dimension).
            result_as_numpy: Whether or not to return the normalized
                observation as a numpy array.
                If left as None (which is the default), then the returned
                type depends on x: a PyTorch tensor is returned if x is a
                PyTorch tensor, and a numpy array is returned otherwise.
                If True, the result is always a numpy array.
                If False, the result is always a PyTorch tensor.
            verify: Whether or not to check the type and dimensions of x.
                This is True by default.
                Note that, if `verify` is False, this function will not
                properly check the type of `x` and will assume that `x`
                is a PyTorch tensor.
        Returns:
            The normalized observation, as a PyTorch tensor or a numpy array.
        """

        if self._count == 0:
            # If this RunningNorm instance has no data yet,
            # then we do not know how to do the normalization.
            # We therefore raise an error.
            raise ValueError("Cannot do normalization because no data is collected yet.")

        if verify:
            # Here we verify the type and shape of x.

            if result_as_numpy is None:
                # If there is not an explicit request about the return type,
                # we infer the return type from the type of x:
                # if x is a tensor, we return a tensor;
                # otherwise, we assume x to be a CPU-bound iterable, and
                # therefore we return a numpy array.
                result_as_numpy = not isinstance(x, torch.Tensor)
            else:
                result_as_numpy = bool(result_as_numpy)

            # We call _verify() to make sure that x is of correct shape
            # and is properly converted to a PyTorch tensor.
            x = self._verify(x)

        # We get the mean and stdev of the collected data
        mean, stdev = self.stats

        # Now we compute the normalized observation, clipped according to the
        # lower and upper bounds expressed by the `clip` tuple, if exists.
        result = _clamp((x - mean) / stdev, self._lb, self._ub)

        if result_as_numpy:
            # If we are to return the result as a numpy array, we do the
            # necessary conversion.
            result = result.cpu().numpy()

        # Finally, return the result
        return result

    @torch.no_grad()
    def update_and_normalize(self, x: Iterable, mask: Optional[Iterable] = None) -> Iterable:
        """
        Update the observation stats according to x, then normalize x.

        Args:
            x: The observation(s), as a PyTorch tensor, or as an Iterable
                which can be converted to a PyTorch tensor.
                The shape of x can be the same with the observaiton shape,
                or it can be augmented with an extra leftmost dimension
                to express a batch of observations.
            mask: Can be given as a 1-dimensional Iterable of booleans ONLY
                if `x` represents a batch of observations.
                If a `mask` is provided, the i-th observation within the
                observation batch `x` will be taken into account only if
                the the i-th item of the `mask` is True.
        Returns:
            The normalized counterpart of the observation(s) expressed by x.
        """
        result_as_numpy = not isinstance(x, torch.Tensor)
        x = self._verify(x)

        self.update(x, mask, verify=False)
        result = self.normalize(x, verify=False)

        if result_as_numpy:
            result = result.cpu().numpy()

        return result

    def to_layer(self) -> "ObsNormLayer":
        """
        Make a PyTorch module which normalizes the its inputs.

        Returns:
            An ObsNormLayer instance.
        """
        mean, stdev = self.stats
        low = self.low
        high = self.high
        return ObsNormLayer(mean=mean, stdev=stdev, low=low, high=high)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}, count: {self.count}>"

    def __copy__(self) -> "RunningNorm":
        return deepcopy(self)


class ObsNormLayer(nn.Module):
    """
    An observation normalizer which behaves as a PyTorch Module.
    """

    def __init__(
        self, mean: torch.Tensor, stdev: torch.Tensor, low: Optional[float] = None, high: Optional[float] = None
    ) -> None:
        """
        `__init__(...)`: Initialize the ObsNormLayer.

        Args:
            mean: The mean according to which the observations are to be
                normalized.
            stdev: The standard deviation according to which the observations
                are to be normalized.
            low: Optionally a real number if the result of the normalization
                is to be clipped. Represents the lower bound for the clipping
                operation.
            high: Optionally a real number if the result of the normalization
                is to be clipped. Represents the upper bound for the clipping
                operation.
        """
        super().__init__()
        self.register_buffer("_mean", mean)
        self.register_buffer("_stdev", stdev)
        self._lb = None if low is None else float(low)
        self._ub = None if high is None else float(high)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize an observation or a batch of observations.

        Args:
            x: The observation(s).
        Returns:
            The normalized counterpart of the observation(s).
        """
        return _clamp((x - self._mean) / self._stdev, self._lb, self._ub)
