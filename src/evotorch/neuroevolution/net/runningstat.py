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

from copy import deepcopy
from typing import Optional, Union

import numpy as np
import torch
from torch import nn

from .runningnorm import RunningNorm


class RunningStat:
    """
    Tool for efficiently computing the mean and stdev of arrays.
    The arrays themselves are not stored separately,
    instead, they are accumulated.

    This RunningStat is implemented as a wrapper around RunningNorm.
    The difference is that the interface of RunningStat is simplified
    to expect only numpy arrays, and expect only non-vectorized
    observations.
    With this simplified interface, RunningStat is meant to be used
    by GymNE, on classical non-vectorized gym tasks.
    """

    def __init__(self):
        """
        `__init__(...)`: Initialize the RunningStat.
        """
        self._rn: Optional[RunningNorm] = None
        self.reset()

    def reset(self):
        """
        Reset the RunningStat to its initial state.
        """
        self._rn = None

    @property
    def count(self) -> int:
        """
        Get the number of arrays accumulated.
        """
        if self._rn is None:
            return 0
        else:
            return self._rn.count

    @property
    def sum(self) -> np.ndarray:
        """
        Get the sum of all accumulated arrays.
        """
        return self._rn.sum.numpy()

    @property
    def sum_of_squares(self) -> np.ndarray:
        """
        Get the sum of squares of all accumulated arrays.
        """
        return self._rn.sum_of_squares.numpy()

    @property
    def mean(self) -> np.ndarray:
        """
        Get the mean of all accumulated arrays.
        """
        return self._rn.mean.numpy()

    @property
    def stdev(self) -> np.ndarray:
        """
        Get the standard deviation of all accumulated arrays.
        """
        return self._rn.stdev.numpy()

    def update(self, x: Union[np.ndarray, "RunningStat"]):
        """
        Accumulate more data into the RunningStat object.
        If the argument is an array, that array is added
        as one more data element.
        If the argument is another RunningStat instance,
        all the stats accumulated by that RunningStat object
        are added into this RunningStat object.
        """
        if isinstance(x, RunningStat):
            if x.count > 0:
                if self._rn is None:
                    self._rn = deepcopy(x._rn)
                else:
                    self._rn.update(x._rn)
        else:
            if self._rn is None:
                x = np.array(x, dtype="float32")
                self._rn = RunningNorm(shape=x.shape, dtype="float32", device="cpu")
            self._rn.update(x)

    def normalize(self, x: Union[np.ndarray, list]) -> np.ndarray:
        """
        Normalize the array x according to the accumulated stats.
        """
        if self._rn is None:
            return x
        else:
            x = np.array(x, dtype="float32")
            return self._rn.normalize(x)

    def __copy__(self):
        return deepcopy(self)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}, count: {self.count}>"

    def to(self, device: Union[str, torch.device]) -> "RunningStat":
        """
        If the target device is cpu, return this RunningStat instance itself.
        A RunningStat object is meant to work with numpy arrays. Therefore,
        any device other than the cpu will trigger an error.

        Args:
            device: The target device. Only cpu is supported.
        Returns:
            The original RunningStat.
        """
        if torch.device(device) == torch.device("cpu"):
            return self
        else:
            raise ValueError(
                f"The received target device is {repr(device)}. However, RunningStat can only work on a cpu."
            )

    def to_layer(self) -> nn.Module:
        """
        Make a PyTorch module which normalizes the its inputs.

        Returns:
            An ObsNormLayer instance.
        """
        return self._rn.to_layer()
