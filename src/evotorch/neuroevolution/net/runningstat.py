# The following code is adapted from:
# https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/es.py.
#
# The license of OpenAI's original code is as follows:
#
# The MIT License
#
# Copyright (c) 2016 OpenAI (http://openai.com)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.


from copy import copy, deepcopy
from typing import Union

import numpy as np


class RunningStat:
    """
    Tool for efficiently computing the mean and stdev of arrays.
    The arrays themselves are not stored separately,
    instead, they are accumulated.
    """

    def __init__(self):
        """
        ``__init__(...)``: Initialize the RunningStat.

        In the beginning, the number of arrays is 0,
        and the sum and the sum of squares are set as NaN.
        """
        # self.sum = np.zeros(shape, dtype='float32')
        # self.sumsq = np.full(shape, eps, dtype='float32')
        # self.count = eps
        self.reset()

    def reset(self):
        """
        Reset the RunningStat to its initial state.
        """
        self._sum = float("nan")
        self._sumsq = float("nan")
        self._count = 0

    def _increment(self, s, ssq, c):
        # self.sum += s
        # self.sumsq += ssq
        # self.count += c
        if self._count == 0:
            self._sum = np.array(s, dtype="float32")
            self._sumsq = np.array(ssq, dtype="float32")
        else:
            self._sum += s
            self._sumsq += ssq
        self._count += c

    @property
    def count(self) -> int:
        """
        Get the number of arrays accumulated.
        """
        return self._count

    @property
    def sum(self) -> np.ndarray:
        """
        Get the sum of all accumulated arrays.
        """
        return self._sum

    @property
    def sum_of_squares(self) -> np.ndarray:
        """
        Get the sum of squares of all accumulated arrays.
        """
        return self._sumsq

    @property
    def mean(self) -> np.ndarray:
        """
        Get the mean of all accumulated arrays.
        """
        return self._sum / self._count

    @property
    def stdev(self) -> np.ndarray:
        """
        Get the standard deviation of all accumulated arrays.
        """
        return np.sqrt(np.maximum(self._sumsq / self._count - np.square(self.mean), 1e-2))

    # def _set_from_init(self, init_mean, init_std, init_count):
    #    init_mean = np.asarray(init_mean, dtype='float32')
    #    init_std = np.asarray(init_std, dtype='float32')
    #    self._sum = init_mean * init_count
    #    self._sumsq = (np.square(init_mean) + np.square(init_std)) * init_count
    #    self._count = init_count

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
                self._increment(x.sum, x.sum_of_squares, x.count)
        else:
            self._increment(x, np.square(x), 1)

    def normalize(self, x: Union[np.ndarray, list]) -> np.ndarray:
        """
        Normalize the array x according to the accumulated stats.
        """
        x = np.array(x, dtype="float32")
        x -= self.mean
        x /= self.stdev
        return x

    def __copy__(self):
        return deepcopy(self)

    def __get_repr(self):
        return "<RunningStat, count: " + str(self._count) + ">"

    def __str__(self):
        return self.__get_repr()

    def __repr__(self):
        return self.__get_repr()
