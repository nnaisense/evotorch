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

from typing import Iterable

import pytest
import torch

from evotorch.testing import assert_allclose
from evotorch.tools.ranking import rank


@pytest.mark.parametrize(
    "x",
    [
        [1, 2, 3, 4],
        [4, 3, 2, 1],
        [10, -10, 5, 3],
        [6, 2, 5],
    ],
)
def test_rank(x):
    def do_the_test(x: Iterable, higher_is_better: bool):
        x = torch.tensor(x, dtype=torch.float32)

        if higher_is_better:
            t = x.clone()
        else:
            t = x * -1

        ranks = torch.empty_like(t)
        ranks[t.argsort()] = (torch.arange(len(t)) / (len(t) - 1)) - 0.5

        ranks2 = rank(x, higher_is_better=higher_is_better, ranking_method="centered")

        assert_allclose(ranks, ranks2, atol=0.00001)

    for higher_is_better in (True, False):
        do_the_test(x, higher_is_better)
