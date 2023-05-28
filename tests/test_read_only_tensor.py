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

import pytest
import torch

import evotorch.tools as ett
from evotorch import testing


def test_read_only_tensor():
    x = ett.make_tensor([-3, -2, -1, 0, 1, 2, 3], dtype="float32")
    y = ett.as_read_only_tensor(x)

    # Ensure that x and y are sharing memory (even though x is a regular tensor and y is a read-only tensor)
    assert ett.storage_ptr(x) == ett.storage_ptr(y)

    clamped = torch.clamp(x, -1.0, 1.0)
    clamped2 = torch.clamp(y, -1.0, 1.0)

    # Ensure that a torch function gives the same result on both x and y
    testing.assert_allclose(clamped2, clamped, atol=0.00001)

    # A ReadOnlyTensor disallows any torch method whose name ends with '_'
    with pytest.raises(AttributeError):
        y.normal_()

    # A ReadOnlyTensor fails when addressed via the `out` keyword of a torch function
    with pytest.raises(TypeError):
        torch.clamp(y, -1.0, 1.0, out=y)

    # A ReadOnlyTensor fails when one tries to modify it
    with pytest.raises(TypeError):
        y += 1
    with pytest.raises(TypeError):
        y[:] = 0

    # The clone of a ReadOnlyTensor is not read-only anymore
    clone = y.clone()
    assert not isinstance(clone, ett.ReadOnlyTensor)
