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


from typing import Type

import pytest
import torch

from evotorch.tools.structures import CBag, CDict, CList, CMemory


def test_cmemory():
    values = torch.arange(10, dtype=torch.int64)

    mem = CMemory(num_keys=5, batch_size=10, dtype=torch.int64)
    mem.data[:] = -1
    keys = torch.randint(0, 5, (10,))
    mem[keys] = values

    equivalent_data = torch.empty(10, 5, dtype=torch.int64)
    equivalent_data[:] = -1
    equivalent_data[torch.tensor(True, dtype=torch.bool).expand(10), keys] = values

    assert mem.batch_shape == torch.Size([10])
    assert mem.batch_ndim == 1
    assert mem.key_shape == torch.Size([])
    assert mem.key_ndim == 0
    assert mem.value_shape == torch.Size([])
    assert mem.value_ndim == 0
    assert equivalent_data.shape == mem.data.shape
    assert torch.all(mem.data == equivalent_data)


def test_multikey_cmemory():
    values = torch.arange(10, dtype=torch.int64)

    mem = CMemory(num_keys=(3, 2), batch_size=10, dtype=torch.int64)
    mem.data[:] = -1
    keys = torch.empty(10, 2, dtype=torch.int64)
    keys[:, 0] = torch.randint(0, 3, (10,))
    keys[:, 1] = torch.randint(0, 2, (10,))
    mem[keys] = values

    equivalent_data = torch.empty(10, 3, 2, dtype=torch.int64)
    equivalent_data[:] = -1
    equivalent_data[torch.tensor(True, dtype=torch.bool).expand(10), keys[:, 0], keys[:, 1]] = values

    assert mem.batch_shape == torch.Size([10])
    assert mem.batch_ndim == 1
    assert mem.key_shape == torch.Size([2])
    assert mem.key_ndim == 1
    assert mem.value_shape == torch.Size([])
    assert mem.value_ndim == 0
    assert equivalent_data.shape == mem.data.shape
    assert torch.all(mem.data == equivalent_data)


def test_matrixstoring_multikey_cmemory():
    values = torch.arange(10, dtype=torch.int64).reshape(-1, 1, 1) * torch.ones(10, 4, 5, dtype=torch.int64)

    mem = CMemory(4, 5, num_keys=(3, 2), batch_size=10, dtype=torch.int64)
    mem.data[:] = -1
    keys = torch.empty(10, 2, dtype=torch.int64)
    keys[:, 0] = torch.randint(0, 3, (10,))
    keys[:, 1] = torch.randint(0, 2, (10,))
    mem[keys] = values

    equivalent_data = torch.empty(10, 3, 2, 4, 5, dtype=torch.int64)
    equivalent_data[:] = -1
    equivalent_data[torch.tensor(True, dtype=torch.bool).expand(10), keys[:, 0], keys[:, 1]] = values

    assert mem.batch_shape == torch.Size([10])
    assert mem.batch_ndim == 1
    assert mem.key_shape == torch.Size([2])
    assert mem.key_ndim == 1
    assert mem.value_shape == torch.Size([4, 5])
    assert mem.value_ndim == 2
    assert equivalent_data.shape == mem.data.shape
    assert torch.all(mem.data == equivalent_data)


@pytest.mark.parametrize("structure_type", [CMemory, CDict, CList])
def test_operations(structure_type: Type):
    kwargs = dict(batch_size=10, dtype=torch.int64)
    if issubclass(structure_type, CList):
        kwargs["max_length"] = 5
    else:
        kwargs["num_keys"] = 5

    mem = structure_type(**kwargs)

    if issubclass(structure_type, CMemory):
        mem.data[:] = -1
    elif issubclass(structure_type, CDict):
        for k in range(5):
            mem.set_([k] * 10, -1)
    elif issubclass(structure_type, CList):
        for _ in range(5):
            mem.append_(-1)
    else:
        assert False, "unrecognized structure type"

    equivalent_data = torch.empty(10, 5, dtype=torch.int64)
    equivalent_data[:] = -1
    each_batch_item = torch.tensor(True, dtype=torch.bool).expand(10)

    def make_keys_mask_values() -> tuple:
        return torch.randint(0, 5, (10,)), torch.randn(10) > 0, torch.randint(0, 10, (10,), dtype=torch.int64)

    keys, mask, values = make_keys_mask_values()
    mem.set_(keys, values, where=mask)
    equivalent_data[each_batch_item, keys] = torch.where(mask, values, torch.tensor(-1, dtype=torch.int64))

    keys, mask, values = make_keys_mask_values()
    mem.add_(keys, values, where=mask)
    equivalent_data[each_batch_item, keys] = torch.where(
        mask,
        equivalent_data[each_batch_item, keys] + values,
        equivalent_data[each_batch_item, keys],
    )

    keys, mask, values = make_keys_mask_values()
    mem.subtract_(keys, values, where=mask)
    equivalent_data[each_batch_item, keys] = torch.where(
        mask,
        equivalent_data[each_batch_item, keys] - values,
        equivalent_data[each_batch_item, keys],
    )

    keys, mask, values = make_keys_mask_values()
    mem.multiply_(keys, values, where=mask)
    equivalent_data[each_batch_item, keys] = torch.where(
        mask,
        equivalent_data[each_batch_item, keys] * values,
        equivalent_data[each_batch_item, keys],
    )

    keys, mask, values = make_keys_mask_values()
    values = torch.where(values <= 0, torch.ones(values.shape, dtype=torch.int64), values)
    mem.divide_(keys, values, where=mask)
    equivalent_data[each_batch_item, keys] = torch.where(
        mask,
        equivalent_data[each_batch_item, keys] * torch.as_tensor(1 / values, dtype=torch.int64),
        equivalent_data[each_batch_item, keys],
    )

    assert torch.all(mem.data == equivalent_data)


def test_clist():
    lst = CList(max_length=3, batch_size=2, dtype=torch.int64)

    lst.append_([1, 2])
    # 1
    # 2
    assert torch.all(lst.length == torch.LongTensor([1, 1]))

    lst.append_([3, 4], where=[True, False])
    # 1 3
    # 2
    assert torch.all(lst.length == torch.LongTensor([2, 1]))

    lst.append_([5, 6], where=[False, True])
    # 1 3
    # 2 6
    assert torch.all(lst.length == torch.LongTensor([2, 2]))

    lst.append_([7, 8])
    # 1 3 7
    # 2 6 8
    assert torch.all(lst.length == torch.LongTensor([3, 3]))
    assert torch.all(lst[[0, 0]] == torch.LongTensor([1, 2]))
    assert torch.all(lst[[1, 1]] == torch.LongTensor([3, 6]))
    assert torch.all(lst[[2, 2]] == torch.LongTensor([7, 8]))
    assert torch.all(lst[[0, 1]] == torch.LongTensor([1, 6]))
    assert torch.all(lst[[-1, 0]] == torch.LongTensor([7, 2]))
    assert torch.all(lst[[1, -2]] == torch.LongTensor([3, 6]))

    popped = lst.popleft_()
    # 3 7
    # 6 8
    assert torch.all(popped == torch.LongTensor([1, 2]))
    assert torch.all(lst.length == torch.LongTensor([2, 2]))

    lst.append_([2, 1])
    # 3 7 2
    # 6 8 1
    assert torch.all(lst.length == torch.LongTensor([3, 3]))
    assert torch.all(lst[[0, 0]] == torch.LongTensor([3, 6]))
    assert torch.all(lst[[1, 1]] == torch.LongTensor([7, 8]))
    assert torch.all(lst[[2, 2]] == torch.LongTensor([2, 1]))
    assert torch.all(lst[[-3, -3]] == torch.LongTensor([3, 6]))
    assert torch.all(lst[[-2, -2]] == torch.LongTensor([7, 8]))
    assert torch.all(lst[[-1, -1]] == torch.LongTensor([2, 1]))

    popped = lst.popleft_(where=[True, False])
    # 7 2
    # 6 8 1
    assert torch.all(lst.length == torch.LongTensor([2, 3]))
    assert int(popped[0]) == 3

    popped = lst.popleft_(where=[False, True])
    # 7 2
    # 8 1
    assert torch.all(lst.length == torch.LongTensor([2, 2]))
    assert int(popped[1]) == 6
    assert torch.all(lst[[0, 0]] == torch.LongTensor([7, 8]))
    assert torch.all(lst[[1, 1]] == torch.LongTensor([2, 1]))
    assert torch.all(lst[[-2, -2]] == torch.LongTensor([7, 8]))
    assert torch.all(lst[[-1, -1]] == torch.LongTensor([2, 1]))

    popped = lst.pop_(where=[True, False])
    # 7
    # 8 1
    assert torch.all(lst.length == torch.LongTensor([1, 2]))
    assert int(popped[0]) == 2

    popped = lst.pop_()
    # <empty>
    # 8
    assert torch.all(lst.length == torch.LongTensor([0, 1]))
    default = torch.LongTensor([-11, -12])
    assert torch.all(lst.get([0, 0], default=default) == torch.LongTensor([-11, 8]))
    assert torch.all(lst.get([-1, -1], default=default) == torch.LongTensor([-11, 8]))


def test_cbag():
    values_for_a = [0, 1, 9, 7, 6]
    values_for_b = [2, 3, 4, 5, 8]
    n = len(values_for_a)
    max_value = max(max(values_for_a), max(values_for_b))

    bag = CBag(max_length=n, value_range=(0, max_value + 1), batch_size=2, dtype=torch.int64)

    for i in range(n):
        element_a = values_for_a[i]
        element_b = values_for_b[i]
        t = torch.LongTensor([element_a, element_b])
        bag.push_(t)

    popped_from_a = set()
    popped_from_b = set()

    for _ in range(n):
        popped = bag.pop_()
        element_a = int(popped[0])
        element_b = int(popped[1])
        assert element_a not in popped_from_a
        assert element_b not in popped_from_b
        popped_from_a.add(element_a)
        popped_from_b.add(element_b)

    assert popped_from_a == set(values_for_a)
    assert popped_from_b == set(values_for_b)
