# Copyright 2025 NNAISENSE SA
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
from torch import FloatTensor
from torch.func import vmap

from evotorch.testing import assert_allclose
from evotorch.tools import TensorFrame, make_tensor


def test_sorting():
    tolerance = 1e-4

    tbl = TensorFrame(
        dict(
            A=[3, 7, 9, 5],
            B=[12, 14, 15, 13],
        ),
    )
    sorted_tbl = tbl.sort_values(by="A")
    assert_allclose(sorted_tbl.B, [12, 13, 14, 15], atol=tolerance)


def test_nlargest_and_nsmallest():
    tolerance = 1e-4

    tbl = TensorFrame(
        dict(
            A=[3, 7, 9, 5],
            B=[12, 14, 15, 13],
        ),
    )

    smallest = tbl.nsmallest(2, "A")
    assert len(smallest) == 2
    assert_allclose(smallest.A, [3, 5], atol=tolerance)
    assert_allclose(smallest.B, [12, 13], atol=tolerance)

    largest = tbl.nlargest(2, "A")
    assert len(largest) == 2
    assert_allclose(largest.A, [9, 7], atol=tolerance)
    assert_allclose(largest.B, [15, 14], atol=tolerance)


def test_inplace_modification():
    tolerance = 1e-4

    source_X = FloatTensor([1, 2, 3, 4, 5])

    original_X = source_X.clone()
    original_Y = original_X * 10

    tbl = TensorFrame(dict(X=original_X, Y=original_Y))

    tbl.pick[1:4, "X"] = FloatTensor([-2, -3, -4])
    assert_allclose(tbl.X, [1, -2, -3, -4, 5], atol=tolerance)
    assert_allclose(original_X, source_X, atol=tolerance)


@pytest.mark.parametrize("rhs_as_frame", [False, True])
def test_inplace_modification_multicolumn(rhs_as_frame: bool):
    tolerance = 1e-4

    source_X = FloatTensor([1, 2, 3, 4, 5])
    source_Y = source_X * 10

    original_X = source_X.clone()
    original_Y = source_Y.clone()

    tbl = TensorFrame(dict(X=original_X, Y=original_Y))

    prepare_rhs = TensorFrame if rhs_as_frame else dict

    tbl.pick[1:4, ["X", "Y"]] = prepare_rhs(
        {
            "X": FloatTensor([-2, -3, -4]),
            "Y": FloatTensor([-20, -30, -40]),
        }
    )

    assert_allclose(tbl.X, [1, -2, -3, -4, 5], atol=tolerance)
    assert_allclose(tbl.Y, [10, -20, -30, -40, 50], atol=tolerance)

    assert_allclose(original_X, source_X, atol=tolerance)
    assert_allclose(original_Y, source_Y, atol=tolerance)


def test_batched_operations():
    tolerance = 1e-4

    batched_X = FloatTensor(
        [
            [
                [2, 3],
                [9, 7],
                [1, 8],
            ],
            [
                [8, 5],
                [10, 11],
                [6, 7],
            ],
        ]
    )

    batched_Y = FloatTensor(
        [
            [1000, 2000, 3000],
            [4000, 5000, 6000],
        ]
    )

    def run(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        tbl = TensorFrame(dict(X=x, Y=y, Z=True))

        def do_for_each_row(row: dict) -> dict:
            x = row["X"]
            y = row["Y"]
            z = row["Z"]
            assert x.shape == (batched_X.shape[-1],)
            assert y.shape == tuple()
            assert z.shape == tuple()
            return {"OUT": z * (torch.max(x) + y)}

        tbl = tbl.each(do_for_each_row)
        return tbl.OUT

    output = vmap(run, in_dims=(0, 0))(batched_X, batched_Y)
    desired = vmap(vmap(torch.max))(batched_X) + batched_Y
    assert_allclose(output, desired, atol=tolerance)


@pytest.mark.parametrize("use_join", [False, True])
def test_hstack_without_override(use_join: bool):
    tolerance = 1e-4

    tbl1 = TensorFrame(
        dict(
            A=[1, 2, 3],
            B=[4, 5, 6],
        ),
    )

    tbl2 = TensorFrame(dict(C=[10, 20, 30]))

    if use_join:
        stacked = tbl1.join(tbl2)
    else:
        stacked = tbl1.hstack(tbl2)

    assert stacked.columns == ["A", "B", "C"]
    assert_allclose(stacked.A, [1, 2, 3], atol=tolerance)
    assert_allclose(stacked.B, [4, 5, 6], atol=tolerance)
    assert_allclose(stacked.C, [10, 20, 30], atol=tolerance)


@pytest.mark.parametrize("override", [False, True])
def test_hstack_with_override(override: bool):
    tolerance = 1e-4

    tbl1 = TensorFrame(
        dict(
            A=[1, 2, 3],
            B=[4, 5, 6],
        ),
    )

    tbl2 = TensorFrame(
        dict(
            B=[-10, -20, -30],
            C=[10, 20, 30],
        ),
    )

    got_value_error = False
    try:
        stacked = tbl1.hstack(tbl2, override=override)
    except ValueError:
        got_value_error = True

    if override:
        assert stacked.columns == ["A", "B", "C"]
        assert_allclose(stacked.A, [1, 2, 3], atol=tolerance)
        assert_allclose(stacked.B, [-10, -20, -30], atol=tolerance)
        assert_allclose(stacked.C, [10, 20, 30], atol=tolerance)
    else:
        assert got_value_error


def test_vstack():
    tolerance = 1e-4

    tbl1 = TensorFrame(
        dict(
            A=[10, 11, 12],
            B=[100, 101, 102],
        ),
    )

    tbl2 = TensorFrame(
        dict(
            A=[13, 14],
            B=[103, 104],
        ),
    )

    stacked = tbl1.vstack(tbl2)

    assert len(stacked) == len(tbl1) + len(tbl2)
    assert_allclose(stacked.A, [10, 11, 12, 13, 14], atol=tolerance)
    assert_allclose(stacked.B, [100, 101, 102, 103, 104], atol=tolerance)


def test_vstack_multidim():
    tolerance = 1e-4

    tbl1 = TensorFrame(
        dict(
            A=[10, 11, 12],
            B=[
                [100, -100],
                [101, -101],
                [102, -102],
            ],
        ),
    )

    tbl2 = TensorFrame(
        dict(
            A=[13, 14, 15],
            B=[
                [103, -103],
                [104, -104],
                [105, -105],
            ],
        ),
    )

    stacked = tbl1.vstack(tbl2)

    assert len(stacked) == len(tbl1) + len(tbl2)
    assert_allclose(stacked.A, [10, 11, 12, 13, 14, 15], atol=tolerance)
    assert_allclose(
        stacked.B,
        [
            [100, -100],
            [101, -101],
            [102, -102],
            [103, -103],
            [104, -104],
            [105, -105],
        ],
        atol=tolerance,
    )


def test_vstack_failure_because_of_ndim_mismatch():
    tbl1 = TensorFrame(
        dict(
            A=[1, 2, 3],
            B=[4, 5, 6],
        )
    )

    tbl2 = TensorFrame(
        dict(
            A=[1, 2, 3],
            B=[
                [4, 5],
                [6, 7],
                [8, 9],
            ],
        )
    )

    with pytest.raises(ValueError):
        tbl1.vstack(tbl2)


def test_vstack_failure_because_of_column_mismatch():
    tbl1 = TensorFrame(
        dict(
            A=[1, 2, 3],
            B=[4, 5, 6],
        )
    )

    tbl2 = TensorFrame(
        dict(
            A=[1, 2, 3],
            C=[11, 22, 33],
        )
    )

    with pytest.raises(ValueError):
        tbl1.vstack(tbl2)


def test_picking_and_slicing():
    tolerance = 1e-4

    tbl = TensorFrame(
        dict(
            A=[1, 2, 3, 4],
            B=[
                [10, 20],
                [30, 40],
                [50, 60],
                [70, 80],
            ],
            C=[-1, -2, -3, -4],
        ),
    )

    subtable = tbl.pick[[1, 3]]
    assert len(subtable) == 2
    assert subtable.columns == ["A", "B", "C"]
    assert_allclose(subtable.A, [2, 4], atol=tolerance)
    assert_allclose(subtable.B, [[30, 40], [70, 80]], atol=tolerance)
    assert_allclose(subtable.C, [-2, -4], atol=tolerance)

    subtable = tbl.pick[[1, 3], "A"]
    assert len(subtable) == 2
    assert subtable.columns == ["A"]
    assert_allclose(subtable.A, [2, 4], atol=tolerance)

    subtable = tbl.pick[[1, 3], ["A", "C"]]
    assert len(subtable) == 2
    assert subtable.columns == ["A", "C"]
    assert_allclose(subtable.A, [2, 4], atol=tolerance)
    assert_allclose(subtable.C, [-2, -4], atol=tolerance)

    for slicer in (slice(None, 2, None), tbl.C > -3):
        sliced = tbl.pick[slicer, "A"]
        assert sliced.columns == ["A"]
        assert_allclose(sliced.A, [1, 2], atol=tolerance)

        sliced = tbl.pick[slicer, ["A", "B"]]
        assert sliced.columns == ["A", "B"]
        assert_allclose(sliced.A, [1, 2], atol=tolerance)
        assert_allclose(sliced.B, [[10, 20], [30, 40]], atol=tolerance)

        sliced = tbl.pick[slicer, ["A", "C"]]
        assert sliced.columns == ["A", "C"]
        assert_allclose(sliced.A, [1, 2], atol=tolerance)
        assert_allclose(sliced.C, [-1, -2], atol=tolerance)

        for column_slicer in (["A", "B", "C"], slice(None, None, None)):
            sliced = tbl.pick[slicer, column_slicer]
            assert sliced.columns == ["A", "B", "C"]
            assert_allclose(sliced.A, [1, 2], atol=tolerance)
            assert_allclose(sliced.B, [[10, 20], [30, 40]], atol=tolerance)
            assert_allclose(sliced.C, [-1, -2], atol=tolerance)


def test_read_only():
    tbl = TensorFrame(dict(A=[1, 2, 3]))
    tbl["B"] = 4
    assert tbl.columns == ["A", "B"]
    assert not tbl.is_read_only

    tbl = tbl.get_read_only_view()
    with pytest.raises(TypeError):
        tbl["C"] = 5

    cloned_tbl = tbl.clone()
    assert not cloned_tbl.is_read_only
    cloned_tbl["C"] = 5
    assert cloned_tbl.columns == ["A", "B", "C"]
    cloned_tbl["A"] *= 2

    objs = make_tensor([None], dtype=object)
    objs[0] = cloned_tbl
    assert objs[0].is_read_only

    with pytest.raises(TypeError):
        objs[0]["D"] = 10

    with pytest.raises(TypeError):
        objs[0]["A"][:] = 2.0


def test_with_columns():
    tolerance = 1e-4

    tbl = TensorFrame(dict(A=[1, 2, 3]))
    tbl = tbl.with_columns(
        B=[4, 5, 6],
        C=7,
    )
    assert set(tbl.columns) == set(["A", "B", "C"])
    assert_allclose(tbl.B, [4, 5, 6], atol=tolerance)
    assert_allclose(tbl.C, [7, 7, 7], atol=tolerance)

    tbl = tbl.get_read_only_view().with_columns(
        A=[10, 20, 30],
        Z=[100, 200, 300],
    )

    assert tbl.is_read_only
    assert set(tbl.columns) == set(["A", "B", "C", "Z"])
    assert_allclose(tbl.A, [10, 20, 30], atol=tolerance)
    assert_allclose(tbl.Z, [100, 200, 300], atol=tolerance)
