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

from typing import Any

import pytest

from evotorch.tools.recursiveprintable import RecursivePrintable


# Test mapping is correclty printed
class Dict(RecursivePrintable, dict):
    pass


# Test iterables are correclty printed
class List(RecursivePrintable, list):
    pass


# Test that the repr is correctly used when printing
class Foo:
    def __init__(self, x: Any):
        self.x = x

    def __repr__(self):
        return f"Foo<{self.x}>"


@pytest.mark.parametrize(
    "item, expected",
    [
        (dict(a=1, b=2), "{'a': 1, 'b': 2}"),
        (Dict(a=1, b=2), "Dict({'a': 1, 'b': 2})"),
        (Foo(42), "Foo<42>"),
        (Foo(Foo(42)), "Foo<Foo<42>>"),
        ([1, 2, 3], "[1, 2, 3]"),
        (List([1, 2, 3]), "List([1, 2, 3])"),
        ("str", "'str'"),
    ],
)
def test_recursiveprintable(item, expected):
    d = Dict(a=item)

    assert repr(d) == str(d) == d.to_string() == f"Dict({{'a': {expected}}})"


def test_recursiveprintable_raises():
    class Unsupported(RecursivePrintable):
        pass

    with pytest.raises(NotImplementedError):
        Unsupported().to_string()


@pytest.mark.parametrize(
    "d, depth, expected",
    [
        (Dict(a=Dict(a=Dict(a=0))), 0, "<...>"),
        (Dict(a=Dict(a=Dict(a=0))), 1, "Dict({'a': <...>})"),
        (Dict(a=Dict(a=Dict(a=0))), 2, "Dict({'a': Dict({'a': <...>})})"),
        (Dict(a=Dict(a=Dict(a=0))), 3, "Dict({'a': Dict({'a': Dict({'a': 0})})})"),
        (
            Dict(a=Dict(a=Dict(a=0)), b=Dict(b=(Dict(b=1)))),
            0,
            "<...>",
        ),
        (
            Dict(a=Dict(a=Dict(a=0)), b=Dict(b=(Dict(b=1)))),
            1,
            "Dict({'a': <...>, 'b': <...>})",
        ),
        (
            Dict(a=Dict(a=Dict(a=0)), b=Dict(b=(Dict(b=1)))),
            2,
            "Dict({'a': Dict({'a': <...>}), 'b': Dict({'b': <...>})})",
        ),
        (
            Dict(a=Dict(a=Dict(a=0)), b=Dict(b=(Dict(b=1)))),
            3,
            "Dict({'a': Dict({'a': Dict({'a': 0})}), 'b': Dict({'b': Dict({'b': 1})})})",
        ),
    ],
)
def test_maxdepth(d, depth, expected):
    assert d.to_string(max_depth=depth) == expected
