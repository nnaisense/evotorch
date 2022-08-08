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

from evotorch.tools.hook import Hook


def test_hook():
    call_order = []

    def f():
        call_order.append("f")

    def g():
        call_order.append("g")

    my_hook = Hook([f, g])
    my_hook()

    assert call_order == ["f", "g"]


def test_hook_with_args():
    call_order = []

    def f(a, b, *, c):
        call_order.append("f")
        assert a == 1
        assert b == 2
        assert c == 3

    def g(a, b, *, c):
        call_order.append("g")
        assert a == 1
        assert b == 2
        assert c == 3

    my_hook = Hook([f, g])
    my_hook(1, 2, c=3)

    assert call_order == ["f", "g"]


def test_hook_with_dict_result():
    call_order = []

    def f():
        call_order.append("f")
        return {"a": 1}

    def h():
        call_order.append("h")
        return None

    def g():
        call_order.append("g")
        return {"b": 2}

    my_hook = Hook([f, h, g])
    result = my_hook()
    assert call_order == ["f", "h", "g"]
    assert set(result.keys()) == {"a", "b"}
    assert result["a"] == 1
    assert result["b"] == 2


def test_hook_with_list_result():
    call_order = []

    def f():
        call_order.append("f")
        return [1]

    def h():
        call_order.append("h")
        return None

    def g():
        call_order.append("g")
        return [2]

    my_hook = Hook([f, h, g])
    result = my_hook()
    assert call_order == ["f", "h", "g"]
    assert result == [1, 2]


def test_hook_with_default_args():
    call_order = []

    def f(a, b, *, c):
        call_order.append("f")
        assert a == 1
        assert b == 2
        assert c == 3

    def g(a, b, *, c):
        call_order.append("g")
        assert a == 1
        assert b == 2
        assert c == 3

    my_hook = Hook([f, g], kwargs={"c": 3})
    my_hook(1, 2)

    assert call_order == ["f", "g"]
