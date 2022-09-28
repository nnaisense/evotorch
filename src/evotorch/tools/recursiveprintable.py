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


from typing import Any, Iterable, Mapping


class RecursivePrintable:
    """
    A base class for making a class printable.

    This base class considers custom container types which can recursively
    contain themselves (even in a cyclic manner). Classes inheriting from
    `RecursivePrintable` will gain a new ready-to-use method named
    `to_string(...)`. This `to_string(...)` method, upon being called,
    checks if the current class is an Iterable or a Mapping, and prints
    the representation accordingly, with a recursion limit to avoid
    `RecursionError`. The methods `__str__(...)` and `__repr__(...)`
    are also defined as aliases of this `to_string` method.
    """

    def to_string(self, *, max_depth: int = 10) -> str:
        if max_depth <= 0:
            return "<...>"

        def item_repr(x: Any) -> str:
            if isinstance(x, RecursivePrintable):
                return x.to_string(max_depth=(max_depth - 1))
            else:
                return repr(x)

        result = []

        def puts(*x: Any):
            for item_of_x in x:
                result.append(str(item_of_x))

        clsname = type(self).__name__
        first_one = True

        if isinstance(self, Mapping):
            puts(clsname, "({")
            for k, v in self.items():
                if first_one:
                    first_one = False
                else:
                    puts(", ")
                puts(item_repr(k), ": ", item_repr(v))
            puts("})")
        elif isinstance(self, Iterable):
            puts(clsname, "([")
            for v in self:
                if first_one:
                    first_one = False
                else:
                    puts(", ")
                puts(item_repr(v))
            puts("])")
        else:
            raise NotImplementedError

        return "".join(result)

    def __str__(self) -> str:
        return self.to_string()

    def __repr__(self) -> str:
        return self.to_string()
