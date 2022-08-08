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

"""
This module contains the Hook class, which is used for event handling,
and for defining additional behaviors to the class instances which own
the Hook.
"""

from collections.abc import Mapping, MutableSequence
from typing import Any, Callable, Iterable, Optional, Union


class Hook(MutableSequence):
    """
    A Hook stores a list of callable objects to be called for handling
    certain events. A Hook itself is callable, which invokes the callables
    stored in its list. If the callables stored by the Hook return list-like
    objects or dict-like objects, their returned results are accumulated,
    and then those accumulated results are finally returned by the Hook.
    """

    def __init__(
        self,
        callables: Optional[Iterable[Callable]] = None,
        *,
        args: Optional[Iterable] = None,
        kwargs: Optional[Mapping] = None,
    ):
        """
        Initialize the Hook.

        Args:
            callables: A sequence of callables to be stored by the Hook.
            args: Positional arguments which, when the Hook is called,
                are to be passed to every callable stored by the Hook.
                Please note that these positional arguments will be passed
                as the leftmost arguments, and, the other positional
                arguments passed via the `__call__(...)` method of the
                Hook will be added to the right of these arguments.
            kwargs: Keyword arguments which, when the Hook is called,
                are to be passed to every callable stored by the Hook.
                Please note that these keyword arguments could be overriden
                by the keyword arguments passed via the `__call__(...)`
                method of the Hook.
        """
        self._funcs: list = [] if callables is None else list(callables)
        self._args: list = [] if args is None else list(args)
        self._kwargs: dict = {} if kwargs is None else dict(kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Optional[Union[dict, list]]:
        """
        Call every callable object stored by the Hook.
        The results of the stored callable objects (which can be dict-like
        or list-like objects) are accumulated and finally returned.

        Args:
            args: Additional positional arguments to be passed to the stored
                callables.
            kwargs: Additional keyword arguments to be passed to the stored
                keyword arguments.
        """

        all_args = []
        all_args.extend(self._args)
        all_args.extend(args)

        all_kwargs = {}
        all_kwargs.update(self._kwargs)
        all_kwargs.update(kwargs)

        result: Optional[Union[dict, list]] = None

        for f in self._funcs:
            tmp = f(*all_args, **all_kwargs)
            if tmp is not None:
                if isinstance(tmp, Mapping):
                    if result is None:
                        result = dict(tmp)
                    elif isinstance(result, list):
                        raise TypeError(
                            f"The function {f} returned a dict-like object."
                            f" However, previous function(s) in this hook had returned list-like object(s)."
                            f" Such incompatible results cannot be accumulated."
                        )
                    elif isinstance(result, dict):
                        result.update(tmp)
                    else:
                        raise RuntimeError
                elif isinstance(tmp, Iterable):
                    if result is None:
                        result = list(tmp)
                    elif isinstance(result, list):
                        result.extend(tmp)
                    elif isinstance(result, dict):
                        raise TypeError(
                            f"The function {f} returned a list-like object."
                            f" However, previous function(s) in this hook had returned dict-like object(s)."
                            f" Such incompatible results cannot be accumulated."
                        )
                    else:
                        raise RuntimeError
                else:
                    raise TypeError(
                        f"Expected the function {f} to return None, or a dict-like object, or a list-like object."
                        f" However, the function returned an object of type {repr(type(tmp))}."
                    )

        return result

    def accumulate_dict(self, *args: Any, **kwargs: Any) -> Optional[Union[dict, list]]:
        result = self(*args, **kwargs)
        if result is None:
            return {}
        elif isinstance(result, Mapping):
            return result
        else:
            raise TypeError(
                f"Expected the functions in this hook to accumulate"
                f" dictionary-like objects. Instead, accumulated"
                f" an object of type {type(result)}."
                f" Hint: are the functions registered in this hook"
                f" returning non-dictionary iterables?"
            )

    def accumulate_sequence(self, *args: Any, **kwargs: Any) -> Optional[Union[dict, list]]:
        result = self(*args, **kwargs)
        if result is None:
            return []
        elif isinstance(result, Mapping):
            raise TypeError(
                f"Expected the functions in this hook to accumulate"
                f" sequences (that are NOT dictionaries). Instead, accumulated"
                f" a dict-like object of type {type(result)}."
                f" Hint: are the functions registered in this hook"
                f" returning objects with Mapping interface?"
            )
        else:
            return result

    def _to_string(self) -> str:
        init_args = [repr(self._funcs)]

        if len(self._args) > 0:
            init_args.append(f"args={self._args}")

        if len(self._kwargs) > 0:
            init_args.append(f"kwargs={self._kwargs}")

        s_init_args = ", ".join(init_args)

        return f"{type(self).__name__}({s_init_args})"

    def __repr__(self) -> str:
        return self._to_string()

    def __str__(self) -> str:
        return self._to_string()

    def __getitem__(self, i: Union[int, slice]) -> Union[Callable, "Hook"]:
        if isinstance(i, slice):
            return Hook(self._funcs[i], args=self._args, kwargs=self._kwargs)
        else:
            return self._funcs[i]

    def __setitem__(self, i: Union[int, slice], x: Iterable[Callable]):
        self._funcs[i] = x

    def __delitem__(self, i: Union[int, slice]):
        del self._funcs[i]

    def insert(self, i: int, x: Callable):
        self._funcs.insert(i, x)

    def __len__(self) -> int:
        return len(self._funcs)

    @property
    def args(self) -> list:
        """Positional arguments that will be passed to the stored callables"""
        return self._args

    @property
    def kwargs(self) -> dict:
        """Keyword arguments that will be passed to the stored callables"""
        return self._kwargs
