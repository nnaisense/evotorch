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


from contextlib import suppress
from typing import Optional


def check_version(mdl, n: int) -> Optional[tuple]:
    """
    Get the leftmost integers from a module's reported `__version__`.

    If the Python module does not report any version, or its reported version
    is not in the expected format (i.e. does not look like integers separated
    by dots) then this function returns None.

    For example, if a module named `mymodule` has a `__version__` attribute
    "1.2.3.4", calling `check_version(mymodule, 2)` will return the tuple
    `(1, 2)`.

    If the `__version__` of a module named `mymodule2` is, for example,
    "1.2", and this function was called with `check_version(mymodule2, 3)`,
    then, the result will be None because 3 integers were requested from
    a 2-element version string.

    If requested items in the `__version__` string of a module cannot be parsed
    as integers, the result will be None.

    The fact that this function does not raise exceptions with incompatible
    version strings allows one to define version-specific behaviors for a
    library, while still ensuring that the execution will continue if the
    library's reported version is differently formatted than expected.

    Args:
        mdl: The module object whose version is being queried.
        n: Number of items that will be retrieved from the version string.
    Returns:
        A tuple of integers if the version was successfully retrieved;
        None otherwise.
    """

    result = None

    if n <= 0:
        raise ValueError(f"The argument `n` was expected as a positive integer. However, the value of `n` is {n}.")

    if hasattr(mdl, "__version__"):
        version = mdl.__version__
        if isinstance(version, str):
            version_parts = version.split(".")
            if len(version_parts) >= n:
                with suppress(Exception):
                    integers = []
                    for i in range(n):
                        integers.append(int(version_parts[i]))
                    result = tuple(integers)

    return result
