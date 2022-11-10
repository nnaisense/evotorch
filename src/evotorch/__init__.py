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

"""Top-level package for evotorch."""

# flake8: noqa
# isort: off

# Import the top-import requests, if any
import os
import importlib

_top_imports = (i.strip() for i in os.getenv("EVOTORCH_TOP_IMPORTS", "").split(","))
for _top_import in _top_imports:
    if len(_top_import) > 0:
        importlib.import_module(_top_import)

# Import the subpackages of EvoTorch
from . import tools
from . import core
from .core import Problem, Solution, SolutionBatch

# isort: on

from . import algorithms, decorators, distributions, logging, neuroevolution, optimizers, testing

__all__ = (
    "__version__",
    "__author__",
    "__email__",
    "Problem",
    "Solution",
    "SolutionBatch",
    "algorithms",
    "core",
    "decorators",
    "distributions",
    "logging",
    "neuroevolution",
    "optimizers",
    "testing",
    "tools",
)
__author__ = "Nihat Engin Toklu, Timothy Atkinson, Vojtech Micka, Rupesh Kumar Srivastava"
__email__ = "engin@nnaisense.com, timothy@nnaisense.com, vojtech@nnaisense.com, rupesh@nnaisense.com"

try:
    from .__version import __version__ as __version__
except ImportError:
    import sys

    print(
        "Please install the package to ensure correct behavior.\nFrom root folder:\n\tpip install -e .", file=sys.stderr
    )
    __version__ = "undefined"
