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

"""Utility classes and functions for neural networks"""

__all__ = (
    "ModuleExpectingFlatParameters",
    "MultiLayered",
    "NetParsingError",
    "Policy",
    "RunningNorm",
    "RunningStat",
    "StatefulModule",
    "count_parameters",
    "device_of_module",
    "fill_parameters",
    "functional",
    "layers",
    "make_functional_module",
    "misc",
    "multilayered",
    "parameter_vector",
    "parser",
    "rl",
    "statefulmodule",
    "str_to_net",
    "vecrl",
)

from . import functional, layers, misc, multilayered, parser, rl, statefulmodule, vecrl
from .functional import ModuleExpectingFlatParameters, make_functional_module
from .misc import count_parameters, device_of_module, fill_parameters, parameter_vector
from .multilayered import MultiLayered
from .parser import NetParsingError, str_to_net
from .runningnorm import RunningNorm
from .runningstat import RunningStat
from .statefulmodule import StatefulModule
from .vecrl import Policy
