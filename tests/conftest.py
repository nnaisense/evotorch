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

import os
import random
import sys

import numpy as np
import torch

# Insert the path of the mock ray implementation to the path list of Python.
# The mock ray implementation within the directory _mock-site-packages/ does not create remote actors,
# and instead executes the tasks in sequence.
# This mock ray implementation is used as a replacement for the now-obsolete local mode of the actual
# ray library.
_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_TESTS_DIR, "_mock-site-packages"))

import ray  # noqa: E402

SEED = 0


def pytest_sessionstart(session):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    # ray.init()  # Let us not do ray.init() and test if EvoTorch initializes ray by itself


def pytest_sessionfinish(session, exitstatus):
    ray.shutdown()
