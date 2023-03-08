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

import random
from pathlib import Path

import numpy as np
import ray

SEED = 0


def pytest_sessionstart(session):
    random.seed(SEED)
    np.random.seed(SEED)

    ray.init(
        num_cpus=1,
        log_to_driver=False,
        local_mode=True,
        include_dashboard=False,
        object_store_memory=256 * 1024**2,
        _memory=512 * 1024**2,
        _redis_max_memory=256 * 1024**2,
        _system_config={
            "object_timeout_milliseconds": 200,
            # "num_heartbeats_timeout": 10,
            "object_store_full_delay_ms": 100,
        },
    )


def pytest_sessionfinish(session, exitstatus):
    ray.shutdown()
