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

"""This namespace various RL-specific utilities."""
from typing import Iterable

import gym
import torch
from torch import nn

from .runningstat import RunningStat


def _accumulate(values: Iterable, method: str):
    if method == "sum":
        return sum(values)
    elif method == "min":
        return min(values)
    elif method == "max":
        return max(values)
    elif method == "avg":
        total = None
        count = 0
        for i, value in enumerate(values):
            count += 1
            if i == 0:
                total = value
            else:
                total += value
        return total / count


def _accumulate_across_dicts(dicts: Iterable[dict], key: str, method: str):
    values = []
    for d in dicts:
        values.append(d[key])
    return _accumulate(values, method)


def _accumulate_all_across_dicts(dicts: Iterable[dict], keys_and_methods: dict):
    result = {}
    for key, method in keys_and_methods.items():
        result[key] = _accumulate_across_dicts(dicts, key, method)
    return result


def reset_env(env: gym.Env) -> Iterable:
    """
    Reset a gym environment.

    For gym 1.0, the plan is to have a `reset(...)` method which returns
    a two-element tuple `(observation, info)` where `info` is an object
    providing any additional information regarding the initial state of
    the agent. However, the old (pre 1.0) gym API (and some environments
    which were written with old gym compatibility in mind) has (or have)
    a `reset(...)` method which returns a single object that is the
    initial observation.
    With the assumption that the observation space of the environment
    is NOT tuple, this function can work with both pre-1.0 and (hopefully)
    after-1.0 versions of gym, and always returns the initial observation.

    Please do not use this function on environments whose observation
    spaces or tuples, because then this function cannot distinguish between
    environments whose `reset(...)` methods return a tuple and environments
    whose `reset(...)` methods return a single observation object but that
    observation object is a tuple.

    Args:
        env: The gym environment which will be reset.
    Returns:
        The initial observation
    """
    result = env.reset()
    if isinstance(result, tuple) and (len(result) == 2):
        result = result[0]
    return result


def take_step_in_env(env: gym.Env, action: Iterable) -> tuple:
    """
    Take a step in the gym environment.
    Taking a step means performing the action provided via the arguments.

    For gym 1.0, the plan is to have a `step(...)` method which returns a
    5-elements tuple containing `observation`, `reward`, `terminated`,
    `truncated`, `info` where `terminated` is a boolean indicating whether
    or not the episode is terminated because of the actions taken within the
    environment, and `truncated` is a boolean indicating whether or not the
    episode is finished because the time limit is reached.
    However, the old (pre 1.0) gym API (and some environments which were
    written with old gym compatibility in mind) has (or have) a `step(...)`
    method which returns 4 elements: `observation`, `reward`, `done`, `info`
    where `done` is a boolean indicating whether or not the episode is
    "done", either because of termination or because of truncation.
    This function can work with both pre-1.0 and (hopefully) after-1.0
    versions of gym, and always returns the 4-element tuple as its result.

    Args:
        env: The gym environment in which the given action will be performed.
    Returns:
        A tuple in the form `(observation, reward, done, info)` where
        `observation` is the observation received after performing the action,
        `reward` is the amount of reward gained,
        `done` is a boolean value indicating whether or not the episode has
        ended, and
        `info` is additional information (usually as a dictionary).
    """
    result = env.step(action)
    if isinstance(result, tuple):
        n = len(result)
        if n == 4:
            observation, reward, done, info = result
        elif n == 5:
            observation, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            raise ValueError(
                f"The result of the `step(...)` method of the gym environment"
                f" was expected as a tuple of length 4 or 5."
                f" However, the received result is {repr(result)}, which is"
                f" of length {len(result)}."
            )
    else:
        raise TypeError(
            f"The result of the `step(...)` method of the gym environment"
            f" was expected as a tuple of length 4 or 5."
            f" However, the received result is {repr(result)}, which is"
            f" of type {type(result)}."
        )
    return observation, reward, done, info


class ObsNormLayer(nn.Module):
    """Observation normalization layer for a policy network"""

    def __init__(self, stats: RunningStat, trainable_stats: bool):
        """`__init__(...)`: Initialize the observation normalization layer

        Args:
            stats: The RunninStat object storing the mean and stdev of
                all of the observations.
            trainable_stats: Whether or not the normalization data
                are to be stored as trainable parameters.
        """
        nn.Module.__init__(self)

        mean = torch.tensor(stats.mean, dtype=torch.float32)
        stdev = torch.tensor(stats.stdev, dtype=torch.float32)

        if trainable_stats:
            self.obs_mean = nn.Parameter(mean)
            self.obs_stdev = nn.Parameter(stdev)
        else:
            self.obs_mean = mean
            self.obs_stdev = stdev

    def forward(self, x):
        x = x - self.obs_mean
        x = x / self.obs_stdev
        return x


class ActClipLayer(nn.Module):
    def __init__(self, box: gym.spaces.Box):
        nn.Module.__init__(self)

        self.lb = torch.as_tensor(box.low, dtype=torch.float32)
        self.ub = torch.as_tensor(box.high, dtype=torch.float32)

    def forward(self, x):
        return torch.min(torch.max(x, self.lb), self.ub)
