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

"""This namespace provides various reinforcement learning utilities."""

from copy import deepcopy
from typing import Any, Iterable, Optional, Union

import gymnasium as gym
import torch
from gymnasium.spaces import Box
from torch import nn

from .misc import device_of_module
from .runningnorm import RunningNorm
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
    Reset a gymnasium environment.

    Even though the `gymnasium` library switched to a new API where the
    `reset()` method returns a tuple `(observation, info)`, this function
    follows the conventions of the classical `gym` library and returns
    only the observation of the newly reset environment.

    Args:
        env: The gymnasium environment which will be reset.
    Returns:
        The initial observation
    """
    result = env.reset()
    if isinstance(result, tuple) and (len(result) == 2):
        result = result[0]
    return result


def take_step_in_env(env: gym.Env, action: Iterable) -> tuple:
    """
    Take a step in the gymnasium environment.
    Taking a step means performing the action provided via the arguments.

    Even though the `gymnasium` library switched to a new API where the
    `step()` method returns a 5-element tuple of the form
    `(observation, reward, terminated, truncated, info)`, this function
    follows the conventions of the classical `gym` library and returns
    a 4-element tuple `(observation, reward, done, info)`.

    Args:
        env: The gymnasium environment in which the action will be performed.
        action: The action to be performed.
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


class ActClipWrapperModule(nn.Module):
    def __init__(self, wrapped_module: nn.Module, obs_space: Box):
        super().__init__()

        device = device_of_module(wrapped_module)

        if not isinstance(obs_space, Box):
            raise TypeError(f"Unrecognized observation space: {obs_space}")

        self.wrapped_module = wrapped_module
        self.register_buffer("_low", torch.from_numpy(obs_space.low).to(device))
        self.register_buffer("_high", torch.from_numpy(obs_space.high).to(device))

    def forward(self, x: torch.Tensor, h: Any = None) -> Union[torch.Tensor, tuple]:
        if h is None:
            result = self.wrapped_module(x)
        else:
            result = self.wrapped_module(x, h)

        if isinstance(result, tuple):
            x, h = result
            got_h = True
        else:
            x = result
            h = None
            got_h = False

        x = torch.max(x, self._low)
        x = torch.min(x, self._high)

        if got_h:
            return x, h
        else:
            return x


class ObsNormWrapperModule(nn.Module):
    def __init__(self, wrapped_module: nn.Module, rn: Union[RunningStat, RunningNorm]):
        super().__init__()

        device = device_of_module(wrapped_module)
        self.wrapped_module = wrapped_module

        with torch.no_grad():
            normalizer = deepcopy(rn.to_layer()).to(device)
        self.normalizer = normalizer

    def forward(self, x: torch.Tensor, h: Any = None) -> Union[torch.Tensor, tuple]:
        x = self.normalizer(x)

        if h is None:
            result = self.wrapped_module(x)
        else:
            result = self.wrapped_module(x, h)

        if isinstance(result, tuple):
            x, h = result
            got_h = True
        else:
            x = result
            h = None
            got_h = False

        if got_h:
            return x, h
        else:
            return x


class AliveBonusScheduleWrapper(gym.Wrapper):
    """
    A Wrapper which awards the agent for being alive in a scheduled manner
    This wrapper is meant to be used for non-vectorized environments.
    """

    def __init__(self, env: gym.Env, alive_bonus_schedule: tuple, **kwargs):
        """
        `__init__(...)`: Initialize the AliveBonusScheduleWrapper.

        Args:
            env: Environment to wrap.
            alive_bonus_schedule: If given as a tuple `(t, b)`, an alive
                bonus `b` will be added onto all the rewards beyond the
                timestep `t`.
                If given as a tuple `(t0, t1, b)`, a partial (linearly
                increasing towards `b`) alive bonus will be added onto
                all the rewards between the timesteps `t0` and `t1`,
                and a full alive bonus (which equals to `b`) will be added
                onto all the rewards beyond the timestep `t1`.
            kwargs: Expected in the form of additional keyword arguments,
                these will be passed to the initialization method of the
                superclass.
        """
        super().__init__(env, **kwargs)
        self.__t: Optional[int] = None

        if len(alive_bonus_schedule) == 3:
            self.__t0, self.__t1, self.__bonus = (
                int(alive_bonus_schedule[0]),
                int(alive_bonus_schedule[1]),
                float(alive_bonus_schedule[2]),
            )
        elif len(alive_bonus_schedule) == 2:
            self.__t0, self.__t1, self.__bonus = (
                int(alive_bonus_schedule[0]),
                int(alive_bonus_schedule[0]),
                float(alive_bonus_schedule[1]),
            )
        else:
            raise ValueError(
                f"The argument `alive_bonus_schedule` was expected to have 2 or 3 elements."
                f" However, its value is {repr(alive_bonus_schedule)} (having {len(alive_bonus_schedule)} elements)."
            )

        if self.__t1 > self.__t0:
            self.__gap = self.__t1 - self.__t0
        else:
            self.__gap = None

    def reset(self, *args, **kwargs):
        self.__t = 0
        return self.env.reset(*args, **kwargs)

    def step(self, action) -> tuple:
        step_result = self.env.step(action)
        self.__t += 1

        observation = step_result[0]
        reward = step_result[1]
        rest = step_result[2:]

        if self.__t >= self.__t1:
            reward = reward + self.__bonus
        elif (self.__gap is not None) and (self.__t >= self.__t0):
            reward = reward + ((self.__t - self.__t0) / self.__gap) * self.__bonus

        return (observation, reward) + rest
