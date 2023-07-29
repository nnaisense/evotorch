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

from collections.abc import Mapping
from typing import Optional

import gym as classical_gym
import gymnasium as gym
from gymnasium.envs.registration import register
from gymnasium.spaces import Box


class WrappedHumanoidBulletEnv(gym.Env):
    ID = "WrappedHumanoidBulletEnv-v0"
    ENTRY_POINT = __name__ + ":WrappedHumanoidBulletEnv"
    MAX_EPISODE_STEPS = 1000

    def __init__(self, trajectory_length: Optional[int] = None, render_mode: Optional[str] = None, **kwargs):
        gym.Env.__init__(self)

        if trajectory_length is None:
            trajectory_length = self.MAX_EPISODE_STEPS

        self.__tlimit = trajectory_length
        self.__done = True
        self.__t = 0
        self.__contained_env = classical_gym.make(
            "pybullet_envs:HumanoidBulletEnv-v0", render=(render_mode == "human"), **kwargs
        )

        self.__render_mode = None if render_mode is None else str(render_mode)

        self.observation_space = Box(
            low=self.__contained_env.observation_space.low,
            high=self.__contained_env.observation_space.high,
            dtype=self.__contained_env.observation_space.dtype,
        )
        self.action_space = Box(
            low=self.__contained_env.action_space.low,
            high=self.__contained_env.action_space.high,
            dtype=self.__contained_env.action_space.dtype,
        )
        self.reward_range = (float("-inf"), float("inf"))

    def step(self, action):
        assert not self.__done, "Trying to progress in a finished trajectory"

        step_results = self.__contained_env.step(action)
        num_step_results = len(step_results)

        if num_step_results == 4:
            observation, reward, done, info = step_results
            if isinstance(info, Mapping) and ("TimeLimit.truncated" in info):
                truncated = info["TimeLimit.truncated"]
                terminated = done and (not truncated)
            else:
                truncated = False
                terminated = done
        elif num_step_results == 5:
            observation, reward, terminated, truncated, info = step_results
            done = terminated or truncated
        else:
            assert False, "Unexpected number of returns from the step method"

        self.__t += 1

        if self.__t >= self.__tlimit:
            done = True

        self.__done = done

        reward = sum(self.__contained_env.rewards[1:])

        return observation, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.__done = False
        self.__t = 0
        contained_obs = self.__contained_env.reset(**kwargs)
        if isinstance(contained_obs, tuple):
            result = contained_obs
        else:
            result = contained_obs, {}
        return result

    def render(self, mode: Optional[str] = None, **kwargs):
        if mode is None:
            mode = self.__render_mode
        return self.__contained_env.render(mode=mode, **kwargs)

    def close(self):
        return self.__contained_env.close()

    def seed(self, seed=None):
        return self.__contained_env.seed(seed)

    @property
    def rewards(self):
        if hasattr(self.__contained_env, "rewards"):
            return self.__contained_env.rewards
        else:
            return None

    @property
    def reward(self):
        if hasattr(self.__contained_env, "reward"):
            return self.__contained_env.reward
        else:
            return None

    @property
    def body_xyz(self):
        if hasattr(self.__contained_env.robot, "body_xyz"):
            return self.__contained_env.robot.body_xyz
        else:
            return None

    @property
    def body_rpy(self):
        if hasattr(self.__contained_env.robot, "body_rpy"):
            return self.__contained_env.robot.body_rpy
        else:
            return None

    def camera_adjust(self):
        return self.__contained_env.camera_adjust()

    @property
    def robot(self):
        return self.__contained_env


class TinyTrajHumanoidBulletEnv(WrappedHumanoidBulletEnv):
    ID = "TinyTrajHumanoidBulletEnv-v0"
    ENTRY_POINT = __name__ + ":TinyTrajHumanoidBulletEnv"
    MAX_EPISODE_STEPS = 200


register(
    id=WrappedHumanoidBulletEnv.ID,
    entry_point=WrappedHumanoidBulletEnv.ENTRY_POINT,
    max_episode_steps=WrappedHumanoidBulletEnv.MAX_EPISODE_STEPS,
)


register(
    id=TinyTrajHumanoidBulletEnv.ID,
    entry_point=TinyTrajHumanoidBulletEnv.ENTRY_POINT,
    max_episode_steps=TinyTrajHumanoidBulletEnv.MAX_EPISODE_STEPS,
)
