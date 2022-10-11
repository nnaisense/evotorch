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


import pickle
from argparse import ArgumentParser
from ast import literal_eval
from pathlib import Path
from time import sleep
from typing import Optional, Union

import gym
import numpy as np
import torch
from packaging.version import Version

from evotorch.neuroevolution.net.rl import reset_env, take_step_in_env

new_render_api = Version(gym.__version__) < Version("0.26")


def make_env_for_rendering(*args, **kwargs):
    """
    Initialize a new gym environment with human-mode rendering.

    Beginning with gym 0.26, it is required to specify the rendering mode
    while initializing the environment. If the gym version is newer than
    or equal to 0.26, this function passes the keyword argument
    `render_mode="human"` to `gym.make(...)`.

    Args:
        args: Expected in the form of positional arguments. These are
            passed directly to `gym.make(...)`.
        kwargs: Expected in the form of keyword arguments. These are
            passed directly to `gym.make(...)`.
    Returns:
        The newly made gym environment.
    """
    if new_render_api:
        env_config = {"render_mode": "human"}
    else:
        env_config = {}

    env_config.update(kwargs)
    return gym.make(*args, **env_config)


def make_env_for_recording(*args, **kwargs):
    """
    Initialize a new gym environment with human-mode rendering.

    Beginning with gym 0.26, it is required to specify the rendering mode
    while initializing the environment. If the gym version is newer than
    or equal to 0.26, this function passes the keyword argument
    `render_mode="rgb_array"` to `gym.make(...)`.

    Args:
        args: Expected in the form of positional arguments. These are
            passed directly to `gym.make(...)`.
        kwargs: Expected in the form of keyword arguments. These are
            passed directly to `gym.make(...)`.
    Returns:
        The newly made gym environment.
    """
    if new_render_api:
        env_config = {"render_mode": "rgb_array"}
    else:
        env_config = {}

    env_config.update(kwargs)
    return gym.make(*args, **env_config)


def rgb_array_from_env(env: gym.Env) -> np.ndarray:
    """
    Render the current state of the environment into numpy array.

    Returns:
        The newly made numpy array containing the rendering result.
    """
    if new_render_api:
        return env.render()
    else:
        return env.render(mode="rgb_array")


def str_if_non_empty(s: Optional[str]) -> Optional[str]:
    if (s is None) or (isinstance(s, str) and (s == "")):
        return None
    else:
        return str(s)


def float_if_positive(x: Optional[float]) -> Optional[float]:
    if (x is None) or (x <= 0):
        return None
    else:
        return float(x)


def int_if_positive(x: Optional[int]) -> Optional[int]:
    if (x is None) or (x <= 0):
        return None
    else:
        return int(x)


def dict_if_non_empty(d: Optional[Union[str, dict]]) -> Optional[dict]:
    if d is None:
        return None
    elif isinstance(d, str):
        if d == "":
            return None
        else:
            return dict(literal_eval(d))
    elif isinstance(d, dict):
        return d
    else:
        raise TypeError(f"Object of unexpected type: {d}")


def main(
    fname: Union[str, Path],
    *,
    num_repeats: int = 1,
    t: Optional[float] = None,
    record_prefix: Optional[Union[str, Path]] = None,
    record_period: int = 1,
    extract: Optional[str] = None,
    set_in_env: Optional[str] = None,
    config: Optional[str] = None,
    env_name: Optional[str] = None,
):
    num_repeats = int(num_repeats)
    t = float_if_positive(t)
    record_prefix = str_if_non_empty(record_prefix)
    record_period = int(record_period)
    extract = str_if_non_empty(extract)
    set_in_env = dict_if_non_empty(set_in_env)
    config = dict_if_non_empty(config)
    env_name = str_if_non_empty(env_name)

    with open(fname, "rb") as f:
        loaded = pickle.load(f)

    if env_name is None:
        env_name = loaded["env_name"]
    policy = loaded["policy"]
    kwargs = {}
    if ("BulletEnv" in env_name) and (record_prefix is None) and (not new_render_api):
        kwargs["render"] = True
    if config is not None:
        kwargs.update(config)

    if record_prefix is None:
        env = make_env_for_rendering(env_name, **kwargs)
    else:
        env = make_env_for_recording(env_name, **kwargs)

    if set_in_env is not None:
        for k, v in set_in_env.items():
            setattr(env.unwrapped, k, v)

    def use_policy(observation):
        with torch.no_grad():
            action = policy(torch.as_tensor(observation, dtype=torch.float32)).numpy()
        if isinstance(env.action_space, gym.spaces.Discrete):
            action = np.argmax(action)
        return action

    if record_prefix is None:

        def render():
            env.render()
            if t is not None:
                sleep(t)

    else:
        scene_index = 0
        save_index = 0

        def render():
            nonlocal scene_index, save_index
            from matplotlib import pyplot as plt

            if scene_index % record_period == 0:
                img = rgb_array_from_env(env)
                if extract is not None:
                    extract_parts = extract.split(",")
                    x1 = int(extract_parts[0])
                    y1 = int(extract_parts[1])
                    x2 = int(extract_parts[2])
                    y2 = int(extract_parts[3])
                    img = img[y1:y2, x1:x2]
                scene_fname = "%s_%05d.png" % (record_prefix, save_index)
                plt.imsave(scene_fname, img)
                print(scene_fname)
                save_index += 1

            scene_index += 1

    repeat_iter = 0
    while repeat_iter < num_repeats or num_repeats <= 0:
        cumulative_reward = 0.0
        observation = reset_env(env)
        render()

        while True:
            action = use_policy(observation)
            observation, reward, done, info = take_step_in_env(env, action)
            render()
            cumulative_reward += float(reward)
            if done:
                break

        print(f"Repeat {repeat_iter}: {cumulative_reward}")
        repeat_iter += 1


if __name__ == "__main__":
    parser = ArgumentParser(
        description=(
            "This is a command-line tool for running, visualizing, and optionally recording the saved agents."
            " This tool works with the `.pickle` files saved by the training script named `rl.py`."
        )
    )

    parser.add_argument("fname", type=str, help="Name of the pickle file which stores the agent saved by rl.py")

    parser.add_argument("-n", type=int, default=1, help="Number of episodes over which the agent will be tested")

    parser.add_argument(
        "-t",
        type=float,
        default=-1.0,
        help=(
            "If given as a positive value, this much time, in seconds, will be introduced between each scene."
            " This option will be ignored if a non-empty string is given via the -r argument"
            " (a non-empty string for -r meaning that the recording mode is enabled)"
        ),
    )
    parser.add_argument(
        "-r",
        type=str,
        default="",
        help=(
            "When given as a non-empty string, this script will behave in recording mode."
            " With the recording mode enabled, each scene will be saved as a png file"
            " (instead of being shown to the screen)."
            " The prefix of the name of each created png file is determined by the string given for this argument."
        ),
    )

    parser.add_argument(
        "-p",
        type=int,
        default=1,
        help=(
            "When recording (i.e. when the argument -r is present), this argument determines the period"
            " for saving the scenes."
            " When given as a positive integer N, one scene from every N scenes will be saved."
            " By default, this is 1, which means there will be no skipped scenes."
        ),
    )

    parser.add_argument(
        "--extract",
        type=str,
        default="",
        help=(
            "When provided, this is expected as a string in the form 'x1,y1,x2,y2' where x and y values are integers."
            " In recording mode, this region will be extracted and this extracted region will be saved,"
            " instead of the whole scene."
            " For example, when the argument --extract '10,20,-30,-40' is given, the extracted region"
            " will be bounded by 10 from the left, 20 from the top, 30 from the right, and 40 from the bottom."
        ),
    )

    parser.add_argument(
        "--set",
        type=str,
        default="",
        help=(
            "When provided, this is expected as a Python dictionary."
            " Each item in this Python dictionary will be set as an attribute of the gym environment."
            " For example, for a PyBullet environment (e.g. bullet_envs:HumanoidBulletEnv-v0),"
            ' one might want to give the argument --set \'{"_render_width": 640, "_render_height": 480}\''
            " to set the attributes _render_width and _render_height as 640 and 480 respectively,"
            " therefore configuring the render size to 640x480 while recording."
        ),
    )

    parser.add_argument(
        "--config",
        type=str,
        default="",
        help=(
            "When provided, this is expected as a Python dictionary."
            " Each item in this Python dictionary will be sent to the environment as a keyword argument"
            " during its initialization phase."
        ),
    )

    parser.add_argument(
        "--env-name",
        type=str,
        default="",
        help=(
            "Name of the environment in which the policy will be tested."
            " If not specified, the environment name stored in the loaded pickle file will be used."
        ),
    )

    parsed = parser.parse_args()

    main(
        fname=parsed.fname,
        num_repeats=parsed.n,
        t=parsed.t,
        record_prefix=parsed.r,
        record_period=parsed.p,
        extract=parsed.extract,
        set_in_env=parsed.set,
        config=parsed.config,
        env_name=parsed.env_name,
    )
