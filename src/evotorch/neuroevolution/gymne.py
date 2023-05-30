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

"""This namespace contains the `GymNE` class."""

import pickle
from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional, Union

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from ..core import BoundsPairLike, Solution, SolutionBatch
from ..tools import Device, ReadOnlyTensor
from .neproblem import NEProblem
from .net import RunningStat
from .net.rl import (
    ActClipWrapperModule,
    AliveBonusScheduleWrapper,
    ObsNormWrapperModule,
    _accumulate_all_across_dicts,
    reset_env,
    take_step_in_env,
)
from .net.statefulmodule import ensure_stateful


def ensure_space_types(env: gym.Env) -> None:
    if not isinstance(env.observation_space, gym.spaces.Box):
        raise TypeError(
            f"The gym environment has an unsupported type of observation space: {type(env.observation_space)}."
            f" The only supported observation space type is gym.spaces.Box."
        )
    if not isinstance(env.observation_space, (gym.spaces.Box, gym.spaces.Discrete)):
        raise TypeError(
            f"The gym environment has an unsupported type of action space: {type(env.action_space)}."
            f" Supported action space types are gym.spaces.Box and gym.spaces.Discrete."
        )


def _make_env(env: Union[str, Callable], **kwargs) -> gym.Env:
    if isinstance(env, str):
        return gym.make(env, **kwargs)
    else:
        return env(**kwargs)


class GymNE(NEProblem):
    """
    Representation of a NEProblem where the goal is to maximize
    the total reward obtained in a `gym` environment.
    """

    def __init__(
        self,
        env: Optional[Union[str, Callable]] = None,
        network: Optional[Union[str, nn.Module, Callable[[], nn.Module]]] = None,
        *,
        env_name: Optional[Union[str, Callable]] = None,
        network_args: Optional[dict] = None,
        env_config: Optional[Mapping] = None,
        observation_normalization: bool = False,
        num_episodes: int = 1,
        episode_length: Optional[int] = None,
        decrease_rewards_by: Optional[float] = None,
        alive_bonus_schedule: Optional[tuple] = None,
        action_noise_stdev: Optional[float] = None,
        num_actors: Optional[Union[int, str]] = None,
        actor_config: Optional[dict] = None,
        num_subbatches: Optional[int] = None,
        subbatch_size: Optional[int] = None,
        initial_bounds: Optional[BoundsPairLike] = (-0.00001, 0.00001),
    ):
        """
        `__init__(...)`: Initialize the GymNE.

        Args:
            env: The gym environment to solve. Expected as a Callable
                (maybe a function returning a gym.Env, or maybe a gym.Env
                subclass), or as a string referring to a gym environment
                ID (e.g. "Ant-v4", "Humanoid-v4", etc.).
            network: A network structure string, or a Callable (which can be
                a class inheriting from `torch.nn.Module`, or a function
                which returns a `torch.nn.Module` instance), or an instance
                of `torch.nn.Module`.
                The object provided here determines the structure of the
                neural network policy whose parameters will be evolved.
                A network structure string is a string which can be processed
                by `evotorch.neuroevolution.net.str_to_net(...)`.
                Please see the documentation of the function
                `evotorch.neuroevolution.net.str_to_net(...)` to see how such
                a neural network structure string looks like.
                Note that this network can be a recurrent network.
                When the network's `forward(...)` method can optionally accept
                an additional positional argument for the hidden state of the
                network and returns an additional value for its next state,
                then the policy is treated as a recurrent one.
                When the network is given as a callable object (e.g.
                a subclass of `nn.Module` or a function) and this callable
                object is decorated via `evotorch.decorators.pass_info`,
                the following keyword arguments will be passed:
                (i) `obs_length` (the length of the observation vector),
                (ii) `act_length` (the length of the action vector),
                (iii) `obs_shape` (the shape tuple of the observation space),
                (iv) `act_shape` (the shape tuple of the action space),
                (v) `obs_space` (the Box object specifying the observation
                space, and
                (vi) `act_space` (the Box object specifying the action
                space). Note that `act_space` will always be given as a
                `gym.spaces.Box` instance, even when the actual gym
                environment has a discrete action space. This because `GymNE`
                always expects the neural network to return a tensor of
                floating-point numbers.
            env_name: Deprecated alias for the keyword argument `env`.
                It is recommended to use the argument `env` instead.
            network_args: Optionally a dict-like object, storing keyword
                arguments to be passed to the network while instantiating it.
            env_config: Keyword arguments to pass to `gym.make(...)` while
                creating the `gym` environment.
            observation_normalization: Whether or not to do online observation
                normalization.
            num_episodes: Number of episodes over which a single solution will
                be evaluated.
            episode_length: Maximum amount of simulator interactions allowed
                in a single episode. If left as None, whether or not an episode
                is terminated is determined only by the `gym` environment
                itself.
            decrease_rewards_by: Some gym env.s are defined in such a way that
                the agent gets a constant reward for each timestep
                it survives. This constant reward can also be called
                "survival bonus". Such a rewarding scheme can lead the
                evolution to local optima where the agent does nothing
                but does not die either, just to collect the survival
                bonuses. To prevent this, it can be desired to
                remove the survival bonuses from each reward obtained.
                If this is the case with the problem at hand,
                the user can set the argument `decrease_rewards_by`
                to a positive float number, and that number will
                be subtracted from each reward.
            alive_bonus_schedule: Use this to add a customized amount of
                alive bonus.
                If left as None (which is the default), additional alive
                bonus will not be added.
                If given as a tuple `(t, b)`, an alive bonus `b` will be
                added onto all the rewards beyond the timestep `t`.
                If given as a tuple `(t0, t1, b)`, a partial (linearly
                increasing towards `b`) alive bonus will be added onto
                all the rewards between the timesteps `t0` and `t1`,
                and a full alive bonus (which equals to `b`) will be added
                onto all the rewards beyond the timestep `t1`.
            action_noise_stdev: If given as a real number `s`, then, for
                each generated action, Gaussian noise with standard
                deviation `s` will be sampled, and then this sampled noise
                will be added onto the action.
                If action noise is not desired, then this argument can be
                left as None.
            num_actors: Number of actors to create for parallelized
                evaluation of the solutions.
                One can also set this as "max", which means that
                an actor will be created on each available CPU.
                When the parallelization is enabled each actor will have its
                own instance of the `gym` environment.
            actor_config: A dictionary, representing the keyword arguments
                to be passed to the options(...) used when creating the
                ray actor objects. To be used for explicitly allocating
                resources per each actor.
                For example, for declaring that each actor is to use a GPU,
                one can pass `actor_config=dict(num_gpus=1)`.
                Can also be given as None (which is the default),
                if no such options are to be passed.
            num_subbatches: If `num_subbatches` is None (assuming that
                `subbatch_size` is also None), then, when evaluating a
                population, the population will be split into n pieces, `n`
                being the number of actors, and each actor will evaluate
                its assigned piece. If `num_subbatches` is an integer `m`,
                then the population will be split into `m` pieces,
                and actors will continually accept the next unevaluated
                piece as they finish their current tasks.
                The arguments `num_subbatches` and `subbatch_size` cannot
                be given values other than None at the same time.
            subbatch_size: If `subbatch_size` is None (assuming that
                `num_subbatches` is also None), then, when evaluating a
                population, the population will be split into `n` pieces, `n`
                being the number of actors, and each actor will evaluate its
                assigned piece. If `subbatch_size` is an integer `m`,
                then the population will be split into pieces of size `m`,
                and actors will continually accept the next unevaluated
                piece as they finish their current tasks.
                When there can be significant difference across the solutions
                in terms of computational requirements, specifying a
                `subbatch_size` can be beneficial, because, while one
                actor is busy with a subbatch containing computationally
                challenging solutions, other actors can accept more
                tasks and save time.
                The arguments `num_subbatches` and `subbatch_size` cannot
                be given values other than None at the same time.
            initial_bounds: Specifies an interval from which the values of the
                initial policy parameters will be drawn.
        """
        # Store various environment information
        if (env is not None) and (env_name is None):
            self._env_maker = env
        elif (env is None) and (env_name is not None):
            self._env_maker = env_name
        elif (env is not None) and (env_name is not None):
            raise ValueError(
                f"Received values for both `env` ({repr(env)}) and `env_name` ({repr(env_name)})."
                f" Please specify the environment to solve via only one of these arguments, not both."
            )
        else:
            raise ValueError("Environment name is missing. Please specify it via the argument `env`.")

        # Make sure that the network argument is not missing.
        if network is None:
            raise ValueError(
                "Received None via the argument `network`."
                "Please provide the network as a string, or as a `Callable`, or as a `torch.nn.Module` instance."
            )

        # Store various environment information
        self._env_config = {} if env_config is None else deepcopy(dict(env_config))
        self._decrease_rewards_by = 0.0 if decrease_rewards_by is None else float(decrease_rewards_by)
        self._alive_bonus_schedule = alive_bonus_schedule
        self._action_noise_stdev = None if action_noise_stdev is None else float(action_noise_stdev)
        self._observation_normalization = bool(observation_normalization)
        self._num_episodes = int(num_episodes)
        self._episode_length = None if episode_length is None else int(episode_length)

        self._info_keys = dict(cumulative_reward="avg", interaction_count="sum")

        self._env: Optional[gym.Env] = None

        self._obs_stats: Optional[RunningStat] = None
        self._collected_stats: Optional[RunningStat] = None

        # Create a temporary environment to read its dimensions
        tmp_env = _make_env(self._env_maker, **(self._env_config))

        # Store the temporary environment's dimensions
        self._obs_length = len(tmp_env.observation_space.low)

        if isinstance(tmp_env.action_space, gym.spaces.Discrete):
            self._act_length = tmp_env.action_space.n
            self._box_act_space = gym.spaces.Box(low=float("-inf"), high=float("inf"), shape=(self._act_length,))
        else:
            self._act_length = len(tmp_env.action_space.low)
            self._box_act_space = tmp_env.action_space

        self._act_space = tmp_env.action_space
        self._obs_space = tmp_env.observation_space
        self._obs_shape = tmp_env.observation_space.low.shape

        # Validate the space types of the environment
        ensure_space_types(tmp_env)

        if self._observation_normalization:
            self._obs_stats = RunningStat()
            self._collected_stats = RunningStat()
        else:
            self._obs_stats = None
            self._collected_stats = None
        self._interaction_count: int = 0
        self._episode_count: int = 0

        super().__init__(
            objective_sense="max",  # RL is maximization
            network=network,  # Using the policy as the network
            network_args=network_args,
            initial_bounds=initial_bounds,
            num_actors=num_actors,
            actor_config=actor_config,
            subbatch_size=subbatch_size,
            device="cpu",
        )

        self.after_eval_hook.append(self._extra_status)

    @property
    def _network_constants(self) -> dict:
        return {
            "obs_length": self._obs_length,
            "act_length": self._act_length,
            "obs_space": self._obs_space,
            "act_space": self._box_act_space,
            "obs_shape": self._obs_space.shape,
            "act_shape": self._box_act_space.shape,
        }

    @property
    def _str_network_constants(self) -> dict:
        return {
            "obs_space": self._obs_space.shape,
            "act_space": self._box_act_space.shape,
        }

    def _instantiate_new_env(self, **kwargs) -> gym.Env:
        env_config = {**kwargs, **(self._env_config)}
        env = _make_env(self._env_maker, **env_config)
        if self._alive_bonus_schedule is not None:
            env = AliveBonusScheduleWrapper(env, self._alive_bonus_schedule)
        return env

    def _get_env(self) -> gym.Env:
        if self._env is None:
            self._env = self._instantiate_new_env()
        return self._env

    def _normalize_observation(self, observation: Iterable, *, update_stats: bool = True) -> Iterable:
        observation = np.asarray(observation, dtype="float32")
        if self.observation_normalization:
            if update_stats:
                self._obs_stats.update(observation)
                self._collected_stats.update(observation)
            return self._obs_stats.normalize(observation)
        else:
            return observation

    def _use_policy(self, observation: Iterable, policy: nn.Module) -> Iterable:
        with torch.no_grad():
            result = policy(torch.as_tensor(observation, dtype=torch.float32, device="cpu")).numpy()
        if self._action_noise_stdev is not None:
            result = (
                result
                + self.make_gaussian(len(result), center=0.0, stdev=self._action_noise_stdev, device="cpu").numpy()
            )
        env = self._get_env()
        if isinstance(env.action_space, gym.spaces.Discrete):
            result = np.argmax(result)
        elif isinstance(env.action_space, gym.spaces.Box):
            result = np.clip(result, env.action_space.low, env.action_space.high)
        return result

    def _prepare(self) -> None:
        super()._prepare()
        self._get_env()

    @property
    def network_device(self) -> Device:
        """The device on which the problem should place data e.g. the network
        In the case of GymNE, supported Gym environments return numpy arrays on CPU which are converted to Tensors
        Therefore, it is almost always optimal to place the network on CPU
        """
        return torch.device("cpu")

    def _rollout(
        self,
        *,
        policy: nn.Module,
        update_stats: bool = True,
        visualize: bool = False,
        decrease_rewards_by: Optional[float] = None,
    ) -> dict:
        """Peform a rollout of a network"""
        if decrease_rewards_by is None:
            decrease_rewards_by = self._decrease_rewards_by
        else:
            decrease_rewards_by = float(decrease_rewards_by)

        policy = ensure_stateful(policy)
        policy.reset()

        if visualize:
            env = self._instantiate_new_env(render_mode="human")
        else:
            env = self._get_env()

        observation = self._normalize_observation(reset_env(env), update_stats=update_stats)
        if visualize:
            env.render()
        t = 0

        cumulative_reward = 0.0

        while True:
            observation, raw_reward, done, info = take_step_in_env(env, self._use_policy(observation, policy))
            reward = raw_reward - decrease_rewards_by
            t += 1
            if update_stats:
                self._interaction_count += 1

            if visualize:
                env.render()

            observation = self._normalize_observation(observation, update_stats=update_stats)

            cumulative_reward += reward

            if done or ((self._episode_length is not None) and (t >= self._episode_length)):
                if update_stats:
                    self._episode_count += 1

                final_info = dict(cumulative_reward=cumulative_reward, interaction_count=t)

                for k in self._info_keys:
                    if k not in final_info:
                        final_info[k] = info[k]

                return final_info

    @property
    def _nonserialized_attribs(self) -> List[str]:
        return super()._nonserialized_attribs + ["_env"]

    def run(
        self,
        policy: Union[nn.Module, Iterable],
        *,
        update_stats: bool = False,
        visualize: bool = False,
        num_episodes: Optional[int] = None,
        decrease_rewards_by: Optional[float] = None,
    ) -> dict:
        """
        Evaluate the policy on the gym environment.

        Args:
            policy: The policy to be evaluated. This can be a torch module
                or a sequence of real numbers representing the parameters
                of a policy network.
            update_stats: Whether or not to update the observation
                normalization data while running the policy. If observation
                normalization is not enabled, then this argument will be
                ignored.
            visualize: Whether or not to render the environment while running
                the policy.
            num_episodes: Over how many episodes will the policy be evaluated.
                Expected as None (which is the default), or as an integer.
                If given as None, then the `num_episodes` value that was given
                while initializing this GymNE will be used.
            decrease_rewards_by: How much each reward value should be
                decreased. If left as None, the `decrease_rewards_by` value
                value that was given while initializing this GymNE will be
                used.
        Returns:
            A dictionary containing the score and the timestep count.
        """
        if not isinstance(policy, nn.Module):
            policy = self.make_net(policy)

        if num_episodes is None:
            num_episodes = self._num_episodes

        try:
            policy.eval()

            episode_results = [
                self._rollout(
                    policy=policy,
                    update_stats=update_stats,
                    visualize=visualize,
                    decrease_rewards_by=decrease_rewards_by,
                )
                for _ in range(num_episodes)
            ]

            results = _accumulate_all_across_dicts(episode_results, self._info_keys)
            return results
        finally:
            policy.train()

    def visualize(
        self,
        policy: Union[nn.Module, Iterable],
        *,
        update_stats: bool = False,
        num_episodes: Optional[int] = 1,
        decrease_rewards_by: Optional[float] = None,
    ) -> dict:
        """
        Evaluate the policy and render its actions in the environment.

        Args:
            policy: The policy to be evaluated. This can be a torch module
                or a sequence of real numbers representing the parameters
                of a policy network.
            update_stats: Whether or not to update the observation
                normalization data while running the policy. If observation
                normalization is not enabled, then this argument will be
                ignored.
            num_episodes: Over how many episodes will the policy be evaluated.
                Expected as None (which is the default), or as an integer.
                If given as None, then the `num_episodes` value that was given
                while initializing this GymNE will be used.
            decrease_rewards_by: How much each reward value should be
                decreased. If left as None, the `decrease_rewards_by` value
                value that was given while initializing this GymNE will be
                used.
        Returns:
            A dictionary containing the score and the timestep count.
        """
        return self.run(
            policy=policy,
            update_stats=update_stats,
            visualize=True,
            num_episodes=num_episodes,
            decrease_rewards_by=decrease_rewards_by,
        )

    def _ensure_obsnorm(self):
        if not self.observation_normalization:
            raise ValueError("This feature can only be used when observation_normalization=True.")

    def get_observation_stats(self) -> RunningStat:
        """Get the observation stats"""
        self._ensure_obsnorm()
        return self._obs_stats

    def _make_sync_data_for_actors(self) -> Any:
        if self.observation_normalization:
            return dict(obs_stats=self.get_observation_stats())
        else:
            return None

    def set_observation_stats(self, rs: RunningStat):
        """Set the observation stats"""
        self._ensure_obsnorm()
        self._obs_stats.reset()
        self._obs_stats.update(rs)

    def _use_sync_data_from_main(self, received: dict):
        for k, v in received.items():
            if k == "obs_stats":
                self.set_observation_stats(v)

    def pop_observation_stats(self) -> RunningStat:
        """Get and clear the collected observation stats"""
        self._ensure_obsnorm()
        result = self._collected_stats
        self._collected_stats = RunningStat()
        return result

    def _make_sync_data_for_main(self) -> Any:
        result = dict(episode_count=self.episode_count, interaction_count=self.interaction_count)

        if self.observation_normalization:
            result["obs_stats_delta"] = self.pop_observation_stats()

        return result

    def update_observation_stats(self, rs: RunningStat):
        """Update the observation stats via another RunningStat instance"""
        self._ensure_obsnorm()
        self._obs_stats.update(rs)

    def _use_sync_data_from_actors(self, received: list):
        total_episode_count = 0
        total_interaction_count = 0

        for data in received:
            data: dict
            total_episode_count += data["episode_count"]
            total_interaction_count += data["interaction_count"]
            if self.observation_normalization:
                self.update_observation_stats(data["obs_stats_delta"])

        self.set_episode_count(total_episode_count)
        self.set_interaction_count(total_interaction_count)

    def _make_pickle_data_for_main(self) -> dict:
        # For when the main Problem object (the non-remote one) gets pickled,
        # this function returns the counters of this remote Problem instance,
        # to be sent to the main one.
        return dict(interaction_count=self.interaction_count, episode_count=self.episode_count)

    def _use_pickle_data_from_main(self, state: dict):
        # For when a newly unpickled Problem object gets (re)parallelized,
        # this function restores the inner states specific to this remote
        # worker. In the case of GymNE, those inner states are episode
        # and interaction counters.
        for k, v in state.items():
            if k == "episode_count":
                self.set_episode_count(v)
            elif k == "interaction_count":
                self.set_interaction_count(v)
            else:
                raise ValueError(f"When restoring the inner state of a remote worker, unrecognized state key: {k}")

    def _extra_status(self, batch: SolutionBatch):
        return dict(total_interaction_count=self.interaction_count, total_episode_count=self.episode_count)

    @property
    def observation_normalization(self) -> bool:
        """
        Get whether or not observation normalization is enabled.
        """
        return self._observation_normalization

    def set_episode_count(self, n: int):
        """
        Set the episode count manually.
        """
        self._episode_count = int(n)

    def set_interaction_count(self, n: int):
        """
        Set the interaction count manually.
        """
        self._interaction_count = int(n)

    @property
    def interaction_count(self) -> int:
        """
        Get the total number of simulator interactions made.
        """
        return self._interaction_count

    @property
    def episode_count(self) -> int:
        """
        Get the total number of episodes completed.
        """
        return self._episode_count

    def _get_local_episode_count(self) -> int:
        return self.episode_count

    def _get_local_interaction_count(self) -> int:
        return self.interaction_count

    def _evaluate_network(self, policy: nn.Module) -> Union[float, torch.Tensor]:
        result = self.run(
            policy,
            update_stats=True,
            visualize=False,
            num_episodes=self._num_episodes,
            decrease_rewards_by=self._decrease_rewards_by,
        )
        return result["cumulative_reward"]

    def to_policy(self, x: Iterable, *, clip_actions: bool = True) -> nn.Module:
        """
        Convert the given parameter vector to a policy as a PyTorch module.

        If the problem is configured to have observation normalization,
        the PyTorch module also contains an additional normalization layer.

        Args:
            x: An sequence of real numbers, containing the parameters
                of a policy. Can be a PyTorch tensor, a numpy array,
                or a Solution.
            clip_actions: Whether or not to add an action clipping layer so
                that the generated actions will always be within an
                acceptable range for the environment.
        Returns:
            The policy expressed by the parameters.
        """

        policy = self.make_net(x)

        if self.observation_normalization and (self._obs_stats.count > 0):
            policy = ObsNormWrapperModule(policy, self._obs_stats)

        if clip_actions and isinstance(self._get_env().action_space, gym.spaces.Box):
            policy = ActClipWrapperModule(policy, self._get_env().action_space)

        return policy

    def save_solution(self, solution: Iterable, fname: Union[str, Path]):
        """
        Save the solution into a pickle file.
        Among the saved data within the pickle file are the solution
        (as a PyTorch tensor), the policy (as a `torch.nn.Module` instance),
        and observation stats (if any).

        Args:
            solution: The solution to be saved. This can be a PyTorch tensor,
                a `Solution` instance, or any `Iterable`.
            fname: The file name of the pickle file to be created.
        """

        # Convert the solution to a PyTorch tensor on the cpu.
        if isinstance(solution, torch.Tensor):
            solution = solution.to("cpu")
        elif isinstance(solution, Solution):
            solution = solution.values.clone().to("cpu")
        else:
            solution = torch.as_tensor(solution, dtype=torch.float32, device="cpu")

        if isinstance(solution, ReadOnlyTensor):
            solution = solution.as_subclass(torch.Tensor)

        policy = self.to_policy(solution).to("cpu")

        # Store the solution and the policy.
        result = {
            "solution": solution,
            "policy": policy,
        }

        # If available, store the observation stats.
        if self.observation_normalization and (self._obs_stats is not None):
            result["obs_mean"] = torch.as_tensor(self._obs_stats.mean)
            result["obs_stdev"] = torch.as_tensor(self._obs_stats.stdev)
            result["obs_sum"] = torch.as_tensor(self._obs_stats.sum)
            result["obs_sum_of_squares"] = torch.as_tensor(self._obs_stats.sum_of_squares)

        # Some additional data.
        result["interaction_count"] = self.interaction_count
        result["episode_count"] = self.episode_count
        result["time"] = datetime.now()

        # If the environment is specified via a string ID, then store that ID.
        if isinstance(self._env_maker, str):
            result["env"] = self._env_maker

        # Save the dictionary which stores the data.
        with open(fname, "wb") as f:
            pickle.dump(result, f)

    def get_env(self) -> gym.Env:
        """
        Get the gym environment stored by this GymNE instance
        """
        return self._get_env()
