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
import random
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Union

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box
from torch import nn

from ..core import Solution, SolutionBatch
from ..tools import Device, ReadOnlyTensor, pass_info_if_needed, split_workload
from .baseneproblem import BaseNEProblem
from .net import str_to_net
from .net.rl import ActClipWrapperModule, ObsNormWrapperModule
from .net.runningnorm import RunningNorm
from .net.vecrl import Policy, TorchWrapper, make_vector_env


def _is_numpy_array_entirely_inf(x: np.ndarray) -> bool:
    return np.all(np.isinf(x))


def _is_numpy_array_entirely_positive_inf(x: np.ndarray) -> bool:
    return _is_numpy_array_entirely_inf(x) and np.all(x > 0)


def _is_numpy_array_entirely_negative_inf(x: np.ndarray) -> bool:
    return _is_numpy_array_entirely_inf(x) and np.all(x < 0)


def _numpy_arrays_specify_no_bounds(low: np.ndarray, high: np.ndarray) -> bool:
    return _is_numpy_array_entirely_negative_inf(low) and _is_numpy_array_entirely_positive_inf(high)


def _numpy_arrays_specify_bounds(low: np.ndarray, high: np.ndarray) -> bool:
    return not _numpy_arrays_specify_no_bounds(low, high)


def _make_dummy_env(env: Union[str, Callable], **kwargs) -> gym.vector.VectorEnv:
    if isinstance(env, str):
        result = make_vector_env(env, num_envs=1, **kwargs)
    else:
        result = env(num_envs=1, **kwargs)
    result = TorchWrapper(result, discrete_to_continuous_act=True)
    return result


def _env_spaces(env: Union[str, Callable], **kwargs) -> tuple:
    tmp_env = _make_dummy_env(env, **kwargs)
    return tmp_env.single_observation_space, tmp_env.single_action_space


def _env_constants_for_str_net(env: Union[str, Callable], **kwargs) -> dict:
    obs_space, act_space = _env_spaces(env, **kwargs)
    return {
        "obs_length": obs_space.shape[0],
        "act_length": act_space.shape[0],
        "obs_shape": obs_space.shape,
        "obs_space": obs_space.shape,
        "act_shape": act_space.shape,
        "act_space": act_space.shape,
    }


def _env_constants_for_callable_net(env: Union[str, Callable], **kwargs) -> dict:
    obs_space, act_space = _env_spaces(env, **kwargs)
    return {
        "obs_length": obs_space.shape[0],
        "act_length": act_space.shape[0],
        "obs_shape": obs_space.shape,
        "obs_space": obs_space,
        "act_shape": act_space.shape,
        "act_space": act_space,
    }


class VecGymNE(BaseNEProblem):
    """
    An EvoTorch problem for solving vectorized gym environments
    """

    def __init__(
        self,
        env: Union[str, Callable],
        network: Union[str, Callable, nn.Module],
        *,
        env_config: Optional[Mapping] = None,
        max_num_envs: Optional[int] = None,
        network_args: Optional[Mapping] = None,
        observation_normalization: bool = False,
        decrease_rewards_by: Optional[float] = None,
        alive_bonus_schedule: Optional[tuple] = None,
        action_noise_stdev: Optional[float] = None,
        num_episodes: int = 1,
        device: Optional[Device] = None,
        num_actors: Optional[Union[int, str]] = None,
        num_gpus_per_actor: Optional[int] = None,
        num_subbatches: Optional[int] = None,
        subbatch_size: Optional[int] = None,
        actor_config: Optional[Mapping] = None,
    ):
        """
        Initialize the VecGymNE.

        Args:
            env: Environment to be solved.
                If this is given as a string starting with "gym::" (e.g.
                "gym::Humanoid-v4", etc.), then it is assumed that the target
                environment is a classical gym environment.
                If this is given as a string starting with "brax::" (e.g.
                "brax::humanoid", etc.), then it is assumed that the target
                environment is a brax environment.
                If this is given as a string which does not contain "::" at
                all (e.g. "Humanoid-v4", etc.), then it is assumed that the
                target environment is a classical gym environment. Therefore,
                "gym::Humanoid-v4" and "Humanoid-v4" are equivalent.
                If this argument is given as a Callable (maybe a function or a
                class), then, with the assumption that this Callable expects
                a keyword argument `num_envs: int`, this Callable is called
                and its result (expected as a `gym.vector.VectorEnv` instance)
                is used as the environment.
            network: A network structure string, or a Callable (which can be
                a class inheriting from `torch.nn.Module`, or a function
                which returns a `torch.nn.Module` instance), or an instance
                of `torch.nn.Module`.
                The object provided here determines the structure of the
                neural network whose parameters will be evolved.
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
                environment has a discrete action space. This because
                `VecGymNE` always expects the neural network to return
                a tensor of floating-point numbers.
            env_config: Keyword arguments to pass to the environment while
                it is being created.
            max_num_envs: Maximum number of environments to be instantiated.
                By default, this is None, which means that the number of
                environments can go up to the population size (or up to the
                number of solutions that a remote actor receives, if the
                problem object is configured to have parallelization).
                For situations where the current reinforcement learning task
                requires large amount of resources (e.g. memory), allocating
                environments as much as the number of solutions might not
                be feasible. In such cases, one can set `max_num_envs` as an
                integer to bring an upper bound (in total, across all the
                remote actors, for when the problem is parallelized) to how
                many environments can be allocated.
            network_args: Any additional keyword argument to be used when
                instantiating the network can be specified via `network_args`
                as a dictionary. If there are no such additional keyword
                arguments, then `network_args` can be left as None.
                Note that the argument `network_args` is expected to be None
                when the network is specified as a `torch.nn.Module` instance.
            observation_normalization: Whether or not online normalization
                will be done on the encountered observations.
            decrease_rewards_by: If given as a float, each reward will be
                decreased by this amount. For example, if the environment's
                reward function has a constant "alive bonus" (i.e. a bonus
                that is constantly added onto the reward as long as the
                agent is alive), and if you wish to negate this bonus,
                you can set `decrease_rewards_by` to this bonus amount,
                and the bonus will be nullified.
                If you do not wish to affect the rewards in this manner,
                keep this as None.
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
                For sampling the noise, the global random number generator
                of PyTorch on the simulator's device will be used.
            num_episodes: Number of episodes over which each policy will
                be evaluated. The default is 1.
            device: The device in which the population will be kept.
                If you wish to do a single-GPU evolution, we recommend
                to set this as "cuda" (or "cuda:0", or "cuda:1", etc.),
                assuming that the simulator will also instantiate itself
                on that same device.
                Alternatively, if you wish to do a multi-GPU evolution,
                we recommend to leave this as None or set this as "cpu",
                so that the main population will be kept on the cpu
                and the remote actors will perform their evaluations on
                the GPUs that are assigned to them.
            num_actors: Number of actors to create for parallelized
                evaluation of the solutions.
                Certain string values are also accepted.
                When given as "max" or as "num_cpus", the number of actors
                will be equal to the number of all available CPUs in the ray
                cluster.
                When given as "num_gpus", the number of actors will be
                equal to the number of all available GPUs in the ray
                cluster, and each actor will be assigned a GPU.
                When given as "num_devices", the number of actors will be
                equal to the minimum among the number of CPUs and the number
                of GPUs available in the cluster (or will be equal to the
                number of CPUs if there is no GPU), and each actor will be
                assigned a GPU (if available).
                If `num_actors` is given as "num_gpus" or "num_devices",
                the argument `num_gpus_per_actor` must not be used,
                and the `actor_config` dictionary must not contain the
                key "num_gpus".
                If `num_actors` is given as something other than "num_gpus"
                or "num_devices", and if you wish to assign GPUs to each
                actor, then please see the argument `num_gpus_per_actor`.
            num_gpus_per_actor: Number of GPUs to be assigned to each
                actor. This can be an integer or a float (for when you
                wish to assign fractional amounts of GPUs to actors).
                When `num_actors` has the special value "num_devices",
                the argument `num_gpus_per_actor` is expected to be left as
                None.
            num_subbatches: For when there are multiple actors, you can
                set this to an integer n if you wish the population
                to be divided exactly into n sub-batches. The actors, as they
                finish their currently assigned sub-batch of solutions,
                will pick the next un-evaluated sub-batch.
                If you specify too large numbers for this argument, then
                each sub-batch will be smaller.
                When working with vectorized simulators on GPU, having too
                many and too small sub-batches can hurt the performance.
                This argument can be left as None, in which case, assuming
                that `subbatch_size` is also None, the population will be
                split to m sub-batches, m being the number of actors.
            subbatch_size: For when there are multiple actors, you can
                set this to an integer n if you wish the population to be
                divided into sub-batches in such a way that each sub-batch
                will consist of exactly n solutions. The actors, as they
                finish their currently assigned sub-batch of solutions,
                will pick the next un-evaluated sub-batch.
                If you specify too small numbers for this argument, then
                there will be many sub-batches, each sub-batch having a
                small number of solutions.
                When working with vectorized simulators on GPU, having too
                many and too small sub-batches can hurt the performance.
                This argument can be left as None, in which case, assuming
                that `num_subbatches` is also None, the population will be
                split to m sub-batches, m being the number of actors.
            actor_config: Additional configuration to be used when creating
                each actor with the help of `ray` library.
                Can be left as None if additional configuration is not needed.
        """

        # Store the string or the Callable that will be used to generate the reinforcement learning environment.
        self._env_maker = env

        # Declare the variable which will store the environment.
        self._env: Optional[TorchWrapper] = None

        # Declare the variable which will store the batch size of the vectorized environment.
        self._num_envs: Optional[int] = None

        # Store the upper bound (if any) regarding how many environments can exist at the same time.
        self._max_num_envs: Optional[int] = None if max_num_envs is None else int(max_num_envs)

        # Actor-specific upper bound regarding how many environments can exist at the same time.
        # This variable will be filled by the `_parallelize(...)` method.
        self._actor_max_num_envs: Optional[int] = None

        # Declare the variable which stores whether or not we properly initialized the `_actor_max_num_envs` variable.
        self._actor_max_num_envs_ready: bool = False

        # Store the additional configurations to be used as keyword arguments while instantiating the environment.
        self._env_config: dict = {} if env_config is None else dict(env_config)

        # Declare the variable that will store the device of the simulator.
        # This variable will be filled when the first observation is received from the environment.
        # The device of the observation array received from the environment will determine the value of this variable.
        self._simulator_device: Optional[torch.device] = None

        # Store the neural network architecture (that might be a string or an `nn.Module` instance).
        self._architecture = network

        if network_args is None:
            # If `network_args` is given as None, change it to an empty dictionary
            network_args = {}

        if isinstance(network, str):
            # If the network is given as a string, then we will need the values for the constants `obs_length`,
            # `act_length`, and `obs_space`. To obtain those values, we use our helper function
            # `_env_constants_for_str_net(...)` which temporarily instantiates the specified environment and returns
            # its needed constants.
            env_constants = _env_constants_for_str_net(self._env_maker, **(self._env_config))
        elif isinstance(network, nn.Module):
            # If the network is an already instantiated nn.Module, then we do not prepare any pre-defined constants.
            env_constants = {}
        else:
            # If the network is given as a Callable, then we will need the values for the constants `obs_length`,
            # `act_length`, and `obs_space`. To obtain those values, we use our helper function
            # `_env_constants_for_callable_net(...)` which temporarily instantiates the specified environment and
            # returns its needed constants.
            env_constants = _env_constants_for_callable_net(self._env_maker, **(self._env_config))

        # Build a `Policy` instance according to the given architecture, and store it.
        if isinstance(network, str):
            instantiated_net = str_to_net(network, **{**env_constants, **network_args})
        elif isinstance(network, nn.Module):
            instantiated_net = network
        else:
            instantiated_net = pass_info_if_needed(network, env_constants)(**network_args)
        self._policy = Policy(instantiated_net)

        # Store the boolean which indicates whether or not there will be observation normalization.
        self._observation_normalization = bool(observation_normalization)

        # Declare the variables that will store the observation-related stats if observation normalization is enabled.
        self._obs_stats: Optional[RunningNorm] = None
        self._collected_stats: Optional[RunningNorm] = None

        # Store the number of episodes configuration given by the user.
        self._num_episodes = int(num_episodes)

        # Store the `decrease_rewards_by` configuration given by the user.
        self._decrease_rewards_by = None if decrease_rewards_by is None else float(decrease_rewards_by)

        if alive_bonus_schedule is None:
            # If `alive_bonus_schedule` argument is None, then we store it as None as well.
            self._alive_bonus_schedule = None
        else:
            # This is the case where the user has specified an `alive_bonus_schedule`.
            alive_bonus_schedule = list(alive_bonus_schedule)
            alive_bonus_schedule_length = len(alive_bonus_schedule)
            if alive_bonus_schedule_length == 2:
                # If `alive_bonus_schedule` was given as a 2-element sequence (t, b), then store it as (t, t, b).
                # This means that the partial alive bonus time window starts and ends at t, therefore, there will
                # be no alive bonus until t, and beginning with t, there will be full alive bonus.
                self._alive_bonus_schedule = [
                    int(alive_bonus_schedule[0]),
                    int(alive_bonus_schedule[0]),
                    float(alive_bonus_schedule[1]),
                ]
            elif alive_bonus_schedule_length == 3:
                # If `alive_bonus_schedule` was given as a 3-element sequence (t0, t1, b), then store those 3
                # elements.
                self._alive_bonus_schedule = [
                    int(alive_bonus_schedule[0]),
                    int(alive_bonus_schedule[1]),
                    float(alive_bonus_schedule[2]),
                ]
            else:
                # `alive_bonus_schedule` sequences with unrecognized lengths trigger an error.
                raise ValueError(
                    f"Received invalid number elements as the alive bonus schedule."
                    f" Expected 2 or 3 items, but got these: {self._alive_bonus_schedule}"
                    f" (having a length of {len(self._alive_bonus_schedule)})."
                )

        # If `action_noise_stdev` is specified, store it.
        self._action_noise_stdev = None if action_noise_stdev is None else float(action_noise_stdev)

        # Initialize the counters for the number of simulator interactions and the number of episodes.
        self._interaction_count: int = 0
        self._episode_count: int = 0

        device_is_cpu = (device is None) or (str(device) == "cpu")
        actors_use_gpu = (
            (num_actors is not None)
            and (num_actors > 1)
            and (num_gpus_per_actor is not None)
            and (num_gpus_per_actor > 0)
        )

        if not device_is_cpu:
            # In the case where the device is something other than the cpu, we tell SyncVectorEnv to use this device.
            self._device_for_sync_vector_env = device
            self._sync_vector_env_uses_aux_device = False
        elif actors_use_gpu:
            # In the case where this problem instance is configured to use multiple actors and the actors are
            # configured to use the available gpu(s), we tell SyncVectorEnv to use the `aux_device`.
            self._device_for_sync_vector_env = None
            self._sync_vector_env_uses_aux_device = True
        else:
            self._device_for_sync_vector_env = None
            self._sync_vector_env_uses_aux_device = False

        # Call the superclass
        super().__init__(
            objective_sense="max",
            initial_bounds=(-0.00001, 0.00001),
            solution_length=self._policy.parameter_length,
            device=device,
            dtype=torch.float32,
            num_actors=num_actors,
            num_gpus_per_actor=num_gpus_per_actor,
            actor_config=actor_config,
            num_subbatches=num_subbatches,
            subbatch_size=subbatch_size,
        )

        self.after_eval_hook.append(self._extra_status)

    def _parallelize(self):
        super()._parallelize()
        if self.is_main:
            if not self._actor_max_num_envs_ready:
                if self._actors is None:
                    self._actor_max_num_envs = self._max_num_envs
                else:
                    if self._max_num_envs is not None:
                        max_num_envs_per_actor = split_workload(self._max_num_envs, len(self._actors))
                        for i_actor, actor in enumerate(self._actors):
                            actor.call.remote("_set_actor_max_num_envs", max_num_envs_per_actor[i_actor])
                self._actor_max_num_envs_ready = True

    def _set_actor_max_num_envs(self, n: int):
        self._actor_max_num_envs = n
        self._actor_max_num_envs_ready = True

    def _extra_status(self, batch: SolutionBatch):
        return dict(total_interaction_count=self.interaction_count, total_episode_count=self.episode_count)

    @property
    def observation_normalization(self) -> bool:
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

    def _get_env(self, num_policies: int) -> TorchWrapper:
        # Get the existing environment instance stored by this VecGymNE, after (re)building it if needed.

        if (self._env is None) or (num_policies > self._num_envs):
            # If this VecGymNE does not have its environment ready yet (i.e. the `_env` attribute is None)
            # or it the batch size of the previously instantiated environment is not enough to deal with
            # the number of policies (i.e. the `_num_envs` attribute is less than `num_policies`), then
            # we (re)build the environment.

            # Keyword arguments to pass to the TorchWrapper.
            torch_wrapper_cfg = dict(
                force_classic_api=True,
                discrete_to_continuous_act=True,
                clip_actions=True,
            )

            # Keyword arguments to use only when the underlying environment is a classical gymnasium environment
            gym_cfg = dict(empty_info=True, num_episodes=self._num_episodes)

            if self._sync_vector_env_uses_aux_device:
                gym_cfg["device"] = self.aux_device
            elif self._device_for_sync_vector_env is not None:
                gym_cfg["device"] = self._device_for_sync_vector_env

            if isinstance(self._env_maker, str):
                # If the environment is specified via a string, then we use our `make_vector_env` function.
                self._env = make_vector_env(
                    self._env_maker,
                    num_envs=num_policies,
                    gym_kwargs=gym_cfg,
                    **torch_wrapper_cfg,
                    **(self._env_config),
                )
            else:
                # If the environment is specified via a Callable, then we call it.
                # We expect this Callable to accept a keyword argument named `num_envs`, and additionally, we pass
                # the environment configuration dictionary as keyword arguments.
                self._env = self._env_maker(num_envs=num_policies, **(self._env_config))

                if not isinstance(self._env, gym.vector.VectorEnv):
                    # If what is returned by the Callable is not a vectorized environment, then we trigger an error.
                    raise TypeError("This is not a vectorized environment")

                # We wrap the returned vectorized environment with a TorchWrapper, so that the actions that we send
                # and the observations and rewards that we receive are PyTorch tensors.
                self._env = TorchWrapper(self._env, **torch_wrapper_cfg)

            if self._env.num_envs != num_policies:
                # If the finally obtained vectorized environment has a different number of batch size, then we trigger
                # an error.
                raise ValueError("Incompatible number of environments")

            # We update the batch size of the created environment.
            self._num_envs = num_policies

            if not isinstance(self._env.single_observation_space, Box):
                # If the observation space is not Box, then we trigger an error.
                raise TypeError(
                    f"Unsupported observation type: {self._env.single_observation_space}."
                    f" Only Box-typed observation spaces are supported."
                )

            try:
                # If possible, use the `seed(...)` method to explicitly randomize the environment.
                # Although the new gym API removed the seed method, some environments define their own `seed(...)`
                # method for randomization.
                new_seed = random.randint(0, (2**32) - 1)
                self._env.seed(new_seed)
            except Exception:
                # Our attempt at manually seeding the environment has failed.
                # This could be because the environment does not have a `seed(...)` method.
                # Nothing to do.
                pass

        return self._env

    @property
    def _nonserialized_attribs(self):
        # Call the `_nonserialized_attribs` property implementation of the superclass to receive the base list
        # of non-serialized attributes, then add "_env" to this base list, and then return the resulting list.
        return super()._nonserialized_attribs + ["_env"]

    @property
    def _grad_device(self) -> torch.device:
        # For distributed mode, this property determines the device in which the temporary populations will be made
        # for gradient computation.

        if self._simulator_device is None:
            # If the simulator device is not known yet, then we return the cpu device.
            return torch.device("cpu")
        else:
            # If the simulator device is known, then we return that device.
            return self._simulator_device

    def _make_running_norm(self, observation: torch.Tensor) -> RunningNorm:
        # Make a new RunningNorm instance according to the observation tensor.
        # The dtype and the device of the new RunningNorm is taken from the observation.
        # This new RunningNorm is empty (i.e. does not contain any stats yet).
        return RunningNorm(shape=observation.shape[1:], dtype=observation.dtype, device=observation.device)

    def _transfer_running_norm(self, rn: RunningNorm, observation: torch.Tensor) -> RunningNorm:
        # Transfer (if necessary) the RunningNorm to the device of the observation tensor.
        # The returned RunningNorm may be the RunningNorm itself (if the device did not change)
        # or a new copy (if the device did change).
        if torch.device(rn.device) != torch.device(observation.device):
            rn = rn.to(observation.device)
        return rn

    def _normalize_observation(
        self, observation: torch.Tensor, *, mask: Optional[torch.Tensor] = None, update_stats: bool = True
    ) -> torch.Tensor:
        # This function normalizes the received observation batch.
        # If a mask is given (as a tensor of booleans), only observations with corresponding mask value set as True
        # will be taken into consideration.
        # If `update_stats` is given as True and observation normalization is enabled, then we will update the
        # RunningNorm instances as well.

        if self._observation_normalization:
            # This is the case where observation normalization is enabled.
            if self._obs_stats is None:
                # If we do not have observation stats yet, we build a new one (according to the dtype and device
                # of the observation).
                self._obs_stats = self._make_running_norm(observation)
            else:
                # If we already have observation stats, we make sure that it is in the correct device.
                self._obs_stats = self._transfer_running_norm(self._obs_stats, observation)

            if update_stats:
                # This is the case where the `update_stats` argument was encountered as True.
                if self._collected_stats is None:
                    # If the RunningNorm responsible to collect new stats is not built yet, we build it here
                    # (according to the dtype and device of the observation).
                    self._collected_stats = self._make_running_norm(observation)
                else:
                    # If the RunningNorm responsible to collect new stats already exists, then we make sure
                    # that it is in the correct device.
                    self._collected_stats = self._transfer_running_norm(self._collected_stats, observation)

                # We first update the RunningNorm responsible for collecting the new stats.
                self._collected_stats.update(observation, mask)

                # We now update the RunningNorm which stores all the stats, and return the normalized observation.
                result = self._obs_stats.update_and_normalize(observation, mask)
            else:
                # This is the case where the `update_stats` argument was encountered as False.
                # Here we normalize the observation but do not update our existing RunningNorm instances.
                result = self._obs_stats.update(observation, mask)
            return result
        else:
            # This is the case where observation normalization is disabled.
            # In this case, we just return the observation as it is.
            return observation

    def _ensure_obsnorm(self):
        if not self.observation_normalization:
            raise ValueError("This feature can only be used when observation_normalization=True.")

    def get_observation_stats(self) -> RunningNorm:
        """Get the observation stats"""
        self._ensure_obsnorm()
        return self._obs_stats

    def _make_sync_data_for_actors(self) -> Any:
        if self.observation_normalization:
            obs_stats = self.get_observation_stats()
            if obs_stats is not None:
                obs_stats = obs_stats.to("cpu")
            return dict(obs_stats=obs_stats)
        else:
            return None

    def set_observation_stats(self, rn: RunningNorm):
        """Set the observation stats"""
        self._ensure_obsnorm()
        self._obs_stats = rn

    def _use_sync_data_from_main(self, received: dict):
        for k, v in received.items():
            if k == "obs_stats":
                self.set_observation_stats(v)

    def pop_observation_stats(self) -> RunningNorm:
        """Get and clear the collected observation stats"""
        self._ensure_obsnorm()
        result = self._collected_stats
        self._collected_stats = None
        return result

    def _make_sync_data_for_main(self) -> Any:
        result = dict(episode_count=self.episode_count, interaction_count=self.interaction_count)

        if self.observation_normalization:
            collected = self.pop_observation_stats()
            if collected is not None:
                collected = collected.to("cpu")
            result["obs_stats_delta"] = collected

        return result

    def update_observation_stats(self, rn: RunningNorm):
        """Update the observation stats via another RunningNorm instance"""
        self._ensure_obsnorm()
        if self._obs_stats is None:
            self._obs_stats = rn
        else:
            self._obs_stats.update(rn)

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

    def _evaluate_batch(self, batch: SolutionBatch):
        if self._actor_max_num_envs is None:
            self._evaluate_subbatch(batch)
        else:
            subbatches = batch.split(max_size=self._actor_max_num_envs)
            for subbatch in subbatches:
                self._evaluate_subbatch(subbatch)

    def _evaluate_subbatch(self, batch: SolutionBatch):
        # Get the number of solutions and the solution batch from the shape of the batch.
        num_solutions, solution_length = batch.values_shape

        # Get (possibly after (re)building) the environment object.
        env = self._get_env(num_solutions)

        # Reset the environment and receive the first observation batch.
        obs_per_env = env.reset()

        # Update the simulator device according to the device of the observation batch received.
        self._simulator_device = obs_per_env.device

        # Get the number of environments.
        num_envs = obs_per_env.shape[0]

        # Transfer (if necessary) the solutions (which are the network parameters) to the simulator device.
        batch_values = batch.values.to(self._simulator_device)

        if num_solutions == num_envs:
            # If the number of solutions is equal to the number of environments, then we declare all of the solutions
            # as the network parameters, and we declare all of these environments active.
            params_per_env = batch_values
            active_per_env = torch.ones(num_solutions, dtype=torch.bool, device=self._simulator_device)
        elif num_solutions < num_envs:
            # If the number of solutions is less than the number of environments, then we allocate a new empty
            # tensor to represent the network parameters.
            params_per_env = torch.empty((num_envs, solution_length), dtype=batch.dtype, device=self._simulator_device)

            # The first `num_solutions` rows of this new parameters tensor is filled with the values of the solutions.
            params_per_env[:num_solutions, :] = batch_values

            # The remaining parameters become the clones of the first solution.
            params_per_env[num_solutions:, :] = batch_values[0]

            # At first, all the environments are declared as inactive.
            active_per_env = torch.zeros(num_envs, dtype=torch.bool, device=self._simulator_device)

            # Now, the first `num_solutions` amount of environments is declared as active.
            # The remaining ones remain inactive.
            active_per_env[:num_solutions] = True
        else:
            assert False, "Received incompatible number of environments"

        # We get the policy and fill it with the parameters stored by the solutions.
        policy = self._policy
        policy.set_parameters(params_per_env)

        # Declare the counter which stores the total timesteps encountered during this evaluation.
        total_timesteps = 0

        # Declare the counters (one for each environment) storing the number of episodes completed.
        num_eps_per_env = torch.zeros(num_envs, dtype=torch.int64, device=self._simulator_device)

        # Declare the scores (one for each environment).
        score_per_env = torch.zeros(num_envs, dtype=torch.float32, device=self._simulator_device)

        if self._alive_bonus_schedule is not None:
            # If an alive_bonus_schedule was provided, then we extract the timesteps.
            # bonus_t0 is the timestep where the partial alive bonus will start.
            # bonus_t1 is the timestep where the full alive bonus will start.
            # alive_bonus is the amount that will be added to reward if the agent is alive.
            bonus_t0, bonus_t1, alive_bonus = self._alive_bonus_schedule

            if bonus_t1 > bonus_t0:
                # If bonus_t1 is bigger than bonus_t0, then we have a partial alive bonus time window.
                add_partial_alive_bonus = True

                # We compute and store the length of the time window.
                bonus_t_gap_as_float = float(bonus_t1 - bonus_t0)
            else:
                # If bonus_t1 is NOT bigger than bonus_t0, then we do NOT have a partial alive bonus time window.
                add_partial_alive_bonus = False

            # To properly give the alive bonus for each solution, we need to keep track of the timesteps for all
            # the running solutions. So, we declare the following variable.
            t_per_env = torch.zeros(num_envs, dtype=torch.int64, device=self._simulator_device)

        def normalize(observations: torch.Tensor, *, mask: torch.Tensor) -> torch.Tensor:
            original_observations = observations
            observations = observations[mask]
            if observations.shape[0] == 0:
                return observations
            else:
                normalized = self._normalize_observation(observations)
                modified_observations = original_observations.clone()
                modified_observations[mask] = normalized
                return modified_observations

        # We normalize the initial observation.
        # obs_per_env = self._normalize_observation(obs_per_env, mask=active_per_env)
        obs_per_env = normalize(obs_per_env, mask=active_per_env)

        while True:
            # Pass the observations through the policy and get the actions to perform.
            action_per_env = policy(torch.as_tensor(obs_per_env, dtype=params_per_env.dtype))

            if self._action_noise_stdev is not None:
                # If we are to apply action noise, we sample from a Gaussian distribution and add the noise onto
                # the actions.
                action_per_env = action_per_env + (torch.rand_like(action_per_env) * self._action_noise_stdev)

            # Apply the actions, get the observations, rewards, and the 'done' flags.
            obs_per_env, reward_per_env, done_per_env, _ = env.step(action_per_env)

            if self._decrease_rewards_by is not None:
                # We decrease the rewards, if we have the configuration to do so.
                reward_per_env = reward_per_env - self._decrease_rewards_by

            if self._alive_bonus_schedule is not None:
                # Here we handle the alive bonus schedule.

                # For each environment, increment the timestep.
                t_per_env[active_per_env] += 1

                # For those who are within the full alive bonus time region, increase the scores by the full amount.
                in_full_bonus_t_per_env = active_per_env & (t_per_env >= bonus_t1)
                score_per_env[in_full_bonus_t_per_env] += alive_bonus

                if add_partial_alive_bonus:
                    # Here we handle the partial alive bonus time window.
                    # We first determine which environments are in the partial alive bonus time window.
                    in_partial_bonus_t_per_env = active_per_env & (t_per_env >= bonus_t0) & (t_per_env < bonus_t1)

                    # Here we compute the partial alive bonuses and add those bonuses to the scores.
                    score_per_env[in_partial_bonus_t_per_env] += alive_bonus * (
                        torch.as_tensor(t_per_env[in_partial_bonus_t_per_env] - bonus_t0, dtype=torch.float32)
                        / bonus_t_gap_as_float
                    )

                # Determine which environments just finished their episodes.
                just_finished_per_env = active_per_env & done_per_env

                # Reset the timestep counters of the environments which are just finished.
                t_per_env[just_finished_per_env] = 0

            # For each active environment, increase the score by the reward received.
            score_per_env[active_per_env] += reward_per_env[active_per_env]

            # Update the total timesteps counter.
            total_timesteps += int(torch.sum(active_per_env))

            # Reset the policies whose episodes are done (so that their hidden states become 0).
            policy.reset(done_per_env)

            # Update the number of episodes counter for each environment.
            num_eps_per_env[done_per_env] += 1

            # Solutions with number of completed episodes larger than the number of allowed episodes become inactive.
            # active_per_env[:num_solutions] = num_eps_per_env[:num_solutions] < self._num_episodes
            active_per_env[:num_solutions] = active_per_env[:num_solutions] & (
                num_eps_per_env[:num_solutions] < self._num_episodes
            )

            if not torch.any(active_per_env[:num_solutions]):
                # If there is not a single active solution left, then we exit this loop.
                break

            # For the next iteration of this loop, we normalize the observation.
            # obs_per_env = self._normalize_observation(obs_per_env, mask=active_per_env)
            obs_per_env = normalize(obs_per_env, mask=active_per_env)

        # Update the interaction count and the episode count stored by this VecGymNE instance.
        self._interaction_count += total_timesteps
        self._episode_count += num_solutions * self._num_episodes

        # Compute the fitnesses
        fitnesses = score_per_env[:num_solutions]
        if self._num_episodes > 1:
            fitnesses /= self._num_episodes

        # Assign the scores to the solutions as fitnesses.
        batch.set_evals(fitnesses)

    def get_env(self) -> Optional[gym.Env]:
        """
        Get the gym environment.

        Returns:
            The gym environment if it is built. If not built yet, None.
        """
        return self._env

    def to_policy(self, solution: Iterable, *, with_wrapper_modules: bool = True) -> nn.Module:
        """
        Convert the given solution to a policy.

        Args:
            solution: A solution which can be given as a `torch.Tensor`, as a
                `Solution`, or as any `Iterable`.
            with_wrapper_modules: Whether or not to wrap the policy module
                with helper modules so that observations are normalized
                and actions are clipped to be within the correct boundaries.
                The default and the recommended value is True.
        Returns:
            The policy, as a `torch.nn.Module` instance.
        """
        # Get the gym environment
        env = self._get_env(1)

        # Get the observation space, its lower and higher bounds.
        obs_space = env.single_action_space
        low = obs_space.low
        high = obs_space.high

        # If the lower and higher bounds are not -inf and +inf respectively, then this environment needs clipping.
        needs_clipping = _numpy_arrays_specify_bounds(low, high)

        # Convert the solution to a PyTorch tensor on cpu.
        if isinstance(solution, torch.Tensor):
            solution = solution.to("cpu")
        elif isinstance(solution, Solution):
            solution = solution.values.clone().to("cpu")
        else:
            solution = torch.as_tensor(solution, dtype=torch.float32, device="cpu")

        # Convert the internally stored policy to a PyTorch module.
        result = self._policy.to_torch_module(solution)

        if with_wrapper_modules:
            if self.observation_normalization and (self._obs_stats is not None):
                # If observation normalization is needed and there are collected observation stats, then we wrap the
                # policy with an ObsNormWrapperModule.
                result = ObsNormWrapperModule(result, self._obs_stats)

            if needs_clipping:
                # If clipping is needed, then we wrap the policy with an ActClipWrapperModule
                result = ActClipWrapperModule(result, obs_space)

        return result

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

        # Store the solution and the policy.
        result = {
            "solution": solution,
            "policy": self.to_policy(solution),
        }

        # If available, store the observation stats.
        if self.observation_normalization and (self._obs_stats is not None):
            result["obs_mean"] = self._obs_stats.mean.to("cpu")
            result["obs_stdev"] = self._obs_stats.stdev.to("cpu")
            result["obs_sum"] = self._obs_stats.sum.to("cpu")
            result["obs_sum_of_squares"] = self._obs_stats.sum_of_squares.to("cpu")

        # Some additional data.
        result["interaction_count"] = self.interaction_count
        result["episode_count"] = self.episode_count
        result["time"] = datetime.now()

        if isinstance(self._env_maker, str):
            # If the environment was specified via a string, store the string.
            result["env"] = self._env_maker

        # Store the network architecture.
        result["architecture"] = self._architecture

        # Save the dictionary which stores the data.
        with open(fname, "wb") as f:
            pickle.dump(result, f)

    @property
    def max_num_envs(self) -> Optional[int]:
        """
        Maximum number of environments to be allocated.

        If a maximum number of environments is not set, then None is returned.
        If this problem instance is the main one, then the overall maximum
        number of environments is returned.
        If this problem instance is a remote one (i.e. is on a remote actor)
        then the maximum number of environments for that actor is returned.
        """
        if self.is_main:
            return self._max_num_envs
        else:
            return self._actor_max_num_envs

    def make_net(self, solution: Iterable) -> nn.Module:
        """
        Make a new policy network parameterized by the given solution.
        Note that this parameterized network assumes that the observation
        is already normalized, and it does not do action clipping to ensure
        that the generated actions are within valid bounds.

        To have a policy network which has its own observation normalization
        and action clipping layers, please see the method `to_policy(...)`.

        Args:
            solution: The solution which stores the parameters.
                This can be a Solution instance, or a 1-dimensional tensor,
                or any Iterable of real numbers.
        Returns:
            The policy network, as a PyTorch module.
        """
        return self.to_policy(solution, with_wrapper_modules=False)

    @property
    def network_device(self) -> Optional[Device]:
        """
        The device on which the policy networks will operate.

        Specific to VecGymNE, the network device is determined only
        after receiving the first observation from the reinforcement
        learning environment. Until then, this property has the value
        None.
        """
        return self._simulator_device
