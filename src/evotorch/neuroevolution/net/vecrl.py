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

"""
This namespace provides various vectorized reinforcement learning utilities.
"""


from collections.abc import Mapping
from copy import deepcopy
from typing import Any, Callable, Iterable, Optional, Union

import gym
import numpy as np
import torch
from functorch import vmap
from gym.spaces import Box, Discrete, Space
from gym.vector import SyncVectorEnv
from torch import nn
from torch.nn import utils as nnu

try:
    import jax
    import jax.dlpack
    from jax import numpy as jnp
except ImportError:
    jax = None
    jnp = None


if jax is not None:
    JaxArray = jnp.ndarray

    def is_jax_array(x: Any) -> bool:
        return isinstance(x, JaxArray)

    def jax_to_torch(x: JaxArray) -> torch.Tensor:
        x = jax.dlpack.to_dlpack(x)
        x = torch.utils.dlpack.from_dlpack(x)
        return x

    def torch_to_jax(x: torch.Tensor) -> JaxArray:
        x = torch.utils.dlpack.to_dlpack(x)
        x = jax.dlpack.from_dlpack(x)
        return x

else:

    def _jax_is_missing():
        raise ImportError("The module `jax` is missing.")

    class JaxArray:
        def __init__(self, *args, **kwargs):
            _jax_is_missing()

    def is_jax_array(x: Any) -> bool:
        return False

    def jax_to_torch(x: JaxArray) -> torch.Tensor:
        _jax_is_missing()

    def torch_to_jax(x: torch.Tensor) -> JaxArray:
        _jax_is_missing()


try:
    import brax
    import brax.envs
except ImportError:
    brax = None


if brax is not None:
    from brax.envs import Env as BraxEnv

    def is_brax_env(env: Any) -> bool:
        return isinstance(env, BraxEnv)

else:

    def _brax_is_missing():
        raise ImportError("The module `brax` is missing.")

    class BraxEnv:
        def __init__(self, *args, **kwargs):
            _brax_is_missing()

    def is_brax_env(env: Any) -> bool:
        return False


def array_type(x: Any, fallback: Optional[str] = None) -> str:
    """
    Get the type of an array as a string ("jax", "torch", or "numpy").
    If the type of the array cannot be determined and a fallback is provided,
    then the fallback value will be returned.

    Args:
        x: The array whose type will be determined.
        fallback: Fallback value, as a string, which will be returned if the
            array type cannot be determined.
    Returns:
        The array type as a string ("jax", "torch", or "numpy").
    Raises:
        TypeError: if the array type cannot be determined and a fallback
            value is not provided.
    """
    if is_jax_array(x):
        return "jax"
    elif isinstance(x, torch.Tensor):
        return "torch"
    elif isinstance(x, np.ndarray):
        return "numpy"
    elif fallback is not None:
        return fallback
    else:
        raise TypeError(f"The object has an unrecognized type: {type(x)}")


def convert_from_torch(x: torch.Tensor, array_type: str) -> Any:
    """
    Convert the given PyTorch tensor to an array of the specified type.

    Args:
        x: The PyTorch array that will be converted.
        array_type: Type to which the PyTorch tensor will be converted.
            Expected as one of these strings: "jax", "torch", "numpy".
    Returns:

    """
    if array_type == "torch":
        return x
    elif array_type == "jax":
        return torch_to_jax(x)
    elif array_type == "numpy":
        return x.cpu().numpy()
    else:
        raise ValueError(f"Unrecognized array type: {array_type}")


def convert_to_torch(x: Any) -> torch.Tensor:
    """
    Convert the given array to PyTorch tensor.

    Args:
        x: Array to be converted. Can be a JAX array, a numpy array,
            a PyTorch tensor (in which case the input tensor will be
            returned as it is) or any Iterable object.
    Returns:
        The PyTorch counterpart of the given array.
    """
    if isinstance(x, torch.Tensor):
        return x
    elif is_jax_array(x):
        return jax_to_torch(x)
    elif isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    else:
        return torch.as_tensor(x)


def _must_be_supported_space(space: Space):
    if not isinstance(space, (Box, Discrete)):
        raise TypeError(
            f"Actions and observations must be in one of these spaces: Box, Discrete."
            f" Encountered an unsupported space: {type(space)}."
        )


def convert_to_torch_bool(x: Any) -> torch.Tensor:
    """
    Convert the given array to a PyTorch tensor of bools.

    If the given object is an array of floating point numbers, then, values
    that are near to 0.0 (with a tolerance of 1e-4) will be converted to
    False, and the others will be converted to True.
    If the given object is an array of integers, then zero values will be
    converted to False, and non-zero values will be converted to True.
    If the given object is an array of booleans, then no change will be made
    to those boolean values.

    The given object can be a JAX array, a numpy array, or a PyTorch tensor.
    The result will always be a PyTorch tensor.

    Args:
        x: Array to be converted.
    Returns:
        The array converted to a PyTorch tensor with its dtype set as bool.
    """
    x = convert_to_torch(x)
    if x.dtype == torch.bool:
        pass  # nothing to do
    elif "float" in str(x.dtype):
        x = torch.abs(x) > 1e-4
    else:
        x = torch.as_tensor(x, dtype=torch.bool)

    return x


class TorchWrapper(gym.Wrapper):
    """
    A gym wrapper which ensures that the actions, observations, rewards, and
    the 'done' values are expressed as PyTorch tensors.
    """

    def __init__(
        self,
        env: Union[gym.Env],
        *,
        force_classic_api: bool = False,
        discrete_to_continuous_act: bool = False,
        clip_actions: bool = False,
        **kwargs,
    ):
        """
        `__init__(...)`: Initialize the TorchWrapper.

        Args:
            env: The gym environment to be wrapped.
            force_classic_api: Set this as True if you would like to enable
                the classic API. In the classic API, the `reset(...)` method
                returns only the observation and the `step(...)` method
                returns 4 elements (not 5).
            discrete_to_continuous_act: When this is set as True and the
                wrapped environment has a Discrete action space, this wrapper
                will transform the action space to Box. A Discrete-action
                environment with `n` actions will be converted to a Box-action
                environment where the action length is `n`.
                The index of the largest value within the action vector will
                be applied to the underlying environment.
            clip_actions: Set this as True if you would like to clip the given
                actions so that they conform to the declared boundaries of the
                action space.
            kwargs: Expected in the form of additional keyword arguments.
                These additional keyword arguments are passed to the
                superclass.
        """
        super().__init__(env, **kwargs)

        # Declare the variable that will store the array type of the underlying environment.
        self.__array_type: Optional[str] = None

        if hasattr(env, "single_observation_space"):
            # If the underlying environment has the attribute "single_observation_space",
            # then this is a vectorized environment.
            self.__vectorized = True

            # Get the observation and action spaces.
            obs_space = env.single_observation_space
            act_space = env.single_action_space
        else:
            # If the underlying environment has the attribute "single_observation_space",
            # then this is a non-vectorized environment.
            self.__vectorized = False

            # Get the observation and action spaces.
            obs_space = env.observation_space
            act_space = env.action_space

        # Ensure that the observation and action spaces are supported.
        _must_be_supported_space(obs_space)
        _must_be_supported_space(act_space)

        # Store the choice of the user regarding "force_classic_api".
        self.__force_classic_api = bool(force_classic_api)

        if isinstance(act_space, Discrete) and discrete_to_continuous_act:
            # The underlying action space is Discrete and `discrete_to_continuous_act` is given as True.
            # Therefore, we convert the action space to continuous (to Box).

            # Take the shape and the dtype of the discrete action space.
            single_action_shape = (act_space.n,)
            single_action_dtype = torch.from_numpy(np.array([], dtype=act_space.dtype)).dtype

            # We store the integer dtype of the environment.
            self.__discrete_dtype = single_action_dtype

            if self.__vectorized:
                # If the environment is vectorized, we declare the new `action_space` and the `single_action_space`
                # for the enviornment.
                action_shape = (env.num_envs,) + single_action_shape
                self.single_action_space = Box(float("-inf"), float("inf"), shape=single_action_shape, dtype=np.float32)
                self.action_space = Box(float("-inf"), float("inf"), shape=action_shape, dtype=np.float32)
            else:
                # If the environment is not vectorized, we declare the new `action_space` for the environment.
                self.action_space = Box(float("-inf"), float("inf"), shape=single_action_shape, dtype=np.float32)
        else:
            # This is the case where we do not transform the action space.
            # The discrete dtype will not be used, so, we set it as None.
            self.__discrete_dtype = None

        if isinstance(act_space, Box) and clip_actions:
            # If the action space is Box and the wrapper is configured to clip the actions, then we store the lower
            # and the upper bounds for the actions.
            self.__act_lb = torch.from_numpy(act_space.low)
            self.__act_ub = torch.from_numpy(act_space.high)
        else:
            # If there will not be any action clipping, then we store the lower and the upper bounds as None.
            self.__act_lb = None
            self.__act_ub = None

    @property
    def array_type(self) -> Optional[str]:
        """
        Get the array type of the wrapped environment.
        This can be "jax", "torch", or "numpy".
        """
        return self.__array_type

    def __infer_array_type(self, observation):
        if self.__array_type is None:
            # If the array type is not determined yet, set it as the array type of the received observation.
            # If the observation has an unrecognized type, set the array type as "numpy".
            self.__array_type = array_type(observation, "numpy")

    def reset(self, *args, **kwargs):
        """Reset the environment"""

        # Call the reset method of the wrapped environment.
        reset_result = self.env.reset(*args, **kwargs)

        if isinstance(reset_result, tuple):
            # If we received a tuple of two elements, then we assume that this is the new gym API.
            # We note that we received an info dictionary.
            got_info = True
            # We keep the received observation and info.
            observation, info = reset_result
        else:
            # If we did not receive a tuple, then we assume that this is the old gym API.
            # We note that we did not receive an info dictionary.
            got_info = False
            # We keep the received observation.
            observation = reset_result
            # We did not receive an info dictionary, so, we set it as an empty dictionary.
            info = {}

        # We understand the array type of the underlying environment from the first observation.
        self.__infer_array_type(observation)

        # Convert the observation to a PyTorch tensor.
        observation = convert_to_torch(observation)

        if self.__force_classic_api:
            # If the option `force_classic_api` was set as True, then we only return the observation.
            return observation
        else:
            # Here we handle the case where `force_classic_api` was set as False.
            if got_info:
                # If we got an additional info dictionary, we return it next to the observation.
                return observation, info
            else:
                # If we did not get any info dictionary, we return only the observation.
                return observation

    def step(self, action, *args, **kwargs):
        """Take a step in the environment"""

        if self.__array_type is None:
            # If the array type is not known yet, then probably `reset()` has not been called yet.
            # We raise an error.
            raise ValueError(
                "Could not understand what type of array this environment works with."
                " Perhaps the `reset()` method has not been called yet?"
            )

        if self.__discrete_dtype is not None:
            # If the wrapped environment is discrete-actioned, then we take the integer counterpart of the action.
            action = torch.argmax(action, dim=-1).to(dtype=self.__discrete_dtype)

        if self.__act_lb is not None:
            # The internal variable `__act_lb` having a value other than None means that the initialization argument
            # `clip_actions` was given as True.
            # Therefore, we clip the actions.
            self.__act_lb = self.__act_lb.to(action.device)
            self.__act_ub = self.__act_ub.to(action.device)
            action = torch.max(action, self.__act_lb)
            action = torch.min(action, self.__act_ub)

        # Convert the action tensor to the expected array type of the underlying environment.
        action = convert_from_torch(action, self.__array_type)

        # Perform the step and get the result.
        result = self.env.step(action, *args, **kwargs)

        if not isinstance(result, tuple):
            # If the `step(...)` method returned anything other than tuple, we raise an error.
            raise TypeError(f"Expected a tuple as the result of the `step()` method, but received a {type(result)}")

        if len(result) == 5:
            # If the result is a tuple of 5 elements, then we note that we are using the new API.
            using_new_api = True
            # Take the observation, reward, two boolean variables done and done2 indicating that the episode(s)
            # has/have ended, and additional info.
            # `done` indicates whether or not the episode(s) reached terminal state(s).
            # `done2` indicates whether or not the episode(s) got truncated because of the timestep limit.
            observation, reward, done, done2, info = result
        elif len(result) == 4:
            # If the result is a tuple of 5 elements, then we note that we are not using the new API.
            using_new_api = False
            # Take the observation, reward, the done boolean flag, and additional info.
            observation, reward, done, info = result
            done2 = None
        else:
            raise ValueError(f"Unexpected number of elements were returned from step(): {len(result)}")

        # Convert the observation, reward, and done variables to PyTorch tensors.
        observation = convert_to_torch(observation)
        reward = convert_to_torch(reward)
        done = convert_to_torch_bool(done)
        if done2 is not None:
            done2 = convert_to_torch_bool(done2)

        if self.__force_classic_api:
            # This is the case where the initialization argument `force_classic_api` was set as True.
            if done2 is not None:
                # We combine the terminal state and truncation signals into a single boolean tensor indicating
                # whether or not the episode(s) ended.
                done = done | done2
            # Return 4 elements, compatible with the classic gym API.
            return observation, reward, done, info
        else:
            # This is the case where the initialization argument `force_classic_api` was set as False.
            if using_new_api:
                # If we are using the new API, then we return the 5-element result.
                return observation, reward, done, done2, info
            else:
                # If we are using the new API, then we return the 4-element result.
                return observation, reward, done, info


def make_brax_env(
    env_name: str,
    *,
    force_classic_api: bool = False,
    num_envs: Optional[int] = None,
    discrete_to_continuous_act: bool = False,
    clip_actions: bool = False,
    **kwargs,
) -> TorchWrapper:
    """
    Make a brax environment and wrap it via TorchWrapper.

    Args:
        env_name: Name of the brax environment, as string (e.g. "humanoid").
        force_classic_api: Whether or not the classic gym API is to be used.
        num_envs: Batch size for the vectorized environment.
        discrete_to_continuous_act: Whether or not the the discrete action
            space of the environment is to be converted to a continuous one.
            This does nothing if the environment's action space is not
            discrete.
        clip_actions: Whether or not the actions should be explicitly clipped
            so that they stay within the declared action boundaries.
        kwargs: Expected in the form of additional keyword arguments, these
            are passed to the environment.
    Returns:
        The brax environment, wrapped by TorchWrapper.
    """

    if brax is not None:
        config = {}
        config.update(kwargs)
        if num_envs is not None:
            config["batch_size"] = num_envs
        env = brax.envs.create_gym_env(env_name, **config)
        env = TorchWrapper(
            env,
            force_classic_api=force_classic_api,
            discrete_to_continuous_act=discrete_to_continuous_act,
            clip_actions=clip_actions,
        )
        return env
    else:
        _brax_is_missing()


def make_gym_env(
    env_name: str,
    *,
    force_classic_api: bool = False,
    num_envs: Optional[int] = None,
    discrete_to_continuous_act: bool = False,
    clip_actions: bool = False,
    **kwargs,
) -> TorchWrapper:
    """
    Make gym environments and wrap them via a SyncVectorEnv and a TorchWrapper.

    Args:
        env_name: Name of the gym environment, as string (e.g. "Humanoid-v4").
        force_classic_api: Whether or not the classic gym API is to be used.
        num_envs: Batch size for the vectorized environment.
        discrete_to_continuous_act: Whether or not the the discrete action
            space of the environment is to be converted to a continuous one.
            This does nothing if the environment's action space is not
            discrete.
        clip_actions: Whether or not the actions should be explicitly clipped
            so that they stay within the declared action boundaries.
        kwargs: Expected in the form of additional keyword arguments, these
            are passed to the environment.
    Returns:
        The gym environments, wrapped by a TorchWrapper.
    """

    def make_the_env():
        return gym.make(env_name, **kwargs)

    env_fns = [make_the_env for _ in range(num_envs)]
    vec_env = TorchWrapper(
        SyncVectorEnv(env_fns),
        force_classic_api=force_classic_api,
        discrete_to_continuous_act=discrete_to_continuous_act,
        clip_actions=clip_actions,
    )

    return vec_env


def make_vector_env(
    env_name: str,
    *,
    force_classic_api: bool = False,
    num_envs: Optional[int] = None,
    discrete_to_continuous_act: bool = False,
    clip_actions: bool = False,
    **kwargs,
) -> TorchWrapper:
    """
    Make a new vectorized environment and wrap it via TorchWrapper.

    Args:
        env_name: Name of the gym environment, as string.
            If the string starts with "gym::" (e.g. "gym::Humanoid-v4", etc.),
            then it is assumed that the target environment is a classical gym
            environment which will first be wrapped via a SyncVectorEnv and
            then via a TorchWrapper.
            If the string starts with "brax::" (e.g. "brax::humanoid", etc.),
            then it is assumed that the target environment is a brax
            environment which will be wrapped via TorchWrapper.
            If the string does not contain "::" at all (e.g. "Humanoid-v4"),
            then it is assumed that the target environment is a classical gym
            environment. Therefore, "gym::Humanoid-v4" and "Humanoid-v4"
            are equivalent.
        force_classic_api: Whether or not the classic gym API is to be used.
        num_envs: Batch size for the vectorized environment.
        discrete_to_continuous_act: Whether or not the the discrete action
            space of the environment is to be converted to a continuous one.
            This does nothing if the environment's action space is not
            discrete.
        clip_actions: Whether or not the actions should be explicitly clipped
            so that they stay within the declared action boundaries.
        kwargs: Expected in the form of additional keyword arguments, these
            are passed to the environment.
    Returns:
        The gym environments, wrapped by a TorchWrapper.
    """

    env_parts = str(env_name).split("::", maxsplit=1)

    if len(env_parts) == 0:
        raise ValueError(f"Invalid value for `env_name`: {repr(env_name)}")
    elif len(env_parts) == 1:
        fn = make_gym_env
    elif len(env_parts) == 2:
        env_name = env_parts[1]
        if env_parts[0] == "gym":
            fn = make_gym_env
        elif env_parts[0] == "brax":
            fn = make_brax_env
        else:
            invalid_value = env_parts[0] + "::"
            raise ValueError(
                f"The argument `env_name` starts with {repr(invalid_value)}, implying that the environment is stored"
                f" in a registry named {repr(env_parts[0])}."
                f" However, the registry {repr(env_parts[0])} is not recognized."
                f" Supported environment registries are: 'gym', 'brax'."
            )
    else:
        assert False, "Unexpected value received from len(env_parts)"

    return fn(
        env_name,
        force_classic_api=force_classic_api,
        num_envs=num_envs,
        discrete_to_continuous_act=discrete_to_continuous_act,
        clip_actions=clip_actions,
        **kwargs,
    )


MaskOrIndices = Union[int, Iterable]


def reset_tensors(x: Any, indices: MaskOrIndices):
    """
    Reset the specified regions of the given tensor(s) as 0.

    Note that the resetting is performed in-place, which means, the provided tensors are modified.

    The regions are determined by the argument `indices`, which can be a sequence of booleans (in which case it is
    interpreted as a mask), or a sequence of integers (in which case it is interpreted as the list of indices).

    For example, let us imagine that we have the following tensor:

    ```python
    import torch

    x = torch.tensor(
        [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9, 10, 11],
            [12, 13, 14, 15],
        ],
        dtype=torch.float32,
    )
    ```

    If we wish to reset the rows with indices 0 and 2, we could use:

    ```python
    reset_tensors(x, [0, 2])
    ```

    The new value of `x` would then be:

    ```
    torch.tensor(
        [
            [0, 0, 0, 0],
            [4, 5, 6, 7],
            [0, 0, 0, 0],
            [12, 13, 14, 15],
        ],
        dtype=torch.float32,
    )
    ```

    The first argument does not have to be a single tensor.
    Instead, it can be a container (i.e. a dictionary-like object or an iterable) that stores tensors.
    In this case, each tensor stored by the container will be subject to resetting.
    In more details, each tensor within the iterable(s) and each tensor within the value part of the dictionary-like
    object(s) will be reset.

    As an example, let us assume that we have the following collection:

    ```python
    a = torch.tensor(
        [
            [0, 1],
            [2, 3],
            [4, 5],
        ],
        dtype=torch.float32,
    )

    b = torch.tensor(
        [
            [0, 10, 20],
            [30, 40, 50],
            [60, 70, 80],
        ],
        dtype=torch.float32,
    )

    c = torch.tensor(
        [
            [100],
            [200],
            [300],
        ],
        dtype=torch.float32,
    )

    d = torch.tensor([-1, -2, -3], dtype=torch.float32)

    my_tensors = [a, {"1": b, "2": (c, d)}]
    ```

    To clear the regions with indices, e.g, (1, 2), we could do:

    ```python
    reset_tensors(my_tensors, [1, 2])
    ```

    and the result would be:

    ```
    >>> print(a)
    torch.tensor(
        [
            [0, 1],
            [0, 0],
            [0, 0],
        ],
        dtype=torch.float32,
    )

    >>> print(b)
    torch.tensor(
        [
            [0, 10, 20],
            [0, 0, 0],
            [0, 0, 0],
        ],
        dtype=torch.float32,
    )

    >>> print(c)
    c = torch.tensor(
        [
            [100],
            [0],
            [0],
        ],
        dtype=torch.float32,
    )

    >>> print(d)
    torch.tensor([-1, 0, 0], dtype=torch.float32)
    ```

    Args:
        x: A tensor or a collection of tensors, whose values are subject to resetting.
        indices: A sequence of integers or booleans, specifying which regions of the tensor(s) will be reset.
    """
    if isinstance(x, torch.Tensor):
        # If the first argument is a tensor, then we clear it according to the indices we received.
        x[indices] = 0
    elif isinstance(x, (str, bytes, bytearray)):
        # str, bytes, and bytearray are the types of `Iterable` that we do not wish to process.
        # Therefore, we explicitly add a condition for them here, and explicitly state that nothing should be done
        # when instances of them are encountered.
        pass
    elif isinstance(x, Mapping):
        # If the first argument is a Mapping (i.e. a dictionary-like object), then, for each value part of the
        # Mapping instance, we call this function itself.
        for key, value in x.items():
            reset_tensors(value, indices)
    elif isinstance(x, Iterable):
        # If the first argument is an Iterable (e.g. a list, a tuple, etc.), then, for each value contained by this
        # Iterable instance, we call this function itself.
        for value in x:
            reset_tensors(value, indices)


class Policy:
    """
    A Policy for deciding the actions for a reinforcement learning environment.

    This can be seen as a stateful wrapper around a PyTorch module.

    Let us assume that we have the following PyTorch module:

    ```python
    from torch import nn

    net = nn.Linear(5, 8)
    ```

    which has 48 parameters (when all the parameters are flattened).
    Let us randomly generate a parameter vector for our module `net`:

    ```python
    parameters = torch.randn(48)
    ```

    We can now prepare a policy:

    ```python
    policy = Policy(net)
    policy.set_parameters(parameters)
    ```

    If we generate a random observation:

    ```python
    observation = torch.randn(5)
    ```

    We can receive our action as follows:

    ```python
    action = policy(observation)
    ```

    If the PyTorch module that we wish to wrap is a recurrent network (i.e.
    a network which expects an optional second argument for the hidden state,
    and returns a second value which represents the updated hidden state),
    then, the hidden state is automatically managed by the Policy instance.

    Let us assume that we have a recurrent network named `recnet`.

    ```python
    policy = Policy(recnet)
    policy.set_parameters(parameters_of_recnet)
    ```

    In this case, because the hidden state of the network is internally
    managed, the usage is still the same with our previous non-recurrent
    example:

    ```python
    action = policy(observation)
    ```

    When using a recurrent module on multiple episodes, it is important
    to reset the hidden state of the network. This is achieved by the
    reset method:

    ```python
    policy.reset()
    action1 = policy(observation1)

    # action2 will be computed with the hidden state generated by the
    # previous forward-pass.
    action2 = policy(observation2)

    policy.reset()

    # action3 will be computed according to the renewed hidden state.
    action3 = policy(observation3)
    ```

    Both for non-recurrent and recurrent networks, it is possible to
    perform vectorized operations. For now, let us return to our
    first non-recurrent example:

    ```python
    net = nn.Linear(5, 8)
    ```

    Instead of generating only one parameter vector, we now generate
    a batch of parameter vectors. Let us say that our batch size is 10:

    ```python
    batch_of_parameters = torch.randn(10, 48)
    ```

    Like we did in the non-batched examples, we can do:

    ```python
    policy = Policy(net)
    policy.set_parameters(batch_of_parameters)
    ```

    Because we are now in the batched mode, `policy` now expects a batch
    of observations and will return a batch of actions:

    ```python
    batch_of_observations = torch.randn(10, 5)
    batch_of_actions = policy(batch_of_observations)
    ```

    When doing vectorized reinforcement learning with a recurrent module,
    it can be the case that only some of the environments are finished,
    and therefore it is necessary to reset the hidden states associated
    with those environments only. The `reset(...)` method of Policy
    has a second argument to specify which of the recurrent network
    instances are to be reset. For example, if the episodes of the
    environments with indices 2 and 5 are about to restart (and therefore
    we wish to reset the states of the networks with indices 2 and 5),
    then, we can do:

    ```python
    policy.reset(torch.tensor([2, 5]))
    ```
    """

    def __init__(self, net: Union[str, Callable, nn.Module], **kwargs):
        """
        `__init__(...)`: Initialize the Policy.

        Args:
            net: The network to be wrapped by the Policy object.
                This can be a string, a Callable (e.g. a `torch.nn.Module`
                subclass), or a `torch.nn.Module` instance.
                When this argument is a string, the network will be
                created with the help of the function
                `evotorch.neuroevolution.net.str_to_net(...)` and then
                wrapped. Please see the `str_to_net(...)` function's
                documentation for details regarding how a network structure
                can be expressed via strings.
            kwargs: Expected in the form of additional keyword arguments,
                these keyword arguments will be passed to the provided
                Callable object (if the argument `net` is a Callable)
                or to `str_to_net(...)` (if the argument `net` is a string)
                at the moment of generating the network.
                If the argument `net` is a `torch.nn.Module` instance,
                having any additional keyword arguments will trigger an
                error, because the network is already instantiated and
                therefore, it is not possible to pass these keyword arguments.
        """
        from ..net import str_to_net
        from ..net.functional import ModuleExpectingFlatParameters, make_functional_module

        if isinstance(net, str):
            self.__module = str_to_net(net, **kwargs)
        elif isinstance(net, nn.Module):
            if len(kwargs) > 0:
                raise ValueError(
                    f"When the network is given as an `nn.Module` instance, extra network arguments cannot be used"
                    f" (because the network is already instantiated)."
                    f" However, these extra keyword arguments were received: {kwargs}."
                )
            self.__module = net
        elif isinstance(net, Callable):
            self.__module = net(**kwargs)
        else:
            raise TypeError(
                f"The class `Policy` expected a string or an `nn.Module` instance, or a Callable, but received {net}"
                f" (whose type is {type(net)})."
            )

        self.__fmodule: ModuleExpectingFlatParameters = make_functional_module(self.__module)
        self.__state: Any = None
        self.__parameters: Optional[torch.Tensor] = None

    def set_parameters(self, parameters: torch.Tensor, indices: Optional[MaskOrIndices] = None, *, reset: bool = True):
        """
        Set the parameters of the policy.

        Args:
            parameters: A 1-dimensional or a 2-dimensional tensor containing
                the flattened parameters to be used with the neural network.
                If the given parameters are two-dimensional, then, given that
                the leftmost size of the parameter tensor is `n`, the
                observations will be expected in a batch with leftmost size
                `n`, and the returned actions will also be in a batch,
                again with the leftmost size `n`.
            indices: For when the parameters were previously given via a
                2-dimensional tensor, provide this argument if you would like
                to change only some rows of the previously given parameters.
                For example, if `indices` is given as `torch.tensor([2, 4])`
                and the argument `parameters` is given as a 2-dimensional
                tensor with leftmost size 2, then the rows with indices
                2 and 4 will be replaced by these new parameters provided
                via the argument `parameters`.
            reset: If given as True, the hidden states of the networks whose
                parameters just changed will be reset. If `indices` was not
                provided at all, then this means that the parameters of all
                networks are modified, in which case, all the hidden states
                will be reset.
                If given as False, no such resetting will be done.
        """
        if self.__parameters is None:
            if indices is not None:
                raise ValueError(
                    "The argument `indices` can be used only if network parameters were previously specified."
                    " However, it seems that the method `set_parameters(...)` was not called before."
                )
            self.__parameters = parameters
        else:
            if indices is None:
                self.__parameters = parameters
            else:
                self.__parameters[indices] = parameters

        if reset:
            self.reset(indices)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass the given observations through the network.

        Args:
            x: The observations, as a PyTorch tensor.
                If the parameters were given (via the method
                `set_parameters(...)`) as a 1-dimensional tensor, then this
                argument is expected to store a single observation.
                If the parameters were given as a 2-dimensional tensor,
                then, this argument is expected to store a batch of
                observations, and the leftmost size of this observation
                tensor must match with the leftmost size of the parameter
                tensor.
        Returns:
            The output tensor, which represents the action to take.
        """
        if self.__parameters is None:
            raise ValueError("Please use the method `set_parameters(...)` before calling the policy.")

        if self.__state is None:
            further_args = (x,)
        else:
            further_args = (x, self.__state)

        parameters = self.__parameters
        ndim = parameters.ndim
        if ndim == 1:
            result = self.__fmodule(parameters, *further_args)
        elif ndim == 2:
            vmapped = vmap(self.__fmodule)
            result = vmapped(parameters, *further_args)
        else:
            raise ValueError(
                f"Expected the parameters as a 1 or 2 dimensional tensor."
                f" However, the received parameters tensor has {ndim} dimensions."
            )

        if isinstance(result, torch.Tensor):
            return result
        elif isinstance(result, tuple):
            result, state = result
            self.__state = state
            return result
        else:
            raise TypeError(f"The torch module used by the Policy returned an unexpected object: {result}")

    def reset(self, indices: Optional[MaskOrIndices] = None, *, copy: bool = True):
        """
        Reset the hidden states, if the contained module is a recurrent network.

        Args:
            indices: Optionally a sequence of integers or a sequence of
                booleans, specifying which networks' states will be
                reset. If left as None, then the states of all the networks
                will be reset.
            copy: When `indices` is given as something other than None,
                if `copy` is given as True, then the resetting will NOT
                be done in-place. Instead, a new copy of the hidden state
                will first be created, and then the specified regions
                of this new copy will be cleared, and then finally this
                modified copy will be declared as the new hidden state.
                It is a common practice for recurrent neural network
                implementations to return the same tensor both as its
                output and as (part of) its hidden state. With `copy=False`,
                the resetting would be done in-place, and the action
                tensor could be involuntarily reset as well.
                This in-place modification could cause silent bugs
                if the unintended modification on the action tensor
                happens BEFORE the action is sent to the reinforcement
                learning environment.
                To prevent such situations, the default value for the argument
                `copy` is True.
        """
        if indices is None:
            self.__state = None
        else:
            if self.__state is not None:
                with torch.no_grad():
                    if copy:
                        self.__state = deepcopy(self.__state)
                    reset_tensors(self.__state, indices)

    @property
    def parameters(self) -> torch.Tensor:
        """
        The currently used parameters.
        """
        return self.__parameters

    @property
    def h(self) -> Optional[torch.Tensor]:
        """
        The hidden state of the contained recurrent network, if any.

        If the contained recurrent network did not generate a hidden state
        yet, or if the contained network is not recurrent, then the result
        will be None.
        """
        return self.__state

    @property
    def parameter_length(self) -> int:
        """
        Length of the parameter tensor.
        """
        return self.__fmodule.parameter_length

    @property
    def wrapped_module(self) -> nn.Module:
        """
        The wrapped `torch.nn.Module` instance.
        """
        return self.__module

    def to_torch_module(self, parameter_vector: torch.Tensor) -> nn.Module:
        """
        Get a copy of the contained network, parameterized as specified.

        Args:
            parameter_vector: The parameters to be used by the new network.
        Returns:
            Copy of the contained network, as a `torch.nn.Module` instance.
        """
        with torch.no_grad():
            net = deepcopy(self.__module).to(parameter_vector.device)
            nnu.vector_to_parameters(parameter_vector, net.parameters())
        return net
