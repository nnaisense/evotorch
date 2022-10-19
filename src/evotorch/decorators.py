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

"""Module defining decorators for evotorch."""

from typing import Callable


def pass_info(fn_or_class: Callable) -> Callable:
    """
    Decorates a callable so that the neuroevolution problem class (e.g. GymNE) will
    pass information regarding the task at hand, in the form of keyword arguments.

    For example, in the case of [GymNE][evotorch.neuroevolution.GymNE] or
    [VecGymNE][evotorch.neuroevolution.VecGymNE], the passed information would
    include dimensions of the observation and action spaces.

    Example:
        ```python
        @pass_info
        class MyModule(nn.Module):
            def __init__(self, obs_length: int, act_length: int, **kwargs):
                # Because MyModule is decorated with @pass_info, it receives
                # keyword arguments related to the environment "CartPole-v0",
                # including obs_length and act_length.
                ...


        problem = GymNE(
            "CartPole-v0",
            network=MyModule,
            ...,
        )
        ```

    Args:
        fn_or_class (Callable): Function or class to decorate

    Returns:
        Callable: Decorated function or class
    """
    setattr(fn_or_class, "__evotorch_pass_info__", True)
    return fn_or_class
