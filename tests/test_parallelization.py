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

import pytest
import torch

import evotorch as et


class MyProblem(et.Problem):
    def __init__(self, num_actors: int):
        super().__init__(
            objective_sense="min",
            initial_bounds=(-10.0, 10.0),
            solution_length=5,
            num_actors=int(num_actors),
        )

    def _evaluate_batch(self, batch: et.SolutionBatch):
        x = batch.values
        norms = torch.linalg.norm(x, dim=-1)
        batch.set_evals(norms)

    def get_actor_index(self) -> int:
        return self.actor_index

    def get_env(self):
        class DummyEnv:
            def __init__(self, actor_index: int):
                self.actor_index = actor_index

            def get_parent_actor_index(self) -> int:
                return self.actor_index

        result = DummyEnv(self.actor_index)
        return result


def test_remote_access():
    num_actors = 4
    problem = MyProblem(num_actors=num_actors)
    expected_indices = set(range(num_actors))
    all_actor_indices = set(problem.all_remote_problems().get_actor_index())
    all_env_actor_indices = set(problem.all_remote_envs().get_parent_actor_index())
    assert all_actor_indices == expected_indices
    assert all_env_actor_indices == expected_indices
    assert len(problem.actors) == num_actors
