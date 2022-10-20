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

from evotorch.algorithms import PGPE
from evotorch.logging import PicklingLogger, StdOutLogger
from evotorch.neuroevolution import GymNE

# Specialized Problem class for Gym environments
problem = GymNE(
    env="Humanoid-v4",
    # Linear policy defined using special string syntax supported by EvoTorch
    network="Linear(obs_length, act_length)",
    observation_normalization=True,
    # Humanoid has an "alive bonus" reward at each step, which often hurts EAs, so we remove it
    decrease_rewards_by=5.0,
    # Use all available CPU cores
    num_actors="max",
)

# Run PGPE with ClipUp for 500 steps and log results to standard output at every step
searcher = PGPE(
    problem,
    popsize=200,
    center_learning_rate=0.01125,
    stdev_learning_rate=0.1,
    optimizer="clipup",
    optimizer_config={"max_speed": 0.015},
    radius_init=0.27,
    num_interactions=150000,
    popsize_max=3200,
)
logger = StdOutLogger(searcher)
pickler = PicklingLogger(searcher, interval=10)  # save the current solution at every 10 generations
searcher.run(500)

# Create a policy to test using the final center of the optimized distribution and visualize its behavior
population_center = searcher.status["center"]
policy = problem.to_policy(population_center)
problem.visualize(policy)
