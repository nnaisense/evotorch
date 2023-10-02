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


# flake8 doesn't like sacred configurations, so it is disabled for this script
# flake8: noqa
import os
import pickle
from datetime import datetime
from typing import Optional

import numpy as np
import sacred

from evotorch.algorithms import PGPE
from evotorch.logging import SacredLogger, StdOutLogger
from evotorch.neuroevolution import GymNE
from evotorch.tools import SuppressSacredExperiment

""" This script uses sacred and gym to re-implement experiments from the paper:

Nihat Engin Toklu, Paweł Liskowski, and Rupesh Kumar Srivastava.
"ClipUp: a simple and powerful optimizer for distribution-based policy evolution."
International Conference on Parallel Problem Solving from Nature. Springer, Cham, 2020.

An extended version of this paper can be found online:
https://arxiv.org/abs/2008.02387

To run an experiment use:

    python rl_clipup.py with [CONFIGURATION NAME]

Where the [CONFIGURATION_NAME] is one of the following:

lunarlander - the LunarLander continuous environment
walker - The Walker2D environment
humanoid - The Humanoid environment
pybullet_humanoid - The pybullet Humanoid environment using the tinytraj_humanoid_bullet.py file for a modified environment with shorter trajectories

It should be noted that this script works for v4 of the Walker2D and Humanoid environments, rather than v2 used in the paper.
Therefore results may very mildly.

It should also be noted that, for when using the ClipUp optimizer, this script uses 75% of the maximum speed as the default learning rate
for the mean of the search distribution.
Although the originally published ClipUp study uses 50% of the maximum speed in most of its experiments, it is also reported that, based on additional
experiments on MuJoCo Humanoid, bringing the mean learning rate closer to the maximum speed might result in a performance boost
(see the section 5.4 of the arXiv version of the paper).
Following those additional results, 75% of the maximum speed (rather than 50%) is used as the default in this script.
This script allows one to manually set the mean learning rate for ClipUp (as euclidean distance, not as percentages) via the hyperparameter
named clipup_alpha.

If you wish to have the sacred log stored in .json format do:

    python rl_clipup.py -F [RESULTS_DIR] with [CONFIGURATION NAME]

You can freely modify the configuration using sacred.
For example, if you wish to run the Humanoid-v2 environment with the Adam optimizer and alpha=0.15 (as studied in the paper) do:

    python rl_clipup.py with humanoid optimizer="adam" adam_alpha=0.15

Once finished, you can view your result using the lightweight 'rl_enjoy.py' script. To visualize a trained agent simply do:

    python rl_enjoy.py [AGENT FILE]

You modify the number of repeated episodes using

    python rl_enjoy.py [AGENT FILE] [REPEATS]

and setting [REPEATS] to -1 gives you indefinite rendering until you kill the process.
"""

if __name__ == "__main__":
    ex = sacred.Experiment()
else:
    ex = SuppressSacredExperiment()


def simplified_env_name(env_name: str) -> str:
    if ":" in env_name:
        colon_pos = env_name.find(":")
        return env_name[colon_pos + 1 :]
    else:
        return env_name


@ex.config
def cfg():
    # String reference to environment. See: https://www.gymlibrary.ml/
    env_name = "NOT SPECIFIED YET"
    # Used to mask the name of the environment e.g. in storage and logs
    actual_env_name = ""

    # String expression of policy e.g. the below is a 1-hidden layer MLP with 16 neurons and Tanh activation.
    policy = "Linear(obs_length, 16) >> Tanh() >> Linear(16, act_length)"

    # Choice of optimizer ('clipup' for using ClipUp, 'adam' for using Adam)
    optimizer = "clipup"

    # Configuration for ClipUp optimizer
    max_speed = 0.15  # Maximum speed for ClipUp
    clipup_alpha = max_speed * 0.75  # Learning rate (i.e. step size) for ClipUp
    momentum = 0.9  # Momentum for ClipUp
    radius_init = max_speed * 15  # Initial radius for the search distribution

    # Configuration for Adam optimizer -- note that it uses the same radius_init as the ClipUp optimizer
    adam_alpha = 4e-3  # Learning rate (i.e. step size) for Adam
    beta1 = 0.9  # beta1 for Adam
    beta2 = 0.999  # beta2 for Adam
    epsilon = 1e-8  # epsilon for Adam

    # Other configuration of PGPE/evolution
    # Learning rate for the standard deviation (omega in the ClipUp paper)
    stdev_learning_rate = 0.1
    # allowed ratio of change in standard deviation (see page 3)
    stdev_max_change = 0.2
    # Base population size (lambda in the paper)
    popsize = 1000
    # Maximum number of interactions (T in the paper) -- if -1, then ignored
    num_interactions = -1
    # Maximum population size (lambda_max in the paper) -- if -1, then ignored
    popsize_max = -1
    # Total number of generations to run evolution
    num_generations = 1000

    # Length of an episode. -1 means episode length will be the default specified by the environment.
    episode_length = -1

    # Configuration of the problem
    # Whether to use observation normalization
    observation_normalization = False
    # Scalar modification of step-wise reward. Can be used to remove 'alive bonus'
    decrease_rewards_by = float("nan")
    # Whether to run the algorithm in distributed mode. If True, then the behaviour may diverge from the paper but performance may be better.
    distributed = False
    # Number of actors -- if -1, then the maximum number of actors is used
    num_actors = -1
    # How frequently to save the policy described by the center of the search distribution. If 'last' then only the final generation is saved.
    save_interval = "last"
    # Whether to evaluate the center of the search distribution.
    evaluate_center = True
    # How many samples to draw when evaluating the center of the search distribution. If -1, then the number of actors is used.
    center_num_evaluations = -1


@ex.named_config
def lunarlander():
    env_name = "LunarLanderContinuous-v2"
    policy = "Linear(obs_length, act_length, False)"  # False means no bias, as in the paper
    max_speed = 0.3
    adam_alpha = 0.2
    popsize = 200
    popsize_max = -1  # Lunar lander experiments used fixed popsize
    num_interactions = -1
    observation_normalization = False  # Lunar lander experiments did not use observation normalization
    num_generations = 50


@ex.named_config
def walker():
    env_name = "Walker2d-v4"
    policy = "Linear(obs_length, act_length)"
    max_speed = 1.5e-2
    adam_alpha = 4e-3
    popsize = 100
    popsize_max = 800
    num_interactions = 0.75 * 1000 * popsize
    observation_normalization = True
    decrease_rewards_by = 1.0
    num_generations = 500


@ex.named_config
def humanoid():
    env_name = "Humanoid-v4"
    policy = "Linear(obs_length, act_length)"
    max_speed = 0.015
    adam_alpha = 6e-4
    popsize = 200
    popsize_max = 3200
    num_interactions = 0.75 * 1000 * popsize
    observation_normalization = True
    decrease_rewards_by = 5.0
    num_generations = 500


@ex.named_config
def pybullet_humanoid():
    env_name = "wrapped_humanoid_bullet:TinyTrajHumanoidBulletEnv-v0"
    actual_env_name = "wrapped_humanoid_bullet:WrappedHumanoidBulletEnv-v0"
    policy = "Linear(obs_length, 64) >> Tanh() >> Linear(64, act_length)"
    max_speed = 0.15
    adam_alpha = 6e-4  # No experiment was ever done with adam for this problem
    popsize = 10000
    popsize_max = 80000
    num_interactions = 0.75 * 200 * popsize
    observation_normalization = True


def none_if_nan(x: float) -> Optional[float]:
    return None if np.isnan(x) else x


def positive_or_default(x, default):
    return x if x > 0 else default


@ex.automain
def main(_config: dict):
    # Get the environment name
    env_name = _config["env_name"]
    actual_env_name = _config["actual_env_name"]
    if actual_env_name == "":
        actual_env_name = env_name

    # Instantiate the problem class
    problem = GymNE(
        env=env_name,
        network=_config["policy"],
        observation_normalization=_config["observation_normalization"],
        decrease_rewards_by=none_if_nan(_config["decrease_rewards_by"]),
        num_actors=positive_or_default(_config["num_actors"], "max"),
        episode_length=positive_or_default(_config["episode_length"], None),
    )

    # Instantiate the searcher
    # Branching on the optimizer name to give different optimizer configs
    if _config["optimizer"] == "clipup":
        optimizer_config = {
            "max_speed": _config["max_speed"],
            "momentum": _config["momentum"],
        }
        center_learning_rate = _config["clipup_alpha"]
    elif _config["optimizer"] == "adam":
        optimizer_config = {
            "beta1": _config["beta1"],
            "beta2": _config["beta2"],
            "epsilon": _config["epsilon"],
        }
        center_learning_rate = _config["adam_alpha"]

    # Using the optimizer config
    searcher = PGPE(
        problem,
        popsize=_config["popsize"],
        center_learning_rate=center_learning_rate,
        stdev_learning_rate=_config["stdev_learning_rate"],
        stdev_max_change=_config["stdev_max_change"],
        optimizer=_config["optimizer"],
        optimizer_config=optimizer_config,
        radius_init=_config["radius_init"],
        num_interactions=positive_or_default(_config["num_interactions"], None),
        popsize_max=positive_or_default(_config["popsize_max"], None),
        distributed=_config["distributed"],
    )

    if _config["evaluate_center"]:
        # Create a test problem instance -- note the difference in configuration

        test_problem = GymNE(
            env=actual_env_name,  # Using the actual environment name, rather than a modified version
            network=_config["policy"],
            observation_normalization=_config["observation_normalization"],
            decrease_rewards_by=0.0,  # Not changing the rewards
            num_actors=1,  # Using only 1 actor avoid clogging the CPU
        )

        # Get the number of center evaluations
        center_num_evaluations = _config["center_num_evaluations"]
        if center_num_evaluations == -1:
            center_num_evaluations = problem._num_actors

        # Create a hook for the searcher
        def evaluate_center():
            if _config["observation_normalization"]:
                # Update the observation normalization stats
                test_problem.set_observation_stats(problem.get_observation_stats())

            # Get the center of the search distribution
            center = searcher.status["center"]

            # As a batch
            center_batch = problem.generate_batch(center_num_evaluations)
            center_batch.set_values(center.unsqueeze(0).repeat(center_num_evaluations, 1))

            # Evaluate
            test_problem.evaluate(center_batch)

            # Return as dict
            return {"center_mean_eval": center_batch.evals.mean().item()}

        # Append hook to searcher
        searcher.after_step_hook.append(evaluate_center)

    # Get some information for logging the center of the search distribution
    save_interval = _config["save_interval"]
    pid = os.getpid()
    now_str = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
    short_env = simplified_env_name(env_name)
    fname_prefix = f"{short_env}_{now_str}"
    # Temporary storage before logging to sacred
    fname = f"{fname_prefix}_{pid}.pickle"

    # Create a logger to standard out and sacred
    StdOutLogger(searcher)
    SacredLogger(searcher, ex, "mean_eval")

    # Step the searcher
    for generation in range(1, 1 + _config["num_generations"]):
        searcher.step()

        # If saving the policy at a fixed interval, check for the interval + save
        if isinstance(save_interval, int) and (generation % save_interval) == 0:
            center_policy = problem.to_policy(searcher.status["center"])
            with open(fname, "wb") as f:
                pickle.dump(
                    {
                        "env_name": actual_env_name,
                        "policy": center_policy,
                    },
                    f,
                )
            artifact_name = f"{fname_prefix}_{pid}_generation{generation}.pickle"
            ex.add_artifact(fname, artifact_name)
            print("Saved", fname)

    # If only saving the policy from the final generation, do it now
    if isinstance(save_interval, str) and save_interval == "last":
        center_policy = problem.to_policy(searcher.status["center"])
        with open(fname, "wb") as f:
            pickle.dump(
                {
                    "env_name": actual_env_name,
                    "policy": center_policy,
                },
                f,
            )
        artifact_name = f"{fname_prefix}_{pid}_generation{generation}.pickle"
        ex.add_artifact(fname, artifact_name)
        print("Saved", fname)

    # While not strictly necessary, ray can sometimes make mistakes with garbage collection of actors. This method is called for safety
    problem.kill_actors()
