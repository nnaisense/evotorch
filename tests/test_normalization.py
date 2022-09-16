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


from typing import Iterable

import numpy as np
import torch

from evotorch.neuroevolution.net.runningnorm import RunningNorm
from evotorch.neuroevolution.net.runningstat import RunningStat
from evotorch.testing import assert_allclose


def _compute_stats(observations: Iterable) -> dict:
    # Compute stats from a collection of observations.
    # The observations are expected as a 2-dimensional tensor or numpy array.

    if isinstance(observations, np.ndarray):

        def convert(x):
            return x.numpy()

    else:

        def convert(x):
            return x

    observations = torch.as_tensor(observations, dtype=torch.float32)

    # Get the sum of observations, sum of squares, and the number of observations
    sum_of_obs = torch.sum(observations, dim=0)
    sum_of_squares = torch.sum(observations**2, dim=0)
    count = observations.shape[0]

    # Compute E[x] and E[x^2]
    E_x = sum_of_obs / count
    E_x2 = sum_of_squares / count

    # Compute the variance as E[x^2] - E[x]^2
    # Additionally, clip the variance so that it cannot go below 1e-2.
    variance = E_x2 - (E_x**2)
    variance = torch.max(variance, torch.tensor(1e-2, dtype=torch.float32))

    # The mean is E[x] and the standard deviation is the square root of the variance.
    mean = E_x
    stdev = torch.sqrt(variance)

    return dict(
        sum=convert(sum_of_obs),
        sum_of_squares=convert(sum_of_squares),
        count=count,
        mean=convert(mean),
        stdev=convert(stdev),
    )


def test_running_stat():
    tolerance = 1e-4

    # Create some random observations
    all_obs = np.random.randn(10, 5)

    # Get the stats of the observations
    stats = _compute_stats(all_obs)
    mean = stats["mean"]
    stdev = stats["stdev"]

    # Now that we know the mean and the standard deviation of our observations, we instantiate a RunningStat,
    # feed observations to it, and confirm that it also reports the same mean and the same standard deviation.
    rs = RunningStat()

    for obs in all_obs:
        # Feed each observation to the RunningStat instance.
        rs.update(obs)

    # Confirm that the mean and the standard deviation are reported correctly.
    assert_allclose(rs.mean, mean, atol=tolerance)
    assert_allclose(rs.stdev, stdev, atol=tolerance)

    # We now instantiate multiple RunningStat instances: rs1, rs2, and rs3. Each of them will get its own portion of
    # the observations. Then, we will add each of these RunningStat instances to yet another RunningStat.
    # Finally, we will test whether or not this last RunningStat, after collecting the data from rs1, rs2, and rs3,
    # will report the mean and the standard deviation correctly.
    rs1 = RunningStat()
    rs2 = RunningStat()
    rs3 = RunningStat()

    obs_batch1 = all_obs[:3]
    obs_batch2 = all_obs[3:7]
    obs_batch3 = all_obs[7:]

    for obs in obs_batch1:
        rs1.update(obs)

    for obs in obs_batch2:
        rs2.update(obs)

    for obs in obs_batch3:
        rs3.update(obs)

    final_rs = RunningStat()
    final_rs.update(rs3)
    final_rs.update(rs2)
    final_rs.update(rs1)

    # Assert that `final_rs` reports the correct mean and standard deviation
    assert_allclose(final_rs.mean, mean, atol=tolerance)
    assert_allclose(final_rs.stdev, stdev, atol=tolerance)


def _make_random_booleans(*shape):
    x = torch.randn(*shape)
    return x > 0


def test_running_norm():
    tolerance = 1e-4

    # We imagine an environment whose observation length is as follows:
    obs_length = 10

    # We imagine a batch of environments
    num_envs = 50

    # We will have this many batch of observations
    num_timesteps = 30

    # For each timestep, we generate a batch of observations
    all_obs = torch.randn(num_timesteps, num_envs, obs_length)

    # For each timestep, we generate a mask tensor
    # This mask tensor will tell us which environment is active at that timestep
    masks = _make_random_booleans(num_timesteps, num_envs)

    # Now, we compute the overall stats over all observations over all 'active' environments over all timesteps
    overall_stats = _compute_stats(all_obs.reshape(-1, obs_length)[masks.reshape(-1)])
    overall_mean = overall_stats["mean"]
    overall_stdev = overall_stats["stdev"]

    # Make a new RunningNorm instance
    rn = RunningNorm(shape=obs_length, dtype=torch.float32)

    for t in range(num_timesteps):
        # For each timestep, feed the batch of observations and the mask tensor to the normalizer
        obs = all_obs[t]
        active = masks[t]
        rn.update(obs, mask=active)

    # Assert that the reported mean and stdev of the RunningNorm instance are correct
    assert_allclose(rn.mean, overall_mean, atol=tolerance)
    assert_allclose(rn.stdev, overall_stdev, atol=tolerance)
