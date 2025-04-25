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

from evotorch import Problem
from evotorch.algorithms import PGPE
from evotorch.logging import PandasLogger, PicklingLogger, StdOutLogger
from evotorch.neuroevolution import GymNE


@pytest.fixture
def searcher_sphere():
    def sphere(x: torch.Tensor) -> torch.Tensor:
        return torch.sum(x.pow(2.0))

    problem = Problem("min", sphere, solution_length=10, initial_bounds=(-1, 1))

    searcher = PGPE(
        problem,
        popsize=10,
        radius_init=2.25,
        center_learning_rate=0.2,
        stdev_learning_rate=0.1,
    )

    return searcher


@pytest.fixture
def searcher_gym():
    problem = GymNE(
        # Name of the environment
        env="CartPole-v1",
        # Linear policy mapping observations to actions
        network="Linear(obs_length, act_length)",
        # Use 4 available CPUs. Note that you can modify this value,
        # or use 'max' to exploit all available GPUs
        num_actors=4,
        observation_normalization=True,
    )

    searcher = PGPE(
        problem,
        popsize=10,
        radius_init=2.25,
        center_learning_rate=0.2,
        stdev_learning_rate=0.1,
    )

    return searcher


@pytest.mark.parametrize(
    "logger_class, kwargs",
    [
        (PandasLogger, {}),
        (PandasLogger, {"after_first_step": True}),
        (PicklingLogger, {"interval": 2, "directory": "{dir}"}),
        (PicklingLogger, {"interval": 2, "directory": "{dir}", "after_first_step": True}),
        (StdOutLogger, {}),
        (StdOutLogger, {"after_first_step": True}),
    ],
)
def test_basic_loggers(logger_class, kwargs, tmpdir, searcher_sphere):
    kwargs = {k: v.format(dir=tmpdir) if isinstance(v, str) else v for k, v in kwargs.items()}

    logger_class(searcher_sphere, **kwargs)

    searcher_sphere.run(5)


def test_pickling_logger_on_rl(tmpdir, searcher_gym):
    PicklingLogger(searcher_gym, interval=1, directory=tmpdir)

    searcher_gym.run(2)

    assert "best" in searcher_gym.status
    assert searcher_gym.step_count == 2


def test_pandas_logger(searcher_gym):
    logger = PandasLogger(searcher_gym, after_first_step=True)

    searcher_gym.run(2)

    assert "best" in searcher_gym.status
    assert searcher_gym.step_count == 2
    assert logger.to_dataframe() is not None
    assert logger.to_dataframe().shape == (2, 7)


@pytest.mark.parametrize(
    "set_client, set_run, raises",
    [
        (False, True, True),
        (False, False, False),
        (True, True, False),
        (True, False, True),
    ],
)
def test_mlflow_logger(set_client, set_run, raises, tmpdir, searcher_gym):
    try:
        import mlflow

        from evotorch.logging import MlflowLogger

        mlflow.set_tracking_uri(f"file://{tmpdir}")
        experiment_id = mlflow.create_experiment(f"exp_{set_client:d}{set_run:d}{raises:d}")
        client = mlflow.tracking.MlflowClient() if set_client else None
        run = mlflow.start_run(experiment_id=experiment_id)
        if not set_run:
            run = None

        if raises:
            with pytest.raises(ValueError):
                MlflowLogger(searcher_gym, client=client, run=run)
        else:
            MlflowLogger(searcher_gym, client=client, run=run)
            searcher_gym.run(2)

            assert "best" in searcher_gym.status
            assert searcher_gym.step_count == 2

        mlflow.end_run()
    except ImportError:
        pass


@pytest.mark.parametrize(
    "pass_run, kwargs",
    [
        (False, {}),
        (True, {}),
    ],
)
def test_neptune_logger(pass_run, kwargs, searcher_gym, tmpdir):
    try:
        import os

        import neptune.new as neptune

        from evotorch.logging import NeptuneLogger

        kwargs["project"] = "test/project"
        kwargs["mode"] = "debug"

        # Change current working directory temporarily
        # to avoid creating a .neptune directory
        cwd = os.getcwd()
        os.chdir(tmpdir)

        if pass_run:
            run = neptune.init_run(**kwargs)
            logger = NeptuneLogger(searcher_gym, run=run)
        else:
            logger = NeptuneLogger(searcher_gym, **kwargs)

        searcher_gym.run(2)

        logger.run.stop()

        # Restore current working directory
        os.chdir(cwd)

        assert "best" in searcher_gym.status
        assert searcher_gym.step_count == 2
    except ImportError:
        pass


@pytest.mark.parametrize(
    "pass_run, kwargs",
    [
        (False, {}),
        (True, {}),
    ],
)
def test_wandb_logger(pass_run, kwargs, searcher_gym, tmpdir):
    try:
        import wandb

        from evotorch.logging import WandbLogger

        kwargs["project"] = "test_project"
        kwargs["mode"] = "disabled"
        kwargs["dir"] = str(tmpdir)

        if pass_run:
            wandb.init(**kwargs)
            WandbLogger(searcher_gym, init=False)
        else:
            WandbLogger(searcher_gym, **kwargs)

        searcher_gym.run(2)

        assert "best" in searcher_gym.status
        assert searcher_gym.step_count == 2
    except ImportError:
        pass
