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

"""This module contains logging utilities."""

import os
import pathlib
import pickle
import weakref
from copy import deepcopy
from datetime import datetime
from typing import Any, Iterable, Optional, Union

import torch
from packaging.version import Version
from torch import nn

from .algorithms.searchalgorithm import SearchAlgorithm
from .core import Problem, Solution
from .neuroevolution.net import device_of_module
from .tools import ReadOnlyTensor, clone, is_dtype_object, is_sequence

try:
    import pandas
except ImportError:
    pandas = None


try:
    import sacred
except ImportError:
    sacred = None


try:
    import mlflow
except ImportError:
    mlflow = None


try:
    import neptune

    if hasattr(neptune, "__version__") and (Version(neptune.__version__) < Version("1.0")):
        import neptune.new as neptune
except ImportError:
    neptune = None


try:
    import wandb
except ImportError:
    wandb = None


class Logger:
    """Base class for all logging classes."""

    def __init__(self, searcher: SearchAlgorithm, *, interval: int = 1, after_first_step: bool = False):
        """`__init__(...)`: Initialize the Logger.

        Args:
            searcher: The evolutionary algorithm instance whose progress
                is to be logged.
            interval: Expected as an integer n.
                Logging is to be done at every n iterations.
            after_first_step: Expected as a boolean.
                Meaningful only if interval is set as an integer greater
                than 1. Let us suppose that interval is set as 10.
                If after_first_step is False (which is the default),
                then the logging will be done at steps 10, 20, 30, and so on.
                On the other hand, if after_first_step is True,
                then the logging will be done at steps 1, 11, 21, 31, and so
                on.
        """
        searcher.log_hook.append(self)
        self._interval = int(interval)
        self._after_first_step = bool(after_first_step)
        self._steps_count = 0

    def __call__(self, status: dict):
        if self._after_first_step:
            n = self._steps_count
            self._steps_count += 1
        else:
            self._steps_count += 1
            n = self._steps_count

        if (n % self._interval) == 0:
            self._log(self._filter(status))

    def _filter(self, status: dict) -> dict:
        return status

    def _log(self, status: dict):
        raise NotImplementedError


class PicklingLogger(Logger):
    """
    A logger which periodically pickles the current result of the search.

    The pickled data includes the current center solution and the best solution
    (if available). If the problem being solved is a reinforcement learning
    task, then the pickled data also includes the observation normalization
    data and the policy.
    """

    def __init__(
        self,
        searcher: SearchAlgorithm,
        *,
        interval: int,
        directory: Optional[Union[str, pathlib.Path]] = None,
        prefix: Optional[str] = None,
        zfill: int = 6,
        items_to_save: Union[str, Iterable] = (
            "center",
            "best",
            "pop_best",
            "median_eval",
            "mean_eval",
            "pop_best_eval",
        ),
        make_policy_from: Optional[str] = None,
        after_first_step: bool = False,
        verbose: bool = True,
    ):
        """
        `__init__(...)`: Initialize the PicklingLogger.

        Args:
            searcher: The evolutionary algorithm instance whose progress
                is to be pickled.
            interval: Expected as an integer n.
                Pickling is to be done at every n iterations.
            directory: The directory into which the pickle files will be
                placed. If given as None, the current directory will be
                used.
            prefix: The prefix to be used when naming the pickle file.
                If left as None, then the prefix will contain the date,
                time, name of the class of the problem object, and the PID
                of the current process.
            zfill: When naming the pickle file, the generation number will
                also be used. This `zfill` argument is used to determine
                the length of the number of generations string.
                For example, if the current number of generations is
                150, and zfill is 6 (which is the default), then the
                number of generations part of the file name will look
                like this: 'generation000150', the numeric part of the
                string taking 6 characters.
            items_to_save: Items to save from the status dictionary.
                This can be a string, or a sequence of strings.
            make_policy_from: If given as a string, then the solution
                represented by this key in the status dictionary will be
                taken, and converted a PyTorch module with the help of the
                problem's `to_policy(...)` method, and then will be added
                into the pickle file. For example, if this argument is
                given as "center", then the center solution will be converted
                to a policy and saved.
                If this argument is left as None, then, this logger will
                first try to obtain the center solution. If there is no
                center solution, then the logger will try to obtain the
                'pop_best' solution. If that one does not exist either, then
                an error will be raised, and the user will be requested to
                explicitly state a value for the argument `make_policy_from`.
                If the problem object does not provide a `to_policy(...)`,
                method, then this configuration will be ignored, and no
                policy will be saved.
            after_first_step: Expected as a boolean.
                Meaningful only if interval is set as an integer greater
                than 1. Let us suppose that interval is set as 10.
                If after_first_step is False (which is the default),
                then the logging will be done at steps 10, 20, 30, and so on.
                On the other hand, if after_first_step is True,
                then the logging will be done at steps 1, 11, 21, 31, and so
                on.
            verbose: If set as True, then a message will be printed out to
                the standard output every time the pickling happens.
        """
        # Call the initializer of the superclass.
        super().__init__(searcher, interval=interval, after_first_step=after_first_step)

        # Each Logger register itself to the search algorithm's log hook. Additionally, in the case of PicklingLogger,
        # we add this object's final saving method to the search algorithm's end-of-run hook.
        # This is to make sure that the latest generation's result is saved, even when the last generation number
        # does not coincide with the saving period.
        searcher.end_of_run_hook.append(self._final_save)

        # Store a weak reference to the search algorithm.
        self._searcher_ref = weakref.ref(searcher)

        # Store the item keys as a tuple of strings.
        if isinstance(items_to_save, str):
            self._items_to_save = (items_to_save,)
        else:
            self._items_to_save = tuple(items_to_save)

        # Store the status key that will be used to get the current solution for making the policy.
        self._make_policy_from = None if make_policy_from is None else str(make_policy_from)

        if prefix is None:
            # If a file name prefix is not given by the user, then we prepare one using the current date and time,
            # name of the problem type, and the PID of the current process.
            strnow = datetime.now().strftime("%Y-%m-%d-%H.%M.%S")
            probname = type(searcher.problem).__name__
            strpid = str(os.getpid())
            self._prefix = f"{probname}_{strnow}_{strpid}"
        else:
            # If there is a file name prefix given by the user, then we use that name.
            self._prefix = str(prefix)

        if directory is None:
            # If a directory name is not given by the user, then we store the directory name as None.
            self._directory = None
        else:
            # If a directory name is given by the user, then we store it as a string, and make sure that it exists.
            self._directory = str(directory)
            os.makedirs(self._directory, exist_ok=True)

        self._verbose = bool(verbose)
        self._zfill = int(zfill)

        self._last_generation: Optional[int] = None
        self._last_file_name: Optional[str] = None

    def _within_dir(self, fname: Union[str, pathlib.Path]) -> str:
        fname = str(fname)
        if self._directory is not None:
            fname = os.path.join(self._directory, fname)
        return fname

    @classmethod
    def _as_cpu_tensor(cls, x: Any) -> Any:
        if isinstance(x, Solution):
            x = cls._as_cpu_tensor(x.values)
        elif isinstance(x, torch.Tensor):
            with torch.no_grad():
                x = x.cpu().clone()
            if isinstance(x, ReadOnlyTensor):
                x = x.as_subclass(torch.Tensor)
        return x

    def _log(self, status: dict):
        self.save()

    def _final_save(self, status: dict):
        # Get the stored search algorithm
        searcher: Optional[SearchAlgorithm] = self._searcher_ref()

        if searcher is not None:
            # If the search algorithm object is still alive, then we check its generation number
            current_gen = searcher.step_count

            if (self._last_generation is None) or (current_gen > self._last_generation):
                # If there was not a save at all, or the latest save was for a previous generation,
                # then save the current status of the search.
                self.save()

    def save(self, fname: Optional[Union[str, pathlib.Path]] = None) -> str:
        """
        Pickle the current status of the evolutionary search.

        If this PicklingLogger was initialized with a `directory` argument,
        then the pickle file will be put into that directory.

        Args:
            fname: The name of the pickle file to be created.
                This can be left as None if the user wishes this PicklingLogger
                to determine the name of the file.
                Alternatively, an explicit name for this file can be specified
                via this argument.
        Returns:
            The name of the pickle file that was generated, as a string.
        """
        # Get the stored search algorithm
        searcher: Optional[SearchAlgorithm] = self._searcher_ref()

        if searcher is not None:
            # If the search algorithm object is still alive, then take the problem object from it.
            problem: Problem = searcher.problem

            # Initialize the data dictionary that will contain the objects to be put into the pickle file.
            data = {}

            for item_to_save in self._items_to_save:
                # For each item key, get the object with that key from the algorithm's status dictionary, and then put
                # that object into our data dictionary.
                if item_to_save in searcher.status:
                    data[item_to_save] = self._as_cpu_tensor(searcher.status[item_to_save])

            if (
                hasattr(problem, "observation_normalization")
                and hasattr(problem, "get_observation_stats")
                and problem.observation_normalization
            ):
                # If the problem object has observation normalization, then put the normalizer object to the data
                # dictionary.
                data["obs_stats"] = problem.get_observation_stats().to("cpu")

            if hasattr(problem, "to_policy"):
                # If the problem object has the method `to_policy(...)`, then we will generate a policy from the
                # current solution, and put that policy into the data dictionary.

                if self._make_policy_from is None:
                    # If the user did not specify the status key of the solution that will be converted to a policy,
                    # then, we first check if the search algorithm has a "center" item, and if not, we check if it
                    # has a "pop_best" item.

                    if "center" in searcher.status:
                        # The status key of the search algorithm has "center". We declare "center" as the key of the
                        # current solution.
                        policy_key = "center"
                    elif "pop_best" in searcher.status:
                        # The status key of the search algorithm has "pop_best". We declare "pop_best" as the key of
                        # the current solution.
                        policy_key = "pop_best"
                    else:
                        # The status key of the search algorithm contains neither a "center" solution and nor a
                        # "pop_best" solution. So, we raise an error.
                        raise ValueError(
                            "PicklingLogger did not receive an explicit value for its `make_policy_from` argument."
                            " The status dictionary of the search algorithm has neither 'center' nor 'pop_best'."
                            " Therefore, it is not clear which status item is to be used for making a policy."
                            " Please try instantiating a PicklingLogger with an explicit `make_policy_from` value."
                        )
                else:
                    # This is the case where the user explicitly specified (via a status key) which solution will be
                    # used for making the current policy. We declare that key as the key to use.
                    policy_key = self._make_policy_from

                # Get the solution.
                policy_solution = searcher.status[policy_key]

                # Make a policy from the solution
                policy = problem.to_policy(policy_solution)

                if isinstance(policy, nn.Module) and (device_of_module(policy) != torch.device("cpu")):
                    # If the created policy is a PyTorch module, and this module is not on the cpu, then we make
                    # a copy of this module on the cpu.
                    policy = clone(policy).to("cpu")

                # We put this policy into our data dictionary.
                data["policy"] = policy

                # We put the datetime-related information
                first_step_datetime = searcher.first_step_datetime
                if first_step_datetime is not None:
                    now = datetime.now()
                    elapsed = now - first_step_datetime
                    data["beginning_time"] = first_step_datetime
                    data["now"] = now
                    data["elapsed"] = elapsed

            if fname is None:
                # If a file name was not given, then we generate one.
                num_gens = str(searcher.step_count).zfill(self._zfill)
                fname = f"{self._prefix}_generation{num_gens}.pickle"

            # We prepend the specified directory name, if given.
            fname = self._within_dir(fname)

            # Here, the pickle file is created and the data is saved.
            with open(fname, "wb") as f:
                pickle.dump(data, f)

            # Store the most recently saved generation number
            self._last_generation = searcher.step_count

            # Store the name of the last pickle file
            self._last_file_name = fname

            # Report to the user that the save was successful
            if self._verbose:
                print("Saved to", fname)

            # Return the file name
            return fname
        else:
            return None

    @property
    def last_generation(self) -> Optional[int]:
        """
        Get the last generation for which a pickle file was created.
        If no pickle file is created yet, the result will be None.
        """
        return self._last_generation

    @property
    def last_file_name(self) -> Optional[str]:
        """
        Get the name of the last pickle file.
        If no pickle file is created yet, the result will be None.
        """
        return self._last_file_name

    def unpickle_last_file(self) -> dict:
        """
        Unpickle the most recently made pickle file.
        The file itself will not be modified.
        Its contents will be returned.
        """
        with open(self._last_file_name, "rb") as f:
            return pickle.load(f)


class ScalarLogger(Logger):
    def _filter(self, status: dict) -> dict:
        short_status = {}
        for k, v in status.items():
            if (not is_sequence(v)) and (not isinstance(v, Solution)):
                short_status[k] = v
        return short_status


class StdOutLogger(ScalarLogger):
    """A logger which prints the status into the screen."""

    def __init__(
        self,
        searcher: SearchAlgorithm,
        *,
        interval: int = 1,
        after_first_step: bool = False,
        leading_keys: Iterable[str] = ("iter",),
    ):
        """`__init__(...)`: Initialize the StdOutLogger.

        Args:
            searcher: The evolutionary algorithm instance whose progress
                is to be logged.
            interval: Expected as an integer n.
                Logging is to be done at every n iterations.
            after_first_step: Expected as a boolean.
                Meaningful only if interval is set as an integer greater
                than 1. Let us suppose that interval is set as 10.
                If after_first_step is False (which is the default),
                then the logging will be done at steps 10, 20, 30, and so on.
                On the other hand, if after_first_step is True,
                then the logging will be done at steps 1, 11, 21, 31, and so
                on.
            leading_keys: A sequence of strings where each string is a status
                key. When printing the status, these keys will be shown first.
        """
        super().__init__(searcher, interval=interval, after_first_step=after_first_step)
        self._leading_keys = list(leading_keys)
        self._leading_keys_set = set(self._leading_keys)

    def _log(self, status: dict):
        max_key_length = max([len(str(k)) for k in status.keys()])

        def report(k, v):
            nonlocal max_key_length
            print(str(k).rjust(max_key_length), ":", v)

        for k in self._leading_keys:
            if k in status:
                v = status[k]
                report(k, v)
        for k, v in status.items():
            if k not in self._leading_keys_set:
                report(k, v)
        print()


if pandas is not None:

    class PandasLogger(ScalarLogger):
        """A logger which collects status information and
        generates a pandas.DataFrame at the end.
        """

        def __init__(self, searcher: SearchAlgorithm, *, interval: int = 1, after_first_step: bool = False):
            """`__init__(...)`: Initialize the PandasLogger.

            Args:
                searcher: The evolutionary algorithm instance whose progress
                    is to be logged.
                interval: Expected as an integer n.
                    Logging is to be done at every n iterations.
                after_first_step: Expected as a boolean.
                    Meaningful only if interval is set as an integer greater
                    than 1. Let us suppose that interval is set as 10.
                    If after_first_step is False (which is the default),
                    then the logging will be done at steps 10, 20, 30, and
                    so on. On the other hand, if after_first_step is True,
                    then the logging will be done at steps 1, 11, 21, 31, and
                    so on.
            """
            super().__init__(searcher, interval=interval, after_first_step=after_first_step)
            self._data = []

        def _log(self, status: dict):
            self._data.append(deepcopy(status))

        def to_dataframe(self, *, index: Optional[str] = "iter") -> pandas.DataFrame:
            """Generate a pandas.DataFrame from the collected
            status information.

            Args:
                index: The column to be set as the index.
                    If passed as None, then no index will be set.
                    The default is "iter".
            """
            result = pandas.DataFrame(self._data)
            if index is not None:
                result.set_index(index, inplace=True)
            return result


if sacred is not None:
    ExpOrRun = Union[sacred.Experiment, sacred.run.Run]

    class SacredLogger(ScalarLogger):
        """A logger which stores the status via the Run object of sacred."""

        def __init__(
            self,
            searcher: SearchAlgorithm,
            run: ExpOrRun,
            result: Optional[str] = None,
            *,
            interval: int = 1,
            after_first_step: bool = False,
        ):
            """`__init__(...)`: Initialize the SacredLogger.

            Args:
                searcher: The evolutionary algorithm instance whose progress
                    is to be logged.
                run: An instance of `sacred.run.Run` or `sacred.Experiment`,
                    using which the progress will be logged.
                result: The key in the status dictionary whose associated
                    value will be registered as the current result
                    of the experiment.
                    If left as None, no result will be registered.
                interval: Expected as an integer n.
                    Logging is to be done at every n iterations.
                after_first_step: Expected as a boolean.
                    Meaningful only if interval is set as an integer greater
                    than 1. Let us suppose that interval is set as 10.
                    If after_first_step is False (which is the default),
                    then the logging will be done at steps 10, 20, 30, and
                    so on. On the other hand, if after_first_step is True,
                    then the logging will be done at steps 1, 11, 21, 31,
                    and so on.
            """
            super().__init__(searcher, interval=interval, after_first_step=after_first_step)
            self._result = result
            self._run = run

        def _log(self, status: dict):
            for k, v in status.items():
                self._run.log_scalar(k, v)
            if self._result is not None:
                self._run.result = status[self._result]


if mlflow is not None:
    MlflowID = Union[str, bytes, int]

    class MlflowLogger(ScalarLogger):
        """A logger which stores the status via Mlflow."""

        def __init__(
            self,
            searcher: SearchAlgorithm,
            client: Optional[mlflow.tracking.MlflowClient] = None,
            run: Union[mlflow.entities.Run, Optional[MlflowID]] = None,
            *,
            interval: int = 1,
            after_first_step: bool = False,
        ):
            """`__init__(...)`: Initialize the MlflowLogger.

            Args:
                searcher: The evolutionary algorithm instance whose progress
                    is to be logged.
                client: The MlflowClient object whose log_metric() method
                    will be used for logging. This can be passed as None,
                    in which case mlflow.log_metrics() will be used instead.
                    Please note that, if a client is provided, the `run`
                    argument is required as well.
                run: Expected only if a client is provided.
                    This is the mlflow Run object (an instance of
                    mlflow.entities.Run), or the ID of the mlflow run.
                interval: Expected as an integer n.
                    Logging is to be done at every n iterations.
                after_first_step: Expected as a boolean.
                    Meaningful only if interval is set as an integer greater
                    than 1. Let us suppose that interval is set as 10.
                    If after_first_step is False (which is the default),
                    then the logging will be done at steps 10, 20, 30, and
                    so on. On the other hand, if after_first_step is True,
                    then the logging will be done at steps 1, 11, 21, 31,
                    and so on.
            """

            super().__init__(searcher, interval=interval, after_first_step=after_first_step)

            self._client = client
            self._run_id: Optional[MlflowID] = None

            if self._client is None:
                if run is not None:
                    raise ValueError("Received `run`, but `client` is missing")
            else:
                if run is None:
                    raise ValueError("Received `client`, but `run` is missing")
                if isinstance(run, mlflow.entities.Run):
                    self._run_id = run.info.run_id
                else:
                    self._run_id = run

        def _log(self, status: dict):
            if self._client is None:
                mlflow.log_metrics(status, step=self._steps_count)
            else:
                for k, v in status.items():
                    self._client.log_metric(self._run_id, k, v, step=self._steps_count)


if neptune is not None:

    class NeptuneLogger(ScalarLogger):
        """A logger which stores the status via neptune."""

        def __init__(
            self,
            searcher: SearchAlgorithm,
            run: Optional[neptune.Run] = None,
            *,
            interval: int = 1,
            after_first_step: bool = False,
            group: Optional[str] = None,
            **neptune_kwargs,
        ):
            """`__init__(...)`: Initialize the NeptuneLogger.

            Args:
                searcher: The evolutionary algorithm instance whose progress
                    is to be logged.
                run: A `neptune.new.run.Run` instance using which the status
                    will be logged. If None, then a new run will be created.
                interval: Expected as an integer n.
                    Logging is to be done at every n iterations.
                after_first_step: Expected as a boolean.
                    Meaningful only if interval is set as an integer greater
                    than 1. Let us suppose that interval is set as 10.
                    If after_first_step is False (which is the default),
                    then the logging will be done at steps 10, 20, 30, and so on.
                    On the other hand, if after_first_step is True,
                    then the logging will be done at steps 1, 11, 21, 31, and so
                    on.
                group: Into which group will the metrics be stored.
                    For example, if the status keys to be logged are "score" and
                    "elapsed", and `group` is set as "training", then the metrics
                    will be sent to neptune with the keys "training/score" and
                    "training/elapsed". `group` can also be left as None,
                    in which case the status will be sent to neptune with the
                    key names unchanged.
                **neptune_kwargs: Any additional keyword arguments to be passed
                    to `neptune.init_run()` when creating a new run.
                    For example, `project="my-project"` or `tags=["my-tag"]`.
            """
            super().__init__(searcher, interval=interval, after_first_step=after_first_step)
            self._group = group
            if run is None:
                self._run = neptune.init_run(**neptune_kwargs)
            else:
                self._run = run

        @property
        def run(self) -> neptune.Run:
            return self._run

        def _log(self, status: dict):
            for k, v in status.items():
                target_key = k if self._group is None else self._group + "/" + k
                self._run[target_key].log(v)


if wandb is not None:

    class WandbLogger(ScalarLogger):
        """A logger which stores the status to Weights & Biases."""

        def __init__(
            self,
            searcher: SearchAlgorithm,
            init: Optional[bool] = True,
            *,
            interval: int = 1,
            after_first_step: bool = False,
            group: Optional[str] = None,
            **wandb_kwargs,
        ):
            """`__init__(...)`: Initialize the WandbLogger.

            Args:
                searcher: The evolutionary algorithm instance whose progress
                    is to be logged.
                init: Run `wandb.init()` in the logger initialization
                interval: Expected as an integer n.
                    Logging is to be done at every n iterations.
                after_first_step: Expected as a boolean.
                    Meaningful only if interval is set as an integer greater
                    than 1. Let us suppose that interval is set as 10.
                    If after_first_step is False (which is the default),
                    then the logging will be done at steps 10, 20, 30, and so on.
                    On the other hand, if after_first_step is True,
                    then the logging will be done at steps 1, 11, 21, 31, and so
                    on.
                group: Into which group will the metrics be stored.
                    For example, if the status keys to be logged are "score" and
                    "elapsed", and `group` is set as "training", then the metrics
                    will be sent to W&B with the keys "training/score" and
                    "training/elapsed". `group` can also be left as None,
                    in which case the status will be sent to W&B with the
                    key names unchanged.
                **wandb_kwargs: If `init` is `True` any additional keyword argument
                    will be passed to `wandb.init()`.
                    For example, WandbLogger(searcher, project=my-project, entity=my-organization)
                    will result in calling `wandb.init(project=my-project, entity=my-organization)`
            """
            super().__init__(searcher, interval=interval, after_first_step=after_first_step)
            self._group = group
            if init:
                wandb.init(**wandb_kwargs)

        def _log(self, status: dict):
            log_status = dict()
            for k, v in status.items():
                target_key = k if self._group is None else self._group + "/" + k
                log_status[target_key] = v

            wandb.log(log_status)
