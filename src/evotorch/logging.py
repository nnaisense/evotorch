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

from copy import copy, deepcopy
from typing import Callable, Iterable, Optional, Union

from .algorithms.searchalgorithm import SearchAlgorithm
from .core import Problem, Solution
from .tools import is_sequence

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
    import neptune.new as neptune
except ImportError:
    neptune = None


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
            run,
            *,
            interval: int = 1,
            after_first_step: bool = False,
            group: Optional[str] = None,
        ):
            """`__init__(...)`: Initialize the NeptuneLogger.

            Args:
                searcher: The evolutionary algorithm instance whose progress
                    is to be logged.
                run: A `neptune.new.run.Run` instance using which the status
                    will be logged.
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
            """
            super().__init__(searcher, interval=interval, after_first_step=after_first_step)
            self._run = run
            self._group = group

        def _log(self, status: dict):
            for k, v in status.items():
                target_key = k if self._group is None else self._group + "/" + k
                self._run[target_key].log(v)
