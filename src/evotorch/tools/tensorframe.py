# Copyright 2025 NNAISENSE SA
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


from collections import OrderedDict
from collections.abc import Mapping, Sequence
from io import StringIO
from typing import Any, Callable, Optional, Union

import numpy as np
import torch
from torch.func import vmap

from .recursiveprintable import DEFAULT_MAX_DEPTH_FOR_PRINTING, RecursivePrintable

try:
    import pandas as pd

    _PandasDataFrame = pd.DataFrame

    def _is_pandas_dataframe(obj: Any) -> bool:
        return isinstance(obj, pd.DataFrame)

except ImportError:
    pd = None

    class _PandasDataFrame:
        pass

    def _is_pandas_dataframe(obj: Any) -> bool:
        return False


def _without_torch_dot(s: object) -> str:
    s = str(s)
    unwanted_prefix = "torch."
    if s.startswith(unwanted_prefix):
        s = s[len(unwanted_prefix) :]
    return s


class TensorFrame(RecursivePrintable):
    """
    A structure which allows one to manipulate tensors in a tabular manner.
    The interface of this structure is inspired by the `DataFrame` class
    of the `pandas` library.

    **Motivation.**
    It is a common scenario to have to work with arrays/tensors that are
    associated with each other (e.g. when working on a knapsack problem, we
    could have arrays `A` and `B`, where `A[i]` represents the value of the
    i-th item and `B[i]` represents the weight of the i-th item).
    A practical approach for such cases is to organize those arrays and
    operate on them in a tabular manner. `pandas` is a popular library
    for doing such tabular operations.

    In the context of evolutionary computation, efficiency via vectorization
    and/or parallelization becomes an important concern. For example, if we
    have a fitness function in which a solution vector is evaluated with the
    help of tabular operations, we would like to be able to obtain a batched
    version of that function, so that not just a solution, but an entire
    population can be evaluated efficiently. Another example is when
    developing custom evolutionary algorithms, where solutions, fitnesses,
    and algorithm-specific (or problem-specific) metadata can be organized
    in a tabular manner, and operating on such tabular data has to be
    vectorized and/or parallelized for increased efficiency.

    `TensorFrame` is introduced to address these concerns. In more details,
    with evolutionary computation in mind, it has these features/behaviors:

    (i) The columns of a `TensorFrame` are expressed via PyTorch tensors
    (or via `evotorch.tools.ReadOnlyTensor` instances), allowing one
    to place the tabular data on devices such as cuda.

    (ii) A `TensorFrame` can be placed into a function that is transformed
    via `torch.vmap` or `evotorch.decorators.expects_ndim` or
    `evotorch.decorators.rowwise`, therefore it can work on batched data.

    (iii) Upon being pickled or cloned, a `TensorFrame` applies the `clone`
    method on all its columns and ensures that the cloned tensors have
    minimally sized storages (even when their originals might have shared
    their storages with larger tensors). Therefore, one can send a
    `TensorFrame` to a remote worker (e.g. using `ray` library), without
    having to worry about oversized shared tensor storages.

    (iv) A `TensorFrame` can be placed as an item into an
    `evotorch.tools.ObjectArray` container. Therefore, it can serve as a
    value in a solution of a problem whose `dtype` is `object`.

    **Basic usage.**
    A tensorframe can be instantiated like this:

    ```python
    from evotorch.tools import TensorFrame
    import torch

    my_tensorframe = TensorFrame(
        {
            "COLUMN1": torch.FloatTensor([1, 2, 3, 4]),
            "COLUMN2": torch.FloatTensor([10, 20, 30, 40]),
            "COLUMN3": torch.FloatTensor([-10, -20, -30, -40]),
        }
    )
    ```

    which represents the following tabular data:

    ```text
      float32    float32     <- dtype of the column
        cpu        cpu       <- device of the column

      COLUMN1    COLUMN2    COLUMN3
     =========  =========  =========
        1.0        10.0      -10.0
        2.0        20.0      -20.0
        3.0        30.0      -30.0
        4.0        40.0      -40.0
    ```

    Rows can be picked and re-organized like this:

    ```python
    my_tensorframe = my_tensorframe.pick[[0, 3, 2]]
    ```

    which causes `my_tensorframe` to now store:

    ```text
      float32    float32    float32
        cpu        cpu        cpu

      COLUMN1    COLUMN2    COLUMN3
     =========  =========  =========
        1.0        10.0      -10.0
        4.0        40.0      -40.0
        3.0        30.0      -30.0
    ```

    A tensor of a column can be received like this:

    ```python
    print(my_tensorframe["COLUMN1"])
    # Note: alternatively: print(my_tensorframe.COLUMN1)

    # Prints: torch.tensor([1.0, 4.0, 3.0], dtype=torch.float32)
    ```

    Multiple columns can be received like this:

    ```python
    print(my_tensorframe[["COLUMN1", "COLUMN2"]])

    # Prints:
    #
    #  float32    float32
    #    cpu        cpu
    #
    #  COLUMN1    COLUMN2
    # =========  =========
    #    1.0        10.0
    #    4.0        40.0
    #    3.0        30.0
    ```

    The values of a column can be changed like this:

    ```python
    my_tensorframe.pick[1:, "COLUMN1"] = torch.FloatTensor([7.0, 9.0])
    ```

    which causes `my_tensorframe` to become:

    ```text
      float32    float32    float32
        cpu        cpu        cpu

      COLUMN1    COLUMN2    COLUMN3
     =========  =========  =========
        1.0        10.0      -10.0
        7.0        40.0      -40.0
        9.0        30.0      -30.0
    ```

    Multiple columns can be changed like this:

    ```python
    my_tensorframe.pick[1:, ["COLUMN1", "COLUMN2"]] = TensorFrame(
        {
            "COLUMN1": torch.FloatTensor([11.0, 12.0]),
            "COLUMN2": torch.FloatTensor([44.0, 55.0]),
        }
    )

    # Note: alternatively, the right-hand side can be given as a dictionary:
    # my_tensorframe.pick[1:, ["COLUMN1", "COLUMN2"]] = {
    #     "COLUMN1": torch.FloatTensor([11.0, 12.0]),
    #     "COLUMN2": torch.FloatTensor([44.0, 55.0]),
    # }
    ```

    which causes `my_tensorframe` to become:

    ```text
      float32    float32    float32
        cpu        cpu        cpu

      COLUMN1    COLUMN2    COLUMN3
     =========  =========  =========
        1.0        10.0      -10.0
       11.0        44.0      -40.0
       12.0        55.0      -30.0
    ```

    **Further notes.**

    - A tensor under a TensorFrame column can have more than one dimension.
        Across different columns, the size of the leftmost dimensions must
        match.
    - Unlike a `pandas.DataFrame`, a TensorFrame does not have a special index
        column.
    """

    def __init__(
        self,
        data: Optional[Union[Mapping, "TensorFrame", _PandasDataFrame]] = None,
        *,
        read_only: bool = False,
        device: Optional[Union[torch.device, str]] = None,
    ):
        """
        `__init__(...)`: Initialize the TensorFrame

        Args:
            data: Optionally a TensorFrame, or a dictionary (where the keys
                are column names and the values are column values), or a
                `pandas.DataFrame` object. If provided, the tabular data
                given here are used for initializing the new TensorFrame.
            read_only: Whether or not the newly made TensorFrame will be
                read-only. A read-only TensorFrame's columns will be
                ReadOnlyTensors, and its columns and values will not change.
            device: If left as None, each column can be on a different device.
                If given as a string or a `torch.device`, each column will
                be forcefully moved to this given device.
        """
        from .readonlytensor import as_read_only_tensor

        self.__is_read_only = False
        self.__device = None if device is None else torch.device(device)

        if read_only:

            def prepare_value(x):
                return as_read_only_tensor(x)

        else:

            def prepare_value(x):
                return x

        self.__data = OrderedDict()
        if data is None:
            pass  # no data is given, nothing to do
        elif _is_pandas_dataframe(data):
            for k in data.columns:
                v = data[k]
                self.__setitem__(k, prepare_value(v))
        elif isinstance(data, Mapping):
            for k, v in data.items():
                self.__setitem__(k, prepare_value(v))
        elif isinstance(data, TensorFrame):
            for k, v in data.__data.items():
                self.__setitem__(k, prepare_value(v))
        else:
            raise TypeError(
                "When constructing a new TensorFrame, the argument `data` was expected as a dictionary-like object"
                " (an instance of Mapping), or as a TensorFrame."
                f" However, the encountered object is of type {type(data)}."
            )

        self.__is_read_only = read_only
        self._initialized = True

    def __get_first_tensor(self) -> Optional[torch.Tensor]:
        return None if len(self.__data) == 0 else next(iter(self.__data.values()))

    def __get_default_device_kwargs(self) -> dict:
        return {} if self.__device is None else {"device": self.__device}

    def __get_first_tensor_device_kwargs(self) -> dict:
        first_tensor = self.__get_first_tensor()
        return {} if first_tensor is None else {"device": first_tensor.device}

    def as_tensor(
        self,
        x: Any,
        *,
        to_work_with: Optional[Union[str, np.str_, torch.Tensor]] = None,
        broadcast_if_scalar: bool = False,
    ) -> torch.Tensor:
        """
        Convert the given object `x` to a PyTorch tensor.

        Args:
            x: The object to be converted to a PyTorch tensor.
            to_work_with: Optionally a string, referring to an existing column
                within this TensorFrame, or a PyTorch tensor. The object `x`
                will be converted to a PyTorch tensor on the same device with
                this given column/tensor. If `to_work_with` is left as None,
                `x` will be converted to a PyTorch tensor on the same device
                with this TensorFrame's first column.
            broadcast_if_scalar: If this argument is given as True and if `x`
                is a scalar, its tensor-counterpart will be broadcast to a
                vector of length `n`, where `n` is the number of rows of this
                TensorFrame.
        Returns:
            The tensor counterpart of `x`.
        """
        from .readonlytensor import as_read_only_tensor

        if to_work_with is not None:
            if isinstance(to_work_with, torch.Tensor):
                pass  # nothing to do
            elif isinstance(to_work_with, (str, np.str_)):
                to_work_with = self.__data[str(to_work_with)]
            else:
                raise TypeError(
                    "The argument `to_work_with` was expected as a PyTorch tensor or as a string"
                    " (that refers to one of the columns of this TensorFrame)."
                    f" However, it was received as an instance of {type(to_work_with)}."
                )
            result = torch.as_tensor(x, device=to_work_with.device)
        else:
            if self.__is_read_only:
                convert = as_read_only_tensor
            else:
                convert = torch.as_tensor

            if isinstance(x, torch.Tensor):
                result = convert(x, **(self.__get_default_device_kwargs()))
            else:
                result = convert(x, **(self.__get_first_tensor_device_kwargs()))

        if broadcast_if_scalar and (result.ndim == 0):
            first_tensor = self.__get_first_tensor()
            if first_tensor is None:
                raise ValueError("The first column cannot be given as a scalar.")
            result = result * torch.ones(first_tensor.shape[0], dtype=result.dtype, device=result.device)

        return result

    def __setitem__(self, column_name: Union[str, np.str_], values: Any):
        if self.__is_read_only:
            raise TypeError("Cannot modify a read-only TensorFrame")

        column_name = str(column_name)
        values = self.as_tensor(values, broadcast_if_scalar=True)
        self.__data[column_name] = values

    def __getitem__(
        self,
        column_name_or_mask: Union[str, np.str_, Sequence[Union[str, np.str_]]],
    ) -> torch.Tensor:
        if (isinstance(column_name_or_mask, np.ndarray) and (column_name_or_mask.dtype == np.bool_)) or (
            isinstance(column_name_or_mask, torch.Tensor) and (column_name_or_mask.dtype == torch.bool)
        ):
            return self.pick[column_name_or_mask]
        elif isinstance(column_name_or_mask, (str, np.str_)):
            return self.__data[str(column_name_or_mask)]
        elif isinstance(column_name_or_mask, Sequence):
            result = TensorFrame()
            for col in column_name_or_mask:
                if not isinstance(col, (str, np.str_)):
                    raise TypeError(f"The sequence of column names has an item of this unrecognized type: {type(col)}")
                result[col] = self[col]
            if self.is_read_only:
                result = result.get_read_only_view()
            return result
        else:
            raise TypeError(
                "Expected a column name (as a string) or a sequence of column names, but encountered an instance of"
                f" {type(column_name_or_mask)}."
            )

    def __setattr__(self, attr_name: str, value: Any) -> torch.Tensor:
        if attr_name == "__dict__":
            object.__setattr__(self, attr_name, value)
        elif ("_initialized" in self.__dict__) and (self.__dict__["_initialized"]):
            if attr_name in self.__dict__:
                self.__dict__[attr_name] = value
            elif attr_name in self.__data:
                raise ValueError(
                    f"Please do not use the dot notation to change the column {repr(attr_name)}."
                    f" Hint: you could use: tensorframe[{repr(attr_name)}] = ..."
                )
            else:
                raise ValueError(
                    f"Unknown attribute: {repr(attr_name)}."
                    f" Hint: to add a new column {attr_name}, you could use: tensorframe[{repr(attr_name)}] = ..."
                )
        else:
            self.__dict__[attr_name] = value

    def __getattr__(self, column_name: str) -> torch.Tensor:
        if column_name in self.__data:
            return self.__data[column_name]
        else:
            raise AttributeError(column_name)

    def without_enforced_device(self) -> "TensorFrame":
        """
        Make a shallow copy of this TensorFrame without any enforced device.

        In the newly made shallow copy, columns will be able to exist on
        different devices.

        Returns:
            A shallow copy of this TensorFrame without any enforced device.
        """
        return TensorFrame(self.__data, read_only=self.__is_read_only, device=None)

    def with_enforced_device(self, device: Union[str, torch.device]) -> "TensorFrame":
        """
        Make a shallow copy of this TensorFrame with an enforced device.

        In the newly made shallow copy, columns will be forcefully moved onto
        the specified device.

        Args:
            device: The device to which the new TensorFrame's columns will move.
        Returns:
            A shallow copy of this TensorFrame with an enforced device.
        """
        if device is None:
            raise TypeError("When using the method `with_enforced_device`, the argument `device` cannot be None")

        return TensorFrame(self.__data, read_only=self.__is_read_only, device=device)

    def to(self, device: Union[str, torch.device]) -> "TensorFrame":
        if self.__device is None:
            enforce_device_kwargs = {}
        else:
            enforce_device_kwargs = {"device": device}

        return TensorFrame(
            {k: v.to(device) for k, v in self.__data.items()},
            read_only=self.__is_read_only,
            **enforce_device_kwargs,
        )

    def cpu(self) -> "TensorFrame":
        """
        Get a shallow copy of this TensorFrame with all columns moved to cpu.
        """
        return self.to("cpu")

    def cuda(self) -> "TensorFrame":
        """
        Get a shallow copy of this TensorFrame with all columns moved to cuda.
        """
        return self.to("cuda")

    @property
    def device(self) -> Optional[Union[torch.device, set]]:
        """
        Get the device(s) of this TensorFrame.

        If different columns exist on different devices, a set of devices
        will be returned. If all the columns exist on the same device, then
        that device will be returned.
        """
        devices = set(v.device for v in self.__data.values())
        n = len(devices)
        if n == 0:
            return None
        elif n == 1:
            [only_device] = devices
            return only_device
        else:
            return devices

    @property
    def columns(self) -> list:
        """
        Columns as a list of strings
        """
        return list(self.__data.keys())

    @property
    def pick(self) -> "Picker":
        """
        Get or set values of this TensorFrame.

        **Usage.**

        ```python
        # Get values
        tensorframe.pick[slice_or_indexlist, column_name]

        # Set values
        tensorframe.pick[slice_or_indexlist, column_name] = (
            ...  # A tensor or a list/tensor of indices or a TensorFrame
        )
        ```
        """
        return Picker(self)

    def __non_tabular_repr(self, *, max_depth: int = DEFAULT_MAX_DEPTH_FOR_PRINTING) -> str:
        f = StringIO()
        if max_depth >= DEFAULT_MAX_DEPTH_FOR_PRINTING:
            indent = " " * 4
            dbl_indent = " " * 8
            colon_for_dict = ":"
            comma_for_dict = ","

            def prn(*items):
                print(*items, sep="", file=f)

        else:
            indent = ""
            dbl_indent = ""
            colon_for_dict = ": "
            comma_for_dict = ", "

            def prn(*items):
                print(*items, sep="", end="", file=f)

        try:
            prn(type(self).__name__, "(")
            prn(indent, "{")
            last_i = len(self.__data) - 1
            for i, (k, v) in enumerate(self.__data.items()):
                if i > 0:
                    prn()
                prn(dbl_indent, repr(k), colon_for_dict)
                comma = "" if i == last_i else comma_for_dict
                if isinstance(v, RecursivePrintable):
                    v_as_str = v.to_string(max_depth=(max_depth - 1))
                elif isinstance(v, torch.Tensor):
                    v_as_str = str(v).replace("\n", "\n" + dbl_indent).strip()
                else:
                    v_as_str = str(v)
                prn(dbl_indent, v_as_str, comma)
            prn(indent, "}")

            if self.__is_read_only and (self.__device is not None):
                prn(indent, comma_for_dict, "read_only=True", comma_for_dict)
                prn(indent, "device=", repr(self.__device))
            elif self.__is_read_only:
                prn(indent, comma_for_dict, "read_only=True")
            elif self.__device is not None:
                prn(indent, comma_for_dict, "device=", repr(self.__device))

            prn(")")
            f.seek(0)
            result = f.read()
        finally:
            f.close()

        return result

    def __tabular_repr(self) -> str:
        from itertools import chain

        half_rows_to_show = 15
        num_rows_to_show = half_rows_to_show * 2

        n = len(self)
        if n <= num_rows_to_show:
            row_indices_to_show = range(n)
            three_dots_after_row = None
        else:
            row_indices_to_show = chain(range(half_rows_to_show), range(n - half_rows_to_show, n))
            three_dots_after_row = half_rows_to_show - 1

        col_lengths = {}

        def update_col_length(col_index: int, new_col_length: int):
            if col_index in col_lengths:
                current_col_length = col_lengths[col_index]
                if new_col_length > current_col_length:
                    col_lengths[col_index] = new_col_length
            else:
                col_lengths[col_index] = new_col_length

        lines = []

        def fill_line(*items):
            current_line = []
            for col_index, item in enumerate(items):
                if isinstance(item, torch.Tensor) and (item.ndim == 0):
                    # item_as_str = " " + str(item.cpu().item()) + " "
                    item_as_str = " " + str(np.asarray(item.cpu()).reshape(1)[0]) + " "
                else:
                    item_as_str = " " + str(item) + " "
                update_col_length(col_index, len(item_as_str))
                current_line.append(item_as_str)
            lines.append(current_line)

        fill_line("", *(_without_torch_dot(self[column].dtype) for column in self.columns))
        fill_line("", *((self[column].device) for column in self.columns))
        fill_line()
        fill_line("", *(column for column in self.columns))
        fill_line("", *(("=" * col_lengths[j + 1]) for j in range(len(self.columns))))
        for row_index in row_indices_to_show:
            fill_line(row_index, *(self[column][row_index] for column in self.columns))
            if (three_dots_after_row is not None) and (row_index == three_dots_after_row):
                fill_line("...", *("..." for column in self.columns))

        f = StringIO()
        try:
            for line in lines:
                for col_index, col in enumerate(line):
                    print(col.center(col_lengths[col_index]), end="", file=f)
                print(file=f)
            needs_another_line = False
            if self.__device is not None:
                print(" device=", repr(self.__device), sep="", end="", file=f)
                needs_another_line = True
            if self.__is_read_only:
                print(" read_only=True", end="", file=f)
                needs_another_line = True
            if needs_another_line:
                print(file=f)
            f.seek(0)
            result = f.read()
        finally:
            f.close()
        return result

    def __all_columns_are_one_dimensional(self) -> bool:
        if len(self.columns) == 0:
            return False

        for column in self.columns:
            if self[column].ndim != 1:
                return False

        return True

    def to_string(self, *, max_depth: int = DEFAULT_MAX_DEPTH_FOR_PRINTING) -> str:
        """
        Return the string representation of this TensorFrame
        """
        if len(self.columns) == 0:
            return type(self).__name__ + "()"

        if (max_depth >= DEFAULT_MAX_DEPTH_FOR_PRINTING) and self.__all_columns_are_one_dimensional():
            return self.__tabular_repr()
        else:
            return self.__non_tabular_repr(max_depth=max_depth)

    def __tabular_repr_html(self) -> str:
        from itertools import chain

        half_rows_to_show = 15
        num_rows_to_show = half_rows_to_show * 2

        n = len(self)
        if n <= num_rows_to_show:
            row_indices_to_show = range(n)
            three_dots_after_row = None
        else:
            row_indices_to_show = chain(range(half_rows_to_show), range(n - half_rows_to_show, n))
            three_dots_after_row = half_rows_to_show - 1

        f = StringIO()

        def safe_html(s: Any) -> str:
            s = str(s)

            replacements = (
                ("&", "&amp;"),
                ("<", "&lt;"),
                (">", "&gt;"),
                ("'", "&apos;"),
                ('"', "&quot;"),
            )

            for a, b in replacements:
                s = s.replace(a, b)

            return s

        def prn(*msg):
            print(*msg, sep="", end="", file=f)

        try:
            prn("<table>")

            prn("<tr>")
            prn("<th></th>")
            for column in self.columns:
                t = self[column]
                prn("<th>")
                prn(safe_html(_without_torch_dot(t.dtype)), "<br/>")
                prn(safe_html(t.device), "<br/><br/>")
                prn(safe_html(column))
                prn("</th>")
            prn("</tr>")

            for i in row_indices_to_show:
                prn("<tr>")
                prn("<td><strong>", i, "</strong></td>")
                for column in self.columns:
                    prn("<td>", safe_html(np.asarray(self[column][i].cpu()).reshape(1)[0]), "</td>")
                prn("</tr>")
                if (three_dots_after_row is not None) and (i == three_dots_after_row):
                    prn("<tr>")
                    prn("<td>...</td>")
                    for column in self.columns:
                        prn("<td>...</td>")
                    prn("</tr>")

            prn("<tr>")
            prn("<th></th>")
            for column in self.columns:
                t = self[column]
                prn("<th>")
                prn(safe_html(column))
                prn("</th>")
            prn("</tr>")

            prn("</table>")
            f.seek(0)
            result = f.read()
        finally:
            f.close()
        return result

    def _repr_html_(self) -> str:
        if len(self.columns) == 0:
            return type(self).__name__ + "()"

        if self.__all_columns_are_one_dimensional():
            return self.__tabular_repr_html()
        else:
            return "<pre>" + self.__non_tabular_repr().replace("<", "[").replace(">", "]") + "</pre>"

    @property
    def is_read_only(self) -> bool:
        """
        True if this TensorFrame is read-only; False otherwise.
        """
        return self.__is_read_only

    def get_read_only_view(self) -> "TensorFrame":
        """
        Get a read-only counterpart of this TensorFrame.
        """
        return TensorFrame(self.__data, read_only=True, device=self.__device)

    def clone(self, *, preserve_read_only: bool = False, memo: Optional[dict] = None) -> "TensorFrame":
        """
        Get a clone of this TensorFrame.

        Args:
            preserve_read_only: If True, the newly made clone will be read-only
                only if this TensorFrame is also read-only.
        Returns:
            The clone of this TensorFrame.
        """
        if memo is None:
            memo = {}

        self_id = id(self)
        if self_id in memo:
            return memo[self_id]

        new_read_only = self.__is_read_only if preserve_read_only else False

        newdata = OrderedDict()
        for k in self.columns:
            newdata[k] = self.__data[k].clone()

        return TensorFrame(newdata, read_only=new_read_only, device=self.__device)

    def __copy__(self) -> "TensorFrame":
        return self.clone(preserve_read_only=True)

    def __deepcopy__(self, memo: Optional[dict]) -> "TensorFrame":
        return self.clone(preserve_read_only=True, memo=memo)

    def __getstate__(self) -> dict:
        cloned_frame = self.clone(preserve_read_only=True)
        return cloned_frame.__dict__

    def __setstate__(self, d: dict):
        object.__setattr__(self, "__dict__", {})
        self.__dict__.update(d)

    def __len__(self) -> int:
        for v in self.__data.values():
            return v.shape[0]
        return 0

    def argsort(
        self,
        by: Union[str, np.str_],
        *,
        indices: Optional[Union[str, np.str_]] = None,
        ranks: Optional[Union[str, np.str_]] = None,
        descending: bool = False,
        join: bool = False,
    ) -> Union[torch.Tensor, "TensorFrame"]:
        """
        Return row indices (also optionally ranks) for sorting the TensorFrame.

        For example, let us assume that we have a TensorFrame named `table`.
        We can sort this `table` like this:

        ```python
        indices_for_sorting = table.argsort(by="A")  # sort by the column A
        sorted_table = table.pick[indices_for_sorting]
        ```

        Args:
            by: The name of the column according to which the TensorFrame
                will be sorted.
            indices: If given as a string `s`, the result will be a new
                TensorFrame, and the sorting indices will be listed under
                a column named `s`.
            ranks: If given as a string `z`, the result will be a new
                TensorFrame, and the ranks will be listed under a column
                named `z`.
            descending: If True, the sorting will be in descending order.
            join: Can be used only if column names are given via `indices`
                and/or `ranks`. If `join` is True, then the resulting
                TensorFrame will have the sorted TensorFrame's columns
                as well.
        Returns:
            A tensor of indices, or a TensorFrame.
        """
        by = str(by)
        target_column = self[by]
        indices_for_sorting = torch.argsort(target_column, descending=descending)
        if (indices is None) and (ranks is None):
            if join:
                raise ValueError(
                    "When the argument `join` is given as True,"
                    " the arguments `indices` and/or `ranks` are also required."
                    " However, both `indices` and `ranks` are encountered as None."
                )
            return indices_for_sorting
        result = TensorFrame()
        if indices is not None:
            result[indices] = indices_for_sorting
        if ranks is not None:
            rank_integers = torch.empty_like(indices_for_sorting)
            [n] = rank_integers.shape
            increasing_indices = torch.arange(n, device=rank_integers.device)
            rank_integers[indices_for_sorting] = increasing_indices
            result[ranks] = rank_integers
        if join:
            return self.hstack(result)
        return result

    def sort(self, by: Union[str, np.str_], *, descending: bool = False) -> "TensorFrame":
        """
        Return a sorted copy of this TensorFrame.

        Args:
            by: Name of the column according to which the sorting will be done.
            descending: If True, the sorting will be in descending order.
        Returns:
            The sorted copy of this TensorFrame.
        """
        indices_for_sorting = self.argsort(by, descending=descending)
        return self.pick[indices_for_sorting]

    def hstack(
        self,
        other: "TensorFrame",
        *,
        override: bool = False,
    ) -> "TensorFrame":
        """
        Horizontally join this TensorFrame with another.

        Args:
            other: The other TensorFrame.
            override: If this is given as True and if the other TensorFrame
                has overlapping columns, the other TensorFrame's values will
                override (i.e. will take priority) in the joined result.
        Returns:
            A new TensorFrame formed from joining this TensorFrame with the
            other TensorFrame.
        """
        if not override:
            left_columns = set(self.columns)
            right_columns = set(other.columns)
            common_columns = left_columns.intersection(right_columns)
            if len(common_columns) > 0:
                raise ValueError(
                    "Cannot horizontally stack these two TensorFrame objects,"
                    f" because they have the following shared column(s): {common_columns}."
                )

        if len(other) != len(self):
            raise ValueError(
                "Cannot horizontally stack these two TensorFrame objects,"
                f" because the number of rows of the first TensorFrame is {len(self)}"
                f" while the length of the second TensorFrame is {len(other)}."
            )

        result = TensorFrame(self, device=self.__device)
        for right_column in other.columns:
            result[right_column] = other[right_column]

        return result

    def vstack(self, other: "TensorFrame") -> "TensorFrame":
        """
        Vertically join this TensorFrame with the other TensorFrame.

        Args:
            other: The other TensorFrame which will be at the bottom.
        Returns:
            The joined TensorFrame.
        """
        if set(self.columns) != set(other.columns):
            raise ValueError(
                "Cannot vertically stack these two TensorFrame objects, because their columns do not perfectly match."
            )

        def combine_tensors(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            b = torch.as_tensor(b, device=a.device)
            if a.ndim != b.ndim:
                raise ValueError("Cannot combine two columns that have different numbers of dimensions")
            if a.ndim == 1:
                return torch.cat([a, b])
            elif a.ndim > 1:
                return torch.vstack([a, b])
            else:
                raise RuntimeError("Execution should not have reached this point. This is probably a bug.")

        newdata = OrderedDict()
        for col in self.columns:
            newdata[col] = combine_tensors(self[col], other[col])

        return TensorFrame(newdata, device=self.__device)

    def each(
        self,
        fn: Callable,
        *,
        chunk_size: Optional[int] = None,
        randomness: str = "error",
        join: bool = False,
        override: bool = False,
    ) -> "TensorFrame":
        """
        For each row of this TensorFrame, perform the operations of `fn`.

        `fn` is executed on the rows in a vectorized manner, with the help
        of `torch.vmap`.

        The function `fn` is expected to have this interface:

        ```python
        def fn(row: dict) -> dict:
            # `row` is a dictionary where the keys are column names.
            # This function is expected to return another dictionary.
            ...
        ```

        For example, if we have a TensorFrame with columns A and B, and
        if we want to create a new column C where, for each row, the
        value under C is the sum of A's value and B's value, then the
        function would look like this:

        ```python
        def do_summation_for_each_row(row: dict) -> dict:
            a = row["A"]
            b = row["B"]
            return {"C": a + b}
        ```

        Now, if our current TensorFrame looks like this:

        ```
         A    B
        ===  ===
         1    4
         2    5
         3    6
        ```

        Running `tensorframe.each(do_summation_for_each_row)` will result in
        the following new TensorFrame:

        ```
         C
        ===
         5
         7
         9
        ```

        Args:
            fn: A function which receives a dictionary as its argument, and
                returns another dictionary.
            chunk_size: For performing `fn` on each row, this `each` method
                uses `torch.vmap`. This `chunk_size` argument configures the
                size of the chunks on which the transformed `fn` will
                operate. If `chunk_size` is ignored, `fn` will operate on the
                whole batch.
            randomness: If given as "error" (which is the default), any random
                number generation operation within `fn` will raise an error.
                If given as "different", random numbers generated by `fn`
                will differ from row to row.
                If given as "same", random numbers generated by `fn` will be
                the same across the rows.
            join: If given as True, the resulting TensorFrame will also contain
                this TensorFrame's columns.
            override: If given as True (and if `join` is also True), and if
                the resulting TensorFrame has overlapping columns, the new
                values under the overlapping columns will take precedence.
        Returns:
            A new TensorFrame which stores the results of `fn`
        """
        if (not join) and override:
            raise ValueError("The argument `override` can be set as True only if `join` is also True.")
        input_dict = {k: self.__data[k] for k in self.columns}
        output_dict = vmap(fn, chunk_size=chunk_size, randomness=randomness)(input_dict)
        result = TensorFrame(output_dict, read_only=self.is_read_only, **(self.__get_default_device_kwargs()))
        if join:
            result = self.hstack(result, override=override)
        return result

    def sort_values(
        self,
        by: Union[str, np.str_, Sequence],
        *,
        ascending: Union[bool, Sequence] = True,
    ) -> "TensorFrame":
        """
        Like the `sort` method, but with a more pandas-like interface.

        Args:
            by: Column according to which this TensorFrame will be sorted.
            ascending: If True, the sorting will be in ascending order.
        Returns:
            The sorted counterpart of this TensorFrame.
        """
        by = _get_only_one_column_name(by)
        ascending = _get_only_one_boolean(ascending)
        return self.sort(by, descending=(not ascending))

    def nlargest(self, n: int, columns: Union[str, np.str_, Sequence]) -> "TensorFrame":
        """
        Sort this TensorFrame and take the largest `n` rows.

        Args:
            n: Number of rows of the resulting TensorFrame.
            columns: The name of the column according to which the rows will
                be sorted. Although the name of this argument is plural
                ("columns") for compatibility with pandas' interface, only
                one column name is supported.
        Returns:
            A new TensorFrame that contains the largest n rows of the original
            TensorFrame.
        """
        return self.sort_values(columns, ascending=False).pick[:n]

    def nsmallest(self, n: int, columns: Union[str, np.str_, Sequence]) -> "TensorFrame":
        """
        Sort this TensorFrame and take the smallest `n` rows.

        Args:
            n: Number of rows of the resulting TensorFrame.
            columns: The name of the column according to which the rows will
                be sorted. Although the name of this argument is plural
                ("columns") for compatibility with pandas' interface, only
                one column name is supported.
        Returns:
            A new TensorFrame that contains the smallest n rows of the original
            TensorFrame.
        """
        return self.sort_values(columns, ascending=True).pick[:n]

    def join(self, t: Union["TensorFrame", Sequence]) -> "TensorFrame":
        """
        Like the `hstack` method, but with a more pandas-like interface.

        Joins this TensorFrame with the other TensorFrame.

        Args:
            t: The TensorFrame that will be horizontally stacked to the right.
        Returns:
            A new TensorFrame which is the result of horizontally joining this
            TensorFrame with the other (`t`).
        """
        t = _get_only_one_tensorframe(t)
        return self.hstack(t)

    def drop(self, *, columns: Union[str, np.str_, Sequence]) -> "TensorFrame":
        """
        Get a new TensorFrame where the given columns are dropped.

        Args:
            columns: A single column name or a sequence of column names
                to be dropped.
        Returns:
            A modified copy of this TensorFrame where the specified `columns`
            are dropped.
        """
        if isinstance(columns, (str, np.str_)):
            columns = [columns]
        elif isinstance(columns, Sequence):
            pass  # nothing to do
        else:
            raise TypeError(
                "The argument `columns` was expected as a string or as a sequence of strings."
                f" However, it was received as an instance of this unrecognized type: {type(columns)}."
            )
        all_columns = set(self.__data.keys())
        columns_to_drop = set(str(s) for s in columns)
        if not columns_to_drop.issubset(all_columns):
            raise ValueError(
                "Some of the `columns` cannot be found within the original TensorFrame,"
                " and therefore, they cannot be dropped."
            )
        result = TensorFrame(device=self.__device)
        for col in self.__data.keys():
            if col not in columns_to_drop:
                result[col] = self.__data[col]
        if self.is_read_only:
            result = result.get_read_only_view()
        return result

    def with_columns(self, **kwargs) -> "TensorFrame":
        """
        Get a modified copy of this TensorFrame with some columns added/updated.

        The columns to be updated or added are expected as keyword arguments.
        For example, if a keyword argument is given as `A=new_a_values`, then,
        if `A` already exists in the original TensorFrame, those values will be
        dropped and the resulting TensorFrame will have the new values
        (`new_a_values`). On the other hand, if `A` does not exist in the
        original TensorFrame, the resulting TensorFrame will have a new column
        `A` with the given `new_a_values`.
        """
        columns_to_update = set(kwargs.keys())
        columns_already_updated = set()
        result = TensorFrame(device=self.__device)
        for col in self.__data.keys():
            if col in columns_to_update:
                result[col] = kwargs[col]
                columns_already_updated.add(col)
            else:
                result[col] = self.__data[col]
        for col in columns_to_update.difference(columns_already_updated):
            result[col] = kwargs[col]
        if self.is_read_only:
            result = result.get_read_only_view()
        return result


def _get_only_one_column_name(s: Union[str, np.str_, Sequence]) -> str:
    if isinstance(s, (str, np.str_)):
        return str(s)
    elif isinstance(s, Sequence):
        n = len(s)
        if n == 0:
            raise ValueError("Expected a column name but received an empty sequence")
        if n != 1:
            raise ValueError("Only a single column is supported")
        [result] = s
        return str(result)
    else:
        raise TypeError(f"Don't know how to get the column name from an instance of {type(s)}")


def _get_only_one_boolean(b: Union[bool, np.bool_, Sequence]) -> bool:
    if isinstance(b, Sequence):
        n = len(b)
        if n == 0:
            raise ValueError("Expected a single boolean item but received an empty sequence")
        if n != 1:
            raise ValueError("Only a single boolean is supported")
        [result] = b
        return bool(result)
    else:
        return bool(b)


def _get_only_one_tensorframe(t: Union[TensorFrame, Sequence]) -> TensorFrame:
    if isinstance(t, Sequence):
        n = len(t)
        if n == 0:
            raise ValueError("Expected a single TensorFrame but received an empty sequence")
        if n != 1:
            raise ValueError("Only a single TensorFrame is supported")
        [t] = t
    if not isinstance(t, TensorFrame):
        raise TypeError(f"Expected a TensorFrame, but got an instance of {type(t)}")
    return t


RowIndex = Union[slice, list, torch.Tensor, np.ndarray]


def _as_int_or_none(x: Any) -> Optional[int]:
    return None if x is None else int(x)


def _prepare_indices(values: torch.Tensor, index: RowIndex) -> Union[torch.Tensor, slice]:
    if isinstance(index, slice):
        old_index = index
        index = slice(
            _as_int_or_none(old_index.start),
            _as_int_or_none(old_index.stop),
            _as_int_or_none(old_index.step),
        )
    elif isinstance(index, (list, torch.Tensor, np.ndarray)):
        index = torch.as_tensor(index, device=values.device)
        if index.ndim != 1:
            raise ValueError("This operation can only work with 1-dimensional index tensors.")
    else:
        raise TypeError(
            "The indices were expected as a slice, or as a tensor, or as a numpy array, or as a list."
            f" However, the encountered object is of type {type(index)}."
        )
    return index


def _set_values(values: torch.Tensor, index: RowIndex, new_values: Any) -> torch.Tensor:
    index = _prepare_indices(values, index)
    got_index_tensor = isinstance(index, torch.Tensor) and (index.dtype != torch.bool)
    got_mask_tensor = isinstance(index, torch.Tensor) and (index.dtype == torch.bool)
    got_slice = isinstance(index, slice)

    if not isinstance(values, torch.Tensor):
        raise TypeError(f"Cannot change the item(s) within an instance of {type(values)}")

    if got_index_tensor:
        return torch.index_put(values, (index,), torch.as_tensor(new_values, device=values.device))
    elif got_slice or got_mask_tensor:
        n = len(values)
        target_indices = torch.arange(n, device=values.device)[index]
        return _set_values(values, target_indices, new_values)
    else:
        raise TypeError(
            "The argument `index` was expected as a tensor of integers, or as a tensor of booleans,"
            " or as a slice, or as an integer."
            f" However, its type is {type(index)}."
        )


def _get_values(values: torch.Tensor, index: RowIndex) -> torch.Tensor:
    index = _prepare_indices(values, index)

    if not isinstance(values, torch.Tensor):
        raise TypeError(f"Cannot get item(s) from within an instance of {type(values)}")

    return values[index]


class Picker:
    def __init__(self, frame: TensorFrame):
        self.__frame = frame

    def __unpack_location(self, location: Union[tuple, RowIndex]) -> tuple:
        if isinstance(location, tuple):
            rows, columns = location
            if isinstance(columns, (str, np.str_)):
                columns = [str(columns)]
            elif isinstance(columns, list):
                columns = [str(s) for s in columns]
            elif isinstance(columns, slice):
                if columns.start is None and columns.stop is None and columns.step is None:
                    columns = self.__frame.columns
                else:
                    raise ValueError("For columns, only unlimited slice (i.e. ':') is supported")
            else:
                raise TypeError(
                    "Columns were expected as a string, or as a list of strings, or as an unlimited slice (i.e. ':')"
                    f" However, the encountered object is of this unrecognized type: {type(columns)}."
                )
        else:
            rows = location
            columns = self.__frame.columns
        return rows, columns

    def __getitem__(self, location: Union[tuple, RowIndex]):
        index, columns = self.__unpack_location(location)

        result = TensorFrame()
        for col in columns:
            result[col] = _get_values(self.__frame[col], index)
        return result

    def __setitem__(
        self, location: Union[tuple, RowIndex], new_values: Union[np.ndarray, torch.Tensor, Mapping, TensorFrame]
    ):
        if self.__frame.is_read_only:
            raise TypeError("Cannot modify a read-only TensorFrame")

        index, columns = self.__unpack_location(location)

        if isinstance(new_values, TensorFrame):
            incoming_columns = set(new_values.columns)
        elif isinstance(new_values, Mapping):
            incoming_columns = set(new_values.keys())
        elif isinstance(new_values, (np.ndarray, torch.Tensor, Sequence)):
            if len(columns) != 1:
                raise ValueError(
                    f" When the right-hand side values are given in a {type(new_values)},"
                    " there must be only one target column on the left-hand side of the assignment."
                    f" However, the number of target columns is {len(columns)}."
                )
            incoming_columns = set(columns)
            [only_column] = columns
            new_values = {only_column: new_values}
        else:
            raise TypeError(
                "The right-hand side values were expected in the form of a numpy array, or a PyTorch tensor,"
                " or a sequence of values, or a dictionary (where the keys are column names and values are the new"
                " values for those columns), or a TensorFrame."
                f" However, the encountered right-hand side object has this unrecognized type: {type(new_values)}."
            )

        if set(columns) != incoming_columns:
            raise ValueError("The columns of the left-hand side do not match the columns of the right-hand side")

        for col in columns:
            self.__frame[col] = _set_values(self.__frame[col], index, new_values[col])
