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

from typing import Optional

import torch
from torch import nn


class MultiLayered(nn.Module):
    def __init__(self, *layers: nn.Module):
        super().__init__()
        self._submodules = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, h: Optional[dict] = None):
        if h is None:
            h = {}

        new_h = {}

        for i, layer in enumerate(self._submodules):
            layer_h = h.get(i, None)
            if layer_h is None:
                layer_result = layer(x)
            else:
                layer_result = layer(x, h[i])

            if isinstance(layer_result, tuple):
                if len(layer_result) == 2:
                    x, layer_new_h = layer_result
                else:
                    raise ValueError(
                        f"The layer number {i} returned a tuple of length {len(layer_result)}."
                        f" A tensor or a tuple of two elements was expected."
                    )
            elif isinstance(layer_result, torch.Tensor):
                x = layer_result
                layer_new_h = None
            else:
                raise TypeError(
                    f"The layer number {i} returned an object of type {type(layer_result)}."
                    f" A tensor or a tuple of two elements was expected."
                )

            if layer_new_h is not None:
                new_h[i] = layer_new_h

        if len(new_h) == 0:
            return x
        else:
            return x, new_h

    def __iter__(self):
        return self._submodules.__iter__()

    def __getitem__(self, i):
        return self._submodules[i]

    def __len__(self):
        return len(self._submodules)

    def append(self, module: nn.Module):
        self._submodules.append(module)
