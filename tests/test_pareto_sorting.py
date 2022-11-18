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

# TODO: come up with a bunch of static fitness examples where there is no equal crowding distance between the points
#       (so that the the sorting is deterministic and the test is guaranteed to work)


from typing import List, Tuple

import numpy as np
import torch

from evotorch import Problem, SolutionBatch
from evotorch.core import ParetoInfo


class DummyMultiObjProblem(Problem):
    def __init__(self):
        super().__init__(
            objective_sense=["min", "max"],
            solution_length=4,
            dtype=torch.float32,
            initial_bounds=(-1000, 1000),
        )

    def _evaluate_batch(self, batch: SolutionBatch):
        f1 = batch.values[:, :2].pow(2).sum(dim=-1)
        f2 = batch.values[:, 2:].pow(2).sum(dim=-1)
        evals = batch.access_evals()
        evals[:, 0] = f1
        evals[:, 1] = f2

    def make_dummy_batch(self) -> SolutionBatch:
        """
        Make a SolutionBatch on which the tests will be done.

        The decision values of this newly made SolutionBatch are fixed
        in such a way that, once pareto-sorted, the solutions' crowding
        distances do not coincide with each other (except for the ones that
        have the inf crodwing distances).
        """
        values = torch.FloatTensor(
            [
                [1148, 1955, 137, 2572],
                [2359, 1333, 1044, 2649],
                [1524, 1131, 2092, 1720],
                [930, 1647, 1040, 1691],
                [3255, 245, 3480, 2880],
                [2415, 2371, 203, 787],
                [2812, 692, 919, 2352],
                [729, 133, 3348, 582],
                [1973, 2582, 286, 3097],
                [1229, 1623, 3134, 3215],
                [3204, 1336, 2821, 465],
                [351, 3406, 1655, 134],
                [2263, 1376, 2395, 523],
                [681, 1156, 1196, 1070],
                [3405, 3053, 1322, 2574],
                [479, 456, 3517, 3032],
                [3360, 2285, 1902, 2869],
                [3926, 143, 463, 1750],
                [2891, 3652, 928, 102],
                [798, 1966, 872, 101],
                [3339, 481, 347, 1599],
                [1898, 1620, 1455, 1506],
                [2224, 2656, 3971, 3369],
                [3444, 3163, 260, 959],
                [1444, 805, 2353, 1238],
                [2513, 3131, 1710, 3919],
                [3254, 1216, 3282, 3607],
                [3266, 2883, 2432, 517],
                [657, 3677, 2317, 1313],
                [284, 604, 2151, 3976],
                [1421, 1013, 909, 2202],
                [1791, 1753, 1231, 962],
                [1023, 2999, 3713, 2661],
                [1114, 385, 2969, 1685],
                [2688, 1118, 1277, 1868],
                [1380, 3835, 2748, 3328],
                [3044, 115, 1646, 2611],
                [656, 496, 956, 1051],
                [3964, 2562, 3455, 1696],
                [1706, 2653, 2312, 2351],
                [3603, 803, 3038, 3794],
                [3868, 387, 606, 3505],
                [2838, 2995, 5, 1382],
                [3057, 621, 467, 2639],
                [58, 3132, 1679, 2261],
                [666, 15, 1387, 3115],
                [205, 960, 2413, 3103],
                [2043, 2743, 1334, 529],
                [1560, 1675, 3703, 714],
                [1377, 2549, 3891, 362],
            ]
        )

        num_solutions = values.shape[0]
        batch = self.generate_batch(num_solutions, empty=True)
        batch.access_values()[:] = values
        return batch


class NonVectorizedParetoTools:
    """
    Class that contains non-vectorized pareto-sorting functions.

    The idea is to compare the results of these non-vectorized functions
    and the results of the newly introduced vectorized pareto-sorting
    methods of SolutionBatch.
    """

    vectorized_crowding = False

    @staticmethod
    def dominates(i: int, j: int, utils: np.ndarray) -> bool:
        return np.all(utils[i, :] >= utils[j, :]) and np.any(utils[i, :] > utils[j, :])

    @staticmethod
    def crowding_distance_assignment(pareto_set: np.ndarray, utils: np.ndarray) -> np.ndarray:
        L = len(pareto_set)
        distances = np.zeros(L, dtype="float32")
        for m in range(utils.shape[1]):
            # U = utils[pareto_set][:, m]
            U = utils[pareto_set, m]
            ordered = np.argsort(U)[::-1]

            # e.g. pareto_set = [1,   7,  3]
            # e.g.          U = [20, 14, 15]

            # ordered = [0, 2, 1]

            distances[ordered[0]] = np.inf
            distances[ordered[-1]] = np.inf

            fmax = np.max(U)
            fmin = np.min(U)

            for i in range(1, L - 1):
                denom = fmax - fmin
                if denom < 1e-8:
                    denom = 1e-8
                distances[ordered[i]] += (U[ordered[i - 1]] - U[ordered[i + 1]]) / denom

        return distances

    @classmethod
    def pareto_sort_np(
        cls,
        utils: np.ndarray,
        crowdsort: bool,
        crowdsort_upto: int,
    ) -> Tuple[List[np.ndarray], np.ndarray]:

        if cls.vectorized_crowding:

            def crowding_distance_assignment(*args, **kwargs):
                return cls.crowding_distance_assignment2(*args, **kwargs).numpy()

        else:
            crowding_distance_assignment = cls.crowding_distance_assignment

        count: int = 0

        n = int(len(utils))
        dominated_by: List[List[int]] = [[0 for __ in range(0)] for _ in range(n)]
        domination_counter: List[int] = [0 for _ in range(n)]
        rank = np.zeros(n, dtype="int64")
        fronts: List[np.ndarray] = [np.array([0], dtype="int64") for _ in range(0)]

        first_front: List[int] = []
        for p in range(n):
            for q in range(n):
                if cls.dominates(p, q, utils):
                    dominated_by[p].append(q)
                elif cls.dominates(q, p, utils):
                    domination_counter[p] += 1
            if domination_counter[p] == 0:
                rank[p] = 0
                # fronts[0].append(p)
                first_front.append(p)

        first_front_array = np.array(first_front, "int64")
        if not crowdsort:
            fronts.append(first_front_array)
        else:
            fronts.append(first_front_array[crowding_distance_assignment(first_front_array, utils).argsort()[::-1]])
        count += len(fronts[-1])

        i = 0
        while True:
            next_front: List[int] = []
            for p in fronts[-1]:
                for q in dominated_by[p]:
                    domination_counter[q] -= 1
                    if domination_counter[q] == 0:
                        rank[q] = i + 1
                        next_front.append(q)
            i += 1

            if len(next_front) == 0:
                break
            else:
                next_front_array = np.array(next_front, "int64")
                if (not crowdsort) or (count > crowdsort_upto):
                    fronts.append(next_front_array)
                else:
                    fronts.append(
                        next_front_array[crowding_distance_assignment(next_front_array, utils).argsort()[::-1]]
                    )
                count += len(fronts[-1])

        return fronts, rank

    @classmethod
    def pareto_sort(
        cls, utils: torch.Tensor, crowdsort: bool, crowdsort_upto: int
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        device = utils.device
        utils = torch.as_tensor(utils, device="cpu").numpy()
        fronts, ranks = cls.pareto_sort_np(utils, crowdsort, crowdsort_upto)

        for i in range(len(fronts)):
            fronts[i] = torch.as_tensor(torch.from_numpy(fronts[i]), device=device)

        ranks = torch.as_tensor(torch.from_numpy(ranks), device=device)

        return fronts, ranks

    @classmethod
    def arg_pareto_sort(cls, batch: SolutionBatch, crowdsort: bool = True) -> ParetoInfo:
        utils = batch.utils()
        fronts, ranks = cls.pareto_sort(utils, crowdsort, len(batch))
        return ParetoInfo(fronts=fronts, ranks=ranks)


def test_pareto_sorting():
    # Make a new instance of the simple multi-objective problem.
    problem = DummyMultiObjProblem()

    # Instantiate and evaluate the test solutions.
    batch = problem.make_dummy_batch()
    problem.evaluate(batch)

    # Compute pareto info using both non-vectorized and vectorized tools
    pareto_info_a = batch.arg_pareto_sort()
    pareto_info_b = NonVectorizedParetoTools.arg_pareto_sort(batch)

    # Ensure that the ranks of the solutions match
    assert torch.all(pareto_info_a.ranks == pareto_info_b.ranks)

    # Ensure that the number of fronts match
    assert len(pareto_info_a.fronts) == len(pareto_info_b.fronts)

    num_fronts = len(pareto_info_a.fronts)

    for i_front in range(num_fronts):
        # For each front, ensure that the crowding-distance-based ordering match
        front_a = pareto_info_a.fronts[i_front]
        front_b = pareto_info_b.fronts[i_front]
        assert len(front_a) == len(front_b)
        if len(front_a) > 2:
            crowd_sorted_a = front_a[2:]
            crowd_sorted_b = front_b[2:]
            assert torch.all(crowd_sorted_a == crowd_sorted_b)
