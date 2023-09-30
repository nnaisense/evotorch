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

import math
from collections.abc import Mapping
from copy import copy, deepcopy
from typing import Optional, Type, Union

import torch

from ...core import Problem, SolutionBatch
from ...distributions import (
    Distribution,
    ExpGaussian,
    ExpSeparableGaussian,
    SeparableGaussian,
    SymmetricSeparableGaussian,
)
from ...optimizers import get_optimizer_class
from ...tools import RealOrVector, modify_tensor, to_stdev_init
from ..searchalgorithm import SearchAlgorithm, SinglePopulationAlgorithmMixin


class GaussianSearchAlgorithm(SearchAlgorithm, SinglePopulationAlgorithmMixin):
    """
    Base class for search algorithms based on Gaussian distribution.
    """

    DISTRIBUTION_TYPE = NotImplemented
    DISTRIBUTION_PARAMS = NotImplemented

    def __init__(
        self,
        problem: Problem,
        *,
        popsize: int,
        center_learning_rate: float,
        stdev_learning_rate: float,
        stdev_init: Optional[RealOrVector] = None,
        radius_init: Optional[RealOrVector] = None,
        num_interactions: Optional[int] = None,
        popsize_max: Optional[int] = None,
        optimizer=None,
        optimizer_config: Optional[dict] = None,
        ranking_method: Optional[str] = None,
        center_init: Optional[RealOrVector] = None,
        stdev_min: Optional[RealOrVector] = None,
        stdev_max: Optional[RealOrVector] = None,
        stdev_max_change: Optional[RealOrVector] = None,
        obj_index: Optional[int] = None,
        distributed: bool = False,
        popsize_weighted_grad_avg: Optional[bool] = None,
        ensure_even_popsize: bool = False,
    ):
        # Ensure that the problem is numeric
        problem.ensure_numeric()

        # The distribution-based algorithms we consider here cannot handle strict lower and upper bound constraints.
        # Therefore, we ensure that the given problem is unbounded.
        problem.ensure_unbounded()

        # Initialize the SearchAlgorithm, which is the parent class
        SearchAlgorithm.__init__(
            self,
            problem,
            center=self._get_mu,
            stdev=self._get_sigma,
            mean_eval=self._get_mean_eval,
        )

        self._ensure_even_popsize = bool(ensure_even_popsize)

        if not distributed:
            # self.add_status_getters({"median_eval": self._get_median_eval})
            if num_interactions is not None:
                self.add_status_getters({"popsize": self._get_popsize})
            if self._ensure_even_popsize:
                if (popsize % 2) != 0:
                    raise ValueError(
                        f"`popsize` was expected as an even number. However, the received `popsize` is {popsize}."
                    )

        if center_init is None:
            # If a starting point for the search distribution is not given,
            # then we use the problem object to generate us one.
            mu = problem.generate_values(1).reshape(-1)
        else:
            # If a starting point for the search distribution is given,
            # then we make sure that its length, dtype, and device
            # are correct.
            mu = problem.ensure_tensor_length_and_dtype(center_init, allow_scalar=False, about="center_init")

        # Get the standard deviation or the radius configuration from the arguments
        stdev_init = to_stdev_init(
            solution_length=problem.solution_length, stdev_init=stdev_init, radius_init=radius_init
        )

        # Make sure that the provided initial standard deviation is
        # of correct length, dtype, and device.
        sigma = problem.ensure_tensor_length_and_dtype(stdev_init, about="stdev_init", allow_scalar=False)

        # Create the distribution
        dist_cls = self.DISTRIBUTION_TYPE
        dist_params = deepcopy(self.DISTRIBUTION_PARAMS) if self.DISTRIBUTION_PARAMS is not None else {}
        dist_params.update({"mu": mu, "sigma": sigma})
        self._distribution: Distribution = dist_cls(dist_params, dtype=problem.dtype, device=problem.device)

        # Store the following keyword arguments to use later
        self._popsize = int(popsize)
        self._popsize_max = None if popsize_max is None else int(popsize_max)
        self._num_interactions = None if num_interactions is None else int(num_interactions)

        self._center_learning_rate = float(center_learning_rate)
        self._stdev_learning_rate = float(stdev_learning_rate)
        self._optimizer = self._initialize_optimizer(self._center_learning_rate, optimizer, optimizer_config)
        self._ranking_method = None if ranking_method is None else str(ranking_method)

        self._stdev_min = (
            None
            if stdev_min is None
            else problem.ensure_tensor_length_and_dtype(stdev_min, about="stdev_min", allow_scalar=True)
        )

        self._stdev_max = (
            None
            if stdev_max is None
            else problem.ensure_tensor_length_and_dtype(stdev_max, about="stdev_max", allow_scalar=True)
        )

        self._stdev_max_change = (
            None
            if stdev_max_change is None
            else problem.ensure_tensor_length_and_dtype(stdev_max_change, about="stdev_max_change", allow_scalar=True)
        )

        self._obj_index = problem.normalize_obj_index(obj_index)

        if distributed and (problem.num_actors > 0):
            # If the algorithm is initialized in distributed mode, and also if the problem is configured
            # for parallelization, then the _step method becomes an alias for _step_distributed
            self._step = self._step_distributed
        else:
            # Otherwise, the _step method becomes an alias for _step_non_distributed
            self._step = self._step_non_distributed

        if popsize_weighted_grad_avg is None:
            self._popsize_weighted_grad_avg = num_interactions is None
        else:
            if not distributed:
                raise ValueError(
                    "The argument `popsize_weighted_grad_avg` can only be used in distributed mode."
                    " (i.e. when the argument `distributed` is given as True)."
                    " When `distributed` is False, please leave `popsize_weighted_grad_avg` as None."
                )
            self._popsize_weighted_grad_avg = bool(popsize_weighted_grad_avg)

        self._mean_eval: Optional[float] = None
        self._population: Optional[SolutionBatch] = None
        self._first_iter: bool = True

        # We would like to add the reporting capabilities of the mixin class `singlePopulationAlgorithmMixin`.
        # However, we exclude "mean_eval" from the reporting services requested from `SinglePopulationAlgorithmMixin`
        # because this class has its own reporting mechanism for `mean_eval`.
        # Additionally, we enable the reporting services of `SinglePopulationAlgorithmMixin` only when we are
        # in the non-distributed mode. This is because we do not have a centrally stored population at all in the
        # distributed mode.
        SinglePopulationAlgorithmMixin.__init__(self, exclude="mean_eval", enable=(not distributed))

    def _initialize_optimizer(
        self, learning_rate: float, optimizer=None, optimizer_config: Optional[dict] = None
    ) -> object:
        if optimizer is None:
            return None
        elif isinstance(optimizer, str):
            center_optim_cls = get_optimizer_class(optimizer, optimizer_config)
            return center_optim_cls(
                stepsize=float(learning_rate),
                dtype=self._distribution.dtype,
                solution_length=self._distribution.solution_length,
                device=self._distribution.device,
            )
        else:
            return optimizer

    def _step(self):
        raise NotImplementedError

    def _step_distributed(self):
        # Use the problem object's `sample_and_compute_gradients` method
        # to do parallelized and distributed gradient computation
        fetched = self.problem.sample_and_compute_gradients(
            self._distribution,
            self._popsize,
            popsize_max=self._popsize_max,
            obj_index=self._obj_index,
            num_interactions=self._num_interactions,
            ranking_method=self._ranking_method,
            ensure_even_popsize=self._ensure_even_popsize,
        )

        # The method `sample_and_compute_gradients(...)` returns a list of dictionaries, each dictionary being
        # the result of a different remote computation.
        # For each remote computation, the list will contain a dictionary that looks like this:
        # {"gradients": <gradients dictionary here>, "num_solutions": ..., "mean_eval": ...}

        # We will now accumulate all the gradients, num_solutions, and mean_evals in their own lists.
        # So, in the end, we will have a list of gradients, a list of num_solutions, and a list of
        # mean_eval.
        # These lists will be stored by the following temporary class:
        class list_of:
            gradients = []
            num_solutions = []
            mean_eval = []

        # We are now filling the lists declared above
        n = len(fetched)
        for i in range(n):
            list_of.gradients.append(fetched[i]["gradients"])
            list_of.num_solutions.append(fetched[i]["num_solutions"])
            list_of.mean_eval.append(fetched[i]["mean_eval"])

        # Here, we get the keys of our gradient dictionaries.
        # For most simple Gaussian distributions, grad_keys should be {"mu", "sigma"}.
        grad_keys = set(list_of.gradients[0].keys())

        # We now find the total number of solutions and the overall average mean_eval.
        # The overall average mean will be reported to the user.
        total_num_solutions = 0
        total_weighted_eval = 0
        for i in range(n):
            total_num_solutions += list_of.num_solutions[i]
            total_weighted_eval += float(list_of.num_solutions[i] * list_of.mean_eval[i])
        avg_mean_eval = total_weighted_eval / total_num_solutions

        # For each gradient (in most cases among 'mu' and 'sigma'), we allocate a new 0-filled tensor.
        avg_gradients = {}
        for key in grad_keys:
            avg_gradients[key] = self._distribution.make_zeros(num_solutions=1).reshape(-1)

        # Below, we iterate over all collected results and add their gradients, in a weighted manner, onto the
        # `avg_gradients` we allocated above.
        # At the end, `avg_gradients` will store the weighted-averaged gradients to be followed by the algorithm.
        for i in range(n):
            # For each collected result, we compute a weight for the gradient, which is the number of solutions
            # sampled divided by the total number of sampled solutions.
            num_solutions = list_of.num_solutions[i]
            if self._popsize_weighted_grad_avg:
                # If we are to weigh each gradient by its popsize (i.e. by its sample size)
                # then the its weight is computed as its number of solutions divided by the
                # total number of solutions
                weight = num_solutions / total_num_solutions
            else:
                # If we are NOT to weigh each gradient by its popsize (i.e. by its sample size)
                # then the weight of this gradient simply becomes 1 divided by the number of gradients.
                weight = 1 / n
            for key in grad_keys:
                grad = list_of.gradients[i][key]
                avg_gradients[key] += weight * grad

        self._update_distribution(avg_gradients)
        self._mean_eval = avg_mean_eval

    def _step_non_distributed(self):
        # First, we define an inner function which fills the current population by sampling from the distribution.
        def fill_and_eval_pop():
            # This inner function is responsible for filling the main population with samples
            # and evaluate them.
            if self._num_interactions is None:
                # If num_interactions is configured as None, this means that we are not going to adapt
                # the population size according to the number of simulation interactions reported
                # by the problem object.

                # We first make sure that the population (which is to be of constant size, since we are
                # not in the adaptive population size mode) is allocated.
                if self._population is None:
                    self._population = SolutionBatch(
                        self.problem, popsize=self._popsize, device=self._distribution.device, empty=True
                    )

                # Now, we do in-place sampling on the population.
                self._distribution.sample(out=self._population.access_values(), generator=self.problem)

                # Finally, here, the solutions are evaluated.
                self.problem.evaluate(self._population)
            else:
                # If num_interactions is not None, then this means that we have a threshold for the number
                # of simulator interactions to reach before declaring the phase of sampling complete.
                # In other words, we have to adapt our population size according to the number of simulator
                # interactions reported by the problem object.

                # The 'total_interaction_count' status reported by the problem object shows the global interaction count.
                # Therefore, to properly count the simulator interactions we made during this generation, we need
                # to get the interaction count before starting our sampling and evaluation operations.
                first_num_interactions = self.problem.status.get("total_interaction_count", 0)

                # We will keep allocating and evaluating new populations until the interaction count threshold is reached.
                # These newly allocated populations will eventually concatenated into one.
                # The not-yet-concatenated populations and the total allocated population size will be stored below:
                populations = []
                total_popsize = 0

                # Below, we repeatedly allocate, sample, and evaluate, until our thresholds are reached.
                while True:
                    # Allocate a new population
                    newpop = SolutionBatch(
                        self.problem,
                        popsize=self._popsize,
                        like=self._population,
                        empty=True,
                    )

                    # Update the total population size
                    total_popsize += len(newpop)

                    # Sample new solutions within the newly allocated population
                    self._distribution.sample(out=newpop.access_values(), generator=self.problem)

                    # Evaluate the new population
                    self.problem.evaluate(newpop)

                    # Add the newly allocated and evaluated population into the populations list
                    populations.append(newpop)

                    # In addition to the num_interactions threshold, we might also have a popsize_max threshold.
                    # We now check this threshold.
                    if (self._popsize_max is not None) and (total_popsize >= self._popsize_max):
                        # If the popsize_max threshold is reached, we leave the loop.
                        break

                    # We now compute the number of interactions we have made during this while loop.
                    interactions_made = self.problem.status["total_interaction_count"] - first_num_interactions

                    if interactions_made > self._num_interactions:
                        # If the number of interactions exceeds our threshold, we leave the loop.
                        break

                # Finally, we concatenate all our populations into one.
                self._population = SolutionBatch.cat(populations)

        if self._first_iter:
            # If we are computing the first generation, we just sample from our distribution and evaluate
            # the solutions.
            fill_and_eval_pop()
            self._first_iter = False
        else:
            # If we are computing next generations, then we need to compute the gradients of the last
            # generation, sample a new population, and evaluate the new population's solutions.
            samples = self._population.access_values(keep_evals=True)
            fitnesses = self._population.access_evals()[:, self._obj_index]
            obj_sense = self.problem.senses[self._obj_index]
            ranking_method = self._ranking_method
            gradients = self._distribution.compute_gradients(
                samples, fitnesses, objective_sense=obj_sense, ranking_method=ranking_method
            )
            self._update_distribution(gradients)
            fill_and_eval_pop()

    def _update_distribution(self, gradients: dict):
        # This is where we follow the gradients with the help of the stored Distribution object.

        # First, we check whether or not we will need to do a controlled update on the
        # standard deviation (do we have imposed lower and upper bounds for the standard deviation,
        # and do we have a maximum change limiter?)
        controlled_stdev_update = (
            (self._stdev_min is not None) or (self._stdev_max is not None) or (self._stdev_max_change is not None)
        )

        if controlled_stdev_update:
            # If the standard deviation update needs to be controlled, we store the standard deviation just before
            # the update. We will use this later.
            old_sigma = self._distribution.sigma

        # Here, we determine for which distribution parameter we have a learning rate and for which distribution
        # parameter we have an optimizer.
        learning_rates = {}
        optimizers = {}

        if self._optimizer is not None:
            # If there is an optimizer, then we declare that "mu" has an optimizer
            optimizers["mu"] = self._optimizer
        else:
            # If we do not have an optimizer, then we declare that "mu" has a raw learning rate coefficient
            learning_rates["mu"] = self._center_learning_rate

        # Here, we declare that "sigma" has a learning rate
        learning_rates["sigma"] = self._stdev_learning_rate

        # With the help of the Distribution object's `update_parameters(...)` method, we follow the gradients
        updated_dist = self._distribution.update_parameters(
            gradients, learning_rates=learning_rates, optimizers=optimizers
        )

        if controlled_stdev_update:
            # If our standard deviation update needs to be controlled, then, considering the pre-update
            # standard deviation, we ensure that the update constraints (lower and upper bounds and maximum change)
            # are not violated.
            updated_dist = updated_dist.modified_copy(
                sigma=modify_tensor(
                    old_sigma,
                    updated_dist.sigma,
                    lb=self._stdev_min,
                    ub=self._stdev_max,
                    max_change=self._stdev_max_change,
                )
            )

        # Now we can declare that our main distribution is the updated one
        self._distribution = updated_dist

    def _get_mu(self) -> torch.Tensor:
        return self._distribution.parameters["mu"]

    def _get_sigma(self) -> torch.Tensor:
        return self._distribution.parameters["sigma"]

    def _get_mean_eval(self) -> Optional[float]:
        if self._population is None:
            return self._mean_eval
        else:
            return float(torch.mean(self._population.evals[:, self._obj_index]))

    # def _get_median_eval(self) -> Optional[float]:
    #    if self._population is None:
    #        return None
    #    else:
    #        return float(torch.median(self._population.evals[:, self._obj_index]))

    def _get_popsize(self) -> int:
        return 0 if self._population is None else len(self._population)

    @property
    def optimizer(self):
        """
        Get the optimizer used by this search algorithm.

        If an optimizer is not being used, the result will be `None`.
        If a PyTorch optimizer is being used, the result will be an instance of
        `torch.optim.Optimizer`.
        If the returned optimizer is "clipup", then the returned object will be
        an instance of `evotorch.optimizers.ClipUp`.

        The returned optimizer object can be used for reading/writing the
        hyperparameters. For example, to read the learning of the optimizer,
        one can do:

        ```python
        learning_rate = my_search_algorithm.optimizer.param_groups[0]["lr"]
        ```

        One can also update the learning rate like this:

        ```python
        my_search_algorithm.optimizer.param_groups[0]["lr"] = new_learning_rate
        ```

        **Note for when updating the learning rate of ClipUp.**
        At the moment of initialization, if one provides `center_learning_rate`
        but the maximum speed is not specified (i.e. the search algorithm is not
        given something like `optimizer_config={"max_speed": ...}`), then, the
        maximum speed is initialized as `2*center_learning_rate`. However, when
        this `center_learning_rate` is later modified (via
        `my_search_algorithm.optimizer.param_groups[0]["lr"] = new_center_learning_rate`
        the maximum speed will NOT be automatically adjusted.
        Therefore, when updating the center learning rate of ClipUp, consider
        also adjusting the maximum speed of ClipUp via:
        `my_search_algorithm.optimizer.param_groups[0]["max_speed"] = ...`
        """
        return None if self._optimizer is None else self._optimizer.contained_optimizer

    @property
    def population(self) -> Optional[SolutionBatch]:
        """
        The population, represented by a SolutionBatch.

        If the population is not initialized yet, the retrieved value will
        be None.
        Also note that, if this algorithm is in distributed mode, the
        retrieved value will be None, since the distributed mode causes the
        population to be generated in the remote actors, and not in the main
        process.
        """
        return self._population

    @property
    def obj_index(self) -> int:
        """
        Index of the focused objective
        """
        return self._obj_index


class PGPE(GaussianSearchAlgorithm):
    """
    PGPE: Policy gradient with parameter-based exploration.

    This implementation is the symmetric-sampling variant proposed
    in the paper Sehnke et al. (2010).

    Inspired by the PGPE implementations used in the studies
    of Ha (2017, 2019), and by the evolution strategy variant of
    Salimans et al. (2017), this PGPE implementation uses 0-centered
    ranking by default.
    The default optimizer for this PGPE implementation is ClipUp
    (Toklu et al., 2020).

    References:

        Frank Sehnke, Christian Osendorfer, Thomas Ruckstiess,
        Alex Graves, Jan Peters, Jurgen Schmidhuber (2010).
        Parameter-exploring Policy Gradients.
        Neural Networks 23(4), 551-559.

        David Ha (2017). Evolving Stable Strategies.
        <http://blog.otoro.net/2017/11/12/evolving-stable-strategies/>

        Salimans, T., Ho, J., Chen, X., Sidor, S. and Sutskever, I. (2017).
        Evolution Strategies as a Scalable Alternative to
        Reinforcement Learning.

        David Ha (2019). Reinforcement Learning for Improving Agent Design.
        Artificial life 25 (4), 352-365.

        Toklu, N.E., Liskowski, P., Srivastava, R.K. (2020).
        ClipUp: A Simple and Powerful Optimizer
        for Distribution-based Policy Evolution.
        Parallel Problem Solving from Nature (PPSN 2020).
    """

    DISTRIBUTION_TYPE = NotImplemented  # To be filled by the PGPE instance
    DISTRIBUTION_PARAMS = NotImplemented  # To be filled by the PGPE instance

    def __init__(
        self,
        problem: Problem,
        *,
        popsize: int,
        center_learning_rate: float,
        stdev_learning_rate: float,
        stdev_init: Optional[RealOrVector] = None,
        radius_init: Optional[RealOrVector] = None,
        num_interactions: Optional[int] = None,
        popsize_max: Optional[int] = None,
        optimizer="clipup",
        optimizer_config: Optional[dict] = None,
        ranking_method: Optional[str] = "centered",
        center_init: Optional[RealOrVector] = None,
        stdev_min: Optional[RealOrVector] = None,
        stdev_max: Optional[RealOrVector] = None,
        stdev_max_change: Optional[RealOrVector] = 0.2,
        symmetric: bool = True,
        obj_index: Optional[int] = None,
        distributed: bool = False,
        popsize_weighted_grad_avg: Optional[bool] = None,
    ):
        """
        `__init__(...)`: Initialize the PGPE algorithm.

        Args:
            problem: The problem object which is being worked on.
                The problem must have its dtype defined
                (which means it works on Solution objects,
                not with custom Solution objects).
                Also, the problem must be single-objective.
            popsize: The population size.
                In the case of PGPE, `popsize` is expected as an even number
                in non-distributed mode. In distributed mode, PGPE will
                ensure that each sub-population size assigned to a remote
                actor is an even number.
                This behavior is because PGPE does symmetric sampling
                (i.e. solutions are sampled in pairs).
            center_learning_rate: The learning rate for the center
                of the search distribution.
            stdev_learning_rate: The learning rate for the standard
                deviation values of the search distribution.
            stdev_init: The initial standard deviation of the search
                distribution, expressed as a scalar or as an array.
                Determines the initial coverage area of the search
                distribution.
                If one wishes to configure the coverage area via the
                argument `radius_init` instead, then `stdev_init` is expected
                as None.
            radius_init: The initial radius of the search distribution,
                expressed as a scalar.
                Determines the initial coverage area of the search
                distribution.
                Here, "radius" is defined as the norm of the search
                distribution.
                If one wishes to configure the coverage area via the
                argument `stdev_init` instead, then `radius_init` is expected
                as None.
            num_interactions: When given as an integer n,
                it is ensured that a population has interacted with
                the GymProblem's environment n times. If this target
                has not been reached yet, then the population is declared
                too small, and gets extended with more samples,
                until n amount of interactions is reached.
                When given as None, popsize is the only configuration
                affecting the size of a population.
            popsize_max: Having `num_interactions` set as an integer
                might cause the effective population size jump to
                unnecesarily large numbers. To prevent this,
                one can set `popsize_max` to specify an upper
                bound for the effective population size.
            optimizer: The optimizer to be used while following the
                estimated the gradients.
                Can be given as None if a momentum-based optimizer
                is not required.
                Otherwise, can be given as a str containing the name
                of the optimizer (e.g. 'adam', 'clipup');
                or as an instance of evotorch.optimizers.TorchOptimizer
                or evotorch.optimizers.ClipUp.
                The default is 'clipup'.
                Note that, for ClipUp, the default maximum speed is set
                as twice the given `center_learning_rate`.
                This maximum speed can be configured by passing
                `{"max_speed": ...}` to `optimizer_config`.
            optimizer_config: Configuration which will be passed
                to the optimizer as keyword arguments.
                See `evotorch.optimizers` for details about
                which optimizer accepts which keyword arguments.
            ranking_method: Which ranking method will be used for
                fitness shaping. See the documentation of
                `evotorch.ranking.rank(...)` for details.
                As in the study of Salimans et al. (2017),
                the default is 'centered'.
                Can be given as None if no such ranking is required.
            center_init: The initial center solution.
                Can be left as None.
            stdev_min: Lower bound for the standard deviation value/array.
                Can be given as a real number, or as an array of real numbers.
            stdev_max: Upper bound for the standard deviation value/array.
                Can be given as a real number, or as an array of real numbers.
            stdev_max_change: The maximum update ratio allowed on the
                standard deviation. Expected as None if no such limiter
                is needed, or as a real number within 0.0 and 1.0 otherwise.
                Like in the implementation of Ha (2017, 2018),
                the default value for this setting is 0.2, meaning that
                the update on the standard deviation values can not be
                more than 20% of their original values.
            symmetric: Whether or not the solutions will be sampled
                in a symmetric/mirrored/antithetic manner.
                The default is True.
            obj_index: Index of the objective according to which the
                gradient estimations will be done.
                For single-objective problems, this can be left as None.
            distributed: Whether or not the gradient computation will
                be distributed. If `distributed` is given as False and
                the problem is not parallelized, then everything will
                be centralized (i.e. the entire computation will happen
                in the main process).
                If `distributed` is given as False, and the problem
                is parallelized, then the population will be created
                in the main process and then sent to remote workers
                for parallelized evaluation, and then the remote fitnesses
                will be collected by the main process again for computing
                the search gradients.
                If `distributed` is given as True, and the problem
                is parallelized, then the search algorithm itself will
                be distributed, in the sense that each remote actor will
                generate its own population (such that the total population
                size across all these actors becomes equal to `popsize`)
                and will compute its own gradient, and then the main process
                will collect these gradients, compute the averaged gradients
                and update the main search distribution.
                Non-distributed mode has the advantage of keeping the
                population in the main process, which is good when one wishes
                to do detailed monitoring during the evolutionary process,
                but has the disadvantage of having to pass the solutions to
                the remote actors and having to collect fitnesses, which
                might result in increased interprocess communication traffic.
                On the other hand, while it is not possible to monitor the
                population in distributed mode, the distributed mode has the
                advantage of significantly reducing the interprocess
                communication traffic, since the only things communicated
                with the remote actors are the search distributions (not the
                solutions) and the gradients.
            popsize_weighted_grad_avg: Only to be used in distributed mode.
                (where being in distributed mode means `distributed` is given
                as True). In distributed mode, each actor remotely samples
                its own solution batches and computes its own gradients.
                These gradients are then collected, and a final average
                gradient is computed.
                If `popsize_weighted_grad_avg` is True, then, while averaging
                over the gradients, each gradient will have its own weight
                that is computed according to how many solutions were sampled
                by the actor that produced the gradient.
                If `popsize_weighted_grad_avg` is False, then, there will not
                be weighted averaging (or, each gradient will have equal
                weight).
                If `popsize_weighted_grad_avg` is None, then, the gradient
                weights will be equal a value for `num_interactions` is given
                (because `num_interactions` affects the number of solutions
                according to the episode lengths, and popsize-weighting the
                gradients could be misleading); and the gradient weights will
                be weighted according to the sub-population (i.e. sub-batch)
                sizes if `num_interactions` is left as None.
                The default value for `popsize_weighted_grad_avg` is None.
                When the distributed mode is disabled (i.e. when `distributed`
                is False), then the argument `popsize_weighted_grad_avg` is
                expected as None.
        """

        if symmetric:
            self.DISTRIBUTION_TYPE = SymmetricSeparableGaussian
            divide_by = "num_directions"
        else:
            self.DISTRIBUTION_TYPE = SeparableGaussian
            divide_by = "num_solutions"

        self.DISTRIBUTION_PARAMS = {"divide_mu_grad_by": divide_by, "divide_sigma_grad_by": divide_by}

        super().__init__(
            problem,
            popsize=popsize,
            center_learning_rate=center_learning_rate,
            stdev_learning_rate=stdev_learning_rate,
            stdev_init=stdev_init,
            radius_init=radius_init,
            popsize_max=popsize_max,
            num_interactions=num_interactions,
            optimizer=optimizer,
            optimizer_config=optimizer_config,
            ranking_method=ranking_method,
            center_init=center_init,
            stdev_min=stdev_min,
            stdev_max=stdev_max,
            stdev_max_change=stdev_max_change,
            obj_index=obj_index,
            distributed=distributed,
            popsize_weighted_grad_avg=popsize_weighted_grad_avg,
            ensure_even_popsize=symmetric,
        )


class SNES(GaussianSearchAlgorithm):
    """
    SNES: Separable Natural Evolution Strategies

    Inspired by the implementation at: http://schaul.site44.com/code/snes.py

    Reference:

        Schaul, T., Glasmachers, T., Schmidhuber, J. (2011).
        High Dimensions and Heavy Tails for Natural Evolution Strategies.
        Proceedings of the 13th annual conference on Genetic and evolutionary
        computation (GECCO 2011).
    """

    DISTRIBUTION_TYPE = ExpSeparableGaussian
    DISTRIBUTION_PARAMS = None

    def __init__(
        self,
        problem: Problem,
        *,
        stdev_init: Optional[RealOrVector] = None,
        radius_init: Optional[RealOrVector] = None,
        popsize: Optional[int] = None,
        center_learning_rate: Optional[float] = None,
        stdev_learning_rate: Optional[float] = None,
        scale_learning_rate: bool = True,
        num_interactions: Optional[int] = None,
        popsize_max: Optional[int] = None,
        optimizer=None,
        optimizer_config: Optional[dict] = None,
        ranking_method: Optional[str] = "nes",
        center_init: Optional[RealOrVector] = None,
        stdev_min: Optional[RealOrVector] = None,
        stdev_max: Optional[RealOrVector] = None,
        stdev_max_change: Optional[RealOrVector] = None,
        obj_index: Optional[int] = None,
        distributed: bool = False,
        popsize_weighted_grad_avg: Optional[bool] = None,
    ):
        """
        `__init__(...)`: Initialize the SNES algorithm.

        Args:
            problem: The problem object which is being worked on.
            stdev_init: The initial standard deviation of the search
                distribution, expressed as a scalar or as an array.
                Determines the initial coverage area of the search
                distribution.
                If one wishes to configure the coverage area via the
                argument `radius_init` instead, then `stdev_init` is expected
                as None.
            radius_init: The initial radius of the search distribution,
                expressed as a scalar.
                Determines the initial coverage area of the search
                distribution.
                Here, "radius" is defined as the norm of the search
                distribution.
                If one wishes to configure the coverage area via the
                argument `stdev_init` instead, then `radius_init` is expected
                as None.
            popsize: Population size. Can be specified as an int,
                or can be left as None to let the solver decide.
                In the case of SNES, `popsize` can be left as None,
                in which case the default `popsize` will be computed
                as `4 + floor(3 * log(n))` where `n` is the length
                of a solution.
            center_learning_rate: Learning rate for updating the mean
                of the search distribution. Default value is 1.0
            stdev_learning_rate: Learning rate for updating the covariance
                matrix of the search distribution.
                The default value is `0.2 * (3 + log(n)) / sqrt(n)`
                where `n` is the length of a solution.
            scale_learning_rate: For SNES, there is a default standard
                deviation learning rate value which is computed as
                `0.2 * (3 + log(n)) / sqrt(n)` (where `n` is the solution
                length).
                If scale_learning_rate is True (which is the default),
                then the effective learning rate for the standard deviation
                becomes the provided `stdev_learning_rate` multiplied by this
                default value. If `scale_learning_rate` is False, then the
                effective standard deviation learning rate becomes
                equal to the provided `stdev_learning_rate` value.
            num_interactions: When given as an integer n,
                it is ensured that a population has interacted with
                the GymProblem's environment n times. If this target
                has not been reached yet, then the population is declared
                too small, and gets extended with more samples,
                until n amount of interactions is reached.
                When given as None, popsize is the only configuration
                affecting the size of a population.
            popsize_max: Having `num_interactions` set as an integer
                might cause the effective population size jump to
                unnecesarily large numbers. To prevent this,
                one can set `popsize_max` to specify an upper
                bound for the effective population size.
            num_interactions: When given as an integer n,
                it is ensured that a population has interacted with
                the GymProblem's environment n times. If this target
                has not been reached yet, then the population is declared
                too small, and gets extended with more samples,
                until n amount of interactions is reached.
                When given as None, popsize is the only configuration
                affecting the size of a population.
            popsize_max: Having `num_interactions` set as an integer
                might cause the effective population size jump to
                unnecesarily large numbers. To prevent this,
                one can set `popsize_max` to specify an upper
                bound for the effective population size.
            optimizer: The optimizer to be used while following the
                estimated the gradients.
                Can be given as None if a momentum-based optimizer
                is not required.
                Otherwise, can be given as a str containing the name
                of the optimizer (e.g. 'adam', 'clipup');
                or as an instance of evotorch.optimizers.TorchOptimizer
                or evotorch.optimizers.ClipUp.
                The default is None.
                Note that, for ClipUp, the default maximum speed is set
                as twice the given `center_learning_rate`.
                This maximum speed can be configured by passing
                `{"max_speed": ...}` to `optimizer_config`.
            optimizer_config: Configuration which will be passed
                to the optimizer as keyword arguments.
                See `evotorch.optimizers` for details about
                which optimizer accepts which keyword arguments.
            ranking_method: Which ranking method will be used for
                fitness shaping. See the documentation of
                `evotorch.ranking.rank(...)` for details.
                The default is 'nes'.
                Can be given as None if no such ranking is required.
            center_init: The initial center solution.
                Can be left as None.
            stdev_min: Minimum values for the standard deviation.
                Expected as a 1-dimensional array to serve as a limiter
                to the diagonals of the covariance matrix's square root.
            stdev_max: Maximum values for the standard deviation.
                Expected as a 1-dimensional array to serve as a limiter
                to the diagonals of the covariance matrix's square root.
            stdev_max_change: Maximum change allowed for when updating
                the square roort of the covariance matrix.
            obj_index: Index of the objective according to which the
                gradient estimations will be done.
                For single-objective problems, this can be left as None.
            distributed: Whether or not the gradient computation will
                be distributed. If `distributed` is given as False and
                the problem is not parallelized, then everything will
                be centralized (i.e. the entire computation will happen
                in the main process).
                If `distributed` is given as False, and the problem
                is parallelized, then the population will be created
                in the main process and then sent to remote workers
                for parallelized evaluation, and then the remote fitnesses
                will be collected by the main process again for computing
                the search gradients.
                If `distributed` is given as True, and the problem
                is parallelized, then the search algorithm itself will
                be distributed, in the sense that each remote actor will
                generate its own population (such that the total population
                size across all these actors becomes equal to `popsize`)
                and will compute its own gradient, and then the main process
                will collect these gradients, compute the averaged gradients
                and update the main search distribution.
                Non-distributed mode has the advantage of keeping the
                population in the main process, which is good when one wishes
                to do detailed monitoring during the evolutionary process,
                but has the disadvantage of having to pass the solutions to
                the remote actors and having to collect fitnesses, which
                might result in increased interprocess communication traffic.
                On the other hand, while it is not possible to monitor the
                population in distributed mode, the distributed mode has the
                advantage of significantly reducing the interprocess
                communication traffic, since the only things communicated
                with the remote actors are the search distributions (not the
                solutions) and the gradients.
            popsize_weighted_grad_avg: Only to be used in distributed mode.
                (where being in distributed mode means `distributed` is given
                as True). In distributed mode, each actor remotely samples
                its own solution batches and computes its own gradients.
                These gradients are then collected, and a final average
                gradient is computed.
                If `popsize_weighted_grad_avg` is True, then, while averaging
                over the gradients, each gradient will have its own weight
                that is computed according to how many solutions were sampled
                by the actor that produced the gradient.
                If `popsize_weighted_grad_avg` is False, then, there will not
                be weighted averaging (or, each gradient will have equal
                weight).
                If `popsize_weighted_grad_avg` is None, then, the gradient
                weights will be equal a value for `num_interactions` is given
                (because `num_interactions` affects the number of solutions
                according to the episode lengths, and popsize-weighting the
                gradients could be misleading); and the gradient weights will
                be weighted according to the sub-population (i.e. sub-batch)
                sizes if `num_interactions` is left as None.
                The default value for `popsize_weighted_grad_avg` is None.
                When the distributed mode is disabled (i.e. when `distributed`
                is False), then the argument `popsize_weighted_grad_avg` is
                expected as None.
        """

        if popsize is None:
            popsize = int(4 + math.floor(3 * math.log(problem.solution_length)))

        if center_learning_rate is None:
            center_learning_rate = 1.0

        def default_stdev_lr():
            n = problem.solution_length
            return 0.2 * (3 + math.log(n)) / math.sqrt(n)

        if stdev_learning_rate is None:
            stdev_learning_rate = default_stdev_lr()
        else:
            stdev_learning_rate = float(stdev_learning_rate)
            if scale_learning_rate:
                stdev_learning_rate *= default_stdev_lr()

        super().__init__(
            problem,
            popsize=popsize,
            center_learning_rate=center_learning_rate,
            stdev_learning_rate=stdev_learning_rate,
            stdev_init=stdev_init,
            radius_init=radius_init,
            popsize_max=popsize_max,
            num_interactions=num_interactions,
            optimizer=optimizer,
            optimizer_config=optimizer_config,
            ranking_method=ranking_method,
            center_init=center_init,
            stdev_min=stdev_min,
            stdev_max=stdev_max,
            stdev_max_change=stdev_max_change,
            obj_index=obj_index,
            distributed=distributed,
            popsize_weighted_grad_avg=popsize_weighted_grad_avg,
        )


class CEM(GaussianSearchAlgorithm):
    """
    The cross-entropy method (CEM) (Rubinstein, 1999).

    This CEM implementation is focused on continuous optimization,
    and follows the variant explained in Duan et al. (2016).

    The adaptive population size mechanism explained in Toklu et al. (2020)
    (and previously used in the accompanying source code of the study
    Salimans et al. (2017)) is supported, where the population size in an
    iteration keeps increasing until a certain numberof interactions with
    the simulator of the reinforcement learning environment is made.
    See the initialization arguments `num_interactions`, `popsize_max`.

    References:

        Rubinstein, R. (1999). The cross-entropy method for combinatorial
        and continuous optimization.
        Methodology and computing in applied probability, 1(2), 127-190.

        Duan, Y., Chen, X., Houthooft, R., Schulman, J., Abbeel, P. (2016).
        Benchmarking deep reinforcement learning for continuous control.
        International conference on machine learning. PMLR, 2016.

        Salimans, T., Ho, J., Chen, X., Sidor, S. and Sutskever, I. (2017).
        Evolution Strategies as a Scalable Alternative to
        Reinforcement Learning.

        Toklu, N.E., Liskowski, P., Srivastava, R.K. (2020).
        ClipUp: A Simple and Powerful Optimizer
        for Distribution-based Policy Evolution.
        Parallel Problem Solving from Nature (PPSN 2020).
    """

    DISTRIBUTION_TYPE = SeparableGaussian
    DISTRIBUTION_PARAMS = NotImplemented  # To be filled by the CEM instance

    def __init__(
        self,
        problem: Problem,
        *,
        popsize: int,
        parenthood_ratio: float,
        stdev_init: Optional[RealOrVector] = None,
        radius_init: Optional[RealOrVector] = None,
        num_interactions: Optional[int] = None,
        popsize_max: Optional[int] = None,
        center_init: Optional[RealOrVector] = None,
        stdev_min: Optional[RealOrVector] = None,
        stdev_max: Optional[RealOrVector] = None,
        stdev_max_change: Optional[Union[float, RealOrVector]] = None,
        obj_index: Optional[int] = None,
        distributed: bool = False,
        popsize_weighted_grad_avg: Optional[bool] = None,
    ):
        """
        `__init__(...)`: Initialize the search algorithm.

        Args:
            problem: The problem object to work on.
            popsize: The population size.
            parenthood_ratio: Expected as a float larger than 0 and smaller
                than 1. For example, setting this value to 0.1 means that
                the top 10% of the population will be declared as the parents,
                and those parents will be used for updating the population.
                The amount of parents is always computed according to the
                specified `popsize`, not according to the adapted population
                size, and not according to `popsize_max`.
            stdev_init: The initial standard deviation of the search
                distribution, expressed as a scalar or as an array.
                Determines the initial coverage area of the search
                distribution.
                If one wishes to configure the coverage area via the
                argument `radius_init` instead, then `stdev_init` is expected
                as None.
            radius_init: The initial radius of the search distribution,
                expressed as a scalar.
                Determines the initial coverage area of the search
                distribution.
                Here, "radius" is defined as the norm of the search
                distribution.
                If one wishes to configure the coverage area via the
                argument `stdev_init` instead, then `radius_init` is expected
                as None.
            num_interactions: When given as an integer n,
                it is ensured that a population has interacted with
                the GymProblem's environment n times. If this target
                has not been reached yet, then the population is declared
                too small, and gets extended with more samples,
                until n amount of interactions is reached.
                When given as None, popsize is the only configuration
                affecting the size of a population.
            popsize_max: Having `num_interactions` set as an integer
                might cause the effective population size jump to
                unnecesarily large numbers. To prevent this,
                one can set `popsize_max` to specify an upper
                bound for the effective population size.
            center_init: The initial center solution.
                Can be left as None.
            stdev_min: The minimum value for the standard deviation
                values of the Gaussian search distribution.
                Can be left as None (which is the default),
                or can be given as a scalar or as a 1-dimensional array.
            stdev_max: The maximum value for the standard deviation
                values of the Gaussian search distribution.
                Can be left as None (which is the default),
                or can be given as a scalar or as a 1-dimensional array.
            stdev_max_change: The maximum update ratio allowed on the
                standard deviation. Expected as None if no such limiter
                is needed, or as a real number within 0.0 and 1.0 otherwise.
                In the PGPE implementation of Ha (2017, 2018), a value of
                0.2 (20%) was used.
                For this CEM implementation, the default is None.
            obj_index: Index of the objective according to which the
                gradient estimations will be done.
                For single-objective problems, this can be left as None.
            distributed: Whether or not the gradient computation will
                be distributed. If `distributed` is given as False and
                the problem is not parallelized, then everything will
                be centralized (i.e. the entire computation will happen
                in the main process).
                If `distributed` is given as False, and the problem
                is parallelized, then the population will be created
                in the main process and then sent to remote workers
                for parallelized evaluation, and then the remote fitnesses
                will be collected by the main process again for computing
                the search gradients.
                If `distributed` is given as True, and the problem
                is parallelized, then the search algorithm itself will
                be distributed, in the sense that each remote actor will
                generate its own population (such that the total population
                size across all these actors becomes equal to `popsize`)
                and will compute its own gradient, and then the main process
                will collect these gradients, compute the averaged gradients
                and update the main search distribution.
                Non-distributed mode has the advantage of keeping the
                population in the main process, which is good when one wishes
                to do detailed monitoring during the evolutionary process,
                but has the disadvantage of having to pass the solutions to
                the remote actors and having to collect fitnesses, which
                might result in increased interprocess communication traffic.
                On the other hand, while it is not possible to monitor the
                population in distributed mode, the distributed mode has the
                advantage of significantly reducing the interprocess
                communication traffic, since the only things communicated
                with the remote actors are the search distributions (not the
                solutions) and the gradients.
            popsize_weighted_grad_avg: Only to be used in distributed mode.
                (where being in distributed mode means `distributed` is given
                as True). In distributed mode, each actor remotely samples
                its own solution batches and computes its own gradients.
                These gradients are then collected, and a final average
                gradient is computed.
                If `popsize_weighted_grad_avg` is True, then, while averaging
                over the gradients, each gradient will have its own weight
                that is computed according to how many solutions were sampled
                by the actor that produced the gradient.
                If `popsize_weighted_grad_avg` is False, then, there will not
                be weighted averaging (or, each gradient will have equal
                weight).
                If `popsize_weighted_grad_avg` is None, then, the gradient
                weights will be equal a value for `num_interactions` is given
                (because `num_interactions` affects the number of solutions
                according to the episode lengths, and popsize-weighting the
                gradients could be misleading); and the gradient weights will
                be weighted according to the sub-population (i.e. sub-batch)
                sizes if `num_interactions` is left as None.
                The default value for `popsize_weighted_grad_avg` is None.
                When the distributed mode is disabled (i.e. when `distributed`
                is False), then the argument `popsize_weighted_grad_avg` is
                expected as None.
        """

        self.DISTRIBUTION_PARAMS = {"parenthood_ratio": float(parenthood_ratio)}

        super().__init__(
            problem,
            popsize=popsize,
            center_learning_rate=1.0,
            stdev_learning_rate=1.0,
            stdev_init=stdev_init,
            radius_init=radius_init,
            popsize_max=popsize_max,
            num_interactions=num_interactions,
            optimizer=None,
            optimizer_config=None,
            ranking_method=None,
            center_init=center_init,
            stdev_min=stdev_min,
            stdev_max=stdev_max,
            stdev_max_change=stdev_max_change,
            obj_index=obj_index,
            distributed=distributed,
            popsize_weighted_grad_avg=popsize_weighted_grad_avg,
        )


class XNES(GaussianSearchAlgorithm):
    """
    XNES: Exponential Natural Evolution Strategies

    Inspired by the implementation at:
    http://schaul.site44.com/code/xnes.py
    https://github.com/pybrain/pybrain/blob/master/pybrain/optimization/distributionbased/xnes.py

    Reference:
        Glasmachers, Tobias, et al.
        Exponential natural evolution strategies.
        Proceedings of the 12th annual conference on Genetic and evolutionary
        computation (GECCO 2010).
    """

    DISTRIBUTION_TYPE = ExpGaussian
    DISTRIBUTION_PARAMS = None

    def __init__(
        self,
        problem: Problem,
        *,
        stdev_init: Optional[RealOrVector] = None,
        radius_init: Optional[RealOrVector] = None,
        popsize: Optional[int] = None,
        center_learning_rate: Optional[float] = None,
        stdev_learning_rate: Optional[float] = None,
        scale_learning_rate: bool = True,
        num_interactions: Optional[int] = None,
        popsize_max: Optional[int] = None,
        optimizer=None,
        optimizer_config: Optional[dict] = None,
        ranking_method: Optional[str] = "nes",
        center_init: Optional[RealOrVector] = None,
        obj_index: Optional[int] = None,
        distributed: bool = False,
        popsize_weighted_grad_avg: Optional[bool] = None,
    ):
        """
        `__init__(...)`: Initialize the XNES algorithm.

        Args:
            problem: The problem object which is being worked on.
            stdev_init: The initial standard deviation of the search
                distribution, expressed as a scalar or as an array.
                Determines the initial coverage area of the search
                distribution.
                If one wishes to configure the coverage area via the
                argument `radius_init` instead, then `stdev_init` is expected
                as None.
            radius_init: The initial radius of the search distribution,
                expressed as a scalar.
                Determines the initial coverage area of the search
                distribution.
                Here, "radius" is defined as the norm of the search
                distribution.
                If one wishes to configure the coverage area via the
                argument `stdev_init` instead, then `radius_init` is expected
                as None.
            popsize: Population size. Can be specified as an int,
                or can be left as None to let the solver decide.
                In the case of SNES, `popsize` can be left as None,
                in which case the default `popsize` will be computed
                as `4 + floor(3 * log(n))` where `n` is the length
                of a solution.
            center_learning_rate: Learning rate for updating the mean
                of the search distribution. Default value is 1.0
            stdev_learning_rate: Learning rate for updating the covariance
                matrix of the search distribution.
                The default value is `0.6 * (3 + log(n)) / (n * sqrt(n))`
                where `n` is the length of a solution.
            scale_learning_rate: For SNES, there is a default standard
                deviation learning rate value which is computed as
                `0.6 * (3 + log(n)) / (n * sqrt(n))` (where `n` is the solution
                length).
                If scale_learning_rate is True (which is the default),
                then the effective learning rate for the standard deviation
                becomes the provided `stdev_learning_rate` multiplied by this
                default value. If `scale_learning_rate` is False, then the
                effective standard deviation learning rate becomes
                equal to the provided `stdev_learning_rate` value.
            num_interactions: When given as an integer n,
                it is ensured that a population has interacted with
                the GymProblem's environment n times. If this target
                has not been reached yet, then the population is declared
                too small, and gets extended with more samples,
                until n amount of interactions is reached.
                When given as None, popsize is the only configuration
                affecting the size of a population.
            popsize_max: Having `num_interactions` set as an integer
                might cause the effective population size jump to
                unnecesarily large numbers. To prevent this,
                one can set `popsize_max` to specify an upper
                bound for the effective population size.
            num_interactions: When given as an integer n,
                it is ensured that a population has interacted with
                the GymProblem's environment n times. If this target
                has not been reached yet, then the population is declared
                too small, and gets extended with more samples,
                until n amount of interactions is reached.
                When given as None, popsize is the only configuration
                affecting the size of a population.
            optimizer: The optimizer to be used while following the
                estimated the gradients.
                Can be given as None if a momentum-based optimizer
                is not required.
                Otherwise, can be given as a str containing the name
                of the optimizer (e.g. 'adam', 'clipup');
                or as an instance of evotorch.optimizers.TorchOptimizer
                or evotorch.optimizers.ClipUp.
                The default is None.
                Note that, for ClipUp, the default maximum speed is set
                as twice the given `center_learning_rate`.
                This maximum speed can be configured by passing
                `{"max_speed": ...}` to `optimizer_config`.
            optimizer_config: Configuration which will be passed
                to the optimizer as keyword arguments.
                See `evotorch.optimizers` for details about
                which optimizer accepts which keyword arguments.
            ranking_method: Which ranking method will be used for
                fitness shaping. See the documentation of
                `evotorch.ranking.rank(...)` for details.
                The default is 'nes'.
                Can be given as None if no such ranking is required.
            center_init: The initial center solution.
                Can be left as None.
            obj_index: Index of the objective according to which the
                gradient estimations will be done.
                For single-objective problems, this can be left as None.
            distributed: Whether or not the gradient computation will
                be distributed. If `distributed` is given as False and
                the problem is not parallelized, then everything will
                be centralized (i.e. the entire computation will happen
                in the main process).
                If `distributed` is given as False, and the problem
                is parallelized, then the population will be created
                in the main process and then sent to remote workers
                for parallelized evaluation, and then the remote fitnesses
                will be collected by the main process again for computing
                the search gradients.
                If `distributed` is given as True, and the problem
                is parallelized, then the search algorithm itself will
                be distributed, in the sense that each remote actor will
                generate its own population (such that the total population
                size across all these actors becomes equal to `popsize`)
                and will compute its own gradient, and then the main process
                will collect these gradients, compute the averaged gradients
                and update the main search distribution.
                Non-distributed mode has the advantage of keeping the
                population in the main process, which is good when one wishes
                to do detailed monitoring during the evolutionary process,
                but has the disadvantage of having to pass the solutions to
                the remote actors and having to collect fitnesses, which
                might result in increased interprocess communication traffic.
                On the other hand, while it is not possible to monitor the
                population in distributed mode, the distributed mode has the
                advantage of significantly reducing the interprocess
                communication traffic, since the only things communicated
                with the remote actors are the search distributions (not the
                solutions) and the gradients.
            popsize_weighted_grad_avg: Only to be used in distributed mode.
                (where being in distributed mode means `distributed` is given
                as True). In distributed mode, each actor remotely samples
                its own solution batches and computes its own gradients.
                These gradients are then collected, and a final average
                gradient is computed.
                If `popsize_weighted_grad_avg` is True, then, while averaging
                over the gradients, each gradient will have its own weight
                that is computed according to how many solutions were sampled
                by the actor that produced the gradient.
                If `popsize_weighted_grad_avg` is False, then, there will not
                be weighted averaging (or, each gradient will have equal
                weight).
                If `popsize_weighted_grad_avg` is None, then, the gradient
                weights will be equal a value for `num_interactions` is given
                (because `num_interactions` affects the number of solutions
                according to the episode lengths, and popsize-weighting the
                gradients could be misleading); and the gradient weights will
                be weighted according to the sub-population (i.e. sub-batch)
                sizes if `num_interactions` is left as None.
                The default value for `popsize_weighted_grad_avg` is None.
                When the distributed mode is disabled (i.e. when `distributed`
                is False), then the argument `popsize_weighted_grad_avg` is
                expected as None.
        """

        if popsize is None:
            popsize = int(4 + math.floor(3 * math.log(problem.solution_length)))

        if center_learning_rate is None:
            center_learning_rate = 1.0

        def default_stdev_lr():
            n = problem.solution_length
            return 0.6 * (3 + math.log(n)) / (n * math.sqrt(n))

        if stdev_learning_rate is None:
            stdev_learning_rate = default_stdev_lr()
        else:
            stdev_learning_rate = float(stdev_learning_rate)
            if scale_learning_rate:
                stdev_learning_rate *= default_stdev_lr()

        super().__init__(
            problem,
            popsize=popsize,
            center_learning_rate=center_learning_rate,
            stdev_learning_rate=stdev_learning_rate,
            stdev_init=stdev_init,
            radius_init=radius_init,
            popsize_max=popsize_max,
            num_interactions=num_interactions,
            optimizer=optimizer,
            optimizer_config=optimizer_config,
            ranking_method=ranking_method,
            center_init=center_init,
            stdev_min=None,
            stdev_max=None,
            stdev_max_change=None,
            obj_index=obj_index,
            distributed=distributed,
            popsize_weighted_grad_avg=popsize_weighted_grad_avg,
        )
