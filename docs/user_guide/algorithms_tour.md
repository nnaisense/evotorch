# Available Evolutionary Algorithms in EvoTorch
EvoTorch provides a number of modern Evolutionary Algorithms. For our initial release, we have limited the available algorithms to those commonly used in Neuroevolution and Planning literature, and we will expand the available algorithms in future releases. The reason for this is that implementing correct evolutionary algorithms is non-trivial, so we take great care before releasing any algorithm so that our library may be used in scientific and industrial settings with confidence.

To discuss the algorithms available in EvoTorch, we will distinguish between *distribution-based* and *population-based* evolutionary algorithms.

## Population-based Algorithms
A population-based evolutionary algorithm follows the 'conventional' model of evolutionary algorithms; there exists a population $X = x_1 \dots x_n$ and associated fitness values $F = f_1 \dots f_n$. In each iteration, an interleaving of Selection $S$, Recombination $R$ and Mutation $C$ operators are applied to produce a new population, biased towards producing solutions which are similar to those best-performing solutions of the previous population, following,

\[
    X = <S,R,C>(X, F)
\]

In the table below you will find details of the currently-available population-based algorithms in EvoTorch.

Algorithm | Description | References
------------ | ------------- | ------------
[SteadyStateGA][evotorch.algorithms.ga.SteadyStateGA] | A steady-state Genetic Algorithm which can be reparameterised with mutation/crossover operators. If used on a multi-objective problem, this algorithm will behave similarly to NSGA-II.  | [Essentials of Metaheuristics](http://cs.gmu.edu/~sean/book/metaheuristics/) [NSGA II](https://ieeexplore.ieee.org/abstract/document/996017)
[CoSyNE][evotorch.algorithms.ga.Cosyne] | The neuroevolution-specialised Genetic Algorithm 'Cooperative Synapse Neuroevolution'. This algorithm maintains subpopulations of individual weights which are cooperatively co-evolved. | [Accelerated Neural Evolution through Cooperatively Coevolved Synapses](https://jmlr.org/beta/papers/v9/gomez08a.html)


## Distribution-based Algorithms
A distribution-based evolutionary algorithm maintains a set of parameters $\theta$ that parameterise a family of probability distributions $\pi$ such that in each generation, a population $X$ may be sampled from the parameterised distribution $\pi(\theta)$,

\[
    X = x_1 \dots x_n \sim \pi(\theta)
\]

The fitness values $f_1 \dots f_n$ are then used to compute a change in $\theta$, $\triangledown \theta$ which is typically followed in a gradient-descent manner,

\[
    \theta = \theta + \alpha \triangledown \theta
\]

The name *distribution-based* comes from the idea that, rather than maintaining an explicit population which competes, survives and reproduces, the population is *modelled* by a distribution $\pi(\theta)$ over solutions, and  the parameters of the distribution $\theta$ are updated according to the samples drawn from the model of the population in each iteration.

Almost all modern distribution-based evolutionary algorithms incorporate the [Natural Gradient](https://ieeexplore.ieee.org/abstract/document/6790500) due to its appealing property of re-parameterisation invariance, and therefore fall under the related frameworks of [Natural Evolution Strategies](https://jmlr.org/papers/v15/wierstra14a.html) and [Information-Geometric Optimization Algorithms](https://jmlr.org/papers/v18/14-467.html).

In the table below you will find details of the currently-available distribution-based algorithms in EvoTorch.

Algorithm | Description | References
------------ | ------------- | ------------
[XNES][evotorch.algorithms.distributed.gaussian.XNES] | Exponential Natural Evolution Strategies. | [Exponential Natural Evolution Strategies](https://dl.acm.org/doi/10.1145/1830483.1830557)
[SNES][evotorch.algorithms.distributed.gaussian.SNES] | Separable Natural Evolution Strategies, the separable variant of Exponential Natural Evolution Strategies. | [High dimensions and Heavy Tails for Natural Evolution Strategies](https://dl.acm.org/doi/10.1145/2001576.2001692)
[PGPE][evotorch.algorithms.distributed.gaussian.PGPE] | Policy-Gradient Parameter Exploration, a variant of Natural Evolution Strategies which was derived specifically for reinforcement learning tasks. It differs from SNES by the exponential parameterisation that SNES inherits from Exponential Natural Evolution Strategies | [Parameter-exploring policy gradients](https://www.sciencedirect.com/science/article/pii/S0893608009003220)
[CEM][evotorch.algorithms.distributed.gaussian.CEM] | The Cross-Entropy Method. This CEM implementation is focused on continuous optimization, and follows the variant explained in Duan et al. (2016). | [Benchmarking Deep Reinforcement Learning for Continuous Control](https://proceedings.mlr.press/v48/duan16.html)
[CMA-ES][evotorch.algorithms.cmaes.CMAES] | The Covariance Matrix Adaptation Evolution Strategy. Due to the complexity and implementation-specific nature of this algorithm, we have opted to create an interface to the popular [pycma](https://github.com/CMA-ES/pycma) library which is created and maintained by the authors of the algorithm. As of writing this documentation, pycma generates its populations as regular lists of separate numpy arrays. Therefore, when using this algorithm with large population sizes, you might notice slow-downs. Also, since it is built on numpy, pycma is CPU-bound. | [Completely Derandomized Self-Adaptation in Evolution Strategies](https://ieeexplore.ieee.org/document/6790628)
