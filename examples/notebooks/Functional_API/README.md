# Functional API for Optimization

As an alternative to its object-oriented stateful API, EvoTorch provides an API that conforms to the functional programming paradigm. This functional API has its own advantages like being able to work on not just a single population, but on a batch of populations.

Here are the examples demonstrating various features of this functional API:

- **[Maintaining a batch of populations using the functional EvoTorch API](batched_searches.ipynb)**: This notebook shows how one can efficiently run multiple searches simultaneously, each with its own population and hyperparameter configuration, by maintaining a batch of populations.
- **[Functional genetic algorithm operators](functional_ops.ipynb)**: This notebook shows how one can implement a custom genetic algorithm by combining the genetic algorithm operator implementations provided by the functional API of EvoTorch.
- **[Solving constrained optimization problems](constrained.ipynb)**: EvoTorch provides batching-friendly constraint penalization functions that can be used with both the object-oriented API and the functional API. In addition, these constraint penalization functions can be used with gradient-based optimization. This notebook demonstrates these features.
- **[Solving reinforcement learning tasks using functional evolutionary algorithms](problem.ipynb)**: The functional evolutionary algorithm implementations of EvoTorch can be used to solve problems that are expressed using the object-oriented core API of EvoTorch. To demonstrate this, this notebook instantiates a `GymNE` problem for the reinforcement learning task "CartPole-v1", and solves it using the functional `pgpe` implementation.
