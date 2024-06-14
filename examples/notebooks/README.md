# Example notebooks

This directory contains Jupyter notebooks for demonstrating both the API and the capabilities of EvoTorch.

## Requirements

These examples require either JupyterLab or Jupyter Notebook. See [here](https://jupyter.org/install) for installation instructions.

The notebook which demonstrates gym experiments requires the gym library with its `box2d` extensions enabled. The following `pip` installation instruction ensures that gym is installed with the `box2d` extensions:

```bash
pip install 'gym[box2d]'
```

## Notebooks

The notebook examples are listed below:

- **[Gym Experiments with PGPE and CoSyNE](Gym_Experiments_with_PGPE_and_CoSyNE.ipynb):** demonstrates how you can solve "LunarLanderContinuous-v2" using both `PGPE` and `CoSyNE` following the configurations described in [the paper proposing ClipUp](https://dl.acm.org/doi/abs/10.1007/978-3-030-58115-2_36) and [the JMLR paper on the CoSyNE algorithm](https://www.jmlr.org/papers/volume9/gomez08a/gomez08a.pdf).
- **[Brax Experiments with PGPE](Brax_Experiments_with_PGPE.ipynb):** demonstrates how you can solve the brax task "humanoid" using PGPE, with GPU support, if available. See also [Brax_Experiments_Visualization.ipynb](Brax_Experiments_Visualization.ipynb) for visualizing evolved brax policies.
- **[Minimizing Lennard-Jones Atom Cluster Potentials](Minimizing_Lennard-Jones_Atom_Cluster_Potentials.ipynb):** recreates experiments from [the paper introducing `SNES`](https://dl.acm.org/doi/abs/10.1145/2001576.2001692), showing that the algorithm can effectively solve the challenging task of [minimising Lennard-Jones atom cluster potentials](https://pubs.acs.org/doi/abs/10.1021/jp970984n).
- **[Model Predictive Control with CEM](Model_Predictive_Control_with_CEM/):** demonstrates the application of [the Cross-Entropy Method `CEM`](https://link.springer.com/article/10.1023/A:1010091220143) to Model Predictive Control (MPC) of the MuJoCo task named "Reacher-v4".
- **[Training MNIST30K](Training_MNIST30K.ipynb):** recreates experiments [from a recent paper](https://www.deepmind.com/publications/non-differentiable-supervised-learning-with-evolution-strategies-and-hybrid-methods) which demonstrates that `SNES` can be used to solve supervised learning problems. The script in particular recreates the training of the 30K-parameter 'MNIST30K' model on the MNIST dataset, but can easily be reconfigured to recreate other experiments from that paper.
- **[Variational Quantum Eigensolvers with SNES](Variational_Quantum_Eigensolvers_with_SNES.ipynb):** re-implements (with some minor changes in experimental setup), [experiments in a recent paper](https://iopscience.iop.org/article/10.1088/2632-2153/abf3ac) demonstrating that `SNES` is a scalable alternative to [analytic gradients on a quantum computer](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.99.032331), and can practically optimize [Quantum Eigensolvers](https://www.nature.com/articles/ncomms5213).
- **[Genetic Programming](Genetic_Programming.ipynb):** demonstrates genetic programming with GPU support.
- **[Feature Space Illumination with MAPElites](Feature_Space_Illumination_with_MAPElites.ipynb):** demonstrates how one can use the MAPElites algorithm to obtain a population organized according to the features of the solutions.
- **[Evolving_Objects](Evolving_Objects.ipynb):** demonstrates how to declare and solve optimization problems with custom structured solutions (storing not-necessarily numeric data and/or having varying lengths). In more details, this example evolves simple gaits for the `Ant-v4` reinforcement learning environment using a custom solution encoding such that each solution contains multiple sublists of integers.
- **[Functional API](Functional_API/)**: As an alternative to its object-oriented stateful API, EvoTorch provides an API that conforms to the functional programming paradigm. This functional API has its own advantages like being able to work on not just a single population, but on a batch of populations. This sub-directory contains examples demonstrating the functional API of EvoTorch.
