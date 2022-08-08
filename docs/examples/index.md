<style>
  .md-typeset h1 {
    display: none;
  }
</style>

We provide a number of reference examples directly on our [GitHub](https://github.com/nnaisense/evotorch/tree/master/examples). Each of these examples demonstrates how to recreate a particular experiment or result from recent evolutionary algorithm literature, to highlight that Evotorch is highly suited to both academic research in and advanced industrial applications of evolutionary algorithms.

We provide a number of examples as [jupyter notebooks](https://jupyter.org/). The easiest way to get started with these examples is to run:

```bash
cd evotorch
pip install jupyterlab
jupyter lab
```

In the `examples/notebooks/` directory, you can find the following notebooks:

- **[Gym Experiments with PGPE and CoSyNE](notebooks/Gym_Experiments_with_PGPE_and_CoSyNE.md)** demonstrates how you can solve "LunarLanderContinuous-v2" using both `PGPE` and `CoSyNE` following the configurations described in [the paper proposing ClipUp](https://dl.acm.org/doi/abs/10.1007/978-3-030-58115-2_36) and [the JMLR paper on the CoSyNE algorithm](https://www.jmlr.org/papers/volume9/gomez08a/gomez08a.pdf).
- **[Minimizing Lennard-Jones Atom Cluster Potentials](notebooks/Minimizing_Lennard-Jones_Atom_Cluster_Potentials.md)** recreates experiments from [the paper introducing `SNES`](https://dl.acm.org/doi/abs/10.1145/2001576.2001692), showing that the algorithm can effectively solve the challenging task of [minimising Lennard-Jones atom cluster potentials](https://pubs.acs.org/doi/abs/10.1021/jp970984n).
- **[Model_Predictive_Control_with_CEM](notebooks/reacher_mpc.md)** demonstrates the application of [the Cross-Entropy Method `CEM`](https://link.springer.com/article/10.1023/A:1010091220143) to Model Predictive Control (MPC) of the MuJoCo task named "Reacher-v4".
- **[Training MNIST30K](notebooks/Training_MNIST30K.md)** recreates experiments [from a recent paper](https://www.deepmind.com/publications/non-differentiable-supervised-learning-with-evolution-strategies-and-hybrid-methods) which demonstrates that `SNES` can be used to solve supervised learning problems. The script in particular recreates the training of the 30K-parameter 'MNIST30K' model on the MNIST dataset, but can easily be reconfigured to recreate other experiments from that paper.
- **[Variational Quantum Eigensolvers with SNES](notebooks/Variational_Quantum_Eigensolvers_with_SNES.md)** re-implements (with some minor changes in experimental setup), [experiments in a recent paper](https://iopscience.iop.org/article/10.1088/2632-2153/abf3ac) demonstrating that `SNES` is a scalable alternative to [analytic gradients on a quantum computer](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.99.032331), and can practically optimize [Quantum Eigensolvers](https://www.nature.com/articles/ncomms5213).

In addition, to help you to implement advanced neuroevolutionary reinforcement learning settings, we have provided 3 python scripts in the `examples/scripts` directory:

- `rl.py` re-implements almost all experiments from [the paper proposing ClipUp](https://dl.acm.org/doi/abs/10.1007/978-3-030-58115-2_36), and is easily reconfigured to replicate any particular experiment using [`sacred`](https://sacred.readthedocs.io/en/stable/quickstart.html).
- `rl_enjoy.py` allows you to easily visualize and enjoy agents trained through `rl.py`.
- `tinytraj_humanoid_bullet.py` implements the modified `"pybullet_envs:HumanoidBulletEnv-v0"` environment from the paper.
