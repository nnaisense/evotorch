---
hide:
  - navigation
---

<style>
  .md-typeset h1 {
    display: none;
  }
</style>

![evotorch](assets/evotorch.svg)

Welcome to the official documentation of EvoTorch!

## What is EvoTorch?

EvoTorch is an advanced evolutionary algorithm library built directly on top of [PyTorch](https://pytorch.org/) and developed by the research scientists at [NNAISENSE](https://nnaisense.com/). It places a variety of modern, state-of-the-art, evolutionary algorithms at your fingertips, allowing you to optimise and evolve solutions to challenging problems on the fly. EvoTorch provides you with many state-of-the-art features:

- Choose among state-of-the-art distribution-based and population-based evolutionary algorithms.
- Scale up your evolutionary search through the use of [Ray](https://www.ray.io/) clusters.
- Accelerate your evolutionary search further by placing both your fitness function and your search algorithm on CUDA-accelerated hardware.
- Incorporate modern machine-learning logging software to track your experiments.
- Exploit the modular nature of EvoTorch to build custom evolutionary algorithms and perform academic and industrial experiments with all of the above, instantly.

## How do I use EvoTorch?

Simply install with pip:
```
pip install evotorch
```

Head to [Quickstart](quickstart.md) for basic instructions using EvoTorch. From there, we recommend you visit our extensive [User Guide](user_guide/general_usage.md) for guidance on using algorithms and logging, creating problems, scaling up experiments with parallel processing and accelerated hardware and specific support for scalable neuroevolution.

For advanced users, we suggest you visit our [Advanced Usage](advanced_usage/solution_batch.md) section, which provides additional information on advanced algorithm usage, creation of custom search algorithms and logging, and using clusters of compute resources for even larger experiments. These resources are great if you are an academic or industrial user wanting to adapt EvoTorch to your research or application needs.

To demonstrate the power and reach of EvoTorch, we've reimplemented experiments from a variety of recent academic papers, covering to black-box optimization, quantum machine learning, model-predictive control, reinforcement learning and supervised neuroevolution. You can find out more by visiting our [Examples](examples/index.md).
