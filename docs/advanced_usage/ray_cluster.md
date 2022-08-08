This short tutorial will guide you through the basics of setting up a [ray cluster](https://docs.ray.io/en/latest/ray-core/configure.html) to run evaluation across multiple CPUs and multiple machines. For more advanced configurations, please refer to the ray documentation.

To use EvoTorch problems across multiple machines, *before* starting the python environment, do the following steps:

1. Make sure that the same Python environment, libraries, and your dependencies exist in all machines.

2. In the terminal of the head node, execute the following command:

```bash
ray start --head
```

3. In the terminal of each non-head node, execute the following command:

```bash
ray start --address=ADDRESS_OF_THE_HEAD_NODE
```

Once these steps have been performed, when you launch your python script, do

```python
import ray
ray.init(address=ADDRESS_OF_THE_HEAD_NODE)
```

This will ensure that your calls to [Problem][evotorch.core.Problem] instances using `ray` will use the cluster that you have created in the previous steps. From here, any [Problem][evotorch.core.Problem] class that is instantiated with

```python
problem = Problem(
    ...
    num_actors = 'max',
)
```

will use *all* CPUs available on the entire cluster.
