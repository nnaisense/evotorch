import numpy as np
import torch

from evotorch.bbo import bbob_noiseless_suite, bbob_problem


def test_bbob_noiseless_suite_global_optima():

    n_functions = len(bbob_noiseless_suite._functions)

    dimensions = [2, 5, 10, 20, 40]

    for dimension in dimensions:

        for function_idx in range(1, n_functions + 1):
            func: bbob_problem.BBOBProblem = bbob_noiseless_suite.get_function_i(function_idx)(dimension)
            batch = func.generate_batch(5)
            batch[0].set_values(func._x_opt)
            func.evaluate(batch)
            eval_of_x_opt = float(batch.evals[0] - func._f_opt)

            assert np.abs(eval_of_x_opt - 0.0) < 1e-7
