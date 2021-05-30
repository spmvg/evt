import unittest

import numpy as np
from evt._compiled_expressions.compiled_expressions import gevmle_fisher_information


class TestGEVMLECompiledExpressions(unittest.TestCase):
    def test_gevmle_fisher_information(self):
        g, a, s = 2, 3, 4
        x = np.array(range(10, 14))
        fisher_information = gevmle_fisher_information(
            x,
            g,
            a,
            s
        )
        self.assertTrue(np.all(np.isclose(
            np.array([
                [-0.07018937, 0.00587819, 0.01286121],
                [0.00587819, -0.01096091, 0.00813489],
                [0.01286121, 0.00813489, 0.01900663]
            ]),
            fisher_information
        )))
