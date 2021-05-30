import unittest

import numpy as np
import pandas as pd
from parameterized import parameterized

from evt import utils


class TestUtils(unittest.TestCase):
    def test_empirical_cdf(self):
        dataset = pd.Series([15, 14, 14, 13], index=list(range(20, 24)))
        result = utils.empirical_cdf(dataset)
        self.assertTrue(np.all(np.isclose(
            [0.2, 0.4, 0.8],
            result.values
        )))
        self.assertTrue(np.all(np.isclose(
            [13, 14, 15],
            result.index
        )))

    def test_mean_excess(self):
        series = pd.Series([1, 2, 3, 3, 4], index=[10, 11, 12, 13, 14])
        mean_excess = utils.mean_excess(series)
        self.assertTrue(np.all(np.isclose(
            [1, 2, 3],
            mean_excess.index
        )))
        self.assertTrue(np.all(np.isclose(
            [2, 4/3, 1],
            mean_excess.values
        )))

    @parameterized.expand([
        (0.012, 2, r'$1.20 \cdot 10^{-2}$'),
        (0.12, 2, r'$1.20 \cdot 10^{-1}$'),
        (1.2, 2, '$1.20 $'),
        (12, 2, r'$1.20 \cdot 10^{1}$'),
        (120, 2, r'$1.20 \cdot 10^{2}$'),
        (0.012, 1, r'$1.2 \cdot 10^{-2}$'),
        (0.12, 1, r'$1.2 \cdot 10^{-1}$'),
        (1.2, 1, '$1.2 $'),
        (12, 1, r'$1.2 \cdot 10^{1}$'),
        (120, 1, r'$1.2 \cdot 10^{2}$'),
        (0.012, 0, r'$1 \cdot 10^{-2}$'),
        (0.12, 0, r'$1 \cdot 10^{-1}$'),
        (1.2, 0, '$1 $'),
        (12, 0, r'$1 \cdot 10^{1}$'),
        (120, 0, r'$1 \cdot 10^{2}$'),
    ])
    def test_scientific_notation(
            self,
            number,
            number_of_significant_digits,
            expected_result
    ):
        result = utils.scientific_notation(number, number_of_significant_digits)
        self.assertEqual(expected_result, result)

    def test_order_statistics(self):
        series = pd.Series([14, 15, 14, 13], index=list(range(20, 24)))
        result = utils.order_statistics(series)
        self.assertTrue(np.all(np.isclose(
            [0, 1, 2, 3],
            result.index
        )))
        self.assertTrue(np.all(np.isclose(
            [15, 14, 14, 13],
            result.values
        )))

    @parameterized.expand([
        (.95, 1.959963984540054),
        (.99, 2.5758293035489004),
        (0, 0),
    ])
    def test_confidence_interval_to_std(self, confidence, desired_result):
        result = utils.confidence_interval_to_std(confidence)
        self.assertAlmostEqual(desired_result, result)
