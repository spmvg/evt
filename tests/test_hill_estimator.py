import hashlib
import io
import unittest

import numpy as np
import pandas as pd
from evt.dataset import Dataset
from evt.estimators.hill import Hill
from evt.methods.peaks_over_threshold import PeaksOverThreshold
import matplotlib.pyplot as plt


class TestHillEstimator(unittest.TestCase):
    def setUp(self) -> None:
        self.series = pd.Series([np.exp(exponent) for exponent in range(5)])
        self.peaks_over_threshold = PeaksOverThreshold(
            Dataset(self.series),
            threshold=0
        )
        self.hill_estimator = Hill(self.peaks_over_threshold)

    def test_zero_order_statistics(self):
        with self.assertRaises(ValueError):
            self.hill_estimator.estimate(0)

    def test_too_many_order_statistics(self):
        with self.assertRaises(ValueError):
            self.hill_estimator.estimate(5)  # there are 5 datapoints, so we cannot get the 5th order statistic

    def test_hill_estimator(self):
        (estimate, ci_lower, ci_upper), = self.hill_estimator.estimate(3)
        self.assertAlmostEqual(2, estimate)
        self.assertAlmostEqual(-0.2631714681523434, ci_lower)
        self.assertAlmostEqual(4.263171468152343, ci_upper)

    def test_plot(self):
        fig = plt.figure(figsize=(8, 6))
        ax = plt.gca()

        out_file = io.BytesIO()
        self.hill_estimator.plot(ax)
        fig.savefig(out_file, format='raw')
        out_file.seek(0)
        hashed = hashlib.md5(out_file.read()).digest()

        self.assertEqual(
            b"\xe5%1\xec\x81\xc5m\xcc\x93\t\xc1+\xee\x12'\x9a",
            hashed
        )
