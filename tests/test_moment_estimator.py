import hashlib
import io
import unittest

import numpy as np
import pandas as pd
from evt.dataset import Dataset
from evt.estimators.moment import Moment
from evt.methods.peaks_over_threshold import PeaksOverThreshold
import matplotlib.pyplot as plt


class TestMomentEstimator(unittest.TestCase):
    def setUp(self) -> None:
        self.series = pd.Series([np.exp(exponent) for exponent in range(5)])
        self.peaks_over_threshold = PeaksOverThreshold(
            Dataset(self.series),
            threshold=0
        )
        self.moment_estimator = Moment(self.peaks_over_threshold)

    def test_zero_order_statistics(self):
        with self.assertRaises(ValueError):
            self.moment_estimator.estimate(0)

    def test_too_many_order_statistics(self):
        with self.assertRaises(ValueError):
            self.moment_estimator.estimate(5)  # there are 5 datapoints, so we cannot get the 5th order statistic

    def test_moment_estimator_negative(self):
        (estimate, ci_lower, ci_upper), = self.moment_estimator.estimate(3)
        self.assertAlmostEqual(-.5, estimate)
        self.assertAlmostEqual(-2.0181815742579925, ci_lower)
        self.assertAlmostEqual(1.0181815742579923, ci_upper)

    def test_moment_estimator_positive(self):
        (estimate, ci_lower, ci_upper), = self.moment_estimator.estimate(4)
        self.assertAlmostEqual(.5, estimate)
        self.assertAlmostEqual(-0.5956531757207271, ci_lower)
        self.assertAlmostEqual(1.595653175720727, ci_upper)

    def test_plot(self):
        fig = plt.figure(figsize=(8, 6))
        ax = plt.gca()

        out_file = io.BytesIO()
        self.moment_estimator.plot(ax)
        fig.savefig(out_file, format='raw')
        out_file.seek(0)
        hashed = hashlib.md5(out_file.read()).digest()

        self.assertEqual(
            b'x\xd6\xf9@bO\x7f\x8b\xd0V\xf7]\x82\xc6l\x1f',
            hashed
        )
