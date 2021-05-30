import hashlib
import io
import unittest

import numpy as np
import pandas as pd
from evt.dataset import Dataset
from evt.estimators.gpdmle import GPDMLE
from evt.methods.peaks_over_threshold import PeaksOverThreshold
import matplotlib.pyplot as plt


class TestGPDMLE(unittest.TestCase):
    def setUp(self) -> None:
        self.series = pd.Series([np.exp(exponent) for exponent in range(5)])
        self.peaks_over_threshold = PeaksOverThreshold(
            Dataset(self.series),
            threshold=0
        )
        self.gpdmle = GPDMLE(self.peaks_over_threshold)

    def test_gpdmle(self):
        (
            (tail_index_estimate, tail_index_ci_lower, tail_index_ci_upper),
            (scale_estimate, scale_ci_lower, scale_ci_upper),
        ) = self.gpdmle.estimate()

        self.assertAlmostEqual(0.4850660575136194, tail_index_estimate)
        self.assertAlmostEqual(-0.816627816142266, tail_index_ci_lower)
        self.assertAlmostEqual(1.7867599311695048, tail_index_ci_upper)

        self.assertAlmostEqual(10.164192059236363, scale_estimate)
        self.assertAlmostEqual(8.594893265235651, scale_ci_lower)
        self.assertAlmostEqual(11.733490853237075, scale_ci_upper)

    def test_runtime_error(self):
        fig = plt.figure(figsize=(8, 6))
        ax = plt.gca()

        out_file = io.BytesIO()
        with self.assertRaises(RuntimeError):
            self.gpdmle.plot_qq_gpd(ax)

    def test_plot_qq_gev(self):
        self.gpdmle.estimate()

        fig = plt.figure(figsize=(8, 6))
        ax = plt.gca()

        out_file = io.BytesIO()
        self.gpdmle.plot_qq_gpd(ax)
        fig.savefig(out_file, format='raw')
        out_file.seek(0)
        hashed = hashlib.md5(out_file.read()).digest()

        self.assertEqual(
            b'y\n\x8b\xaa\x17\x85\x01\x85\xe8m\xc7\xb6<-\x03F',
            hashed
        )
