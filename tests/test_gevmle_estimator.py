import hashlib
import io
import unittest

import pandas as pd
from evt.dataset import Dataset
from evt.estimators.gevmle import GEVMLE
from evt.methods.block_maxima import BlockMaxima
from matplotlib import pyplot as plt


class TestGEVMLE(unittest.TestCase):
    def setUp(self) -> None:
        self.series = pd.Series(range(1000))
        self.block_maxima = BlockMaxima(
            Dataset(self.series),
            number_of_datapoints_per_block=100
        )
        self.gevmle = GEVMLE(self.block_maxima)

    def test_estimate(self):
        estimates = self.gevmle.estimate()
        self.assertAlmostEqual(
            -0.4647220177842265,
            estimates[0].estimate
        )
        self.assertAlmostEqual(
            -1.230942675038333,
            estimates[0].ci_lower
        )
        self.assertAlmostEqual(
            0.3014986394698801,
            estimates[0].ci_upper
        )
        self.assertAlmostEqual(
            473.51719544433735,
            estimates[1].estimate
        )
        self.assertAlmostEqual(
            246.21093397263186,
            estimates[1].ci_lower
        )
        self.assertAlmostEqual(
            700.8234569160428,
            estimates[1].ci_upper
        )
        self.assertAlmostEqual(
            305.7608338850753,
            estimates[2].estimate
        )
        self.assertAlmostEqual(
            111.82303722965494,
            estimates[2].ci_lower
        )
        self.assertAlmostEqual(
            499.6986305404956,
            estimates[2].ci_upper
        )

    def test_runtime_error(self):
        fig = plt.figure(figsize=(8, 6))
        ax = plt.gca()

        out_file = io.BytesIO()
        with self.assertRaises(RuntimeError):
            self.gevmle.plot_qq_gev(ax)

    def test_plot_qq_gev(self):
        self.gevmle.estimate()

        fig = plt.figure(figsize=(8, 6))
        ax = plt.gca()

        out_file = io.BytesIO()
        self.gevmle.plot_qq_gev(ax)
        fig.savefig(out_file, format='raw')
        out_file.seek(0)
        hashed = hashlib.md5(out_file.read()).digest()

        self.assertEqual(
            b'\xe8\xb7\xb5)\x18"$\xa0\x0em\xcb\x0e\xf8\x97\x14\xf5',
            hashed
        )
