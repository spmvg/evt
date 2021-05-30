import hashlib
import io
import unittest

import numpy as np
import pandas as pd
from evt.dataset import Dataset
from evt.methods.peaks_over_threshold import PeaksOverThreshold
import matplotlib.pyplot as plt


class TestPeaksOverThreshold(unittest.TestCase):
    def test_tail_threshold(self):
        series = pd.Series(range(1, 101))
        dataset = Dataset(series)
        pot = PeaksOverThreshold(dataset, threshold=98)
        self.assertAlmostEqual(
            98,
            pot.threshold
        )
        self.assertTrue(np.all(np.isclose(
            [99, 100],
            pot.series_tail
        )))

    def test_raise_negative_threshold(self):
        series = pd.Series(range(10))
        dataset = Dataset(series)
        with self.assertRaises(ValueError):
            PeaksOverThreshold(dataset, threshold=-1)

    def test_plot_tail(self):
        series = pd.Series(range(1, 10))
        dataset = Dataset(series)
        pot = PeaksOverThreshold(dataset, threshold=0)

        fig = plt.figure(figsize=(8, 6))
        ax = plt.gca()

        out_file = io.BytesIO()
        pot.plot_tail(ax)
        fig.savefig(out_file, format='raw')
        out_file.seek(0)
        hashed = hashlib.md5(out_file.read()).digest()

        self.assertEqual(
            b'\x12\xb6tS\xc9m\xacL\xe8\xef\xdfQ\x02T\xac\xad',
            hashed
        )

    def test_plot_qq_exponential(self):
        series = pd.Series(range(1, 10))
        dataset = Dataset(series)
        pot = PeaksOverThreshold(dataset, threshold=0)

        fig = plt.figure(figsize=(8, 6))
        ax = plt.gca()

        out_file = io.BytesIO()
        pot.plot_qq_exponential(ax)
        fig.savefig(out_file, format='raw')
        out_file.seek(0)
        hashed = hashlib.md5(out_file.read()).digest()

        self.assertEqual(
            b"\xa4>\x1f\xe0\x91\xcf&\x83\xdd\x88\xb47\x90'~-",
            hashed
        )

    def test_plot_zipf(self):
        series = pd.Series(range(1, 10))
        dataset = Dataset(series)
        pot = PeaksOverThreshold(dataset, threshold=0)

        fig = plt.figure(figsize=(8, 6))
        ax = plt.gca()

        out_file = io.BytesIO()
        pot.plot_zipf(ax)
        fig.savefig(out_file, format='raw')
        out_file.seek(0)
        hashed = hashlib.md5(out_file.read()).digest()

        self.assertEqual(
            b'\x8c\xc1\xc5u\xe2!B\xe8\xfa\x19\x9d\xfb!\xa0\x0f\xce',
            hashed
        )
