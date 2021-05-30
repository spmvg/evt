import hashlib
import io
import unittest

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from evt.dataset import Dataset


class TestDataset(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_series(self):
        series = pd.Series(data=[1, 2, 3])
        dataset = Dataset(series)
        self.assertTrue(np.all(np.isclose(
            [1, 2, 3],
            dataset.series
        )))

    def test_nans(self):
        series = pd.Series(data=[1, 2, np.nan, 3])
        with self.assertRaises(ValueError):
            Dataset(series)

    def test_inf(self):
        series = pd.Series(data=[1, 2, np.inf, 3])
        with self.assertRaises(ValueError):
            Dataset(series)

    def test_deepcopy(self):
        series = pd.Series(data=[1, 2, 3])
        dataset = Dataset(series)
        self.assertNotEqual(id(series), id(dataset.series))

    def test_duplicated(self):
        series = pd.Series(data=[1, 2, 3, 4], index=[1, 1, 1, 2])
        with self.assertRaises(ValueError):
            Dataset(series)

    def test_plot_dataset(self):
        series = pd.Series(data=[1, 2, 3])
        dataset = Dataset(series)
        fig = plt.figure(figsize=(8, 6))
        ax = plt.gca()

        out_file = io.BytesIO()
        dataset.plot_dataset(ax)
        fig.savefig(out_file, format='raw')
        out_file.seek(0)
        hashed = hashlib.md5(out_file.read()).digest()

        self.assertEqual(
            b'\xc5\xb5\x90\r\x88\xe3\x96\xe1\xa2\x1c\x9eg\xcf\xbc\xd2\xd9',
            hashed
        )

    def test_plot_boxplot(self):
        series = pd.Series(data=[1, 2, 3])
        dataset = Dataset(series)
        fig = plt.figure(figsize=(8, 6))
        ax = plt.gca()

        out_file = io.BytesIO()
        dataset.plot_boxplot(ax)
        fig.savefig(out_file, format='raw')
        out_file.seek(0)
        hashed = hashlib.md5(out_file.read()).digest()

        self.assertEqual(
            b',;\x01\xe9\x95\xb8\xb8\xeb\xc2V\xb4\n\xf3\xc5\x9f\x90',
            hashed
        )

    def test_plot_mean_excess(self):
        series = pd.Series(data=[1, 2, 3])
        dataset = Dataset(series)
        fig = plt.figure(figsize=(8, 6))
        ax = plt.gca()

        out_file = io.BytesIO()
        dataset.plot_mean_excess(ax)
        fig.savefig(out_file, format='raw')
        out_file.seek(0)
        hashed = hashlib.md5(out_file.read()).digest()

        self.assertEqual(
            b'\xe2\x11\x0bS\xc5\x11^\xb2\x84\x0f\x87\x9d\x9c\xfc\xfb\x89',
            hashed
        )

    def test_plot_maximum_to_sum(self):
        series = pd.Series(data=[1, 2, 3])
        dataset = Dataset(series)
        fig = plt.figure(figsize=(8, 6))
        ax = plt.gca()

        out_file = io.BytesIO()
        dataset.plot_maximum_to_sum(ax)
        fig.savefig(out_file, format='raw')
        out_file.seek(0)
        hashed = hashlib.md5(out_file.read()).digest()

        self.assertEqual(
            b'\xec\xac\x9f\x1cl\xbdB\xf5d\xf2\xb2;\x9a\x05\xc7\x99',
            hashed
        )
