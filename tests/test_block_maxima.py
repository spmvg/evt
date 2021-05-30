import hashlib
import io
import unittest

import numpy as np
import pandas as pd
from evt.dataset import Dataset
from evt.methods.block_maxima import BlockMaxima
import matplotlib.pyplot as plt


class TestPeaksOverThreshold(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = Dataset(pd.Series(
            range(5),
            index=range(10, 15),
            name='dataset'
        ))
        self.dataset.series.index.name = 'dataset index'
        self.uniform_dataset = Dataset(pd.Series(
            np.ones(5),
            index=range(10, 15)
        ))

    def test_zero_number_of_datapoints_per_block(self):
        with self.assertRaises(ValueError):
            BlockMaxima(self.dataset, 0)

    def test_negative_number_of_datapoints_per_block(self):
        with self.assertRaises(ValueError):
            BlockMaxima(self.dataset, -1)

    def test_dataset_name(self):
        block_maxima = BlockMaxima(self.dataset, 1)
        self.assertIsInstance(block_maxima.block_maxima, pd.Series)
        self.assertEqual('dataset', block_maxima.block_maxima.name)
        self.assertEqual('dataset index', block_maxima.block_maxima.index.name)

    def test_one_datapoint_per_block(self):
        block_maxima = BlockMaxima(self.dataset, 1)
        self.assertTrue(np.all(np.isclose(
            [10, 11, 12, 13, 14],
            block_maxima.block_maxima.index
        )))
        self.assertTrue(np.all(np.isclose(
            [0, 1, 2, 3, 4],
            block_maxima.block_maxima.values
        )))

    def test_one_datapoint_per_block_uniform(self):
        block_maxima = BlockMaxima(self.uniform_dataset, 1)
        self.assertTrue(np.all(np.isclose(
            [10, 11, 12, 13, 14],
            block_maxima.block_maxima.index
        )))
        self.assertTrue(np.all(np.isclose(
            [1, 1, 1, 1, 1],
            block_maxima.block_maxima.values
        )))

    def test_two_datapoints_per_block(self):
        block_maxima = BlockMaxima(self.dataset, 2)
        self.assertTrue(np.all(np.isclose(
            [11, 13, 14],
            block_maxima.block_maxima.index
        )))
        self.assertTrue(np.all(np.isclose(
            [1, 3, 4],
            block_maxima.block_maxima.values
        )))

    def test_two_datapoints_per_block_uniform(self):
        block_maxima = BlockMaxima(self.uniform_dataset, 2)
        self.assertTrue(np.all(np.isclose(
            [10, 12, 14],
            block_maxima.block_maxima.index
        )))
        self.assertTrue(np.all(np.isclose(
            [1, 1, 1],
            block_maxima.block_maxima.values
        )))

    def test_plot_block_maxima(self):
        block_maxima = BlockMaxima(self.dataset, 1)

        fig = plt.figure(figsize=(8, 6))
        ax = plt.gca()

        out_file = io.BytesIO()
        block_maxima.plot_block_maxima(ax)
        fig.savefig(out_file, format='raw')
        out_file.seek(0)
        hashed = hashlib.md5(out_file.read()).digest()

        self.assertEqual(
            b'\xeamE\x8e\xac\xde\xc9\xae\xaf8\x8df|\x1eU\x17',
            hashed
        )

    def test_plot_block_maxima_boxplot(self):
        block_maxima = BlockMaxima(self.dataset, 1)

        fig = plt.figure(figsize=(8, 6))
        ax = plt.gca()

        out_file = io.BytesIO()
        block_maxima.plot_block_maxima_boxplot(ax)
        fig.savefig(out_file, format='raw')
        out_file.seek(0)
        hashed = hashlib.md5(out_file.read()).digest()

        self.assertEqual(
            b'H\xd7rE\x82\x01x-\x12\xdf%\xd81t\x9c\x0b',
            hashed
        )
