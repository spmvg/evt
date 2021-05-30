import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from evt import utils


class Dataset:
    """
    Represents a raw dataset, which forms the basis of extreme value analysis in this package.
    The core of the ``Dataset`` is a ``pd.Series`` ``series``, which is stored in the attribute ``self.series``.

    The following sanity-checks are performed on the data ``series``:

    * The data cannot contain ``nan``,
    * The data cannot contain non-finite values,
    * The data cannot contain duplicate indices.
    """
    def __init__(
            self,
            series: pd.Series,
    ):
        self.series = self._validate_raw_data(series.copy())

    @staticmethod
    def _validate_raw_data(
            series: pd.Series
    ):
        if series.isna().any():
            raise ValueError(
                f'There are {series.isna().sum()} NaNs in the series.'
            )
        if not np.isfinite(series.to_numpy()).all():
            raise ValueError(f'There are {np.sum(~np.isfinite(series.to_numpy()))} non-finite values in the dataset.')
        if series.index.duplicated().any():
            raise ValueError(f'There are {series.index.duplicated().sum()} duplicate indices in the series.')
        return series

    def plot_dataset(
        self,
        ax: plt.Axes
    ):
        """
        Plot the dataset against the original index.
        """
        ax.plot(
            self.series,
            '-k',
            alpha=.8
        )
        ax.set_xlabel(self.series.index.name or '')
        ax.set_ylabel(self.series.name or '')
        ax.grid()
        ax.set_title('Dataset')

    def plot_boxplot(
            self,
            ax: plt.Axes
    ):
        """
        Boxplot of the dataset.
        """
        ax2 = ax.twiny()
        ax.hist(
            self.series,
            bins=40,
            orientation='horizontal',
            color='k',
            alpha=.3
        )
        ax2.boxplot(
            self.series.values,
            sym='x',
            whis=[5, 95],
            labels=['']
        )
        ax.set_xlabel('Number of observations')
        ax.set_ylabel(self.series.name or '')
        ax.set_title('Boxplot with $(5, 25, 75, 95)$-quantiles')
        ax.grid()

    def plot_mean_excess(
            self,
            ax: plt.Axes
    ):
        """
        Plot the empirical mean excess (average excess of a threshold) as a function of the threshold.
        """
        ax.plot(
            utils.mean_excess(self.series),
            'kx',
            alpha=.8
        )
        ax.set_xlabel(self.series.name or '')
        ax.set_ylabel('Mean excess')
        ax.set_title('Empirical mean excess')
        ax.grid()

    def plot_maximum_to_sum(
            self,
            ax: plt.Axes,
            number_of_moments: int = 4
    ):
        """
        Cumulative absolute-maximum-to-absolute-sum plot of the dataset for ``number_of_moments`` moments against
        the original index.
        """
        absolute_series = self.series.abs()
        for moment in range(1, number_of_moments + 1):
            maximum_to_sum = (absolute_series ** moment).cummax() / (absolute_series ** moment).cumsum()
            ax.plot(maximum_to_sum, '-', label=f'Moment {moment}')
        ax.set_ylim(0, 1)
        ax.legend(loc='upper left')
        ax.grid()
        ax.set_xlabel(self.series.index.name or '')
        ax.set_ylabel('Ratio of abs. maximum to abs. sum')
        ax.set_title('Absolute maximum to absolute sum')
