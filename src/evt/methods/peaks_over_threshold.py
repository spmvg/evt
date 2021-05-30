from numbers import Real

import matplotlib.pyplot as plt
from evt import utils
from evt.dataset import Dataset
from scipy.stats import expon


class PeaksOverThreshold:
    """
    The peaks over threshold method is one of the two fundamental approaches in extreme value theory.
    The peaks of the ``Dataset`` ``dataset`` are determined by applying a threshold ``threshold`` :math:`\geq 0`
    and the resulting peaks are stored in ``self.series_tail``.
    """
    def __init__(
            self,
            dataset: Dataset,
            threshold: Real
    ):
        if threshold < 0:
            raise ValueError('The threshold must be positive. Consider shifting the data.')
        self.threshold = threshold

        self.dataset = dataset
        self.series_tail = dataset.series[dataset.series > threshold].copy()

    def plot_tail(
            self,
            ax: plt.Axes
    ):
        """
        Plot the peaks over threshold against the index of the original data.
        The original dataset is shown for comparison.
        """
        ax.plot(
            self.dataset.series[
                self.dataset.series.index.isin(self.series_tail.index)
            ],
            'xr',
            label='Tail',
            zorder=101
        )
        ax.plot(
            self.dataset.series,
            '-k',
            alpha=.3,
            label='Dataset',
            zorder=100
        )
        ax.axhline(
            y=self.threshold,
            linestyle='--',
            alpha=.8,
            color='k',
            label='Threshold',
            zorder=102
        )
        ax.set_xlabel(self.dataset.series.index.name or '')
        ax.set_ylabel(self.dataset.series.name or '')
        ax.grid()
        ax.set_title('Peaks over threshold')
        ax.legend(loc='lower right').set_zorder(200)

    def plot_qq_exponential(
            self,
            ax: plt.Axes
    ):
        """
        Quantile-quantile plot of the empirical survival function of the peaks over threshold against a fitted
        exponential distribution.
        """
        empirical_survival = 1 - utils.empirical_cdf(self.series_tail)
        loc, scale = expon.fit(self.series_tail)
        survival_function = expon.sf(empirical_survival.index, loc=loc, scale=scale)
        ax.loglog(
            survival_function,
            empirical_survival,
            'xk',
            alpha=.8
        )
        ax.plot(
            survival_function,
            survival_function,
            'r--',
            alpha=.8,
            label='Diagonal'
        )
        ax.invert_xaxis()
        ax.invert_yaxis()
        ax.set_xlabel('Exponential survival function')
        ax.set_ylabel('Empirical survival function')
        ax.legend(loc='upper left')
        ax.set_title('Q–Q plot against exponential distribution')

    def plot_zipf(
            self,
            ax: plt.Axes
    ):
        """
        Log-log plot of the empirical survival function. The :math:`x`-axis corresponds to the values of the original
        dataset.
        """
        empirical_survival = 1 - utils.empirical_cdf(self.series_tail)

        ax.loglog(
            empirical_survival,
            'xk',
            label='Empirical survival',
            alpha=.8
        )
        ax.set_xlabel(self.series_tail.name or '')
        ax.set_ylabel('Empirical survival function')
        ax.set_title('Zipf plot')
        ax.grid()
        ax.legend(loc='upper right')
