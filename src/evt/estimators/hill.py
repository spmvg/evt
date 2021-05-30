from typing import List

import matplotlib.pyplot as plt
import numpy as np

from evt import utils
from evt.estimators.estimator_abc import Estimator, Estimate
from evt.methods.peaks_over_threshold import PeaksOverThreshold


class Hill(Estimator):
    """
    Hill estimator for the tail index in the peaks over threshold approach. [1]

    The number of order statistics ``number_of_order_statistics`` to use in the estimate must be specified.
    Contains a plotting routine for the Hill estimator against the number of order statistics.
    Confidence intervals are based on the asymptotic behaviour of the variance of the estimate. [2]
    Bias is not taken into account.

    1. Hill, Bruce M. "A simple general approach to inference about the tail of a distribution."
       *The annals of statistics* (1975): 1163-1174.
    2. De Haan, Laurens, and Ana Ferreira. *Extreme value theory: an introduction.*
       Springer Science & Business Media, 2007.
    """
    def __init__(
            self,
            peaks_over_threshold: PeaksOverThreshold
    ):
        super().__init__()

        self.peaks_over_threshold = peaks_over_threshold
        self.order_statistics = utils.order_statistics(peaks_over_threshold.series_tail)

    def estimate(
            self,
            number_of_order_statistics: int
    ) -> List[Estimate]:
        r"""
        Returns the Hill estimate for the tail index :math:`\gamma > 0`, calculated with ``number_of_order_statistics``
        order statistics. [1]

        Confidence intervals are based on the asymptotic behaviour of the variance of the estimate. [2]
        Bias is not taken into account.

        :param number_of_order_statistics: number of order statistics to use in the Hill estimator. Must be bigger
            than zero and smaller than the number of peaks in the peaks over threshold approach.
        :return: ``Estimate`` of the tail index.

        1. Hill, Bruce M. "A simple general approach to inference about the tail of a distribution."
           *The annals of statistics* (1975): 1163-1174.
        2. De Haan, Laurens, and Ana Ferreira. *Extreme value theory: an introduction.*
           Springer Science & Business Media, 2007.
        """
        if number_of_order_statistics == 0:
            raise ValueError('number_of_order_statistics cannot be 0')
        if number_of_order_statistics >= len(self.order_statistics):
            raise ValueError(f'number_of_order_statistics {number_of_order_statistics} cannot exceed the '
                             f'number of datapoints in the tail {len(self.order_statistics)}')

        log_of_kth = np.log(self.order_statistics[number_of_order_statistics])
        tail_index = np.sum(
            np.log(
                self.order_statistics[:number_of_order_statistics].values
            ) - log_of_kth
        ) / number_of_order_statistics
        standard_deviation = tail_index / np.sqrt(number_of_order_statistics)
        std_factor = utils.confidence_interval_to_std(Estimate.confidence_level)
        return [Estimate(
            tail_index,
            tail_index - std_factor * standard_deviation,
            tail_index + std_factor * standard_deviation,
        )]

    def plot(
            self,
            ax: plt.Axes,
            max_number_of_order_statistics: int = None
    ):
        """
        Plots the Hill estimate against the number of order statistics.

        The maximum number of order statistics to use in the plot can be specified by
        ``max_number_of_order_statistics``.
        If ``None``, the maximum number of order statistics will be the number of peaks in the peaks over threshold
        approach, minus one.
        """
        if max_number_of_order_statistics is None:
            max_number_of_order_statistics = len(self.order_statistics) - 1

        x_axis = self.order_statistics.index.values[1:max_number_of_order_statistics]
        estimates, ci_lowers, ci_uppers = map(np.array, zip(*[
            self.estimate(number_of_order_statistics)[0]
            for number_of_order_statistics in x_axis
        ]))
        ax.plot(
            x_axis,
            estimates,
            'k',
            alpha=.8,
            label='Estimated tail index'
        )
        ax.plot(
            x_axis,
            ci_uppers,
            'k:',
            alpha=.6,
            label='STD 95% CI'
        )
        ax.plot(
            x_axis,
            ci_lowers,
            'k:',
            alpha=.6
        )
        ax.grid()
        ax.set_xlabel('Number of order statistics')
        ax.set_ylabel('Estimated tail index')
        ax.set_title('Hill estimator')
        ax.legend(loc='upper left')
