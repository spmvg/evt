from typing import List

import matplotlib.pyplot as plt
import numpy as np

from evt import utils
from evt.estimators.estimator_abc import Estimator, Estimate
from evt.methods.peaks_over_threshold import PeaksOverThreshold


class Moment(Estimator):
    """
    Moment estimator for the tail index in the peaks over threshold approach. [1]
    Also known as the Dekkers-Einmahl-De Haan estimator. [2]

    The number of order statistics ``number_of_order_statistics`` to use in the estimate must be specified.
    Contains a plotting routine for the moment estimator against the number of order statistics.
    Confidence intervals are based on the asymptotic behaviour of the variance of the estimate.
    Bias is not taken into account.

    1. De Haan, Laurens, and Ana Ferreira. *Extreme value theory: an introduction.*
       Springer Science & Business Media, 2007.
    2. Dekkers, Arnold LM, John HJ Einmahl, and Laurens De Haan.
       "A moment estimator for the index of an extreme-value distribution." *The Annals of Statistics* (1989): 1833-1855.
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
        Returns the moment estimate for the tail index :math:`\gamma \in \mathbb{R}`, calculated with ``number_of_order_statistics``
        order statistics. [1]

        Confidence intervals are based on the asymptotic behaviour of the variance of the estimate. [1]
        Bias is not taken into account.

        :param number_of_order_statistics: number of order statistics to use in the moment estimator. Must be bigger
            than zero and smaller than the number of peaks in the peaks over threshold approach.
        :return: ``Estimate`` of the tail index.

        1. De Haan, Laurens, and Ana Ferreira. *Extreme value theory: an introduction.*
           Springer Science & Business Media, 2007.
        """
        if number_of_order_statistics == 0:
            raise ValueError('number_of_order_statistics cannot be 0')
        if number_of_order_statistics >= len(self.order_statistics):
            raise ValueError(f'number_of_order_statistics {number_of_order_statistics} cannot exceed the '
                             f'number of datapoints in the tail {len(self.order_statistics)}')

        log_of_kth = np.log(self.order_statistics[number_of_order_statistics])
        hill_part = np.sum(
            np.log(
                self.order_statistics[:number_of_order_statistics].values
            ) - log_of_kth
        ) / number_of_order_statistics
        squared_part = np.sum(
            np.square(np.log(
                self.order_statistics[:number_of_order_statistics].values
            ) - log_of_kth)
        ) / number_of_order_statistics
        tail_index = hill_part + 1 - .5 / (1 - hill_part ** 2 / squared_part)

        variance = tail_index ** 2 + 1 if tail_index >= 0 else (
            (
                (1 - tail_index) ** 2
                * (1 - 2 * tail_index)
                * (1 - tail_index + 6 * tail_index ** 2)
            )
            /
            (
                (1 - 3 * tail_index)
                * (1 - 4 * tail_index)
            )
        )
        standard_deviation = np.sqrt(variance) / np.sqrt(number_of_order_statistics)
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
        Plots the moment estimate against the number of order statistics.

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
        ax.set_title('Moment estimator')
        ax.legend(loc='upper left')
