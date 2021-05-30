from typing import List

import numpy as np
import matplotlib.pyplot as plt
from evt import utils
from evt.estimators.estimator_abc import Estimator, Estimate
from evt.methods.peaks_over_threshold import PeaksOverThreshold
from scipy.stats import genpareto


class GPDMLE(Estimator):
    r"""
    Maximum likelihood estimator for the generalized Pareto distribution in the peaks over threshold approach with
    distribution

    .. math::
        1 - (1+ \gamma (x - \mu) / \sigma) ^{-1/\gamma}

    where

    * ``self.tail_index`` corresponds to the tail index :math:`\gamma`,
    * ``self.loc`` corresponds to the location parameter :math:`\mu`,
    * ``self.scale`` corresponds to the scale parameter :math:`\sigma`.

    The tail index :math:`\gamma` and scale parameter :math:`\sigma` can be estimated.
    The location parameter :math:`\mu` is taken from the peaks over threshold method ``peaks_over_threshold``.
    """
    def __init__(
            self,
            peaks_over_threshold: PeaksOverThreshold
    ):
        super().__init__()

        self.peaks_over_threshold = peaks_over_threshold
        self.tail_index, self.loc, self.scale = None, self.peaks_over_threshold.threshold, None

    def estimate(self) -> List[Estimate]:
        r"""
        Returns maximum likelihood estimates including confidence intervals for the tail index
        and scale of the generalized Pareto distribution.

        The estimator behaves irregularly for :math:`\gamma \leq -\frac{1}{2}`. [1]
        Confidence intervals are based on the asymptotic behaviour of the variance of the estimate. [1]
        Bias is not taken into account.
        The returned estimate might be only locally optimal or fail altogether.

        :return: maximum likelihood ``Estimate`` including confidence intervals for the tail index
            and scale of the generalized extreme value distribution.

        1. De Haan, Laurens, and Ana Ferreira. *Extreme value theory: an introduction.*
           Springer Science & Business Media, 2007.
        """
        self.tail_index, _, self.scale = genpareto.fit(
            self.peaks_over_threshold.series_tail,
            floc=self.loc
        )

        std_factor = utils.confidence_interval_to_std(Estimate.confidence_level)
        tail_index_std = (1 + np.abs(self.tail_index)) / np.sqrt(len(self.peaks_over_threshold.series_tail))
        scale_std = (
            np.sqrt(
                1 + np.square(1 + self.tail_index)
            ) / np.sqrt(
                len(self.peaks_over_threshold.series_tail)
            )
        )

        return [
            Estimate(
                self.tail_index,
                self.tail_index - std_factor * tail_index_std,
                self.tail_index + std_factor * tail_index_std,
            ),
            Estimate(
                self.scale,
                self.scale - std_factor * scale_std,
                self.scale + std_factor * scale_std,
            ),
        ]

    def plot_qq_gpd(
            self,
            ax: plt.Axes
    ):
        """
        Quantile-quantile plot of the empirical survival function of the peaks against the fitted generalized
        Pareto distribution.
        The ``.estimate`` method must be called before this function.
        """
        if self.tail_index is None:
            raise RuntimeError('The .estimate method must be called before plotting is possible.')

        empirical_survival = 1 - utils.empirical_cdf(self.peaks_over_threshold.series_tail)
        survival_function = genpareto.sf(
            empirical_survival.index,
            self.tail_index,
            loc=self.loc,
            scale=self.scale
        )
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
        ax.set_xlabel('GPD survival function')
        ax.set_ylabel('Empirical survival function')
        ax.legend(loc='upper left')
        ax.set_title('Q–Q plot against generalized Pareto distribution')
