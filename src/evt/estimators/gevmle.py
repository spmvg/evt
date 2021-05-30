from typing import List

import matplotlib.pyplot as plt
import numpy as np
from evt import utils
from evt._compiled_expressions.compiled_expressions import gevmle_fisher_information
from evt.estimators.estimator_abc import Estimator, Estimate
from evt.methods.block_maxima import BlockMaxima
from scipy.stats import genextreme


class GEVMLE(Estimator):
    r"""
    Maximum likelihood estimator for the generalized extreme value distribution in the block maxima approach with
    distribution

    .. math::
        \exp \bigg[
            -(1+ \gamma (x - \mu) / \sigma) ^{-1/\gamma}
        \bigg]

    where

    * ``self.tail_index`` corresponds to the tail index :math:`\gamma`,
    * ``self.loc`` corresponds to the location parameter :math:`\mu`,
    * ``self.scale`` corresponds to the scale parameter :math:`\sigma`.

    Confidence intervals are estimated using the observed Fisher information.
    """
    def __init__(
            self,
            block_maxima: BlockMaxima
    ):
        super().__init__()

        self.block_maxima = block_maxima
        self.tail_index, self.loc, self.scale = None, None, None

    def estimate(self) -> List[Estimate]:
        """
        Returns maximum likelihood estimates including confidence intervals for the tail index, location parameter
        and scale of the generalized extreme value distribution.

        Estimates for the confidence intervals are based on asymptotic behaviour of the observed Fisher information
        for the generalized extreme value distribution.

        The returned estimate might be only locally optimal or fail altogether.
        Moreover, if the confidence intervals are unable to be determined numerically, the ``.ci_lower`` and
        ``.ci_upper`` of the estimate will be ``nan``.

        :return: maximum likelihood ``Estimate`` including confidence intervals for the tail index,
            location parameter and scale of the generalized extreme value distribution.
        """
        tail_index, self.loc, self.scale = genextreme.fit(self.block_maxima.block_maxima)
        self.tail_index = -tail_index  # scipy uses opposite sign for tail index

        std_tail_index, std_loc, std_scale = np.sqrt(np.diag(np.linalg.inv(gevmle_fisher_information(
            self.block_maxima.block_maxima.to_numpy(),
            self.tail_index,
            self.loc,
            self.scale
        )))) / np.sqrt(len(self.block_maxima.block_maxima))
        std_factor = utils.confidence_interval_to_std(Estimate.confidence_level)
        return [
            Estimate(
                self.tail_index,
                self.tail_index - std_factor * std_tail_index,
                self.tail_index + std_factor * std_tail_index,
            ),
            Estimate(
                self.loc,
                self.loc - std_factor * std_loc,
                self.loc + std_factor * std_loc,
            ),
            Estimate(
                self.scale,
                self.scale - std_factor * std_scale,
                self.scale + std_factor * std_scale,
            ),
        ]

    def plot_qq_gev(
            self,
            ax: plt.Axes
    ):
        """
        Quantile-quantile plot of the empirical survival function of the block maxima against the fitted generalized
        extreme value distribution.
        The ``.estimate`` method must be called before this function.
        """
        if self.tail_index is None:
            raise RuntimeError('The .estimate method must be called before plotting is possible.')

        empirical_survival = 1 - utils.empirical_cdf(self.block_maxima.block_maxima)
        survival_function = genextreme.sf(
            empirical_survival.index,
            -self.tail_index,
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
        ax.set_xlabel('GEV survival function')
        ax.set_ylabel('Empirical survival function')
        ax.legend(loc='upper left')
        ax.set_title('Q–Q plot against generalized extreme value distribution')
