from abc import abstractmethod
from dataclasses import dataclass
from numbers import Real
from typing import List


@dataclass
class Estimate:
    r"""
    Represents an estimate of a quantity. Includes lower- and upper-bounds of a confidence interval at a determined
    confidence level.

    Can be unpacked like a tuple:

    >>> estimate, ci_lower, ci_upper = Estimate(...)

    """
    estimate: Real
    """ Numerical estimate. """
    ci_lower: Real
    """ Lower bound of the confidence interval with confidence level ``confidence_level``. """
    ci_upper: Real
    """ Upper bound of the confidence interval with confidence level ``confidence_level``. """
    confidence_level: Real = .95
    """ Confidence level of the confidence intervals. Between 0 and 1 inclusive. """

    def __iter__(self):
        return iter([self.estimate, self.ci_lower, self.ci_upper])


class Estimator:
    """
    Abstract class representing an estimator for one or multiple parameters.
    """
    @abstractmethod
    def estimate(self, *args) -> List[Estimate]:
        """
        Returns a list of ``Estimate`` objects corresponding to the estimated values with confidence intervals.
        """
        pass  # pragma: no cover
