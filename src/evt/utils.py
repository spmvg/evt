from numbers import Real
from pathlib import Path

import pandas as pd
from scipy.stats import norm


def empirical_cdf(
        series: pd.Series
) -> pd.Series:
    """
    Calculates the empirical distribution function given a ``pd.Series`` ``series``. The resulting CDF will have values
    between 0 and 1, exclusive. The index is ignored.

    :param series: ``pd.Series`` of which to calculate the empirical distribution function.
    :return: ``pd.Series``, the empirical distribution function. The index corresponds to the values of the ``series``.
    """
    cdf = series.sort_values().reset_index(drop=True)

    cdf.index += 1  # 0 < lowest <= highest < 1
    size = cdf.shape[0] + 1
    cdf.index = cdf.index / size
    cdf.drop_duplicates(
        keep='first',
        inplace=True
    )
    return pd.Series(cdf.index.values, index=cdf)


def mean_excess(series: pd.Series) -> pd.Series:
    """
    Calculates the mean excess (average excess of a threshold) for values in the ``pd.Series`` ``series``.
    For every value in ``series``, the mean excess of the rest of the distribution will be calculated.

    :param series: ``pd.Series`` of which the mean excess will be calculated. The index is ignored.
    :return: ``pd.Series`` corresponding to the mean excesses. The index corresponds to the threshold.
    """
    sorted_values = series.sort_values(ascending=False).reset_index(drop=True)

    mean_excess_duplicates = (
        sorted_values.cumsum().shift(1) - sorted_values.index * sorted_values
    ) / sorted_values.index
    mean_excess_duplicates.index = sorted_values.values
    mean_excess = mean_excess_duplicates[~mean_excess_duplicates.index.duplicated()]

    return mean_excess.dropna().sort_index()


def scientific_notation(
        number: Real,
        number_of_significant_digits: int = 1
) -> str:
    """
    Returns the scientific notation of ``number`` in ``number_of_significant_digits`` significant digits.
    """
    significand, exponent = f'{number:.{number_of_significant_digits}E}'.split('E')
    exponent_int = int(exponent)
    exponent_text = rf'\cdot 10^{{{exponent_int}}}' if exponent_int else ''
    return rf'${significand} {exponent_text}$'


def order_statistics(
        series: pd.Series
) -> pd.Series:
    """
    Calculates the order statistics (sorted maxima) of a ``pd.Series``.

    :param series: ``pd.Series`` of which to calculate the order statistics. The index is ignored.
    :return: ``pd.Series``, where the index corresponds to the ascending index of the order statistic, where 0 is the
        biggest.
    """
    return series.sort_values(
        ascending=False
    ).reset_index(
        drop=True
    )


def repo_root() -> Path:
    """ Returns the root path of the repository. """
    return Path(__file__).parent.parent.parent  # pragma: no cover


def confidence_interval_to_std(
        confidence: Real
) -> Real:
    """
    Returns the number of standard deviations of a standard normal distribution, corresponding to a confidence
    interval with confidence level ``confidence``.

    :param confidence: confidence interval, between 0 (inclusive) and 1 (exclusive).
    :return: ``Real``
    """
    return norm.ppf(.5 + confidence / 2)
