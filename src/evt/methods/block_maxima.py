import pandas as pd
from evt.dataset import Dataset
import matplotlib.pyplot as plt


class BlockMaxima:
    """
    The block maxima method is one of the two fundamental approaches in extreme value theory. The ``Dataset``
    ``dataset`` is divided into blocks of size ``number_of_datapoints_per_block`` :math:`\geq 1`.
    For every block, the maximum is determined.
    """
    def __init__(
            self,
            dataset: Dataset,
            number_of_datapoints_per_block: int,
    ):
        if number_of_datapoints_per_block < 1:
            raise ValueError(f'Number of datapoints per block {number_of_datapoints_per_block} must be >= 1')

        self.dataset = dataset
        self.number_of_datapoints_per_block = number_of_datapoints_per_block
        self.block_maxima = self._block_maxima(dataset.series, number_of_datapoints_per_block)

    @staticmethod
    def _block_maxima(
            series: pd.Series,
            number_of_datapoints_per_block: int
    ) -> pd.Series:
        values_name, index_name = 'values', 'index'
        renamed_series = series.copy()
        renamed_series.name, renamed_series.index.name = values_name, index_name  # prevent collision

        maxima_series_frame = renamed_series.to_frame().reset_index()
        maxima_series_indices = maxima_series_frame.groupby(
            by=lambda row_number: row_number // number_of_datapoints_per_block
        )[values_name].idxmax()
        maxima_series = maxima_series_frame.iloc[maxima_series_indices].set_index(index_name)[values_name]

        maxima_series.name, maxima_series.index.name = series.name, series.index.name  # put back to prevent collision
        return maxima_series

    def plot_block_maxima(
            self,
            ax: plt.Axes
    ):
        """
        Plots the block maxima as stars against the original dataset. The blocks are indicated by vertical separators.
        """
        numbered_indices = self.dataset.series.index.to_series().reset_index(drop=True)
        block_separator_indices = numbered_indices[numbered_indices.index % self.number_of_datapoints_per_block == 0]
        block_separators = self.dataset.series.loc[block_separator_indices].index

        ax.plot(
            self.block_maxima,
            '*r',
            label='Block maxima',
            markersize=9,
            zorder=102
        )
        ax.plot(
            self.dataset.series,
            '-k',
            alpha=.7,
            label='Dataset',
            zorder=100
        )
        for i, block_separator in enumerate(block_separators):
            ax.axvline(
                x=block_separator,
                linestyle=':',
                alpha=.3,
                color='k',
                label=None if i else 'Block separator',
                zorder=101
            )
        ax.set_xlabel(self.dataset.series.index.name or '')
        ax.set_ylabel(self.dataset.series.name or '')
        ax.grid(axis='y')
        ax.set_title('Block maxima')
        ax.legend(loc='lower right').set_zorder(200)

    def plot_block_maxima_boxplot(
            self,
            ax: plt.Axes
    ):
        """
        Plots a boxplot of the block maxima on the left side. On the right side, a boxplot of the original dataset is
        shown for comparison.
        """
        ax.boxplot(
            [
                self.block_maxima.values,
                self.dataset.series.values
            ],
            sym='x',
            whis=[5, 95],
            labels=[
                'Block maxima',
                'Dataset'
            ]
        )
        ax.set_ylabel(self.dataset.series.name or '')
        ax.set_title('Block maxima\nBoxplot with $(5, 25, 75, 95)$-quantiles')
        ax.grid()
