Estimators and analysis for extreme value theory (EVT).
The package is structured as follows.
Example notebooks are provided as links.

* The `Dataset` [object](https://github.com/spmvg/evt/blob/master/tutorial/0_the_dataset_object.ipynb) performs sanity-checks and contains plotting routines.
  * Generic plots of the dataset.
  * Maximum-to-sum plot.
  * Mean excess function.
* The [peaks over threshold](https://github.com/spmvg/evt/blob/master/tutorial/1_peaks_over_threshold.ipynb) method.
  * Plot the tail.
  * QQ-plot against exponential.
  * Zipf-plot.
* The [block maxima](https://github.com/spmvg/evt/blob/master/tutorial/2_block_maxima.ipynb) method.
  * Plot block maxima against the dataset.
* Estimators: calculate estimates and [confidence intervals](https://github.com/spmvg/evt/blob/master/tutorial/3_hill_estimator_and_the_estimate_object.ipynb). Plotting routines for analysis.
   * [Hill estimator](https://github.com/spmvg/evt/blob/master/tutorial/3_hill_estimator_and_the_estimate_object.ipynb).
     Plot against order statistics.
   * [Moment estimator](https://github.com/spmvg/evt/blob/master/tutorial/4_moment_estimator.ipynb) (Dekkers-Einmahl-De Haan).
     Plot against order statistics.
   * [Maximum likelihood for the generalized Pareto distribution](https://github.com/spmvg/evt/blob/master/tutorial/5_maximum_likelihood_generalized_pareto.ipynb).
     Plot fit quality.
   * [Maximum likelihood for the generalized extreme value distribution](https://github.com/spmvg/evt/blob/master/tutorial/6_maximum_likelihood_generalized_extreme_value.ipynb).
     Plot fit quality.

Documentation
-------------
Documentation is provided [here](https://evt.readthedocs.io/en/latest/).
Example notebooks are provided [here](https://github.com/spmvg/evt/blob/master/tutorial/).

Installation
------------
Releases are made available on PyPi.
The recommended installation method is via `pip`:

```python
pip install evt
```

For a development setup, the requirements are in `dev-requirements.txt`.
Subsequently, the repo can be locally `pip`-installed.