Quantile Regression Benchmark
=============================
|Build Status| |Python 3.6+|

Benchopt is a package to simplify and make more transparent and
reproducible the comparisons of optimization algorithms.
This benchmark is dedicated to the the L1-regularized quantile regression problem:


$$\\min_{\\beta, \\beta_0} \\frac{1}{n} \\sum_{i=1}^{n} \\text{pinball}(y_i, x_i^\\top \\beta + \\beta_0) + \\lambda \\lveft w \\rvert_1$$

where

$$\\text{pinball}(y, \\hat{y}) = \\alpha \\max(y - \\hat{y}, 0) + (1 - \\alpha) \\max(\\hat{y} - y, 0)$$

where $n$ (or ``n_samples``) stands for the number of samples, $p$ (or ``n_features``) stands for the number of features and

$$X = [x_1^\\top, \\dots, x_n^\\top]^\\top \\in \\mathbb{R}^{n \\times p}$$


Install
--------

This benchmark can be run using the following commands:

.. code-block::

   $ pip install -U benchopt
   $ git clone https://github.com/benchopt/benchmark_quantile_regression
   $ benchopt run benchmark_quantile_regression

Apart from the problem, options can be passed to ``benchopt run``, to restrict the benchmarks to some solvers or datasets, e.g.:

.. code-block::

	$ benchopt run benchmark_quantile_regression -s scipy -d simulated --max-runs 10 --n-repetitions 10


Use ``benchopt run -h`` for more details about these options, or visit https://benchopt.github.io/api.html.

.. |Build Status| image:: https://github.com/benchopt/benchmark_quantile_regression/workflows/Tests/badge.svg
   :target: https://github.com/benchopt/benchmark_quantile_regression/actions
.. |Python 3.6+| image:: https://img.shields.io/badge/python-3.6%2B-blue
   :target: https://www.python.org/downloads/release/python-360/
