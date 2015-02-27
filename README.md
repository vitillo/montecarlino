Montecarlino
============

Parellel implementation of bootstrap confidence intervals and permutation tests.

## Installation
```bash
pip install montecarlino
```

## Examples

Let's compute the 95% confidence interval for the median of a set of values:

```python
from montecarlino import bootstrap_interval

x1 = np.array([0.80, 0.83, 1.89, 1.04, 1.45, 1.38, 1.91, 1.64, 0.73, 1.46])
bootstrap_interval(np.median, x1)
```

Now let's compare the medians of two samples, i.e. what's the p-value
of the difference of the medians assuming there is no difference?

```python
from montecarlino import grouped_permutation_test

def _median_difference(x, y):
    return np.median(y[x == 0]) - np.median(y[x == 1])

x1 = np.array([0.80, 0.83, 1.89, 1.04, 1.45, 1.38, 1.91, 1.64, 0.73, 1.46])
x2 = np.array([1.15, 0.88, 0.90, 0.74, 1.21])
grouped_permutation_test(_median_difference, [x1, x2])
```



