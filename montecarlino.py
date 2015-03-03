import numpy as np
import multiprocessing as mp

from functools import partial
from scipy.stats import norm

def _bootstrap_interval(statistic, data, num_samples):
    np.random.seed()
    n = len(data)
    idx = np.random.randint(0, n, (num_samples, n))
    samples = data[idx]
    return [statistic(sample) for sample in samples]

def _jackknife_indexes(data):
    base = np.arange(0, len(data))
    return (np.delete(base, i) for i in base)

def bootstrap_interval(statistic, data, num_samples=100000, alpha=0.05):
    """
    Returns a confidence interval for the specified statistic. Uses the BCA
    algorithm, see An Introduction to the Bootstrap. Chapman & Hall 1993.
    """

    cores = mp.cpu_count()
    pool = mp.Pool(processes=cores)
    job_samples = num_samples/cores + 1

    stat = [pool.apply_async(_bootstrap_interval, args=(statistic, data, job_samples)) for i in range(cores)]
    stat = [s.get() for s in stat]
    stat = np.sort([item for list in stat for item in list])
    num_samples = len(stat)

    pool.close()
    pool.join()

    # Compute the bias correction, i.e. the discrepancy between the median of the
    # bootstrap distribution and the original median of the sample, in normal units.
    z0 = norm.ppf(np.sum(stat < statistic(data))/float(num_samples))

    # Jackknife statistics
    jindexes = _jackknife_indexes(data)
    jstat = [statistic(data[index]) for index in jindexes]
    jmean = np.mean(jstat)

    # Compute the acceleration value, i.e. the rate of change of the standard error
    # of jstat with respect to the true parameter value.
    a = np.sum((jmean - jstat)**3)/(6.0*np.sum((jmean - jstat)**2)**1.5)

    alphas = np.array((alpha/2, 1 - alpha/2))
    zs = z0 + norm.ppf(alphas).reshape(alphas.shape+(1,)*z0.ndim)
    avals = norm.cdf(z0 + zs/(1-a*zs))
    return stat[(avals*num_samples).astype(int)]

def _permutation_test(statistic, x, y, num_samples, alternative, grouped=False):
    np.random.seed()
    distribution = []
    observed_stat = statistic(x, y)

    for i in range(num_samples):
        perm_x = np.random.permutation(x)
        perm_stat = statistic(perm_x, y)
        distribution.append(perm_stat)

    if alternative == "two sided":
        return np.sum(np.abs(distribution) >= np.abs(observed_stat))
    elif alternative == "greater":
        return np.sum(distribution >= observed_stat)
    else:
        return np.sum(distribution <= observed_stat)

def _grouped_statistic_wrapper(statistic, ngroups, x, y):
    groups = []

    for i in range(ngroups + 1):
        groups.append(y[x == i])

    return statistic(*groups)

def permutation_test(statistic, x, y, num_samples=100000, alternative="two sided", grouped=False):
    """Tests the independence of the statistic for the variables X and Y and returns a p-value."""

    assert(alternative == "lower" or alternative == "greater" or alternative == "two sided")

    statistic = partial(_grouped_statistic_wrapper, statistic, np.max(x)) if grouped else statistic
    cores = mp.cpu_count()
    job_samples = num_samples/cores + 1
    num_samples = job_samples * cores
    pool = mp.Pool(processes=cores)

    densities = [pool.apply_async(_permutation_test, args=(statistic, x, y, job_samples, alternative, grouped)) for i in range(cores)]
    densities = [density.get() for density in densities]

    pool.close()
    pool.join()

    return np.sum(densities)/float(num_samples)

def grouped_permutation_test(statistic, groups, num_samples=100000, alternative="two sided"):
    """ Tests the independence of the statistic for the supplied groups and returns a p-value."""

    idx = []
    data = []

    for i, group in enumerate(groups):
        n = len(group)
        idx.extend([i]*len(group))
        data.extend(group)

    return permutation_test(statistic, np.array(idx), np.array(data), num_samples, alternative, True)

def _median_difference(x, y):
    return np.median(x) - np.median(y)

if __name__ == "__main__":
    x1 = np.array([0.80, 0.83, 1.89, 1.04, 1.45, 1.38, 1.91, 1.64, 0.73, 1.46])
    x2 = np.array([1.15, 0.88, 0.90, 0.74, 1.21])
    print grouped_permutation_test(_median_difference, [x1, x2])
    print bootstrap_interval(np.median, x1)
