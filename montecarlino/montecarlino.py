import numpy as np
import multiprocessing as mp
from functools import partial

def _bootstrap_interval(statistic, data, num_samples):
    np.random.seed()
    n = len(data)
    idx = np.random.randint(0, n, (num_samples, n))
    samples = data[idx]
    return [statistic(sample) for sample in samples]

def bootstrap_interval(statistic, data, num_samples=100000, alpha=0.05):
    """Returns a confidence interval for the specified statistic."""

    cores = mp.cpu_count()
    pool = mp.Pool(processes=cores)
    job_samples = num_samples/cores + 1

    stat = [pool.apply_async(_bootstrap_interval, args=(statistic, data, job_samples)) for i in range(cores)]
    stat = [s.get() for s in stat]
    stat = np.sort([item for list in stat for item in list])

    pool.close()
    pool.join()

    return (stat[int((alpha/2.0)*num_samples)],
            stat[int((1-alpha/2.0)*num_samples)])

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
