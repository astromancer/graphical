from math import ceil, floor

import numpy as np


def get_percentile(data, p):
    """
    Get percentile value on (possibly masked) `data`.  Negative values for
    `p` are interpreted as percentile distance below minimum.  Similarly for
    values of `p` greater than 100.

    Parameters
    ----------
    data: array-like
    p: float

    Returns
    -------

    """
    p = p / 100
    a = abs(p)
    s = [-1, 1][p > 0]
    r, q = divmod(a, 1)
    c = abs(float(p > 1) - s * q) * 100
    d = 0
    if c > 0:
        d = np.percentile(np.ma.compressed(data), c)

    mn, mx, = data.min(), data.max()
    p1 = int(p > 1)
    s2 = 1 if 0 < p < 1 else -1

    # m = p1 - s * ceil(a) + 1
    # n = p1 + s * floor(a)
    # for x in 'psrqcmn':
    #     print(f'{x} = {eval(x):<4g}', end='\t')
    # print()

    return (p1 - s * ceil(a) + 1) * mn + (p1 + s * floor(a)) * mx + s2 * d
    # print('p = ', p, 'lim', l, 'expected', expect)


def get_percentile_limits(data, plims=(-5, 105), e=()):
    """
    Return suggested axis limits based on the extrema of `data`, optional
    errorbars `e`.

    data: array-like
        data on display
    e - uncertainty (stddev, measurement errors)
        can be either single array of same shape as x, or 2 arrays (δx+, δx-)
    plims: 2-tuple
        Data limits expressed as percentiles of the data distribution.
        0 corresponds to the 0th percentile, 100 to the 100th percentile.
        numbers outside range (0, 100) are allowed, in which case they will be
        interpreted as distances from the 0th and 100th percentile
        respectively and the unit distance is a 100th of the data peak-to-peak
        distance.
    """

    x = get_data_pm_1sigma(data, e)
    lims = np.empty(2, data.dtype)
    for i, (x, p) in enumerate(zip(x, plims)):
        lims[i] = get_percentile(x, p)

    return lims


def get_data_pm_1sigma(x, e=()):
    """
    Compute the 68.27% confidence interval given the 1-sigma measurement
    uncertainties `e` are given, else return a 2-tuple with data duplicated.

    Parameters
    ----------
    x: array-like
    e: optional, array-like or 2-tuple of array-like
        If array like, assume this is the 1-sigma measurement uncertainties
        If 2-tuple, assume these are the upper and lower confidence distance
        from the mean

    Returns
    -------

    """
    if e is None:
        return x, x

    n = len(e)
    if n == 0:
        return x, x
    elif n == 2:
        return x - e[0], x + e[1]
    else:
        return x - e, x + e
