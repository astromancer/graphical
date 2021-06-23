import numpy as np


def percentile(data, p, axis=None):
    """
    Get percentile value on (possibly masked) `data`.  Negative values for
    `p` are interpreted as percentile distance below minimum.  Similarly for
    values of `p` greater than 100. Useful for scaling the axes of plots.

    Parameters
    ----------
    data: array-like
    p: array-like
    axis: None, int, tuple

    Returns
    -------

    """

    data = np.asanyarray(data)
    signum = np.array([-1, 1])

    p = np.array(p, ndmin=1)
    p = np.divide(p, 100)
    a = np.abs(p)
    s = signum[(p > 0).astype(int)]
    _, q = np.divmod(a, 1)
    c = np.abs((p > 1).astype(float) - s * q) * 100

    # remove masked points
    if np.ma.is_masked(data):
        if axis is not None:
            raise NotImplementedError

        data = np.ma.compressed(data)

    # get shape of output array
    out_shape = (len(p),)
    if axis is not None:
        out_shape += tuple(
            np.take(data.shape, np.delete(np.arange(data.ndim), axis))
        )
    ndo = len(out_shape)
    #
    d = np.zeros(out_shape)
    d[c > 0] = np.percentile(data, c[c > 0], axis)

    mn, mx, = data.min(axis, keepdims=True), data.max(axis, keepdims=True)
    p1 = (p > 1).astype(int)
    s2 = np.array(signum[((0 < p) & (p < 1)).astype(int)], ndmin=ndo).T
    u = np.array(p1 - s * np.ceil(a) + 1, ndmin=ndo).T
    v = np.array(p1 + s * np.floor(a), ndmin=ndo).T
    return np.squeeze(u * mn + v * mx + s2 * d)


def get_percentile_limits(data, plims=(-5, 105), e=(), axis=None):
    """
    Return suggested axis limits based on the extrema of `data` and optional
    errorbars `e`.

    data: array-like
        data on display
    plims: 2-tuple
        Data limits expressed as percentiles of the data distribution.
        0 corresponds to the 0th percentile, 100 to the 100th percentile.
        numbers outside range (0, 100) are allowed, in which case they will be
        interpreted as distances from the 0th and 100th percentile
        respectively and the unit distance is a 100th of the data peak-to-peak
        distance.
    e: uncertainty (stddev, measurement errors)
        can be either single array of same shape as x, or 2 arrays (δx+, δx-)
    axis: None, int, tuple
        axis along which to compute percentile
    """

    x = get_data_pm_1sigma(data, e)
    lims = np.empty(2, data.dtype)
    for i, (x, p) in enumerate(zip(x, plims)):
        lims[i] = percentile(x, p, axis)

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
    if n == 2:
        return x - e[0], x + e[1]
    return x - e, x + e
