
# std
import math
from pathlib import Path

# third-party
import numpy as np
import matplotlib.transforms as mtransforms
from matplotlib.colors import to_rgba
from matplotlib import patheffects as path_effects
from loguru import logger

# local
from recipes.containers import ensure


def is_none(*args):
    for a in args:
        yield a is None


def not_none(*args):
    for t in is_none(*args):
        yield not t


def percentile(data, p, axis=None):
    """
    Get percentile value on (possibly masked) `data`. Negative values for
    `p` are interpreted as percentile distance below minimum.  Similarly for
    values of `p` greater than 100. Useful for scaling the axes of plots.

    Parameters
    ----------
    data : array-like
        Data to compute percentile values for.
    p : array-like
        Percentile value(s) in the interval (-100, 100)
    axis : int or tuple, optional
        Axis along which to do the computation, by default None, which uses all
        data

    Examples
    --------
    >>> 

    Returns
    -------
    [type]
        [description]

    Raises
    ------
    NotImplementedError
        [description]
    """

    data = np.asanyarray(data)
    signum = np.array([-1, 1])

    p = np.array(p, ndmin=1) / 100
    # p = np.divide(p, 100)
    a = np.abs(p)
    s = signum[(p > 0).astype(int)]
    # _, q = np.divmod(a, 1)
    c = np.abs((p > 1).astype(float) - s * (a % 1)) * 100

    # remove masked points
    if np.ma.is_masked(data):
        if axis is not None:
            raise NotImplementedError

        data = np.ma.compressed(data)

    # create output array
    d = np.zeros((len(p),
                  *(np.take(data.shape, np.delete(np.arange(data.ndim), axis))
                    if axis is not None else ())))
    d[c > 0] = np.percentile(data, c[c > 0], axis)

    # sourcery skip: flip-comparison
    mn, mx, = data.min(axis, keepdims=True), data.max(axis, keepdims=True)
    p1 = (p > 1).astype(int)
    s2 = np.array(signum[((0 < p) & (p < 1)).astype(int)], ndmin=d.ndim).T
    u = np.array(p1 - s * np.ceil(a) + 1, ndmin=d.ndim).T
    v = np.array(p1 + s * np.floor(a), ndmin=d.ndim).T
    return np.squeeze(u * mn + v * mx + s2 * d)


def get_percentiles(data, plims=(-5, 105), errorbars=(), axis=None):
    """
    Return suggested axis limits based on the extrema of `data` and optional
    1 sigma standard deviation `errorbars`.

    data : array-like
        data on display
    plims : 2-tuple
        Data limits expressed as percentiles of the data distribution.
        0 corresponds to the 0th percentile, 100 to the 100th percentile.
        numbers outside range (0, 100) are allowed, in which case they will be
        interpreted as distances from the 0th and 100th percentile
        respectively and the unit distance is a 100th of the data peak-to-peak
        distance.
    e : uncertainty (stddev, measurement errors)
        can be either single array of same shape as x, or 2 arrays (δx+, δx-)
    axis :  int, tuple
        axis along which to compute percentile
    """
    if np.ma.ptp(plims) == 0:
        raise ValueError('Percentile values for colour limits must differ. '
                         f'Received: {plims}.')

    x = get_data_pm_1sigma(data, errorbars)
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

    # sourcery skip: assign-if-exp, reintroduce-else
    if n == 2:
        return x - e[0], x + e[1]

    return x - e, x + e


def emboss(art, linewidth=2, color='k', alpha=1):
    # add border around artists to make them stand out
    art.set_path_effects([
        path_effects.Stroke(linewidth=linewidth,
                            foreground=to_rgba(color, alpha)),
        path_effects.Normal()
    ])
    return art


# alias
embossed = emboss

# ---------------------------------------------------------------------------- #


def _check_log_scalable(x, name):
    if np.any(x <= 0.0):
        raise ValueError(
            f'{name} contains non-positive values, so cannot be log-scaled.'
        )


def hexbin(x, y, C=None, gridsize=100,
           xscale='linear', yscale='linear', extent=None,
           reduce_C_function=np.mean, mincnt=None):
    """
    Function to support histogramming over hexagonal tesselations.
    """

    # Set the size of the hexagon grid
    nx2, ny2 = nxy = np.array(gridsize if np.iterable(gridsize) else
                              (gridsize, int(gridsize / math.sqrt(3))))

    # Will be log()'d if necessary, and then rescaled.
    tx, ty = txy = np.array([x, y])

    for i, scale in enumerate((xscale, yscale)):
        if scale == 'log':
            _check_log_scalable((xy := txy[i]), 'xy'[i])
            txy[i] = np.log10(xy)

    if extent is None:
        xmin, xmax = (tx.min(), tx.max()) if len(x) else (0, 1)
        ymin, ymax = (ty.min(), ty.max()) if len(y) else (0, 1)

        # to avoid issues with singular data, expand the min/max pairs
        xmin, xmax = mtransforms.nonsingular(xmin, xmax, expander=0.1)
        ymin, ymax = mtransforms.nonsingular(ymin, ymax, expander=0.1)
    else:
        xmin, xmax, ymin, ymax = extent

    nx1, ny1 = nxy1 = nxy + 1
    n = nxy1.prod() + nxy.prod()

    # In the x-direction, the hexagons exactly cover the region from
    # xmin to xmax. Need some padding to avoid roundoff errors.
    padding = 1.e-9 * (xmax - xmin)
    xmin -= padding
    xmax += padding
    sx = (xmax - xmin) / nx2
    sy = (ymax - ymin) / ny2
    # Positions in hexagon index coordinates.
    ix = (tx - xmin) / sx
    iy = (ty - ymin) / sy
    ix1, iy1 = np.round([ix, iy]).astype(int)
    ix2, iy2 = np.floor([ix, iy]).astype(int)
    # flat indices, plus one so that out-of-range points go to position 0.
    i1 = np.where((0 <= ix1) & (ix1 < nx1) & (0 <= iy1) & (iy1 < ny1),
                  ix1 * ny1 + iy1 + 1, 0)
    i2 = np.where((0 <= ix2) & (ix2 < nx2) & (0 <= iy2) & (iy2 < ny2),
                  ix2 * ny2 + iy2 + 1, 0)

    d1 = (ix - ix1) ** 2 + 3.0 * (iy - iy1) ** 2
    d2 = (ix - ix2 - 0.5) ** 2 + 3.0 * (iy - iy2 - 0.5) ** 2
    bdist = (d1 < d2)

    if C is None:  # [1:] drops out-of-range points.
        counts1 = np.bincount(i1[bdist], minlength=1 + nx1 * ny1)[1:]
        counts2 = np.bincount(i2[~bdist], minlength=1 + nx2 * ny2)[1:]
        accum = np.concatenate([counts1, counts2]).astype(float)
        if mincnt is not None:
            accum[accum < mincnt] = np.nan

    else:
        # store the C values in a list per hexagon index
        Cs_at_i1 = [[] for _ in range(1 + nx1 * ny1)]
        Cs_at_i2 = [[] for _ in range(1 + nx2 * ny2)]
        for i in range(len(x)):
            if bdist[i]:
                Cs_at_i1[i1[i]].append(C[i])
            else:
                Cs_at_i2[i2[i]].append(C[i])
        if mincnt is None:
            mincnt = 0
        accum = np.array(
            [reduce_C_function(acc) if len(acc) >= mincnt else np.nan
                for Cs_at_i in [Cs_at_i1, Cs_at_i2]
                for acc in Cs_at_i[1:]],  # [1:] drops out-of-range points.
            float)

    good_idxs = ~np.isnan(accum)

    offsets = np.zeros((n, 2), float)
    offsets[:nx1 * ny1, 0] = np.repeat(np.arange(nx1), ny1)
    offsets[:nx1 * ny1, 1] = np.tile(np.arange(ny1), nx1)
    offsets[nx1 * ny1:, 0] = np.repeat(np.arange(nx2) + 0.5, ny2)
    offsets[nx1 * ny1:, 1] = np.tile(np.arange(ny2), nx2) + 0.5
    offsets[:, 0] *= sx
    offsets[:, 1] *= sy
    offsets[:, 0] += xmin
    offsets[:, 1] += ymin
    # remove accumulation bins with no data
    offsets = offsets[good_idxs, :]
    accum = accum[good_idxs]

    return (*offsets.T, accum), (xmin, xmax), (ymin, ymax), nxy


def save_figure(fig, filenames=(), overwrite=False, **kws):

    filenames = ensure.list(filenames)
    if fn := kws.pop('filename', ()):
        filenames.append(fn)
    filenames = ensure.tuple(filenames, Path)

    if not filenames:
        logger.debug(
            'Not saving figure {}: Could not resolve any filenames from {!r}.',
            fig, filenames
        )
        return 0

    if not fig:
        logger.warning('No figure {!r} for filenames: {}', fig, filenames)
        return 0

    if not fig.axes:
        logger.warning("Not saving figure {} at {} since it's empty.",
                       fig, filenames)
        return 0

    saved = 0
    for filename in filenames:
        if filename.exists():
            if overwrite:
                logger.info('Overwriting figure: {!s}.', filename)
            else:
                logger.info('Not overwriting: {!s}', filename)
                continue
        else:
            logger.info('Saving figure: {!s}.', filename)

        try:
            fig.savefig(filename, **kws)
            saved += 1
        except Exception as err:
            logger.warning('Could not save figure at filename {} due to: {}',
                           filename, err)

    return saved


# alias
save_fig = save_figure
