"""
better corner plots
"""

# std
import numbers
import warnings
import itertools as itt

# third-party
import numpy as np
from loguru import logger
from matplotlib.gridspec import GridSpec
from matplotlib import colormaps, pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# local
from recipes.config import ConfigNode

# relative
from . import density
from .utils import percentile
from .hist import Histogram, get_bins


# ---------------------------------------------------------------------------- #
# Module config
CONFIG = ConfigNode.load_module(__file__)

# ---------------------------------------------------------------------------- #


def truncate_colormap(cmap, lo=0, hi=1, n=255):
    cmap = colormaps.get_cmap(cmap)
    return LinearSegmentedColormap.from_list(
        f'{cmap.name}[{lo:.2f}, {lo:.2f}]',
        cmap(np.linspace(lo, hi, n)))


# DEFAULT_CMAP = CONFIG.cmap #truncate_colormap(CONFIG.cmap, 0, 0.9)


def corner(samples, bins=CONFIG.nbins, plims=(0.5, 99.5),
           labels=None, label_kws=None, tessellation='hex', cmap=CONFIG.cmap,
           min_count_density=3, scatter_kws=None, density_kws=None,
           truths=None, truth_kws=None, hist_kws=None, priors=None,
           prior_kws=None):
    #  original parameters that are not implemented

    # smooth=None,
    # smooth1d=None,

    # show_titles=False,
    # title_fmt='.2f',
    # title_kws=None,

    # scale_hist=False,
    # quantiles=None,
    # verbose=False,
    # fig=None,
    # max_n_ticks=5,
    # top_ticks=False,
    # use_math_text=False,
    # hist_kws=None,
    # **hist2d_kws,
    """
    WIP
    Better corner plots.

    Extensions of the original functionality includes:
        * support for hexagonal tessellation in 2d histograms
        * y-range of axes in the same rows and x-range of axes in the same
             columns are connected, so that zooming/panning on one of the axes
             automatically adjusts the range of all the other axes that plot the
             same parameter!
        * Bin computation via bayesian block etc via `astropy.stats.histogram`
            and `numpy.histogram_bin_edges
        * performance: Only adding the axes we need to the figure.  This is
            different from `corner.corner` which adds a full grid of axes and
            then removes the upper right triangle.

        * TODO: plim range

    Parameters
    ----------
    samples
    bins: str, int
        str - name of the algorithm to use for computing the bin edges
        int - the number of bins to use
    plims: 2-tuple
        Percentile limits for range of histogram
    labels
    label_kws
    tessellation
    min_count_density
    scatter_kws
    density_kws
    truths
    truth_kws
    hist_kws
    priors: tuple, list
        Prior probability functions.  If given, the priors will be plotted
        with the marginal histograms.  This is sometimes useful in that it can
        indicate when your posterior samples pile up near the edges of the
        prior, ie. Your prior range is too small and you should choose a
        less informative prior.
    prior_kws: dict


    Returns
    -------

    """

    # intention: This function is a superset of `corner.corner` in
    #  terms of functionality.  It includes all the original options, and the

    *_, dof = samples.shape
    n = dof + 1
    if samples.ndim > 2:
        samples = samples.reshape(-1, dof)

    if dof > 10:
        raise NotImplementedError

    # check priors
    do_priors = (priors is not None)
    if do_priors:
        assert len(priors) == dof, 'Invalid number of priors. Received ' \
            f'{len(priors)}, expected {dof}'
        for i, pr in enumerate(priors):
            assert callable(pr), f'Prior {i} is not callable'

    # get defaults for dict params
    # prior_kws_ = dict(ls='--', color='grey')

    prior_kws = {**CONFIG.priors, **(prior_kws or {})}
    scatter_kws = {**CONFIG.scatter, **(scatter_kws or {})}
    density_kws = {**CONFIG.density, 'cmap': cmap, **(density_kws or {})}
    label_kws = {**CONFIG.labels, **(label_kws or {})}  # text params
    hist_kws = {'cmap': cmap, **(hist_kws or {})}
    tick_kws = CONFIG.ticks

    # setup figure
    fig = plt.figure()
    gs = GridSpec(dof, dof, figure=fig, **CONFIG.figure.margins)

    # get ranges
    # default to min-max scaling (same as range=None for histogram)
    if not np.any(plims):
        plims = (0, 100)
    lims = percentile(samples, plims, 0).T
    logger.debug('Ranges: {!s}.', lims)

    # get bins
    if tessellation == 'hex':
        if not isinstance(bins, numbers.Integral):
            warnings.warn(
                f'Ignoring bins {bins}, since tessellation is {tessellation!r}.'
                f' Falling back to default bins = {CONFIG.nbins}'
            )
            bins = CONFIG.nbins
        bins = np.full(dof, bins)
    else:
        # compute bins
        bins = [get_bins(s, bins, rng)
                for s, rng in zip(samples.T, lims)]

    # loopy loop!
    axes = np.ma.masked_all((n, n), 'O')
    for i, j in itt.combinations_with_replacement(range(dof), 2):
        logger.debug('plotting {} {}.', i, j)

        # ii, jj the row-, column indices from upper left corner of figure
        ii, jj = dof - i - 1, dof - j - 1

        ax = axes[ii, jj] = fig.add_subplot(gs[ii:ii + 1, jj:jj + 1])
        # ax.text(0.5, 0.5, f'{ii}, {jj}', transform=ax.transAxes)

        # connect row and column axes limits for same parameter
        if i > 0:
            ax.get_shared_x_axes().joined(ax, axes[ii + 1, jj])
        if j > i + 1:
            ax.get_shared_y_axes().joined(ax, axes[ii, jj + 1])

        xlims, ylims = lims[[jj, ii]]
        if ii == jj:
            # marginal density plot
            logger.debug('Plotting marginal histogram {}.', ii)
            h = Histogram(samples[:, ii], bins[ii], range=xlims, density=True)
            bars = h.plot(ax, **hist_kws)

            if do_priors:
                x = np.linspace(*xlims, 100)
                ax.plot(x, priors[ii](x), **prior_kws)
                # this will not affect the chosen axes limits

            # set axes limits
            ax.set(xlim=xlims, ylim=(0, percentile(h.counts, 102.5)))

            # top row
            if ii == 0:
                ylbl = ax.yaxis.label
                ax.text(-0.4, 0.5, ylbl.get_text(), transform=ax.transAxes)
                # # new.update_from(ylbl)
                # new.set_transform(ylbl.get_transform().frozen())
                # ax.yaxis.set_label_position('left')
                # fixme: this moves out of place when you resize the figure

            # label / tick y-axis on right
            ax.yaxis.set_label_position('right')
            if labels is not None:
                label = labels[jj]
                ax.set_ylabel(f'p({label})', **label_kws)


            ax.tick_params(right=True,
                           labelright=True,
                           labelleft=(ii == 0),
                           labelbottom=(i == 0),
                           **CONFIG.ticks)
            for tick in ax.get_yticklabels():
                tick.set_va('bottom')

            #
            ax.grid()

        else:
            # plot scatter / density
            hvals, poly_coll, points = density.scatter_map(
                ax, samples[:, [jj, ii]], (bins[jj], bins[ii]), None, None,
                min_count_density, tessellation, cmap,
                scatter_kws=scatter_kws, density_kws=density_kws)

            if labels is not None:
                # labels / ticks
                if i == 0:
                    # bottom row
                    ax.set_xlabel(labels[jj],  **label_kws)
                if jj == 0:
                    # left column
                    ax.set_ylabel(labels[ii], **label_kws)

            #
            ax.tick_params(labelleft=(jj == 0),
                           labelbottom=(i == 0),
                           **tick_kws)

            ax.set(xlim=xlims, ylim=ylims)

    return fig, axes

    # TODO: good guess here for limits

    # ylim, xlim = np.sort(np.percentile(pair, (0.001, 99.99), 0))
    # ax.set(xlim=xlim, ylim=ylim)
