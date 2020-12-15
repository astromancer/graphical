"""
better corner plots
"""

import itertools as itt
from graphing.scatter import scatter_density

from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import warnings
import numbers
from .hist import Histogram, get_bins
from .utils import percentile

import logging
from recipes.logging import get_module_logger


# module level logger
logger = get_module_logger()

DEFAULT_NBINS = 30


def truncate_colormap(cmap, lo=0, hi=1, n=255):
    cmap = plt.get_cmap(cmap)
    return LinearSegmentedColormap.from_list(
        f'{cmap.name}[{lo:.2f}, {lo:.2f}]',
        cmap(np.linspace(lo, hi, n)))


DEFAULT_CMAP = truncate_colormap('magma', 0, 0.9)


def corner(samples, bins=DEFAULT_NBINS, plims=(0.5, 99.5),
           labels=None, label_kws=None, tessellation='hex',
           min_count_density=3, scatter_kws=None, density_kws=None,
           truths=None, truth_kws=None, hist_kws=None, priors=None,
           prior_kws=None):
    #  original parameters that are not implemented

    # smooth=None,
    # smooth1d=None,

    # show_titles=False,
    # title_fmt='.2f',
    # title_kwargs=None,

    # scale_hist=False,
    # quantiles=None,
    # verbose=False,
    # fig=None,
    # max_n_ticks=5,
    # top_ticks=False,
    # use_math_text=False,
    # hist_kwargs=None,
    # **hist2d_kwargs,
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
    prior_kws_ = dict(ls='--', color='grey')
    prior_kws_.update(prior_kws or {})
    scatter_kws_ = dict(marker='.', ms=1, alpha=0.75)
    scatter_kws_.update(scatter_kws or {})
    density_kws_ = dict(alpha=0.75, edgecolors='none')
    density_kws_.update(density_kws or {})
    hist_kws = hist_kws or {}
    hist_kws.setdefault('cmap', density_kws_.setdefault('cmap', DEFAULT_CMAP))

    # setup figure
    fig = plt.figure()
    gridspec_kw = dict(hspace=0.05, wspace=0.05)
    gs = GridSpec(dof, dof, figure=fig, **gridspec_kw)

    # text params
    tick_kws = dict(labelrotation=45, pad=0, length=2)
    label_kws_ = dict(labelpad=15, rotation=0, va='center', rotation_mode=None)
    label_kws_.update(label_kws or {})

    # get ranges
    # default to min-max scaling (same as range=None for histogram)
    if not np.any(plims):
        plims = (0, 100)
    lims = percentile(samples, plims, 0).T
    logger.debug('Ranges: %s', lims)

    # get bins
    if tessellation == 'hex':
        if not isinstance(bins, numbers.Integral):
            warnings.warn(
                f'Ignoring bins {bins}, since tessellation is {tessellation!r}.'
                f' Falling back to default bins = {DEFAULT_NBINS}'
            )
            bins = DEFAULT_NBINS
        bins = np.full(dof, bins)
    else:
        # compute bins
        bins = [get_bins(s, bins, rng)
                for s, rng in zip(samples.T, lims)]

    # loopy loop!
    axes = np.ma.masked_all((n, n), 'O')
    for i, j in itt.combinations_with_replacement(range(dof), 2):
        # ii, jj the row-, column indices from upper left corner of figure
        ii, jj = dof - i - 1, dof - j - 1
        label = labels[jj]
        logger.debug('plotting %i %i', i, j)

        ax = axes[ii, jj] = fig.add_subplot(gs[ii:ii + 1, jj:jj + 1])
        # ax.text(0.5, 0.5, f'{ii}, {jj}', transform=ax.transAxes)

        # connect row and column axes limits for same parameter
        if i > 0:
            ax.get_shared_x_axes().join(ax, axes[ii + 1, jj])
        if j > i + 1:
            ax.get_shared_y_axes().join(ax, axes[ii, jj + 1])

        xlims, ylims = lims[[jj, ii]]
        if ii == jj:
            # marginal density plot
            logger.debug('Plotting marginal histogram %i' % ii)
            h = Histogram(samples[:, ii], bins[ii], range=xlims, density=True)
            bars = h.plot(ax, **hist_kws)

            if do_priors:
                x = np.linspace(*xlims, 100)
                ax.plot(x, priors[ii](x), **prior_kws_)
                # this will not affect the chosen axes limits

            # set axes limits
            ax.set(xlim=xlims, ylim=(0, percentile(h.counts, 102.5)))

            # label / tick y-axis on right
            ax.yaxis.set_label_position('right')
            ax.set_ylabel('p(%s)' % label, **label_kws_)

            # top row
            if ii == 0:
                ylbl = ax.yaxis.label
                ax.text(-0.4, 0.5, ylbl.get_text(), transform=ax.transAxes)
                # # new.update_from(ylbl)
                # new.set_transform(ylbl.get_transform().frozen())
                # ax.yaxis.set_label_position('left')
                # fixme: this moves out of place when you resize the figure

            # bottom row
            if i == 0:
                ax.set_xlabel(label, **label_kws_)

            ax.tick_params(right=True,
                           labelright=True,
                           labelleft=(ii == 0),
                           labelbottom=(i == 0),
                           **tick_kws)
            for tick in ax.get_yticklabels():
                tick.set_va('bottom')

            #
            ax.grid()

        else:
            # plot scatter / density
            hvals, poly_coll, points = scatter_density(
                ax, samples[:, [jj, ii]], (bins[jj], bins[ii]), None,
                min_count_density, tessellation=tessellation,
                scatter_kws=scatter_kws_, density_kws=density_kws_)

            # labels / ticks
            if i == 0:
                # bottom row
                ax.set_xlabel(label)
            if jj == 0:
                # left column
                ax.set_ylabel(labels[ii], **label_kws_)

            #
            ax.tick_params(labelleft=(jj == 0),
                           labelbottom=(i == 0),
                           **tick_kws)

            ax.set(xlim=xlims, ylim=ylims)

    return fig, axes

    # TODO: good guess here for limits

    # ylim, xlim = np.sort(np.percentile(pair, (0.001, 99.99), 0))
    # ax.set(xlim=xlim, ylim=ylim)
