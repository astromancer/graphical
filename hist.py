import six

import numpy as np
import matplotlib.pyplot as plt


def hist(x, bins=100, range=None, weights=None, **kws):
    """
    Plot a nice looking histogram.

    Parameters
    ----------
    x:          sequence
        Values to histogram

    Keywords
    --------
    axlabels:   sequence
        One or two axis labels (x,y)
    title:      str
        The figure title
    show_stats: str; option ('mode',)
        Show the given statistic of the distribution
    * Remaining keywords are passed to ax.hist

    Returns
    -------
    h:          tuple
        bins, values
    ax:         axes
    """

    show_stats = kws.pop('show_point_estimates', ())
    fmt_stats = kws.pop('fmt_stats', None)
    lbls = kws.pop('axlabels', ())
    title = kws.pop('title', '')
    # ax = ax.plot

    # if bins is a string, first compute bin edges with the desired heuristic
    if isinstance(bins, six.string_types):
        a = np.asarray(x).ravel()

        # TODO: if weights is specified, we need to modify things.
        #       e.g. we could use point measures fitness for Bayesian blocks
        if weights is not None:
            raise NotImplementedError("weights are not yet supported "
                                      "for the enhanced histogram")

        # if range is specified, we need to truncate the data for
        # the bin-finding routines
        if range is not None:
            a = a[(a >= range[0]) & (a <= range[1])]

        if bins == 'blocks':
            from astropy.stats import bayesian_blocks
            bins = bayesian_blocks(a)
        elif bins == 'knuth':
            from astropy.stats import knuth_bin_width
            da, bins = knuth_bin_width(a, True)
        elif bins == 'scott':
            from astropy.stats import scott_bin_width
            da, bins = scott_bin_width(a, True)
        elif bins == 'freedman':
            from astropy.stats import freedman_bin_width
            da, bins = freedman_bin_width(a, True)
        else:
            raise ValueError("unrecognized bin code: '{}'".format(bins))

    alpha = kws.setdefault('alpha', 0.5)
    Q = kws.pop('percentile', [])
    named_quantiles = {25: 'lower  quartile',  # https://en.wikipedia.org/wiki/Quantile#Specialized_quantiles
                       50: 'median',
                       75: 'upper quartile'}

    # Create figure
    ax = kws.pop('ax', None)
    if ax is None:
        _, ax = plt.subplots(tight_layout=1, figsize=(12, 8))
        # else:
        # fig = ax.figure

    # Plot the histogram
    h = counts, bins, patches = ax.hist(x, bins, range, weights, **kws)

    # Make axis labels and title
    xlbl = lbls[0] if len(lbls)     else ''
    ylbl = lbls[1] if len(lbls) > 1   else 'Counts'
    ax.set_xlabel(xlbl)
    ax.set_ylabel(ylbl)
    ax.set_title(title)
    ax.grid()

    # Extra summary statistics (point estimators)
    stats = {}
    if len(show_stats):
        from matplotlib.transforms import blended_transform_factory as btf

    if 'min' in show_stats:
        stats['min'] = x.min()

    if 'max' in show_stats:
        stats['max'] = x.max()

    if 'mode' in show_stats:
        from scipy.stats import mode
        mr = mode(x)
        xmode = mr.mode.squeeze()
        stats['mode'] = xmode

    if 'mean' in show_stats:
        stats['mean'] = x.mean()
    if 'median' in show_stats:
        Q.append(50)

    if len(Q):  # 'percentile' in show_stats:
        P = np.percentile(x, Q)
        for p, q in zip(P, Q):
            name = named_quantiles.get(q, '$p_{%i}$' % q)
            stats[name] = p

    if fmt_stats is None:
        from recipes.string import minfloatfmt
        fmt_stats = minfloatfmt

    for key, val in stats.items():
        ax.axvline(val, color='r', alpha=alpha, ls='--', lw=2)
        trans = btf(ax.transData, ax.transAxes)
        txt = '%s = %s' % (key, fmt_stats(val))
        ax.text(val, 1, txt,
                rotation='vertical', transform=trans, va='top', ha='right')

        # if 'percentile' in show_stats:
        # pass

    return h, ax
