# third-party
import numpy as np
import matplotlib.pyplot as plt

# local
from recipes.logging import LoggingMixin

# relative
from .utils import percentile
from .image.utils import _sanitize_data


def get_bins(data, bins, range=None):
    # superset of the automated binning from astropy / numpy
    if isinstance(bins, str) and bins in ('blocks', 'knuth', 'freedman'):
        from astropy.stats import calculate_bin_edges
        return calculate_bin_edges(data, bins, range)
    else:
        return np.histogram_bin_edges(data, bins, range)


class Histogram(LoggingMixin):
    """A histogram that carries some state"""

    bins = 'auto'
    range = None

    def __init__(self, data, bins=bins, range=None, plims=None, **kws):
        # create
        super().__init__()
        # run
        self(data, bins, range, plims, **kws)

    def __call__(self, data, bins=bins, range=None, plims=None, **kws):

        # compute histogram
        data = _sanitize_data(data)
        if plims is not None:
            # choose range based on data percentile limits
            range = percentile(data, plims)  # sourcery skip: avoid-builtin-shadow

        self.bin_edges = self.auto_bins(data, bins, range)
        self.counts, _ = np.histogram(data, self.bin_edges, range, **kws)

    @property
    def bin_centers(self):
        return self.bin_edges[:-1] + np.diff(self.bin_edges)

    @property
    def n(self):
        return len(self.counts)

    def auto_bins(self, data, bins=bins, range=None):
        return get_bins(data, bins, range)

    def get_verts(self):
        """vertices for vertical bars"""
        if len(self.counts) == 0:
            # empty histogram
            return []

        x01 = [self.bin_edges[:-1], self.bin_edges[1:]]
        x = x01 + x01[::-1]

        ymin = 0
        y = list(np.full((2, self.n), ymin)) + [self.counts] * 2

        return np.array([x, y]).T

    def get_bars(self, **kws):
        # create collection

        from matplotlib.collections import PolyCollection
        return PolyCollection(self.get_verts(),
                              array=self.counts / self.counts.max(),
                              **kws)

    def plot(self, ax, **kws):
        bars = self.get_bars(**kws)
        ax.add_collection(bars)
        return bars


def hist(x, bins=100, range=None, normed=False, weights=None, **kws):
    """
    Plot a nice looking histogram.

    Parameters
    ----------
    x:          sequence
        Values to histogram

    Keywords
    --------
    axes_labels:   sequence
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

    # https://en.wikipedia.org/wiki/Quantile#Specialized_quantiles
    named_quantiles = {25: 'lower  quartile',
                       50: 'median',
                       75: 'upper quartile'}

    show_stats = kws.pop('show_stats', ())
    show_stats_labels = kws.pop('show_stats_labels', True)
    fmt_stats = kws.pop('fmt_stats', None)
    lbls = kws.pop('axes_labels', ())
    title = kws.pop('title', '')
    alpha = kws.setdefault('alpha', 0.75)
    ax = kws.pop('ax', None)
    Q = kws.pop('percentile', [])

    # Create figure
    if ax is None:
        _, ax = plt.subplots(tight_layout=True)
        # else:
        # fig = ax.figure

    # compute bins if heuristic
    bins = get_bins(x, bins, range)

    # Plot the histogram
    h = counts, bins, patches = ax.hist(x, bins, range, normed, weights, **kws)

    # Make axis labels and title
    xlbl = lbls[0] if len(lbls) else ''
    ylbl = lbls[1] if len(lbls) > 1 else ('Density' if normed else 'Counts')
    ax.set_xlabel(xlbl)
    ax.set_ylabel(ylbl)
    ax.set_title(title)
    ax.grid()

    # Extra summary statistics (point estimators)
    stats = {}
    if 'min' in show_stats:
        stats['min'] = x.min()

    if 'max' in show_stats:
        stats['max'] = x.max()

    if 'mode' in show_stats:
        from scipy.stats import mode

        stats['mode'] = mode(x).mode.item()

    if 'mean' in show_stats:
        stats['mean'] = x.mean()
    if 'median' in show_stats:
        Q.append(50)

    # if 'percentile' in show_stats:
    # pass

    if len(Q):  # 'percentile' in show_stats:
        P = np.percentile(x, Q)
        for p, q in zip(P, Q):
            name = named_quantiles.get(q, '$p_{%i}$' % q)
            stats[name] = p

    if fmt_stats is None:
        from recipes.pprint import decimal as fmt_stats

    if stats:
        from matplotlib.transforms import blended_transform_factory as btf

        for key, val in stats.items():
            c = patches[0].get_facecolor()
            ax.axvline(val, color=c, alpha=1, ls='--', lw=2)
            trans = btf(ax.transData, ax.transAxes)
            if show_stats_labels:
                txt = '%s = %s' % (key, fmt_stats(val))
                ax.text(val, 1, txt,
                        transform=trans,
                        rotation='vertical', va='top', ha='right')

    return h, ax
