"""
Implements a pixel colour value histogram intended to accompany image colorbars.
"""


# third-party
import numpy as np
from matplotlib import ticker
from matplotlib.colors import ListedColormap

# local
from recipes.logging import LoggingMixin
from recipes.utils import duplicate_if_scalar

# relative
from .utils import _sanitize_data


# ---------------------------------------------------------------------------- #
_sci = ticker.LogFormatterSciNotation()


def fmt_log_tick(x, pos=None):
    # x = float(x)
    if 1 <= x <= 100:
        return str(int(x))

    if 0.1 <= x < 1:
        return f'{x:.1f}'

    return _sci(x)

# ---------------------------------------------------------------------------- #


class PixelHistogram(LoggingMixin):  # PixelHistogram
    """
    Histogram of colour values in an image
    """

    # TODO: get to work with RGB data

    _default_n_bins = 50

    @classmethod
    def from_image(cls, image_artist):
        pass

    # todo. better with data?
    def __init__(self, ax, image_plot, orientation='horizontal', use_blit=True,
                 outside_colour=None, outside_alpha=0.5, **kws):
        """
        Display a histogram for colour values in an image.

        Parameters
        ----------
        ax
        image_plot
        use_blit
        outside_colour
        kws
        """

        # TODO: dynamic recompute histogram when too few bins are shown..
        # TODO: integrate color stretch functionality
        #  FIXME: fails for all zero data

        from matplotlib.collections import PolyCollection

        self.log = kws.pop('log', True)
        self.ax = ax
        self.image_plot = image_plot
        self.norm = image_plot.norm

        assert orientation.lower().startswith(('h', 'v'))
        self.orientation = orientation

        self.set_cmap(image_plot, outside_colour, outside_alpha)

        # compute histogram
        self.bins = kws.pop('bins', self._default_n_bins)
        self.counts = self.bin_edges = self.bin_centers = ()
        self.compute(self.get_array())

        # create collection
        self.bars = PolyCollection(self.get_verts(self.counts, self.bin_edges),
                                   array=self.norm(self.bin_centers),
                                   cmap=self.cmap)
        ax.add_collection(self.bars)

        if use_blit:
            # image_plot.set_animated(True)
            self.bars.set_animated(True)

        # set axes limits
        if self.log:
            ax.set(xscale='log', xlim=(0.75, None))
            ax.xaxis.major.formatter = ticker.FuncFormatter(fmt_log_tick)

        # rescale if non-empty histogram
        if len(self.counts):
            self.autoscale_view()

    def set_cmap(self, image_plot, outside_colour, outside_alpha):
        # setup colormap (copy)
        cmap = image_plot.get_cmap()
        self.cmap = ListedColormap(cmap(np.linspace(0, 1, 256)))

        # optionally gray out out-of-bounds values
        if outside_colour is None:
            outside_colours = self.cmap([0., 1.])  # note float
            outside_colours[:, -1] = outside_alpha
            under, over = outside_colours
        else:
            under, over = duplicate_if_scalar(outside_colour)
        #
        self.cmap.set_over(over)
        self.cmap.set_under(under)

    def get_array(self):
        return self.image_plot.get_array()

    def set_array(self, data):

        # compute histogram
        self.compute(data)

        # create collection
        self.bars.set_verts(self.get_verts(self.counts, self.bin_edges))
        self.bars.set_array(self.norm(self.bin_centers))

    def compute(self, data, bins=None, range=None):
        # compute histogram
        self.bins = self._auto_bins(self.bins if bins is None else bins)
        if range is None:  # FIXME: should be included in bins vector
            range = self._auto_range()  # sourcery skip: avoid-builtin-shadow

        self.counts, self.bin_edges = np.histogram(
            _sanitize_data(data), self.bins, range)
        self.bin_centers = self.bin_edges[:-1] + np.diff(self.bin_edges)

    def get_verts(self, counts, bin_edges):
        """vertices for horizontal bars"""
        # FIXME: order swaps for vertical bars

        if len(counts) == 0:
            # empty histogram
            return []

        xmin = 0
        ywidth = np.diff(bin_edges[:2])[0]
        return [[(xmin, ymin),
                 (xmin, ymin + ywidth),
                 (xmin + xwidth, ymin + ywidth),
                 (xmin + xwidth, ymin),
                 (xmin, ymin)]
                for xwidth, ymin in zip(counts, bin_edges)]

    def update(self):

        # data = self.image_plot.get_array()
        # rng = self._auto_range()
        #
        # self.counts, self.bin_edges = counts, bin_edges =\
        #     np.histogram(_sanitize_data(data), self.bins, rng)

        self.bars.set_verts(
            self.get_verts(self.counts, self.bin_edges)
        )

        # bin_centers = bin_edges[:-1] + np.diff(bin_edges)
        # self.bars.set_array(self.norm(self.bin_centers))
        # note set_array doesn't seem to work correctly. bars outside the
        #  range get coloured for some reason

        self.bars.set_facecolors(self.cmap(self.norm(self.bin_centers)))
        return self.bars  # TODO: xtick labels if necessary

    def _auto_bins(self, n=None):
        bins = self.bins if n is None else n
        data = self.get_array()

        # unit bins for integer arrays containing small range of numbers
        if data.dtype.kind == 'i':  # integer array
            lo, hi = np.nanmin(data), np.nanmax(data)
            bins = np.arange(min(hi - lo, n) + 1)
            # NOTE: this actually sets the range as well...
        return bins

    def _auto_range(self, stretch=1.2):
        # choose range based on image colour limits
        vmin, vmax = self.image_plot.get_clim()
        if vmin == vmax:
            self.logger.warning('Colour range is 0! Falling back to min-max '
                                'range.')
            image = self.get_array()
            return image.min(), image.max()

        # set the axes limits slightly wider than the clims
        m = 0.5 * (vmin + vmax)
        w = (vmax - vmin) * stretch
        return m - w / 2, m + w / 2

    def autoscale_view(self):
        # set the axes limits slightly wider than the clims
        # the naming here assumes horizontal histogram orientation
        xmin = 0.7 if self.log else 0
        xlim = (xmin, self.counts.max())
        ylim = self._auto_range()

        if self.orientation.startswith('v'):
            xlim, ylim = ylim, xlim

        # self.logger.debug('Ax lims: ({:.1f}, {:.1f})', *lim)
        self.ax.set(xlim=xlim, ylim=ylim)
