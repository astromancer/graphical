"""
Implements a pixel colour value histogram intended to accompany image colorbars.
"""


# std
from types import MappingProxyType

# third-party
import numpy as np
from matplotlib import colormaps, ticker
from matplotlib.colors import ListedColormap
from matplotlib.collections import PolyCollection

# local
from recipes.logging import LoggingMixin
from recipes.utils import duplicate_if_scalar

# relative
from .utils import _sanitize_data


# ---------------------------------------------------------------------------- #
# _sci = ticker.LogFormatterSciNotation()


def fmt_log_tick(x, pos=None):
    # x = float(x)
    return str(int(x)) if 1 <= x <= 100 else f'{x:.1f}'
    # if 0.1 <= x < 1:

    # return

# ---------------------------------------------------------------------------- #


class PixelHistogram(LoggingMixin):  # PixelHistogram
    """
    Histogram of colour values in an image
    """

    # TODO: get to work with RGB data

    _default_n_bins = 50

    _default_outer_style = dict(facecolor=None,
                                edgecolor='0.75',
                                linewidth=0.5,
                                alpha=0.75)

    # @classmethod
    # def from_image(cls, image_plot):
    #     pass

    # todo. better with data?
    def __init__(self, ax, image,
                 orientation='horizontal',
                 outer_bar_style=MappingProxyType(_default_outer_style),
                 use_blit=True,
                 **kws):
        """
        Display a histogram for colour values in an image.

        Parameters
        ----------
        ax
        image
        use_blit
        outside_colour
        kws
        """

        # TODO: dynamic recompute histogram when too few bins are shown..
        # TODO: integrate color stretch functionality
        #  FIXME: fails for all zero data

        self.log = kws.pop('log', True)
        self.ax = ax
        self.image = image
        self.image.callbacks.connect('changed', self._update_cmap_from_image)

        assert orientation.lower().startswith(('h', 'v'))
        self.orientation = orientation

        # compute histogram
        self.bins = kws.pop('bins', self._default_n_bins)
        self.counts = self.bin_edges = self.bin_centers = ()
        self.compute(self.get_array())

        # colour map
        self._outer_style = {**self._default_outer_style, **outer_bar_style}
        cmap = self._compute_cmap(self.image.get_cmap(),
                                  self._outer_style['facecolor'],
                                  self._outer_style['alpha'])

        # create collection
        self.bars = PolyCollection([]) # cmap=cmap, norm=self.image.norm
        self.set_cmap(cmap)
        ax.add_collection(self.update())

        if use_blit:
            # image.set_animated(True)
            self.bars.set_animated(True)

        # set axes limits
        if self.log:
            ax.set(xscale='log', xlim=(0.75, None))
            ax.xaxis.major.formatter = ticker.FuncFormatter(fmt_log_tick)
            ax.format_xdata = self.ax.format_ydata

        # rescale if non-empty histogram
        if len(self.counts):
            self.autoscale_view()

    @property
    def figure(self):
        return self.ax.figure

    @property
    def cmap(self):
        return self._cmap

    @cmap.setter
    def cmap(self, cmap):
        self.set_cmap(cmap)

    def set_cmap(self, cmap, outside_colour=None, outside_alpha=0.75):
        self._cmap = self._compute_cmap(cmap, outside_colour, outside_alpha)
        self.bars.set_cmap(self._cmap)

    def _compute_cmap(self, cmap, outside_colour, outside_alpha):
        # setup colormap (copy)
        self.logger.debug('Adapting cmap {} for {}.', cmap, self)
        cmap = colormaps.get_cmap(cmap)
        cmap = ListedColormap(cmap(np.linspace(0, 1, 256)))

        # optionally gray out out-of-bounds values
        if outside_colour is None:
            outside_colours = cmap([0., 1.])  # note float
            outside_colours[:, -1] = outside_alpha
            under, over = outside_colours
        else:
            under, over = duplicate_if_scalar(outside_colour)
        #
        cmap.set_over(over)
        cmap.set_under(under)
        return cmap

    def _update_cmap_from_image(self, image):
        self.set_cmap(image.get_cmap(),
                      self._outer_style['facecolor'], 
                      self._outer_style['alpha'])

    def get_array(self):
        return self.image.get_array()

    def set_array(self, data):  # set_image
        # compute histogram
        self.compute(data)
        self.update()

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
        clim = self.image.get_clim()
        outside = ~np.ma.masked_inside(self.bin_centers, *clim).mask
        self.bars.set(
            verts=self.get_verts(self.counts, self.bin_edges),
            linewidth=outside * self._outer_style['linewidth'],
            edgecolor=self._outer_style['edgecolor'],
            array=self.bin_centers,
            clim=clim
        )

        # draw_list = [self.bars]

        # Axes grids. this doesn't work..
        # for axis in (self.ax.xaxis, self.ax.yaxis):
        #     z = axis.get_zorder()
        #     if axis._major_tick_kw['gridOn']:
        #         for t in axis.majorTicks:
        #             t.gridline.set_zorder(z)
        #             draw_list.append((t.gridline for t in axis.majorTicks))

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
        vmin, vmax = self.image.get_clim()
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

        # self.logger.debug('Ax lims: ({:.1f}, {:.1f}).', *lim)
        self.ax.set(xlim=xlim, ylim=ylim)
