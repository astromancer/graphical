"""
Routines for displaying images and video
"""

# std
import time
import warnings
import itertools as itt

# third-party
import numpy as np
import matplotlib.pylab as plt
from matplotlib import ticker
from matplotlib.figure import Figure
from matplotlib.widgets import Slider
from matplotlib.gridspec import GridSpec
from loguru import logger
from mpl_toolkits.mplot3d import art3d
from mpl_toolkits.axes_grid1 import AxesGrid, make_axes_locatable
from astropy.visualization.stretch import BaseStretch
from astropy.visualization.interval import BaseInterval
from astropy.visualization.mpl_normalize import ImageNormalize

# local
from recipes.functionals import echo0
from recipes.dicts import AttrReadItem
from recipes.logging import LoggingMixin
from recipes.misc import duplicate_if_scalar
from recipes.array.neighbours import neighbours

# relative
from .bar3d import bar3d
from .sliders import TripleSliders
from .utils import get_percentile_limits
from .connect import ConnectionMixin, mpl_connect


# from obstools.aps import ApertureCollection

# from .zscale import zrange

# from astropy.visualization import mpl_normalize  # import ImageNormalize as _

# TODO: docstrings (when stable)
# TODO: unit tests
# TODO: maybe display things like contrast ratio ??

# TODO: middle mouse resets axes limits


def _sanitize_data(data):
    """
    Removes nans and masked elements
    Returns flattened array
    """
    if np.ma.is_masked(data):
        data = data[~data.mask]
    return np.asarray(data[~np.isnan(data)])


def move_axes(ax, x, y):
    """Move the axis in the figure by x, y"""
    l, b, w, h = ax.get_position(True).bounds
    ax.set_position((l + x, b + y, w, h))


def get_norm(image, interval, stretch):
    """

    Parameters
    ----------
    image
    interval
    stretch

    Returns
    -------

    """
    # choose colour interval algorithm based on data type
    if image.dtype.kind == 'i':  # integer array
        if image.ptp() < 1000:
            interval = 'minmax'

    # determine colour transform from `interval` and `stretch`
    if isinstance(interval, str):
        interval = interval,
    interval = Interval.from_name(*interval)
    #
    if isinstance(stretch, str):
        stretch = stretch,
    stretch = Stretch.from_name(*stretch)

    # Create an ImageNormalize object
    return ImageNormalize(image, interval, stretch=stretch)


def get_screen_size_inches():
    """
    Use QT to get the size of the primary screen in inches

    Returns
    -------
    size_inches: list

    """
    import sys
    from PyQt5.QtWidgets import QApplication, QDesktopWidget

    # Note the check on QApplication already running and not executing the exit
    #  statement at the end.
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    else:
        logger.debug(f'QApplication instance already exists: {app}')

    # TODO: find out on which screen the focus is

    w = QDesktopWidget()
    s = w.screen()
    size_inches = [s.width() / s.physicalDpiX(), s.height() / s.physicalDpiY()]
    # app.exec_()
    w.close()
    return size_inches

    # screens = app.screens()
    # size_inches = np.empty((len(screens), 2))
    # for i, s in enumerate(screens):
    #     g = s.geometry()
    #     size_inches[i] = np.divide(
    #             [g.height(), g.width()], s.physicalDotsPerInch()
    #     )
    # app.exec_()
    # return size_inches


def guess_figsize(image, fill_factor=0.75, max_pixel_size=0.2):
    """
    Make an educated guess of the size of the figure needed to display the
    image data.

    Parameters
    ----------
    image: np.ndarray
        Sample image
    fill_factor: float
        Maximal fraction of screen size allowed in any direction
    min_size: 2-tuple
        Minimum allowed size (width, height) in inches
    max_pixel_size: float
        Maximum allowed pixel size

    Returns
    -------
    size: tuple
        Size (width, height) of the figure in inches


    """

    # Sizes reported by mpl figures seem about half the actual size on screen
    shape = np.array(np.shape(image)[::-1])
    return _guess_figsize(shape, fill_factor, max_pixel_size)


def _guess_figsize(image_shape, fill_factor=0.75, max_pixel_size=0.2,
                   min_size=(2, 2)):
    # screen dimensions
    screen_size = np.array(get_screen_size_inches())

    # change order of image dimensions since opposite order of screen
    max_size = np.multiply(image_shape, max_pixel_size)

    # get upper limit for fig size based on screen and data and fill factor
    max_size = np.min([max_size, screen_size * fill_factor], 0)

    # get size from data
    aspect = image_shape / image_shape.max()
    size = max_size[aspect == 1] * aspect

    # enlarge =
    size *= max(np.max(min_size / size), 1)

    logger.debug('Guessed figure size: (%.1f, %.1f)', *size)
    return size


def auto_grid(n):
    x = int(np.floor(np.sqrt(n)))
    y = int(np.ceil(n / x))
    return x, y


def get_clim(data, plims=(0.25, 99.75)):
    """
    Get colour scale limits for data.
    """

    if np.all(np.ma.getmask(data)):
        return None, None

    clims = get_percentile_limits(_sanitize_data(data), plims)

    bad_clims = (clims[0] == clims[1])
    if bad_clims:
        logger.warning('Ignoring bad colour interval: (%.1f, %.1f). ', *clims)
        return None, None

    return clims


def set_clim_connected(x, y, artist, sliders):
    artist.set_clim(*sliders.positions)
    return artist


def plot_image_grid(images, layout=(), titles=(), title_kws=None, figsize=None,
                    plims=None, clim_all=False, **kws):
    """

    Parameters
    ----------
    images
    layout
    titles
    clim_all:
        Compute colour limits from the full set of pixel values for all
        images.  Choose this if your images are all normalised to roughly the
        same scale. If False clims will be computed individually and the
        colourbar sliders will be disabled.

    Returns
    -------

    """

    # TODO: plot individual histograms - clim_each
    # todo: guess fig size

    n = len(images)
    assert n, 'No images to plot!'
    # assert clim_mode in ('all', 'row')

    # get grid layout
    if not layout:
        layout = auto_grid(n)
    n_rows, n_cols = layout
    if n_rows == -1:
        n_rows = int(np.ceil(n / n_cols))
    if n_cols == -1:
        n_cols = int(np.ceil(n / n_rows))

    # create figure
    fig = plt.figure(figsize=figsize)

    # ticks
    tick_par = dict(color='w', direction='in',
                    bottom=1, top=1, left=1, right=1)

    # Use gridspec rather than ImageGrid since the latter tends to resize
    # the axes
    if clim_all:
        cbar_size, hist_size = 3, 5
    else:
        cbar_size = hist_size = 0

    gs = GridSpec(n_rows, n_cols * (100 + cbar_size + hist_size),
                  hspace=0.005,
                  wspace=0.005,
                  left=0.03,  # fixme: need more for ticks
                  right=0.97,
                  bottom=0.03,
                  top=0.98
                  )  # todo: maybe better with tight layout.

    # create colourbar and pixel histogram axes

    #
    kws = {**dict(origin='lower',
                  cbar=False, sliders=False, hist=False,
                  clim=not clim_all,
                  plims=plims),
           **kws}
    title_kws = {**dict(color='w',
                        va='top',
                        fontweight='bold'),
                 **(title_kws or {})}

    art = []
    w = len(str(int(n)))
    axes = np.empty((n_rows, n_cols), 'O')
    indices = enumerate(np.ndindex(n_rows, n_cols))
    for (i, (j, k)), title in itt.zip_longest(indices, titles, fillvalue=''):
        if i == n:
            break

        # last
        if (i == n - 1) and clim_all:
            # do colourbar + pixel histogram if clim all
            kws.update(cbar=True, sliders=True, hist=True,
                       cax=fig.add_subplot(
                           gs[:, -(cbar_size + hist_size) * n_cols:]),
                       hax=fig.add_subplot(gs[:, -hist_size * n_cols:]))

        # create axes!
        axes[j, k] = ax = fig.add_subplot(
            gs[j:j + 1, (100 * k):(100 * (k + 1))])

        # plot image
        imd = ImageDisplay(images[i], ax=ax, **kws)
        art.append(imd.imagePlot)

        # do ticks
        top = (j == 0)
        bot = (j == n_rows - 1)
        left = (k == 0)  # leftmost
        # right = (j == n_cols - 1)

        # set the ticks to white and visible on all spines for aesthetic
        ax.tick_params('both', **{**dict(labelbottom=bot, labeltop=top,
                                         labelleft=left, labelright=0),
                                  **tick_par})

        for lbl, spine in ax.spines.items():
            spine.set_color('w')

        # add title text
        title = title.replace("\n", "\n     ")
        ax.text(0.025, 0.95, f'{i: <{w}}: {title}',
                transform=ax.transAxes, **title_kws)

    # Do colorbar
    # fig.colorbar(imd.imagePlot, cax)
    img = ImageGrid(fig, axes, imd)
    if clim_all:
        img._clim_all(images, plims)
    return img


class ImageGrid:
    def __init__(self, fig, axes, imd):
        self.fig = fig
        self.axes = axes
        self.imd = imd

    def __iter__(self):
        yield from (self.fig, self.axes, self.imd)

    def save(self, filenames):
        from matplotlib.transforms import Bbox

        fig = self.fig

        assert len(filenames) == self.axes.size

        ax_per_image = (len(fig.axes) // self.axes.size)
        # axit = mit.chunked(self.fig.axes, ax_per_image)

        for ax, name in zip(self.axes.ravel(), filenames):
            # mn, mx = (np.inf, np.inf), (0, 0)
            # for ax in axes[::-1]:
            #     # Save just the portion _inside_ the second axis's boundaries
            #     mn1, mx1 = ax.get_window_extent().transformed(
            #         fig.dpi_scale_trans.inverted()).get_points()
            #     mn = np.min((mn, mn1), 0)
            #     mx = np.max((mx, mx1), 0)

            #     ticklabels = ax.xaxis.get_ticklabels() + ax.yaxis.get_ticklabels()
            #     for txt in ticklabels:
            #         mn1, mx1 = txt.get_window_extent().transformed(
            #         fig.dpi_scale_trans.inverted()).get_points()
            #         mn = np.min((mn, mn1), 0)
            #         mx = np.max((mx, mx1), 0)

            # remove ticks
            # ax.set_axis_off()
            if len(ax.texts):
                ax.texts[0].set_visible(False)

            # Pad the saved area by 10% in the x-direction and 20% in the y-direction
            fig.savefig(name, bbox_inches=ax.get_window_extent().transformed(
                fig.dpi_scale_trans.inverted()).expanded(1.2, 1))

    @property
    def images(self):
        return [ax.images[0].get_array() for ax in self.fig.axes]

    def _clim_all(art, imd, images, plims):
        # connect all image clims to the sliders.
        for image in art:
            # noinspection PyUnboundLocalVariable
            imd.sliders.lower.on_move.add(set_clim_connected, image,
                                          imd.sliders)
            imd.sliders.upper.on_move.add(set_clim_connected, image,
                                          imd.sliders)

        # The same as above can be accomplished in pure matplolib as follows:
        # https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/multi_image.html
        # Make images respond to changes in the norm of other images (e.g. via
        # the "edit axis, curves and images parameters" GUI on Qt), but be
        # careful not to recurse infinitely!
        # def update(changed_image):
        #     for im in art:
        #         if (changed_image.get_cmap() != im.get_cmap()
        #                 or changed_image.get_clim() != im.get_clim()):
        #             im.set_cmap(changed_image.get_cmap())
        #             im.set_clim(changed_image.get_clim())
        #
        # for im in art:
        #     im.callbacksSM.connect('changed', update)

        # update clim for all plots

        # for the general case where images are non-uniform shape, we have to
        # flatten them all to get the colour percentile values.
        # TODO: will be more efficient for large number of images to sample
        #  evenly from each image
        pixels = []
        for im in images:
            # getattr(im, ('ravel', 'compressed')[np.ma.isMA(im)])()
            pixels.extend(im.compressed() if np.ma.isMA(im) else im.ravel())
        pixels = np.array(pixels)

        clim = self.imd.clim_from_data(pixels, plims=plims)
        self.imd.sliders.set_positions(clim, draw_on=False)  # no canvas yet!

        # Update histogram with data from all images
        self.imd.histogram.set_array(pixels)
        self.imd.histogram.autoscale_view()


class FromNameMixin:
    @classmethod
    def from_name(cls, method, *args, **kws):
        """
        Construct derived subtype from `method` string and `kws`
        """

        from recipes.oo import iter_subclasses

        if not isinstance(method, str):
            raise TypeError('method should be a string.')

        allowed_names = set()
        for sub in iter_subclasses(cls.__bases__[0]):
            name = sub.__name__
            if name.lower().startswith(method.lower()):
                break
            else:
                allowed_names.add(name)

        else:
            raise ValueError('Unrecognized method %r. Please use one of '
                             'the following %s' %
                             (method, tuple(allowed_names)))

        return sub(*args, **kws)


class Interval(BaseInterval, FromNameMixin):
    def get_limits(self, values):
        print('hi')  # FIXME: this is missed
        return BaseInterval.get_limits(self, _sanitize_data(values))


class Stretch(BaseStretch, FromNameMixin):
    pass


# class ImageNormalize(mpl_normalize.ImageNormalize):
#
#     # FIXME: ImageNormalize fills masked arrays with vmax instead of removing
# them.
#     # this skews the colour distribution.  TODO: report bug
#
#     def __init__(self, data=None, *args, **kws):
#         if data is not None:
#             data = _sanitize_data(data)
#
#         mpl_normalize.ImageNormalize.__init__(self, data, *args, **kws)
#
#     def __call__(self, values, clip=None):
#         return mpl_normalize.Normalize.__call__(
#                 self, _sanitize_data(values), clip)


# from .hist import Histogram
class ColourBarHistogram(LoggingMixin):  # PixelHistogram
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
        from matplotlib.colors import ListedColormap

        self.log = kws.pop('log', True)
        self.ax = ax
        self.image_plot = image_plot
        self.norm = image_plot.norm

        assert orientation.lower().startswith(('h', 'v'))
        self.orientation = orientation

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
            ax.set_xscale('log')

        # rescale if non-empty histogram
        if len(self.counts):
            self.autoscale_view()

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
            range = self._auto_range()

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
        xmin = 0.1 if self.log else 0
        xlim = (xmin, self.counts.max())
        ylim = self._auto_range()

        if self.orientation.startswith('v'):
            xlim, ylim = ylim, xlim

        # self.logger.debug('Ax lims: (%.1f, %.1f)', *lim)
        self.ax.set(xlim=xlim, ylim=ylim)


class ImageDisplay(LoggingMixin):
    # TODO: move cursor with arrow keys when hovering over figure (like ds9)
    # TODO: optional zoomed image window

    # FIXME: Dragging too slow for large images: option for update on release
    #  instead of update on drag!!

    # FIXME: hist: outside alpha changes with middle slider move.....

    # TODO: optional Show which region on the histogram corresponds to colorbar
    # TODO: better histogram for integer / binary data with narrow ranges
    # TODO: method pixels corresponding to histogram bin?

    # TODO: remove ticks on cbar ax
    # TODO: plot scale func on hist axis

    sliderClass = TripleSliders  # AxesSliders
    _default_plims = (0.25, 99.75)
    _default_hist_kws = dict(bins=100)

    def __init__(self, image, *args, **kws):
        """

        Parameters
        ----------
        image
        args
        kws


        Keywords
        --------
        cbar
        cax
        hist
        hax
        sliders
        ax

        remaining keywords passed to ax.imshow


        """
        # ax      :       Axes object
        #     Axes on which to display

        self.has_cbar = kws.pop('cbar', True)
        hist_kws = kws.pop('hist', self._default_hist_kws)
        if hist_kws is True:
            hist_kws = self._default_hist_kws
        self.has_hist = bool(hist_kws)
        self.has_sliders = kws.pop('sliders', True)
        self.use_blit = kws.pop('use_blit', False)
        connect = kws.pop('connect', self.has_sliders)
        # set origin
        kws.setdefault('origin', 'lower')

        # check data
        image = np.ma.asarray(image).squeeze()  # remove redundant dimensions
        if image.ndim != 2:
            msg = f'{self.__class__.__name__} cannot image {image.ndim}D data.'
            if image.ndim == 3:
                msg += 'Use `VideoDisplay` class to image 3D data.'
            raise ValueError(msg)

        # convert boolean to integer (for colour scale algorithm)
        if image.dtype.name == 'bool':
            image = image.astype(int)

        self.data = image
        self.ishape = self.data.shape

        # create the figure if needed
        self.divider = None
        self.figure, axes = self.init_figure(kws)
        self.ax, self.cax, self.hax = axes
        ax = self.ax

        # use imshow to do the plotting
        self.clim_from_data(image, kws)

        self.imagePlot = ax.imshow(image, *args, **kws)
        self.norm = self.imagePlot.norm
        # self.imagePlot.set_clim(*clim)

        # create the colourbar / histogram / sliders
        self.cbar = None
        if self.has_cbar:
            self.cbar = self.make_cbar()

        # create sliders after histogram so they display on top
        self.sliders, self.histogram = self.make_sliders(hist_kws)

        # connect on_draw for debugging
        self._draw_count = 0
        # self.cid = ax.figure.canvas.mpl_connect('draw_event', self._on_draw)

        if connect:
            self.connect()

    def __iter__(self):
        yield self.figure
        yield self.ax

    # def _clean_kws(self):
    #     s = inspect.signature(self.ax.imshow)
    #     set(s.parameters.keys()) - {'X', 'data', 'kwargs'}

    def init_figure(self, kws):
        """
        Create the figure and add the axes

        Parameters
        ----------
        kws

        Returns
        -------

        """
        # note intentionally not unpacking keyword dict so that the keys
        #  removed here reflect at the calling scope
        ax = kws.pop('ax', None)
        cax = kws.pop('cax', None)
        hax = kws.pop('hax', None)
        title = kws.pop('title', None)
        figsize = kws.pop('figsize', 'auto')
        # sidebar = kws.pop('sidebar', True)

        # create axes if required
        if ax is None:
            if figsize == 'auto':
                # automatically determine the figure size based on the data
                figsize = self.guess_figsize(self.data)
                # FIXME: the guessed size does not account for the colorbar
                #  histogram

            fig = plt.figure(figsize=figsize)
            self._gs = gs = GridSpec(1, 1,
                                     left=0.05, right=0.95,
                                     top=0.98, bottom=0.05, )
            ax = fig.add_subplot(gs[0, 0])
            ax.tick_params('x', which='both', top=True)

        # axes = namedtuple('AxesContainer', ('image',))(ax)
        if self.has_cbar and (cax is None):
            self.divider = make_axes_locatable(ax)
            cax = self.divider.append_axes('right', size=0.2, pad=0)

        if self.has_hist and (hax is None):
            hax = self.divider.append_axes('right', size=1, pad=0.2)

        # set the axes title if given
        if title is not None:
            ax.set_title(title)

        # setup coordinate display
        ax.format_coord = self.format_coord
        ax.grid(False)
        return ax.figure, (ax, cax, hax)

    def guess_figsize(self, data, fill_factor=0.75, max_pixel_size=0.2):
        """
        Make an educated guess of the size of the figure needed to display the
        image data.

        Parameters
        ----------
        data: np.ndarray
            Sample image
        fill_factor: float
            Maximal fraction of screen size allowed in any direction
        max_pixel_size: 2-tuple
            Maximum allowed pixel size (heigh, width) in inches

        Returns
        -------
        size: tuple
            Size (width, height) of the figure in inches


        """

        #
        image = self.data[0] if data is None else data
        return guess_figsize(image, fill_factor, max_pixel_size)

    def make_cbar(self):
        fmt = None
        if self.has_hist:
            # No need for the data labels on the colourbar since it will be on
            # the histogram axis.
            from matplotlib import ticker
            fmt = ticker.NullFormatter()

        # cax = self.axes.cbar
        cbar = self.figure.colorbar(self.imagePlot, cax=self.cax, format=fmt)
        return cbar

    def make_sliders(self, hist_kws=None):
        # data = self.imagePlot.get_array()

        # FIXME: This is causing recursive repaint!!

        sliders = None
        if self.has_sliders:
            clim = self.imagePlot.get_clim()
            sliders = self.sliderClass(self.hax, clim, 'y',
                                       color='rbg',
                                       ms=(2, 1, 2),
                                       extra_markers='>s<')

            sliders.lower.on_move.add(self.update_clim)
            sliders.upper.on_move.add(self.update_clim)

        cbh = None
        if self.has_hist:
            cbh = ColourBarHistogram(self.hax, self.imagePlot, 'horizontal',
                                     self.use_blit, **hist_kws)

            # set ylim if reasonable to do so
            # if data.ptp():
            #     # avoid warnings in setting upper/lower limits identical
            #     hax.set_ylim((data.min(), data.max()))
            # NOTE: will have to change with different orientation

            self.hax.yaxis.tick_right()
            self.hax.grid(True)
        return sliders, cbh

    def clim_from_data(self, data, kws=None, **kws_):
        """
        Get colour scale limits for data.
        """
        # first arg is dict from which we remove 'extra' keywords that
        # are not allowed in imshow. This allows a dict from the calling
        # scope to be edited here without global statement.
        # This function can also still be used with unpacked keyword args
        kws = kws or {}
        clim = kws.pop('clim', True)
        plims = kws.pop('plims', None)
        if clim:
            plims = kws_.setdefault('plims', plims)
            if plims is None:
                kws_['plims'] = self._default_plims

            if np.all(np.ma.getmask(data)):
                return None, None

            clims = get_percentile_limits(_sanitize_data(data), **kws_)
            self.logger.debug('Colour limits: ({:.1f}, {:.1f})', *clims)
            kws['vmin'], kws['vmax'] = clims

            bad_clims = (clims[0] == clims[1])
            if bad_clims:
                self.logger.warning('Bad colour interval: (%.1f, %.1f). '
                                    'Ignoring', *clims)
                return None, None

            return clims
        return None, None

    def set_clim(self, *clim):
        self.imagePlot.set_clim(*clim)

        if not self.has_hist:
            return self.imagePlot

        self.histogram.update()
        self.sliders.min_span = (clim[0] - clim[1]) / 100

        # TODO: return COLOURBAR ticklabels?
        return self.imagePlot, self.histogram.bars

    def update_clim(self, *xydata):
        """Set colour limits on slider move"""
        return self.set_clim(*self.sliders.positions)

    def _on_draw(self, event):
        self.logger.debug('DRAW %i', self._draw_count)  # ,  vars(event)
        if self._draw_count == 0:
            self._on_first_draw(event)
        self._draw_count += 1

    def _on_first_draw(self, event):
        self.logger.debug('FIRST DRAW')

    def format_coord(self, x, y, precision=3, masked_str='masked'):
        """
        Create string representation for cursor position in data coordinates

        Parameters
        ----------
        x: float
            x data coordinate position
        y: float
            y data coordinate position
        precision: int, optional
            decimal precision for string format
        masked_str: str, optional
            representation for masked elements in array
        Returns
        -------
        str
        """

        # MASKED_STR = 'masked'

        if not hasattr(self, 'imagePlot'):
            # prevents a swap of repeated errors flooding the terminal for
            # mouse over when no image has been drawn
            return 'no image'

        # xy repr
        xs = 'x=%1.{:d}f'.format(precision) % x
        ys = 'y=%1.{:d}f'.format(precision) % y
        # z
        col, row = int(x + 0.5), int(y + 0.5)
        nrows, ncols = self.ishape
        if (0 <= col < ncols) and (0 <= row < nrows):
            data = self.imagePlot.get_array()
            z = data[row, col]
            # handle masked data
            if np.ma.is_masked(z):
                # prevent Warning: converting a masked element to nan.
                zs = 'z=%s' % masked_str
            else:
                zs = 'z=%1.{:d}f'.format(precision) % z
            return ',\t'.join((xs, ys, zs)).expandtabs()
        else:
            return ', '.join((xs, ys))

    def connect(self):
        if self.sliders is None:
            return

        # connect sliders' interactions + save background for blitting
        self.sliders.connect()

    def save(self, filename, *args, **kws):
        self.figure.savefig(filename, *args, **kws)


# class AstroImageDisplay(ImageDisplay):

#     def clim_from_data(self, data, kws):
#         # colour transform / normalize
#         interval = kws.pop('interval', 'zscale')
#         stretch = kws.pop('stretch', 'linear')

#         # note: ImageNormalize fills masked values.. this is clearly WRONG
#         #  since it will skew statistics on the image. important that data
#         #  passed to this function has been cleaned of masked values
#         # HACK: get limits ignoring masked pixels
#         #         # set the slider positions / color limits

#         self.norm = get_norm(data, interval, stretch)
#         # kws['norm'] = norm
#         clim = self.norm.interval.get_limits(data)
#         return clim


class VideoDisplay(ImageDisplay):
    # FIXME: blitting not working - something is leading to auto draw
    # FIXME: frame slider bar not drawing on blit
    # FIXME: HISTOGRAM values not updating on scroll
    # TODO: lock the sliders in place with button??

    _scroll_wrap = True  # scrolling past the end leads to the beginning

    def _check_data(self, data):
        if not isinstance(data, np.ndarray):
            data = np.ma.asarray(data)

        n_dim = data.ndim
        if n_dim == 2:
            warnings.warn('Loading single image frame as 3D data cube. Use '
                          '`ImageDisplay` instead to view single frames.')
            data = np.ma.atleast_3d(data)

        if n_dim != 3:
            raise ValueError(f'Cannot image {n_dim}D data')
        return data, len(data)

    def __init__(self, data, **kws):
        """
        Image display for 3D data. Implements frame slider and image scroll.

        subclasses optionally implement `update` method

        Parameters
        ----------
        data:       np.ndarray or np.memmap
            initial display data

        clim_every: int
            How frequently to re-run the color normalizer algorithm to set
            the colour limits. Setting this to `False` may have a positive
            effect on performance.

        kws are passed directly to ImageDisplay.
        """

        #
        data, nframes = self._check_data(data)

        self.nframes = int(nframes)
        self.clim_every = kws.pop('clim_every', 1)

        # don't connect methods yet
        connect = kws.pop('connect', True)

        # setup image display
        # parent sets data as 2D image.
        n = self._frame = 0
        ImageDisplay.__init__(self, data[n], connect=False, **kws)
        # save data (this can be array_like (or np.mmap))
        self.data = data

        # make observer container for scroll
        # self.on_scroll = Observers()

        # make frame slider
        fsax = self.divider.append_axes('bottom', size=0.1, pad=0.3)
        self.frameSlider = Slider(fsax, 'frame', n, self.nframes, valfmt='%d')
        self.frameSlider.on_changed(self.update)
        fsax.xaxis.set_major_locator(ticker.AutoLocator())

        if self.use_blit:
            self.frameSlider.drawon = False

        # # save background for blitting
        # self.background = self.figure.canvas.copy_from_bbox(
        #     self.ax.bbox)

        if connect:
            self.connect()

    def connect(self):
        ImageDisplay.connect(self)

        # enable frame scroll
        self.figure.canvas.mpl_connect('scroll_event', self._scroll)

    # def init_figure(self, **kws):
    #     fig, ax = ImageDisplay.init_figure(self, **kws)
    #     return fig, ax

    # def init_axes(self, fig):
    #     gs = GridSpec(100, 100,
    #                   left=0.05, right=0.95,
    #                   top=0.98, bottom=0.05,
    #                   hspace=0, wspace=0)
    #     q = 97
    #     return self._init_axes(fig,
    #                            image=gs[:q, :80],
    #                            cbar=gs[:q, 80:85],
    #                            hbar=gs[:q, 87:],
    #                            fslide=gs[q:, :80])

    def guess_figsize(self, data, fill_factor=0.55, max_pixel_size=0.2):
        # TODO: inherit docstring
        size = super().guess_figsize(data, fill_factor, max_pixel_size)
        # create a bit more space below the figure for the frame nr indicator
        size[1] += 0.5
        self.logger.debug('Guessed figure size: (%.1f, %.1f)', *size)
        return size

    @property
    def frame(self):
        """Index of image currently being displayed"""
        return self._frame

    @frame.setter
    def frame(self, i):
        """Set frame data respecting scroll wrap"""
        self.set_frame(i)

    def set_frame(self, i):
        """
        Set frame data respecting scroll wrap
        """
        # wrap scrolling if desired
        if self._scroll_wrap:
            # wrap around! scroll past end ==> go to beginning
            i %= self.nframes
        else:  # stop scrolling at the end
            i = max(i, self.nframes)

        i = int(round(i, 0))  # make sure we have an int
        self._frame = i  # store current frame

    def get_image_data(self, i):
        """
        Get the image data to be displayed.

        Parameters
        ----------
        i: int
            Frame number

        Returns
        -------
        np.ndarray

        """
        return self.data[i]

    def update(self, i, draw=True):
        """
        Display the image associated with index `i` frame in the sequence. This
        method can be over-written to change image switching behaviour
        in subclasses.

        Parameters
        ----------
        i: int
            Frame index
        draw: bool
            Whether the canvas should be redrawn after the data is updated


        Returns
        -------
        draw_list: list
            list of artists that have been changed and need to be redrawn
        """
        self.set_frame(i)

        image = self.get_image_data(self.frame)
        # set the image data
        # TODO: method set_image_data here??
        self.imagePlot.set_data(image)  # does not update normalization

        # FIXME: normalizer fails with boolean data
        #  File "/usr/local/lib/python3.6/dist-packages/matplotlib/colorbar.py", line 956, in on_mappable_changed
        #   self.update_normal(mappable)
        # File "/usr/local/lib/python3.6/dist-packages/matplotlib/colorbar.py", line 987, in update_normal

        draw_list = [self.imagePlot]

        # set the slider axis limits
        if self.sliders:
            # find min / max as float
            min_max = float(np.nanmin(image)), float(np.nanmax(image))
            if not np.isnan(min_max).any():
                self.sliders.ax.set_ylim(min_max)
                self.sliders.valmin, self.sliders.valmax = min_max

                # since we changed the axis limits, need to redraw tick labels
                draw_list.extend(
                    getattr(self.histogram.ax,
                            f'get_{self.sliders.slide_axis}ticklabels')())

        # update histogram
        if self.has_hist:
            self.histogram.compute(image, self.histogram.bin_edges)
            draw_list.append(self.histogram.update())

        if not (self._draw_count % self.clim_every):
            # set the slider positions / color limits
            vmin, vmax = self.clim_from_data(image)
            self.imagePlot.set_clim(vmin, vmax)

            if self.sliders:
                draw_list = self.sliders.set_positions((vmin, vmax),
                                                       draw_on=False)

            # set the axes limits slightly wider than the clims
            if self.has_hist:
                self.histogram.autoscale_view()

            # if getattr(self.norm, 'interval', None):
            #     vmin, vmax = self.norm.interval.get_limits(
            #             _sanitize_data(image))

            # else:
            #     self.logger.debug('Auto clims: (%.1f, %.1f)', vmin, vmax)

        #
        if draw:
            self.sliders.draw(draw_list)

        return draw_list
        # return i, image

    def _scroll(self, event):

        # FIXME: drawing on scroll.....
        # try:
        inc = [-1, +1][event.button == 'up']
        new = self._frame + inc
        if self.use_blit:
            self.frameSlider.drawon = False
        self.frameSlider.set_val(new)  # calls connected `update`
        self.frameSlider.drawon = True

        # except Exception as err:
        #     self.logger.exception('Scroll failed:')

    def play(self, start=None, stop=None, pause=0):
        """
        Show a video of images in the stack

        Parameters
        ----------
        n: int
            number of frames in the animation
        pause: int
            interval between frames in milliseconds

        Returns
        -------

        """

        if stop is None and start:
            stop = start
            start = 0
        if start is None:
            start = 0
        if stop is None:
            stop = self.nframes

        # save background for blitting
        # FIXME: saved bg should be without
        tmp_inviz = [self.frameSlider.poly, self.frameSlider.valtext]
        # tmp_inviz.extend(self.histogram.ax.yaxis.get_ticklabels())
        tmp_inviz.append(self.histogram.bars)
        for s in tmp_inviz:
            s.set_visible(False)

        fig = self.figure
        fig.canvas.draw()
        self.background = fig.canvas.copy_from_bbox(self.figure.bbox)

        for s in tmp_inviz:
            s.set_visible(True)

        self.frameSlider.eventson = False
        self.frameSlider.drawon = False

        # pause: inter-frame pause (millisecond)
        seconds = pause / 1000
        i = int(start)

        # note: the fastest frame rate achievable currently seems to be
        #  around 20 fps
        try:
            while i <= stop:
                self.frameSlider.set_val(i)
                draw_list = self.update(i)
                draw_list.extend([self.frameSlider.poly,
                                  self.frameSlider.valtext])

                fig.canvas.restore_region(self.background)

                # FIXME: self.frameSlider.valtext doesn't dissappear on blit

                for art in draw_list:
                    self.ax.draw_artist(art)

                fig.canvas.blit(fig.bbox)

                i += 1
                time.sleep(seconds)
        except Exception as err:
            raise err
        finally:
            self.frameSlider.eventson = True
            self.frameSlider.drawon = True

    # def blit_setup(self):

    # @expose.args()
    # def draw_blit(self, artists):
    #
    #     self.logger.debug('draw_blit')
    #
    #     fig = self.figure
    #     fig.canvas.restore_region(self.background)
    #
    #     for art in artists:
    #         try:
    #             self.ax.draw_artist(art)
    #         except Exception as err:
    #             self.logger.debug('drawing FAILED %s', art)
    #             traceback.print_exc()
    #
    #     fig.canvas.blit(fig.bbox)

    # def format_coord(self, x, y):
    #     s = ImageDisplay.format_coord(self, x, y)
    #     return 'frame %d: %s' % (self.frame, s)

    # def format_coord(self, x, y):
    #     col, row = int(x + 0.5), int(y + 0.5)
    #     nrows, ncols, _ = self.data.shape
    #     if (col >= 0 and col < ncols) and (row >= 0 and row < nrows):
    #         z = self.data[self._frame][row, col]
    #         return 'x=%1.3f,\ty=%1.3f,\tz=%1.3f' % (x, y, z)
    #     else:
    #         return 'x=%1.3f, y=%1.3f' % (x, y)


class VideoDisplayX(VideoDisplay):
    # FIXME: redraw markers after color adjust
    # TODO: improve memory performance by allowing coords to update via func

    marker_properties = dict(c='r', marker='x', alpha=1, ls='none', ms=5)

    def __init__(self, data, coords=None, **kws):
        """

        Parameters
        ----------
        data: array-like
            Image stack. shape (N, ypix, xpix)
        coords:  array_like, optional
            coordinate positions (yx) of apertures to display. This must be
            array_like with
            shape (N, k, 2) where k is the number of apertures per frame, and N
            is the number of frames.
        kws:
            passed to `VideoDisplay`
        """

        VideoDisplay.__init__(self, data, **kws)

        # create markers
        self.marks, = self.ax.plot([], [], **self.marker_properties)

        # check coords
        self.coords = coords
        self.has_coords = (coords is not None)
        if self.has_coords:
            coords = np.asarray(coords)
            if coords.ndim not in (2, 3) or (coords.shape[-1] != 2):
                raise ValueError('Coordinate array has incorrect shape: %s',
                                 coords.shape)
            if coords.ndim == 2:
                # Assuming single coordinate point per frame
                coords = coords[:, None]
            if len(coords) < len(data):
                self.logger.warning(
                    'Coordinate array contains fewer points (%i) than '
                    'the number of frames (%i).', len(coords), len(data))

            # set for frame 0
            self.marks.set_data(coords[0, :, ::-1].T)
            self.get_coords = self.get_coords_internal

    def get_coords(self, i):
        return

    def get_coords_internal(self, i):
        i = int(round(i))
        return self.coords[i, :, ::-1].T

    def update(self, i, draw=True):
        # self.logger.debug('update')
        # i = round(i)
        draw_list = VideoDisplay.update(self, i, False)
        #
        coo = self.get_coords(i)
        if coo is not None:
            self.marks.set_data(coo)
            draw_list.append(self.marks)

        return draw_list


class VideoDisplayA(VideoDisplayX):
    # default aperture properties
    apProps = dict(ec='m', lw=1,
                   picker=False,
                   widths=7.5, heights=7.5)

    def __init__(self, data, coords=None, ap_props={}, **kws):
        """
        Optionally also displays apertures if coordinates provided.
        """
        VideoDisplayX.__init__(self, data, coords, **kws)

        # create apertures
        props = VideoDisplayA.apProps.copy()
        props.update(ap_props)
        self.aps = self.create_apertures(**props)

    def create_apertures(self, **props):
        props.setdefault('animated', self.use_blit)
        aps = ApertureCollection(**props)
        # add apertures to axes.  will not display yet if coordinates not given
        aps.add_to_axes(self.ax)
        return aps

    def update_apertures(self, i, *args, **kws):
        coords, *_ = args
        self.aps.coords = coords
        return self.aps

    def update(self, i, draw=True):
        # get all the artists that changed by calling parent update
        draw_list = VideoDisplay.update(self, i, False)
        #
        coo = self.get_coords(i)
        if coo is not None:
            self.marks.set_data(coo)
            draw_list.append(self.marks)

        art = self.update_apertures(i, coo.T)
        draw_list.append(art)

        return draw_list

        # self.ap_updater(self.aps, i)
        # self.aps.coords = coo.T

        # except Exception as err:
        #     self.logger.exception('Aperture update failed at %i', i)
        #     self.aps.coords = np.empty((0, 2))
        # else:
        # draw_list.append(self.aps)
        # finally:
        # return draw_list


DEFAULT_TITLES = ('Data', 'Fit', 'Residual')


class ImageModelPlot3D(ConnectionMixin):
    """
    Base class for plotting image data, model and residual for comparison.
    """
    # TODO: profile & speed up!
    # TODO: blit for view angle change...
    # TODO: optionally Include info as text in figure??????

    images_axes_kws = dict(nrows_ncols=(1, 3),
                           axes_pad=0.1,
                           label_mode='L',  # THIS DOESN'T WORK!
                           # share_all = True,
                           cbar_location='right',
                           cbar_mode='each',
                           cbar_size='12%',
                           cbar_pad='0%')

    _3d_axes_kws = dict(azim=-125, elev=30)

    colorbar_ticks = AttrReadItem(
        major=(major := dict(axis='y',
                             which='major',
                             colors='orangered',
                             direction='in',
                             labelsize=10,
                             pad=-11,
                             length=4)),
        minor={**major,
               **dict(which='minor',
                      length=2)},
        right={**major,
               **dict(pad=10,
                      direction='inout')}
    )

    # @profile()

    def __init__(self, x=(), y=(), z=(), data=(),
                 fig=None, titles=DEFAULT_TITLES,
                 image_kws=(), art3d_kws=(), residual_funcs=(),
                 **kws):

        self.art3d = []
        self.images = []
        self.titles = list(titles)
        self.fig = self.setup_figure(fig, **kws)

        self.residual_funcs = dict(residual_funcs)
        #
        self.update(x, y, z, data, image_kws, art3d_kws)

        # link viewlims of the 3d axes
        ConnectionMixin.__init__(self, self.fig.canvas)

    def __call__(self,  x, y, z, data):
        """
        Plot the data.

        Parameters
        ----------
        (x, y, z, data): np.ndarray
            xy-grid, model, data to plot.
        """
        self.update(x, y, z, data)

    # @unhookPyQt

    def setup_figure(self, fig=None, **kws):
        """
        Initialize grid of 2x3 subplots. Top 3 are colour images, bottom 3 are
        3D wireframe plots of data, fit and residual.
        """

        # Plots for current fit
        fig = fig or plt.figure(**kws)
        # gridpec_kw=dict(left=0.05, right=0.95,
        #                 top=0.98, bottom=0.01))
        if not isinstance(fig, Figure):
            raise TypeError(f'Expected Figure, received {type(fig)}')

        self.axes_images = self.setup_image_axes(fig)
        self.axes_3d = self.setup_3d_axes(fig)

        # fig.suptitle('PSF Fitting')

        return fig

    def setup_3d_axes(self, fig):
        # Create the plot grid for the 3D plots
        # axes_3d = AxesGrid(fig, 212, **self._3d_axes_kws)
        axes_3d = []
        for i in range(4, 7):
            ax = fig.add_subplot(2, 3, i, projection='3d', 
                                 **self._3d_axes_kws)
            ax.set_facecolor('None')
            # ax.patch.set_linewidth( 1 )
            # ax.patch.set_edgecolor( 'k' )
            axes_3d.append(ax)

        return axes_3d

    def setup_image_axes(self, fig):
        # Create the plot grid for the images
        self.axes_images = axes = AxesGrid(fig, 211, **self.images_axes_kws)

        for i, (ax, cax) in enumerate(zip(axes, axes.cbar_axes)):
            # image
            im = ax.imshow(np.empty((1, 1)), origin='lower')
            self.images.append(im)

            # title above image
            ax.set_title(self.titles[i], {'weight': 'bold'},  y=1)

            # colorbar
            cbar = cax.colorbar(im)
            self.setup_cbar_ticks(cbar, cax)

        return axes

    def setup_cbar_ticks(self, cbar, cax):
        # make the colorbar ticks look nice
        
        rightmost = cax is self.axes_images.cbar_axes[-1]
        params = self.colorbar_ticks['right' if rightmost else 'major']
        cax.axes.tick_params(**params)
        cax.axes.tick_params(**self.colorbar_ticks.minor)
        
        # make the colorbar spine invisible
        cbar.outline.set_visible(False)
        #
        for w in ('top', 'bottom', 'right'):
            cax.spines[w].set_visible(True)
            cax.spines[w].set_color(self.colorbar_ticks.major['colors'])
        cax.minorticks_on()

        for t in cax.axes.yaxis.get_ticklabels():
            t.set(weight='bold',
                  ha='center',
                  va='center')

    def get_clim(self, data):
        return get_clim(data)

    def update_images(self, *data, **kws):
        # data, model, residual
        # NOTE: mask shape changes, which breaks things below.
        for image, z in zip(self.images, data):
            image.set_data(z)
            image.update(kws)

    def update_3d(self, *data, **kws):
        raise NotImplementedError()

    def update(self, x, y, z, data, image_props=(), art3d_props=()):
        """Update plots with new data."""
        if x == () or x is None:
            return

        res = data - z
        res_img = self.residual_funcs.get('image', echo0)(res)
        self.update_images(z, data, res_img, **dict(image_props))

        res_3d = self.residual_funcs.get('3d', echo0)(res)
        self.update_3d(x, y, z, data, res_3d, **dict(art3d_props))

        # def set_axes_limits():
        # plims = 0.25, 99.75                       #percentiles
        # clims = np.percentile( data, plims )      #colour limits for data
        # rlims = np.percentile( res, plims )       #colour limits for residuals
        xlims = x[0, [0, -1]]
        ylims = y[[0, -1], 0]
        zlims = [z.min(), z.max()]

        # image colour limits
        rlims = [res_img.min(), res_img.max()]
        clims = self.get_clim(data)
        logger.info('clim {}', clims)
        for im, clim in zip(self.images, (clims, clims, rlims)):
            im.set_clim(clim)
            im.set_extent(np.r_[xlims, ylims])

        # 3D limits
        xr, yr = xlims.ptp(), ylims.ptp()
        rlims = [res_3d.min(), res_3d.max()]
        for ax, zlim in zip(self.axes_3d, (zlims, zlims, rlims)):
            ax.set_zlim(zlim)
            if xr and yr:
                # artificially set axes limits --> applies to all since
                # share_all=True in constructor
                ax.set(xlim=xlims, ylim=ylims)

        # self.fig.canvas.draw()

    @mpl_connect('motion_notify_event')
    def on_move(self, event):
        if ((ax := event.inaxes) not in self.axes_3d
                or ax.button_pressed not in ax._rotate_btn):
            return

        for other in set(self.axes_3d) - {event.inaxes}:
            other.azim = ax.azim
            other.elev = ax.elev


class ImageModelWireframe(ImageModelPlot3D):

    def __init__(self, x=(), y=(), z=(), data=(),
                 fig=None, titles=DEFAULT_TITLES,
                 image_kws=(), art3d_kws=(), **kws):

        super().__init__(fig=fig, titles=titles)

        for ax in self.axes_3d:
            art = art3d.Line3DCollection([])
            ax.add_collection(art)
            self.art3d.append(art)

        self.update(x, y, z, data, art3d_props=kws)

    @staticmethod
    def make_segments(x, y, z):
        """Update segments of wireframe plots."""
        # NOTE: Does not seem to play well with masked data - mask shape changes...
        return [*(xlines := np.r_['-1,3,0', x, y, z]),
                *xlines.transpose(1, 0, 2)]  # swap x-y axes

    def update_3d(self, x, y, z, data, residual, **kws):
        """update plots with new data."""

        # NOTE: mask shape changes, which breaks things below.
        for i, zz in enumerate((data, z, residual)):
            self.art3d[i].set_segments(self.make_segments(x, y, zz))


class ImageModelBar3D(ImageModelPlot3D):

    def update_3d(self, x, y, z, data, residual, **kws):
        """update plots with new data."""
        # TODO: will be much faster to update the  Poly3DCollection verts
        for bars in self.art3d:
            for bar in bars.ravel():
                bar.remove()

        for ax, zz in zip(self.axes_3d, (z, data, abs(residual))):
            bars = bar3d(ax, x, y, zz, **kws)
            self.art3d.append(bars)


class ImageModelContour3D(ImageModelPlot3D):
    def setup_image_axes(self, fig):
        # Create the plot grid for the contour plots
        self.grid_contours = AxesGrid(fig, 212,  # similar to subplot(211)
                                      nrows_ncols=(1, 3),
                                      axes_pad=0.2,
                                      label_mode='L',
                                      # This is necessary to avoid
                                      # AxesGrid._tick_only throwing
                                      share_all=True)

    def update(self, X, Y, Z, data):
        """update plots with new data."""
        res = data - Z
        plots = self.art3d

        for i, (ax, z) in enumerate(zip(self.grid_contours, (data, Z, res))):
            plots[i].set_segments(self.make_segments(X, Y, z))
            cs = ax.contour(X, Y, z)
            ax.clabel(cs, inline=1, fontsize=7)  # manual=manual_locations

        zlims = [Z.min(), Z.max()]
        rlims = [res.min(), res.max()]
        # plims = 0.25, 99.75                       #percentiles
        # clims = np.percentile( data, plims )      #colour limits for data
        # rlims = np.percentile( res, plims )       #colour limits for residuals
        for i, pl in enumerate(plots):
            ax = pl.axes
            ax.set_zlim(zlims if (i + 1) % 3 else rlims)
        ax.set_xlim([X[0, 0], X[0, -1]])
        ax.set_ylim([Y[0, 0], Y[-1, 0]])

        # for i,im in enumerate(images):
        # ax = im.axes
        # im.set_clim( zlims if (i+1)%3 else rlims )
        # artificially set axes limits --> applies to all since share_all=True in constuctor
        # im.set_extent( [X[0,0], X[0,-1], Y[0,0], Y[-1,0]] )

        # self.fig.canvas.draw()


# from recipes.array import ndgrid


class PSFPlotter(ImageModelPlot3D, VideoDisplay):
    def __init__(self, filename, model, params, coords, window, **kws):
        self.model = model
        self.params = params
        self.coords = coords
        self.window = w = int(window)
        self.grid = np.mgrid[:w, :w]
        extent = np.array([0, w, 0, w]) - 0.5  # l, r, b, t

        ImageModelPlot3D.__init__(self)
        axData = self.axes_images[0]

        FitsCubeDisplay.__init__(self, filename, ax=axData, extent=extent,
                                 sidebar=False, figsize=None)
        self.update(0)  # FIXME: full frame drawn instead of zoom
        # have to draw here for some bizarre reason
        # self.axes_images[0].draw(self.fig._cachedRenderer)

    def get_image_data(self, i):
        # coo = self.coords[i]
        data = neighbours(self[i], self.coords[i], self.window)
        return data

    def update(self, i, draw=False):
        """Set frame data. draw if requested """
        i %= len(self)  # wrap around! (eg. scroll past end ==> go to beginning)
        i = int(round(i, 0))  # make sure we have an int
        self._frame = i  # store current frame

        image = self.get_image_data(i)
        p = self.params[i]
        Z = self.model(p, self.grid)
        Y, X = self.grid
        self.update(X, Y, Z, image)

        if draw:
            self.fig.canvas.draw()

        return i
