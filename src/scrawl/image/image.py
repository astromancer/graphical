
# std
import itertools as itt
from time import time
from collections import abc

# third-party
import numpy as np
import cmasher as cmr
import matplotlib.pylab as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable

# local
from recipes.logging import LoggingMixin
from recipes.string import remove_prefix

# relative
from ..sliders import RangeSliders
from ..utils import get_percentile_limits
from ..moves.callbacks import CallbackManager, mpl_connect
from .hist import PixelHistogram
from .utils import _sanitize_data, guess_figsize, set_clim_connected


# from .zscale import zrange

# from astropy.visualization import mpl_normalize  # import ImageNormalize as _

# TODO: docstrings (when stable)
# TODO: unit tests
# TODO: maybe display things like contrast ratio ??

# TODO: middle mouse resets axes limits

SCROLL_CMAPS = cmr.get_cmap_list()
SCROLL_CMAPS_R = [_ for _ in SCROLL_CMAPS if _.endswith('_r')]


def auto_grid(n):
    x = int(np.floor(np.sqrt(n)))
    y = int(np.ceil(n / x))
    return x, y


class ImageDisplay(CallbackManager, LoggingMixin):
    # TODO: move cursor with arrow keys when hovering over figure (like ds9)
    # TODO: optional zoomed image window
    # TODO: scroll colorbar to switch cmap

    # FIXME: hist: outside alpha changes with middle slider move.....

    # TODO: optional Show which region on the histogram corresponds to colorbar
    # TODO: better histogram for integer / binary data with narrow ranges
    # TODO: method pixels corresponding to histogram bin?

    # TODO: remove ticks on cbar ax
    # TODO: plot scale func on hist axis

    sliderClass = RangeSliders
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

        # get colour limits
        if 'vmin' not in kws and 'vmax' not in kws:
            self.clim_from_data(image, kws)

        # use imshow to draw the image
        self.image = ax.imshow(image, *args, **kws)
        self.norm = self.image.norm
        # self.image.set_clim(*clim)

        # create the colourbar / histogram / sliders
        self.cbar = None
        if self.has_cbar:
            self.cbar = self.make_cbar()

        # create sliders after histogram so they display on top
        self.sliders, self.histogram = self.make_sliders(hist_kws)
        self._cbar_hist_connectors = {}
        self.connect_cbar_hist()

        # connect on_draw for debugging
        self._draw_count = 0
        # self.cid = ax.figure.canvas.mpl_connect('draw_event', self._on_draw)

        # link viewlims of the 3d axes
        CallbackManager.__init__(self, self.figure.canvas)

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

    def _on_draw(self, event):
        self.logger.debug('DRAW %i', self._draw_count)  # ,  vars(event)
        if self._draw_count == 0:
            self._on_first_draw(event)
        self._draw_count += 1

    def _on_first_draw(self, event):
        self.logger.debug('FIRST DRAW')

    # TODO: custom cbar class?
    def make_cbar(self):
        fmt = None
        if self.has_hist:
            # No need for the data labels on the colourbar since it will be on
            # the histogram axis.
            from matplotlib import ticker
            fmt = ticker.NullFormatter()

        return self.figure.colorbar(self.image, cax=self.cax, format=fmt)

    # TODO: manage timeout through CallbackManager and mpl_connect
    _cmap_scroll_timeout = 0.1
    _cmap_switch_time = -1

    @mpl_connect('scroll_event')
    def _cmap_scroll(self, event):

        if self.cbar is None or self.cax is not event.inaxes:
            return

        now = time()
        if now - self._cmap_switch_time < self._cmap_scroll_timeout:
            self.debug('Scroll timeout')
            return

        self._cmap_switch_time = now
        cmap = remove_prefix(self.image.get_cmap().name, 'cmr.')
        self.logger.debug('Current cmap: {}', cmap)

        available = SCROLL_CMAPS_R
        i = available.index(cmap) if cmap in available else -1
        new = f'cmr.{available[(i + 1) % len(available)]}'
        self.logger.info('Scrolling cmap: {} -> {}', cmap, new)
        self.set_cmap(new)

        self.canvas.draw()

    def connect_cbar_hist(self):
        if not (self.histogram and self.sliders):
            raise ValueError()

        from matplotlib.patches import ConnectionPatch

        c0, c1 = self.image.get_clim()
        x0, _ = self.hax.get_xlim()
        xy_from = [(1, 0),
                   (1, 1),
                   (1, 0.5)]
        xy_to = [(x0, c0),
                 (x0, c1),
                 (x0, (c0 + c1) / 2)]
        for mv, xy_from, xy_to in zip(self.sliders, xy_from, xy_to):
            p = ConnectionPatch(
                xyA=xy_from, coordsA=self.cax.transAxes,
                xyB=xy_to, coordsB=self.hax.transData,
                arrowstyle="-",
                edgecolor=mv.artist.get_color(),
                alpha=0.5
            )
            self.figure.add_artist(p)
            self._cbar_hist_connectors[mv.artist] = p

        self.sliders.centre.on_move.add(self._update_connectors)

        return self._cbar_hist_connectors

    def _update_connectors(self, x, y):
        positions = [*self.sliders.positions, self.sliders.positions.mean()]
        for patch, pos in zip(self._cbar_hist_connectors.values(), positions):
            xy = np.array(patch.xy2)
            xy[self.sliders._ifree] = pos
            patch.xy2 = xy
        return self._cbar_hist_connectors.values()

    def make_sliders(self, hist_kws=None):
        # data = self.image.get_array()

        # FIXME: This is causing recursive repaint!!

        sliders = None
        if self.has_sliders:
            clim = self.image.get_clim()
            sliders = self.sliderClass(self.hax, clim, 'y',
                                       color='rbg',
                                       ms=(2, 1, 2),
                                       extra_markers='>s<')

            sliders.lower.on_move.add(self.update_clim)
            sliders.upper.on_move.add(self.update_clim)

        hist = None
        if self.has_hist:
            hist = PixelHistogram(self.hax, self.image, 'horizontal',
                                  use_blit=self.use_blit, **(hist_kws or {}))

            # set ylim if reasonable to do so
            # if data.ptp():
            #     # avoid warnings in setting upper/lower limits identical
            #     hax.set_ylim((data.min(), data.max()))

            # NOTE: will have to change with different orientation
            self.hax.yaxis.tick_right()
            self.hax.grid(True)

        return sliders, hist

    def set_cmap(self, cmap):
        self.image.set_cmap(cmap)
        if self.has_hist:
            self.histogram.set_cmap(cmap)

        self.canvas.draw()

    def clim_from_data(self, data, kws=None, **kws_):
        """
        Get colour scale limits for data.
        """
        # `kws` dict from which we remove 'extra' keywords that are not allowed
        # in imshow. This allows a dict from the calling scope to be edited here
        # without global statement. This function can also still be used with
        # unpacked keyword args
        kws = kws or {}
        if 'clim' in kws:
            clim = kws.pop('clim')
            if clim is None or (clim is False):
                return None, None

            if isinstance(clim, abc.Sized) and len(clim) == 2:
                kws['vmin'], kws['vmax'] = clim
                return clim

            if clim is not True:
                raise ValueError(f'Invalid value for `clim`: {clim!r}.')

        plims = kws.pop('plims', None)
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
            self.logger.warning('Bad colour interval: ({:.1f}, {:.1f}). '
                                'Ignoring', *clims)
            return None, None

        return clims

    def set_clim(self, *clim):
        self.image.set_clim(*clim)

        if not self.has_hist:
            return self.image

        self.histogram.update()
        self.sliders.min_span = (clim[0] - clim[1]) / 100

        # TODO: return COLOURBAR ticklabels?
        return self.image, self.histogram.bars

    def update_clim(self, *xydata):
        """Set colour limits on slider move"""
        return self.set_clim(*self.sliders.positions)

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

        if not hasattr(self, 'image'):
            # prevents a swap of repeated errors flooding the terminal for
            # mouse over when no image has been drawn
            return 'no image'

        # xy repr
        xs = f'x={x:1.{precision:d}f}'
        ys = f'y={y:1.{precision:d}f}'
        # z
        col, row = int(x + 0.5), int(y + 0.5)
        nrows, ncols = self.ishape
        if (0 <= col < ncols) and (0 <= row < nrows):
            data = self.image.get_array()
            z = data[row, col]
            # handle masked data
            # prevent Warning: converting a masked element to nan.
            zs = f'z={masked_str}' if np.ma.is_masked(z) else f'z={z:1.{precision:d}f}'
            return ',\t'.join((xs, ys, zs)).expandtabs()
        else:
            return ', '.join((xs, ys))

    def connect(self):
        super().connect()

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

# ---------------------------------------------------------------------------- #

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
    w = len(str(n))
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
        art.append(imd.image)

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
    # fig.colorbar(imd.image, cax)
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

    def _clim_all(self, art, imd, images, plims):
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
