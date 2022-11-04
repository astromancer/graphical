
# std
import itertools as itt
from collections import abc

# third-party
import numpy as np
import cmasher as cmr
import matplotlib.pyplot as plt
import matplotlib.colorbar as cbar
from matplotlib import ticker
from matplotlib.gridspec import GridSpec
from matplotlib.cm import ScalarMappable, get_cmap
from mpl_toolkits.axes_grid1 import make_axes_locatable

# local
from recipes.pprint import describe
from recipes.logging import LoggingMixin
from recipes.string import remove_prefix
from recipes.functionals import ignore_params

# relative
from ..depth.bar3d import Bar3D
from ..sliders import RangeSliders
from ..utils import get_percentile_limits
from ..moves import TrackAxesUnderMouse, mpl_connect
from ..moves.machinery import CanvasBlitHelper, Observers
from .hist import PixelHistogram
from .utils import _sanitize_data, guess_figsize, set_clim_connected


# TODO: docstrings (when stable)
# TODO: unit tests
# TODO: maybe display things like contrast ratio ??

# ---------------------------------------------------------------------------- #
SCROLL_CMAPS = cmr.get_cmap_list()
SCROLL_CMAPS_R = [_ for _ in SCROLL_CMAPS if _.endswith('_r')]

# ---------------------------------------------------------------------------- #


def auto_grid(n):
    x = int(np.floor(np.sqrt(n)))
    y = int(np.ceil(n / x))
    return x, y

# ---------------------------------------------------------------------------- #


class Colorbar(cbar.Colorbar, LoggingMixin):

    def __init__(self, ax, mappable=None, scroll=True, **kws):
        super().__init__(ax, mappable, **kws)

        self.scroll = None
        if scroll:
            self.scroll = CMapScroll(self)


class ScrollAction(CanvasBlitHelper):

    def __init__(self, artists=(), connect=False, use_blit=True, rate_limit=4):
        super().__init__(artists, connect, use_blit)
        # observer container for scroll callbacks
        self.on_scroll = Observers(rate_limit)

    @mpl_connect('scroll_event', rate_limit=4)
    def _on_scroll(self, event):

        # run callbacks
        if art := self.on_scroll(event):
            # draw the artists that were changed by scroll action
            self.draw(art)


class CMapScroll(ScrollAction, TrackAxesUnderMouse, LoggingMixin):
    # TODO: manage timeout through CallbackManager and mpl_connect

    def __init__(self, cbar, cmaps=SCROLL_CMAPS_R, timeout=0.25, use_blit=True):

        self.colorbar = cbar
        self.available = sorted(set(cmaps))
        self._letter_index = next(zip(self.available))
        self.mappables = {self.colorbar.mappable}

        # canvas = cbar.ax.figure.canvas if cbar.ax else None
        ScrollAction.__init__(self,
                              (self.mappables, self.colorbar.solids),
                              use_blit, timeout)
        # NOTE: Adding `self.colorbar.solids` here so we get the background
        # correct on first draw. A new QuadMesh object will be created whenever
        # the cmap is changed, so we need to remove the original in
        # `_on_first_draw` after saving the initial background
        self.on_scroll.add(self._scroll_cmap)

    def add_artist(self, artist):
        if isinstance(artist, ScalarMappable):
            self.mappables.add(artist)
        return super().add_artist(artist)

    @property
    def mappable(self):
        return self.colorbar.mappable

    def get_cmap(self):
        return remove_prefix(self.mappable.get_cmap().name, 'cmr.')

    def set_cmap(self, cmap):
        for sm in self.mappables:
            sm.set_cmap(cmap)

    def _on_scroll(self, event):
        if self.colorbar is None or event.inaxes is not self.colorbar.ax:
            return

        super()._on_scroll(event)

    def _on_first_draw(self, _):
        super()._on_first_draw(_)
        self.artists.remove(self.colorbar.solids)

    def _scroll_cmap(self, event):
        # called during scroll callback
        cmap = self.get_cmap()
        self.logger.debug('Current cmap: {}', cmap)

        avail = self.available
        inc = [-1, +1][event.button == 'up']
        idx = avail.index(cmap) if cmap in avail else -1
        new = f'cmr.{avail[(idx + inc) % len(avail)]}'
        self.logger.info('Scrolling cmap: {} -> {}', cmap, new)
        self.set_cmap(new)

        cb = self.colorbar
        return self.artists, cb.solids, cb.lines,  cb.dividers

    @mpl_connect('key_press_event')
    def _on_key(self, event):

        if self._axes_under_mouse is not self.colorbar.ax:
            self.logger.debug(f'Ignoring key press since {self._axes_under_mouse = }')
            return

        if event.key not in self._letter_index:
            self.logger.debug('No cmaps starting with {!r}', event.key)
            return

        current = self.get_cmap()
        i = self._letter_index.index(event.key)
        new = f'cmr.{self.available[i]}'
        self.logger.info('Scrolling cmap: {} -> {}', current, new)
        self.set_cmap(new)

        cb = self.colorbar
        self.draw((self.mappables,  cb.solids, cb.lines,  cb.dividers))


# ---------------------------------------------------------------------------- #


class ImageDisplay(CanvasBlitHelper, LoggingMixin):
    # TODO: move cursor with arrow keys when hovering over figure (like ds9)
    # TODO: optional zoomed image window

    # TODO: better histogram for integer / binary data with narrow ranges
    # TODO: method pixels corresponding to histogram bin?

    # TODO: remove ticks on cbar ax
    # TODO: plot scale func on hist axis

    sliderClass = RangeSliders
    _default_plims = (0.25, 99.75)
    _default_hist_kws = dict(bins=100)

    def __init__(self, image, *args, cbar=True, hist=True, sliders=True,
                 use_blit=True, connect=True, **kws):
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

        # set origin
        kws.setdefault('origin', 'lower')

        self.has_cbar = bool(cbar)
        self.has_hist = bool(hist)
        self.has_sliders = bool(sliders)

        hist = (self._default_hist_kws if hist is True else dict(hist)) if hist else {}

        # check data
        image = np.ma.asarray(image).squeeze()  # remove redundant dimensions
        if image.ndim != 2:
            msg = f'{describe(type(self))} cannot display {image.ndim}D data.'
            if image.ndim == 3:
                msg += ('Use the `VideoDisplay` class to display 3D data as '
                        'scrollable video.')
            raise ValueError(msg)

        # convert boolean to integer (for colour scale algorithm)
        if image.dtype.name == 'bool':
            image = image.astype(int)

        self.data = image
        self.ishape = self.data.shape

        # create the figure if needed
        self.divider = None
        self.figure, axes = self.setup_figure(kws)
        self.ax, self.cax, self.hax = axes
        ax = self.ax

        # get colour limits
        if 'vmin' not in kws and 'vmax' not in kws:
            self.clim_from_data(image, kws)

        # use imshow to draw the image
        self.image = ax.imshow(image, *args, **kws)
        self.norm = self.image.norm
        self.get_clim = self.image.get_clim
        # self.image.set_clim(*clim)

        # create the colourbar / histogram / sliders
        self.cbar = None
        if self.has_cbar:
            self.cbar = self.colorbar()

        # init blit and callbacks
        CanvasBlitHelper.__init__(self, self.image, connect=False, active=use_blit)

        # create sliders and pixel histogram
        self.sliders, self.histogram = self.make_sliders(use_blit, hist)

        self._cbar_hist_connectors = {}
        if self.cbar:
            if hist and sliders:
                self._cbar_hist_connectors = self.connect_cbar_hist()
                # redraw connectors on scroll
                self.cbar.scroll.add_art(self._cbar_hist_connectors.values())

            # cmap scroll blit
            self.cbar.scroll.on_scroll.add(ignore_params(self.save_background))
            # NOTE: Updating via method below causes a undesirable white flash
            # from the background even though it's faster.
            # self.cbar.scroll.on_scroll.add(
            #     self.update_background,
            #     (self.cbar.solids, self._cbar_hist_connectors.values())
            # )

        # callbacks
        if connect:
            self.connect()

    def __iter__(self):
        yield self.figure
        yield self.ax

    # def set_canvas(self, canvas):
    #     super().set_canvas(canvas)
    #     if self.sliders:
    #         self.sliders.set_canvas(canvas)

    def setup_figure(self, kws):
        # NOTE intentionally not unpacking keyword dict so that the keys
        #  removed here reflect at the calling scope
        """
        Create the figure and add the axes

        Parameters
        ----------
        kws

        Returns
        -------

        """

        fig = kws.pop('fig', None)
        ax = kws.pop('ax', None)
        cax = kws.pop('cax', None)
        hax = kws.pop('hax', None)
        title = kws.pop('title', None)
        figsize = kws.pop('figsize', 'auto')
        # sidebar = kws.pop('sidebar', True)

        # create axes if required
        if fig is None:
            if figsize == 'auto':
                # automatically determine the figure size based on the data
                figsize = self.guess_figsize(self.data)
                # FIXME: the guessed size does not account for the colorbar
                #  histogram

            fig = plt.figure(figsize=figsize)

        if ax is None:
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

        image = self.data[0] if data is None else data
        return guess_figsize(image, fill_factor, max_pixel_size)

    # TODO: custom cbar class?

    def colorbar(self, **kws):
        # No need for the data labels on the colourbar since it will be on the
        # histogram axis.
        fmt = ticker.NullFormatter() if self.has_hist else None
        return Colorbar(self.cax, self.image,  format=fmt, ticks=[], **kws)

    def connect_cbar_hist(self):
        if not (self.histogram and self.sliders):
            raise ValueError()

        from matplotlib.patches import ConnectionPatch

        connectors = {}
        c0, c1 = self.image.get_clim()
        x0, _ = self.hax.get_xlim()
        xy_from = [(1, 0),
                   (1, 1),
                   (1, 0.5)]
        xy_to = [(x0, c0),
                 (x0, c1),
                 (x0, (c0 + c1) / 2)]
        for mv, xy_from, xy_to in zip(self.sliders, xy_from, xy_to):
            self.figure.add_artist(
                p := ConnectionPatch(
                    xyA=xy_from, coordsA=self.cax.transAxes,
                    xyB=xy_to, coordsB=self.hax.transData,
                    arrowstyle="-",
                    edgecolor=mv.artist.get_color(),
                    alpha=0.5
                )
            )
            connectors[mv.artist] = p

        self.sliders.centre.on_move.add(self._update_connectors)
        self.sliders.add_art(connectors.values())

        return connectors

    def _update_connectors(self, x, y):
        positions = [*self.sliders.positions, self.sliders.positions.mean()]
        for patch, pos in zip(self._cbar_hist_connectors.values(), positions):
            xy = np.array(patch.xy2)
            xy[self.sliders._ifree] = pos
            patch.xy2 = xy

        return self._cbar_hist_connectors.values()

    def make_sliders(self, use_blit, hist_kws=None):
        # data = self.image.get_array()

        sliders = None
        cbar = self.cbar
        if self.has_sliders:
            clim = self.image.get_clim()
            sliders = self.sliderClass(self.hax, clim, 'y',
                                       color='rbg',
                                       ms=(2, 1, 2),
                                       extra_markers='>s<',
                                       use_blit=use_blit)

            sliders.lower.on_move.add(self.update_clim)
            sliders.upper.on_move.add(self.update_clim)
            for mv in sliders.movable.values():
                # add sliders to animated art for blitting
                self.add_art(mv.draw_list)
            #     # mv.on_pick.add(self.sliders.save_background)
                mv.on_release.add(ignore_params(sliders.save_background))

            if cbar:
                # cmap scroll blit
                # sliders to redraw on cmap scroll
                cbar.scroll.add_art(*(set(mv.draw_list)
                                      for mv in sliders.movable.values()))
                # sliders to save background on cmap scrol
                cbar.scroll.on_scroll.add(ignore_params(sliders.save_background))

                # for mv in sliders.movable.values():
                #     # save slider positioons in cmap scroll background
                #     mv.on_release.add(ignore_params(cbar.scroll.save_background))

        hist = None
        if self.has_hist:
            hist = PixelHistogram(self.hax, self.image, 'horizontal',
                                  use_blit=use_blit, **(hist_kws or {}))
            self.add_art(hist.bars)

            if sliders:
                # add for blit
                sliders.artists.add(hist.bars)

            # cmap scroll blit
            cbar.scroll.mappables.add(hist)  # will call hist.set_cmap on scroll
            cbar.scroll.artists.add(hist.bars)

            # set ylim if reasonable to do so
            # if data.ptp():
            #     # avoid warnings in setting upper/lower limits identical
            #     hax.set_ylim((data.min(), data.max()))

            # FIXME: will have to change with different orientation
            self.hax.yaxis.tick_right()
            self.hax.grid(True)

        return sliders, hist

    def set_cmap(self, cmap):
        self.image.set_cmap(cmap)
        if self.has_hist:
            self.histogram.set_cmap(cmap)

        # FIXME: blit!
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
        self.sliders.min_span = (clim[0] - clim[1]) / self.histogram.bins

        # TODO: return COLOURBAR ticks?
        return self.image, self.histogram.bars

    def update_clim(self, *xydata):
        """Set colour limits on slider move"""
        return self.set_clim(*self.sliders.positions)

    def format_coord(self, x, y, precision=3, masked_str='--'):
        """
        Create string representation for cursor position in data coordinates.

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

        # connect sliders' interactions + save background for blitting
        if self.sliders:
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
        self.figure = fig
        self.axes = axes
        self.imd = imd

    def __iter__(self):
        yield from (self.figure, self.axes, self.imd)

    def save(self, filenames):
        from matplotlib.transforms import Bbox

        fig = self.figure

        assert len(filenames) == self.axes.size

        ax_per_image = (len(fig.axes) // self.axes.size)
        # axit = mit.chunked(self.figure.axes, ax_per_image)

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
        return [ax.images[0].get_array() for ax in self.figure.axes]

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


class Image3D:
    def __init__(self, image,
                 origin=(0, 0), cmap=None, figure=None,
                 bar3d_kws=None, **image_kws):

        self.figure, (axi, ax3) = self.setup_figure(figure)
        self.axi, self.ax3 = axi, ax3

        y0, x0 = origin = np.array(origin)
        y1, x1 = origin + image.shape
        y, x = np.indices(image.shape) + origin[:, None, None]

        cmap = get_cmap(cmap)
        self.image = im = ImageDisplay(image, ax=self.axi, cmap=cmap,
                                       extent=(x0, x1, y0, y1),
                                       **{**dict(hist=False, sliders=False),
                                          **image_kws, })

        # remove cbar ticks / labels
        im.cax.yaxis.set_tick_params(length=0)
        im.cax.yaxis.major.formatter = ticker.NullFormatter()

        self.bar3d = bar3d = Bar3D(self.ax3, x, y, image, cmap=cmap,
                                   **{**dict(zaxis_cbar=True),
                                       **(bar3d_kws or {})})

        # Cmap scroll callbacks
        scroll = im.cbar.scroll
        scroll.add_artist(bar3d.bars)
        scroll.add_artist(bar3d.cbar.line)

        scroll.on_scroll.add(bar3d.bars.do_3d_projection)
        scroll.on_scroll.add(bar3d.cbar.line.do_3d_projection)

        self.bar3d.on_rotate.add(scroll.save_background)

    def set_cmap(self, cmap):
        # change cmap for all mappables
        self.image.cbar.scroll.set_cmap(cmap)

    def setup_figure(self, fig=None):
        if fig is None:
            fig = plt.figure()  # size = (8.5, 5)

        axi = fig.add_subplot(1, 2, 1)
        ax3 = fig.add_subplot(1, 2, 2,
                              projection='3d',
                              azim=-125, elev=30)

        # label axes
        for xy in 'xy':
            for ax in (axi, ax3):
                getattr(ax, f'set_{xy}label')(f'${xy}$')
                getattr(ax, f'{xy}axis').set_major_locator(
                    ticker.MaxNLocator('auto', steps=[1, 2, 5, 10]))

        ax3.set(facecolor='none')  # xlabel='$x$', ylabel='$y$'

        return fig, (axi, ax3)
