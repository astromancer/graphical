
# std
from collections import abc

# third-party
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colorbar as cbar
from matplotlib import cm, ticker
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable

# local
from recipes.pprint import describe
from recipes.logging import LoggingMixin
from recipes.functionals import ignore_params, ignore_returns

# relative
from ..depth.bar3d import Bar3D
from ..sliders import RangeSliders
from ..utils import get_percentile_limits
from ..moves import TrackAxesUnderMouse, mpl_connect
from ..moves.machinery import CanvasBlitHelper, Observers
from .hist import PixelHistogram
from .utils import _sanitize_data, guess_figsize


# TODO: docstrings (when stable)
# TODO: unit tests
# TODO: maybe display things like contrast ratio ??

# ---------------------------------------------------------------------------- #
# SCROLL_CMAPS = cmr.get_cmap_list()
# SCROLL_CMAPS_R = [_ for _ in SCROLL_CMAPS if _.endswith('_r')]

cmap_categories = {
    'Perceptually Uniform Sequential':
        {'viridis', 'plasma', 'inferno', 'magma', 'cividis'},
    'Sequential':
        {'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
         'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
         'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn'},
    'Sequential (2)':
        {'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
         'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
         'hot', 'afmhot', 'gist_heat', 'copper'},
    'Diverging':
        {'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
         'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic'},
    'Cyclic':
        {'twilight', 'twilight_shifted', 'hsv'},
    'Qualitative':
        {'Pastel1', 'Pastel2', 'Paired', 'Accent',
         'Dark2', 'Set1', 'Set2', 'Set3',
         'tab10', 'tab20', 'tab20b', 'tab20c'},
    'Miscellaneous':
        {'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
         'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg',
         'gist_rainbow', 'rainbow', 'jet', 'turbo', 'nipy_spectral',
         'gist_ncar'}
}

_remove = cmap_categories['Qualitative'] | {'flag', 'prism'}
_remove |= {f'{r}_r' for r in _remove}
SCROLL_CMAPS = ({*cm._colormaps._cmaps.keys()} - _remove)


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

    def __init__(self, cbar, cmaps=SCROLL_CMAPS, timeout=0.25, use_blit=True):

        self.colorbar = cbar

        prefixes = {}
        for cmap in cmaps:
            prefix = ''
            if '.' in cmap:
                prefix, cmap = cmap.split('.')
                prefix += '.'
            prefixes[cmap] = prefix

        self.available = sorted(prefixes.keys(), key=str.lower)
        self.prefixes = prefixes
        self._letter_index = next(zip(*self.available))
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

    def _add_artist(self, artist, mappable=None):

        if mappable is None:
            mappable = isinstance(artist, cm.ScalarMappable)

        if mappable:
            self.mappables.add(artist)

        return super()._add_artist(artist)

    @property
    def mappable(self):
        return self.colorbar.mappable

    # def get_cmap(self):
    #     return remove_prefix(self.mappable.get_cmap().name, 'cmr.')

    def set_cmap(self, cmap):
        for sm in self.mappables:
            sm.set_cmap(cmap)

    def _on_scroll(self, event):
        if self.colorbar is None or event.inaxes is not self.colorbar.ax:
            return

        super()._on_scroll(event)

    def _on_first_draw(self, _):
        super()._on_first_draw(_)
        if self.colorbar.solids in self.artists:
            self.artists.remove(self.colorbar.solids)

    def _scroll_cmap(self, event):
        # called during scroll callback
        current = self.mappable.get_cmap().name
        self.logger.debug('Current cmap: {}', current)
        if '.' in current:
            _, current = current.split('.')

        avail = self.available
        inc = [-1, +1][event.button == 'up']
        idx = avail.index(current) if current in avail else -1
        new = f'{avail[(idx + inc) % len(avail)]}'
        new = f'{self.prefixes[new]}{new}'

        self.logger.info('Scrolling cmap: {} -> {}', current, new)
        self.set_cmap(new)

        cb = self.colorbar
        return self.artists, cb.solids, cb.lines, cb.dividers

    @mpl_connect('key_press_event')
    def _on_key(self, event):

        if self._axes_under_mouse is not self.colorbar.ax:
            self.logger.debug(f'Ignoring key press since {self._axes_under_mouse = }')
            return

        if event.key not in self._letter_index:
            self.logger.debug('No cmaps starting with letter {!r}', event.key)
            return

        current = self.mappable.get_cmap().name
        if '.' in current:
            _, current = current.split('.')

        i = self._letter_index.index(event.key)
        new = self.available[i]
        new = f'{self.prefixes[new]}{new}'
        self.logger.info('Scrolling cmap: {} -> {}', current, new)
        self.set_cmap(new)

        cb = self.colorbar
        self.draw((self.artists, cb.solids, cb.lines, cb.dividers))


# ---------------------------------------------------------------------------- #


class ImageDisplay(CanvasBlitHelper, LoggingMixin):

    # TODO: move cursor with arrow keys when hovering over figure (like ds9)
    # TODO: optional zoomed image window

    # TODO: better histogram for integer / binary data with narrow ranges
    # TODO: method pixels corresponding to histogram bin?

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
        kws.setdefault('interpolation', 'none')

        self.has_cbar = bool(cbar)
        self.has_hist = bool(hist)
        self.has_sliders = bool(sliders)

        hist = (self._default_hist_kws if hist is True else dict(hist)) if hist else {}

        image = self._resolve_image(image)

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
            cbar_kws = {} if cbar is True else dict(cbar)
            self.cbar = self.colorbar(**cbar_kws)

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
            # save cbar in image background after cmap scroll
            # TODO: optimize between sliders and cmap scroll which both save
            # background and share some artists
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

    def _resolve_image(self, image):
        # check data
        image = np.ma.asarray(image)   # try remove redundant dimensions
        if (nd := image.ndim) != 2 and (image := image.squeeze()).ndim != 2:
            msg = f'{describe(type(self))} cannot display {nd}D data.'
            if image.ndim == 3:
                msg += ('Use the `VideoDisplay` class to display 3D data as'
                        ' scrollable video.')

            raise ValueError(msg)

        # convert boolean to integer (for colour scale algorithm)
        if image.dtype.name == 'bool':
            image = image.astype(int)

        return image

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

        # create figure / axes if required
        if ax is None:
            if fig is None:
                if figsize == 'auto':
                    # automatically determine the figure size based on the data
                    figsize = self.guess_figsize(self.data)
                    # FIXME: the guessed size does not account for the colorbar
                    #  histogram

                fig = plt.figure(figsize=figsize)

            # axes
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
        if self.has_hist:
            kws = {**dict(ticks=[], format=ticker.NullFormatter()),
                   **kws}

        return Colorbar(self.cax, self.image, **kws)

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
        cbar = self.cbar
        sliders = self._make_sliders(use_blit, cbar) if self.has_sliders else None
        hist = self._make_hist(use_blit, hist_kws, sliders, cbar) if self.has_hist else None
        return sliders, hist

    def _make_hist(self, use_blit, hist_kws, sliders, cbar):
        hist = PixelHistogram(self.hax,
                              self.image,
                              'horizontal',
                              use_blit=use_blit,
                              **(hist_kws or {}))
        self.add_art(hist.bars)

        if sliders:
            # set minimun slider separation 1 bin
            sliders.min_span = np.diff(hist.bin_edges).max()
            # add for blit
            sliders.add_art(hist.bars)
            # sliders.link([t.gridline for t in self.hax.xaxis.majorTicks])

            # cmap scroll blit
        cbar.scroll.mappables.add(hist)
        cbar.scroll.add_art(hist.bars)

        # set ylim if reasonable to do so
        # if data.ptp():
        #     # avoid warnings in setting upper/lower limits identical
        #     hax.set_ylim((data.min(), data.max()))

        # FIXME: will have to change with different orientation
        self.hax.yaxis.tick_right()
        self.hax.grid(True)

        return hist

    def _make_sliders(self, use_blit, cbar):
        clim = self.image.get_clim()
        slide = self.sliderClass(self.hax,
                                 clim,
                                 'y',
                                 color='rbg',
                                 ms=(2, 1, 2),
                                 extra_markers='>s<',
                                 use_blit=use_blit)

        # lambda *_: self.set_clim(*self.sliders.positions)
        slide.lower.on_move.add(self.update_clim)
        slide.upper.on_move.add(self.update_clim)

        for mv in slide.movable.values():
            # add sliders to animated art for blitting
            self.add_art(mv.draw_list)
            #     # mv.on_pick.add(self.sliders.save_background)
            mv.on_release.add(ignore_params(slide.save_background))

        if cbar:
            # cmap scroll blit
            # sliders to redraw on cmap scroll
            cbar.scroll.add_art(*(set(mv.draw_list) for mv in slide.movable.values()))
            # sliders to save background on cmap scrol
            cbar.scroll.on_scroll.add(ignore_params(slide.save_background))

            # for mv in sliders.movable.values():
            #     # save slider positioons in cmap scroll background
            #     mv.on_release.add(ignore_params(cbar.scroll.save_background))

        return slide

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
        # NOTE: for some unfathomable reason, this only works if done twice. ???
        self.image.set_clim(*clim)
        self.image.set_clim(*clim)

        if not self.has_hist:
            return self.image

        self.histogram.update()
        # minimum slider range is max bin size
        self.sliders.min_span = np.diff(self.histogram.bin_edges).max()

        return self.image, self.histogram.bars

    def update_clim(self, *xydata):
        """Set colour limits on slider move."""
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
        xs = f'{x:=1.{precision:d}f}'
        ys = f'{y:=1.{precision:d}f}'
        # z
        col, row = int(x + 0.5), int(y + 0.5)
        nrows, ncols = self.ishape
        if (0 <= col < ncols) and (0 <= row < nrows):
            data = self.image.get_array()
            z = data[row, col]
            # handle masked data
            # prevent Warning: converting a masked element to nan.
            zs = f'z={masked_str}' if np.ma.is_masked(z) else f'{z=:1.{precision:d}f}'
            return ',    '.join((xs, ys, zs))

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


class Image3D:
    def __init__(self, image,
                 origin=(0, 0), cmap=None, figure=None,
                 bar3d_kws=None, **image_kws):

        self.figure, (axi, ax3) = self.setup_figure(figure)
        self.axi, self.ax3 = axi, ax3

        y0, x0 = origin = np.array(origin)
        y1, x1 = origin + image.shape
        y, x = np.indices(image.shape) + origin[:, None, None]

        cmap = cm.get_cmap(cmap)
        self.image = im = ImageDisplay(
            image, ax=self.axi, cmap=cmap, extent=(x0, x1, y0, y1),
            **{**dict(hist=False, sliders=False,
                      cbar={'format': ticker.NullFormatter()}),
               **image_kws}
        )

        # remove cbar ticks / labels
        # im.cax.yaxis.set_tick_params(length=0)
        # im.cax.yaxis.major.formatter = ticker.NullFormatter()

        self.bars = bars = Bar3D(self.ax3, x, y, image, cmap=cmap,
                                 **{**dict(zaxis_cbar=True),
                                    **(bar3d_kws or {})})

        # Cmap scroll callbacks
        scroll = im.cbar.scroll
        for art in (bars.bars, bars.cbar.line):
            scroll.add_art(art)
            scroll.on_scroll.add(ignore_returns(ignore_params(art.do_3d_projection)))

        # save background after 3d rotation
        # FIXME: this slows down the rotation interaction significantly
        self.bars.on_rotate.add(ignore_params(scroll.save_background))

    def set_cmap(self, cmap):
        # change cmap for all mappables
        self.image.cbar.scroll.set_cmap(cmap)

    def set_data(self, image, clim=None):

        self.image.image.set_data(image)

        if (clim is None) or (clim is not False):
            clim = self.image.clim_from_data(image)
            self.image.set_clim(*clim)

        self.bars.set_z(image, clim)

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
