
# third-party
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colorbar as cbar
from matplotlib import cm, ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable

# local
from recipes import dicts
from recipes.pprint import describe
from recipes.config import ConfigNode
from recipes.logging import LoggingMixin
from recipes.functionals import ignore_params, ignore_returns

# relative
from ..depth.prisms import Bar3D
from ..sliders import RangeSliders
from ..moves.machinery import CanvasBlitHelper
from .cbar import Colorbar
from .hist import PixelHistogram
from .utils import guess_figsize, resolve_clim


# TODO: docstrings (when stable)
# TODO: unit tests
# TODO: maybe display things like contrast ratio ??

# ---------------------------------------------------------------------------- #
# module config
CONFIG = ConfigNode.load_module(__file__)


# ---------------------------------------------------------------------------- #
class FigureSetup:

    def __init__(self,  cbar=True, hist=True, sliders=True):

        self.divider = None
        self.has_cbar = bool(cbar)
        self.has_hist = bool(hist)
        self.has_sliders = bool(sliders)

    def setup_figure(self, fig=None,
                     ax=None, cax=None, hax=None,
                     title=None, figsize=CONFIG.fig.size, data=None,
                     subplot=111, subplot_kws=None):
        """
        Create the figure and add the axes.
        """

        # create figure / axes if required
        if ax is None:
            if figsize == 'auto':
                # automatically determine the figure size based on the data
                figsize = self.guess_figsize(data)
                # FIXME: the guessed size does not account for the colorbar
                #  histogram

            if fig is None:
                fig = plt.figure()

            # axes
            ax = fig.add_subplot(subplot, **(subplot_kws or {}))
            ax.tick_params('x', which='both', top=True)
            fig.set_size_inches(figsize)

        # axes = namedtuple('AxesContainer', ('image',))(ax)
        if self.has_cbar and (cax is None):
            self.divider = make_axes_locatable(ax)
            cax = self.divider.append_axes('right', **CONFIG.cbar)

        if self.has_hist and (hax is None):
            hax = self.divider.append_axes('right',
                                           size=CONFIG.hist.size,
                                           pad=CONFIG.hist.pad)

        # set the axes title if given
        if title is not None:
            ax.set_title(title)

        return ax.figure, (ax, cax, hax)

    def guess_figsize(self, data, fill_factor=CONFIG.fig.fill,
                      max_pixel_size=CONFIG.fig.max_pixel_size):
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
        if data is None:
            raise ValueError('Need image data to guess appropriate figure size.')

        return guess_figsize(data, fill_factor, max_pixel_size)

    def colorbar(self, **kws):
        # No need for the data labels on the colourbar since it will be on the
        # histogram axis.
        if self.has_hist:
            kws = {**dict(ticks=[], format=ticker.NullFormatter()), **kws}

        return Colorbar(self.cax, self.image, **kws)


class ImageDisplay(CanvasBlitHelper, FigureSetup, LoggingMixin):

    # TODO: move cursor with arrow keys when hovering over figure (like ds9)
    # TODO: optional zoomed image window

    # TODO: better histogram for integer / binary data with narrow ranges
    # TODO: method pixels corresponding to histogram bin?

    # TODO: plot scale func on hist axis

    sliderClass = RangeSliders

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
        ax: Axes
            Axes on which to display

        remaining keywords passed to ax.imshow


        """

        # init figure setup mixin
        FigureSetup.__init__(self, cbar, hist, sliders)

        # get data
        self.data = image = self._resolve_image(image)
        self.ishape = self.data.shape

        # create the figure if needed
        kws, figure_setup = dicts.split(kws, 'fig', 'ax', 'cax', 'hax', 'title', 'figsize')
        self.figure, axes = self.setup_figure(**figure_setup, data=image)
        self.ax, self.cax, self.hax = axes
        ax = self.ax

        # get colour limits
        kws['vmin'], kws['vmax'] = resolve_clim(image, **kws)
        # remove 'extra' keywords that are not allowed in imshow.
        dicts.remove(kws, 'clim', 'plim')

        # use imshow to draw the image
        self.image = ax.imshow(image, *args,
                               **{**kws,
                                  'origin': CONFIG.origin,
                                  'interpolation': CONFIG.interpolation})
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
        hist = dict((CONFIG.hist if hist is True else hist) or {})
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

    #     # setup coordinate display
    #     ax.format_coord = self.format_coord
    #     ax.grid(False)

    def guess_figsize(self, data, fill_factor=CONFIG.fig.fill,
                      max_pixel_size=CONFIG.fig.max_pixel_size):

        image = self.data[0] if data is None else data
        return super().guess_figsize(image, fill_factor, max_pixel_size)

    def connect_cbar_hist(self):
        if not (self.histogram and self.sliders):
            raise ValueError()

        from matplotlib.patches import ConnectionPatch

        connectors = {}
        c0, c1 = self.image.get_clim()
        x0, _ = self.hax.get_xlim()
        mapping = {(1, 0):      (x0, c0),
                   (1, 1):      (x0, c1),
                   (1, 0.5):    (x0, (c0 + c1) / 2)}
        for mv, (xy_from, xy_to) in zip(self.sliders, mapping.items()):
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
        slide = self.sliderClass(self.hax, clim, 'y',
                                 **CONFIG.sliders, use_blit=use_blit)

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

    def set_data(self, data):
        self.image.set_data(data)

    def set_cmap(self, cmap):
        self.image.set_cmap(cmap)
        if self.has_hist:
            self.histogram.set_cmap(cmap)

        # FIXME: blit!
        self.canvas.draw()

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

    # def inset_axes(self):
    #     ax_histx = ax.inset_axes([0, 1.05, 1, 0.25], sharex=ax)


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


class Image3DBase(FigureSetup):

    def __init__(self, image, origin=(0, 0),
                 cmap=None, figure=None,
                 bar3d_kws=None, **image_kws):

        self.figure, (axi, ax3) = self.setup_figure(figure, data=image)
        self.axi, self.ax3 = axi, ax3

        y0, x0 = origin = np.array(origin)
        y1, x1 = origin + image.shape
        y, x = np.indices(image.shape) + origin[:, None, None]

        cmap = cm.get_cmap(cmap)
        self.image = im = self.plot_image(image, cmap, **image_kws,
                                          extent=(x0, x1, y0, y1))

        # remove cbar ticks / labels
        # im.cax.yaxis.set_tick_params(length=0)
        # im.cax.yaxis.major.formatter = ticker.NullFormatter()

        self.bars = bars = self.plot_bars(x, y, image, cmap=cmap,
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

    def setup_figure(self, *args, **kws):

        fig, axes = _, (axi, *_) = super().setup_figure(*args, **kws, subplot=121)

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


class Image3D(Image3DBase):

    def plot_image(self, image, cmap, **kws):
        return ImageDisplay(
            image, ax=self.axi, cmap=cmap,
            **{**dict(hist=False, sliders=False,
                      cbar={'format': ticker.NullFormatter()}),
               **kws}
        )

    def plot_bars(self, x, y, z, **kws):
        return Bar3D(self.ax3, x, y, z, **kws)

    def set_cmap(self, cmap):
        # change cmap for all mappables
        self.image.cbar.scroll.set_cmap(cmap)

    def set_data(self, image, clim=None):

        self.image.set_data(image)

        clim = resolve_clim(image, plim=CONFIG.plim)
        self.image.set_clim(*clim)

        self.bars.set_z(image, clim)


# from recipes.oo.property import cached_property


class MouseOverEffect:
    def __init__(self, ax):
        self.ax = ax

    def on_enter(self, event):
        if self.event.axes != self.ax:
            return


class ZoomInset(LoggingMixin):
    def __init__(self, ax, image, position=(0, 1.05), size=(0.2, 0.2),
                 origin=(0, 0), shape=(16, 16), **kws):
        self.ax = ax
        self.image = image
        self.origin = origin
        self.shape = shape

        ax_histx = ax.inset_axes([*position, *size], sharex=ax)
        self.art = ax.imshow(self.sub, **kws)

    @property
    def origin(self):
        return self._origin

    @origin.setter
    def origin(self, origin):
        self._origin = np.array(origin, int)

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, shape):
        self._shape = np.array(shape, int)

    # @cached_property(depends_on=(origin, shape))
    @property
    def segment(self):
        return tuple(map(slice, self.origin, self.shape))

    @property
    def sub(self):
        return self.image[self.segment]
