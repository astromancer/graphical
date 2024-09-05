"""
Plot image models.
"""

# std
from collections import abc

# third-party
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.colors import LightSource
from loguru import logger
from mpl_toolkits.mplot3d import art3d
from mpl_toolkits.axes_grid1 import AxesGrid

# local
from recipes.config import ConfigNode
from recipes.functionals import echo0

# relative
from ..utils import emboss
from ..depth.prisms import Bar3D
from ..moves.callbacks import CallbackManager, mpl_connect
from . import ImageDisplay
from .utils import resolve_clim
from recipes.decorators import update_defaults

# ---------------------------------------------------------------------------- #
# Load config
CONFIG = ConfigNode.load_module(__file__)




# ---------------------------------------------------------------------------- #

class ImageBar3D(CallbackManager):

    def __init__(self, data, fig=None, image_kws=(), **kws):

        self.fig, self.axes = self.setup_figure(fig, **kws)
        ax1, ax2 = self.axes

        # image
        self.image = ImageDisplay(data, ax=ax1, **CONFIG.image.merge(image_kws),
                                  hist=False, sliders=False, use_blit=False)

        # bars
        nrows, ncols = data.shape
        y, x = np.mgrid[:nrows, :ncols]
        self.bars = Bar3D(ax2, x, y, data,
                          **CONFIG.bars.merge(kws, cmap=CONFIG.image.cmap).coerce(
                              lightsource=LightSource, unpack=True))

        # axes setup
        altaz = CONFIG.axes['3d']
        ax2.set(xlim=(-0.5, nrows), ylim=(-0.5, ncols), **kws)
        ax2.azim, ax2.elev = altaz.azim, altaz.elev

        ax2.tick_params(pad=-1)

        # color limits
        # p = CONFIG.psf.params
        # ylim = (p.amplitude + p.background)

    def setup_figure(self, fig=None, **kws):

        if isinstance(fig, abc.MutableMapping):
            kws = {**fig, **kws}
            fig = None

        if fig is None:
            fig = plt.figure(figsize=CONFIG.image_bar_plot.figure.size, **kws)
            ax1 = fig.add_subplot(1, 2, 1)
            ax2 = fig.add_subplot(1, 2, 2, projection='3d')

        fig.subplots_adjust(**CONFIG.image_bar_plot.figure.margins)

        return fig, (ax1, ax2)

    def set_clim(self, clim):
        self.image.set_clim(clim)
        self.bars.bars.set_clim(clim)
        # self.image.cax.set_ylim(0, ylim)


# ---------------------------------------------------------------------------- #

class ImageModel3DPlot(CallbackManager):
    """
    Base class for plotting image data, model and residual for comparison.
    """

    # TODO: profile & speed up!
    # TODO: blit for view angle change...
    # TODO: optionally Include info as text in figure??????

    # @profile()

    def __init__(self, x=(), y=(), z=(), data=(),
                 fig=None, titles=CONFIG.axes.titles, title_kws=(),
                 image_kws=(), art3d_kws=(), residual_funcs=(),
                 **kws):

        self.art3d = []
        self.images = []
        self.titles = list(titles)
        self.title_kws = dict(title_kws)
        self.fig = self.setup_figure(fig, **kws)

        self.residual_funcs = dict(residual_funcs)
        #
        self.update(x, y, z, data, image_kws, art3d_kws)

        # link viewlims of the 3d axes
        CallbackManager.__init__(self, self.fig.canvas)

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

    def setup_figure(self, fig=None, axes_image=(), axes_3d=(), **kws):
        """
        Initialize grid of 2x3 subplots. Top 3 are colour images, bottom 3 are
        3D wireframe plots of data, fit and residual.
        """

        # Plots for current fit
        fig = fig or plt.figure(**kws)

        if not isinstance(fig, Figure):
            raise TypeError(f'Expected Figure, received {type(fig)}')

        axes_image = CONFIG.axes.image.merge(axes_image)
        self.axes_images = self.setup_image_axes(fig, **axes_image)
        axes_3d = CONFIG.axes['3d'].merge(axes_3d)
        self.axes_3d = self.setup_3d_axes(fig, **axes_3d)

        return fig

    def setup_3d_axes(self, fig, **kws):
        # Create the plot grid for the 3D plots
        # axes_3d = AxesGrid(fig, 212, **self._3d_axes_kws)
        axes_3d = []
        for i in range(4, 7):
            ax = fig.add_subplot(2, 3, i, projection='3d', **kws)
            ax.set_facecolor('None')
            # ax.patch.set_linewidth( 1 )
            # ax.patch.set_edgecolor( 'k' )
            axes_3d.append(ax)

        return axes_3d

    # @update_defaults(config=)
    def setup_image_axes(self, fig, **config):
        # Create the plot grid for the images
        
        kws, cbar = ConfigNode(config).split('cbar')
        ticks = cbar.pop(('cbar', 'ticks'), ())
        kws.update(cbar.transform('_'.join))

        self.axes_images = axes = AxesGrid(fig, 211, nrows_ncols=(1, 3), **kws)

        for i, (ax, cax) in enumerate(zip(axes, axes.cbar_axes)):
            # image
            im = ax.imshow(np.empty((1, 1)), origin='lower')
            self.images.append(im)

            # title above image
            ax.set_title(self.titles[i], **self.title_kws)

            # colorbar
            cbar = cax.colorbar(im)
            self.setup_cbar_ticks(cbar, cax, **ticks)

        return axes

    def setup_cbar_ticks(self, cbar, cax, **config):
        # make the colorbar ticks look nice
            
        config = CONFIG.axes.image.cbar.ticks.merge(config)
        config = ConfigNode(
            major=(major := {'which': 'major', 'axis': 'y', **config.major}),
            minor={**major, **config.minor, 'which': 'minor'},
            right={**major, **config.right}
        )
        
        rightmost = cax is self.axes_images.cbar_axes[-1]
        cax.axes.tick_params(**config['right' if rightmost else 'major'])
        cax.axes.tick_params(**config.minor)

        # make the colorbar spine invisible
        cbar.outline.set_visible(False)
        #
        for w in ('top', 'bottom', 'right')[(2 * (not rightmost)):]:
            cax.spines[w].set_visible(True)
            cax.spines[w].set_color(config.major.colors)
        cax.minorticks_on()

        labels = dict(CONFIG.axes.image.cbar.ticks.labels)
        embossed = labels.pop('emboss', None)
        for t in cax.axes.yaxis.get_ticklabels():
            t.set(**labels)
            if embossed:
                emboss(t, *embossed)

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

        xlims = xlims if (xlims := x[0, [0, -1]]).ptp() else x[[0, -1], 1]
        ylims = ylims if (ylims := y[0, [0, -1]]).ptp() else y[[0, -1], 1]
        zlims = [z.min(), z.max()]

        # image colour limits
        rlims = [res_img.min(), res_img.max()]
        clims = resolve_clim(data, plim=CONFIG.image.plim)
        logger.info('clim {}.', clims)

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


class ImageModelWireframe(ImageModel3DPlot):

    def __init__(self, x=(), y=(), z=(), data=(),
                 fig=None, titles=CONFIG.axes.titles,
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


class ImageModelBar3D(ImageModel3DPlot):

    def __init__(self, *args, dxy=0.8, **kws):
        self.dxy = dxy
        super().__init__(*args, **kws)

    def update_3d(self, x, y, z, data, residual, dxy=0.8, **kws):
        """update plots with new data."""
        # TODO: will be much faster to update the  Poly3DCollection verts

        for bars in self.art3d:
            for bar in bars.ravel():
                bar.remove()

        # print(kws)
        for ax, zz in zip(self.axes_3d, (z, data, abs(residual))):
            bars = Bar3D(ax, x, y, zz, dxy, **kws)
            self.art3d.append(bars)

    def set_clim(self, clim):
        for im in self.images[:-1]:
            im.set_clim(clim)

        for bars in self.art3d[:-1]:
            bars.bars.set_clim(clim)


class ImageModelContour3D(ImageModel3DPlot):
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


# class PSFPlotter(ImageModel3DPlot, VideoDisplay):
#     def __init__(self, filename, model, params, coords, window, **kws):
#         self.model = model
#         self.params = params
#         self.coords = coords
#         self.window = w = int(window)
#         self.grid = np.mgrid[:w, :w]
#         extent = np.array([0, w, 0, w]) - 0.5  # l, r, b, t

#         ImageModel3DPlot.__init__(self)
#         axData = self.axes_images[0]

#         FitsCubeDisplay.__init__(self, filename, ax=axData, extent=extent,
#                                  sidebar=False, figsize=None)
#         self.update(0)  # FIXME: full frame drawn instead of zoom
#         # have to draw here for some bizarre reason
#         # self.axes_images[0].draw(self.fig._cachedRenderer)

#     def get_image_data(self, i):
#         # coo = self.coords[i]
#         data = neighbours(self[i], self.coords[i], self.window)
#         return data

#     def update(self, i, draw=False):
#         """Set frame data. draw if requested """
#         i %= len(self)  # wrap around! (eg. scroll past end ==> go to beginning)
#         i = int(round(i, 0))  # make sure we have an int
#         self._frame = i  # store current frame

#         image = self.get_image_data(i)
#         p = self.params[i]
#         Z = self.model(p, self.grid)
#         Y, X = self.grid
#         self.update(X, Y, Z, image)

#         if draw:
#             self.fig.canvas.draw()

#         return i
