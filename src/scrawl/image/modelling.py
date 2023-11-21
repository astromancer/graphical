

# third-party
import numpy as np
import matplotlib.pylab as plt
from matplotlib.figure import Figure
from loguru import logger
from mpl_toolkits.mplot3d import art3d
from mpl_toolkits.axes_grid1 import AxesGrid

# local
from recipes.functionals import echo0
from recipes.dicts import AttrReadItem
from recipes.array.neighbours import neighbours

# relative
from ..depth.bar3d import bar3d
from ..moves.callbacks import CallbackManager, mpl_connect
from ..video import VideoDisplay


DEFAULT_TITLES = ('Data', 'Fit', 'Residual')


class ImageModelPlot3D(CallbackManager):
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
