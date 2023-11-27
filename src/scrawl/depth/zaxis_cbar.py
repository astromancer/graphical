
# third-party
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# local
from recipes.array import fold

# relative
from ..moves.callbacks import CallbackManager, mpl_connect


class ZAxisCbar(CallbackManager):
    @classmethod
    def from_image(cls, image, corner=(0, 1), nseg=50, **kws):
        return cls(image.axes, image.get_cmap(), image.get_clim(), corner, nseg,
                   **kws)

    from_scalar_mappable = from_image

    def __init__(self, ax, cmap=None, zrange=(), corner=(0, 1), nseg=50, **kws):
        self.ax = ax
        self.xyz = xyz = np.empty((3, nseg))
        xy = np.array([ax.get_xlim()[corner[0]],
                       ax.get_ylim()[corner[1]]])
        xyz[:2] = np.array(xy, ndmin=2).T
        xyz[2] = np.linspace(*(zrange or ax.get_zlim()), nseg)

        self.line = Line3DCollection(fold.fold(xyz.T, 2, 1, pad=False),
                                     cmap=cmap,
                                     array=xyz[2],
                                     **{**dict(zorder=10,
                                               lw=3),
                                        **kws})

        ax.add_collection(self.line, autolim=False)
        ax.zaxis.line.set_visible(False)

        CallbackManager.__init__(self, self.ax.figure.canvas, connect=True)

    @mpl_connect('motion_notify_event')
    def update(self, event=None):
        if self.ax.M is not None:
            zax = self.ax.zaxis
            mins, maxs, *_, highs = zax._get_coord_info(self.ax.figure._cachedRenderer)
            (x1, y1, _), _ = zax._get_axis_line_edge_points(
                np.where(highs, maxs, mins),
                np.where(~highs, maxs, mins)
            )

            self.xyz[:2] = np.array((x1, y1), ndmin=2).T

        self.line.set_segments(fold.fold(self.xyz.T, 2, 1, pad=False))
        # self.line.do_3d_projection()
