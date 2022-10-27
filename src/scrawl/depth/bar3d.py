"""
3D bar plots that actually render correctly.
"""

# std
import numbers
import itertools as itt

# third-party
import numpy as np
from loguru import logger
from matplotlib.cm import ScalarMappable

# relative
from ..moves import CallbackManager, TrackAxesUnderMouse, mpl_connect
from . import camera
from .zaxis_cbar import ZAxisCbar


# ---------------------------------------------------------------------------- #
# shape (6, 4, 3)
# All faces are oriented facing outwards - when viewed from the
# outside, their vertices are in a counterclockwise ordering.
CUBOID = np.array([
    # -z
    (
        (0, 0, 0),
        (0, 1, 0),
        (1, 1, 0),
        (1, 0, 0),
    ),
    # +z
    (
        (0, 0, 1),
        (1, 0, 1),
        (1, 1, 1),
        (0, 1, 1),
    ),
    # -y
    (
        (0, 0, 0),
        (1, 0, 0),
        (1, 0, 1),
        (0, 0, 1),
    ),
    # +y
    (
        (0, 1, 0),
        (0, 1, 1),
        (1, 1, 1),
        (1, 1, 0),
    ),
    # -x
    (
        (0, 0, 0),
        (0, 0, 1),
        (0, 1, 1),
        (0, 1, 0),
    ),
    # +x
    (
        (1, 0, 0),
        (1, 1, 0),
        (1, 1, 1),
        (1, 0, 1),
    ),
])
# ---------------------------------------------------------------------------- #


def is_one_colour(c):
    return (isinstance(c, (str, numbers.Real)) or np.size(c) in {3, 4})


class Bar3D(CallbackManager):
    """
    3D bar plot that renders correctly for different viewing angles.
    """
    #  TODO inherit from collection ??

    @classmethod
    def from_image(cls, ax, image, dxy=0.8, **kws):
        origin = np.array(image.get_extent())[[0, 2]]
        y, x = np.indices(image.shape) + origin[:, None, None]
        cmap = image.get_cmap()
        return cls(ax, x, y, image.get_array(), dxy,
                   color=cmap(image.norm(image)),
                   **kws)

    def __init__(self, ax, x, y, z, dxy=0.8,
                 color=None, cmap=None, norm=None, vmin=None, vmax=None,
                 zaxis_cbar=False,
                 **kws):
        #
        assert hasattr(ax, 'bar3d')
        assert 0 < dxy <= 1

        self.sm = ScalarMappable(norm, cmap)
        self.set_cmap = self.sm.set_cmap

        self.ax = ax
        xyz = np.array([x, y, z])  # .reshape(3, -1)
        self.xy, self.z = xyz[:2], xyz[-1]
        self.dxy = float(dxy)

        # bar width and breadth
        (dx,) = self.dxy * np.diff(x[0, :2])
        (dy,) = self.dxy * np.diff(y[:2, 0])
        assert (dx != 0) & (dy != 0)

        if color is None:
            self.set_cmap(cmap)
            # self.set_array(z)
            self.sm.set_clim(vmin, vmax)
            color = self.sm.to_rgba(z)

        if not (color is None or is_one_colour(color)):
            color = np.asanyarray(color)
            if color.shape[:x.ndim] == x.shape:
                color = color.reshape(-1, 4)
        else:
            color = itt.repeat(color)

        # "distance" of bar base from camera
        shape = _, n = x.shape
        self.bars = np.empty(shape, dtype=object)

        for i, (xx, yy, dz, c) in enumerate(zip(*xyz.reshape(3, -1), color)):
            self.bars[tuple(divmod(i, n))] = \
                ax.bar3d(xx, yy, 0, dx, dy, dz, color=c, **kws)

        self.cbar = None
        if zaxis_cbar:
            self.cbar = ZAxisCbar(self.ax, cmap)

        # 
        TrackAxesUnderMouse.__init__(self, ax.figure.canvas, connect=True)

        # embed()

        # self.sm.callbacks.connect('changed', self._update_bars)

    @mpl_connect('draw_event', 1)
    def on_first_draw(self, _):
        # HACK to get the zorder right at first draw
        self.set_zorder()
        self.remove_callback('draw_event', 1)
        self.ax.figure.canvas.draw()

    def set_array(self, z):
        self.sm.set_array(z)
        self._update_bars(z)

    # def set_cmap(self, cmap):
        # self.sm.set_cmap
        # # in_init = self.cmap
        # in_init = self.cmap is None
        # super().set_cmap(cmap)
        # if not in_init:
        #     self._update_bars(self.z)

    def _update_bars(self, z=None):
        if z is not None:
            self.z = z

        # indexed by [bar, face, vertex, coord]
        shape = (*self.bars.shape, 1, 1, 3)
        xyz = np.zeros(shape)
        xyz[..., 0, 0, :2] = np.moveaxis(self.xy, 0, -1)
        dxyz = np.full(shape, self.dxy)
        dxyz[..., 0, 0, -1] = z

        polys = xyz + dxyz * CUBOID
        # collapse the first two axes
        polys = polys.reshape((-1,) + polys.shape[2:])

        colours = self.to_rgba(z).reshape(-1, 4)
        for bar, verts, c in zip(self.bars.ravel(), polys, colours):
            bar.set_verts(verts)
            bar.set_color(
                self.ax._shade_colors(c, self.ax._generate_normals(verts))
            )

    def get_zorder(self):
        return -camera.distance(self.ax, *self.xy).ravel()

    def set_zorder(self):
        for i, o in enumerate(self.get_zorder()):
            bar = self.bars[tuple(divmod(i, self.bars.shape[1]))]
            bar._sort_zpos = o
            bar.set_zorder(o)

    @mpl_connect('motion_notify_event')
    def on_rotate(self, event):  # sourcery skip: de-morgan
        # Redo zorder when rotating axes
        if ((event.inaxes is not self.ax) or
                (self.ax.button_pressed not in self.ax._rotate_btn)):
            return

        logger.debug('Setting zorder on rotate: {}', self.ax)
        self.set_zorder()


# alias
bar3d = Bar3d = Bar3D
