"""
3D bar plots that actually render properly.
"""

import numpy as np


from recipes.transforms import sph2cart
from loguru import logger


def sphview(ax):
    """returns the camera position for 3D axes in spherical coordinates"""
    r = np.square(np.max([ax.get_xlim(),
                          ax.get_ylim()], 1)).sum()
    theta, phi = np.radians((90 - ax.elev, ax.azim))
    return r, theta, phi


def camera_distance(ax, x, y):
    return np.sqrt(np.square(
        # location of bar base
        [x, y, np.zeros_like(x)] -
        # camera position in xyz
        np.array(sph2cart(*sphview(ax)), ndmin=3).T
    ).sum(0))


class bar3d:
    def __init__(self, ax, x, y, z, dxy=0.8, *args, **kws):
        """
        3D bar plot that actually renders properly.
        """
        assert hasattr(ax, 'bar3d')
        self.ax = ax
        self._xyz = np.array([x, y, np.zeros_like(z)])

        # bar width and bredth
        (dx,), (dy,) = dxy * np.diff(x[0, :2]), dxy * np.diff(y[:2, 0])
        assert (dx != 0) & (dy != 0)

        # "distance" of bar base from camera
        shape = _, n = x.shape
        self.bars = np.empty(shape, dtype=object)
        for i, (xx, yy, dz) in enumerate(zip(*map(np.ravel, (x, y, z)))):
            self.bars[tuple(divmod(i, n))] = \
                ax.bar3d(xx, yy, 0, dx, dy, dz, **kws)
            
        # Redo zorder when rotating axes
        canvas= ax.figure.canvas
        canvas.mpl_connect('motion_notify_event', self.on_rotate)
        self._cid = canvas.mpl_connect('draw_event', self.on_first_draw)

    def on_first_draw(self, event):
        # HACK to get the zorder right at first draw
        self.set_zorder()
        canvas = self.ax.figure.canvas
        canvas.mpl_disconnect(self._cid)
        canvas.draw()
    
    def get_zorder(self):
        return 1. / camera_distance(self.ax, *self._xyz[:2]).ravel()
    
    def set_zorder(self):
        for i, o in enumerate(self.get_zorder()):
            bar = self.bars[tuple(divmod(i, self.bars.shape[1]))]
            bar._sort_zpos = o
            bar.set_zorder(o)
    
    def on_rotate(self, event):  # sourcery skip: de-morgan
        if ((event.inaxes is not self.ax) or
                (self.ax.button_pressed not in self.ax._rotate_btn)):
            return

        logger.debug('Setting zorder on rotate: {}', self.ax)
        self.set_zorder()

# def get_zorder(ax, x, y):
#     camera_distance(ax, x, y)


# def bar3d(ax, x, y, z, dxy=0.8, **kws):
#     """
#     3D bar plot that actually renders properly.
#     """
#     assert hasattr(ax, 'bar3d')

#     # bar width and bredth
#     (dx,), (dy,) = dxy * np.diff(x[0, :2]), dxy * np.diff(y[:2, 0])
#     assert (dx != 0) & (dy != 0)

#     # "distance" of bar base from camera
#     zo = 1. / camera_distance(ax, x, y)
#     shape = _, n = x.shape
#     bars = np.empty(shape, dtype=object)
#     for i, (xx, yy, dz, o) in enumerate(zip(*map(np.ravel, (x, y, z, zo)))):
#         bars[tuple(divmod(i, n))] = art = ax.bar3d(xx, yy, 0, dx, dy, dz, **kws)
#         art._sort_zpos = o



#     return bars
