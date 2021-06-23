import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from recipes.transforms import sph2cart


def sphview(ax):
    """returns the camera position for 3D axes in spherical coordinates"""
    r = np.square(np.max([ax.get_xlim(),
                          ax.get_ylim()], 1)).sum()
    theta, phi = np.radians((90 - ax.elev, ax.azim))
    return r, theta, phi


# class bar3d:
#     def __init__(self, ax, x, y, z, dxy=0.8, *args, **kws):
#         self.ax = ax
#         self._xyz = np.array([x, y, np.zeros_like(z)])
        
#         zo = self.get_camera_distance()
        
#         (dx,), (dy,) = dxy * np.diff(x[0, :2]), dxy * np.diff(y[:2, 0])
#         assert (dx != 0) & (dy != 0)
#         bars = np.empty(x.shape, dtype=object)
#         n = x.shape[1]
#         for i, (xx, yy, dz, o) in enumerate(zip(*map(np.ravel, (x, y, z, zo)))):
#             j, k = divmod(i, n)
#             bars[j, k] = pl = ax.bar3d(xx, yy, 0, dx, dy, dz, *args, **kws)
#             pl._sort_zpos = o

#     def get_camera_distance(self):
#         # camera position in xyz
#         xyz = np.array(sph2cart(*sphview(self.ax)), ndmin=3).T
#         # "distance" of bars from camera
#         return np.multiply(self._xyz, xyz).sum(0)


def bar3d(ax, x, y, z, dxy=0.8, *args, **kws):
    # camera position in xyz
    xyz = np.array(sph2cart(*sphview(ax)), ndmin=3).T
    # "distance" of bars from camera
    zo = np.multiply([x, y, np.zeros_like(z)], xyz).sum(0)

    (dx,), (dy,) = dxy * np.diff(x[0, :2]), dxy * np.diff(y[:2, 0])
    assert (dx != 0) & (dy != 0)
    
    bars = np.empty(x.shape, dtype=object)
    n = x.shape[1]
    for i, (xx, yy, dz, o) in enumerate(zip(*map(np.ravel, (x, y, z, zo)))):
        j, k = divmod(i, n)
        bars[j, k] = pl = ax.bar3d(xx, yy, 0, dx, dy, dz, *args, **kws)
        pl._sort_zpos = o

    return bars
