

# third-party
import numpy as np

# local
from recipes.math.transforms import sph2cart


def sphview(ax):
    """
    Returns the camera position for 3D axes in spherical coordinates.
    """
    r = np.square(np.max([ax.get_xlim(),
                          ax.get_ylim()], 1)).sum()
    theta, phi = np.radians((90 - ax.elev, ax.azim))
    return r, theta, phi


def xyz(ax):
    return sph2cart(*sphview(ax))


def distance(ax, x, y, z=None):
    z = np.zeros_like(x) if z is None else z
    # camera = xyz(ax)
    # print(camera)
    return np.sqrt(np.square(
        # location of points
        [x, y, z] -
        # camera position in xyz
        np.array(xyz(ax), ndmin=x.ndim + 1).T
    ).sum(0))
