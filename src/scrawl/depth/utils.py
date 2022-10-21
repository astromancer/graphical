

# third-party
import numpy as np

# local
from recipes.transforms import sph2cart


def sphview(ax):
    """
    Returns the camera position for 3D axes in spherical coordinates.
    """
    r = np.square(np.max([ax.get_xlim(),
                          ax.get_ylim()], 1)).sum()
    theta, phi = np.radians((90 - ax.elev, ax.azim))
    return r, theta, phi


def camera_distance(ax, x, y, z=None):
    z = np.zeros_like(x) if z is None else z
    return np.multiply(
        # location of points
        [x, y, z],
        # camera position in xyz
        np.array(sph2cart(*sphview(ax)), ndmin=3).T
    ).sum(0)

# return np.sqrt(np.square(
    #     # location of points
    #     [x, y, z] -
    #     # camera position in xyz
    #     np.array(sph2cart(*sphview(ax)), ndmin=3).T
    # ).sum(0))
