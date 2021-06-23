"""
Bezier curve recipe
"""

# adapted from:
#   https://stackoverflow.com/a/12644499/1098683

import numpy as np
from scipy.misc import comb


def bernstein(i, n, t):
    """
    The iᵗʰ Bernstein basis polynomial of degree n, i as a function of t
    """

    # return comb(n, i) * (t ** (n - i)) * (1 - t) ** i
    return comb(n, i) * (t ** i) * ((1 - t) ** (n - i))


def bezier_curve(points, steps=50):
    """
    Given a set of control points, return the
    bezier curve defined by the control points.

    points should be a list of lists, or list of tuples
    such as [[1,1],
             [2,3],
             [4,5],
             ..
             [Xn, Yn]]
    steps is the number of time steps, defaults to 1000

    See http://processingjs.nihongoresources.com/bezierinfo/
    """

    n = len(points)
    x, y = np.array(points).T
    t = np.linspace(0, 1, steps)
    polys = np.array([bernstein(i, n - 1, t) for i in range(n)])
    return (np.dot(x, polys), np.dot(y, polys))


# if __name__ == "__main__":
#     from matplotlib import pyplot as plt

#     nPoints = 3
#     points = np.random.rand(nPoints, 2) * 200
#     xpoints = [p[0] for p in points]
#     ypoints = [p[1] for p in points]

#     xvals, yvals = bezier_curve(points)
#     plt.plot(xvals, yvals)
#     plt.plot(xpoints, ypoints, "ro")
#     for nr in range(len(points)):
#         plt.text(points[nr][0], points[nr][1], nr)

#     plt.show()
