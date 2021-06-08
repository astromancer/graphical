"""
Bezier curve recipe
"""

# adapted from:
#   https://stackoverflow.com/a/12644499/1098683

import numpy as np
from scipy.misc import comb


def bernstein_poly(i, n, t):
    """
    The iᵗʰ Bernstein basis polynomial of degree n, i as a function of t
    """

    # return comb(n, i) * (t ** (n - i)) * (1 - t) ** i
    return comb(n, i) * (t ** i) * ((1 - t) ** (n - i))


def bezier_curve(points, nTimes=50):
    """
    Given a set of control points, return the
    bezier curve defined by the control points.

    points should be a list of lists, or list of tuples
    such as [[1,1],
             [2,3],
             [4,5],
             ..
             [Xn, Yn]]
    nTimes is the number of time steps, defaults to 1000

    See http://processingjs.nihongoresources.com/bezierinfo/
    """

    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0, 1, nTimes)

    polynomial_array = np.array(
            [bernstein_poly(i, nPoints - 1, t) for i in range(0, nPoints)])

    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    return xvals, yvals


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    nPoints = 3
    points = np.random.rand(nPoints, 2) * 200
    xpoints = [p[0] for p in points]
    ypoints = [p[1] for p in points]

    xvals, yvals = bezier_curve(points)
    plt.plot(xvals, yvals)
    plt.plot(xpoints, ypoints, "ro")
    for nr in range(len(points)):
        plt.text(points[nr][0], points[nr][1], nr)

    plt.show()
