"""tests"""

#
# from motley.profiling import profile
#
# profiler = profile()
#
#
# @profile.histogram

import numpy as np

from graphing import ts
from matplotlib import pyplot as plt
import pytest

# class TestTSPlot(object):

# generate some data
n = 250
n2 = 100
np.random.seed(666)
t = np.linspace(0, 2 * np.pi, n)
t2 = np.linspace(0, np.pi, n2)
y = [np.sin(3 * t),
     # np.cos(10*t),
     np.cos(10 * np.sqrt(t))]
y2 = np.random.rand(n2)
e = np.random.randn(len(y), n)
m = np.random.rand(len(y), n) > 0.8
ym = np.ma.array(y, mask=m)


@pytest.mark.parametrize(
        "args",
        [  # basic
            (y[0],),
            # multiple series by index
            (y,),
            # multiple series, single time vector
            (t, y),
            # multiple series with uncertainties, single time vector
            (t, y, e),
            # masked data
            (t, ym, e),  # show_masked='x',
            #  multiple series non-uniform sizes
            ([t, t2], [ym[0], y2], [e[1], None])
        ]
)
def test_plot(args, **kws):
    tsp = ts.plot(*args, **kws)


# kws = {}
# tsp = ts.plot(y[0], **kws)
# tsp = ts.plot(y, **kws)
# tsp = ts.plot(t, y, **kws)
# tsp = ts.plot(t, y, e, **kws)
# tsp = ts.plot(t, ym, e, show_masked='x', **kws)
# tsp = ts.plot([t, np.arange(n2)],
#               [ym[0], np.random.rand(n2)],
#               [e[1], None],
#               **kws)


plt.show()
# raise err

# TODO: more tests
#  everything with histogram # NOTE: significantly slower
#  test raises
