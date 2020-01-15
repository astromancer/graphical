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


def test_plot(**kws):
    # generate some data
    n = 250
    np.random.seed(666)
    t = np.linspace(0, 2 * np.pi, n)
    y = [np.sin(3 * t),
         # np.cos(10*t),
         np.cos(10 * np.sqrt(t))]
    e = np.random.randn(len(y), n)
    m = np.random.rand(len(y), n) > 0.8

    # case 1:    bare minimum
    # print('CASE1')
    # tsp = ts.plot(y[0], **kws)

    # case 2:    multiple series, no time
    print('CASE2')
    tsp = ts.plot(y, **kws)

    # case 3:    multiple series, single time
    print('CASE3')
    tsp = ts.plot(t, y, **kws)

    # case 4:    full args
    print('CASE4')
    t2 = np.power(t, np.c_[1:len(y) + 1])  # power stretch time
    tsp = ts.plot(t2, y, e, **kws)

    # case 5: masked data
    print('CASE5')
    ym = np.ma.array(y, mask=m)
    tsp = ts.plot(t, ym, e,
                  show_masked='x',
                  **kws)

    # case 4: masked data
    # mask = [(np.pi/4*(1+i) < t)&(t > np.pi/8*(1+i)) for i in range(len(y))]
    # ym = np.ma.array(y, mask=mask)
    # fig, plots, *stuff = ts.plot(t, ym, e,
    # show_masked='span')
    # FIXME:
    # File "/home/hannes/.local/lib/python3.4/site-packages/draggable/errorbars.py", line 257, in __init__
    # self.to_orig[handel.markers] = NamedErrorbarContainer(origart)
    # AttributeError: 'Rectangle' object has no attribute 'markers'

    # #case 4: non-uniform data
    # y2 = [_[:np.random.randint(int(0.7*N), N)] for _ in y]
    # #fig, plots, *stuff = ts.plot(y2)
    #
    # t2 = [t[:len(_)] for _ in y2]
    # fig, plots, *stuff = ts.plot(t2, y2, **kws)
    #
    # e2 = [np.random.randn(len(_)) for _ in y2]
    # fig, plots, *stuff = ts.plot(t2, y2, e2, **kws)

    # case: non-uniform masked data
    # except Exception as err:
    plt.show()
    # raise err

    # TODO: more tests


# tests()
# everything with histogram #NOTE: significantly slower
# tests(show_hist=True)
# plt.show()

test_plot()
