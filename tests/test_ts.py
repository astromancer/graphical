"""tests"""

#
# from motley.profiling import profile
#
# profiler = profile()
#
#
# @profile.histogram

import numpy as np

import graphing.utils
from graphing import ts
from matplotlib import pyplot as plt


def test_plims(data):
    mn, mx = data.min(), data.max()
    p20, p30, p50, p80 = np.percentile(data, (20, 30, 50, 80))
    ptp = mx - mn
    md = p50 - mn

    expected = {-2.5: mn - (2 * ptp + md),  # :     +4 * mn - 2 * mx - p50
                -2.0: mn - (2 * ptp),  # :          +3 * mn - 2 * mx
                -1.5: mn - (1 * ptp + md),  # :     +3 * mn - 1 * mx - p50
                -1.2: mn - ptp - (p20 - mn),  # :   +3 * mn - 1 * mx - p20
                -1.0: mn - (1 * ptp),  # :          +2 * mn - 1 * mx
                -0.5: mn - md,  # :                 +2 * mn - 0 * mx - p50
                +0.0: mn,  # :                      +1 * mn - 0 * mx
                +0.3: p30,  # :                     +0 * mn - 0 * mx + p30
                +0.5: p50,  # :                     +0 * mn - 0 * mx + p50
                +1.0: mx,  # :                      +0 * mn + 1 * mx
                +1.2: mx + (mx - p80),  # :         +0 * mn + 2 * mx - p80
                +1.5: mx + (mx - p50),  # :         +0 * mn + 2 * mx - p50
                +2.7: mx + ptp + (mx - p30)  # :    -1 * mn + 3 * mx - p30
                }

    for p, e in expected.items():
        assert graphing.utils.get_percentile(data, p) == e


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
