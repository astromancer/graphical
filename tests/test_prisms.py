
# std
import itertools as itt

# third-party
import pytest
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import CSS4_COLORS
from scipy.stats import multivariate_normal

# local
from scrawl.depth.prisms import Bar3DCollection, HexBar3DCollection, hexbin
# from mpl_toolkits.mplot3d.art3d import Bar3DCollection


# ---------------------------------------------------------------------------- #
# helper functions

def get_gaussian_bars(mu=(0, 0),
                      sigma=([0.8,  0.3],
                             [0.3,  0.5]),
                      range=(-3, 3),
                      res=2 ** 3,
                      seed=123):
    np.random.seed(seed)
    rv = multivariate_normal(mu, np.array(sigma))
    sl = slice(*range, complex(res))
    xy = np.array(np.mgrid[sl, sl][::-1])
    z = rv.pdf(xy.transpose(1, 2, 0)).T

    return *xy, z


def get_gaussian_hexs(mu=(0, 0),
                      sigma=([0.8,  0.3],
                             [0.3,  0.5]),
                      n=10_000,
                      res=8,
                      seed=123):
    np.random.seed(seed)
    rv = multivariate_normal(mu, np.array(sigma))
    xyz, (xmin, xmax), (ymin, ymax), (nx, ny) = hexbin(*rv.rvs(n).T, gridsize=res)
    dxy = np.array([(xmax - xmin) / nx, (ymax - ymin) / ny / np.sqrt(3)]) * 0.9
    return *xyz, dxy


data_generators = {
    Bar3DCollection: get_gaussian_bars,
    HexBar3DCollection: get_gaussian_hexs
}


# ---------------------------------------------------------------------------- #
# fixtures

@pytest.fixture(params=(HexBar3DCollection, Bar3DCollection))  #
def bar3d_class(request):
    return request.param


# ---------------------------------------------------------------------------- #
# tests

mpl_image_compare = pytest.mark.mpl_image_compare(baseline_dir='images',
                                                  style='default')


@mpl_image_compare
def test_bar3d_1d_data(bar3d_class):
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    _plot_bar3d(ax, bar3d_class, 0, 0, 1, ec='0.5', lw=0.5)
    return fig


@pytest.mark.parametrize('z0', (0, 1))
@mpl_image_compare
def test_bar3d_zsort(bar3d_class, z0):

    fig, axes = plt.subplots(2, 4, subplot_kw={'projection': '3d'})
    elev = 45
    azim0, astep = 0, 45  # -22.5
    camera = itt.product(np.r_[azim0:(180 + azim0):astep], (elev, 0))
    # sourcery skip: no-loop-in-tests
    for ax, (azim, elev) in zip(axes.T.ravel(), camera):
        _plot_bar3d(ax, bar3d_class,
                    [0, 1], [0, 1], [1, 2],
                    z0=z0,
                    azim=azim, elev=elev,
                    ec='0.5', lw=0.5)

    return fig


@mpl_image_compare
def test_bar3d_with_2d_data(bar3d_class):
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    _plot_bar3d(ax, bar3d_class, *data_generators[bar3d_class](),
                ec='0.5', lw=0.5)
    return fig


@pytest.mark.parametrize('shade', (0, 1))
@mpl_image_compare
def test_bar3d_colors(bar3d_class, shade):
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

    xyz = data_generators[bar3d_class]()
    bars = _plot_bar3d(ax, bar3d_class, *xyz,
                       facecolors=list(CSS4_COLORS)[:xyz[0].size],
                       edgecolors='0.5', lw=0.5,
                       shade=shade)
    return fig


@pytest.mark.parametrize('shade', (0, 1))
@mpl_image_compare
def test_bar3d_cmap(bar3d_class, shade):
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

    xyz = data_generators[bar3d_class]()
    bars = _plot_bar3d(ax, bar3d_class, *xyz,
                       cmap='viridis',
                       shade=shade,
                       edgecolors='0.5', lw=0.5)
    return fig


def _plot_bar3d(ax, kls, x, y, z, dxy='0.8', azim=None, elev=None, **kws):

    bars = kls(x, y, z, dxy=dxy, **kws)
    ax.add_collection(bars)

    viewlim = np.array([(np.min(x), np.max(np.add(x, bars.dx))),
                        (np.min(y), np.max(np.add(y, bars.dy))),
                        (bars.z0, np.max(np.add(bars.z0, z)))])

    if kls is HexBar3DCollection:
        viewlim[:2, 0] = viewlim[:2, 0] - np.array([bars.dx / 2, bars.dy / 2]).T

    ax.auto_scale_xyz(*viewlim, False)
    # ax.set(xlabel='x', ylabel='y', zlabel='z')

    if azim:
        ax.azim = azim
    if elev:
        ax.elev = elev

    return bars

# _test_bar3d_with_2d_data(Bar3DCollection)
# fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
# _plot_bar3d(ax, HexBar3DCollection, *data_generators[HexBar3DCollection](),
#                 ec='0.5', lw=0.5)
# fig.savefig('prisms.png')
# plt.show()


# if __name__ == '__main__':
#     # bars = test_hex3d_basic()
#     # plt.show()

#     fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
#     bar3d_class = HexBar3DCollection
#     xyz = data_generators[bar3d_class]()
#     bars = _plot_bar3d(ax, bar3d_class,
#                        [0, 1], [0, 1], [1, 2],
#                     #    dxy=1,
#                        cmap='rainbow', alpha=0.35)


# # if __name__ == '__main__':
# #     # [0, 1], [0, 1], [1, 2]
# #     test_bar3d([0, 1], [0, 1], [1, 2])

# #     # test_bar3d()
#     plt.show()
