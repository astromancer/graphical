"""
Scatter plots with density inlays / Density maps with scatter for sparse areas
"""

# std
import itertools as itt
import copy

# third-party
import numpy as np
import matplotlib.pyplot as plt

# local
from recipes.config import ConfigNode
from recipes.oo.slots import _sanitize_locals
from recipes.utils import duplicate_if_scalar

# relative
from .image import Image3DBase
from .utils import hexbin as _hexbin_helper
from .depth.prisms import PRISM_WORKERS, Bar3D


# ---------------------------------------------------------------------------- #
# module config
CONFIG = ConfigNode.load_module(__file__)  # , 'yaml'


# ---------------------------------------------------------------------------- #


def _sanitize_data(data, allow_dim):
    # sanitize data
    data = np.asanyarray(data).squeeze()
    assert data.size > 0, 'No data!'
    assert data.ndim == allow_dim, f'Invalid dimensionality: {data.ndim}'
    assert data.shape[-1] == 2, f'Invalid shape: {data.shape}'

    # mask nans
    # data = np.ma.MaskedArray(data, np.isnan(data))
    # return data

    # remove nans:
    data = data[~np.isnan(data).any(1)]
    assert len(data) > 0
    return data


def _api_update_kws(default=None, user=None, **add_kws):
    # default arg
    user = {} if user is None else user
    return {**(default or {}), **user, **add_kws}

# ---------------------------------------------------------------------------- #


def hist2d(ax, data, *args, **kws):
    return scatter_map(ax, data, *args, **kws, tessellation='rect')


def hexbin(ax, data, *args, **kws):
    return scatter_map(ax, data, *args, **kws, tessellation='hex')


def scatter_map(ax, data,
                bins=CONFIG.bins, range=None,
                max_points=CONFIG.max_points,
                min_count=CONFIG.min_count,
                tessellation=CONFIG.tessellation,
                cmap=CONFIG.cmap, alpha=CONFIG.alpha,
                scatter_kws=None, density_kws=None, **kws):
    """
    Point cloud visualization combining a density map and scatter plot. Regions
    with high point density are plotted as a 2D histogram image using either
    rectangular or hexagonal tesselation.

    Parameters
    ----------
    ax: Axes

    data: array-like (N, 2)
        Data to histogram.
    bins: int
        Number of bins.
    range:
        Grid range for density map. Points outside this range will be scatter plot.
    max_points: int
        Number of points required to trigger the density map. For data sets
        containing fewer points than this, we only plot the scatter points. To
        always produce pure density plot, set this value to 0. To always only
        produce pure scatter plot set this value to `None` or `np.inf`. 
    min_count: int
        Point density threshold for triggering density plot instead of
        scatter in a bin cell. Bins containing more points than this number will
        be plotted as density cells. Points in sparse regions will be plotted as
        actual markers. For pure scatter plot set this value `None` or `np.inf`.
        For pure density map, set `min_count` to 0.
    tessellation: str
        Shape of the bins.
    scatter_kws
    density_kws

    Returns
    -------

    """

    data = _sanitize_data(data, 2)

    # plot only scatter if small amount of data
    if max_points is None and min_count == 0:
        raise ValueError()

    if len(data) < max_points:
        min_count = None

    if (worker := MAP_WORKERS.get(tessellation.lower())) is None:
        raise ValueError(f'Invalid tessellation {tessellation!r}: Valid '
                         f'choices are {MAP_WORKERS}')

    # set default args
    shared = dict(cmap=cmap, alpha=alpha)
    scatter_kws = _api_update_kws(CONFIG.scatter, scatter_kws, **shared)
    density_kws = _api_update_kws(CONFIG.density, density_kws, **shared, **kws)

    # do density plot
    art, hvals, xy_scatter = None, [], data
    if min_count and np.isfinite(min_count):
        art, hvals, xy_scatter = worker(ax, data, bins, range, min_count, density_kws)

    # plot scatter points
    # set colour of markers to match cmap
    points = ax.scatter(*xy_scatter.T,
                        **scatter_kws,
                        c=np.ones(xy_scatter.shape[0]))

    return hvals, art, points

    # return worker(ax, data, bins, range, min_count, cmap,
    #               scatter_kws, density_kws)

    # div = make_axes_locatable(ax)
    # cax = div.append_axes('right', '5%')
    # cbar = ax.figure.colorbar(im, cax)
    # cbar.ax.set_ylabel('Density')

    # ax.set_title('Coord scatter')
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.grid()

    # return returns


# aliases
map = density_map = map_scatter = scatter_map

# ---------------------------------------------------------------------------- #


def _hist2d(ax, data, bins, range, min_count, density_kws):
    bins = duplicate_if_scalar(bins)

    # plot density map # x_edges, y_edges
    hvals, *_, qmesh = ax.hist2d(*data.T, bins, range, **density_kws)
    xy_scatter = _remove_sparse_cells_qmesh(qmesh, data, min_count)
    return qmesh, hvals, xy_scatter


def _remove_sparse_cells_qmesh(qmesh, data, min_count):
    # remove low density faces
    z = qmesh.get_array()
    xy_scatter = _get_sparse_points(data.T, z.T,
                                    (qmesh._coordinates[0, :, 0],
                                     qmesh._coordinates[:, 0, 1]),
                                    min_count)

    # set cmap under color transparent
    _adapt_cmap(qmesh, min_count)

    return xy_scatter


def _get_sparse_points(points, bincounts, edges, min_count):
    x, y = points
    x_edges, y_edges = edges
    
    # select points within the range
    bins = bincounts.shape
    ix_x = np.digitize(x, x_edges)
    ix_y = np.digitize(y, y_edges)
    ind = ((ix_x > 0) & (ix_x <= bins[0]) &
           (ix_y > 0) & (ix_y <= bins[1]))
    # low density points
    counts = bincounts[ix_x[ind] - 1, ix_y[ind] - 1]
    return points.T[ind][counts < min_count]


def _adapt_cmap(art, min_count):
    # copy the art (avoid deprecation warning for mpl 3.3)
    cm = copy.copy(art.get_cmap())
    # make the bins with few points invisible
    cm.set_under(art.axes.get_fc(), alpha=0)
    art.set(cmap=cm, clim=min_count)


# ---------------------------------------------------------------------------- #


def _hexbin(ax, data, bins, range, min_count, density_kws):
    # do density plot
    sparse_point_indices = []

    extent = None
    if range is not None:
        extent = np.r_[range]
        assert len(extent) == 4

    def collect_indices(idx):
        counts = len(idx)
        if counts < min_count:
            sparse_point_indices.extend(idx)
        return counts

    # plot density map
    indices = np.arange(len(data))
    polygons = ax.hexbin(*data.T, indices,
                         gridsize=bins,
                         reduce_C_function=collect_indices,
                         extent=extent,
                         **density_kws)

    hvals = polygons.get_array()

    # set cmap under color transparent
    _adapt_cmap(polygons, min_count)

    return polygons, hvals, data[sparse_point_indices]


def _hexbin_sparse(data, bins, range, min_count):
    sparse_point_indices = []

    extent = None
    if range is not None:
        extent = np.r_[range]
        assert len(extent) == 4

    def collect_indices(idx):
        counts = len(idx)
        if counts < min_count:
            sparse_point_indices.extend(idx)
        return counts

    # plot density map
    indices = np.arange(len(data))
    hexbin_data = _hexbin_helper(
        *data.T, indices,
        gridsize=bins,
        reduce_C_function=collect_indices,
        extent=extent,
    )
    return hexbin_data, sparse_point_indices


# ---------------------------------------------------------------------------- #
MAP_WORKERS = dict(
    hex=_hexbin,
    rect=_hist2d
)


# ---------------------------------------------------------------------------- #


def bar3d(ax, data,
          bins=CONFIG.bins, range=None,
          max_points=CONFIG.max_points,
          min_count=CONFIG.min_count,
          tessellation=CONFIG.tessellation,
          cmap=CONFIG.cmap, alpha=CONFIG.alpha,
          scatter_kws=None, density_kws=None, *kws):

    # set default args
    # set default args
    shared = dict(cmap=cmap, alpha=alpha)
    scatter_kws = _api_update_kws(CONFIG.scatter, scatter_kws, **shared)
    density_kws = _api_update_kws(CONFIG.density, density_kws, **shared, **kws)

    if len(data) > max_points:
        if tessellation == 'hex':
            # histogram
            ((x, y, z), (xmin, xmax), (ymin, ymax), (nx, ny)), sparse_point_indices = \
                _hexbin_sparse(data, bins, range, min_count)
            dxy = np.array([(xmax - xmin) / nx,
                            (ymax - ymin) / ny / np.sqrt(3)]) * 0.9

            xy_scatter = data[sparse_point_indices].T

            # remove low density cells
            x, y, z = np.take([x, y, z], *np.where(z >= min_count), 1)
        else:
            # histogram
            z, x_edges, y_edges = np.histogram2d(*data.T, bins, range)
            x, y = x_edges[:-1], y_edges[:-1]
            dxy = np.diff([x[:2], y[:2]], axis=0)

            # remove low density cells
            xy_scatter = _remove_sparse_cells((x, y, z),
                                              (x_edges, y_edges),
                                              min_count)

        #
        Bar3dClass = PRISM_WORKERS[tessellation]
        bars = Bar3dClass(x, y, z, dxy=dxy, **density_kws)
        ax.add_collection(bars)

        viewlim = np.array([(np.min(x), np.max(np.add(x, bars.dx))),
                            (np.min(y), np.max(np.add(y, bars.dy))),
                            (bars.z0, np.max(np.add(bars.z0, z)))])
        if tessellation == 'hex':
            viewlim[:2, 0] = viewlim[:2, 0] - np.array([bars.dx, bars.dy] / 2).T

        ax.auto_scale_xyz(*viewlim, False)

        scatter_kws['zorder'] = bars.get_zorder() + 1

    else:
        *xy_scatter, _ = data.T

    # set default colour of markers to match colormap
    # scatter_kws.setdefault('color', polygons.get_cmap()(0))
    points = ax.scatter3D(*xy_scatter, **scatter_kws,
                          c=np.ones(xy_scatter.shape[1]))

    return bars, points


def map23(data,
          bins=CONFIG.bins, range=None,
          max_points=CONFIG.max_points,
          min_count=CONFIG.min_count,
          tessellation=CONFIG.tessellation,
          cmap=CONFIG.cmap, alpha=CONFIG.alpha,
          scatter_kws=None, density_kws=None):

    # 2D
    fig = plt.figure()
    ax = fig.add_subplot(121)

    # call below updates `scatter_kws` with the alpha value as well as default
    # colour of markers to match colormap
    z, art, points = scatter_map(ax, data, bins, range,
                                 max_points, min_count,
                                 tessellation,
                                 cmap, alpha,
                                 scatter_kws, density_kws)

    # # 3D
    ax3 = fig.add_subplot(122, projection='3d')
    bars, points3d = bar3d(ax3, data, bins, range,
                           max_points, min_count,
                           tessellation,
                           cmap, alpha,
                           scatter_kws, density_kws)

    # x, y = art.get_offsets().T
    # dxy = art.get_paths()[0].vertices.ptp(0)

    # Bar3D = PRISM_WORKERS[tessellation]

    # bars = Bar3D(x, y, z, dxy=dxy, cmap=art.get_cmap(), **density_kws)
    # ax3.add_collection(bars)

    # #
    # points3d, = ax3.plot(*points.get_data(), 0, **scatter_kws)

    # viewlim = np.array([(np.min(x), np.max(np.add(x, bars.dx))),
    #                     (np.min(y), np.max(np.add(y, bars.dy))),
    #                     (bars.z0, np.max(np.add(bars.z0, z)))])
    # viewlim[:2, 0] = viewlim[:2, 0] - np.array([bars.dx / 2, bars.dy / 2]).T

    # ax3.auto_scale_xyz(*viewlim, False)

    return fig, (art, points), (bars, points3d)
