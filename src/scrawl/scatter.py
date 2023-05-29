"""
Scatter plots with density inlays.
"""

# std
import copy

# third-party
import numpy as np

# local
from recipes.config import ConfigNode
from recipes.utils import duplicate_if_scalar


# ---------------------------------------------------------------------------- #
# module config
KNOWN_TESSELATIONS = {'hex', 'rect'}

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


def density(ax, data,
            bins=CONFIG.bins, range=None,
            min_count=CONFIG.min_count, max_points=CONFIG.max_points,
            tessellation=CONFIG.tessellation,
            cmap=CONFIG.cmap, alpha=CONFIG.alpha,
            scatter_kws=None, density_kws=None):
    """
    Point cloud visualization combining a density map and scatter plot. Regions
    with high point density are plotted as a 2D histogram image using either
    rectangular or hexagonal tesselation.

    Parameters
    ----------
    ax
    data
    bins: int
    range
    max_points:
        Maximum number of points to plot. For sets with fewer points than this, 
        only plot the scatter points, not the denisty map. To always only produce
        pure scatter plot use `max_points=np.inf`.
    min_count: int
        Point density threshold. Bins with more points than this number will
        be plotted as density map. Points not in dense regions will be
        plotted as actual markers. For pure scatter plot set this value `None`
        or `np.inf`. For pure density map, set `min_count` to 0.
    tessellation
    scatter_kws
    density_kws

    Returns
    -------

    """

    data = _sanitize_data(data, 2)

    # plot only scatter if small amount of data
    if len(data) < max_points:
        min_count = None

    # default arg
    scatter_kws = scatter_kws or {}
    scatter_kws.setdefault('alpha', alpha)
    density_kws = density_kws or {}
    density_kws.setdefault('alpha', alpha)

    tessellation = tessellation.lower()
    if tessellation not in KNOWN_TESSELATIONS:
        raise ValueError(f'Invalid tessellation {tessellation!r}: Valid '
                         f'choices are {KNOWN_TESSELATIONS}')

    if tessellation == 'rect':
        return hist2d(ax, data, bins, range, min_count, cmap,
                      scatter_kws, density_kws)

    if tessellation == 'hex':
        return hexbin(ax, data, bins, range, min_count, cmap,
                      scatter_kws, density_kws)

    # div = make_axes_locatable(ax)
    # cax = div.append_axes('right', '5%')
    # cbar = ax.figure.colorbar(im, cax)
    # cbar.ax.set_ylabel('Density')

    # ax.set_title('Coord scatter')
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.grid()

    # return returns


def hist2d(ax, data, bins=CONFIG.bins, range=None, min_count=None,
           cmap=None, scatter_kws=None, density_kws=None):
    """

    Parameters
    ----------
    ax
    data
    bins
    range
    min_count
    scatter_kws
    density_kws

    Returns
    -------

    """
    assert len(data) > 0

    if min_count and np.isfinite(min_count):
        density_kws = dict(density_kws or {}, cmap=cmap)
        bins = duplicate_if_scalar(bins)

        x_data, y_data = data.T
        # plot density map
        hvals, x_edges, y_edges, qmesh = ax.hist2d(x_data, y_data,
                                                   bins, range, **density_kws)
        # remove low density points
        fc = qmesh.get_facecolor()
        fc[np.ravel(hvals < min_count)] = 0
        qmesh.set_facecolor(fc)

        ix_x = np.digitize(x_data, x_edges)
        ix_y = np.digitize(y_data, y_edges)

        # select points within the range
        ind = (ix_x > 0) & (ix_x <= bins[0]) & (ix_y > 0) & (ix_y <= bins[1])
        # values of the histogram where there are enough points
        hhsub = hvals[ix_x[ind] - 1, ix_y[ind] - 1]
        x_scatter = x_data[ind][hhsub < min_count]  # low density points
        y_scatter = y_data[ind][hhsub < min_count]
    else:
        hvals = []
        qmesh = None
        x_scatter, y_scatter = data

    # plot scatter points
    points = ax.plot(x_scatter, y_scatter,
                     **dict(**scatter_kws or {},
                            **CONFIG.style,
                            color=qmesh.get_cmap()(0)))

    return hvals, qmesh, points


def hexbin(ax, data, bins=CONFIG.bins, range=None, min_count=None,
           cmap=None, scatter_kws=(), density_kws=()):
    """

    Parameters
    ----------
    ax
    data
    bins
    min_count
    scatter_kws
    density_kws

    Returns
    -------

    """
    assert len(data) > 0

    scatter_kws = scatter_kws or {}
    
    if min_count and np.isfinite(min_count):
        # do density plot
        density_kws = dict(density_kws or {}, cmap=cmap)
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
        # set default colour of markers to match colormap
        scatter_kws.setdefault('color', polygons.get_cmap()(0))

        # copy the colormap (avoid deprecation warning for mpl 3.3)
        cm = copy.copy(polygons.get_cmap())
        # make the bins with few points invisible
        cm.set_under(ax.get_fc(), alpha=0)
        polygons.set(cmap=cm, clim=min_count)
        # polygons.set_clim(min_count)

    else:
        sparse_point_indices = ...
        hvals = []
        polygons = None

    # plot scatter points
    points = ax.plot(*data[sparse_point_indices].T,
                     **{**CONFIG.style, **scatter_kws})

    return hvals, polygons, points


#
# FUNC_MAP = dict(hex=hexbin,
#                 rect=hist2d)
