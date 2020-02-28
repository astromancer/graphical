import numpy as np
from recipes.misc import duplicate_if_scalar

DEFAULT_BINS = 50


def _sanitize_data(data, allow_dim):
    # sanitize data
    data = np.asanyarray(data).squeeze()
    assert data.size > 0, 'No data!'
    assert data.ndim == allow_dim
    assert data.shape[-1] == 2

    # mask nans
    data = np.ma.MaskedArray(data, np.isnan(data))
    return data


def scatter_density(ax, data, bins=DEFAULT_BINS, range=None, min_count=3,
                    tessellation='hex', scatter_kws=None,
                    density_kws=None):
    """
    Point cloud visualization with density map and scatter plot. Regions
    with high point density are plotted as a 2d histogram image using either
    rectangular or hexagonal binning.

    Parameters
    ----------
    ax
    data
    bins: int
    range
    min_count: int
        point density threshold. Bins with more points than this number will
        be plotted as density map. Points not in dense regions will be
        plotted as actual markers. For pure scatter plot set this value `None`
        or `numpy.inf`.  For pure density map, set `min_count` to 0.
    tessellation
    scatter_kws
    density_kws

    Returns
    -------

    """

    data = _sanitize_data(data, 2)

    # default arg
    # cmap = get_cmap(density_kws.get('cmap', DEFAULT_CMAP))
    scatter_kws = scatter_kws or {}
    density_kws = density_kws or {}

    if tessellation not in FUNC_MAP:
        raise ValueError('Invalid tessellation %r: Valid choices are %s',
                         tessellation, tuple(FUNC_MAP.keys()))

    if tessellation == 'rect':
        returns = hist2d_scatter(ax, data, bins, range, min_count,
                                 scatter_kws, density_kws)

    if tessellation == 'hex':
        returns = hexbin_scatter(ax, data, bins, range, min_count,
                                 scatter_kws, density_kws)

    # div = make_axes_locatable(ax)
    # cax = div.append_axes('right', '5%')
    # cbar = ax.figure.colorbar(im, cax)
    # cbar.ax.set_ylabel('Density')

    # ax.set_title('Coord scatter')
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    ax.grid()

    return returns


def hist2d_scatter(ax, data, bins=DEFAULT_BINS, range=None, min_count=None,
                   scatter_kws=None,
                   density_kws=None):
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
    do_density_plot = (min_count is not None) and np.isfinite(min_count)
    if do_density_plot:
        density_kws = density_kws or {}
        bins = duplicate_if_scalar(bins)

        x_data, y_data = data.T
        # plot density map
        hvals, x_edges, y_edges, qmesh = ax.hist2d(x_data, y_data,
                                                   bins, range,
                                                   **density_kws)
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
    scatter_kws = scatter_kws or {}
    scatter_kws.setdefault('color', qmesh.get_cmap()(0))
    scatter_kws.setdefault('marker', 'o')
    scatter_kws.setdefault('ls', '')

    points = ax.plot(x_scatter, y_scatter, **scatter_kws)

    return hvals, qmesh, points


def hexbin_scatter(ax, data, bins=DEFAULT_BINS, range=None, min_count=None,
                   scatter_kws=None, density_kws=None):
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
    scatter_kws = scatter_kws or {}
    do_density_plot = (min_count is not None) and np.isfinite(min_count)
    if do_density_plot:
        density_kws = density_kws or {}
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
    else:
        sparse_point_indices = ...
        hvals = []
        polygons = None

    # plot scatter points
    scatter_kws.setdefault('marker', 'o')
    scatter_kws.setdefault('ls', '')
    points = ax.plot(*data[sparse_point_indices].T, **scatter_kws)

    # make the bins with few points invisible
    cm = polygons.get_cmap()
    cm.set_under((1, 1, 1), alpha=1)
    polygons.set_clim(min_count)
    return hvals, polygons, points


#
FUNC_MAP = dict(hex=hexbin_scatter,
                rect=hist2d_scatter)
