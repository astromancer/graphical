
# std
import itertools as itt

# third-party
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# relative
from .utils import set_clim_connected


# ---------------------------------------------------------------------------- #
GS_GRID = dict(hspace=0.005,
               wspace=0.005,
               left=0.03,  # fixme: need more for ticks
               right=0.97,
               bottom=0.03,
               top=0.98)

TITLE_STYLE = dict(color='w',
                   va='top',
                   fontweight='bold')

# ---------------------------------------------------------------------------- #


def resolve_layout(layout, n):
    if not layout:
        layout = auto_grid_layout(n)
    n_rows, n_cols = layout
    if n_rows == -1:
        n_rows = int(np.ceil(n / n_cols))
    if n_cols == -1:
        n_cols = int(np.ceil(n / n_rows))
    return n_rows, n_cols



def auto_grid_layout(n):
    x = int(np.floor(np.sqrt(n)))
    y = int(np.ceil(n / x))
    return x, y



def plot_image_grid(images, layout=(), titles=(), title_kws=None, figsize=None,
                    plims=None, clim_all=False, **kws):
    """

    Parameters
    ----------
    images
    layout
    titles
    clim_all:
        Compute colour limits from the full set of pixel values for all
        images.  Choose this if your images are all normalised to roughly the
        same scale. If False clims will be computed individually and the
        colourbar sliders will be disabled.

    Returns
    -------

    """

    # TODO: plot individual histograms - clim_each
    # todo: guess fig size

    n = len(images)
    assert n, 'No images to plot!'
    # assert clim_mode in ('all', 'row')

    # get grid layout
    n_rows, n_cols = resolve_layout(layout, n)

    # create figure
    fig = plt.figure(figsize=figsize)

    # ticks
    tick_par = dict(color='w', direction='in',
                    bottom=1, top=1, left=1, right=1)

    # Use gridspec rather than ImageGrid since the latter tends to resize
    # the axes
    if clim_all:
        cbar_size, hist_size = 3, 5
    else:
        cbar_size = hist_size = 0

    gs = GridSpec(n_rows, n_cols * (100 + cbar_size + hist_size), **GS_GRID)

    #
    kws = {**dict(origin='lower',
                  cbar=False, sliders=False, hist=False,
                  clim=not clim_all,
                  plims=plims),
           **kws}
    title_kws = {**TITLE_STYLE, **(title_kws or {})}

    art = []
    w = len(str(n))
    axes = np.empty((n_rows, n_cols), 'O')
    indices = enumerate(np.ndindex(n_rows, n_cols))
    for (i, (j, k)), title in itt.zip_longest(indices, titles, fillvalue=''):
        if i == n:
            break

        # last
        if (i == n - 1) and clim_all:
            # do colourbar + pixel histogram if clim all
            kws.update(cbar=True, sliders=True, hist=True,
                       cax=fig.add_subplot(
                           gs[:, -(cbar_size + hist_size) * n_cols:]),
                       hax=fig.add_subplot(gs[:, -hist_size * n_cols:]))

        # create axes!
        axes[j, k] = ax = fig.add_subplot(
            gs[j:j + 1, (100 * k):(100 * (k + 1))])

        # plot image
        imd = ImageDisplay(np.asanyarray(images[i]), ax=ax, **kws)
        art.append(imd.image)

        # do ticks
        top = (j == 0)
        bot = (j == n_rows - 1)
        left = (k == 0)  # leftmost
        # right = (j == n_cols - 1)

        # set the ticks to white and visible on all spines for aesthetic
        ax.tick_params('both', **{**dict(labelbottom=bot, labeltop=top,
                                         labelleft=left, labelright=0),
                                  **tick_par})

        for lbl, spine in ax.spines.items():
            spine.set_color('w')

        # add title text
        title = title.replace("\n", "\n     ")
        ax.text(0.025, 0.95, f'{i: <{w}}: {title}',
                transform=ax.transAxes, **title_kws)

    # Do colorbar
    # fig.colorbar(imd.image, cax)
    img = ImageGrid(fig, axes, imd)
    if clim_all:
        img._clim_all(images, plims)

    return img



# ---------------------------------------------------------------------------- #


class ImageGrid:
    def __init__(self, fig, axes, imd):
        self.figure = fig
        self.axes = axes
        self.imd = imd

    def __iter__(self):
        yield from (self.figure, self.axes, self.imd)

    def save(self, filenames):
        from matplotlib.transforms import Bbox

        fig = self.figure

        assert len(filenames) == self.axes.size

        ax_per_image = (len(fig.axes) // self.axes.size)
        # axit = mit.chunked(self.figure.axes, ax_per_image)

        for ax, name in zip(self.axes.ravel(), filenames):
            # mn, mx = (np.inf, np.inf), (0, 0)
            # for ax in axes[::-1]:
            #     # Save just the portion _inside_ the second axis's boundaries
            #     mn1, mx1 = ax.get_window_extent().transformed(
            #         fig.dpi_scale_trans.inverted()).get_points()
            #     mn = np.min((mn, mn1), 0)
            #     mx = np.max((mx, mx1), 0)

            #     ticklabels = ax.xaxis.get_ticklabels() + ax.yaxis.get_ticklabels()
            #     for txt in ticklabels:
            #         mn1, mx1 = txt.get_window_extent().transformed(
            #         fig.dpi_scale_trans.inverted()).get_points()
            #         mn = np.min((mn, mn1), 0)
            #         mx = np.max((mx, mx1), 0)

            # remove ticks
            # ax.set_axis_off()
            if len(ax.texts):
                ax.texts[0].set_visible(False)

            # Pad the saved area by 10% in the x-direction and 20% in the y-direction
            fig.savefig(name, bbox_inches=ax.get_window_extent().transformed(
                fig.dpi_scale_trans.inverted()).expanded(1.2, 1))

    @property
    def images(self):
        return [ax.images[0].get_array() for ax in self.figure.axes]

    def _clim_all(self, art, imd, images, plims):
        # connect all image clims to the sliders.
        for image in art:
            # noinspection PyUnboundLocalVariable
            imd.sliders.lower.on_move.add(set_clim_connected, image,
                                          imd.sliders)
            imd.sliders.upper.on_move.add(set_clim_connected, image,
                                          imd.sliders)

        # The same as above can be accomplished in pure matplolib as follows:
        # https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/multi_image.html
        # Make images respond to changes in the norm of other images (e.g. via
        # the "edit axis, curves and images parameters" GUI on Qt), but be
        # careful not to recurse infinitely!
        # def update(changed_image):
        #     for im in art:
        #         if (changed_image.get_cmap() != im.get_cmap()
        #                 or changed_image.get_clim() != im.get_clim()):
        #             im.set_cmap(changed_image.get_cmap())
        #             im.set_clim(changed_image.get_clim())
        #
        # for im in art:
        #     im.callbacksSM.connect('changed', update)

        # update clim for all plots

        # for the general case where images are non-uniform shape, we have to
        # flatten them all to get the colour percentile values.
        # TODO: will be more efficient for large number of images to sample
        #  evenly from each image
        pixels = []
        for im in images:
            # getattr(im, ('ravel', 'compressed')[np.ma.isMA(im)])()
            pixels.extend(im.compressed() if np.ma.isMA(im) else im.ravel())
        pixels = np.array(pixels)

        clim = self.imd.clim_from_data(pixels, plims=plims)
        self.imd.sliders.set_positions(clim, draw_on=False)  # no canvas yet!

        # Update histogram with data from all images
        self.imd.histogram.set_array(pixels)
        self.imd.histogram.autoscale_view()
