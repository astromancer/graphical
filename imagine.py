# TODO: docstrings (when stable)
# TODO: unit tests

import logging
import warnings
import traceback
import functools
from collections import Callable

import numpy as np
import matplotlib

matplotlib.use('qt5agg')
import matplotlib.pylab as plt
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Slider  # AxesWidget,
# from matplotlib.patches import FancyArrow, Circle
# from matplotlib.transforms import Affine2D
# from matplotlib.transforms import blended_transform_factory as btf

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from mpl_toolkits.axes_grid1 import AxesGrid, make_axes_locatable

# from recipes.io.tracewarn import warning_traceback_on
# from .interactive import ConnectionMixin
# from draggables.machinery import DragMachinery
from .sliders import AxesSliders, ThreeSliders
# from decor import expose


# from recipes.iter import grouper

# from PyQt4.QtCore import pyqtRemoveInputHook, pyqtRestoreInputHook
from IPython import embed


# warning_traceback_on()


def scale_unity(a):
    a = a - a.min()
    a /= a.max()
    return a


def _sanitize_data(data):
    """
    Removes nans and masked elements
    Returns flatened array
    """
    if np.ma.is_masked(data):
        data = data[~data.mask]
    return data[~np.isnan(data)]


def get_colour_scaler(method='ptp', **kws):
    """
    get vmin, vmax for image auto-scaling.

    Parameters
    ----------
    method       :       str - {'ptp', 'minmax', 'percentile', 'zscale'};
                             default 'ptp'
        which algorithm to use

    other keywords specify options for specific algorithm.

    Returns
    -------
    dict of modified keywords
    """

    method = kws.get('colourscale', method)
    method = kws.get('colorscale', method)

    # if both vmin and vmax are given, use those as limits TODO: either
    vmin = kws.get('vmin')
    vmax = kws.get('vmax')
    if vmin and vmax:
        kws.pop('clims', None)
        f = lambda: vmin, vmax
        # if either vmin or vmax provided they will supersede
        return f, kws

    # clims means the same as vmin, vmax
    clims = kws.pop('clims', None)
    if clims is not None:
        f = lambda: clims
        # function simply returns provided clim values
        return f, kws

    if method.startswith('per'):
        pmin = kws.pop('pmin', 2.25)
        pmax = kws.pop('pmax', 99.75)
        plims = kws.pop('plims', (pmin, pmax))
        # TODO: Display next to sliders??
        f = functools.partial(np.percentile, q=plims)
        return f, kws

    if method.startswith('z'):
        from .zscale import zrange
        contrast = kws.pop('contrast', 1 / 100)
        sigma_clip = kws.pop('sigma_clip', 3.5)
        maxiter = kws.pop('maxiter', 10)
        num_points = kws.pop('num_points', 1000)
        f = functools.partial(zrange,
                              contrast=contrast,
                              sigma_clip=sigma_clip,
                              maxiter=maxiter,
                              num_points=num_points)
        return f, kws

    if method.lower() in ('minmax', 'ptp'):
        f = lambda x: (x.min(), x.max())
        return f, kws

    raise ValueError('Say what?')


get_color_scaler = get_colour_scaler


# class ColourSliders():
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# def __init__(self, ax, x1, x2, slide_on='y', axpos=0., **kws):
# """ """
##self.setup_axes(ax, axpos)

# markers = ax.plot(axpos, x1, 'b>',
# axpos, np.mean((x1,x2)), 'go',
# axpos, x2, 'r<',
# ms=7.5,
# clip_on=False, zorder=3)

# self.knobs = DragMachinery(markers, annotate=False) #TODO: inherit???
# self.max_knob, self.centre_knob, self.min_knob = self.knobs

##FIXME: knobs don't update immediately when shifting center knob - i guess this could be desirable
##TODO: options for plot_on_motion / plot_on_release behaviour
##self.centre_knob.on_changed(self.center_shift)

# self.min_knob.on_changed(self.recentre0)
# self.max_knob.on_changed(self.recentre1)
##self.min_knob.on_changed(lambda o: self.recentre(-o))
##FIXME: update centre knob when shifting others...

# self.knobs.connect()


##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# def setup_axes(self, ax, axpos):
# """ """
##hide axis patch
# ax.patch.set_visible(0)
##setup ticks
# self.ax.tick_params(axis=self.slide_on, direction='inout', which='both')
# axsli, axoth = [ax.xaxis, ax.yaxis][::self._order]
# axoth.set_ticks([])
# which_spine = 'right' if self._index else 'bottom'
# axsli.set_ticks_position(which_spine)
##hide the axis spines
# ax.spines[which_spine].set_position(('axes', axpos))
# for where, spine in ax.spines.items():
# if where != which_spine:
# spine.set_visible(0)

##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# def get_vals(self):
# return tuple(k.get_ydata()[0] + d.offset
# for i, (k,d) in enumerate(self.knobs.draggables.items())
# if not i==1)
##SAY WHAAAAT???

##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# def on_changed(self, func):
# for knob in (self.min_knob, self.max_knob):
# knob.on_changed(func)

##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
##@expose.args()
# def recentre0(self, offset):
##momentarily disable observers to reposition center knob
##offset = (self.max_knob.offset + self.min_knob.offset) / 2
# print('offset', offset)
# self.centre_knob.shift(offset/2, observers_active=False)
# self.centre_knob.offset = offset/2

# def recentre1(self, offset):
# self.recentre0(-offset)

##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
##@expose.args()
##def center_shift(self, offset):
##for knob in (self.min_knob, self.max_knob):
##knob.shift(offset)
##knob.offset = offset


# class ColourBarHistogram():
#     def __init__(self):
#         ''
#     def update(self):
#         ''
#     def make_colours(self):
#         ''




# ****************************************************************************************************
class ImageDisplay(object):
    # TODO: Subclass AstroImageDisplay
    # TODO: Option for orientation markers - NE arrows
    # TODO: move cursor with arrow keys when hovering over figure (like ds9)
    # TODO: Sidebar grey things should be same colour as limiting colours
    # TODO: Sidebar: zoom

    def __init__(self, data, *args, **kws):
        """ """
        # ax      :       Axes object
        #     Axes on which to display

        # self.sscale = kws.pop('sscale', 'linear')
        # print('ImageDisplay')
        # print(kws)

        ax = kws.pop('ax', None)
        title = kws.pop('title', None)
        self.has_hist = kws.pop('hist', True)
        self.use_blit = kws.pop('use_blit', False)  # FIXME: blit broken
        self.colourscale = cs = kws.pop('colourscale',
                                        kws.pop('colorscale', 'percentile'))
        autoscale_figure = kws.pop('autoscale_figure', True)
        sidebar = kws.pop('sidebar', True)

        # check data
        data = np.atleast_2d(data)
        # convert boolean to integer (for colour scale algorithm)
        if data.dtype.name == 'bool':
            data = data.astype(int)

        if data.ndim > 2:
            raise ValueError('Cannot image %iD data' % data.ndim)
        self.data = data

        # create axes if required
        if ax is None:
            if autoscale_figure:
                # automatically determine the figure size based on the data
                figsize = self.guess_figsize(data)
            else:
                figsize = None

            self._gs = gs = GridSpec(1, 1,
                                     left=0.05, right=0.95,
                                     top=0.98, bottom=0.05)
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(gs[0, 0])

            # fig, ax = plt.subplots(figsize=figsize,
            #                        gridspec_kw=dict(left=0.05, right=0.95,
            #                                         top=0.98, bottom=0.05))

        self.figure = ax.figure
        self.ax = ax
        # self.divider = make_axes_locatable(ax)
        self.ax.format_coord = self.cooDisplayFormatter

        # set the clims vmin and vmax in kws according to requested auto scaling
        self.scaler, kws = get_colour_scaler(cs, **kws)
        kws['vmin'], kws['vmax'] = self.get_clim(data)

        # set the axes title if given
        if not title is None:
            ax.set_title(title)

            # re-shape the figure to fit the data
            # if autoscale_figure:
            # self.adjust_figure_to_data_size()
            # self.figure.subplots_adjust(left=0.05, right=0.95,
            #                             top=0.98, bottom=0.01)

        # use imshow to do the plotting
        self.imagePlot = ax.imshow(data, *args, **kws)

        if sidebar:
            # createHistogramreate the colourbar and the AxesSliders
            sidebar = SideBar(ax, self.has_hist, self.use_blit)
            self.sidebar = sidebar
            self.sliders = sidebar.sliders
        else:
            self.sidebar = None
            self.sliders = None

        # if self.use_blit:
        #     self.imagePlot.set_animated(True)
        #     self.sliders.ax.set_animated(True)

        self._draw_count = 0
        self.cid = ax.figure.canvas.mpl_connect('draw_event',
                                                self._on_draw)

    def _on_draw(self, event):
        logging.debug('DRAW %i', self._draw_count)  # ,  vars(event)
        if self._draw_count == 0:
            self._on_first_draw(event)
        self._draw_count += 1

    def _on_first_draw(self, event):
        print('FIRST DRAW')

    def guess_figsize(self, data):
        """
        Guess the figure size in inches based on screen size and shape of image
        data
        """
        # if np.squeeze(data).ndim != 2:
        #     raise ValueError('Data must be 2D not %sD' % str(data.ndim))

        shape = self.data.shape if (data is None) else np.shape(data)
        shape = shape[-2:]
        screenSize = 13, 16.5  # max scale  # TODO: automated way of getting this?
        fillFactor = 0.85
        figSize = (shape[::-1] / np.max(shape)) * screenSize[np.argmax(shape)]
        figSize *= fillFactor
        # make sure the figure isn't too small
        minsize = 2.25, 2.25
        figSize = np.where(figSize < minsize, minsize, figSize)

        # logging.debug('ImageDisplay: Guessed figure size: (%.1f, %.1f)', *figSize)
        return figSize

    def adjust_figure_to_data_size(self, data=None):
        if data is None:
            data = self.data

        figsize = self.guess_figsize(data)
        self.ax.figure.set_size_inches(figsize)

    def get_clim(self, data):
        """Get colour scale limits for data"""
        # remove masked data / nans for scaling algorithm
        data = _sanitize_data(data)
        return self.scaler(data)

    # def get_colourscale_limits(self, data, **kws):
    #     """Get colour scale limits for data"""
    #     kws = self.get_colour_scaler(data, **kws)
    #     return kws['vmin'], kws['vmax']

    # get_colorscale_limits = get_colourscale_limits


    def cooDisplayFormatter(self, x, y):
        col, row = int(x + 0.5), int(y + 0.5)
        Nrows, Ncols = self.data.shape  # [:2]
        # print('hello')
        if (col >= 0 and col < Ncols) and (row >= 0 and row < Nrows):
            # print(col, row)
            z = self.data[row, col]
            return 'x=%1.3f,\ty=%1.3f,\tz=%1.3f' % (x, y, z)
        else:
            return 'x=%1.3f, y=%1.3f' % (x, y)


class SideBar():
    # FIXME: Dragging too slow!!

    # TODO: Show which region on the histogram corresponds to colorbar with line indicators
    # TODO: optional connect histogram range with colorbar

    sliderClass = ThreeSliders  # AxesSliders

    def __init__(self, ax, hist=True, use_blit=False):
        # create the colourbar and the sliders

        self.ax = ax
        self.use_blit = use_blit
        self.has_hist = hist
        self.imagePlot = ax.images[0]
        self.divider = make_axes_locatable(ax)

        self.cax, self.cbar = self.createColorbar()
        self.sliders, h = self.createSliders()
        self.hvals, self.bin_edges, self.patches = h

        self.connect()

    def createColorbar(self):
        cax = self.divider.append_axes('right', size=0.2, pad=0)

        # No need for the data labels on the colourbar since it will be on the
        # histogram axis.
        fmt = None
        if self.has_hist:
            from matplotlib import ticker
            fmt = ticker.NullFormatter()

        cbar = self.ax.figure.colorbar(self.imagePlot, cax=cax, format=fmt)
        return cax, cbar

    def createSliders(self):

        sax = self.divider.append_axes('right', size=1, pad=0.2)  # .5)
        data = self.imagePlot.get_array()

        # set ylim if reasonable to do so
        if data.ptp():
            # avoid warnings in setting upper/lower limits identical
            sax.set_ylim((data.min(), data.max()))

        sax.yaxis.tick_right()

        # sax.yaxis.set_label_position('right')
        # sax.set_yscale( self.sscale )
        # self.sliders = type('null', (), {})()
        # self.sliders.ax = sax

        if self.has_hist:
            h = self.createHistogram(sax, data)
        else:
            h = [], [], []

        # create sliders after histogram so they display on top
        clim = self.imagePlot.get_clim()
        sliders = self.sliderClass(sax, *clim, slide_on='y')

        sliders.lower.on_changed.add(self.set_clim)
        sliders.upper.on_changed.add(self.set_clim)
        return sliders, h

    def createHistogram(self, ax, data):
        """histogram data on slider axis"""
        # from matplotlib.collections import PatchCollection

        h = hvals, bin_edges, patches = \
            ax.hist(_sanitize_data(data),
                    bins=100,
                    log=True,
                    orientation='horizontal')
        self.hvals, self.bin_edges, self.patches = h

        # TODO: use PatchCollection?????
        if self.use_blit:
            for p in patches:
                p.set_animated(True)

        clims = self.imagePlot.get_clim()
        self.updateHistogram(clims)

        ax.grid()  # which='both'

        return h

    def updateHistogram(self, clims):
        """Update histogram colours"""
        for i, (p, c) in enumerate(zip(self.patches, self.get_hcol(clims))):
            p.set_fc(c)

    def get_hcol(self, clims):
        """Get the colours to use for the histogram patches"""
        cmap = self.imagePlot.get_cmap()
        vm = np.ma.masked_outside(self.bin_edges, *clims)
        colours = cmap(scale_unity(vm))
        if np.ma.is_masked(vm):
            # grey out histogram patches outside of colour range
            colours[vm.mask, :3] = 0.25
            colours[vm.mask, -1] = 1
        return colours

    def set_clim(self, *xydata):

        clim = self.sliders.positions
        self.imagePlot.set_clim(clim)

        if not self.has_hist:
            return self.imagePlot  # TODO: COLOURBAR ticklabels??

        self.updateHistogram(clim)
        return self.imagePlot, self.patches  # TODO: COLOURBAR ticklabels??

    def connect(self):
        self.sliders.connect()


# ****************************************************************************************************
class CubeDisplayBase(ImageDisplay):
    _scroll_wrap = True
    autoscale_colours = False

    def __init__(self, data, **kws):
        """
        Image display for 3D data. Implements frame slider and image scroll.
        Optionally also displays apertures if coordinates provided.

        subclasses must implement set_frame, get_frame methods

        Parameters
        ----------
        data    :       array-like
            initial display data
        coords  :       optional, np.ndarray
            coordinates of apertures to display.  This must be an np.ndarray with
            shape (k, N, 2) where k is the number of apertures per frame, and N
            is the number of frames

        kws are passed directly to ImageDisplay.
        """
        # setup image display
        ImageDisplay.__init__(self, data[0], **kws)

        # save data (this can be array or FitsCube instance)
        self.data = data

        # remember the colour limits

        # setup frame slider
        self._frame = 0
        # if len(self.fig.axes) == 1:
        #     # the divider works for a single axes, but not when figure contains multiple axes
        #     self.fsax = self.divider.append_axes('bottom', size=0.2, pad=0.25)
        # else:

        # rect = 0.1, 0.025, 0.8, 0.02            # l, b, w, h
        bbox = self.ax.get_position()
        rect = bbox.x0, 0.025, bbox.width - .1, 0.025
        self.fsax = self.figure.add_axes(rect)
        self._gs.update(bottom=0.1)  # make space for the frame slider

        # TODO: eliminated this SHIT Slider class!!!
        self.frame_slider = Slider(self.fsax, 'frame', 0, len(self), valfmt='%d')
        self.frame_slider.on_changed(self.set_frame)
        if self.use_blit:
            self.frame_slider.drawon = False

        # save background for blitting
        self.background = self.figure.canvas.copy_from_bbox(self.ax.bbox)

        # enable frame scroll
        self.figure.canvas.mpl_connect('scroll_event', self._scroll)


        # @property
        # def has_coords(self):
        # return self.coords is not None

    def guess_figsize(self, data):
        size = super().guess_figsize(data)
        # create a bit more space below the figure for the frame nr indicator
        size[1] += 0.5
        # logging.debug('CubeDisplayBase: Guessed figure size: (%.1f, %.1f)', *size)
        return size

        # def adjust_figure_to_data_size(self, data=None):
        #     if data is None:
        #         data = self.data
        #
        #     figsize = self.guess_figsize(data)
        #     # make the figure slightly taller (room for slider)
        #     figsize *= (1, 1.35)
        #     self.ax.figure.set_size_inches(figsize)


        # def _needs_drawing(self):
        #     # NOTE: this method is a temp hack to return the artists that need to be
        #     # drawn when the frame is changed (for blitting). This is in place while
        #     # the base class is being refined.
        #     # TODO: proper observers as modelled on draggables.machinery
        #
        #     needs_drawing = [self.imagePlot]
        #     # if self.has_hist:
        #     #     needs_drawing.extend(self.patches)  # TODO: PatchCollection...
        #
        #     if self.sliders:
        #         needs_drawing.extend(self.sliders.sliders)

        ##[#self.imagePlot.colorbar, #self.sliders.centre_knob])


        return needs_drawing

    def get_image_data(self, i):
        return self.data[i]

    def get_frame(self):
        return self._frame

    def set_frame(self, i, draw=False):
        """Set frame data. draw if requested """

        # wrap scroll if desired
        if self._scroll_wrap:  # wrap around! (eg. scroll past end ==> go to beginning)
            i %= len(self)
        else:  # stop scrolling at the end
            i = max(i, len(self))

        i = int(round(i, 0))  # make sure we have an int
        if self._frame == i:
            # nothing to do
            return i

        self._frame = i  # store current frame
        image = self.get_image_data(i)

        # TODO: method set_image_data here??

        # ImageDisplay.draw_blit??
        # set the slider axis limits
        if self.sliders:
            # find min / max as float
            imin, imax = float(np.nanmin(image)), float(np.nanmax(image))
            self.sliders.ax.set_ylim(imin, imax)
            self.sliders.valmin, self.sliders.valmax = imin, imax
            # needs_drawing.append()???

        # set the image data
        self.imagePlot.set_data(image)
        # needs_drawing = [self.imagePlot]

        # TODO: lock the sliders in place with button??
        if self.autoscale_colours:
            # set the slider positions / color limits
            vmin, vmax = self.get_clim(image)
            self.imagePlot.set_clim(vmin, vmax)
            if self.sliders:
                self.sliders.set_positions((vmin, vmax))

                # TODO: update histogram values etc...
                # ImageDisplay.draw_blit??
                # if draw:
                # needs_drawing = self._needs_drawing()
                # self.draw_blit(needs_drawing)

        return i
        # FIXME: feels unnatural to return from setter.
        # Maybe introduce extra function that does the wrap. Then updates all the
        # plot elements, then does the draw
        # def update()

    frame = property(get_frame, set_frame)

    def _scroll(self, event):

        try:
            self.frame += [-1, +1][event.button == 'up']
            self.frame_slider.set_val(self.frame)
        except Exception as err:
            import traceback
            logging.debug('Scroll failed:')
            traceback.print_exc()

    # @expose.args()
    def draw_blit(self, artists):

        logging.debug('draw_blit')

        fig = self.ax.figure
        fig.canvas.restore_region(self.background)

        for art in artists:
            try:
                self.ax.draw_artist(art)
            except Exception as err:
                logging.debug('drawing FAILED %s', art)
                traceback.print_exc()

        fig.canvas.blit(fig.bbox)

    # def cooDisplayFormatter(self, x, y):
    #     s = ImageDisplay.cooDisplayFormatter(self, x, y)
    #     return 'frame %d: %s' % (self.frame, s)

    def cooDisplayFormatter(self, x, y):
        col, row = int(x + 0.5), int(y + 0.5)
        Nrows, Ncols, _ = self.data.shape
        # print('hello')
        if (col >= 0 and col < Ncols) and (row >= 0 and row < Nrows):
            # print(col, row)
            z = self.data[self._frame][row, col]
            return 'x=%1.3f,\ty=%1.3f,\tz=%1.3f' % (x, y, z)
        else:
            return 'x=%1.3f, y=%1.3f' % (x, y)


# ****************************************************************************************************
class ImageCubeDisplay(CubeDisplayBase):
    # TODO: frame switch buttons;

    # FIXME: histogram / colorbar labels not displaying nicely


    def __init__(self, data, **kws):
        CubeDisplayBase.__init__(self, data, **kws)
        self.data = np.atleast_3d(data)
        self.ishape = self.data.shape[1:]

    def __len__(self):
        return len(self.data)


# ****************************************************************************************************
from obstools.aps import ApertureCollection


class ImageCubeDisplayA(ImageCubeDisplay):
    aperture_properties = dict(ec='m', lw=1,
                               # animated=True,
                               picker=False,
                               widths=7.5, heights=7.5)

    def __init__(self, data, ap_prop_dict={}, ap_updater=None, **kws):
        """with Apertures
        ap_update_dict is dict keyed on aperture aperture_properties, values of which
        are either array-like sequence of values indeces corresponing to frame number,
        or callables that generate values.
        """
        CubeDisplayBase.__init__(self, data, **kws)
        # self.data = data

        # create apertures if data provided
        props = ImageCubeDisplayA.aperture_properties.copy()
        props['animated'] = self.use_blit
        props.update(ap_prop_dict)
        self.aps = ApertureCollection(**props)
        self.aps.coords = np.empty((0, 2))
        # add apertures to axes.  will not display yet if coordinates not given
        self.aps.axadd(self.ax)

        # check updater

        if (ap_updater is not None) and (not isinstance(ap_updater, Callable)):
            # embed()
            raise TypeError
            # else:
            # check updater working?? self.ap_updater(self.aps, 0)
            # NOTE: this is a good idea, since matplotlib callbacks will silently ignore Exceptions

        self.ap_updater = ap_updater

    def set_frame(self, i, draw=True):

        # logging.debug('set_frame')

        i = super().set_frame(i, False)
        # needs_drawing = self._needs_drawing()

        if self.ap_updater is not None:
            try:
                self.ap_updater(self.aps, i)
                # needs_drawing.append(self.aps)
            except Exception as err:
                logging.warning(str(err))
                self.aps.coords = np.empty((0, 2))

                # if self.use_blit:
                # self.draw_blit(needs_drawing)


                # def update(self, draggable, xydata):
                # """draw all artists that where changed by the motion"""
                # artists = filter(None,
                # (func(xydata) for cid, func in draggable.observers.items()))    #six.iteritems(self.observers):
                # if self._draw_on:
                # if self._use_blit:
                # self.canvas.restore_region(self.background)
                # for art in flatiter((artists, draggable.ref_art)): #WARNING: will flatten containers etc
                # art.draw(self.canvas.renderer)
                # self.canvas.blit(self.figure.bbox)
                # else:
                # self.canvas.draw()


                # def draw_blit(self, artists):
                # fig = self.ax.figure
                # fig.canvas.restore_region(self.background)

                # for art in artists:
                # self.ax.draw_artist(art)

                # fig.canvas.blit(fig.bbox)

        return i


# ****************************************************************************************************
class ImageCubeDisplayX(ImageCubeDisplay):
    marker_properties = dict(c='r', marker='x', alpha=1, ls='none', ms=5)

    def __init__(self, data, coords=None, **kws):
        CubeDisplayBase.__init__(self, data, **kws)

        if coords is not None:
            self.coords = np.asarray(coords)

        # if self.has_coords:
        self.marks, = self.ax.plot([], [], **self.marker_properties)

    def set_frame(self, i, draw=True):

        # logging.debug('set_frame')

        i = super().set_frame(i, False)
        needs_drawing = self._needs_drawing()
        self.marks.set_data(self.coords[i])

        if self.use_blit:
            self.draw_blit(needs_drawing)


# ****************************************************************************************************
from obstools.fastfits import FitsCube


# from recipes.iter import interleave


class FitsCubeDisplay(ImageCubeDisplayA):
    # FIXME: switching with slider messes up the aperture indexes
    # FIXME: colour hist does not display
    # FIXME: redraw when touching self.aps?????
    # TODO: frame switch buttons;
    # TODO: option to set clim from first frame??
    # TODO: autoscroll button
    # TODO: option for marking centres




    # DisplayClass = ImageCubeDisplayA


    def __init__(self, filename, ap_prop_dict={}, ap_updater=None, **kws):
        """ """
        # setup data access
        data = FitsCube(filename)

        full_extent = np.c_[(0, 0), data.ishape[::-1]].flatten() - 0.5
        kws.setdefault('extent', full_extent)
        kws.setdefault('origin', 'llc')
        # autoscale_figure=False,

        ImageCubeDisplayA.__init__(self, data,
                                   ap_prop_dict,
                                   ap_updater,
                                   **kws)

        # embed()

        # self.set_frame(0)                     # WANT this without triggering warnings: Attempting to set identical left==right results
        # self.adjust_figure_to_data_size()
        # self.ax.figure.tight_layout()

        if self.ap_updater is not None:
            self.ap_updater(self.aps, 0)

    def __len__(self):
        return len(self.data)

    def get_image_data(self, i):
        # logging.debug('FitsCubeDisplay.get_image_data(%s, %s)', self, i)
        return self.data[i]

    def get_frame(self):
        return self._frame


        # def set_frame(self, i):

        # self._frame = i
        # data = self[int(i%len(self))]
        # if self.autoscale:
        ##set the slider axis limits
        # dmin, dmax = data.min(), data.max()
        # self.sliders.ax.set_ylim(dmin, dmax)
        # self.sliders.valmin, self.sliders.valmax = dmin, dmax

        ##set the slider positiions / color limits
        # vmin, vmax = self.get_colourscale_limits(data, autoscale=self.autoscale)
        # self.imagePlot.set_data(data)
        # zzz.set_clim(vmin, vmax)
        # self.sliders.set_positions((vmin, vmax))

        ##update the apertures if needed
        # if self.has_coords:
        # self.aps.coords = self.coords[:, i, :]

        ##TODO: BLIT!!
        # self.ax.figure.canvas.draw()
        ##self.draw_blit([self.imgplt, self.aps])


        # frame = property(get_frame, set_frame)

        # class FitsCubeDisplay(ImageCubeDisplay,



        # class MultiImageCubeDisplay():
        # def __init__(self, *args, **kws):

        # assert not len(args)%2
        # self.axes, self.data = grouper(args, 2)

        # for
        # super().__init__(self.axes[-1], self.data[-1], **kws)



        ##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # def get_frame(self):
        # return self._frame

        # def set_frame(self, f):

        # f %= len(self.data[0])  #wrap around!
        # for
        # self.imgplt.set_data(self.data[f])
        # self.ax.figure.canvas.draw()
        # self._frame = f

        # frame = property(get_frame, set_frame)


# ****************************************************************************************************
class Compare3DImage():
    # TODO: profile & speed up!
    # TODO: link viewing angles!!!!!!!!!
    # MODE = 'update'
    """Class for plotting image data for comparison"""

    # @profile()
    def __init__(self, *args, **kws):
        """

        Parameters
        ----------
        args : tuple
            () or (X, Y, Z, data)  or  (fig, X, Y, Z, data)   or   ()
        kws :
        """

        self.plots = []
        self.images = []
        self.titles = kws.get('titles', ['Data', 'Fit', 'Residual'])
        self.scaler = get_colour_scaler(**kws)

        nargs = len(args)
        if nargs == 0:
            fig = None
            data = ()
        elif nargs == 4:
            fig = None
            data = args
        elif nargs == 5:
            fig, *data = args
        else:
            raise ValueError('Incorrect number of parameters')

        self.fig = self.setup_figure(fig)
        if len(data):
            # X, Y, Z, data
            self.update(*data)

    # @unhookPyQt
    def setup_figure(self, fig=None):
        # TODO: Option for colorbars
        # TODO:  Include info as text in figure??????
        """
        Initialize grid of 2x3 subplots. Top 3 are 3D wireframe, bottom 3 are colour images of
        data, fit, residual.
        """
        # Plots for current fit
        fig = fig or plt.figure(figsize=(14, 10), )
        # gridpec_kw=dict(left=0.05, right=0.95,
        #                 top=0.98, bottom=0.01))
        if not isinstance(fig, Figure):
            raise ValueError('Expected Figure, received %s' % type(fig))

        self.grid_3D = self.setup_3D_axes(fig)
        self.grid_images = self.setup_image_axes(fig)

        # fig.suptitle('PSF Fitting')    # NOTE:  Does not display correctly with tight layout
        return fig

    def setup_3D_axes(self, fig):
        # Create the plot grid for the 3D plots
        grid_3D = AxesGrid(fig, 211,  # similar to subplot(211)
                           nrows_ncols=(1, 3),
                           axes_pad=-0.2,
                           label_mode=None,  # This is necessary to avoid AxesGrid._tick_only throwing
                           share_all=True,
                           axes_class=(Axes3D, {}))

        for ax, title in zip(grid_3D, self.titles):
            # pl = ax.plot_wireframe([],[],[])     # since matplotlib 1.5 can no longer initialize this way
            pl = Line3DCollection([])
            ax.add_collection(pl)

            # set title to display above axes
            title = ax.set_title(title, dict(fontweight='bold',
                                             fontsize=14))
            x, y = title.get_position()
            title.set_position((x, 1.0))
            ax.set_facecolor('None')
            # ax.patch.set_linewidth( 1 )
            # ax.patch.set_edgecolor( 'k' )
            self.plots.append(pl)

        return grid_3D

    def setup_image_axes(self, fig):
        # Create the plot grid for the images
        grid_images = AxesGrid(fig, 212,  # similar to subplot(212)
                               nrows_ncols=(1, 3),
                               axes_pad=0.1,
                               label_mode='L',  # THIS DOESN'T FUCKING WORK!
                               # share_all = True,
                               cbar_location='right',
                               cbar_mode='each',
                               cbar_size='7.5%',
                               cbar_pad='0%')

        for i, (ax, cax) in enumerate(zip(grid_images, grid_images.cbar_axes)):
            im = ax.imshow(np.zeros((1, 1)), origin='lower')
            with warnings.catch_warnings():
                # UserWarning: Attempting to set identical bottom==top resultsin singular transformations; automatically expanding.
                warnings.filterwarnings('ignore', category=UserWarning)
                cbar = cax.colorbar(im)

            # make the colorbar ticks look nice
            c = 'orangered'  # > '0.85'
            cax.axes.tick_params(axis='y',
                                 pad=-7,
                                 direction='in',
                                 length=3,
                                 colors=c,
                                 labelsize='x-small')
            # make the colorbar spine invisible
            cax.spines['left'].set_visible(False)
            # for w in ('top', 'bottom', 'right'):
            cax.spines['right'].set_color(c)

            for t in cax.axes.yaxis.get_ticklabels():
                t.set_weight('bold')
                t.set_ha('center')
                t.set_va('center')
                t.set_rotation(90)

                # if i>1:
                # ax.set_yticklabels( [] )       #FIXME:  This kills all ticklabels
            self.images.append(im)

        return grid_images

    @staticmethod
    def make_segments(X, Y, Z):
        """Update segments of wireframe plots."""
        # NOTE: Does not seem to play well with masked data - mask shape changes...
        xlines = np.r_['-1,3,0', X, Y, Z]
        ylines = xlines.transpose(1, 0, 2)  # swap x-y axes
        return list(xlines) + list(ylines)

    def get_clim(self, data):
        data = _sanitize_data(data)
        return self.scaler(data)

    def update(self, X, Y, Z, data):
        """update plots with new data."""

        res = data - Z
        plots, images = self.plots, self.images
        # NOTE: mask shape changes, which breaks things below.
        plots[0].set_segments(self.make_segments(X, Y, data.copy()))
        plots[1].set_segments(self.make_segments(X, Y, Z))
        plots[2].set_segments(self.make_segments(X, Y, res.copy()))
        images[0].set_data(data)
        images[1].set_data(Z)
        images[2].set_data(res)

        zlims = [Z.min(), Z.max()]
        rlims = [res.min(), res.max()]
        clims = self.get_clim(data)
        # plims = 0.25, 99.75                             #percentiles
        # clims = np.percentile( data, plims )            #colour limits for data
        # rlims = np.percentile( res, plims )             #colour limits for residuals
        for i, pl in enumerate(plots):
            ax = pl.axes
            ax.set_zlim(zlims if (i + 1) % 3 else rlims)

        xr = X[0, [0, -1]]
        yr = Y[[0, -1], 0]
        with warnings.catch_warnings():
            # filter `UserWarning: Attempting to set identical bottom==top resultsin singular transformations; automatically expanding.`
            warnings.filterwarnings("ignore", category=UserWarning)
            ax.set_xlim(xr)
            ax.set_ylim(yr)

            # artificially set axes limits --> applies to all since share_all=True in constructor
            for i, im in enumerate(images):
                lims = clims if (i + 1) % 3 else rlims
                im.set_clim(lims)
                im.set_extent(np.r_[xr, yr])

                # self.fig.canvas.draw()
                # TODO: SAVE FIGURES.................


# ****************************************************************************************************
class Compare3DContours(Compare3DImage):
    def setup_image_axes(self, fig):
        # Create the plot grid for the contour plots
        self.grid_contours = AxesGrid(fig, 212,  # similar to subplot(211)
                                      nrows_ncols=(1, 3),
                                      axes_pad=0.2,
                                      label_mode='L',  # This is necessary to avoid AxesGrid._tick_only throwing
                                      share_all=True)

    def update(self, X, Y, Z, data):
        """update plots with new data."""
        res = data - Z
        plots, images = self.plots, self.images

        plots[0].set_segments(self.make_segments(X, Y, data))
        plots[1].set_segments(self.make_segments(X, Y, Z))
        plots[2].set_segments(self.make_segments(X, Y, res))
        # images[0].set_data( data )
        # images[1].set_data( Z )
        # images[2].set_data( res )

        for ax, z in zip(self.grid_contours, (data, Z, res)):
            cs = ax.contour(X, Y, z)
            ax.clabel(cs, inline=1, fontsize=7)  # manual=manual_locations

        zlims = [Z.min(), Z.max()]
        rlims = [res.min(), res.max()]
        # plims = 0.25, 99.75                             #percentiles
        # clims = np.percentile( data, plims )            #colour limits for data
        # rlims = np.percentile( res, plims )             #colour limits for residuals
        for i, pl in enumerate(plots):
            ax = pl.axes
            ax.set_zlim(zlims if (i + 1) % 3 else rlims)
        ax.set_xlim([X[0, 0], X[0, -1]])
        ax.set_ylim([Y[0, 0], Y[-1, 0]])

        # for i,im in enumerate(images):
        # ax = im.axes
        # im.set_clim( zlims if (i+1)%3 else rlims )
        ##artificially set axes limits --> applies to all since share_all=True in constuctor
        # im.set_extent( [X[0,0], X[0,-1], Y[0,0], Y[-1,0]] )

        # self.fig.canvas.draw()


# from recipes.array import ndgrid
from recipes.array.neighbours import neighbours


class PSFPlotter(Compare3DImage, FitsCubeDisplay):
    def __init__(self, filename, model, params, coords, window, **kws):
        self.model = model
        self.params = params
        self.coords = coords
        self.window = w = int(window)
        self.grid = np.mgrid[:w, :w]
        extent = np.array([0, w, 0, w]) - 0.5  # l, r, b, t

        Compare3DImage.__init__(self)
        axData = self.grid_images[0]

        FitsCubeDisplay.__init__(self, filename, ax=axData, extent=extent, sidebar=False, autoscale_figure=False)
        self.set_frame(0)  # FIXME: full frame drawn instead of zoom
        # have to draw here for some bizarre reason
        # self.grid_images[0].draw(self.fig._cachedRenderer)

    def get_image_data(self, i):
        # coo = self.coords[i]
        data = neighbours(self[i], self.coords[i], self.window)
        return data

    def set_frame(self, i, draw=False):
        """Set frame data. draw if requested """
        i %= len(self)  # wrap around! (eg. scroll past end ==> go to beginning)
        i = int(round(i, 0))  # make sure we have an int
        self._frame = i  # store current frame

        image = self.get_image_data(i)
        p = self.params[i]
        Z = self.model(p, self.grid)
        Y, X = self.grid
        self.update(X, Y, Z, image)

        if draw:
            self.fig.canvas.draw()

        return i


if __name__ == '__main__':
    import pylab as plt

    data = np.random.random((100, 100))
    ImageDisplay(data)

    # fig, ax = plt.subplots(1,1, figsize=(2.5, 10), tight_layout=True)
    # ax.set_ylim(0, 250)
    # sliders = AxesSliders(ax, 0.2, 0.7, slide_on='y')
    # sliders.connect()


    plt.show()






# class Imager(object):

# def __init__(self, ax, z, x, y):
# self.ax = ax
# self.x  = x
# self.y  = y
# self.z  = z
# self.dx = self.x[1] - self.x[0]
# self.dy = self.y[1] - self.y[0]
# self.numrows, self.numcols = self.z.shape
# self.ax.format_coord = self.format_coord

# def format_coord(self, x, y):
# col = int(x/self.dx+0.5)
# row = int(y/self.dy+0.5)
##print "Nx, Nf = ", len(self.x), len(self.y), "    x, y =", x, y, "    dx, dy =", self.dx, self.dy, "    col, row =", col, row
# xyz_str = ''
# if (col>=0 and col<self.numcols and row>=0 and row<self.numrows):
# zij = self.z[row,col]
##print "zij =", zij, '  |zij| =', abs(zij)
# if (np.iscomplex(zij)):
# amp, phs = abs(zij), np.angle(zij) / np.pi
# signz = '+' if (zij.imag >= 0.0) else '-'
# xyz_str = 'x=' + str('%.4g' % x) + ', y=' + str('%.4g' % y) + ',' \
# + ' z=(' + str('%.4g' % zij.real) + signz + str('%.4g' % abs(zij.imag)) + 'j)' \
# + '=' + str('%.4g' % amp) + r'*exp{' + str('%.4g' % phs) + u' π j})'
# else:
# xyz_str = 'x=' + str('%.4g' % x) + ', y=' + str('%.4g' % y) + ', z=' + str('%.4g' % zij)
# else:
# xyz_str = 'x=%1.4f, y=%1.4f'%(x, y)
# return xyz_str



# def supershow(ax, x, y, z, *args, **kws):

# assert len(x) == z.shape[1]
# assert len(y) == z.shape[0]

# dx = x[1] - x[0]
# dy = y[1] - y[0]
# zabs = abs(z) if np.iscomplex(z).any() else z

## Use this to center pixel around (x,y) values
# extent = (x[0]-dx/2.0, x[-1]+dx/2.0, y[0]-dy/2.0, y[-1]+dy/2.0)

# im = ax.imshow(zabs, extent = extent, *args, **kws)
# imager = Imager(ax, z, x, y)
# ax.set_xlim((x[0], x[-1]))
# ax.set_ylim((y[0], y[-1]))

# return im
