# import logging

import numpy as np
import matplotlib.pyplot as plt

# from recipes.decor import print_args

from recipes.oo.meta import flaggerFactory
from recipes.containers.lists import flatten
from recipes.logging import LoggingMixin

# from recipes.decor import profile
# from recipes.decor.misc import unhookPyQt
# from PyQt4.QtCore import pyqtRemoveInputHook, pyqtRestoreInputHook
# from IPython import embed

# *******************************************************************************
ConnectionManager, mpl_connect = flaggerFactory(collection='_connections')


class ConnectionMixin(ConnectionManager, LoggingMixin):
    """Mixin for connecting the decorated methods to the figure canvas"""

    def __init__(self, canvas=None):
        """ """
        ConnectionManager.__init__(self)
        self.connections = {}  # connection ids
        # TODO: check that it's a canvas
        self._canvas = canvas

    @property
    def canvas(self):
        return self._canvas

    def add_connection(self, name, method):
        self.logger.debug('Adding connection %r: %s', name, method)
        self.connections[name] = self.canvas.mpl_connect(name, method)

    def remove_connection(self, name):
        self.logger.debug('Removing connection %r', name)
        self.canvas.mpl_disconnect(self.connections[name])
        self.connections.pop(name)

    def connect(self):
        """connect the flagged methods to the canvas"""
        for (name,), method in self._connections.items():
            self.add_connection(name, method)

    def disconnect(self):
        """
        Disconnect from figure canvas.
        """
        for name, cid in self.connections.items():
            self.canvas.mpl_disconnect(cid)
        self.logger.debug('Disconnected from figure %s', self.figure.canvas)


class CanvasSaver(ConnectionMixin):
    def __init__(self, fig):
        ConnectionMixin.__init__(self, fig)
        # print('RAAAR')
        # print(self.connections)
        # print(self._connections)
        # save the background after the first draw
        saving = True

    @mpl_connect('button_release_event')
    def _on_release(self, event):
        """ """
        # connect method to save bg when axes panned/zoomed
        self.saving = self.canvas.manager.toolbar._active in ('PAN', 'ZOOM')
        # FIXME: remove markers before bg save!

    def save(self, axes=None):
        """save the figure content as background"""
        print('HALLELUJA!')
        self.background = self.canvas.copy_from_bbox(self.figure.bbox)

    @mpl_connect('draw_event')
    def _on_draw(self, event):
        """Saves the canvas background after the first draw"""

        print('UH!')

        # save background for bliting
        if self.saving:
            self.save()

        # prevent saving after EVERY draw
        self.saving = False

    def save_after_draw(self, event):
        """Saves the canvas background after the canvas draw"""
        # save background for bliting
        self.save()
        # remove this method callback
        # self.remove_connection('draw_event')


# *******************************************************************************
# from matplotlib.transforms import ScaledTranslation

class PointSelector(ConnectionMixin):  # LineGrab?
    """Click on the axes to return the closest index point along Line2D"""
    fmt = '{line}: Point {i}; {coo}'
    marker_props = dict(marker=r'$\downarrow$',  # 'o'
                        ms=25,
                        color='r',
                        mec='r',
                        ls='None')
    marker_offset = (0, 15)  # in display (pixel) coordinates

    def __init__(self, lines):
        self.lines = flatten(lines)

        self.ax = ax = self.lines[0].axes
        self.figure = fig = ax.figure
        ConnectionMixin.__init__(self, fig)

        # save the background after the first draw
        saving = True

        # print('RAAAR')
        # print(self.connections)
        # print(self._connections)

        # create the position markers
        self.markers, = ax.plot([], [], **self.marker_props)
        self.markers.set_zorder(20)
        # apply the marker offset

    def ignore(self, event):
        print('ignoring')
        return not (event.inaxes and
                    self.canvas.manager.toolbar._active is None)

    def get_index(self, line, event):
        """
        Return the index corresponding to closest matching point on line to
        mouse position on click.
        """
        xy = line.get_xydata()
        xye = (event.xdata, event.ydata)
        return np.linalg.norm(xy - xye, axis=1).argmin()

    def get_index_coords(self, line, event):
        """
        Return the index corresponding to closest matching point on line to
        mouse position on click.
        """
        xy = line.get_xydata()
        ix = np.argmin(abs(xy[:, 0] - event.xdata))
        return ix, xy[ix]

    @mpl_connect('button_press_event')
    def _on_click(self, event):
        print('click')
        if self.ignore(event):
            return

        indeces, xy = [], []
        ax = event.inaxes
        for line in self.lines:
            ix, coo = self.get_index_coords(line, event)
            indeces.append(ix)
            xy.append(coo)

            msg = self.fmt.format(i=ix,
                                  line=line,
                                  coo=ax.format_coord(*coo))
            print(msg)
        print()

        # update marker positions
        xy = np.array(xy).T
        self.markers.set_data(xy)

        # offset = ScaledTranslation(*self.marker_offset,
        # scale_trans=fig.dpi_scale_trans)
        # self.markers.set_transform(ax.transData + offset)

        # self.markers.
        # self.draw_blit([self.markers])
        self.canvas.draw()

        return indeces

        # @mpl_connect('button_release_event')
        # def _on_release(self, event):
        # """ """
        ##connect method to save bg when axes panned/zoomed
        # self.saving = self.canvas.manager.toolbar._active in ('PAN', 'ZOOM')
        ##FIXME: remove markers before bg save!

        # @mpl_connect('draw_event')
        # def _on_draw(self, event):
        # """Saves the canvas background after the first draw"""

        # print( 'UH!' )

        ##save background for bliting
        # if self.saving:
        # self.background = self.canvas.copy_from_bbox(self.figure.bbox)

        ##prevent saving after EVERY draw
        # self.saving = False

    def draw_blit(self, artists):

        self.canvas.restore_region(self.background)
        for art in artists:
            art.draw(self.canvas.renderer)

        self.canvas.blit(self.ax.bbox)


# *******************************************************************************


from matplotlib.widgets import RectangleSelector
from matplotlib.lines import Line2D


# from magic.iter import first_true_index, last_true_index, first_false_index
# from IPython import embed

# *******************************************************************************
class LineSelector(RectangleSelector):
    # FIXME: rectangle dissappears on key press.  This may not be desired
    # FIXME: single press creates a 0-size box at (0,0)???
    # FIXME: box dissapears when resizing with _edge_handles
    # FIXME: box dissappears, but highlighting does not (on 'r')
    # FIXME: Replot highlighted after restore
    def __init__(self, lines, drawtype='box',
                 minspanx=None, minspany=None, useblit=False,
                 lineprops=None, rectprops=None, spancoords='data',
                 button=None, maxdist=10, marker_props=None,
                 interactive=False, state_modifier_keys=None):

        self.verbose = True
        self.lines = flatten(lines)
        ax = self.lines[0].axes

        RectangleSelector.__init__(self, ax, self.select_lines, drawtype,
                                   minspanx, minspany, useblit,
                                   lineprops, rectprops, spancoords,
                                   button, maxdist, marker_props,
                                   interactive, state_modifier_keys)

        hprops = dict(linewidth=10, alpha=0.5, linestyle='-')  # marker='s'
        self.selection = [np.zeros(l.get_xdata().shape, bool)
                          for l in self.lines]

        # Create Line2D for highlighting selected sections
        self.highlighted = []
        for line in self.lines:
            hline, = ax.plot([], [], color=line.get_color(), **hprops)
            self.highlighted.append(hline)
            self.artists.append(
                hline)  # enable blitting for the highlighted segments

    # def contains(self, ):?????
    # @print_args()
    # @profile()
    # @unhookPyQt
    def select_lines(self, eclick, erelease):
        """eclick and erelease are the press and release events"""
        xmin, xmax, ymin, ymax = extent = self.extents
        for i, line in enumerate(self.lines):
            x, y = line.get_data()
            l = (xmin < x) & (x < xmax) & \
                (ymin < y) & (y < ymax)  # points contained within selector box
            self.selection[i] = l

            # set the highlighted line data
            # need to mask a section if selection is discontinuous (i.e. has gaps)
            if l.any():
                ix, = np.where(l)
                nix, = np.where(~l)
                fti, lti = ix[0], ix[-1]
                ffi = nix[0]

                if ffi < lti:  # discontinuous selection
                    sx = slice(fti, lti + 1)
                    xh = np.ma.masked_where(~l[sx], x[sx])
                    yh = y[sx]
                else:
                    xh, yh = x[l], y[l]
            else:
                xh, yh = x[l], y[l]

            hline = self.highlighted[i]
            hline.set_data(xh, yh)

            if self.verbose:
                print('Masking {} points on {} '
                      'within x=({:.3f},{:.3f});'
                      'y=({:.3f}, {:.3f})'.format(l.sum(), line, *extent))

    # @profile( follow=[] )
    def _release(self, event):
        """on button release event"""
        if not self.interactive:
            self.to_draw.set_visible(False)

        if self.spancoords == 'data':
            xmin, ymin = self.eventpress.xdata, self.eventpress.ydata
            xmax, ymax = self.eventrelease.xdata, self.eventrelease.ydata
            # calculate dimensions of box or line get values in the right
            # order
        elif self.spancoords == 'pixels':
            xmin, ymin = self.eventpress.x, self.eventpress.y
            xmax, ymax = self.eventrelease.x, self.eventrelease.y
        else:
            raise ValueError('spancoords must be "data" or "pixels"')

        if xmin > xmax:
            xmin, xmax = xmax, xmin
        if ymin > ymax:
            ymin, ymax = ymax, ymin

        spanx = xmax - xmin
        spany = ymax - ymin
        xproblems = self.minspanx is not None and spanx < self.minspanx
        yproblems = self.minspany is not None and spany < self.minspany

        if (((self.drawtype == 'box') or (self.drawtype == 'line')) and
                (xproblems or yproblems)):
            # check if drawn distance (if it exists) is not too small in
            # neither x nor y-direction

            for h in self.highlighted:
                h.set_data([[], []])
                print(h, h.get_data())

            self.extents = [0] * 4  # FIXME: This draws all the handles at (0,0)

            return

        # update the eventpress and eventrelease with the resulting extents
        x1, x2, y1, y2 = self.extents
        self.eventpress.xdata = x1
        self.eventpress.ydata = y1
        xy1 = self.ax.transData.transform_point([x1, y1])
        self.eventpress.x, self.eventpress.y = xy1

        self.eventrelease.xdata = x2
        self.eventrelease.ydata = y2
        xy2 = self.ax.transData.transform_point([x2, y2])
        self.eventrelease.x, self.eventrelease.y = xy2

        self.onselect(self.eventpress, self.eventrelease)
        # call desired function
        self.update()

        return False

    # @profile( follow=[RectangleSelector._press] )
    def _press(self, event):
        if event.button == 1:
            pass

        if event.button == 2:
            self.restart()
            self.canvas.draw()  # TODO: blit
            return

        RectangleSelector._press(self, event)

        # @profile( follow=[RectangleSelector._release] )
        # def _release(self, event):
        # RectangleSelector._release(self, event)

    def restart(self):
        for i, line in enumerate(self.lines):
            x = line.get_xdata()
            if np.ma.is_masked(x):
                x.mask = False
                line.set_xdata(x)

            self.highlighted[i].set_data([[], []])

    # @unhookPyQt
    def _on_key_press(self, event):
        if event.key.lower().startswith('d'):  # 'delete'/'d'/'D'
            # print( 'delete', '!'*10 )
            for i, line in enumerate(self.lines):
                x = line.get_xdata()
                x = np.ma.masked_where(self.selection[i], x)
                line.set_xdata(x)
                self.highlighted[i].set_data([[], []])

        if event.key.lower() == 'r':
            for i, line in enumerate(self.lines):
                hline = self.highlighted[i]
                x = line.get_xdata()
                if np.ma.is_masked(
                        x):  # FIXME: obviate this check by making ma from start
                    l = self.selection[i]
                    x.mask &= ~l
                line.set_xdata(x)
                # unmask highlighted section that was just restored
                # embed()

                xh = hline.get_xdata()
                print('len(xh), len(l), l.sum()')
                print(len(xh), len(l), l.sum())
                print(l)
                print(np.where(l))
                embed()
                # print( xh.mask | x.mask[l] )
                # hline.set_xdata(xh)

        self.canvas.draw()  # TODO: blit

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# *******************************************************************************
class LCFrameDisplay(ConnectionMixin):
    def __init__(self, fitsfile, fluxes, coo):

        self.coords = np.asarray(coo)

        figlc, axlc = plt.subplots(figsize=(18, 8))
        for f in fluxes:
            axlc.plot(f)

        ConnectionMixin.__init__(self, figlc)
        # ps = PointSelect(figlc)
        # ps.connect()

        self.frame = FITSFrame(fitsfile)
        sx = self.frame.shape
        extent = interleave((0,) * len(sx), sx)

        self.figi, axi = plt.subplots(tight_layout=True)
        self.image = axi.imshow([[0]],
                                origin='llc',
                                extent=extent)  # np.zeros(self.frame.shape) / empty?
        # TODO: colorbar; set clim from first frame; sliders; frame switch buttons;

        self.aps = ApertureCollection(radii=10, ec='m', lw=2)
        self.aps.axadd(axi)

    def get_index(self, line, event):
        # r = np.linalg.norm( line.get_xydata() - event.xydata, axis=0 )
        # ix = r.argmin()
        xy = line.get_xydata()
        return np.argmin(abs(xy[:, 0] - event.xdata))

    def ignore(self, event):
        return not (event.inaxes and
                    self.canvas.manager.toolbar._active is None)

    @mpl_connect('button_press_event')
    def _on_click(self, event):
        if self.ignore(event):
            print('ignoring')
            return

        ax = event.inaxes
        line = ax.lines[0]
        ix = self.get_index(line, event)
        frame = self.frame[ix]

        # print('YO! '*10)
        # print( frame )
        self.aps.coords = self.coords[:, ix, :]

        self.image.set_data(frame)
        self.image.set_clim(np.percentile(frame, (0.25, 99.75)))
        self.figi.canvas.draw()


# *******************************************************************************
# from .imagine import FitsCubeDisplay

class LCFrameDisplay2(PointSelector):
    """ """

    def __init__(self, cube, **kw):
        coords = np.array([star.coo for star in cube])

        tkw = kw.get('tkw', 'utsec')
        starplots = cube.plot_lc(tkw=tkw,
                                 mode='flux',
                                 relative_time=True,
                                 twinx='sexa',
                                 **kw)
        # starplots.connect()

        PointSelector.__init__(self, starplots.draggable)

        fitsfile = cube.filename
        figure, ax = plt.subplots(tight_layout=True)
        self.frame = FITSCubeDisplay(ax, fitsfile, coords)

    def ignore(self, event):
        return not (event.inaxes
                    and
                    self.canvas.manager.toolbar._active is None
                    and
                    event.button == 1)

    @mpl_connect('button_press_event')
    def _on_click(self, event):
        if self.ignore(event):
            print('ignoring')
            return

        # print the selected points
        indeces = PointSelector._on_click(self, event)
        ix = indeces[0]

        print(ix)

        self.frame.frame_slider.set_val(ix)
        # NOTE: This will call self.frame.set_frame(ix)

        # FIXME: frame selection with slider - not updating coordinates!!


if __name__ == '__main__':
    fig, ax = plt.subplots()
    N = 1e2  # If N is large one can see
    x = np.linspace(0.0, 10.0, N)  # improvement by use blitting!

    # plot something
    pl1, = ax.plot(x, +np.sin(.2 * np.pi * x), lw=3.5, c='b', alpha=.7)
    pl2, = ax.plot(x, +np.cos(.2 * np.pi * x), lw=3.5, c='r', alpha=.5)
    pl3, = ax.plot(x, -np.sin(.2 * np.pi * x), lw=3.5, c='g', alpha=.3)
    plots = pl1, pl2, pl3

    # drawtype is 'box' or 'line' or 'none'
    s = LineSelector(plots,
                     drawtype='box', useblit=False,
                     minspanx=5, minspany=5,
                     spancoords='pixels',
                     interactive=True)

    plt.show()
