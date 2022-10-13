"""
Efficient, scrollable image sequence visualisation.
"""


# std
import time
import warnings

# third-party
import numpy as np
from matplotlib import ticker
from matplotlib.widgets import Slider

# relative
from ..image import ImageDisplay


class VideoDisplay(ImageDisplay):
    # FIXME: blitting not working - something is leading to auto draw
    # FIXME: frame slider bar not drawing on blit
    # FIXME: HISTOGRAM values not updating on scroll
    # TODO: lock the sliders in place with button??

    _scroll_wrap = True  # scrolling past the end leads to the beginning

    def _check_data(self, data):
        if not isinstance(data, np.ndarray):
            data = np.ma.asarray(data)

        n_dim = data.ndim
        if n_dim == 2:
            warnings.warn('Loading single image frame as 3D data cube. Use '
                          '`ImageDisplay` instead to view single frames.')
            data = np.ma.atleast_3d(data)

        if n_dim != 3:
            raise ValueError(f'Cannot image {n_dim}D data')
        return data, len(data)

    def __init__(self, data, clim_every=1, **kws):
        """
        Image display for 3D data. Implements frame slider and image scroll.

        subclasses optionally implement `update` method

        Parameters
        ----------
        data:       np.ndarray or np.memmap
            initial display data

        clim_every: int
            How frequently to re-run the color normalizer algorithm to set
            the colour limits. Setting this to `False` may have a positive
            effect on performance.

        kws are passed directly to ImageDisplay.
        """

        #
        data, nframes = self._check_data(data)

        self.nframes = int(nframes)
        self.clim_every = clim_every

        # don't connect methods yet
        connect = kws.pop('connect', True)

        # setup image display
        # parent sets data as 2D image.
        n = self._frame = 0
        ImageDisplay.__init__(self, data[n], connect=False, **kws)
        # save data (this can be array_like (or np.mmap))
        self.data = data

        # make observer container for scroll
        # self.on_scroll = Observers()

        # make frame slider
        fsax = self.divider.append_axes('bottom', size=0.1, pad=0.3)
        self.frameSlider = Slider(fsax, 'frame', n, self.nframes, valfmt='%d')
        self.frameSlider.on_changed(self.update)
        fsax.xaxis.set_major_locator(ticker.AutoLocator())

        if self.use_blit:
            self.frameSlider.drawon = False

        # # save background for blitting
        # self.background = self.figure.canvas.copy_from_bbox(
        #     self.ax.bbox)

        if connect:
            self.connect()

    def connect(self):
        ImageDisplay.connect(self)

        # enable frame scroll
        self.figure.canvas.mpl_connect('scroll_event', self._scroll)

    # def init_figure(self, **kws):
    #     fig, ax = ImageDisplay.init_figure(self, **kws)
    #     return fig, ax

    # def init_axes(self, fig):
    #     gs = GridSpec(100, 100,
    #                   left=0.05, right=0.95,
    #                   top=0.98, bottom=0.05,
    #                   hspace=0, wspace=0)
    #     q = 97
    #     return self._init_axes(fig,
    #                            image=gs[:q, :80],
    #                            cbar=gs[:q, 80:85],
    #                            hbar=gs[:q, 87:],
    #                            fslide=gs[q:, :80])

    def guess_figsize(self, data, fill_factor=0.55, max_pixel_size=0.2):
        # TODO: inherit docstring
        size = super().guess_figsize(data, fill_factor, max_pixel_size)
        # create a bit more space below the figure for the frame nr indicator
        size[1] += 0.5
        self.logger.debug('Guessed figure size: ({:.1f}, {:.1f})', *size)
        return size

    @property
    def frame(self):
        """Index of image currently being displayed"""
        return self._frame

    @frame.setter
    def frame(self, i):
        """Set frame data respecting scroll wrap"""
        self.set_frame(i)

    def set_frame(self, i):
        """
        Set frame data respecting scroll wrap
        """
        # wrap scrolling if desired
        if self._scroll_wrap:
            # wrap around! scroll past end ==> go to beginning
            i %= self.nframes
        else:  # stop scrolling at the end
            i = max(i, self.nframes)

        i = int(round(i, 0))  # make sure we have an int
        self._frame = i  # store current frame

    def get_image_data(self, i):
        """
        Get the image data to be displayed.

        Parameters
        ----------
        i: int
            Frame number

        Returns
        -------
        np.ndarray

        """
        return self.data[i]

    def update(self, i, draw=True):
        """
        Display the image associated with index `i` frame in the sequence. This
        method can be over-written to change image switching behaviour
        in subclasses.

        Parameters
        ----------
        i: int
            Frame index
        draw: bool
            Whether the canvas should be redrawn after the data is updated


        Returns
        -------
        draw_list: list
            list of artists that have been changed and need to be redrawn
        """
        self.set_frame(i)

        image = self.get_image_data(self.frame)
        # set the image data
        # TODO: method set_image_data here??
        self.imagePlot.set_data(image)  # does not update normalization

        # FIXME: normalizer fails with boolean data
        #  File "/usr/local/lib/python3.6/dist-packages/matplotlib/colorbar.py", line 956, in on_mappable_changed
        #   self.update_normal(mappable)
        # File "/usr/local/lib/python3.6/dist-packages/matplotlib/colorbar.py", line 987, in update_normal

        draw_list = [self.imagePlot]

        # set the slider axis limits
        if self.sliders:
            # find min / max as float
            min_max = float(np.nanmin(image)), float(np.nanmax(image))
            if not np.isnan(min_max).any():
                # self.sliders.ax.set_ylim(min_max)
                self.sliders.valmin, self.sliders.valmax = min_max

                # since we changed the axis limits, need to redraw tick labels
                draw_list.extend(
                    getattr(self.histogram.ax,
                            f'get_{self.sliders.slide_axis}ticklabels')()
                )

        # update histogram
        if self.has_hist:
            self.histogram.compute(image, self.histogram.bin_edges)
            draw_list.append(self.histogram.update())

        if self.clim_every and (self._draw_count % self.clim_every) == 0:
            # set the slider positions / color limits
            vmin, vmax = self.clim_from_data(image)
            self.imagePlot.set_clim(vmin, vmax)

            if self.sliders:
                draw_list = self.sliders.set_positions((vmin, vmax),
                                                       draw_on=False)

            # set the axes limits slightly wider than the clims
            if self.has_hist:
                self.histogram.autoscale_view()

            # if getattr(self.norm, 'interval', None):
            #     vmin, vmax = self.norm.interval.get_limits(
            #             _sanitize_data(image))

            # else:
            #     self.logger.debug('Auto clims: ({:.1f}, {:.1f})', vmin, vmax)

        #
        if draw:
            self.sliders.draw(draw_list)

        return draw_list
        # return i, image

    def _scroll(self, event):

        # FIXME: drawing on scroll.....
        # try:
        inc = [-1, +1][event.button == 'up']
        new = self._frame + inc
        if self.use_blit:
            self.frameSlider.drawon = False
        self.frameSlider.set_val(new)  # calls connected `update`
        self.frameSlider.drawon = True

        # except Exception as err:
        #     self.logger.exception('Scroll failed:')

    def play(self, start=None, stop=None, pause=0):
        """
        Show a video of images in the stack

        Parameters
        ----------
        n: int
            number of frames in the animation
        pause: int
            interval between frames in milliseconds

        Returns
        -------

        """

        if stop is None and start:
            stop = start
            start = 0
        if start is None:
            start = 0
        if stop is None:
            stop = self.nframes

        # save background for blitting
        # FIXME: saved bg should be without
        tmp_inviz = [self.frameSlider.poly, self.frameSlider.valtext]
        # tmp_inviz.extend(self.histogram.ax.yaxis.get_ticklabels())
        tmp_inviz.append(self.histogram.bars)
        for s in tmp_inviz:
            s.set_visible(False)

        fig = self.figure
        fig.canvas.draw()
        self.background = fig.canvas.copy_from_bbox(self.figure.bbox)

        for s in tmp_inviz:
            s.set_visible(True)

        self.frameSlider.eventson = False
        self.frameSlider.drawon = False

        # pause: inter-frame pause (millisecond)
        seconds = pause / 1000
        i = int(start)

        # note: the fastest frame rate achievable currently seems to be
        #  around 20 fps
        try:
            while i <= stop:
                self.frameSlider.set_val(i)
                draw_list = self.update(i)
                draw_list.extend([self.frameSlider.poly,
                                  self.frameSlider.valtext])

                fig.canvas.restore_region(self.background)

                # FIXME: self.frameSlider.valtext doesn't dissappear on blit

                for art in draw_list:
                    self.ax.draw_artist(art)

                fig.canvas.blit(fig.bbox)

                i += 1
                time.sleep(seconds)
        except Exception as err:
            raise err
        finally:
            self.frameSlider.eventson = True
            self.frameSlider.drawon = True

    # def blit_setup(self):

    # @expose.args()
    # def draw_blit(self, artists):
    #
    #     self.logger.debug('draw_blit')
    #
    #     fig = self.figure
    #     fig.canvas.restore_region(self.background)
    #
    #     for art in artists:
    #         try:
    #             self.ax.draw_artist(art)
    #         except Exception as err:
    #             self.logger.debug('drawing FAILED %s', art)
    #             traceback.print_exc()
    #
    #     fig.canvas.blit(fig.bbox)

    # def format_coord(self, x, y):
    #     s = ImageDisplay.format_coord(self, x, y)
    #     return 'frame %d: %s' % (self.frame, s)

    # def format_coord(self, x, y):
    #     col, row = int(x + 0.5), int(y + 0.5)
    #     nrows, ncols, _ = self.data.shape
    #     if (col >= 0 and col < ncols) and (row >= 0 and row < nrows):
    #         z = self.data[self._frame][row, col]
    #         return 'x=%1.3f,\ty=%1.3f,\tz=%1.3f' % (x, y, z)
    #     else:
    #         return 'x=%1.3f, y=%1.3f' % (x, y)


class VideoDisplayX(VideoDisplay):
    # TODO: improve memory performance by allowing coords to update via func

    marker_properties = dict(alpha=1, s=5, marker='x', color='r')

    def __init__(self, data, coords=None, **kws):
        """

        Parameters
        ----------
        data: array-like
            Image stack. shape (N, ypix, xpix)
        coords:  array_like, optional
            coordinate positions (yx) of apertures to display. This must be
            array_like with
            shape (N, k, 2) where k is the number of apertures per frame, and N
            is the number of frames.
        markers: str
            Sequence of markers
        kws:
            passed to `VideoDisplay`
        """

        VideoDisplay.__init__(self, data, **kws)

        #
        # check coords
        self.coords = coords
        self.has_coords = (coords is not None)
        if self.has_coords:
            coords = np.asarray(coords)
            if coords.ndim not in (2, 3) or (coords.shape[-1] != 2):
                raise ValueError('Coordinate array `coords` has incorrect '
                                 f'dimensions: {coords.shape}')
            if coords.ndim == 2:
                # Assuming single coordinate point per frame
                coords = coords[:, None]
            if len(coords) < len(data):
                self.logger.warning(
                    'Coordinate array contains fewer points ({}) than the '
                    'number of frames ({}).', len(coords), len(data)
                )

            # coord getter
            self.get_coords = self.get_coords_internal

            # create markers
            # set for frame 0
            self.marks = self.ax.scatter(*coords[0, :, ::-1].T,
                                         **self.marker_properties)
        else:
            # create markers
            self.marks = self.ax.scatter([0], [0], **self.marker_properties)
            self.marks.set_visible(False)

        # redraw markers after color adjust
        self.sliders.link(self.marks)

    def get_coords(self, i):
        return

    def get_coords_internal(self, i):
        i = int(round(i))
        return self.coords[i, :, ::-1]

    def update(self, i, draw=True):
        self.logger.debug('update')
        # i = round(i)
        draw_list = VideoDisplay.update(self, i, False)
        #
        if (coo := self.get_coords(i)) is not None:
            self.marks.set_offsets(coo)
            self.marks.set_visible(True)
            draw_list.append(self.marks)

        return draw_list


class VideoDisplayA(VideoDisplayX):
    # default aperture properties
    apProps = dict(ec='m', lw=1,
                   picker=False,
                   widths=7.5, heights=7.5)

    def __init__(self, data, coords=None, ap_props=None, **kws):
        """
        Optionally also displays apertures if coordinates provided.
        """
        if ap_props is None:
            ap_props = {}
        VideoDisplayX.__init__(self, data, coords, **kws)

        # create apertures
        props = VideoDisplayA.apProps.copy()
        props.update(ap_props)
        self.aps = self.create_apertures(**props)

    def create_apertures(self, **props):
        from obstools.aps import ApertureCollection

        props.setdefault('animated', self.use_blit)
        aps = ApertureCollection(**props)
        # add apertures to axes.  will not display yet if coordinates not given
        aps.add_to_axes(self.ax)
        return aps

    def update_apertures(self, i, *args, **kws):
        coords, *_ = args
        self.aps.coords = coords
        return self.aps

    def update(self, i, draw=True):
        # get all the artists that changed by calling parent update
        draw_list = VideoDisplayX.update(self, i, False)
        coo = self.marks.get_offsets()

        art = self.update_apertures(i, coo.T)
        draw_list.append(art)

        return draw_list
