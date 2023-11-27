"""
Efficient, scrollable image sequence visualisation.
"""

# std
import time
import warnings
import contextlib as ctx
from collections import abc

# third-party
import cycler
import numpy as np
from matplotlib.widgets import Slider

# local
from recipes import dicts
from recipes.config import ConfigNode

# relative
from ..utils import embossed
from ..image import ImageDisplay
from ..moves import ScrollAction


# ---------------------------------------------------------------------------- #
CONFIG = ConfigNode.load_module(__file__)

# ---------------------------------------------------------------------------- #


class VideoDisplay(ImageDisplay, ScrollAction):

    # TODO: lock the sliders in place with doubleclick

    # scrolling past the end leads to the beginning
    _scroll_wrap = CONFIG.scroll.wrap

    def __init__(self, data, clim_every=CONFIG.clim_every, **kws):
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
        data, self.nframes = self._check_data(data)
        self.clim_every = clim_every

        # Init scroll actions
        ScrollAction.__init__(self, rate_limit=CONFIG.scroll.rate_limit)
        self.on_scroll.add(self._scroll_frame)  # enable frame scrolling

        # setup image display
        connect = kws.pop('connect', True)  # don't connect methods just yet
        # parent needs 2D data.
        n = self._frame = 0
        ImageDisplay.__init__(self, data[n], connect=False, **kws)
        # save data (this can be array_like (or np.mmap))
        self.data = data

        # make frame slider
        pos_kws, slide_kws = dicts.split(CONFIG.slider.copy(), 'label', 'valfmt')
        label = slide_kws.pop('label')
        fsax = self.divider.append_axes(**pos_kws)
        self.frame_slider = fs = Slider(fsax, label, n, self.nframes, **slide_kws)
        self.frame_slider.on_changed(self.update)
        # fsax.xaxis.set_major_locator(ticker.AutoLocator())

        if self.use_blit:
            self.frame_slider.drawon = False
            self.add_art(fs.poly, fs._handle, fs.valtext,
                         self._cbar_hist_connectors.values())

            if self.has_sliders:
                # save slider positions in frame scroll background
                self.add_art(*(set(mv.draw_list)
                               for mv in self.sliders.movable.values()))

        # connect callbacks
        if connect:
            self.connect()

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
        self.logger.debug('Guessed figure size: ({:.1f}, {:.1f}).', *size)
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
            self.logger.debug('Reached the last frame. Scroll wrap is OFF.')
            i = max(i, self.nframes)

        i = int(round(i, 0))  # make sure we have an int
        self._frame = i  # store current frame

    def _scroll_frame(self, event):
        # move to next / previous frame
        if event.inaxes is not self.ax:
            return

        inc = [-1, +1][event.button == 'up']
        self.logger.debug('Scrolling to {} image.', ('next', 'previous')[inc < 0])
        i = self._frame + inc

        if self.use_blit:
            self.frame_slider.drawon = False

        # TODO: make frame_slider an AxesSlider to get the updated artistst
        # direct from set_positions
        with self.frame_slider._observers.blocked():
            self.frame_slider.set_val(i)  # don't call connected `update`
            art = self.update(i)

        self.frame_slider.drawon = True

        return self.artists, art

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
        draw_list = self.set_image_data(image)

        if draw:
            self.draw(draw_list)

        return draw_list

    def set_image_data(self, image):
        # set the image data
        self.image.set_data(image)  # does not update normalization

        # FIXME: normalizer fails with boolean data
        #  File "/usr/local/lib/python3.6/dist-packages/matplotlib/colorbar.py", line 956, in on_mappable_changed
        #   self.update_normal(mappable)
        # File "/usr/local/lib/python3.6/dist-packages/matplotlib/colorbar.py", line 987, in update_normal

        draw_list = list(self.artists)

        # update histogram
        if self.has_hist:
            self.histogram.compute(image, self.histogram.bin_edges)
            draw_list.append(self.histogram.update())

        if self.clim_every and (self._draw_count % self.clim_every) == 0:
            # set the slider positions / color limits
            vmin, vmax = self.clim_from_data(image)
            self.image.set_clim(vmin, vmax)

            if self.sliders:
                draw_list.extend(
                    self.sliders.set_positions((vmin, vmax), draw_on=False)
                )

            # set the axes limits slightly wider than the clims
            if self.has_hist:
                self.histogram.autoscale_view()

                # since we changed the axis limits, need to redraw tick labels
                # axis = 'xy'[int(self.histogram.orientation.lower().startswith('h'))]
                # draw_list.extend(getattr(self.histogram.ax,f'get_{axis}ticklabels')())

            # if getattr(self.norm, 'interval', None):
            #     vmin, vmax = self.norm.interval.get_limits(
            #             _sanitize_data(image))

            # else:
            #     self.logger.debug('Auto clims: ({:.1f}, {:.1f}).', vmin, vmax)
        return draw_list

    def play(self, start=None, stop=None, pause=0):
        """
        Show a video of images in the stack.

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
        # tmp_inviz.extend(self.histogram.ax.yaxis.get_ticklabels())
        tmp_inviz = [
            self.frame_slider.poly,
            self.frame_slider.valtext,
            self.histogram.bars,
        ]

        for s in tmp_inviz:
            s.set_visible(False)

        fig = self.figure
        fig.canvas.draw()
        self.background = fig.canvas.copy_from_bbox(self.figure.bbox)

        for s in tmp_inviz:
            s.set_visible(True)

        self.frame_slider.eventson = False
        self.frame_slider.drawon = False

        # pause: inter-frame pause (millisecond)
        seconds = pause / 1000
        i = int(start)

        # note: the fastest frame rate achievable currently seems to be
        #  around 20 fps
        try:
            while i <= stop:
                self.frame_slider.set_val(i)
                draw_list = self.update(i)
                draw_list.extend([self.frame_slider.poly,
                                  self.frame_slider.valtext])

                fig.canvas.restore_region(self.background)

                # FIXME: self.frame_slider.valtext doesn't dissappear on blit

                for art in draw_list:
                    self.ax.draw_artist(art)

                fig.canvas.blit(fig.bbox)

                i += 1
                time.sleep(seconds)
        except Exception as err:
            raise err
        finally:
            self.frame_slider.eventson = True
            self.frame_slider.drawon = True

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


with ctx.suppress(ValueError):
    MARKER_CYCLE = cycler.cycler(marker=list(CONFIG.features.pop('marker_cycle')))


class VideoFeatureDisplay(VideoDisplay):

    def __init__(self, data, coords=None,
                 marker_cycle=MARKER_CYCLE,
                 marker_style: dict = (),
                 **kws):
        """

        Parameters
        ----------
        data: array-like
            Image stack. shape (N, ypix, xpix)
        coords:  array_like, optional
            Coordinate positions (yx) of features to display. `coords` must be
            array_like with shape (N, [l], p, 2) where  N is the number of
            frames l is the (optional) number of features, and p is the number
            of data points per frame.
        markers: str
            Sequence of markers
        kws:
            passed to `VideoDisplay`
        """
        VideoDisplay.__init__(self, data, **kws)

        first = self.set_coords(coords)

        # create marker art
        if isinstance(marker_cycle, abc.MutableMapping):
            marker_cycle = cycler.cycler(**marker_cycle)
        else:
            marker_cycle = cycler.cycler(marker_cycle)

        marker_style = {**CONFIG.features, **(marker_style or {})}
        self.marks = self.mark(first, marker_cycle, **marker_style)

        if self.cbar:
            # NOTE, we don't want the cmap to switch for these when scrolling
            self.cbar.scroll.artists |= {*self.marks}

        if self.sliders:
            self.sliders.add_art(*self.marks)
            self.sliders.link(*self.marks)

    def set_coords(self, coords):
        # check coords
        if coords is None:
            # create masked xy coords for frame 0
            self.coords = None
            return np.array([[None], [None]])

        coords = np.asanyarray(coords)
        nd = coords.ndim
        n = len(self.data)
        m, *_, d = coords.shape

        if nd not in {2, 3, 4} or (d != 2):
            raise ValueError(
                'Coordinate array `coords` has incorrect dimensions: '
                f'{coords.shape}. Expected array dimentions to be: '
                '([n_frames], [n_features], n_points, 2)'
            )

        if n != m and nd in {2, 3}:
            # use as initializing coordinates for first frame
            self.logger.info('Using coordinates as initializer: {}.', coords)
            self.coords = None
            return coords

        # have array
        self.get_coords = self._get_coords_internal

        if nd == 2:
            # Assuming single source point per frame with single feature
            coords = coords[:, None, None]

        self.coords = coords
        return coords[0]

    def mark(self, xy, marker_cycle, emboss=1.5, alpha=1, **style):
        # create markers

        if xy.ndim == 2:
            xy = xy[None, ...]

        art = []
        style = {**dict(style or CONFIG.features),
                 'alpha': alpha}

        for points, kws in zip(xy, marker_cycle):
            if kws['marker'] not in '+x':
                kws['edgecolor'] = kws.pop('color', None)
                kws['facecolor'] = 'none'

            #
            print({**style, **kws})
            marks = self.ax.scatter(*points.T, **{**style, **kws})

            if emboss:
                embossed(marks, emboss, alpha=alpha)

            # redraw markers after color adjust
            self.sliders.link(marks)
            art.append(marks)

        self.add_art(*art)
        return art

    def get_coords(self, i):
        # Subclasses can implement
        # return None by default
        return

    def _get_coords_internal(self, i):
        i = int(round(i))
        return self.coords[i]

    # def set_frame(self, i):
    #     super().set_frame(i)

    def update(self, i, draw=True):

        # update image
        draw_list = super().update(i, False)
        i = self.frame  # rounded and wrapped

        # Try get feature coordinates
        coords = self.get_coords(i)
        self.logger.debug('Coords: {}\n{}.', coords.shape, coords)

        if coords is None:
            self.logger.debug('Not updating marks since coords is {}.', coords)
            return draw_list

        # update markers
        if coords.ndim == 2:
            coords = coords[None]

        self.logger.debug('Updating coords: {}.', coords)
        for coo, marks in zip(coords, self.marks):
            marks.set_offsets(coo)
            marks.set_visible(True)
            draw_list.append(marks)

        return draw_list


class VideoApertureDisplay(VideoFeatureDisplay):  # VideoApertureDisplay
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
        VideoFeatureDisplay.__init__(self, data, coords, **kws)

        # create apertures
        props = VideoApertureDisplay.apProps.copy()
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
        draw_list = super().update(i, False)
        coo = self.marks.get_offsets()

        art = self.update_apertures(i, coo.T)
        draw_list.append(art)

        return draw_list


# aliases
VideoDisplayX = VideoFeatureDisplay
VideoDisplayA = VideoApertureDisplay
