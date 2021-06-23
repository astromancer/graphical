
# std libs
import logging
import itertools as itt
from collections import defaultdict, OrderedDict

# third-party libs
import numpy as np
import more_itertools as mit
from matplotlib.lines import Line2D
from matplotlib.artist import Artist
from matplotlib.container import ErrorbarContainer
from matplotlib.transforms import Affine2D, blended_transform_factory as btf

# local libs
from scrawl.connect import ConnectionMixin, mpl_connect
from recipes import pprint
from recipes.logging import LoggingMixin, get_module_logger

# from matplotlib.axes import Axes
# from matplotlib.offsetbox import DraggableBase


# module level logger
logger = get_module_logger()
logging.basicConfig()
logger.setLevel(logging.INFO)


# from recipes.decor.misc import unhookPyQt

# logging.basicConfig(level=logging.DEBUG)


# ==============================================================================
def draggable_artist_factory(art, offset, annotate, **kws):
    if isinstance(art, ErrorbarContainer):
        from scrawl.draggable.errorbars import DraggableErrorbarContainer
        draggable = DraggableErrorbarContainer(art,
                                               offset=offset,
                                               annotate=annotate,
                                               **kws)
        markers, _, _ = draggable
        return markers, draggable
        # map Line2D to DraggableErrorbarContainer. The picker unavoidably
        # returns the markers.

    if isinstance(art, Line2D):
        # from scrawl.draggable.lines import DraggableLine
        return art, DraggableBase(art, offset, annotate, **kws)

    else:
        raise ValueError


# ==============================================================================
# @expose.args()
def fpicker(artist, event):
    """
    an artist picker that works for clicks outside the axes. ie. artist
    that are not clipped
    """
    logger.debug('fpicker: %s', artist)

    if event.button != 1:  # otherwise intended reset will select
        logger.debug('wrong button!')
        return False, {}

    tf = artist.contains(event)
    logger.debug('fpicker: artist.contains(event) %s', tf)
    return tf
    #
    # transformed_path = artist._get_transformed_path()
    # path, affine = transformed_path.get_transformed_path_and_affine()
    # path = affine.transform_path(path)
    # xy = path.vertices
    # xt, yt = xy.T#[:, 0]
    # # yt = xy[:, 1]
    #
    # # Convert pick radius from points to pixels
    # pixels = artist.figure.dpi / 72. * artist.pickradius
    #
    # xd, yd = xt - event.x, yt - event.y
    # distance of click from points in pixels (display coords)
    # prox = np.sqrt(xd ** 2 + yd ** 2)
    #
    # picked = prox - pixels < 0 #
    #
    # logging.debug('%s', (pixels, prox, artist.pickradius, picked))
    # props = dict(proximity=prox, hello='world')
    #
    # logging.debug('picked: %r: point %s', artist, np.where(picked))
    # return picked.any(), props


def filter_non_artist(objects):
    for o in mit.collapse(objects):
        if o is None:
            continue
        if isinstance(o, Artist):
            yield o
            continue

        # warn if not art
        logger.warning('Object %r is not a matplotlib Artist' % o)


def art_summary(artists):
    col = defaultdict(int)
    if artists is None:
        return ''
    if isinstance(artists, Artist):
        return {artists: 1}

    for art in artists:
        col[type(art)] += 1
    return str(col)


def null(_):
    return ''


class IndexableOrderedDict(OrderedDict):
    def __missing__(self, key):
        if isinstance(key, int):
            return self[list(self.keys())[key]]

        raise KeyError(str(key))


class Observers(LoggingMixin):
    """Container class for observer functions"""

    def __init__(self):
        self.counter = itt.count()
        self.funcs = OrderedDict()
        self.active = {}

    def __repr__(self):
        return '\n'.join(
            (self.__class__.__name__,
             '\n'.join(map(self._repr_observer, self.funcs.keys()))))

    def _repr_observer(self, id_):
        f, a, args, kws = self.funcs[id_]
        s = '%i%s: %s' % (id_, ' *'[a], pprint.method(f))
        par = ['x', 'y']
        par.extend(map(str, args))
        par.extend(map('='.join, kws.items()))
        return s + ', '.join(par).join('()')

    def add(self, func, *args, **kws):
        """
        Add an observer function.

        When the artist is moved / picked, *func* will be called with the new
        coordinate position as arguments.  *func* should return any artists
        that it changes. These will be drawn if blitting is enabled.
        The signature of *func* is therefor:

            draw_list = func(x, y, *args, **kws)`

        Parameters
        ----------
        func
        args
        kws

        Returns
        -------
        A connection id is returned which can be used to remove the method
        """
        if not callable(func):
            raise TypeError('`func` should be callable')

        id_ = next(self.counter)
        self.funcs[id_] = (func, args, kws)
        self.active[func] = True
        return id_

    def remove(self, id_):
        fun, args, kws = self.funcs.pop(id_, None)
        self.active.pop(fun)
        return fun

    def activate(self, fun_or_id):
        """
        Reactivate a non-active observer.  This method is useful for toggling
        the active state of an observer function without removing and re-adding
        it (and it's parameters) to the dict of functions. The function will use
        parameters and keywords (if any) that were initially passed when it was
        added.

        Parameters
        ----------
        fun_or_id: callable, int
            The function (or its identifier) that will be activated 
        """
        self._set_active(fun_or_id, True)

    def deactivate(self, fun_or_id):
        """
        Deactivate an active observer. 

        Parameters
        ----------
        fun_or_id: callable, int
            The function (or its identifier) that will be activated 
        """
        self._set_active(fun_or_id, False)

    def _set_active(self, fun_or_id, tf):
        if not callable(fun_or_id) and fun_or_id in self.funcs:
            # function id passed instead of function itself
            fun, *_ = self.funcs[fun_or_id]
        else:
            fun = fun_or_id

        if fun in self.active:
            self.active[fun] = tf
        else:
            self.logger.warning(
                'Function %r is not an observer! Use `add(fun, *args, **kws)'
                'to make it an observer',  fun)

    def __call__(self, x, y):
        """
        Run all active observers for current data point

        Parameters
        ----------
        x, y

        Returns
        -------
        Artists that need to be drawn
        """
        
        if self.logger.getEffectiveLevel() < logging.DEBUG:
            _art_summary = null
        else:
            _art_summary = art_summary

        # Artists that need to be drawn (from observer functions)
        artists = []
        for cid, (func, args, kws) in self.funcs.items():
            if self.active[func]:
                try:
                    art = func(x, y, *args, **kws)
                    self.logger.debug('observer: %s(%.3f, %.3f): %s',
                                      func.__name__, x, y, _art_summary(art))

                    if isinstance(art, (list, tuple)):  # np.ndarray
                        artists.extend(art)

                    elif art is not None:
                        artists.append(art)

                except Exception:
                    self.logger.exception('Observers error')

        return artists


# ******************************************************************************
class DraggableBase(LoggingMixin):  # TODO: use as mixin?
    """base class for draggable artists"""

    annotation_format = '[%+3.2f]'

    # @classmethod
    # def from_data(cls, data):
    #     # construct from data?
    #     raise NotImplementedError

    def __init__(self, artist, offset=(0., 0.), annotate=False, haunted=False,
                 trapped=False, **kws):
        """

        Parameters
        ----------
        artist
        offset
        annotate
        haunted
        trapped
        kws
        """

        # Line2D.__init__(self, *line.get_data())
        # self.update_from(line)
        self.artist = artist
        self._original_transform = artist.get_transform()

        self.clipped = False
        self.trapped = trapped
        self._xmin, self._xmax = np.nan, np.nan
        self._ymin, self._ymax = np.nan, np.nan
        self._offset = np.array(offset)

        self.ref_point = np.array([artist.get_xdata()[0],
                                   artist.get_ydata()[0]])
        # NOTE: this method of getting a reference point is not appropriate for
        #  all artists

        #
        self.annotated = annotate
        self.ghost = None

        # make the lines pickable
        if not artist.get_picker():
            # artist.update()
            artist.set_pickradius(10)
            artist.set_picker(fpicker)  # fpicker  # set_pickradius

        # Manage with ConnectionMixin?
        self.linked = []
        self.on_picked = Observers()
        self.on_move = Observers()
        self.on_release = Observers()
        # self.on_clipped = Observers()

        # add the shift method to observers
        self.on_move.add(self.move_to)
        self.on_release.add(self.update)

        # control whether linked artists are updated when the parent is updated
        # self._propagate = True

        # Initialize offset texts
        ax = artist.axes
        if self.annotated:  # TODO: manage through linked
            self.text_trans = btf(ax.transAxes, ax.transData)
            self.ytxt = np.mean(artist.get_ydata())
            self.annotation = ax.text(1.005, self.ytxt, '')
            # self.on_move(self.shift_text)

        if haunted:
            self.haunt()

        self._locked_at = np.full(2, np.nan)

    def __str__(self):
        return 'Draggable' + str(self.artist)  # TODO if bounded show bounds

    def get_offset(self):
        return self._offset

    def set_offset(self, offset):
        self._offset = np.where(np.isnan(self._locked_at), offset,
                                self._locked_at)

    offset = property(get_offset, set_offset)

    def get_position(self):
        return self.ref_point + self.offset

    position = property(get_position)

    @property
    def draw_list(self):
        return [self.artist] + [lnk.artist for lnk in self.linked]

    def lock(self, which):
        """Lock movement for x or y coordinate"""
        ix = 'xy'.index(which.lower())
        self._locked_at[ix] = self.offset[ix]

    def free(self, which):
        """Release movement lock for x or y coordinate"""
        ix = 'xy'.index(which.lower())
        self._locked_at[ix] = None

    def lock_x(self):
        """Lock x coordinate at current position"""
        self.lock('x')

    def free_x(self):
        """Release x coordinate lock"""
        self.free('x')

    def lock_y(self):
        """Lock y coordinate at current position"""
        self.lock('y')

    def free_y(self):
        """Release y coordinate lock"""
        self.free('y')

    @property
    def xlim(self):
        if self.trapped:
            xmin, xmax = self.artist.axes.get_xlim()
            return np.nanmax((xmin, self._xmin)), np.nanmin((xmax, self._xmax))
        return self._xmin, self._xmax

    @xlim.setter
    def xlim(self, values):
        self.xmin, self.xmax = np.sort(values)

    @property
    def xmin(self):
        return self.xlim[0]

    @xmin.setter
    def xmin(self, value):
        self._xmin = value

    @property
    def xmax(self):
        return self.xlim[1]

    @xmax.setter
    def xmax(self, value):
        self._xmax = value

    @property
    def ylim(self):
        if self.trapped:
            ymin, ymax = self.artist.axes.get_ylim()
            return np.nanmax((ymin, self._ymin)), np.nanmin((ymax, self._ymax))
        return self._ymin, self._ymax

    @ylim.setter
    def ylim(self, values):
        self.ymin, self.ymax = np.sort(values)

    @property
    def ymin(self):
        return self.ylim[0]

    @ymin.setter
    def ymin(self, value):
        self._ymin = value

    @property
    def ymax(self):
        return self.ylim[1]

    @ymax.setter
    def ymax(self, value):
        self._ymax = value

    def limit(self, xb=None, yb=None):
        """
        Restrict movement of the artist to a particular interval / box

        Parameters
        ----------
        xb
        yb

        Returns
        -------

        """
        if (xb is None) and (yb is None):
            raise ValueError('Need either x, or y limits (or both)')

        if xb is not None:
            self._xmin, self._xmax = np.sort(xb)

        if yb is not None:
            self._ymin, self._ymax = np.sort(yb)

    def contains(self, event):
        if event.button != 1:  # otherwise intended reset will select
            self.logger.debug('wrong button for picking!')
            return False, {}
        return self.artist.contains(self, event)

    def link(self, *artists):
        """
        Link another artist or artist to this one to make them co-moving

        Parameters
        ----------
        artists: a sequence of artists that will be linked to this one

        Returns
        -------

        """
        linked = []
        for drg in artists:
            if not isinstance(drg, DraggableBase):
                drg = DraggableBase(drg)
            linked.append(drg)

        self.linked.extend(linked)
        return linked

    def unlink(self, *artists):
        """ """
        for drg in artists:
            if drg in self.linked:
                i = self.linked.index(drg)
                self.linked.pop(i)

    def clip(self, x, y):
        # TODO: validate method for more complex movement restrictions
        # set min / max here
        self.clipped = False
        xlim = xmin, xmax = self.xlim
        if not np.isnan(xlim).all():
            self.logger.debug('clipping %s: x [%.2f, %.2f]', self, xmin, xmax)
            x = np.clip(x, xmin, xmax)
            if x in xlim:
                self.clipped = True

        ylim = ymin, ymax = self.ylim
        if not np.isnan(ylim).all():
            self.logger.debug('clipping %s: y [%.2f, %.2f]', self, ymin, ymax)
            y = np.clip(y, ymin, ymax)
            if y in ylim:
                self.clipped = True

        return x, y

    def move_to(self, x, y):
        """
        Shift the artist to the position (x, y) in data coordinates.  Note
        the input position will be changed before applying the shift if the
        draggable is restricted

        Parameters
        ----------
        x
        y

        Returns
        -------

        """
        # constrain positions
        x, y = self.clip(x, y)
        offset = np.subtract((x, y), self.ref_point)

        self.logger.debug('shifting %s to (%.3f, %.3f)', self, x, y)
        self.move_by(offset)
        self.logger.debug('offset %s is (%.3f, %.3f)', self, *self.offset)

        return self.artist

    def move_by(self, offset):
        """
        move the artist by offsetting from initial position

        Parameters
        ----------
        offset

        Returns
        -------

        """

        self.offset = offset  # will adhere to positional locks
        self.logger.debug('moving: %s %s', self, offset)

        # add the offset with transform
        offset_trans = Affine2D().translate(*self.offset)
        trans = offset_trans + self._original_transform
        self.artist.set_transform(trans)

    def update(self, x, y):
        self.logger.debug('update: %r', self)

        # Artists that need to be drawn (from observers)
        pos = self.position
        draw_list = self.on_move(x, y)
        # get the actual delta (respecting position locks etc)
        delta = self.position - pos
        self.logger.debug('DELTA %s', delta)

        # if propagate:
        for lnk in self.linked:
            # The linked artists will have a different offset since they can
            # moved independently
            to_draw = lnk.update_offset(delta)
            draw_list.extend(to_draw)

        return draw_list

    def update_offset(self, offset):
        xy = self.position + offset  # new position
        # self.logger.debug('update_offset %s to (%.3f, %.3f)', lnk, x, y)
        return self.update(*xy)

    # def set_axes_lim(self):
    #     self.position
    #
    # NOTE: this function can be avoided if you make a DraggableText?
    # def shift_text(self, offset):
    #     """Shift the annotation by an offset"""
    #     # offset = val - self.ytxt
    #     offset_trans = Affine2D().translate(*offset)
    #     trans = offset_trans + self._original_transform
    #
    #     txt = self.annotation
    #     txt.set_transform(offset_trans + self.text_trans)
    #     txt.set_text(self.annotation_format % offset)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def haunt(self, alpha=0.25):
        """ """
        # create a ghost artist of the same class
        self.ghost = self.__class__(self.artist, offset=self.offset,
                                    annotate=self.annotated)

        self.ghost.artist.set_alpha(alpha)
        self.ghost.set_visible(False)
        self.artist.axes.add_artist(self.ghost)

        # NOTE: can avoid this if statement by subclassing...
        if self.annotated:
            self.ghost.annotation.set_alpha(0.2)

    # TODO: inherit from the artist to avoid redefining these methods????
    def set_transform(self, trans):
        self.artist.set_transform(trans)

    def set_visible(self, vis):
        self.artist.set_visible(vis)

        # NOTE: can avoid this if statement by subclassing artists
        if self.annotated:
            self.annotation.set_visible(vis)

    def set_active(self, b):
        self._active = bool(b)

    def set_animated(self, b=None):
        """set animation state for all linked artists"""
        if b is None:
            b = not self.artist.get_animated()

        self.artist.set_animated(b)
        for drg in self.linked:
            drg.set_animated(b)  # note recurrence

    def draw(self, renderer=None):
        if renderer is None:
            renderer = self.artist.figure.canvas.renderer
        self.artist.draw(renderer)


# ****************************************************************************************************
class DragMachinery(ConnectionMixin):
    # TODO: some of the functionality here belongs in BlitHelper class
    """
    Class for managing draggable artists.  Artists are moved by applying a
    translation (transform) in the data space. This allows objects that live
    in arbitrary coordinates to be moved by dragging them with the mouse.
    """
    # TODO: haunt, link
    # TODO: #incorp best methods from mpl.DraggableBase
    supported_artists = (Line2D, ErrorbarContainer)

    @staticmethod
    def artist_factory(art, offset, annotate, **kws):
        return draggable_artist_factory(art, offset, annotate, **kws)

    def __init__(self, artists=None, offsets=None, annotate=True, haunted=False,
                 auto_legend=True, use_blit=True, **legendkw):
        """


        Parameters
        ----------
        artists
        offsets
        annotate
        linked
        haunted
        auto_legend
        use_blit
        legendkw
        """
        self._ax = None
        if artists is None:
            artists = []

        self.selection = None
        self.ref_point = None
        self.up = None

        if offsets is None:
            offsets = np.zeros((len(artists), 2))
        else:
            offsets = np.asarray(offsets)
            if offsets.ndim < 2:
                raise ValueError

        # remember (will need for reset)
        self._original_offsets = offsets
        self.delta = np.zeros(2)  # in case of pick without motion

        # initialize mapping
        self.draggable = IndexableOrderedDict()

        # initialize auto-connect
        ConnectionMixin.__init__(self)

        # flag for blitting behaviour
        self._use_blit = use_blit

        # build the draggable objects
        for art, offset in zip(artists, offsets):
            self.add_artist(art, offset, annotate, haunted)

            # TODO:
            # enable legend picking
            # self.legend = None
            # if legendkw or auto_legend:
            # self.legend = DynamicLegend(ax, artists, legendkw)
            # self.legend.connect()

        self._draw_count = 0
        # self.canvas.mpl_connect('draw_event', self._on_draw)

        self.background = None

    def __getitem__(self, key):
        """hack for quick indexing"""
        # OR inherit from dict????
        return self.draggable[key]

    @property
    def offsets(self):
        return np.array([drag.offset for drag in self])

    def add_artist(self, artist, offset=(0, 0), annotate=True, haunted=False,
                   **kws):
        """add a draggable artist"""
        key, drg = self.artist_factory(artist,
                                       offset=offset,
                                       annotate=annotate,
                                       haunted=haunted, **kws)
        self.draggable[key] = drg
        self._original_offsets = np.r_['0,2', self._original_offsets, offset]

        return drg

    @property
    def ax(self):
        if self._ax is not None:
            return self._ax

        if len(self.draggable) == 0:
            raise ValueError('%s does not contain any artists yet.'
                             % self.__class__.__name__)
        return self.draggable[0].artist.axes

    @property
    def figure(self):
        return self.ax.figure

    @property
    def canvas(self):
        return self.figure.canvas

    @property
    def artists(self):
        return list(self.draggable.keys())

    @property
    def use_blit(self):
        return self._use_blit and (
            self.canvas is not None) and self.canvas.supports_blit

    def lock(self, which):
        for art, drg in self.draggable.items():
            drg.lock(which)

    def free(self, which):
        for art, drg in self.draggable.items():
            drg.free(which)

    def lock_x(self):
        self.lock('x')

    def free_x(self):
        self.free('x')

    def lock_y(self):
        self.lock('y')

    def free_y(self):
        self.free('y')

    # def validation(self, func):
    #     # Note: that validation is not the best way of enforcing limits since
    #      fast mouse movements can leave the axis while the previous known
    #      position is nowhere near the limits.
    #     for art, drg in self.draggable.items():
    #         drg.validation(func)

    def limit(self, x=None, y=None):
        """
        Set x and/or y limits for all draggable artists

        Parameters
        ----------
        x
        y

        """
        self.logger.debug('limit %s, %s', x, y)
        for art, drg in self.draggable.items():
            drg.limit(x, y)

    def reset(self):
        """reset the plot positions to original"""
        self.logger.debug('resetting!')
        for draggable, off in zip(self.draggable.values(),
                                  self._original_offsets):
            self.update(draggable, draggable.ref_point)

            # self.canvas.draw()

    @mpl_connect('button_press_event')
    def on_click(self, event):

        # print( 'on_click', repr(self.selection ))
        """reset plot on middle mouse"""
        if event.button == 2:
            self.reset()
        else:
            return

    def _ignore_pick(self, event):
        """Filter pick events"""
        if event.mouseevent.button != 1:
            return True

        if event.artist not in self.draggable:
            return True

        # avoid picking multiple artist simultaneously
        # #TODO more intelligence
        if self.selection:
            #  prefer the artist with closest proximity to mouse event
            self.logger.debug('Multiple picks! ignoring: %s', event.artist)
            return True

        return False

    @mpl_connect('pick_event')
    def on_pick(self, event):
        """Pick event handler."""

        if self._ignore_pick(event):
            return

        self.logger.debug('picked: %r: %s', event.artist, vars(event))

        # get data coordinates of pick
        self.selection = event.artist
        draggable = self.draggable[self.selection]
        # get data coordinates
        # xy display coordinate
        xy_disp = event.mouseevent.x, event.mouseevent.y
        # xy in data coordinates
        xy_data = self.ax.transData.inverted().transform(xy_disp)
        # set reference point (to calculate distance of movement)
        self.ref_point = np.subtract(xy_data, draggable.offset)

        # run the on_picked methods for this draggable
        draggable.on_picked(*xy_data)

        # connect motion_notify_event for dragging the selected artist
        self.add_connection('motion_notify_event', self.on_motion)

        if self.use_blit:
            # TODO: need method to get artists that will be changed by this
            #  artist
            draggable.set_animated(True)

            # call update here to avoid artists disappearing on click and hold
            # without move.  Also, this gives us the artists which are animated
            # by the move
            draw_list = draggable.update_offset((0, 0))
            for art in filter_non_artist(draw_list):
                art.set_animated(True)

                # self.update(draggable, xy_data)

                # draggable.save_offset()
                # current_offset = draggable.offset

                # make the ghost artists visible
                # for link in draggable.linked:
                # set the ghost vis same as the original vis for linked
                # for l, g in zip(link.get_children(),
                # link.ghost.get_children()):
                # g.set_visible(l.get_visible())
                # print( 'visible!', g, g.get_visible() )

                # print('picked', repr(self.selection), event.artist)

    def on_motion(self, event):
        """
        Handle movement of the selected artist by the mouse.
        """
        # TODO: pull in draggableBase on_motion_blit
        # if we want the artist to respond to events only inside the axes -
        # may not be desirable
        # if event.inaxes != self.ax:
        # return

        if event.button != 1:
            return

        if self.selection:
            self.logger.debug('dragging: %s', self.selection)
            draggable = self.draggable[self.selection]

            xy_disp = event.x, event.y
            xy_data = x, y = self.ax.transData.inverted().transform(xy_disp)
            # offset from original data
            self.delta = delta = xy_data - self.ref_point

            # difference between current position and previous offset position
            self.logger.debug('on_motion: delta %s; ref %s', delta,
                              self.ref_point)

            # move this artist and all its dependants
            # pos = draggable.position
            self.update(draggable, xy_data)
            # FIXME: only do if dragging = True ??
            # self._shift = draggable.position - pos

            # if draggable.clipped:
            #     draggable.on_clipped(x, y)
            #     print('HI')

            # if dragging set offset??

            # for link in draggable.linked:
            # link.ghost.move(delta)

            # print( [(ch.get_visible(), ch.get_alpha())
            # for ch in link.ghost.get_children()] )

            # link.ghost.draw(self.canvas.renderer)
            # print('...')

            # self.canvas.blit(self.figure.bbox)  #self.ax.bbox??

    @mpl_connect('button_release_event')
    def on_release(self, event):

        # print( 'release top', repr(self.selection ))

        if event.button != 1:
            return

        if self.selection:
            self.logger.debug('on_release: %r', self.selection)
            # Remove dragging method for selected artist
            self.remove_connection('motion_notify_event')

            xy_disp = event.x, event.y  # NOTE: may be far outside allowed range
            xy_data = x, y = self.ax.transData.inverted().transform(xy_disp)
            # xy_data = self.delta + self.ref_point
            self.logger.debug('on_release: delta %s', self.delta)

            draggable = self.draggable[self.selection]
            draw_list = draggable.on_release(x, y)

            print('release', draw_list)

            self.logger.debug('on_release: offset %s %s', draggable,
                              draggable.offset)

            if self.use_blit:
                self.draw_blit(draw_list)
                for art in filter_non_artist(draw_list):
                    art.set_animated(False)

                    # save offset
                    # for linked in draggable.linked:
                    # linked.move( self.delta )
                    # linked.offset = self.delta
                    # linked.ghost.set_visible(False)

                    # do_legend()
                    # self.canvas.draw()
                    # if self._draw_on:
                    # if self._use_blit:
                    # self.canvas.restore_region(self.background)
                    # draggable.draw()
                    # self.canvas.blit(self.figure.bbox)
                    # self.selection.set_animated(False)
                    # else:
                    # self.canvas.draw()

        self.selection = None
        # self.up = None
        # self.delta.fill(0)
        # print( 'release', repr(self.selection ))
        # print()

    @mpl_connect('resize_event')
    def on_resize(self, event):
        """Save the background for blit after canvas resize"""
        self.save_background()

    # @unhookPyQt
    def update(self, draggable, xy_data, draw_on=True):
        """
        Draw all artists that where changed by the motion

        Parameters
        ----------
        draggable
        xy_data
        draw_on

        Returns
        -------
        list of artists
        """
        draw_list = draggable.update(*xy_data)

        # draw the canvas / blit if required
        if draw_on:
            self.draw(draw_list)

        return draw_list

    def blit_setup(self, artists=()):  # save_background
        """
        Setup canvas for blitting. First make all the artists in the list
        invisible, then redraw the canvas and save, then redraw all the artists.
        """
        fig = self.figure
        artists = list(filter_non_artist(artists))

        # set artist animated
        for art in artists:
            art.set_animated(True)
            art.set_visible(False)

        fig.canvas.draw()
        background = fig.canvas.copy_from_bbox(fig.bbox)

        for art in artists:
            art.draw(fig.canvas.renderer)
            art.set_animated(False)
            art.set_visible(True)

        fig.canvas.blit(fig.bbox)
        return background

    @mpl_connect('draw_event')
    def _on_draw(self, event):
        self.logger.debug('draw %s', self._draw_count)
        self._draw_count += 1

    def draw(self, artists):
        # TODO: could set this method dynamically
        # print('draw')
        if self.use_blit:
            self.draw_blit(artists)
        else:
            self.canvas.draw()

    def draw_blit(self, artists):
        self.canvas.restore_region(self.background)
        # TODO: check for uniqueness to prevent unnecessary duplicate draw
        for art in sorted(filter_non_artist(artists), key=Artist.get_zorder):
            # self.logger.info('drawing: %s' % art)
            art.draw(self.canvas.renderer)

        # After rendering all the needed artists, blit each axes individually.
        # for ax in set(self.figure.axes):
        #     ax.figure.canvas.blit(ax.figure.bbox)

        self.canvas.blit(self.figure.bbox)

    def save_background(self):
        # get all the movable artists by "moving" them with zero offset
        artists = []
        for drg in self.draggable.values():
            art = drg.update_offset((0, 0))
            artists.extend(art)

        # save background
        self.background = self.blit_setup(artists)

    # def connect(self):
    #     # connect the methods decorated with `@mpl_connect`
    #     ConnectionMixin.connect(self)
    #
    #     self.save_background()
