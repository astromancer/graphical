import numpy as np
from copy import copy

import collections as coll

from matplotlib.lines import Line2D
import matplotlib.collections as mcoll
from matplotlib.container import ErrorbarContainer
from matplotlib.legend_handler import HandlerErrorbar
from matplotlib.transforms import blended_transform_factory as btf
from matplotlib.transforms import Affine2D

import itertools as itt
import more_itertools as mit

from recipes.containers.list_ import flatten

from graphing.misc import ConnectionMixin, mpl_connect

# from PyQt4.QtCore import pyqtRemoveInputHook, pyqtRestoreInputHook
# from decor.misc import unhookPyQt#, expose

from IPython import embed
# from pprint import pprint

# TODO: Convenience methods (factory) for draggable artists

# TODO: Convert to general class for draggable artists.  see: matplotlib.offsetbox.DraggableBase
from matplotlib.offsetbox import DraggableBase


class Foo(DraggableBase):
    artist_picker = 5

    def __init__(self, ref_artist, use_blit=False, free_axis='both'):
        DraggableBase.__init__(self, ref_artist, use_blit=False)
        self._original_transform = ref_artist.get_transform

    def save_offset(self):
        art = self.ref_artist
        self.ghost = ghost = copy(art)
        art.set_alpha(0.2)
        ax = art.get_axes()
        ax.add_artist(ghost)

    def update_offset(self, dx, dy):
        print(dx, dy)
        trans_offset = ScaledTranslation(dx, dy, IdentityTransform())
        trans = ax.transData + trans_offset

        self.ref_artist.set_transform(trans)

    def finalize_offset(self):
        # print( vars(self) )
        self.ghost.remove()
        self.ref_artist.set_alpha(1.0)
        self.canvas.draw()


# ****************************************************************************************************
def is_line(o):
    return isinstance(o, Line2D)


# def flatten_nested_dict(d):
# items = []
# for v in d.values():
# if isinstance(v, coll.MutableMapping):
# items.extend( flatten_nested_dict(v) )
# else:
# items.append(v)
# return items


# ****************************************************************************************************
class ReorderedErrorbarHandler(HandlerErrorbar):
    """
    Sub-class the standard errorbar handler to make the legends pickable.
    We pick on the markers (Line2D) of the ErrorbarContainer.  We know that the
    markers will always be created, even if not displayed.  The bar stems and
    caps may not be created.  We therefore re-order the ErrorbarContainer
    where it is created by the legend handler by intercepting the create_artist
    method
    """

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __init__(self, xerr_size=0.5, yerr_size=None,
                 marker_pad=0.3, numpoints=None, **kw):
        HandlerErrorbar.__init__(self, xerr_size, yerr_size,
                                 marker_pad, numpoints, **kw)
        self.eventson = True

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # @expose.args()
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        """
        x, y, w, h in display coordinate w/ default dpi (72)
        fontsize in points
        """
        xdescent, ydescent, width, height = self.adjust_drawing_area(
                legend, orig_handle,
                handlebox.xdescent,
                handlebox.ydescent,
                handlebox.width,
                handlebox.height,
                fontsize)

        artists = self.create_artists(legend, orig_handle,
                                      xdescent, ydescent, width, height,
                                      fontsize, handlebox.get_transform())

        container = NamedErrorbarContainer(artists[:-1],
                                           # NOTE: we are completely ignoring legline here
                                           orig_handle.has_xerr,
                                           orig_handle.has_yerr)

        # create_artists will return a list of artists.
        for art in container.get_children():
            handlebox.add_artist(art)

        # these become the legendHandles
        return container

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize,
                       trans):
        # from matplotlib.collections import LineCollection

        #  call the parent class function
        artists = HandlerErrorbar.create_artists(self, legend, orig_handle,
                                                 xdescent, ydescent,
                                                 width, height, fontsize,
                                                 trans)

        # Identify the artists. just so we know what each is
        # NOTE: The order here is different to that in ErrorbarContainer, so we
        #  re-order
        barlinecols, lines = mit.partition(is_line, artists)
        barlinecols = tuple(barlinecols)
        *caplines, legline, legline_marker = lines

        xerr_size, yerr_size = self.get_err_size(legend, xdescent, ydescent,
                                                 width, height,
                                                 fontsize)
        # NOTE: second time calling this (already done in
        # HandlerErrorbar.create_artists)

        legline_marker.set_pickradius(xerr_size * 1.25)
        legline_marker.set_picker(ErrorbarPicker())

        return [legline_marker, caplines, barlinecols, legline]


# ****************************************************************************************************
class ErrorbarPicker():
    """Hadles picking artists in the legend"""
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    parts = ('markers', 'stems', 'caps')

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @staticmethod
    def __call__(artist, event):
        """
        Hack the picker to emulate true artist picking in the legend. Specific
        implementation for errorbar plots. Pretent that the points / bars / caps
        are selected, based on the radius from the central point. Pass the
        "part" description str as a property to the event from where it can be
        used by the on_pick method.
        """
        # print( 'Picker', )
        # print( 'Picker', event.button )

        if event.button != 1:  # otherwise intended reset will select
            # print('wrong button asshole!')
            return False, {}

        # print('........')

        props = {}

        # Convert points to pixels
        transformed_path = artist._get_transformed_path()
        path, affine = transformed_path.get_transformed_path_and_affine()
        path = affine.transform_path(path)
        xy = path.vertices
        xt = xy[:, 0]
        yt = xy[:, 1]

        # Convert pick radius from points to pixels
        pixels = artist.figure.dpi / 72. * artist.pickradius

        # Split the pick area in 3
        Rpix = np.c_[1 / 3:1:3j] * pixels  # radii of concentric circles

        xd, yd = xt - event.x, yt - event.y
        prox = xd ** 2 + yd ** 2  # distance of click from points in pixels
        c = prox - Rpix ** 2  # 2D array. columns rep. points in line; rows rep. distance
        picked = np.any(c < 0)

        if picked:
            # part index (which part of the errorbar container) and point index (which point along the line)
            # NOTE: the indeces here are wrt the re-ordered container. We therefore
            partix, pointix = np.unravel_index(abs(c).argmin(), c.shape)
            props['part'] = ErrorbarPicker.parts[partix]
            props['partxy'] = 'yx'[np.argmin(np.abs((xd, yd)))]

        return picked, props


# ****************************************************************************************************
class DynamicLegend(ConnectionMixin):  # TODO: move to separate script....
    # TODO: subclass Legend??

    # FIXME: redesign so you can handel other artists

    """
    Enables toggling marker / bar / cap visibility by selecting on the legend.
    """
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    _default_legend = dict(fancybox=True,
                           framealpha=0.5,
                           handler_map={ErrorbarContainer:
                                            ReorderedErrorbarHandler(
                                                numpoints=1)})
    label_map = {ErrorbarContainer: 'errorbar{}',
                 Line2D: 'line{}'}

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __init__(self, ax, plots, legendkw={}):
        """enable legend picking"""

        # initialize auto-connect
        ConnectionMixin.__init__(self, ax.figure)

        # Auto-generate labels
        # NOTE: This needs to be done to enable legend picking. if the artists
        # are unlabeled, no legend will be created and we therefor cannot pick them!
        i = 0
        for plot in plots:
            if not plot.get_label():  # NOTE: this may be _line0 etc...
                lbl = self.label_map[type(plot)].format(i)
                plot.set_label(lbl)
                i += 1

        # update default legend props with user specified props
        lkw = self._default_legend
        lkw.update(legendkw)

        # create the legend

        # print('PING!!'*10 )
        # embed()

        # self.legend = ax.legend( plots, labels, **lkw )
        self.legend = ax.legend(**lkw)

        if self.legend:  # if no labels --> no legend, and we are done!
            # create mapping between the picked legend artists (markers), and the
            # original (axes) artists
            self.to_orig = {}
            self.to_leg = {}
            self.to_handle = {}

            # enable legend picking by setting the picker method
            for handel, origart in zip(self.legend.legendHandles,
                                       plots):  # get_lines()
                # FIXME: redesign so you don't have to turn everything into NamedErrorbarContainer
                if isinstance(origart, Line2D):
                    origart = [origart]

                else:
                    self.to_orig[handel.markers] = NamedErrorbarContainer(
                        origart)
                    self.to_leg[handel.markers] = handel
                    self.to_handle[origart[0]] = handel

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @mpl_connect('pick_event')
    def on_pick(self, event):
        """Pick event handler."""
        if event.artist in self.to_orig:
            self.toggle_vis(event)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # @unhookPyQt
    def toggle_vis(self, event):
        """
        on the pick event, find the orig line corresponding to the
        legend proxy line, and toggle the visibility.
        """

        def get_part(mapping, event):
            part = getattr(mapping[event.artist], event.part)

            if event.part in ('stems', 'caps'):
                artists = getattr(part, event.partxy)
            else:
                artists = part

            yield from flatten([artists])

        for art in get_part(self.to_orig, event):
            vis = not art.get_visible()
            art.set_visible(vis)

        for art in get_part(self.to_leg, event):
            vis = not art.get_visible()
            art.set_alpha(1.0 if vis else 0.2)

        # FIXME UnboundLocalError: local variable 'vis' referenced before assignment
        # TODO: BLIT
        self.canvas.draw()


def get_xy(container, has_yerr):
    """
    Return the constituents of an ErrorbarContainer object as dictionary keyed
    on 'x' and 'y'.
    """
    markers, caps, stems = container
    order = reversed if has_yerr else lambda x: x

    def getter(item, g):
        return coll.defaultdict(list,
                                itt.zip_longest(order('xy'),
                                                mit.grouper(g, order(item)),
                                                fillvalue=())
                                )

    _stems = getter(stems, 1)
    _caps = getter(caps, 2)
    return markers, _stems, _caps


# ****************************************************************************************************
class NamedErrorbarContainer(ErrorbarContainer,
                             coll.namedtuple('_NamedErrorbarContainer',
                                             'markers stems caps')):
    """
    Wrapped ErrorbarContainer that identifies constituents explicitly and
    allows dictionary-like item access.
    """

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __new__(cls, container, has_xerr=False, has_yerr=False, **kws):
        if isinstance(container, list):
            if len(container) != 3:
                container = cls._partition(container)

        markers, caps, stems = get_xy(container, has_yerr)

        stems = coll.namedtuple('Stems', 'x y')(**stems)
        caps = coll.namedtuple('Caps', 'x y')(**caps)

        return super().__new__(cls, (markers, caps, stems))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @staticmethod
    def _partition(artists):
        """ """

        stems, lines = mit.partition(is_line, artists)
        markers, *caps = lines
        return markers, caps, tuple(stems)

    def partition(self):
        return self._partition(self.get_children())

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __init__(self, container, has_xerr=False, has_yerr=False, **kws):
        """ """
        if isinstance(container, ErrorbarContainer):
            ErrorbarContainer.__init__(self, container,
                                       container.has_xerr, container.has_yerr,
                                       label=container.get_label())
        else:
            ErrorbarContainer.__init__(self, container, has_xerr, has_yerr,
                                       **kws)


# ****************************************************************************************************
class DraggableErrorbarContainer(NamedErrorbarContainer):
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    annotation_format = '[%+g]'  # '[%+3.2f]'

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __init__(self, container, has_xerr=False, has_yerr=False, **kws):
        """ """
        self.offset = kws.pop('offset', 0)
        self.annotated = kws.pop('annotate', True)
        # haunted = kws.pop('haunted', True)

        NamedErrorbarContainer.__init__(self, container, has_xerr, has_yerr,
                                        **kws)

        # by default the container is self-linked
        self.linked = [self]

        # Save copy of original transform
        markers = self[0]
        self._original_transform = markers.get_transform()

        # make the lines pickable
        if not markers.get_picker():
            markers.set_picker(5)

        # Initialize offset texts
        ax = markers.axes
        self.text_trans = btf(ax.transAxes, ax.transData)
        ytxt = markers.get_ydata().mean()


        self.annotation = ax.text(1.005, ytxt, '')
        # transform=self.text_trans )

        # shift to the given offset (default 0)
        self.shift(self.offset)

        # if haunted:
        # self.haunt()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __repr__(self):
        return "<%s object of %d artists>" % (
        self.__class__.__name__, len(self))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def set_transform(self, trans):
        for art in self.get_children():  # flatten list of containers into list of individual artists
            art.set_transform(trans)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def set_visible(self, vis):
        for art in self.get_children():
            art.set_visible(vis)

        # NOTE: can avoid this if statement by subclassing...
        if self.annotated:
            self.annotation.set_visible(vis)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def haunt(self):
        """ """
        # embed()
        ghost_artists = self._partition(map(copy, self.get_children()))
        container = ErrorbarContainer(ghost_artists,
                                      self.has_xerr, self.has_yerr,
                                      label=self._label)
        self.ghost = DraggableErrorbarContainer(container, offset=self.offset,
                                                annotate=self.annotated)

        ax = self[0].axes
        for g in self.ghost.get_children():
            g.set_alpha(0.2)
            g.set_visible(False)
            ax.add_artist(g)

        # NOTE: can avoid this if statement by subclassing...
        if self.annotated:
            self.ghost.annotation.set_alpha(0.2)

            # .add_container( self.ghost )

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def shift(self, offset):
        """Shift the data by offset by setting the transform """
        # add the offset to the y coordinate
        offset_trans = Affine2D().translate(0, offset)
        trans = offset_trans + self._original_transform
        self.set_transform(trans)

        # NOTE: can avoid this if statement by subclassing...
        if self.annotated:
            txt = self.annotation
            txt.set_transform(offset_trans + self.text_trans)
            txt.set_text(self.annotation_format % offset)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def draw(self, renderer, *args, **kwargs):

        for art in self.get_children():
            art.draw(renderer)

        # NOTE: can avoid this if statement by subclassing...
        if self.annotated:
            self.annotation.draw(renderer)


# ****************************************************************************************************
class DraggableErrorbar(ConnectionMixin):  # TODO: rename
    # TODO:      inherit from DragMachinery()???
    # TODO:      Use offsetbox????
    # TODO:      BLITTING !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # TODO:      one-way links??
    # FIXME:     links -- display offset text??
    # FIXME:     right click shifts - not desired!
    # FIXME:     ghosts for links not plotting ???
    """
    Class which takes ErrorbarCollection objects and makes them draggable on the figure canvas.
    """

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __init__(self, plots, offsets=None, linked=[],
                 auto_legend=True, **legendkw):
        # TODO: update docstring
        # (optionally nested) Line2D objects or ErrorbarContainers
        """
        Parameters
        ----------
        plots : list of plot objects
            List of ErrorbarContainers
        offsets : sequence of floats
            Initial offsets to use for Line2D objects
        linked : sequence of linked artists (or sequence of sequences of linked artists)
            all artists in the linked sequence will move together with all the
            others in the sequence when one of them is dragged.  The offset text
            will only be displayed for the first artist in the sequence.
        auto_legend : boolean
            whether to create a legend or not

        Keywords
        --------
        legendkw : dict, optional
            Keywords passed directly to axes.legend
        """
        self.selection = None
        self.select_point = None

        if offsets is None:
            offsets = np.zeros(len(plots))
        # self.offsets            = {}                    #map pickable artist to offset value for that artist
        self.tmp_offset = 0  # in case of pick without motion

        # self.plots              = plots #= flatten(plots)
        self.ax = ax = plots[0][0].axes
        self.figure = ax.figure

        # initialize mapping
        self.draggables = {}  # TODO: make this a indexable Ordered dict

        # initialize auto-connect
        ConnectionMixin.__init__(self, ax.figure)

        # esure linked argument is a nested list
        if len(linked) and isinstance(linked[0], ErrorbarContainer):  # HACK
            linked = [
                linked]  # np.atleast_2d does not work because it unpacks the ErrorbarContainer
        else:
            linked = list(linked)

        _friends = [f for _, *friends in linked for f in friends]

        # build the draggable objects
        for plot, offset in zip(plots, offsets):
            # if isinstance(plot, ErrorbarContainer):    #NOTE: will need to check this when generalizing
            annotate = not plot in _friends  # only annotate the "first" linked artist
            draggable = DraggableErrorbarContainer(plot,
                                                   offset=offset,
                                                   annotate=annotate)
            markers, _, _ = draggable

            # map Line2D to DraggableErrorbarContainer. The picker returns the markers.
            self.draggables[markers] = draggable

            # create ghost artists
            draggable.haunt()

        self.lines = list(self.draggables.keys())

        # establish links
        for links in linked:
            link_set = [self.draggables[m] for m, _, _ in
                        links]  # set of linked DraggableErrorbarContainers
            for i, draggable in enumerate(link_set):
                draggable.linked = link_set  # each linked artist carries a copy of all those artists linked to it

        # enable legend picking
        self.legend = None
        if legendkw or auto_legend:
            self.legend = DynamicLegend(ax, plots, legendkw)
            self.legend.connect()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def reset(self):
        """reset the plot positions to zero offset"""
        # print('resetting!')
        for draggable in self.draggables.values():
            draggable.shift(0)  # NOTE: should this be the original offsets??

        self.canvas.draw()

        # print( repr(self.selection ))
        # print()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @mpl_connect('button_press_event')
    def on_click(self, event):

        # print( 'on_click', repr(self.selection ))

        """reset plot on middle mouse"""
        if event.button == 2:
            self.reset()
        else:
            return

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @mpl_connect('pick_event')
    def on_pick(self, event):
        """Pick event handler.  On """

        # print('picked', repr(self.selection))

        if event.artist in self.draggables:
            ax = self.ax
            xy = event.mouseevent.xdata, event.mouseevent.ydata
            self.selection = event.artist

            # connect motion_notify_event for dragging the selected artist
            self.add_connection('motion_notify_event', self.on_motion)
            # save the background for blitting
            self.background = self.canvas.copy_from_bbox(self.figure.bbox)

            draggable = self.draggables[self.selection]
            self.select_point = np.subtract(xy,
                                            draggable.offset)  # current_offset = draggable.offset

            # make the ghost artists visible
            for link in draggable.linked:
                # set the ghost vis same as the original vis for linked
                for l, g in zip(link.get_children(),
                                link.ghost.get_children()):
                    g.set_visible(l.get_visible())
                    # print( 'visible!', g, g.get_visible() )

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def on_motion(self, event):

        if event.inaxes != self.ax:
            return

        if self.selection:
            self.tmp_offset = tmp_offset = event.ydata - self.select_point[
                1]  # from original data

            self.canvas.restore_region(self.background)

            draggable = self.draggables[self.selection]
            # print('...')
            for link in draggable.linked:
                link.ghost.shift(tmp_offset)

                # print( [(ch.get_visible(), ch.get_alpha()) for ch in link.ghost.get_children()] )

                link.ghost.draw(self.canvas.renderer)
            # print('...')

            self.canvas.blit(self.figure.bbox)  # self.ax.bbox??

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @mpl_connect('button_release_event')
    def on_release(self, event):

        # print( 'release top', repr(self.selection ))

        # if event.button != 1:
        # return

        if self.selection:
            # Remove dragging method for selected artist
            self.remove_connection('motion_notify_event')

            draggable = self.draggables[self.selection]
            for linked in draggable.linked:
                linked.shift(self.tmp_offset)
                linked.offset = self.tmp_offset
                linked.ghost.set_visible(False)

            # do_legend()
            self.canvas.draw()

        self.selection = None
        # print( 'release', repr(self.selection ))
        # print()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def set_annotation(self, TF):
        """enable / disable offset text annotation"""
        for art in self.draggables.values():
            art.annotation.set_visible(TF)
            art.annotated = TF

            art.ghost.annotation.set_visible(TF)
            art.ghost.annotated = TF


#######################################################################################################################
# Alias
# ****************************************************************************************************
class DraggableErrorbars(DraggableErrorbar):
    pass

#######################################################################################################################
