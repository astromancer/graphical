"""
Extensible slider widgets.
"""

# std
import operator as op

# third-party
import numpy as np
import more_itertools as mit
from cycler import cycler
from matplotlib.lines import Line2D

# relative
from .moves.machinery import MotionManager


# from decor import expose

# def picker(artist, event):
#     # print(vars(event))
#     mouse_position = (event.xdata, event.ydata)
#     if None in mouse_position:
#         return False, {}

#     ax = artist.axes
#     _data2ax_trans = ax.transData + ax.transAxes.inverted()
#     mouse_ax_pos = _data2ax_trans.transform(mouse_position)

#     centre = artist.get_centre()

#     prox = np.linalg.norm(mouse_ax_pos - centre)
#     hit = prox < 0.5
#     print('mouse_position, mouse_ax_pos,  centre')
#     print(mouse_position, mouse_ax_pos, centre)
#     print('prox', prox)
#     print('hit', hit)
#     return hit, {}


class AxesSliders(MotionManager):
    """
    Sliders that move along an axes and hold min/max value state.

    Basic interactions include:
        movement - click & drag with left mouse
        reset - middle mouse

    Connect listeners to the sliders with eg.:
    >>> sliders.upper.on_move.add(func)
    """

    # TODO: Dragging sliders too slow for large images: option for update on
    #  release instead of update on drag!!

    # indices of the movable objects that can be set by assigning to the
    # `positions` attribute
    _use_positions = slice(None)

    # marker_size = 10
    # valfmt = '%1.2f',

    def __init__(self, ax, positions, slide_axis='x',
                 dragging=True,  # FIXME: clarify what this does. is it useful??
                 trapped=False, annotate=False, haunted=False,
                 use_blit=True,
                 extra_markers=(),
                 **props):
        """

        Parameters
        ----------
        ax
        positions
        slide_axis
        dragging
        trapped
        annotate
        haunted
        use_blit
        extra_markers:
            additional markers per line for aesthetic
        props:
            dict of properties for the lines. each property is a 2-tuple
        """

        # initial values
        self.slide_axis = slide_axis.lower()
        self._ifree = i = int(slide_axis == 'y')
        # `self._ifree`: 0 for x-axis slider, 1 for y-axis
        self._ilock = int(not bool(i))
        self._locked = 'yx'[i]
        self._order = o = slice(None, None, [1, -1][i])
        # minimal distance
        self.min_span = None
        self._changed_axes_lim = False

        # check positions
        # assert np.size(positions) == 2, 'Positions should have size 2'
        self._original_position = positions  # np.sort(

        # create the movement UI
        MotionManager.__init__(self, use_blit=use_blit)

        prop_cycle = cycler(**props) if props else [{}] * len(positions)
        # get transform (like axhline / axvline)
        transform = getattr(ax, f'get_{self.slide_axis}axis_transform')()

        # add the sliding artists (Line2D)
        sliders = []
        nem = len(extra_markers)
        clip_on = use_blit or trapped
        for pos, props in zip(self._original_position, prop_cycle):
            # create sliders Line2D
            # add middle point to sliders since picking usually happens near the
            # middle!
            x, y = [[pos], [0, 0.5, 1]][o]
            line = Line2D(x, y, transform=transform, clip_on=clip_on, **props)
            ax.add_artist(line)
            sliders.append(line)
            # dart =
            self.add_artist(line, (0, 0), annotate, haunted,
                            trapped=trapped)

            # add some markers for aesthetic           '>s<'
            for x, m in zip(np.linspace(0, 1, nem, 1), extra_markers):
                x, y = [[pos], [x]][o]
                tied = Line2D(x, y, color=props['color'], marker=m,
                              transform=transform, clip_on=clip_on)
                ax.add_line(tied)  # move to link function??
                self[line].tie(tied)

        # set upper / lower attributes for convenience
        self.lower = self.movable[sliders[0]]
        self.upper = self.movable[sliders[1]]
        self.lock(self._locked)

        # constrain movement
        # x0, x1 = self._original_position[:2]
        # self.lower.ymax = x1 - self.delta_min
        # self.upper.ymin = x0 + self.delta_min

        # if sliders are not `trapped` within axes, allow dragging beyong
        # axes limits and relim axes when that happens
        if not trapped:
            self.upper.on_move.add(self.set_axes_lim, 1)
            self.lower.on_move.add(self.set_axes_lim, 0)

        self.upper.on_move.add(self.set_lower_ymax)
        self.lower.on_move.add(self.set_upper_ymin)

        self.dragging = dragging
        # FIXME: get this working to you update on release not motion for speed
        # create sliders & add to axis

    def setup_axes(self, ax):
        """ """
        ax.set_navigate(False)  # turn off nav for this axis

    @property
    def positions(self):
        return self._original_position[self._use_positions] + \
            self.offsets[self._use_positions, self._ifree]

    def set_positions(self, values, draw_on=True):
        # not a property since it returns a list of artists to draw
        # assert len(values) == len(self.artists)
        draw_list = []
        for drg, v in zip(self.movable.values(), values):
            new = (v, drg.position[self._ilock])[self._order]
            art = self.update(drg, new, False)
            draw_list.extend(art)
            # avoid drawing here with `draw_on=False`

        if draw_on:
            self.draw(draw_list)
        return draw_list

    def set_axes_lim(self, x, y, upper):
        relate = op.gt if upper else op.lt
        expanding = relate(self.delta[self._ifree], 0)
        lim = getattr(self.ax, f'get_{self.slide_axis}lim')()
        beyond = relate(self.positions[upper], lim[upper])
        axis = getattr(self.ax, f'{self.slide_axis}axis')
        if expanding and beyond:
            limits = [None, None]
            limits[upper] = (x, y)[self._ifree]
            # set limits
            getattr(self.ax, f'set_{self.slide_axis}lim')(limits)
            self._changed_axes_lim = True
        elif self._changed_axes_lim:
            limits = [None, None]
            relate = (min, max)[upper]
            limits[upper] = relate(self._original_axes_limits[self._ifree, upper],
                                   (x, y)[self._ifree])

            # set limits
            getattr(self.ax, f'set_{self.slide_axis}lim')(limits)

        # note: only actually have to draw these if the axis limits have
        #  changed, but need to always return these so blit_setup gets the
        #  axis when initializing and therefore knows to redraw all the ticks
        #  etc
        return axis

    def set_upper_ymin(self, x, y):
        # pos = self.get_positions()
        if self.min_span is not None:
            self.upper.ymin = y + self.min_span
            self.logger.trace('upper ymin: {:.2f}, {:.2f}', self.upper.ymin, y)

    def set_lower_ymax(self, x, y):
        # pos = self.get_positions()
        if self.min_span is not None:
            self.lower.ymax = y - self.min_span
            self.logger.trace('lower ymax: {:.2f}, {:.2f}', self.lower.ymax, y)

    # def reset(self):
    #     return super().reset()

    def link(self, *artists):
        """
        Link artists for co-draw.
        """
        for mv in self.movable.values():
            mv.link(*artists)

    def unlink(self, *artists):
        """
        Link artists for co-draw.
        """
        for mv in self.movable.values():
            mv.unlink(*artists)


class RangeSliders(AxesSliders):  # MinMaxMeanSliders # RangeSliders
    """
    A set of 3 sliders for setting min/max/mean value.  Middle slider is
    tied to both the upper and lower slider and will move both those when
    changed.  This allows for easily altering the upper/lower range values
    simultaneously.  The middle slider will also always be at the mean
    position of the upper/lower sliders.
    """
    _use_positions = [0, 1]

    def __init__(self, ax, positions, slide_axis='x', dragging=True,
                 trapped=False, annotate=False, haunted=False, use_blit=True,
                 extra_markers=(), **props):
        """

        Parameters
        ----------
        ax
        x0
        x1
        slide_axis
        dragging
        trapped
        annotate
        haunted
        use_blit
        kwargs
        """

        pos3 = np.hstack([positions, np.mean(positions)])
        AxesSliders.__init__(self, ax, pos3, slide_axis, dragging, trapped,
                             annotate, haunted, use_blit, extra_markers,
                             **props)
        #
        self.centre = self.middle = self.movable[2]
        self.centre.lock(self._locked)
        # self.lower.on_picked.add(lambda x, y: self.centre.set_animate(True))

        # add method to move central slider when either other is moved
        self._cid_lwr_mv_ctr = self.lower.on_move.add(self.set_centre)
        self._cid_upr_mv_ctr = self.upper.on_move.add(self.set_centre)

        # make sure centre slide is between upper and lower
        self.lower.on_release.add(self.set_centre_max)
        self.upper.on_release.add(self.set_centre_min)

        # disconnect links to centre to avoid infinite recursion
        self.centre.on_picked.add(self.deactivate_centre_control)
        # re-link to the centre slider
        self.centre.on_release.add(self.activate_centre_control)
        # make sure we draw all the tied artists on centre move
        # NOTE: not sure why this is necessary since linking should draw all
        # tied artists, but trailing slider does not draw...
        self.centre.on_move.add(
            lambda x, y: self.lower.draw_list + self.upper.draw_list)

    def set_centre(self, x, y):
        """
        set the position of the central slider when either of the other sliders
        is moved
        """
        y = self.positions.mean()
        return self.centre.update(x, y)

    def set_centre_min(self, x, y):
        """set minimum position of the central slider"""
        self.centre.ymin = self.lower.ymin + self.min_span / 2
        self.logger.debug('centre min: {:.2f}, {:.2f}', self.centre.ymin, y)

    def set_centre_max(self, x, y):
        """set maximum position of the central slider"""
        self.centre.ymax = self.upper.ymax + self.min_span / 2
        self.logger.debug('centre ymax: {:.2f}, {:.2f}', self.centre.ymax, y)

    def _animate(self, b):
        for drg in self.movable.values():
            drg.set_animated(b)

    def animate(self, x, y):
        self._animate(True)

    def deanimate(self, x, y):
        self._animate(False)

    def centre_moves(self, x, y):

        pos = self.centre.position  # previous position
        art1 = self.centre.update(x, y)
        shift = self.centre.position - pos
        up = shift[self._ifree] > 0  # True if movement is upwards

        # upper if moving up else lower. i.e. the one that may get clipped
        lead = self.movable[int(up)]
        trail = self.movable[int(not up)]

        art2 = lead.update(*(lead.position + shift))
        art3 = trail.update(*(trail.position + shift))

        return art1 + art2 + art3

    def activate_centre_control(self, x=None, y=None):
        # print('act')
        self.lower.on_move.activate(self._cid_lwr_mv_ctr)
        self.upper.on_move.activate(self._cid_upr_mv_ctr)

        self.centre.untie(self.lower, self.upper)

    def deactivate_centre_control(self, x=None, y=None):
        # print('deact')
        self.lower.on_move.deactivate(self._cid_lwr_mv_ctr)
        self.upper.on_move.deactivate(self._cid_upr_mv_ctr)

        self.centre.tie(self.lower, self.upper)
