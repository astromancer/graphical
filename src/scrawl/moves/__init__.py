"""
Movable artists.
"""

from .errorbars import *
from .callbacks import *
from .machinery import CanvasBlitHelper, Observers


class TrackAxesUnderMouse(CallbackManager):
    _axes_under_mouse = None

    @mpl_connect('axes_enter_event')
    def _on_enter(self, event):
        self.logger.trace('Mouse over: {}.', event.inaxes)
        self._axes_under_mouse = event.inaxes

    @mpl_connect('axes_leave_event')
    def _on_leave(self, event):
        self.logger.trace('Mouse left axes: {}.', event.inaxes)
        self._axes_under_mouse = None


class ScrollAction(CanvasBlitHelper):

    def __init__(self, artists=(), connect=False, use_blit=True, rate_limit=4):
        super().__init__(artists, connect, use_blit)
        # observer container for scroll callbacks
        self.on_scroll = Observers(rate_limit)

    @mpl_connect('scroll_event', rate_limit=4)
    def _on_scroll(self, event):

        # run callbacks
        if art := self.on_scroll(event):
            # draw the artists that were changed by scroll action
            self.draw(art)
