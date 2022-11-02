"""
Movable artists.
"""

from .errorbars import *
from .callbacks import *


class TrackAxesUnderMouse(CallbackManager):
    _axes_under_mouse = None

    @mpl_connect('axes_enter_event')
    def _on_enter(self, event):
        self.logger.trace('Mouse over: {}', event.inaxes)
        self._axes_under_mouse = event.inaxes

    @mpl_connect('axes_leave_event')
    def _on_leave(self, event):
        self.logger.trace('Mouse left axes: {}', event.inaxes)
        self._axes_under_mouse = None
