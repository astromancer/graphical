"""
Manage canvas callbacks.
"""

from recipes.oo.meta import tagger
from recipes.logging import LoggingMixin


__all__ = ['mpl_connect', 'CallbackManager']

#
TagManager, mpl_connect = tagger.factory(collection='_callbacks')


class CallbackManager(TagManager, LoggingMixin):
    """
    Mixin for connecting callbacks of decorated methods to the figure canvas.
    """

    def __init__(self, canvas=None):
        """ """
        TagManager.__init__(self)
        # connection ids
        self.callbacks = {}
        self._canvas = canvas  # TODO: check that it's a canvas

    @property
    def canvas(self):
        return self._canvas

    def add_callback(self, id_, method):
        if id_ in self.callbacks:
            raise ValueError(
                f'A callback with ID {id_} already exists! The following '
                f' callbacks are currently registered: {self.callbacks}\n'
                f'Please choose a unique name for the {method} callback.'
            )

        if isinstance(id_, tuple):
            name = id_[0]
        else:
            name = id_
            id_ = (id_, )
        
        self.logger.debug('Adding callback {!r}: {}', id_, method)
        self.callbacks[id_] = self.canvas.mpl_connect(name, method)

    def remove_callback(self, *id_):
        self.logger.debug('Removing callback {!r}', id_)
        self.canvas.mpl_disconnect(self.callbacks.pop(id_))

    def connect(self):
        """
        Connect the flagged methods to the canvas as callback functions.
        """
        for method, id_ in self._callbacks.items():
            self.add_callback(id_, method)

    def disconnect(self):
        """
        Disconnect all callbacks from figure canvas.
        """
        for cid in self.callbacks.values():
            self.canvas.mpl_disconnect(cid)
        self.logger.debug('Disconnected from figure {!s}', self.figure.canvas)
