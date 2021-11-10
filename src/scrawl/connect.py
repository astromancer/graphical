from recipes.oo.meta import flagger
from recipes.logging import LoggingMixin

#
ConnectionManager, mpl_connect = flagger.factory(collection='_connections')


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
        self.logger.debug('Adding connection {!r}: {}', name, method)
        self.connections[name] = self.canvas.mpl_connect(name, method)

    def remove_connection(self, name):
        self.logger.debug('Removing connection {!r}', name)
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
        self.logger.debug('Disconnected from figure {!s}', self.figure.canvas)
