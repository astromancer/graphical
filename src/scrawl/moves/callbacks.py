"""
Manage canvas callbacks.
"""

# third-party
from matplotlib.cbook import CallbackRegistry

# local
import recipes.pprint as pp
from recipes.oo.meta import TagManagerBase
from recipes.logging import LoggingMixin


__all__ = ['mpl_connect', 'CallbackManager']

# #
# TagManager, mpl_connect = tagger.factory(tag='_is_callback',
#                                          collection='_callbacks')

# class CallbackManager(LoggingMixin,
#                       TagManager(tag='_is_callback',
#                                  collection='_callbacks'))


class CallbackManager(LoggingMixin, TagManagerBase,
                      tag='_is_callback', collection='_callbacks'):
    """
    Mixin for managing canvas / artist callbacks as decorated methods.
    """

    def __init__(self, object=None, connect=False):
        """ """
        # `TagManagerMeta` collects the functions tagged by `@mpl_connect(*args)`
        # and sets the `_callbacks` attribute which maps the (bound) methods to
        # the argument tuple passed to the decorator.
        # TagManager.__init__()

        # connection id proxies
        self.cid_proxies = {}
        if isinstance(object, CallbackRegistry):
            callbacks = object
        elif not isinstance((callbacks := getattr(object, 'callbacks')),
                            CallbackRegistry):
            raise TypeError(f'Cannot not instantiate {type(self)!r} from '
                            f'object of type {type(object)}.')

        #
        self.callbacks = callbacks

        # connect all decorated methods
        if connect:
            self.connect()

    def add_callback(self, signal, method, identifier=None):
        identifier = identifier or type(self)
        if identifier in self.cid_proxies:
            raise ValueError(
                f'A callback with ID {identifier} already exists! The '
                'following callbacks are currently registered:'
                f' {pp.mapping(self.callbacks)}\n'
                f'Please choose a unique (hashable) identifier for the {method}'
                ' callback.'
            )

        self.logger.debug('Adding {} callback with id {!r}', method, identifier)
        self.cid_proxies[identifier] = self.callbacks.connect(signal, method)

    def remove_callback(self, *identifier):
        self.logger.debug('Removing callback {!r}', identifier)
        self.callbacks.disconnect(self.cid_proxies.get(identifier, identifier))

    def connect(self):
        """
        Connect the decorated methods to the canvas callback registry.
        """
        for method, (signal, *identifier) in self._callbacks.items():
            assert callable(method)
            # print(signal, method, *identifier)
            self.add_callback(signal, method, *identifier)

    def disconnect(self):
        """
        Disconnect all callbacks for class.
        """
        [*map(self.callbacks.disconnect, self.cid_proxies.values())]
        self.cid_proxies = {}
        self.logger.debug('Disconnected from figure {!s}', self.canvas)


# decorator
mpl_connect = CallbackManager.tag
