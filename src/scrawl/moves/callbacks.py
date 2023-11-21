"""
Manage canvas callbacks.
"""

# std
import time
from warnings import warn

# third-party
from matplotlib.cbook import CallbackRegistry

# local
import recipes.pprint as pp
from recipes.logging import LoggingMixin
from recipes.oo.meta import MethodTagger, MethodTaggerFactory, TagManagerBase


__all__ = ['mpl_connect', 'CallbackManager']


# ---------------------------------------------------------------------------- #
null = object()

# ---------------------------------------------------------------------------- #


class CallbackFactory(MethodTaggerFactory):
    def __call__(self, *info, **kws):
        return CallbackDecorator(self.tag, info, **kws)


class CallbackDecorator(MethodTagger):

    def __init__(self, tag, info, timeout=None, rate_limit=None):

        if timeout and rate_limit:
            raise ValueError('Either `timeout` or `rate_limit` can be give, but'
                             ' not both.')

        if rate_limit:
            timeout = 1. / rate_limit

        #
        super().__init__(tag, info)

        self.timeout = float(timeout or 0)
        self.previous_call_time = -1

    def __wrapper__(self, func, *args, **kws):
        now = time.time()
        if (elapsed := now - self.previous_call_time) < self.timeout:
            self.logger.debug('Function {.__name__!r} is rate limited at {:.3f}.'
                              ' Hz. Time elapsed since previous call: {:.3f} s.',
                              func, 1. / self.timeout, elapsed)
            return

        self.previous_call_time = now
        return func(*args, **kws)


class CallbackManager(LoggingMixin, TagManagerBase,
                      tag='_signal', collection='_callbacks'):
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

        if object is None:
            callbacks = None
        elif isinstance(object, CallbackRegistry):
            callbacks = object
        elif not isinstance((callbacks := getattr(object, 'callbacks')),
                            CallbackRegistry):
            raise TypeError(f'Cannot not instantiate {type(self)!r} from '
                            f'object of type {type(object)}.')

        #
        self.callbacks = callbacks

        # connect all decorated methods
        if connect:
            if not callbacks:
                warn('No callback registry available. Ignoring `connect=True` '
                     'request.')
            self.connect()

    def _check(self):
        if self.callbacks is None:
            raise ValueError('No callback registry available!')

    def add_callback(self, signal, method, identifier=null):

        self._check()

        identifier = signal if identifier is null else (signal, identifier)

        if identifier in self.cid_proxies:
            raise ValueError(
                f'A callback with ID {identifier!r} already exists! The '
                'following callbacks are currently registered:\n'
                f'{self.cid_proxies = }\n'
                f'{pp.pformat(self.callbacks.callbacks)}\n'
                f'Please choose a unique (hashable) identifier for the {method}'
                ' callback.'
            )

        self.logger.debug('Adding {} callback with id {!r}.',
                          pp.describe(method), identifier)
        self.cid_proxies[identifier] = self.callbacks.connect(signal, method)

    def remove_callback(self, signal, *identifier):
        self._check()
        identifier = (signal, *identifier) if identifier else signal
        if cid := self.cid_proxies.pop(identifier, None):
            self.logger.debug('Removing callback with id {!r} for {}.',
                              identifier, self)
            self.callbacks.disconnect(cid)
        else:
            self.logger.debug('No method with id {!r} in {}.',
                              identifier, cid, self)

    def connect(self):
        """
        Connect the decorated methods to the canvas callback registry.
        """
        self._check()
        self.logger.debug('Connecting all callbacks for {}.', self)
        for method, (signal, *identifier) in self._callbacks.items():
            assert callable(method)
            # print(signal, method, *identifier)
            self.add_callback(signal, method, *identifier)

    def disconnect(self):
        """
        Disconnect all callbacks for class.
        """
        # connect the methods decorated with `@mpl_connect`
        self._check()
        [*map(self.callbacks.disconnect, self.cid_proxies.values())]
        self.cid_proxies = {}
        self.logger.debug('Disconnected from figure {!s}.', self.canvas)


# decorator
mpl_connect = CallbackManager.tag = CallbackFactory(CallbackManager.tag.tag)
