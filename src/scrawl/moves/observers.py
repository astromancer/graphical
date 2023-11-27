
# std
import sys
import time
import itertools as itt
from collections import OrderedDict

# third-party
from better_exceptions import format_exception

# local
from recipes import pprint
from recipes.logging import LoggingMixin

# relative
from .utils import art_summary, filter_non_artist


class Observers(LoggingMixin):
    """
    Container class for observer functions.
    """

    # TODO: Integrate with the activities already handled by CallbackRegister

    def __init__(self, rate_limit=-1):
        self.counter = itt.count()
        self.funcs = OrderedDict()
        self.active = {}

        self.timeout = float(1. / rate_limit)
        self._previous_call_time = -1

    def __repr__(self):
        return '\n'.join((self.__class__.__name__,
                          '\n'.join(map(self._repr_observer, self.funcs.keys()))))

    def _repr_observer(self, id_):
        func, args, kws = self.funcs[id_]
        active = self.active[func]
        observers = pprint.caller(func, args=args, kws=kws).replace('\n', '\n    ')
        return f'{id_}{" *"[active]}: {observers}'

    def add(self, func, *args, **kws):
        """
        Add an observer function.

        When the artist is moved / picked, *func* will be called with the new
        coordinate position as arguments. *func* should return any artists
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
            raise TypeError('Parameter `func` should be a callable.')

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
        Reactivate a non-active observer. This method is useful for toggling
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
                'Function {!r} is not an observer! Use `add(fun, *args, **kws)'
                'to make it an observer', fun
            )

    def __call__(self, *args, **kws):
        """
        Call all active observers.

        Parameters
        ----------
        x, y

        Returns
        -------
        Artists that need to be drawn
        """
        now = time.time()
        if (elapsed := now - self._previous_call_time) < self.timeout:
            self.logger.debug('Observer timed out for {}s. Time elapsed since '
                              'previous call: {:.3f}s', self.timeout, elapsed)
            return

        # Artists that need to be drawn (from observer functions)
        artists = []
        self._previous_call_time = now
        for _, (func, static_args, static_kws) in self.funcs.items():
            if not self.active[func]:
                continue

            try:
                self.logger.debug('Calling observer function: {!r}.', func.__name__)
                art = func(*args, *static_args, *static_kws, **kws)
                art = list(filter_non_artist(art))
                artists.extend(art)
                if art:
                    self.logger.opt(lazy=True).debug(
                        'The following artists have been changed by observer function '
                        '{0[0]!r}:\n{0[1]}', lambda: (func.__name__, art_summary(art))
                    )
                else:
                    self.logger.debug('No artists returned by observer {}.',
                                      func.__name__)

            except Exception:
                self.logger.exception(
                    '\n'.join(('Observer error!',
                               *format_exception(*sys.exc_info())))
                )

        return artists
