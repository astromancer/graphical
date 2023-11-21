
# third-party
import matplotlib.colorbar as cbar
from matplotlib import cm

# local
from recipes.logging import LoggingMixin

# relative
from ..moves import ScrollAction, TrackAxesUnderMouse, mpl_connect


# ---------------------------------------------------------------------------- #
# SCROLL_CMAPS = cmr.get_cmap_list()
# SCROLL_CMAPS_R = [_ for _ in SCROLL_CMAPS if _.endswith('_r')]

cmap_categories = {
    'Perceptually Uniform Sequential':
        {'viridis', 'plasma', 'inferno', 'magma', 'cividis'},
    'Sequential':
        {'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
         'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
         'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn'},
    'Sequential (2)':
        {'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
         'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
         'hot', 'afmhot', 'gist_heat', 'copper'},
    'Diverging':
        {'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
         'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic'},
    'Cyclic':
        {'twilight', 'twilight_shifted', 'hsv'},
    'Qualitative':
        {'Pastel1', 'Pastel2', 'Paired', 'Accent',
         'Dark2', 'Set1', 'Set2', 'Set3',
         'tab10', 'tab20', 'tab20b', 'tab20c'},
    'Miscellaneous':
        {'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
         'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg',
         'gist_rainbow', 'rainbow', 'jet', 'turbo', 'nipy_spectral',
         'gist_ncar'}
}

_remove = cmap_categories['Qualitative'] | {'flag', 'prism'}
_remove |= {f'{r}_r' for r in _remove}
SCROLL_CMAPS = ({*cm._colormaps._cmaps.keys()} - _remove)

# ---------------------------------------------------------------------------- #

class Colorbar(cbar.Colorbar, LoggingMixin):

    def __init__(self, ax, mappable=None, scroll=True, **kws):
        super().__init__(ax, mappable, **kws)

        self.scroll = None
        if scroll:
            self.scroll = CMapScroll(self)


Colourbar = Colorbar


class CMapScroll(ScrollAction, TrackAxesUnderMouse, LoggingMixin):
    # TODO: manage timeout through CallbackManager and mpl_connect

    def __init__(self, cbar, cmaps=SCROLL_CMAPS, timeout=0.25, use_blit=True):

        self.colorbar = cbar

        prefixes = {}
        for cmap in cmaps:
            prefix = ''
            if '.' in cmap:
                prefix, cmap = cmap.split('.')
                prefix += '.'
            prefixes[cmap] = prefix

        self.available = sorted(prefixes.keys(), key=str.lower)
        self.prefixes = prefixes
        self._letter_index = next(zip(*self.available))
        self.mappables = {self.colorbar.mappable}

        # canvas = cbar.ax.figure.canvas if cbar.ax else None
        ScrollAction.__init__(self,
                              (self.mappables, self.colorbar.solids),
                              use_blit, timeout)
        # NOTE: Adding `self.colorbar.solids` here so we get the background
        # correct on first draw. A new QuadMesh object will be created whenever
        # the cmap is changed, so we need to remove the original in
        # `_on_first_draw` after saving the initial background
        self.on_scroll.add(self._scroll_cmap)

    def _add_artist(self, artist, mappable=None):

        if mappable is None:
            mappable = isinstance(artist, cm.ScalarMappable)

        if mappable:
            self.mappables.add(artist)

        return super()._add_artist(artist)

    @property
    def mappable(self):
        return self.colorbar.mappable

    # def colormaps[self]:
    #     return remove_prefix(self.mappable.get_cmap().name, 'cmr.')

    def set_cmap(self, cmap):
        for sm in self.mappables:
            sm.set_cmap(cmap)

    def _on_scroll(self, event):
        if self.colorbar is None or event.inaxes is not self.colorbar.ax:
            return

        super()._on_scroll(event)

    def _on_first_draw(self, _):
        super()._on_first_draw(_)
        if self.colorbar.solids in self.artists:
            self.artists.remove(self.colorbar.solids)

    def _scroll_cmap(self, event):
        # called during scroll callback
        current = self.mappable.get_cmap().name
        self.logger.debug('Current cmap: {}.', current)
        if '.' in current:
            _, current = current.split('.')

        avail = self.available
        inc = [-1, +1][event.button == 'up']
        idx = avail.index(current) if current in avail else -1
        new = f'{avail[(idx + inc) % len(avail)]}'
        new = f'{self.prefixes[new]}{new}'

        self.logger.info('Scrolling cmap: {} -> {}.', current, new)
        self.set_cmap(new)

        cb = self.colorbar
        return self.artists, cb.solids, cb.lines, cb.dividers

    @mpl_connect('key_press_event')
    def _on_key(self, event):

        if self._axes_under_mouse is not self.colorbar.ax:
            self.logger.debug(f'Ignoring key press since {self._axes_under_mouse = }')
            return

        if event.key not in self._letter_index:
            self.logger.debug('No cmaps starting with letter {!r}.', event.key)
            return

        current = self.mappable.get_cmap().name
        if '.' in current:
            _, current = current.split('.')

        i = self._letter_index.index(event.key)
        new = self.available[i]
        new = f'{self.prefixes[new]}{new}'
        self.logger.info('Scrolling cmap: {} -> {}.', current, new)
        self.set_cmap(new)

        cb = self.colorbar
        self.draw((self.artists, cb.solids, cb.lines, cb.dividers))
