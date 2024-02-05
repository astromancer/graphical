
"""
Additional matplotlib axes tick formatters.
"""


# std
import warnings
import datetime

# third-party
import numpy as np
from scipy.stats import mode
from matplotlib import ticker
from matplotlib.transforms import IdentityTransform, Transform

# local
from recipes import pprint as ppr

# relative
from .transforms import ReciprocalTransform


# ---------------------------------------------------------------------------- #
SPD = 86_400

# ---------------------------------------------------------------------------- #
# Locators                                                                     #
# ---------------------------------------------------------------------------- #

# TODO: NoOverlappingTicksFormatter.
#  NOTE: Probably not necessary if you choose appropriate locators


def locator_transform_factory(locator, transform):
    """
    Create a tick formatter class which, when called applies the transformation
    given in ``transform`` before invoking the parent formatter's __call__
    method.
    """
    # _locator = expose.args()(locator)

    # **************************************************************************
    class TransLocator(locator.__class__):
        def __call__(self):
            locs = locator()
            locs = transform.transform(locs)  # [locs > transform.thresh]
            # FIXME: ValueError: total size of new array must be unchanged.
            #  when only a single tick on axis

            # if np.ma.is_masked(locs):
            # return locs[~locs.mask]

            # return super(TransLocator, self).__call__()
            return locs[~np.isinf(locs)]

    return TransLocator


locator_factory = locator_transform_factory


class AutoMinorLocator(ticker.AutoMinorLocator):
    """
    For some reason ticker.AutoMinorLocator does not remove overlapping minor ticks
    adequately.  This class explicitly removes minor ticks that are in the same
    location as major ticks.
    """
    tolerance = 1e-6

    def __call__(self):
        """
        Return unique minor tick locations (ensure no duplicates with major ticks)
        """
        majorlocs = self.axis.get_majorticklocs()
        locs = super().__call__()
        kill = np.any(np.abs(majorlocs[:, None] - locs) < self.tolerance, 0)
        return locs[~kill]


class OffsetLocator(ticker.MaxNLocator):
    """
    Get the same tick locations as you would if plotting the variables with
    offset subtracted
    """

    def __call__(self):
        """Return the locations of the ticks."""
        # Note, these are untransformed coordinates
        off = self.axis.major.formatter.offset or 0
        vmin, vmax = self.axis.get_view_interval() - off
        return self.tick_values(vmin, vmax) + off


# ---------------------------------------------------------------------------- #
# Formatters                                                                   #
# ---------------------------------------------------------------------------- #

def formatter_factory(formatter, tolerance=1e-6):
    """
    Create a tick formatter class which, when called eliminates duplicates
    between major/minor ticks (to within given tolerance before invoking the
    parent formatter's call method.
    """

    # TODO: formatter_unique_factory     better name for this factory
    # NOTE: This class is esentially a HACK
    # NOTE: probably better to have some structure that takes both major and
    #  minor locators
    # GloballyUniqueLocator?? with  get_ticklocs    method

    class NoDuplicateTicksFormatter(formatter.__class__):
        def __call__(self, x, pos=None):
            """function that eliminates duplicate tick labels"""

            if np.any(abs(self.axis.get_ticklocs() - x) < tolerance):
                return ''
            return super().__call__(x, pos)

    return NoDuplicateTicksFormatter


class DegreeFormatter(ticker.Formatter):
    def __init__(self, precision=0, radians=False):
        self.precision = precision
        self.scale = 180. / np.pi if radians else 1

    def __call__(self, x, pos=None):
        # \u00b0 : unicode degree symbol
        return '{:.{}f}\u00b0'.format(x * self.scale, self.precision)


class SexagesimalFormatter(ticker.Formatter):

    def __init__(self, precision=None, sep='hms', base_unit='h', short=False,
                 unicode=False, wrap24=True):

        self.precision = precision
        self.sep = sep
        self.base_unit = base_unit
        self.short = short
        self.unicode = unicode
        self.wrap24 = bool(wrap24)

    def __call__(self, x, pos=None):

        if self.wrap24:
            x %= SPD

        return ppr.hms(x, self.precision, self.sep, self.base_unit,
                       self.short, self.unicode)


class InfiniteAwareness:
    """
    Mixin class for TransFormatters that handles representations for
    infinity ∞
    """
    unicode = True
    latex = False

    def __call__(self, x, pos=None):
        xs = super(InfiniteAwareness, self).__call__(x, pos)

        if xs == 'inf':
            if self.latex:
                return r'$\infty$'

            return '∞' if self.unicode else 'inf'

        return xs  #


class TransFormatter(ticker.ScalarFormatter):
    _transform = IdentityTransform()

    def __init__(self, transform=None, precision=None, infinite=1e15,
                 useOffset=None, useMathText=True, useLocale=None):
        """

        Parameters
        ----------
        transform
        precision
        infinite
        useOffset
        useMathText
        useLocale
        """
        super(TransFormatter, self).__init__(useOffset, useMathText, useLocale)
        self.inf = infinite
        self.precision = precision

        if transform is not None:
            if isinstance(transform, Transform):
                self._transform = transform
            else:
                raise ValueError('bork!')

    def __call__(self, x, pos=None):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            x = self._transform.transform(x)

    #     return super(TransFormatter, self).__call__(xt, pos)

    # def pprint_val(self, x):
        # make infinite if beyond threshold
        if abs(x) > self.inf:
            x = np.sign(x) * np.inf

        if (abs(x) == np.inf) and self.useMathText:
            sign = '-' * int(x < 0)
            return rf'{sign}$\infty$'

        return ppr.decimal(x, self.precision)


# TODO: do this as a proper Scale.
#  then ax.set_yscale('linear_rescale', scale=5)
# or even ax.set_yscale('translate', offset=5)


class LinearScaleTransform(Transform):

    input_dims = 1
    output_dims = 1
    is_separable = False
    has_inverse = True

    def __init__(self, scale):
        Transform.__init__(self, 'scaled')
        self.scale = float(scale)

    def transform_affine(self, x):
        return x * self.scale


class LinearRescaleFormatter(TransFormatter):

    def __init__(self, scale=1., infinite=1e15, useOffset=None,
                 useMathText=True, useLocale=None):

        tr = LinearScaleTransform(scale)
        TransFormatter.__init__(self, tr, infinite, useOffset, useMathText,
                                useLocale)

        # FIXME: tick locations now at arb positions. re-locate...


class ReciprocalFormatter(InfiniteAwareness, TransFormatter):
    """
    Reciprocal transform f(x) -> 1 / x
    Useful for quantities that are inversely related. Like period - frequency
    """
    _transform = ReciprocalTransform()


class SciFormatter(ticker.Formatter):
    """Switch between scientific and decimal formatting based on precision"""

    def __init__(self, precision=2, times=r'\times'):
        self.precision = int(precision)
        self.times = times

    def __call__(self, x, pos=None):
        return ppr.sci(x, self.precision, times=self.times)


class MetricFormatter(ticker.Formatter):
    """
    Formats axis values using metric prefixes to represent powers of 1000,
    plus a specified unit, e.g., 10 MHz instead of 1e7.
    """

    # the prefix for -6 is the greek letter mu
    # represented here by a TeX string

    # The SI metric prefixes  # TODO: this now in recipes.pprint
    METRIC_PREFIXES = {-24: 'y',
                       -21: 'z',
                       -18: 'a',
                       -15: 'f',
                       -12: 'p',
                       -9: 'n',
                       -6: r'$\mu$',
                       -3: 'm',
                       0: '',
                       3: 'k',
                       6: 'M',
                       9: 'G',
                       12: 'T',
                       15: 'P',
                       18: 'E',
                       21: 'Z',
                       24: 'Y'}

    def __init__(self, unit="", precision=None, uselabel=True):
        self.baseunit = unit
        self.pow10 = 0

        if precision is None:
            self.format_str = '{:g}'
        elif precision > 0:
            self.format_str = '{0:.%if}' % precision

        self.uselabel = uselabel

    def __call__(self, x, pos=None):
        s = self.metric_format(x)
        # if x==self.locs[-1]:
        # s += self.unit
        return self.fix_minus(s)

    def set_locs(self, locs):
        # print( 'set_locs' )
        self.locs = locs

        sign = np.sign(locs)
        logs = sign * np.log10(sign * locs)
        pow10 = int(mode(logs // 3)[0][0] * 3)

        self.pow10 = pow10 = np.clip(pow10, -24, 24)
        # self.mantissa = locs / (10 ** pow10)
        self.prefix = self.METRIC_PREFIXES[pow10]
        self.unit = self.prefix + self.baseunit

        # if self.unitlabel is None:
        # self.unitlabel = self.axis.axes.text( 1, 1, self.unit,
        # transform=self.axis.axes.transAxes )
        # else:
        # self.unitlabel.set_text( self.unit )
        if self.uselabel:
            label = self.axis.label.get_text()
            if self.unit not in label:
                ix = label.find('(') if '(' in label else None
                label = label[:ix].strip()
                self.axis.label.set_text(f'{label} ({self.unit})')

    def metric_format(self, num):

        mant = num / (10. ** self.pow10)
        formatted = self.format_str.format(mant)

        return formatted.strip()


# ---------------------------------------------------------------------------- #

def _rotate_tick_labels(ax, angle, minor=False, pad=0):

    ax.tick_params('x', pad=pad)
    ticklabels = ax.xaxis.get_ticklabels(minor)
    for label in ticklabels:
        label.set(ha='left', va='bottom',
                  rotation=angle)
        #   rotation_mode='anchor')


class DateTick(ticker.Formatter):

    def __init__(self, date):
        self.date = datetime.date(*map(int, date.split('-')))
        self._ticks = {}

    def set_locs(self, locs):
        super().set_locs(locs)

        if len(locs):
            majloc = self.axis.major.locator()
            i = np.array([0, *np.diff(majloc // SPD)], bool)
            minor_interval = np.diff(locs).min()
            dateloc = majloc[i] - minor_interval

            x0 = self.axis.axes.get_xlim()[0]
            maj0 = majloc[np.digitize(x0, majloc)] - minor_interval
            min0 = locs[np.digitize(x0, locs)]
            first = min(min0, maj0).item()

            self._ticks = {t: str(self.date + datetime.timedelta(i))
                           for i, t in enumerate((first, *dateloc))}

    def __call__(self, x, pos=None):
        return self._ticks.get(x, '')


# ---------------------------------------------------------------------------- #

def axis_add_custom_ticks(axis, ticks):
    locator = axis.get_major_locator()
    formatter = axis.get_major_formatter()
    axis.set_major_locator(AdditionalTickLocator(locator, ticks.keys()))
    axis.set_major_formatter(AdditionalTickFormatter(formatter, ticks))


class AdditionalTickLocator(ticker.Locator):
    '''This locator chains whatever locator given to it, and then add addition custom ticks to the result'''

    def __init__(self, chain: ticker.Locator, ticks) -> None:
        super().__init__()
        assert chain is not None
        self._chain = chain
        self._additional_ticks = np.asarray(list(ticks))

    def _add_locs(self, locs):
        locs = np.unique(np.concatenate([
            np.asarray(locs),
            self._additional_ticks
        ]))
        return locs

    def tick_values(self, vmin, vmax):
        locs = self._chain.tick_values(vmin, vmax)
        return self._add_locs(locs)

    def __call__(self):
        # this will call into chain's own tick_values,
        # so we also add ours here
        locs = self._chain.__call__()
        return self._add_locs(locs)

    def nonsingular(self, v0, v1):
        return self._chain.nonsingular(v0, v1)

    def set_params(self, **kws):
        return self._chain.set_params(**kws)

    def view_limits(self, vmin, vmax):
        return self._chain.view_limits(vmin, vmax)


class AdditionalTickFormatter(ticker.Formatter):
    '''This formatter chains whatever formatter given to it, and
    then does special formatting for those passed in custom ticks'''

    def __init__(self, chain: ticker.Formatter, ticks) -> None:
        super().__init__()
        assert chain is not None
        self._chain = chain
        self._additional_ticks = ticks

    def __call__(self, x, pos=None):
        if x in self._additional_ticks:
            return self._additional_ticks[x]
        res = self._chain.__call__(x, pos)
        return res

    def format_data_short(self, value):
        if value in self._additional_ticks:
            return self.__call__(value)
        return self._chain.format_data_short(value)

    def get_offset(self):
        return self._chain.get_offset()

    def _set_locator(self, locator):
        self._chain._set_locator(locator)

    def set_locs(self, locs):
        self._chain.set_locs(locs)
