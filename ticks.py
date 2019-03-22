import warnings

import numpy as np
from scipy.stats import mode

from matplotlib import ticker
from matplotlib.transforms import (Transform,
                                   IdentityTransform,
                                   ScaledTranslation)

from recipes.pprint import decimal_repr, sci_repr

from .transforms import ReciprocalTransform


# ====================================================================================================
def formatter_factory(formatter, tolerance=1e-6):  # TODO: formatter_unique_factory     better name for this factory
    # NOTE: This class is esentially a HACK
    # NOTE: probably better to have some structure that takes both major and minor locators
    # GloballyUniqueLocator?? with  get_ticklocs    method
    """
    Create a tick formatter class which, when called eliminates duplicates between major/minor
    ticks (to within given tolerance before invoking the parent formatter's call method.
    """
    FormatterClass = formatter.__class__

    # ****************************************************************************************************
    class NoDuplicateTicksFormatter(FormatterClass):
        def __call__(self, x, pos=None):
            """function that eliminates duplicate tick labels"""

            if np.any(abs(self.axis.get_ticklocs() - x) < tolerance):
                # print( x, 'fuckt!'  )
                return ''
            return super(NoDuplicateTicksFormatter, self).__call__(x, pos)

    return NoDuplicateTicksFormatter


# TODO: NoOverlappingTicksFormatter.  NOTE: Probably not necessary if you choose appropriate locators

# ====================================================================================================
def locator_transform_factory(locator, transform):
    """
    Create a tick formatter class which, when called applies the transformation given in ``transform``
     before invoking the parent formatter's __call__ method.
    """
    # _locator = expose.args()(locator)

    LocatorClass = locator.__class__

    # ****************************************************************************************************
    class TransLocator(LocatorClass):
        def __call__(self):
            locs = locator()
            # try:
            locs = transform.transform(locs)  # [locs > transform.thresh]
            # FIXME: ValueError: total size of new array must be unchanged. when only a single tick on axis
            # except ValueError as err:
            #     print('FUCKup:',  locs, err)
            # if np.ma.is_masked(locs):
            # return locs[~locs.mask]

            # return super(TransLocator, self).__call__()
            return locs[~np.isinf(locs)]

    return TransLocator


locator_factory = locator_transform_factory


# ****************************************************************************************************
class SwitchLogFormatter(ticker.Formatter):
    """Switch between log and scalar format based on precision"""

    def __init__(self, precision=2, mult_symb=r'\times'):
        self.precision = int(precision)
        self.mult_symb = mult_symb

    def __call__(self, x, pos=None):
        return sci_repr(x, self.precision, times=self.mult_symb)


# ****************************************************************************************************
# class TransFormatter(ticker.ScalarFormatter):
# _transform = IdentityTransform()

# def __call__(self, x, pos=None):
# with warnings.catch_warnings():
# warnings.simplefilter("ignore")
# xt = self._transform.transform(x)

# return decimal_repr(xt, 3)


# ****************************************************************************************************
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
            xt = self._transform.transform(x)

        return super(TransFormatter, self).__call__(xt, pos)

    def pprint_val(self, x):
        # make infinite if beyond threshold

        if abs(x) > self.inf:
            x = np.sign(x) * np.inf

        if abs(x) == np.inf:
            if self.useMathText:
                sign = '-' * int(x < 0)
                return r'{}$\infty$'.format(sign)

        return decimal_repr(x, self.precision)


# ****************************************************************************************************
class InfiniteAwareness(object):
    """
    Mixin class that formats inf as infinity symbol using either TeX or unicode
    """

    def __call__(self, x, pos=None):
        xs = super(InfiniteAwareness, self).__call__(x, pos)
        if xs == 'inf':
            return r'$\infty$'  # NOTE: unicode infinity:  u'\u221E'
        else:
            return xs  #


class ReciprocalFormatter(InfiniteAwareness, TransFormatter):
    _transform = ReciprocalTransform()


# ****************************************************************************************************
class DegreeFormatter(ticker.Formatter):
    def __init__(self, precision=0, radians=False):
        self.precision = precision
        if radians:
            self.scale = 180. / np.pi
        else:
            self.scale = 1

    def __call__(self, x, pos=None):
        # \u00b0 : unicode degree symbol
        return '{:.{}f}\u00b0'.format(x * self.scale, self.precision)


# ****************************************************************************************************
class MetricFormatter(ticker.Formatter):
    """
    Formats axis values using metric prefixes to represent powers of 1000,
    plus a specified unit, e.g., 10 MHz instead of 1e7.
    """

    # the prefix for -6 is the greek letter mu
    # represented here by a TeX string

    # The SI metric prefixes
    METRIC_PREFIXES = {-24: "y",
                       -21: "z",
                       -18: "a",
                       -15: "f",
                       -12: "p",
                       -9: "n",
                       -6: "$\mu$",
                       -3: "m",
                       0: "",
                       3: "k",
                       6: "M",
                       9: "G",
                       12: "T",
                       15: "P",
                       18: "E",
                       21: "Z",
                       24: "Y"}

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
            if not self.unit in label:
                ix = label.find('(') if '(' in label else None
                label = label[:ix].strip()
                self.axis.label.set_text('{}    ({})'.format(label, self.unit))

    def metric_format(self, num):

        mant = num / (10 ** self.pow10)
        formatted = self.format_str.format(mant)

        return formatted.strip()


# ****************************************************************************************************
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
        locs = super(self.__class__, self).__call__()
        kill = np.any(np.abs(majorlocs[:, None] - locs) < self.tolerance, 0)
        return locs[~kill]
