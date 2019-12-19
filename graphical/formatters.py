"""
Additional matplotlib axes tick formatters
"""

import warnings

import numpy as np
from scipy.stats import mode
from matplotlib import ticker
from matplotlib.transforms import IdentityTransform, \
    Transform  # , ScaledTranslation

# recipes.pprint import decimal_repr
from recipes import pprint

from .transforms import ReciprocalTransform

class TransFormatter(ticker.ScalarFormatter):
    _transform = IdentityTransform()

    def __init__(self, transform=None, infinite=1e15, useOffset=None,
                 useMathText=True, useLocale=None):
        super(TransFormatter, self).__init__(useOffset, useMathText, useLocale)
        self.inf = infinite

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
        # print('PPP')
        if abs(x) > self.inf:
            x = np.sign(x) * np.inf

        if abs(x) == np.inf:
            if self.useMathText:
                sign = '-' * int(x < 0)
                return r'{}$\infty$'.format(sign)

        return pprint.decimal(x, 2)


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
            elif self.unicode:
                return '∞'
            else:
                return 'inf'
        else:
            return xs  #




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
        return pprint.sci(x, self.precision, times=self.times)




class MetricFormatter(ticker.Formatter):
    """
    Formats axis values using metric prefixes to represent powers of 1000,
    plus a specified unit, e.g., 10 MHz instead of 1e7.
    """

    # the prefix for -6 is the greek letter mu
    # represented here by a TeX string

    # The SI metric prefixes  # TODO: this now in recipes.pprint
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
                self.axis.label.set_text('{} ({})'.format(label, self.unit))

    def metric_format(self, num):

        mant = num / (10. ** self.pow10)
        formatted = self.format_str.format(mant)

        return formatted.strip()
