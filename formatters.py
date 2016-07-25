import warnings

from scipy.stats import mode

from matplotlib import ticker
from matplotlib.transforms import (Transform, 
                                   IdentityTransform,
                                   ScaledTranslation)

from recipes.string import minfloatformat


class TransFormatter(ticker.ScalarFormatter):
    _transform = IdentityTransform()
        
    def __call__(self, x, pos=None):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            xt = self._transform.transform(x)
            
        return minfloatformat(xt, 3)

class InfiniteAwareness():
    def __call__(self, x, pos=None):
        xs = super(InfiniteAwareness, self).__call__(x, pos)

        if xs == 'inf':
            return r'$\infty$'
        else:
            return xs #
 
    
    
#****************************************************************************************************
class MetricFormatter(ticker.Formatter):
    """
    Formats axis values using metric prefixes to represent powers of 1000,
    plus a specified unit, e.g., 10 MHz instead of 1e7.
    """

    # the prefix for -6 is the greek letter mu
    # represented here by a TeX string

    # The SI metric prefixes
    METRIC_PREFIXES = { -24: "y",
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
                        24: "Y"  }

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __init__(self, unit="", precision=None, uselabel=True):
        self.baseunit = unit
        self.pow10 = 0
        
        if precision is None:
            self.format_str = '{:g}'
        elif precision > 0:
            self.format_str = '{0:.%if}' % precision
            
        self.uselabel = uselabel

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __call__(self, x, pos=None):
        s = self.metric_format(x)
        #if x==self.locs[-1]:
            #s += self.unit
        return self.fix_minus(s)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def set_locs(self, locs):
        #print( 'set_locs' )
        self.locs = locs
        
        sign = np.sign( locs )
        logs = sign*np.log10(sign*locs)
        pow10 = int( mode( logs//3 )[0][0] * 3 )

        self.pow10 = pow10 = np.clip( pow10, -24, 24 )
        #self.mantissa = locs / (10 ** pow10)
        self.prefix = self.METRIC_PREFIXES[pow10]
        self.unit = self.prefix + self.baseunit
        
        #if self.unitlabel is None:
            #self.unitlabel = self.axis.axes.text( 1, 1, self.unit, 
                                                 #transform=self.axis.axes.transAxes )
        #else:
            #self.unitlabel.set_text( self.unit )
        if self.uselabel:
            label = self.axis.label.get_text()
            if not self.unit in label:
                ix = label.find('(') if '(' in label else None
                label = label[:ix].strip()
                self.axis.label.set_text( '{}    ({})'.format(label, self.unit) )

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def metric_format(self, num):

        mant = num / (10 ** self.pow10)
        formatted = self.format_str.format( mant )

        return formatted.strip()