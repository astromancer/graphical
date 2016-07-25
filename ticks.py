import warnings

import numpy as np
from scipy.stats import mode

from matplotlib import ticker
from matplotlib.transforms import (Transform, 
                                   IdentityTransform,
                                   ScaledTranslation)

from recipes.string import minfloatformat



#====================================================================================================
def formatter_factory(formatter, tolerance=1e-6):
    '''
    Create a tick formatter class which, when called elliminates duplicates between major/minor
    ticks (to within given tolerance before invoking the parent formatter's call method.
    '''
    FormatterClass = formatter.__class__
    #****************************************************************************************************
    class NoDuplicateTicksFormatter(FormatterClass):
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        def __call__(self, x, pos=None):
            '''function that elliminates duplicate tick labels'''

            if np.any(abs(self.axis.get_ticklocs() - x) < tolerance):
                #print( x, 'fuckt!'  )
                return ''
            return super(NoDuplicateTicksFormatter, self).__call__(x, pos)
        
    return NoDuplicateTicksFormatter


#====================================================================================================
def locator_factory(locator, transform):
    '''
    Create a tick formatter class which, when called applies the
    transformation given in ``trans`` before invoking the parent formatter's
    call.
    '''
    #_locator = expose.args()(locator)
    
    LocatorClass = locator.__class__
    #****************************************************************************************************
    class TransLocator(LocatorClass):
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        def __call__(self):
            locs = locator()
            locs = transform.transform(locs) #[locs > transform.thresh]
            #if np.ma.is_masked(locs):
                #return locs[~locs.mask]
                
            #return super(TransLocator, self).__call__()
            return locs[~np.isinf(locs)]
        
    return TransLocator




#****************************************************************************************************
#class TransFormatter(ticker.ScalarFormatter):
    #_transform = IdentityTransform()
        
    #def __call__(self, x, pos=None):
        #with warnings.catch_warnings():
            #warnings.simplefilter("ignore")
            #xt = self._transform.transform(x)
            
        #return minfloatformat(xt, 3)


class TransFormatter(ticker.ScalarFormatter):
    _transform =  IdentityTransform()
    
    def __init__(self, transform=None, infinite=1e15, useOffset=None, useMathText=True, useLocale=None):
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
        #make infinite if beyond threshold
        #print('PPP')
        if abs(x) > self.inf:
            x = np.sign(x) * np.inf
        
        if abs(x) == np.inf:
            if self.useMathText:
                sign = '-' * int(x<0)
                return r'{}$\infty$'.format(sign)
        
        return minfloatformat(x,2)
#         #return super().pprint_val(x)   #FIXME: does not produce correct range of ticks


#****************************************************************************************************
class InfiniteAwareness():
    def __call__(self, x, pos=None):
        xs = super(InfiniteAwareness, self).__call__(x, pos)

        if xs == 'inf':
            return r'$\infty$'
        else:
            return xs #
 

#****************************************************************************************************
class DegreeFormatter(ticker.Formatter):
    def __init__(self, precision=0):
        self.precision = precision
        
    def __call__(self, x, pos=None):
        # \u00b0 : degree symbol
        return '{:.{}f}\u00b0'.format(x, self.precision)

    
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
    
    
#****************************************************************************************************
class AutoMinorLocator(ticker.AutoMinorLocator):
    '''
    For some reason ticker.AutoMinorLocator does not remove overlapping minor ticks
    adequately.  This class explicitly removes minor ticks that are in the same 
    location as major ticks.
    '''
    tolerance = 1e-6
    def __call__(self):
        '''Return unique minor tick locations (ensure no duplicates with major ticks)'''
        majorlocs = self.axis.get_majorticklocs()
        locs = super(self.__class__, self).__call__()
        kill = np.any(np.abs(majorlocs[:, None]-locs) < self.tolerance, 0)
        return locs[~kill]

