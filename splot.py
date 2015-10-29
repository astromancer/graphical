import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.parasite_axes import SubplotHost
import mpl_toolkits.axisartist as AA
from matplotlib import ticker
from matplotlib import scale as mscale
from matplotlib import transforms as mtransforms

#from .lc import LCplot

from IPython import embed

class ReciprocalTransform(mtransforms.Transform): 
    input_dims = 1 
    output_dims = 1 
    is_separable = False 
    has_inverse = True 

    def __init__(self, thresh=1e-6):
        mtransforms.Transform.__init__(self)
        self.thresh = thresh

    def transform_non_affine(self, x): 
        mask = abs(x) < self.thresh
        x[mask] = self.thresh
        #if any(mask):
            #masked = np.ma.masked_where(mask, x)
            #return 1. / masked
        #else:
        return 1. / x   

    def inverted(self): 
        return self


class ReciprocalScale(mscale.ScaleBase):
    '''
    Scales data in range (0, inf) using a reciprocal scale.

    The (inverse) scale function:
    1 / x
    '''
    name = 'reciprocal'

    def __init__(self, axis, **kwargs):
        """
        Any keyword arguments passed to ``set_xscale`` and
        ``set_yscale`` will be passed along to the scale's
        constructor.
        """
        mscale.ScaleBase.__init__(self)

    def get_transform(self):
        """
        Override this method to return a new instance that does the
        actual transformation of the data.
        """
        return ReciprocalTransform()

    def limit_range_for_scale(self, vmin, vmax, minpos):
        """
        Override to limit the bounds of the axis to the domain of the
        transform.  Unlike the autoscaling provided by the tick locators,
        this range limiting will always be adhered to, whether the axis 
        range is set manually, determined automatically or changed through 
        panning and zooming.
        """
        print( vmin, vmax )
        return min(vmin,1e9), min(vmax, 1e9)

    #@print_args()
    def set_default_locators_and_formatters(self, axis):
        """
        Override to set up the locators and formatters to use with the
        scale.  This is only required if the scale requires custom 
        locators and formatters."""
        
        majloc = axis.major.locator
        #print( majloc )
        #MajLoc = type('Foo', (majloc.__class__,), 
        #              {'__call__' : lambda obj: self.get_transform().transform( majloc() )}  )
        #print(majloc.__class__)
        
        class TransMajLoc(majloc.__class__):
            def __call__(obj):
                s = super(TransMajLoc, obj).__call__()
                r = self.get_transform().transform( s ) 
                print( 's', s )
                print( 'r', r )
                return r
        
        #print( TransMajLoc )
        #print( TransMajLoc() )
        #print( TransMajLoc()() )
        #axis.set_major_locator( TransMajLoc() )
        #q = ticker.ScalarFormatter()
        #embed()
        #axis.set_major_formatter( q )


mscale.register_scale( ReciprocalScale )


#****************************************************************************************************
class NoDuplicateTicks(type):
    '''meta class to elliminate replication of major tick labels in minor tick labels.'''
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __new__(meta, name, bases, attrs):
        attrs['__call__'] = elliminateDuplicateTickLabels
        return super(NoDuplicateTicks, meta).__new__(meta, name, bases, attrs)

#====================================================================================================
def elliminateDuplicateTickLabels(formatter, x, pos=None):
    '''function that elliminates duplicate ticx labels'''
    precision = 9
    if round(x, precision) in formatter.axis.get_ticklocs():
        return ''
    return super(formatter.__class__, formatter).__call__(x, pos)
    
#====================================================================================================
def formatter_factory(formatter):
    '''
    Create a tick formatter class which, when called applies the
    transformation given in ``trans`` before invoking the parent formatter's
    call.
    '''
    FormatterClass = formatter.__class__
    #****************************************************************************************************
    class TransFormatter(FormatterClass): #, metaclass=meta
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        def __call__(self, x, pos=None):
            #print( 'x', x )
            #if np.isnan(x) or not np.isfinite(x):
            if np.ma.is_masked(x):
                return ''
            
            #print( 'xtrans', xtrans )
            #print( FormatterClass )
            #print( 'FormatterClass.__call__(self, xtrans, pos=None)', 
                    #FormatterClass.__call__(self, xtrans, pos=None) )
            
            return FormatterClass.__call__(self, x, pos=None)
        
    return TransFormatter

from scipy.stats import mode
class MetricFormatter(ticker.Formatter):
    """
    Formats axis values using metric prefixes to represent powers of 1000,
    plus a specified unit, e.g., 10 MHz instead of 1e7.
    """

    # the unicode for -6 is the greek letter mu
    # commeted here due to bug in pep8
    # (https://github.com/jcrocholl/pep8/issues/271)

    # The SI metric prefixes
    METRIC_PREFIXES = {
        -24: "y",
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
         24: "Y"
    }

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
        #if x==self.locs[-1]:
            #s += self.unit
        return self.fix_minus(s)

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

    def metric_format(self, num):

        mant = num / (10 ** self.pow10)
        formatted = self.format_str.format( mant )

        return formatted.strip()




def locator_factory(locator, transform):
    '''
    Create a tick formatter class which, when called applies the
    transformation given in ``trans`` before invoking the parent formatter's
    call.
    '''
    LocatorClass = locator.__class__
    #****************************************************************************************************
    class TransLocator(LocatorClass):
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        def __call__(self):
            locs = locator()
            locs = transform.transform( locs[locs > transform.thresh] )
            #if np.ma.is_masked(locs):
                #return locs[~locs.mask]
            return locs
    return TransLocator


#****************************************************************************************************
class TwinnedAxes(SubplotHost):
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __init__(self, *args, **kwargs):
        
        super(SubplotHost, self).__init__(*args, **kwargs)
        
        aux_trans = mtransforms.BlendedGenericTransform(ReciprocalTransform(), mtransforms.IdentityTransform()) 
        
        #Initialize the parasite axis
        self.parasite = self.twin( aux_trans ) # ax2 is responsible for "top" axis and "right" axis
        
   
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def setup_ticks(self):
        
        #Tick setup for host axes
        self.xaxis.set_tick_params( 'both', tickdir='out' )   
        
        #Set the minor tick formatter / locator
        minorTickSize = 8
        self.xaxis.set_tick_params( 'minor', labelsize=minorTickSize )
        
        #mfmtr = self.xaxis.get_minor_formatter()
        #if isinstance(mfmtr, ticker.NullFormatter):
            #sf = ticker.ScalarFormatter()
            
        MinorScalarFormatter = NoDuplicateTicks('MinorScalarFormatter', 
                                            (ticker.ScalarFormatter,),
                                            {})
        msf = MinorScalarFormatter()
        self.xaxis.set_minor_formatter( msf )

        mloc = self.xaxis.get_minor_locator()
        if isinstance(mloc, ticker.NullLocator):
            mloc = ticker.AutoMinorLocator()
            self.xaxis.set_minor_locator( mloc )
            
            
        #Tick setup for parasite axes
        self.parasite.xaxis.tick_top()
        self.parasite.axis['right'].major_ticklabels.set_visible(False)
        self.parasite.xaxis.set_tick_params( 'both', tickdir='out' )
        
        #self.parasite.set_xscale( 'reciprocal' )
        #self.parasite.set_viewlim_mode( 'transform' )
        
        #mtransforms.Affine2D().scale(x_scale, y_scale)
        
        #Set the major tick formatter / locator
        TransLocator = locator_factory( self.xaxis.get_major_locator() , ReciprocalTransform() )
        self.parasite.xaxis.set_major_locator( TransLocator() )
        
        #MajForm = formatter_factory( ticker.ScalarFormatter() ) #self.xaxis.get_major_formatter()
        MajForm = MetricFormatter('Hz', precision=1)
        self.parasite.xaxis.set_major_formatter( MajForm )
        
        #Set the minor tick formatter / locator
        TransMinorLocator = locator_factory( self.xaxis.get_minor_locator() , ReciprocalTransform() )
        self.parasite.xaxis.set_minor_locator( TransMinorLocator() )
        
        #MajForm = formatter_factory( self.xaxis.get_major_formatter() )
        minform = MetricFormatter('Hz', precision=1, uselabel=False)
        self.parasite.xaxis.set_minor_formatter( minform  )    #MajForm()
        
        self.parasite.xaxis.set_tick_params( 'minor', labelsize=minorTickSize )
        
        #print( TransMinorLocator() )
        #print( TransMinorLocator()() )
        
        #MinorParasite = NoDuplicateTicks( 'MinorParasite', 
                                          #(formatter_factory(sf, trans),),
                                          #{} )
        
        #self.parasite.xaxis.set_minor_formatter( MinorParasite() )
        
        self.figure.tight_layout( rect=[0.02, 0.01, None, 0.98] )
        

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def set_xscale(self, value, **kw):
        super(TwinnedAxes, self).set_xscale(value, **kw)
        
        self.parasite.set_xscale(value, **kw)
        #self.parasite.xaxis.set_major_formatter( ReciprocalFormatter() )


#class Splot(LCplot):
    ##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #_default_errorbar = dict( fmt='-', 
                              #capsize=0 )
    ##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #def setup_figure_geometry(self, ax): 
        #fig = plt.figure()
        #ax = TwinnedAxes(fig, 1, 1, 1)
        #fig.add_subplot(ax)
        #ax.setup_ticks()
        #return fig, ax
        
    ##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #@staticmethod
    #def _set_labels(ax, title, axlabels, relative_time, t0):
        #'''axis title + labels'''
        #ax.set_title( title )
        
        #ax.set_xlabel( axlabels[0]      if len(axlabels)     else 'Frequency (Hz)' )
        #ax.set_ylabel( axlabels[1]      if len(axlabels)==2  else 'Power' )
        
        #plab, flab = 'Period (s)', 'Frequency (Hz)'
        #ax.parasite.set_xlabel( flab )
        #ax.set_xlabel( plab )
        #ax.set_ylabel( r'$\Theta^{-1}$' )


#splot = Splot()
