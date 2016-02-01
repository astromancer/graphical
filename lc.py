import itertools as itt
from collections import defaultdict

#from myio import ProcessCommunicator
from recipes.iter import interleave#, flatiter
#from myio import warn

import numpy as np
#import pyfits
#from scipy.optimize import leastsq

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable 
#from imagine import supershow
from matplotlib.patches import Rectangle
from draggables import DraggableErrorbar
from matplotlib.transforms import blended_transform_factory as btf

from .dualaxes import DateTimeDualAxes

from IPython import embed
            

#====================================================================================================
def get_axlim(x, whitefrac, e=0):
    '''Return suggested axis limits based on the extrema of x, and the desired fractional whitespace 
    on plot.'''
    
    if ~np.any(e):
        e = 0
    
    xl, xu = (x-e).min(), (x+e).max()
    xd = xu - xl
    return xl-whitefrac*xd, xu+whitefrac*xd

#====================================================================================================
def hist( x, **kws ):
    '''Plot a nice looking histogram.
    
    Parameters
    ----------
    x:          sequence
        Values to histogram
    
    Keywords
    --------
    axlabels:   sequence
        One or two axis labels (x,y)
    title:      str
        The figure title
    show_stats: str; option ('mode',)
        Show the given statistic of the distribution
    * Remaining keywords are passed to ax.hist
    
    Returns
    -------
    h:          tuple
        bins, values
    ax:         axes
    '''
    
    show_stats  = kws.pop( 'show_stats', () )
    lbls        = kws.pop( 'axlabels', () )
    title       = kws.pop( 'title', '' )
    
    kws.setdefault( 'bins', 100 )
    alpha = kws.setdefault( 'alpha', 0.5 )
    
    #Create figure
    fig, ax = plt.subplots( tight_layout=1, figsize=(12,8) )

    #Plot the histogram
    h = ax.hist( x, **kws )

    #Make axis labels and title
    xlbl = lbls[0]      if len(lbls)     else ''
    ylbl = lbls[1]      if len(lbls)>1   else 'Counts'
    ax.set_xlabel( xlbl )
    ax.set_ylabel( ylbl )
    ax.set_title( title )
    ax.grid()

    #Extra stats
    if 'mode' in show_stats:
        from scipy.stats import mode
        xmode, count = mode( x )
        ax.axvline(xmode, color='r', alpha=alpha, ls='--', lw=2 )
        trans = btf( ax.transData, ax.transAxes )
        txt = 'mode = {}'.format( xmode[0] )
        ax.text( xmode, 1, txt, rotation='vertical', transform=trans, va='top', ha='right' )
    
    return h, ax
    
#====================================================================================================
from matplotlib import rcParams
from cycler import cycler
class LCplot(object):
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #TODO: Keyword translation!
    allowed_kws = ('labels', 'axlabels', 'title', 'relative_time', 'colours', 
                   'show_errors',     'errorbar', 'show_masked', 'spans', 
                   'hist', 'show_hist', 'draggable', 'ax', 'whitefrac', 'twinx',
                   'timescale', 'start')
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    whitefrac = 0.025
    _default_spans =    dict( label='filtered', 
                                alpha=0.2, 
                                color='r' )
    _default_hist =     dict( bins=100,
                                alpha=0.75, 
                                color='b',
                                orientation='horizontal' )
    _default_errorbar = dict( fmt='o', 
                                ms=2.5,
                                capsize=0 )
    #nested dict of default plot settings
    _defaults = { 'spans'       : _default_spans,
                  'hist'        : _default_hist,
                  'errorbar'    : _default_errorbar }
    
    _default_legend = dict( loc         =       'upper right',
                            fancybox    =       True,
                            framealpha  =       0.25,
                            numpoints   =       1,
                            markerscale =       3       )
    @staticmethod
    def _set_defaults(props, defaults):
        for k,v in defaults.items():
            props.setdefault(k,v)
    
    #def _get_defaults():
        
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #def __init__(self):

    
    def __call__(self, *data, **kws):
        '''Plot light curve(s)'''
        #TODO: docstring
        #TODO: astropy.units
        
        self.mask_shown = False
        self.Hist = []
        
        #Check keyword argument validity
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        for kw in kws:
            if not kw in self.allowed_kws:
                raise KeyError('Invalid keyword argument {}.\n'
                               'Only the following keywords are recognised: {}'
                               ''.format(kw, self.allowed_kws))
        
        
        #Set parameter defaults
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        labels                  =       kws.pop( 'labels',               [] )
        axlabels                =       kws.pop( 'axlabels',             [] )
        title                   =       kws.pop( 'title',                '' )
        self.relative_time      =       kws.pop( 'relative_time',        False )
        
        colours                 =       kws.get( 'colours',              [] ) #colour cycle
        
        show_errors             =       kws.pop( 'show_errors',  'bar' ) #{'bar','contour'}
        self.errorbar_props     =       kws.pop( 'errorbar',     {} )
        self._set_defaults( self.errorbar_props, self._default_errorbar )
        #self.error_contours     =       kws.pop( 'error_contours',     False )
        
        show_masked             =       kws.pop( 'show_masked',  False )#'spans' in kws
        self.span_props         =       kws.pop( 'spans',        {} )
        self._set_defaults( self.span_props, self._default_spans )
        
        self.hist_props         =       kws.pop( 'hist',           {} )
        self.show_hist          =       kws.pop( 'show_hist', bool(len(self.hist_props)) )
        self._set_defaults( self.hist_props, self._default_hist )
        
        draggable               =       kws.pop( 'draggable',            True )
        ax                      =       kws.pop( 'ax',                   None )
        whitefrac               =       kws.pop( 'whitefrac',            self.whitefrac )
        whitefrac = np.atleast_1d(whitefrac)
        whitefrac = np.r_[whitefrac, whitefrac] if len(whitefrac)==1 else whitefrac[:2]
        
        #Upper time axis display
        #NOTE: this might get messy! Consider setting up the axes outside and passing
        self.twinx              =       kws.pop( 'twinx',           None )
        self.timescale          =       kws.pop( 'timescale',       's' )
        self.start              =       kws.pop( 'start',        None )
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if isinstance(labels, str):
            labels = [labels]
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        Times, Rates, Errors = self.get_data(data)
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #Ensure we plot with unique colours
        if len(colours) < Rates.shape[0]:               #if Rates.shape[0] > len(rcParams['axes.prop_cycle']):
            cm = plt.get_cmap( 'spectral' )
            colours =  cm( np.linspace(0, 1, Rates.shape[0]) )
        #else:
            #colours = colours[:Rates.shape[0]]
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #Do the plotting
        fig, ax = self.setup_figure_geometry(ax, colours)
        
        plots, masked_plots  = [], []
        _linked = []
        _labels = []
        for t, rate, err, label in itt.zip_longest(Times, Rates, Errors, labels):  #zip_longest in case no Errors are given
            #print( 'plotting' )
            err = err if show_errors else None
            if not np.size(err):
                err = None              #ax.errorbar borks for empty error sequences
                        
            if (not err is None) & (show_errors == 'contour'):
                from tsa.tsa import smooth
                c, = ax.plot(t, smooth(rate + 3*err))
                colour = self.errorbar_props['color'] = c.get_color()   #preserve colour cycle
                ax.plot(t, smooth(rate - 3*err), colour)
                err = None
                
            pl = ax.errorbar(t, rate, err, label=label, **self.errorbar_props)
            plots.append(pl)
            _labels.append(label)       #FIXME: this is messy!!
            
            
            #Histogram
            if self.show_hist:
                self.plot_histogram( rate, **self.hist_props )

            #Get / Plot GTIs
            if show_masked == 'span':
                self.plot_masked_intervals(ax, t, rate.mask)
            
            elif show_masked == 'x':
                unmasked = rate.copy()
                #print( np.where(unmasked.mask) )
                unmasked.mask = ~unmasked.mask
                
                mp = ax.errorbar(t, unmasked, color=pl[0].get_color(),
                                    marker='x', ls='None',
                                    label='_nolegend_')
                masked_plots.append(mp)
                plots.append(mp)
                _linked.append((pl, mp))
                #_labels.append('_nolegend_') #FIXME: this is messy!!
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self._set_labels( ax, title, axlabels )
        self._set_axes_limits( ax, Times, Rates, Errors, whitefrac )
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #Setup plots for canvas interaction
        if draggable:
            #make space for offset text
            plots = DraggableErrorbar(plots, linked=_linked,
                                      **self._default_legend) #FIXME: legend with linked!
        else:
            self._make_legend( ax, plots, _labels )
            
        if hist: 
            return fig, plots, Times, Rates, Errors, self.Hist
        else:
            return fig, plots, Times, Rates, Errors
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #@staticmethod TODO:  handle data of different lenghts!!!!!
    def get_data(self, data):
        
        if len(data)==1:
            Rates = np.ma.asarray( data[0] )            #Assume each row gives rates for TS
            Times = np.arange( Rates.shape[-1] )        #No time given, plot by array index
            Errors = []                                 #No errors given
        
        if len(data)==2:                                #No errors given
            Times, Rates = data
            Errors = []
        
        if len(data)==3:
            Times, Rates, Errors = data
        
        Rates = np.ma.asarray(Rates).squeeze()
        Times = np.ma.asarray(Times).squeeze()
        
            #print( Rates, Times )
            #print( Rates.shape, Times.shape )
            #print( Rates.size, Times.size )
        
        if Times.ndim < Rates.ndim:
            #Assume same times for each sequence of rates
            tmp = np.empty_like( Rates )
            #embed()
            tmp[:] = Times                      #tiles the Times
            Times = tmp
        
        Rates = np.ma.atleast_2d( Rates )
        Rates = np.ma.masked_where( np.isnan(Rates), Rates )
        Times = np.atleast_2d( Times )
        Errors = np.atleast_2d( Errors )
        
        self.t0 = np.floor( Times[:,0].min() )      if self.relative_time   else 0.
        if self.t0:
            Times -= self.t0
        
        return Times, Rates, Errors
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #def _get_ax(self, fig):
        #if self.twin =='sexa':
            #ax = SexaTimeDualAxes(fig, 1, 1, 1)
            #ax.setup_ticks()
        #else:
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #def 
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def setup_figure_geometry(self, ax, colours):
        #FIXME:  leave space on the right of figure to display offsets
        '''Setup figure geometry'''
        
        #TODO:  AUTOMATICALLY DETERMINE THESE VALUES!!
        rect = left, bottom, right, top = [0.025, 0.01, 0.97, .98]
        
        if ax is None:
            if self.twinx =='sexa':
                
                from astropy import units as u
                from astropy.coordinates.angles import Angle
                
                h,m,s = Angle((self.t0 * u.Unit(self.timescale)).to(u.h)).hms       #start time in (h,m,s)
                hms = int(h), int(m), int(round(s))                  #datetime hates floats# although the seconds should be integer, (from np.floor for self.t0) the conversion might have some floating point error
                ymd = tuple(map(int, self.start.split('-')))
                start = ymd + hms
                
                print( start )
                
                fig = plt.figure(figsize=(18,8))
                ax = DateTimeDualAxes(fig, 1, 1, 1, 
                                  timescale=self.timescale,
                                  start=start)
                ax.setup_ticks()
                fig.add_subplot(ax)
                top = .94       #need extra space for the tick labels
            else:
                fig, ax  = plt.subplots(figsize=(18,8))
            
            #fig, ax = plt.subplots( figsize=(18,8) )
            #ax.set_prop_cycle( cycler('color', colours) )      #mpl >1.4 only
            ax.set_color_cycle( colours )
            
            #Add subplot for histogram
            if self.show_hist:
                divider = make_axes_locatable(ax)
                self.hax = divider.append_axes('right', size='25%', pad=0.3, sharey=ax)
                self.hax.grid()
                #self.hax.set_prop_cycle( cycler('color', colours) )    #mpl >1.4 only
                ax.set_color_cycle( colours )
            else:
                self.hax = None         #TODO: can be at init OR NullObject
        else:
            fig = ax.figure
        
        fig.tight_layout( rect=[left, bottom, right, top] )
        ax.grid( b=True, which='both' )
        
        return fig, ax
   
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def plot_masked_intervals(self, ax, t, mask):
        '''Highlight the masked values within the time series with a span accross the axis'''
        spans = self.mask2intervals(t, mask)
        for s in spans:
            #kws = self.span_props
            ax.axvspan( *s, **self.span_props )
        
        self.mask_shown = True#bool(bti)         #just so we don't make a legend entry for this if it's empty
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @staticmethod
    def mask2intervals(a, mask=None):
        '''Retrun index tuples of contiguous masked values.'''
        if mask is None:
            mask = a.mask       #NOTE: If a is a masked array, this function will return masked values!!! 
        if ~np.any(mask):
            return ()

        w, = np.where(mask)
        l1 = w - np.roll(w,1) > 1
        l2 = np.roll(w,-1) -w > 1
        idx = [w[0]] + interleave( w[l2], w[l1] ) + [w[-1]]
        return a[idx].reshape(-1,2)
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def plot_histogram(self, rate, **props):
        #print( 'Plotting H' )
        
        if np.ma.is_masked(rate):
            r = rate[ ~rate.mask ]
        else:
            r = rate
        
        #embed()
        
        h = self.hax.hist(r, **props)
        self.Hist.append(h)
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _set_labels(self, ax, title, axlabels):
        '''axis title + labels'''
        title_text = ax.set_title( title, fontweight='bold' )
        if self.twinx:
            title_text.set_position((0.5,1.09))         #make space for the tick labels
            
        ax.set_xlabel( axlabels[0]      if len(axlabels)     else 't (s)' )     #xlabel =
        ax.set_ylabel( axlabels[1]      if len(axlabels)==2  else 'Counts/s' )  #ylabel =

        if self.relative_time:
            ax.xoffsetText = ax.text( 1, ax.xaxis.labelpad, 
                                      '[{:+.1f}]'.format(self.t0),
                                      ha='right',
                                      transform=ax.xaxis.label.get_transform() )
        
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @staticmethod
    def _set_axes_limits(ax, X, Y, E, whitefrac):
        '''Axes limits'''
        
        if np.ma.is_masked(Y):          #unmask
            X = X[~Y.mask]
        
        xfrac, yfrac = whitefrac
        ax.set_xlim( *get_axlim(X, xfrac) )
        ax.set_ylim( *get_axlim(Y, yfrac, E) )
         
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _make_legend(self, ax, plots, labels):
        '''Legend'''
        
        #print( labels, '!'*10 )
        
        if len(labels):
            if self.mask_shown:
                span_label = self.span_props.pop( 'label' )
                r = Rectangle( (0, 0), 1, 1, **self.span_props )   #span proxy artist for legend
                
                plots += [r]
                labels+= [span_label]
                
            #from misc import make_ipshell
            #ipshell = make_ipshell()
            #ipshell()
            
            #print( plots, labels )
            
            ax.legend( plots, labels,  **self._default_legend)
            
            

    
#====================================================================================================            
lcplot = LCplot()
#====================================================================================================