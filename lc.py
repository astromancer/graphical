import itertools as itt
from collections import defaultdict

#from myio import ProcessCommunicator
from misc import interleave#, flatiter
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
def hist( x, **kw ):
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
    
    show_stats  = kw.pop( 'show_stats', () )
    lbls        = kw.pop( 'axlabels', () )
    title       = kw.pop( 'title', '' )
    
    kw.setdefault( 'bins', 100 )
    alpha = kw.setdefault( 'alpha', 0.5 )
    
    #Create figure
    fig, ax = plt.subplots( tight_layout=1, figsize=(12,8) )

    #Plot the histogram
    h = ax.hist( x, **kw )

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
class LCplot(object):
    #TODO: Keyword translation!
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
                    'hist'      : _default_hist,
                    'errorbar'  : _default_errorbar }
    
    
    
    @staticmethod
    def _set_defaults(props, defaults):
        for k,v in defaults.items():
            props.setdefault(k,v)
    
    #def _get_defaults():
        
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #def __init__(self):

    
    def __call__(self, *data, **kw):
        '''Plot light curve(s)'''
        #TODO: docstring
        
        self.mask_shown = False
        self.Hist = []
        
        #Set parameter defaults
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        labels                  =       kw.pop( 'labels',               [] )
        axlabels                =       kw.pop( 'axlabels',             [] )
        title                   =       kw.pop( 'title',                '' )
        relative_time           =       kw.pop( 'relative_time',        False )
        
        colours                 =       kw.pop( 'colours',      rcParams['axes.color_cycle'] ) #colour cycle
        
        show_errors             =       kw.pop( 'show_errors',  True )
        self.errorbar_props     =       kw.pop( 'errorbar',     {} )
        self._set_defaults( self.errorbar_props, self._default_errorbar )
        
        show_masked             =       kw.pop( 'show_masked',  False )#'spans' in kw
        self.span_props         =       kw.pop( 'spans',        {} )
        self._set_defaults( self.span_props, self._default_spans )
        
        self.hist_props         =       kw.pop( 'hist',           {} )
        self.show_hist          =       kw.pop( 'show_hist', bool(len(self.hist_props)) )
        self._set_defaults( self.hist_props, self._default_hist )
        
        draggable               =       kw.pop( 'draggable',            True )
        ax                      =       kw.pop( 'ax',                   None )
        whitefrac               =       kw.pop( 'whitefrac',            self.whitefrac )
        whitefrac = np.atleast_1d(whitefrac)
        whitefrac = np.r_[whitefrac, whitefrac] if len(whitefrac)==1 else whitefrac[:2]
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if isinstance(labels, str):
            labels = [labels]
        
        fig, ax = self.setup_figure_geometry(ax)
        Times, Rates, Errors = self.get_data(data)
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #Make sure we plot with unique colours
        if Rates.shape[0] > len(colours):
            cm = plt.get_cmap( 'spectral' )
            colours =  cm( np.linspace(0,1,Rates.shape[0]) )
            ax.set_color_cycle( colours )
        else:
            colours = colours[:Rates.shape[0]]
        
        #print( colours )
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #Do the plotting
        
        #embed()
        
        plots  = []
        for t, rate, err, label, colour in itt.zip_longest(Times, Rates, Errors, labels, colours):  #zip_longest in case no Errors are given
            #print( 'plotting' )
            err = err if show_errors else None
            if not np.size(err):
                err = None              #ax.errorbar borks for empty error sequences
            

            if relative_time:
                t0 = np.ceil( t[0] )
                t -= t0
            else:
                t0 = None
            
            self.errorbar_props['label'] = label
            self.errorbar_props['color'] = colour
            
            #print( t, rate, err, self.errorbar_props )
            #embed()
            
            pl = ax.errorbar(t, rate, err, **self.errorbar_props)
            plots.append(pl)

            #Histogram
            if self.show_hist:
                self.hist_props.update( color=colour )
                self.plot_histogram( rate, **self.hist_props )

            #Get / Plot GTIs
            if not show_masked is None:
                if show_masked == 'span':
                    self.plot_masked_intervals(ax, t, rate.mask)
                elif show_masked == 'x':
                    #TODO: CONNECT THESE TO DRAG ALONG!!
                    unmasked = rate.copy()
                    unmasked.mask = ~unmasked.mask
                    ax.plot( t, unmasked, color=colour, marker='x', ls='None' )
                    
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self._set_labels( ax, title, axlabels, relative_time, t0 )
        self._set_axes_limits( ax, Times, Rates, Errors, whitefrac )
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #Setup plots for canvas interaction
        if draggable:
            #make space for offset text
            plots = DraggableErrorbar( plots, markerscale=3 )
        else:
            self._make_legend( ax, plots, labels )
            
        if hist: 
            return fig, plots, Times, Rates, Errors, self.Hist
        else:
            return fig, plots, Times, Rates, Errors
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #@staticmethod TODO:  handle data of different lenghts!!!!!
    def get_data(self, data):
        
        if len(data)==1:
            Rates = np.ma.asarray( data[0] )               #Assume each row gives rates for TS
            Times = np.arange( Rates.shape[-1] )        #No time given, plot by array index
            Errors = []                                 #No errors given
        
        if len(data)==2:                                #No errors given
            Times, Rates = data
            Errors = []
        
        if len(data)==3:
            Times, Rates, Errors = data
        
        Rates = np.ma.asarray(Rates).squeeze()
        Times = np.ma.asarray(Times).squeeze()
        if Times.ndim < Rates.ndim:
            #Assume same times for each sequence of rates
            tmp = np.empty_like( Rates )
            tmp[:] = Times
            Times = tmp
        
        Rates = np.ma.atleast_2d( Rates )
        Rates = np.ma.masked_where( np.isnan(Rates), Rates )
        Times = np.atleast_2d( Times )
        Errors = np.atleast_2d( Errors )
        
        return Times, Rates, Errors
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def setup_figure_geometry(self, ax):        #FIXME:  leave space on the right of figure to display offsets
        '''Setup figure geometry'''
        if ax is None:
            fig, ax = plt.subplots( figsize=(18,8) )
            #Add subplot for histogram
            if self.show_hist:
                divider = make_axes_locatable(ax)
                self.hax = divider.append_axes('right', size='25%', pad=0.3, sharey=ax)
                self.hax.grid()
            else:
                self.hax = None
        else:
            fig = ax.figure
        
        #TODO:  AUTOMATICALLY DETERMINE THESE VALUES!!
        rect = [0.025, 0.01, 0.96, None]
        fig.tight_layout( rect=rect )
        
        ax.grid( b=True, which='both' )
        
        return fig, ax
   
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def plot_masked_intervals(self, ax, t, mask):
        '''Highlight the masked values within the time series with a span accross the axis'''
        spans = self.mask2intervals(t, mask)
        for s in spans:
            #kw = self.span_props
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
    def _set_labels(self, ax, title, axlabels, relative_time, t0):
        '''axis title + labels'''
        ax.set_title( title )
        
        ax.set_xlabel( axlabels[0]      if len(axlabels)     else 't (s)' )     #xlabel =
        ax.set_ylabel( axlabels[1]      if len(axlabels)==2  else 'Counts/s' )  #ylabel =

        if relative_time:
            ax.text( 1, ax.xaxis.labelpad, 
                    '[{:+.1f}]'.format(t0),
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
        
        print( labels, '!'*10 )
        
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
            
            ax.legend( plots, labels )
            
            

    
#====================================================================================================            
lcplot = LCplot()
#====================================================================================================