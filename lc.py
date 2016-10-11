import itertools as itt

import numpy as np
from matplotlib import rcParams
import matplotlib.pyplot as plt
#import colormaps as cmaps
#plt.register_cmap(name='viridis', cmap=cmaps.viridis)

from recipes.array import grid_like
from recipes.string import minlogfmt

from IPython import embed


#====================================================================================================
def get_axlim(x, whitefrac, e=0):
    '''
    Return suggested axis limits based on the extrema of x, and the desired
    fractional whitespace on plot.
    whitespace  - can be number, or (bottom, top) tuple
    e - uncertainty
        can be either single array of same shape as x, or 2 arrays (upper, lower)
    '''
    if np.ma.is_masked(x):
        x = x[~x.mask]          #unmask
    else:
        x = np.asarray(x)
    x = x.flatten()

    e = np.asarray(e)
    e_ = np.zeros((2, len(x)))
    if e.size:      #passed array may have 0 size
        e_[:] = e#.flatten()
    el, eu = e_

    xl, xu = np.nanmin(x-el), np.nanmax(x+eu)
    xd = xu - xl
    wf = np.empty(2)
    wf[:] = whitefrac
    wxd = wf * xd
    return xl - wxd[0], xu + wxd[1]

#====================================================================================================
def hist(x, **kws):
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

    show_stats  = kws.pop('show_stats', ())
    fmt_stats   = kws.pop('fmt_stats', None)
    lbls        = kws.pop('axlabels', ())
    title       = kws.pop('title', '')
    #ax = ax.plot

    kws.setdefault('bins', 100)
    alpha = kws.setdefault('alpha', 0.5)
    Q = kws.pop('percentile', [])
    named_quantiles = {25 : 'lower  quartile',      #https://en.wikipedia.org/wiki/Quantile#Specialized_quantiles
                       50 : 'median',
                       75 : 'upper quartile'}


    #Create figure
    ax                      =       kws.pop('ax',        None)
    if ax is None:
        _, ax = plt.subplots(tight_layout=1, figsize=(12,8))
    #else:
        #fig = ax.figure

    #Plot the histogram
    h = ax.hist(x, **kws)

    #Make axis labels and title
    xlbl = lbls[0]      if len(lbls)     else ''
    ylbl = lbls[1]      if len(lbls)>1   else 'Counts'
    ax.set_xlabel(xlbl)
    ax.set_ylabel(ylbl)
    ax.set_title(title)
    ax.grid()

    #Extra stats #FIXME bad nomenclature
    if len(show_stats):
        from matplotlib.transforms import blended_transform_factory as btf
        stats = {}
    if 'min' in show_stats:
        stats['min'] = x.min()

    if 'max' in show_stats:
        stats['max'] = x.max()

    if 'mode' in show_stats:
        from scipy.stats import mode
        mr = mode(x)
        xmode = mr.mode.squeeze()
        stats['mode'] = xmode

    if 'mean' in show_stats:
        stats['mean'] = x.mean()
    if 'median' in show_stats:
        Q.append(50)

    if len(Q): #'percentile' in show_stats:
        P = np.percentile(x, Q)
        for p, q in zip(P, Q):
            name = named_quantiles.get(q, '$p_{%i}$' % q)
            stats[name] = p

    if fmt_stats is None:
        from recipes.string import minfloatfmt
        fmt_stats = minfloatfmt

    for key, val in stats.items():
        ax.axvline(val, color='r', alpha=alpha, ls='--', lw=2)
        trans = btf(ax.transData, ax.transAxes)
        txt = '%s = %s' % (key, fmt_stats(val))
        ax.text(val, 1, txt,
                rotation='vertical', transform=trans, va='top', ha='right')

    #if 'percentile' in show_stats:
        #pass

    return h, ax

#****************************************************************************************************
class LCplot(object):
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #TODO: Keyword translation!
    allowed_kws = ('labels', 'axlabels', 'title', 'relative_time', 'colours', 'colormap',
                   'show_errors',     'errorbar', 'show_masked', 'spans',
                   'hist', 'show_hist', 'draggable', 'ax', 'whitefrac', 'twinx',
                   'timescale', 'start', 'offsets', 'legend_kw', 'auto_legend')

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    maxpoints = 1e4             #TODO
    whitefrac = 0.025
    _default_spans =    dict(label='filtered',
                             alpha=0.2,
                             color='r' )
    _default_hist =     dict(bins=50,
                             alpha=0.75,
                             #color='b',
                             orientation='horizontal' )
    _default_errorbar = dict(fmt='o',
                             ms=2.5,
                             mec='none',
                             capsize=0,
                             elinewidth=0.5)
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
        #TODO: max points = 1e4 ??

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
        labels                  =       kws.pop('labels',               None )
        axlabels                =       kws.pop('axlabels',             [] )
        title                   =       kws.pop('title',                '' )
        self.relative_time      =       kws.pop('relative_time',        False )

        colours                 =       kws.get('colours',             [] ) #colour cycle
        cmap                    =       kws.get('colormap',            None )

        show_errors             =       kws.pop('show_errors',  'bar' ) #{'bar','contour'} #TODO: sampled
        self.errorbar_props     =       kws.pop('errorbar',     {} )
        self._set_defaults(self.errorbar_props, self._default_errorbar)
        #self.error_contours     =       kws.pop( 'error_contours',     False )

        show_masked             =       kws.pop('show_masked',  False )#'spans' in kws
        self.span_props         =       kws.pop('spans',        {} )
        self._set_defaults(self.span_props, self._default_spans)

        self.hist_props         =       kws.pop('hist',           {} )
        self.show_hist          =       kws.pop('show_hist', bool(len(self.hist_props)) )
        self._set_defaults(self.hist_props, self._default_hist)

        draggable               =       kws.pop('draggable', True)
        ax                      =       kws.pop('ax',        None)
        #TODO: x,y,upper, lower
        whitefrac               =       kws.pop('whitefrac', self.whitefrac)
        whitefrac = np.atleast_1d(whitefrac)
        whitefrac = np.r_[whitefrac, whitefrac] if len(whitefrac)==1 else whitefrac[:2]

        #Upper time axis display
        #NOTE: this might get messy! Consider setting up the axes outside and passing
        self.twinx              =       kws.pop('twinx',           None )
        self.timescale          =       kws.pop('timescale',       's' )
        self.start              =       kws.pop('start',        None )

        legend_kw               =       kws.pop('legend_kw',        {} )
        offsets                 =       kws.pop('offsets',        None )

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #check for structured data
        l = [isinstance(d, dict) for d in data]
        if any(l):
            dgen = (list(d.values()) if ll else d  for d, ll in zip(data, l))
            keys = [tuple(data[i].keys()) for i in np.where(l)[0]]
            data = tuple(dgen)
            if len(keys) > 1:
                assert keys[0] == keys[1], "dict keys don't match"
            if labels is None:
                labels = keys[0]

        #parse data args
        Times, Rates, Errors = self.get_data(data)
        N, _ = Rates.shape

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #check if single label specified
        if labels is None:
            labels = []
        if isinstance(labels, str):
            labels = [labels]
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #Ensure we plot with unique colours

        if ((not cmap is None) #colour map given - superceeds colours kw
        or ((not len(colours)) and len(rcParams['axes.prop_cycle']) < N)):
            cm =  plt.get_cmap(kws.get('colormap', 'spectral'))
            colours =  cm(np.linspace(0, 1, N))

        elif len(colours) < N:
            'Given colour sequence less than number of time series. Colours will repeat'

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #Do the plotting
        fig, ax = self.setup_figure_geometry(ax, colours)

        plots, masked_plots  = [], []
        _linked = []

        #NOTE: masked array behaves badly in mpl<1.5.
        #see: https://github.com/matplotlib/matplotlib/issues/5016/
        #Rates = Rates.filled(np.nan)

        for i, (t, rate, stddev, label) in enumerate(itt.zip_longest(
                Times, Rates, Errors, labels)):  #zip_longest in case no Errors are given

            #catch in case for some reason the user provides a sequence of empty
            #error sequences
            if not (show_errors and np.size(stddev)):
                stddev = None          #ax.errorbar borks for empty error sequences

            if (stddev is not None) & (show_errors == 'contour'):
                self.uncertainty_contours(ax, t, rate, stddev, lw=1)

            #embed()
            pl = ax.errorbar(t, rate, stddev,
                             label=label,
                             **self.errorbar_props)
            plots.append(pl)

            #Histogram
            if self.show_hist:
                self.plot_histogram(rate, **self.hist_props)

            #Get / Plot GTIs
            if show_masked:
                valid = self._vall[i]
                unmasked = rate.copy()[:valid]
                tum = t[:valid]

            if show_masked == 'span':
                self.plot_masked_intervals(ax, tum, unmasked.mask)

            elif show_masked == 'x':
                #print( np.where(unmasked.mask) )
                unmasked.mask = ~unmasked.mask

                #NOTE: using errorbar here so we can easily convert to DraggableErrorbarContainers
                #Can fix this once DraggableLines are supported
                mp = ax.errorbar(tum, unmasked, color=pl[0].get_color(),
                                marker='x', ls='None',
                                label='_nolegend_')
                masked_plots.append(mp)
                plots.append(mp)
                _linked.append((pl, mp))

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self._set_labels( ax, title, axlabels )
        self._set_axes_limits(ax, Times, Rates, Errors, whitefrac, offsets)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #Setup plots for canvas interaction
        if draggable and not self.show_hist:
            #make the artists draggable
            from draggables import DraggableErrorbar

            lkw = self._default_legend.copy()
            lkw.update(legend_kw)
            plots = DraggableErrorbar(plots, offsets=offsets, linked=_linked,
                                      **lkw)
            #TODO: legend with linked plots!
        else:
            self._make_legend( ax, plots, labels )

        if hist:
            return fig, plots, Times, Rates, Errors, self.Hist
        else:
            return fig, plots, Times, Rates, Errors

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def uncertainty_contours(self, ax, t, rate, stddev, **kws):
        #NOTE: interpret uncertainties as stddev of distribution
        from tsa.tsa import smoother
        c, = ax.plot(t, smoother(rate + 3 * stddev), **kws)
        colour = self.errorbar_props['color'] = c.get_color()   #preserve colour cycle
        ax.plot(t, smoother(rate - 3*stddev), colour, **kws)

   #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #TODO:  handle data of different lenghts!!!!!
    def get_data(self, data):
        '''parse data arguments'''

        #rates only
        if len(data)==1:
            Rates = np.ma.asarray(data[0])      #Assume each row gives rates for TS
            Times = []                          #No time given, plot by array index
            #NOTE: Rates here may be non-uniform, so we will set times when we blockify
            Errors = []                         #No errors given

        #times & rates given
        elif len(data)==2:                        #No errors given
            Times, Rates = data
            Errors = []

        #times, rates, errors given
        elif len(data)==3:
            Times, Rates, Errors = data

        else:
            raise ValueError('Invalid number of arguments: {}'.format(len(data)))

        #Convert to masked and remove unwanted dimensionality
        Rates = np.ma.asarray(Rates).squeeze()
        Times = np.ma.asarray(Times).squeeze()
        Errors = np.array(Errors).squeeze()

        #at this point the data might be a masked_array of arrays (of non-uniform length)
        #support for non-uniform data length

        #print(Rates, Times, Errors)

        Times, Rates, Errors  = self.blockify(Times, Rates, Errors)

        #print( Rates, Times )
        #print( Rates.shape, Times.shape )
        #print( Rates.size, Times.size )

        self.t0 = 0.
        #NOTE: relative time is ambiguous when multiple time sequences are given
        #we will plot relatively to the minimum of all times
        if self.relative_time:
            self.t0 = np.floor(Times[:,0].min())
            Times -= self.t0

        return Times, Rates, Errors

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def blockify(self, Times, Rates, Errors):
        '''support for non-uniform data length'''

        uniform, L = self.is_uniform(Rates)
        self._vall = L

        if not uniform:
            #deal with non-uniform data
            Rates = self._blockify(Rates, L)            #TODO: IS THIS REALLY NECESSARY??

        #at this point Rates is uniform
        #Times might be non-uniform nD or uniform 1D.  Will blockify below
        if np.size(Times):
            #case for time nD, rates nD (n in (1,2))
            #NOTE: times might be 1D containing non-uniform arrays
            uniform, tL = self.is_uniform(Times)
            assert np.equal(tL, L).all()

            if not uniform:
                Times = self._blockify(Times, L)        #TODO: IS THIS REALLY NECESSARY??

            elif Times.ndim == 1:       #case uniform 1D time array
                #Assume same times for each sequence of rates
                Times = self.chronify(Rates, Times)
        else:
            #case for time 0D, rates nD uniform
            Times = self.chronify(Rates) #no time array given. use index grid as "time"

        #do the same for the errors
        if np.size(Errors):
            uniform, eL = self.is_uniform(Errors)
            assert np.equal(eL, L).all()
            #embed()
            if not uniform:
                Errors = self._blockify(Errors, L)      #TODO: IS THIS REALLY NECESSARY??

            assert Errors.ndim == Rates.ndim

        #ensure 2D
        Rates = np.ma.atleast_2d(Rates)
        Times = np.atleast_2d(Times)
        Errors = np.atleast_2d(Errors)

        #if Rates.shape != Times.shape:
        #embed()

        #mask nans
        Rates = np.ma.masked_where(np.isnan(Rates), Rates)
        Errors = np.ma.masked_where(np.isnan(Errors), Errors)


        return Times, Rates, Errors

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @staticmethod
    def is_uniform(data):
        '''Determine if data are of uniform length'''
        try:
            L = list(map(len, data))
        except TypeError:
            #data is legitimately 1D
            L = len(data)
            return True, L
        else:
            uni = np.allclose(L, L[0])
            return uni, L

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @staticmethod
    def chronify(Rates, Times=None):
        '''impute time data'''
        if Times is None:
            return grid_like(Rates)[int(Rates.ndim > 1)] #Times = #NOTE: may be 1D
            #return Times
        else:
            nTimes = np.empty_like(Rates)
            nTimes[:] = Times                   #duplicates the Times
            return nTimes

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @staticmethod
    def _blockify(data, L):
        #data has non-uniform lengths
        lR = len(data)
        mxL = max(L)

        #allocate memory
        ndata = np.ma.array(np.empty((lR, mxL)), mask=False)
        for i in range(lR):
            #mask everything beyond original data size
            ndata[i, :L[i]] = data[i]
            ndata.mask[i, L[i]:] = True

        return ndata

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #def _get_ax(self, fig):
        #if self.twin =='sexa':
            #ax = SexaTimeDualAxes(fig, 1, 1, 1)
            #ax.setup_ticks()
        #else:

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def setup_figure_geometry(self, ax, colours):
        #FIXME:  leave space on the right of figure to display offsets
        '''Setup figure geometry'''

        #TODO:  AUTOMATICALLY DETERMINE THESE VALUES!!
        rect = left, bottom, right, top = [0.025, 0.01, 0.97, .98]

        if ax is None:
            if self.twinx =='sexa':
                from .dualaxes import DateTimeDualAxes
                from astropy import units as u
                from astropy.coordinates.angles import Angle

                #TODO: support for longer scale time JD, MJD, etc...

                #h,m,s = Angle((self.t0 * u.Unit(self.timescale)).to(u.h)).hms       #start time in (h,m,s)
                #hms = int(h), int(m), int(round(s))                  #datetime hates floats# although the seconds should be integer, (from np.floor for self.t0) the conversion might have some floating point error
                ##FIXME: self.start might be None
                ##if self.start is None:

                #ymd = tuple(map(int, self.start.split('-')))
                #start = ymd + hms

                #print('MOTHERFUCKER!!!!!!!!!!!')
                #print( start )
                start= self.start

                fig = plt.figure(figsize=(18,8))
                ax = DateTimeDualAxes(fig, 1, 1, 1,
                                  timescale=self.timescale,
                                  start=start)
                ax.setup_ticks()
                fig.add_subplot(ax)
                top = .94       #need extra space for the tick labels
            else:
                fig, ax  = plt.subplots(figsize=(18,8))

        else:
            fig = ax.figure
            #ax.set_color_cycle( colours )

        #Add subplot for histogram
        if self.show_hist:
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(ax)
            self.hax = divider.append_axes('right', size='25%', pad=0., sharey=ax)
            self.hax.grid()
            self.hax.yaxis.tick_right()
            #ax.set_color_cycle(colours)
        else:
            self.hax = None         #TODO: can be at init OR NullObject


        #NOTE: #mpl >1.4 only
        if len(colours):
            from cycler import cycler
            ccyc = cycler('color', colours)
            ax.set_prop_cycle(ccyc)
            if self.show_hist:
                self.hax.set_prop_cycle(ccyc)


        fig.tight_layout(rect=[left, bottom, right, top])
        ax.grid(b=True, which='both')

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

        from recipes.iter import interleave

        w, = np.where(mask)
        l1 = w - np.roll(w,1) > 1
        l2 = np.roll(w,-1) -w > 1
        idx = [w[0]] + interleave(w[l2], w[l1]) + [w[-1]]
        return a[idx].reshape(-1,2)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def plot_histogram(self, rate, **props):
        #print( 'Plotting H' )

        if np.ma.is_masked(rate):
            r = rate[~rate.mask]
        else:
            r = rate

        #embed()

        h = self.hax.hist(r, **props)
        self.Hist.append(h)

        self.hax.grid(True)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _set_labels(self, ax, title, axlabels):
        '''axis title + labels'''
        title_text = ax.set_title( title, fontweight='bold' )
        if self.twinx:
            title_text.set_position((0.5,1.09))         #make space for the tick labels

        ax.set_xlabel(axlabels[0] if len(axlabels)      else 't (s)' )
        ax.set_ylabel(axlabels[1] if len(axlabels)==2   else 'Counts/s')
        if len(axlabels)==3 and self.twinx:
            ax.parasite.set_xlabel(axlabels[2])

        if self.relative_time:
            ax.xoffsetText = ax.text( 1, ax.xaxis.labelpad,
                                      '[{:+.1f}]'.format(self.t0),
                                      ha='right',
                                      transform=ax.xaxis.label.get_transform() )

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @staticmethod
    def _set_axes_limits(ax, X, Y, E, whitefrac, offsets):
        '''Axes limits'''

        xfrac, yfrac = whitefrac
        xlims = get_axlim(X, xfrac)
        ylims = get_axlim(Y, yfrac, E)
        if not offsets is None:
            ylims = np.add(ylims, (min(offsets), max(offsets)))

        ax.set_xlim(xlims)
        ax.set_ylim(ylims)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _make_legend(self, ax, plots, labels):
        '''Legend'''

        #print( labels, '!'*10 )

        if len(labels):
            if self.mask_shown:
                from matplotlib.patches import Rectangle
                span_label = self.span_props.pop( 'label' )
                r = Rectangle( (0, 0), 1, 1, **self.span_props )   #span proxy artist for legend

                plots += [r]
                labels+= [span_label]

            #from misc import make_ipshell
            #ipshell = make_ipshell()
            #ipshell()

            #print( plots, labels )

            ax.legend( plots, labels,  **self._default_legend)


#****************************************************************************************************
#NOTE: you can probs use the std plt.subplots machinery if you register your axes classes

def time_phase_plot(P, toff=0):
    from matplotlib.transforms import Affine2D
    from .dualaxes import DualAxes
    fig = plt.figure()
    aux_trans = Affine2D().translate(-toff, 0).scale(P)
    ax = DualAxes(fig, 1,1,1, aux_trans=aux_trans)
    ax.setup_ticks()
    ax.parasite.set_xlabel('Orbital Phase')
    fig.add_subplot(ax)
    return fig, ax


def plot_folded_lc(ax, phase, lcdata, P_s, twice=True, orientation='h'):
    '''plot folded lc mean/max/min/std'''

    lcmean, lcmn, lcmx, lcstd = np.tile(lcdata, (twice+1))
    if twice:
        phase = np.r_[phase, phase+1]

    t = phase * P_s
    lcerr = lcmean + lcstd * 1. * np.c_[1,-1].T

    linevars = (lcmean, lcmn, lcmx)
    colours = ('b', '0.5', '0.5')
    args = zip(itt.repeat(t), linevars)
    if orientation.startswith('v'):
        args = map(reversed, args)
        fill_between = ax.fill_betweenx
    else:
        fill_between = ax.fill_between


    lines = []
    for args, colour in zip(args, colours):
        pl, = ax.plot(*args, color=colour, lw=1)
        lines.append(pl)
    plm, plmn, plmx = lines

    fill_between(t, *lcerr, color='grey')

    #ax.hlines(phbins, *ax.get_ylim(), color='g', linestyle='--')
    ax.grid()
    if orientation.startswith('h'):
        ax.set_xlim(0, (twice+1)*P_s)
        ax.set_xlabel('t (s)')
    else:
        ax.set_ylim(0, (twice+1)*P_s)
        ax.set_ylabel('t (s)')

    r = Rectangle( (0, 0), 1, 1, fc='grey', ec='none')            #rectangle proxy art for legend.
    leg = ax.legend((plm, plmn, r), ('mean', 'extrema', r'$1\sigma$'))

    ax.figure.tight_layout()
    #return fig




#====================================================================================================


#====================================================================================================
#initialise
lcplot = LCplot()
#====================================================================================================

if __name__ == '__main__':
    '''tests'''
    from decor.profile import profile
    profiler = profile()
    @profiler.histogram
    def tests(**kws):
        #generate some data
        N = 250
        np.random.seed(666)
        t = np.linspace(0, 2*np.pi, N)
        y = [np.sin(3*t),
            #np.cos(10*t),
            np.cos(10*np.sqrt(t))]
        e = np.random.randn(len(y), N)
        m = np.random.rand(len(y), N) > 0.8

        ##try:
        #case 1:    bare minimum
        fig, plots, *stuff = lcplot(y[0], **kws )

        #case 1:    multiple series, no time
        fig, plots, *stuff = lcplot(y, **kws)

        #case 1:    multiple series, single time
        fig, plots, *stuff = lcplot(t, y, **kws)

        #case 2:    full args
        t2 = np.power(t, np.c_[1:len(y)+1])  #power stretch time
        fig, plots, *stuff = lcplot(t2, y, e, **kws)

        #case 3: masked data
        ym = np.ma.array(y, mask=m)
        fig, plots, *stuff = lcplot(t, ym, e,
                                    show_masked='x',
                                    **kws)

        #case 4: masked data
        #mask = [(np.pi/4*(1+i) < t)&(t > np.pi/8*(1+i)) for i in range(len(y))]
        #ym = np.ma.array(y, mask=mask)
        #fig, plots, *stuff = lcplot(t, ym, e,
                                    #show_masked='span')
        #FIXME:
        #File "/home/hannes/.local/lib/python3.4/site-packages/draggables/errorbars.py", line 257, in __init__
        #self.to_orig[handel.markers] = NamedErrorbarContainer(origart)
        #AttributeError: 'Rectangle' object has no attribute 'markers'

        #case 4: non-uniform data
        y2 = [_[:np.random.randint(int(0.7*N), N)] for _ in y]
        #fig, plots, *stuff = lcplot(y2)

        t2 = [t[:len(_)] for _ in y2]
        fig, plots, *stuff = lcplot(t2, y2, **kws)

        e2 = [np.random.randn(len(_)) for _ in y2]
        fig, plots, *stuff = lcplot(t2, y2, e2, **kws)

            #case: non-uniform masked data
        #except Exception as err:
            ##plt.show()
            #raise err

        #TODO: more tests

    #tests()
    #everything with histogram #NOTE: significantly slower
    tests(show_hist=True)
    plt.show()