"""
Versatile functions for plotting time-series data
"""

import itertools as itt

import numpy as np
from matplotlib import rcParams
import matplotlib.pyplot as plt
# import colormaps as cmaps
# plt.register_cmap(name='viridis', cmap=cmaps.viridis)

from recipes.array import ndgrid
from recipes.dict import AttrDict


# from recipes.string import minlogfmt

from IPython import embed

# ****************************************************************************************************
class TSplotter(object):        # TODO: evolve to multiprocessed TS plotter
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # TODO: Keyword translation!

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    figsize = (18, 8)
    maxpoints = 1e4  # TODO

    # Set parameter defaults
    defaults = AttrDict(
        labels=None,
        title='',
        #
        relative_time=False,
        timescale='s',
        start=None,

        # NOTE: this might get messy! Consider setting up the axes outside and passing
        ax=None,
        axlabels=(('t (s)', ''),  # bottom and top x-axes       # TODO: axes_labels
                  'Counts/s'),
        # TODO: axes_label_position: (left, right, center, <,>,^)
        twinx=None,  # Top x-axis display format
        xscale=None,
        yscale=None,
        #
        colours=[],
        cmap=None,  # TODO: colormap
        #
        show_errors='bar',
        show_masked=False,
        show_hist=False,

        draggable=True,
        offsets=None,

        whitefrac=0.025)  # TODO: x,y,upper, lower

    # Default options for plotting related stuff
    default_opts = AttrDict(
        errorbar=dict(fmt='o',  # TODO: sampled lcs from distribution implied by errorbars
                      ms=2.5,
                      mec='none',
                      capsize=0,
                      elinewidth=0.5),
        spans=dict(label='filtered',
                   alpha=0.2,
                   color='r'),
        hist=dict(bins=50,
                  alpha=0.75,
                  # color='b',
                  orientation='horizontal'),
        legend=dict(loc='upper right',  # TODO: option for no legend
                    fancybox=True,
                    framealpha=0.25,
                    numpoints=1,
                    markerscale=3)
    )

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @staticmethod
    def _set_defaults(props, defaults):
        for k, v in defaults.items():
            props.setdefault(k, v)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def check_kws(self, kws):
        allowed_kws = list(self.defaults.keys()) + list(self.default_opts.keys())
        opts = self.defaults.copy()
        dopts = self.default_opts.copy()
        for key, val in kws.items():
            if not key in allowed_kws:
                raise KeyError('Invalid keyword argument {}.\n'
                               'Only the following keywords are recognised: {}'
                               ''.format(key, allowed_kws))
            if key in dopts:
                dopts[key].update(val)
        opts.update(kws)

        return opts, dopts

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __call__(self, *data, **kws):  # def __init__(self):
        """
        Plot light curve(s)

        Parameters
        ----------
        data        :   tuple of array-likes
            (signal,)   -   in which case t is infered
            (t, signal) -   in which case uncertainty is ignored
            (t, signal, uncertainty)
        t           :   (optional; array-like or dict)
            time sequence of corresponding data
        signal      :   (array-like or dict)
            time series data values
        uncertainty :   (optional; array-like or dict)
            standard deviation uncertainty associated with signal

        """
        # TODO: docstring
        # TODO: astropy.units ??
        # TODO: max points = 1e4 ??

        self.mask_shown = False
        self.Hist = []

        # Check keyword argument validity
        opts, dopts = self.opts, self.dopts = self.check_kws(kws)

        self.show_hist = kws.pop('show_hist', bool(len(self.opts.get('hist', {}))))

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # check for structured data (dict keyed on labels and containing data)
        labels = self.opts.labels
        l = [isinstance(d, dict) for d in data]
        if any(l):
            dgen = (list(d.values()) if ll else d for d, ll in zip(data, l))
            keys = [tuple(data[i].keys()) for i in np.where(l)[0]]
            data = tuple(dgen)
            if len(keys) > 1:
                assert keys[0] == keys[1], "dict keys don't match"
            if labels is None:
                labels = keys[0]

        # parse data args
        Times, Signal, Errors = data = self.get_data(data)
        N, _ = Signal.shape

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # check labels
        if labels is None:
            labels = []
        if isinstance(labels, str):
            labels = [labels]  # check if single label given
        labels = labels[:N]

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Ensure we plot with unique colours
        # cmap, colours = self.opts.cmap, self.opts.colours
        if ((not opts.cmap is None)  # colour map given - superceeds colours kw
            or ((not len(opts.colours)) and len(rcParams['axes.prop_cycle']) < N)):
            cm = plt.get_cmap(kws.get('colormap', 'nipy_spectral'))
            opts.colours = cm(np.linspace(0, 1, N))  # linear colour map for ts

        elif len(opts.colours) < N:
            'Given colour sequence less than number of time series. Colours will repeat'

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        fig, ax = self.setup_figure_geometry(opts.ax, opts.colours)

        # Do the plotting
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        plots = []
        _linked = []

        # NOTE: masked array behaves badly in mpl < 1.5.
        # see: https://github.com/matplotlib/matplotlib/issues/5016/
        # Signal = Signal.filled(np.nan)

        for i, (t, signal, stddev, label) in enumerate(itt.zip_longest(
                Times, Signal, Errors, labels)):  # zip_longest in case no Errors are given
            ebplt = self._plot_ts(ax, t, signal, stddev, label, opts.show_errors)
            plots.append(ebplt)

            if opts.show_masked:
                col = ebplt[0].get_color()
                mskplt = self._plot_masked(ax, i, t, signal, col, opts.show_masked)
                if mskplt:
                    plots.append(mskplt)
                    _linked.append((ebplt, mskplt))

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        axes_labels = self.get_labels(opts.axlabels)
        self._set_labels(ax, opts.title, *axes_labels)
        xlims, ylims = self.get_axes_limits(data)
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        # self._set_axes_limits(ax,

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Setup plots for canvas interaction
        if opts.draggable and not self.show_hist:
            # make the artists draggable
            from draggables import DraggableErrorbar
            plots = DraggableErrorbar(plots, offsets=opts.offsets, linked=_linked,
                                      **dopts.legend)
            # TODO: legend with linked plots!
        else:
            self._make_legend(ax, plots, labels)

        if self.show_hist:
            return fig, plots, Times, Signal, Errors, self.Hist
        else:
            return fig, plots, Times, Signal, Errors

    # alias the call method
    plot = __call__

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _plot_ts(self, ax, t, signal, stddev, label, show_errors):
        # catch in case for some reason the user provides a sequence of empty error sequences
        if not (show_errors and np.size(stddev)):
            stddev = None  # ax.errorbar borks for empty error sequences

        if (stddev is not None) & (show_errors == 'contour'):
            self.uncertainty_contours(ax, t, signal, stddev, lw=1)

        # main plot
        ebpl = ax.errorbar(t, signal, stddev,
                           label=label,
                           **self.dopts.errorbar)

        # Histogram
        if self.show_hist:
            self.plot_histogram(signal, **self.dopts.hist)

        return ebpl

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _plot_masked(self, ax, i, t, signal, colour, how):
        # Get / Plot GTIs

        valid = self._original_sizes[i]
        unmasked = signal.copy()[:valid]
        tum = t[:valid]

        mskplt = None
        if how == 'span':
            self.plot_masked_intervals(ax, tum, unmasked.mask)

        elif how == 'x':
            # print( np.where(unmasked.mask) )
            unmasked.mask = ~unmasked.mask

            # NOTE: using errorbar here so we can easily convert to DraggableErrorbarContainers
            # FIXME: Can fix this once DraggableLines are supported
            mskplt = ax.errorbar(tum, unmasked, color=colour,
                                 marker='x', ls='None', alpha=0.7,
                                 label='_nolegend_')

        return mskplt

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def uncertainty_contours(self, ax, t, signal, stddev, **kws):
        # NOTE: interpret uncertainties as stddev of distribution
        from tsa.tsa import smoother
        c, = ax.plot(t, smoother(signal + 3 * stddev), **kws)
        colour = self.dopts.errorbar['color'] = c.get_color()  # preserve colour cycle
        ax.plot(t, smoother(signal - 3 * stddev), colour, **kws)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # TODO:  handle data of different lenghts!!!!!
    def get_data(self, data):
        """parse data arguments"""

        # signals only
        if len(data) == 1:
            Signal = np.ma.asarray(data[0])  # Assume each row gives signals for TS
            Times = []  # No time given, plot by array index
            # NOTE: Signal here may be non-uniform, so we will set times when we blockify
            Errors = []  # No errors given

        # times & signals given
        elif len(data) == 2:  # No errors given
            Times, Signal = data
            Errors = []

        # times, signals, errors given
        elif len(data) == 3:
            Times, Signal, Errors = data

        else:
            raise ValueError('Invalid number of arguments: {}'.format(len(data)))

        # Convert to masked and remove unwanted dimensionality
        Signal = np.ma.asarray(Signal).squeeze()
        Times = np.ma.asarray(Times).squeeze()
        Errors = np.array(Errors).squeeze()

        # print('FUCK YOU 1 ' * 10)
        # embed()


        # at this point the data might be a masked_array of arrays (of non-uniform length)
        # support for non-uniform data length
        Times, Signal, Errors = self.blockify(Times, Signal, Errors)
        # print( Signal, Times )
        # print( Signal.shape, Times.shape )
        # print( Signal.size, Times.size )

        self.t0 = 0.
        # NOTE: relative time is ambiguous when multiple time sequences are given
        # we will plot relatively to the minimum of all times
        if self.opts.relative_time:
            self.t0 = np.floor(Times[:, 0].min())
            Times -= self.t0

        return Times, Signal, Errors

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def blockify(self, Times, Signal, Errors):
        """support for non-uniform data length"""

        uniform, L = self.is_uniform(Signal)
        self._original_sizes = L

        if not uniform:
            # deal with non-uniform data
            Signal = self._blockify(Signal, L)  # TODO: IS THIS REALLY NECESSARY??

        # at this point Signal is uniform
        # Times might be non-uniform nD or uniform 1D.  Will blockify below
        if np.size(Times):
            # case for time nD, signals nD (n in (1,2))
            # NOTE: times might be 1D containing non-uniform arrays
            uniform, tL = self.is_uniform(Times)
            assert np.equal(tL, L).all()

            if not uniform:
                Times = self._blockify(Times, L)  # TODO: IS THIS REALLY NECESSARY??

            elif Times.ndim == 1:  # case uniform 1D time array
                # Assume same times for each sequence of signals
                Times = self.chronify(Signal, Times)
        else:
            # case for time 0D, signals nD uniform
            Times = self.chronify(Signal)  # no time array given. use index grid as "time"

        # embed()

        # try:
        # do the same for the errors
        if np.size(Errors):
            uniform, eL = self.is_uniform(Errors)
            # FIXME: Errors might be shorter that rates
            assert np.equal(eL, L).all()
            # embed()
            if not uniform:
                Errors = self._blockify(Errors, L)
                # FIXME: IS THIS REALLY NECESSARY??
                # FIXME: probs quite inefficient when eg. 2 ts with very different lengths are given

            assert Errors.ndim == Signal.ndim
        # except Exception as err:
        #     embed()
        #     raise err

        # ensure 2D
        Signal = np.ma.atleast_2d(Signal)
        Times = np.atleast_2d(Times)
        Errors = np.atleast_2d(Errors)

        # if Signal.shape != Times.shape:
        # embed()

        # mask nans
        Signal = np.ma.masked_where(np.isnan(Signal), Signal)
        Errors = np.ma.masked_where(np.isnan(Errors), Errors)

        return Times, Signal, Errors

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @staticmethod
    def _blockify(data, L):

        # data has non-uniform lengths
        lR = len(data)
        mxL = max(L)

        # allocate memory
        ndata = np.ma.array(np.empty((lR, mxL)), mask=False)
        for i in range(lR):
            # mask everything beyond original data size
            ndata[i, :L[i]] = data[i]
            ndata.mask[i, L[i]:] = True

        return ndata

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @staticmethod
    def is_uniform(data):
        """Determine if data arrays are of uniform length"""
        try:
            L = list(map(len, data))
        except TypeError:
            # data are legitimately 1D
            L = len(data)
            return True, L
        else:
            uni = np.allclose(L, L[0])
            return uni, L

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @staticmethod
    def chronify(Signal, Times=None):
        """impute time data"""
        if Times is None:
            return ndgrid.like(Signal)[int(Signal.ndim > 1)]  # Times = #NOTE: may be 1D
            # return Times
        else:
            nTimes = np.empty_like(Signal)
            nTimes[:] = Times  # duplicates the Times
            return nTimes

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # def _get_twin(self, fig):
            # if self.twin ==
            # 'sexa':DateTimeDualAxes
            #
            # else:

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def get_axes(self, ax):

        if ax is not None:
            return ax.figure, ax

        if not self.opts.twinx:
            return plt.subplots(figsize=self.figsize)

        # TODO: a factory that gets the appropriate class and keyword
        if self.opts.twinx == 'sexa':
            from .dualaxes import DateTimeDualAxes
            # from astropy import units as u
            # from astropy.coordinates.angles import Angle

            # TODO: support for longer scale time JD, MJD, etc...

            # h,m,s = Angle((self.t0 * u.Unit(self.timescale)).to(u.h)).hms       #start time in (h,m,s)
            # hms = int(h), int(m), int(round(s))                  #datetime hates floats# although the seconds should be integer, (from np.floor for self.t0) the conversion might have some floating point error
            ##FIXME: self.start might be None
            ##if self.start is None:

            # ymd = tuple(map(int, self.start.split('-')))
            # start = ymd + hms

            fig = plt.figure(figsize=self.figsize)
            ax = DateTimeDualAxes(fig, 1, 1, 1,
                                  timescale=self.opts.timescale,
                                  start=self.opts.start)
            ax.setup_ticks()
            fig.add_subplot(ax)

        else:
            # TODO: centralise error handling
            raise NotImplementedError('Option %s not understood' % self.opts.twinx)

        return fig, ax

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def setup_figure_geometry(self, ax, colours):
        # FIXME:  leave space on the right of figure to display offsets
        """Setup figure geometry"""

        rect = left, bottom, right, top = [0.025, 0.01, 0.97, .98]  # TODO:  AUTOMATICALLY DETERMINE THESE VALUES!!
        if self.opts.twinx:
            top = .94  # need extra space for the top tick labels

        fig, ax = self.get_axes(ax)

        # Add subplot for histogram
        if self.show_hist:
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(ax)
            self.hax = divider.append_axes('right', size='25%', pad=0., sharey=ax)
            self.hax.grid()
            self.hax.yaxis.tick_right()
            # ax.set_color_cycle(colours)
        else:
            self.hax = None  # TODO: can be at init OR NullObject

        # NOTE: #mpl >1.4 only
        if len(colours):
            from cycler import cycler
            ccyc = cycler('color', colours)
            ax.set_prop_cycle(ccyc)
            if self.show_hist:
                self.hax.set_prop_cycle(ccyc)

        fig.tight_layout(rect=[left, bottom, right, top])
        ax.grid(b=True, which='both')

        return fig, ax

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def plot_masked_intervals(self, ax, t, mask):
        """Highlight the masked values within the time series with a span accross the axis"""
        spans = convert_mask_to_intervals(t, mask)
        for s in spans:
            ax.axvspan(*s, **self.dopts.spans)

        self.mask_shown = True  # bool(bti)         #just so we don't make a legend entry for this if it's empty

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def plot_histogram(self, signal, **props):
        # print( 'Plotting H' )

        if np.ma.is_masked(signal):
            r = signal[~signal.mask]
        else:
            r = signal

        h = self.hax.hist(r, **props)
        self.Hist.append(h)

        self.hax.grid(True)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def get_labels(self, axlabels):
        if len(axlabels) == 0:
            xlabels, ylabel = self.opts.axlabels
        elif len(axlabels) == 2:
            xlabels, ylabel = axlabels
            if isinstance(xlabels, str):
                xlabels = (xlabels, '')
        else:
            raise ValueError

        return xlabels, ylabel

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _set_labels(self, ax, title, xlabels, ylabel):
        """axis title + labels"""
        title_text = ax.set_title(title, fontweight='bold')

        xlb, xlt = xlabels
        ax.set_xlabel(xlb)
        if self.opts.twinx:
            title_text.set_position((0.5, 1.09))  # make space for the tick labels
            if xlt:
                ax.parasite.set_xlabel(xlt)
        ax.set_ylabel(ylabel)

        # display time offset
        if self.opts.relative_time:
            ax.xoffsetText = ax.text(1, ax.xaxis.labelpad,
                                     '[{:+.1f}]'.format(self.t0),
                                     ha='right',
                                     transform=ax.xaxis.label.get_transform())

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def get_axes_limits(self, data):
        """Axes limits"""
        X, Y, E = data

        # Fractional 'white space' in figure
        whitefrac = np.atleast_1d(self.opts.whitefrac)
        xfrac, yfrac = np.r_[whitefrac, whitefrac] if len(whitefrac) == 1 else whitefrac[:2]
        xl, xu = axes_limit_from_data(X, xfrac)
        yl, yu = ylims = axes_limit_from_data(Y, yfrac, E)

        offsets = self.opts.offsets
        if not offsets is None:
            yl, yu = np.add(ylims, (min(offsets), max(offsets)))

        if (self.opts.yscale == 'log') and (yl <= 0):
            yl = np.sort(Y[Y > 0])[0]  # nonzeromin
            # TODO: WARN
            # TODO: indicate more data points with arrows???????????????????????????
            #       : Would be cool if done while hovering mouse on legend
            # yl = max(Y.min(), nonzeromin)

        if (self.opts.xscale == 'log') and (xl <= 0):
            xl = np.sort(X[X > 0])[0]  # nonzeromin
            # xl = max(X.min(), nonzeromin)

        return (xl, xu), (yl, yu)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _make_legend(self, ax, plots, labels):
        """Legend"""

        # print( labels, '!'*10 )

        if len(labels):
            if self.mask_shown:
                from matplotlib.patches import Rectangle
                span_label = self.span_props.pop('label')
                r = Rectangle((0, 0), 1, 1, **self.span_props)  # span proxy artist for legend

                plots += [r]
                labels += [span_label]

            ax.legend(plots, labels, **self.dopts.legend)


# ****************************************************************************************************
def axes_limit_from_data(x, whitefrac, *e):
    """
    Return suggested axis limits based on the extrema of x, and the desired
    fractional whitespace on plot.
    whitespace  - can be number, or (bottom, top) tuple
    e - uncertainty
        can be either single array of same shape as x, or 2 arrays (upper, lower)
    """
    if np.size(e) == 0:
        el = eu = 0
    elif len(e) == 1:
        el = eu = e
    elif len(e) == 2:
        el, eu = e
    else: #basically ignoring any weird e values here
        el = eu = 0

    #TODO: deal with infs?
    try:
        xl, xu = np.nanmin(x - el), np.nanmax(x + eu)
        xd = xu - xl
    except Exception as err:
        # from IPython import embed
        # embed()
        print('axes_limit_from_data FAIL.:', str(err))
        return x.max(), x.min()

    wf = np.empty(2)
    wf[:] = whitefrac
    wxd = wf * xd
    return xl - wxd[0], xu + wxd[1]


# ====================================================================================================
def convert_mask_to_intervals(a, mask=None):
    """Return index tuples of contiguous masked values."""
    if mask is None:
        mask = a.mask  # NOTE: If a is a masked array, this function will return masked values!!!

    if ~np.any(mask):
        return ()

    from recipes.iter import interleave
    w, = np.where(mask)
    l1 = w - np.roll(w, 1) > 1
    l2 = np.roll(w, -1) - w > 1
    idx = [w[0]] + interleave(w[l2], w[l1]) + [w[-1]]
    return a[idx].reshape(-1, 2)


# ====================================================================================================


# ****************************************************************************************************
# NOTE: you can probs use the std plt.subplots machinery if you register your axes classes

def time_phase_plot(P, toff=0, **figkws):
    from matplotlib.transforms import Affine2D
    from .dualaxes import DualAxes
    fig = plt.figure(**figkws)


    aux_trans = Affine2D().translate(-toff, 0).scale(P)
    ax = DualAxes(fig, 1, 1, 1, aux_trans=aux_trans)
    ax.setup_ticks()
    ax.parasite.set_xlabel('Orbital Phase')
    fig.add_subplot(ax)

    return fig, ax


def plot_folded_lc(ax, phase, lcdata, P_s, twice=True, orientation='h'):
    """plot folded lc mean/max/min/std"""

    from matplotlib.patches import Rectangle

    lcmean, lcmn, lcmx, lcstd = np.tile(lcdata, (twice + 1))
    if twice:
        phase = np.r_[phase, phase + 1]

    t = phase * P_s
    lcerr = lcmean + lcstd * 1. * np.c_[1, -1].T

    linedata = (lcmean, lcmn, lcmx)
    colours = ('b', '0.5', '0.5')
    args = zip(itt.repeat(t), linedata)
    if orientation.startswith('v'):
        args = map(reversed, args)
        fill_between = ax.fill_betweenx
    else:
        fill_between = ax.fill_between

    #embed()

    lines = []
    for a, colour in zip(args, colours):
        # print(list(map(len, a)))
        pl, = ax.plot(*a, color=colour, lw=1)
        lines.append(pl)
    plm, plmn, plmx = lines

    fill_between(t, *lcerr, color='grey')

    # ax.hlines(phbins, *ax.get_ylim(), color='g', linestyle='--')
    ax.grid()
    if orientation.startswith('h'):
        ax.set_xlim(0, (twice + 1) * P_s)
        ax.set_xlabel('t (s)')
    else:
        ax.set_ylim(0, (twice + 1) * P_s)
        ax.set_ylabel('t (s)')

    r = Rectangle((0, 0), 1, 1, fc='grey', ec='none')  # rectangle proxy art for legend.
    leg = ax.legend((plm, plmn, r), ('mean', 'extrema', r'$1\sigma$'))

    ax.figure.tight_layout()
    # return fig


# ====================================================================================================


# ====================================================================================================
# initialise
lcplot = TSplotter()

# ====================================================================================================

if __name__ == '__main__':
    """tests"""
    from decor.profiler import profile

    profiler = profile()


    @profiler.histogram
    def tests(**kws):
        # generate some data
        N = 250
        np.random.seed(666)
        t = np.linspace(0, 2 * np.pi, N)
        y = [np.sin(3 * t),
             # np.cos(10*t),
             np.cos(10 * np.sqrt(t))]
        e = np.random.randn(len(y), N)
        m = np.random.rand(len(y), N) > 0.8

        ##try:
        # case 1:    bare minimum
        fig, plots, *stuff = lcplot(y[0], **kws)

        # case 2:    multiple series, no time
        fig, plots, *stuff = lcplot(y, **kws)

        # case 3:    multiple series, single time
        fig, plots, *stuff = lcplot(t, y, **kws)

        # case 4:    full args
        t2 = np.power(t, np.c_[1:len(y) + 1])  # power stretch time
        fig, plots, *stuff = lcplot(t2, y, e, **kws)

        # case 5: masked data
        ym = np.ma.array(y, mask=m)
        fig, plots, *stuff = lcplot(t, ym, e,
                                    show_masked='x',
                                    **kws)

        # case 4: masked data
        # mask = [(np.pi/4*(1+i) < t)&(t > np.pi/8*(1+i)) for i in range(len(y))]
        # ym = np.ma.array(y, mask=mask)
        # fig, plots, *stuff = lcplot(t, ym, e,
        # show_masked='span')
        # FIXME:
        # File "/home/hannes/.local/lib/python3.4/site-packages/draggables/errorbars.py", line 257, in __init__
        # self.to_orig[handel.markers] = NamedErrorbarContainer(origart)
        # AttributeError: 'Rectangle' object has no attribute 'markers'

        # #case 4: non-uniform data
        # y2 = [_[:np.random.randint(int(0.7*N), N)] for _ in y]
        # #fig, plots, *stuff = lcplot(y2)
        #
        # t2 = [t[:len(_)] for _ in y2]
        # fig, plots, *stuff = lcplot(t2, y2, **kws)
        #
        # e2 = [np.random.randn(len(_)) for _ in y2]
        # fig, plots, *stuff = lcplot(t2, y2, e2, **kws)

        # case: non-uniform masked data
        # except Exception as err:
        ##plt.show()
        # raise err

        # TODO: more tests


    # tests()
    # everything with histogram #NOTE: significantly slower
    tests(show_hist=True)
    plt.show()
