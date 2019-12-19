"""
Versatile functions for plotting time-series data
"""

# TODO:
#  MAJOR REFACTOR REQUIRED
#  illiminate singleton patters for TSplotter in favour of `plot_ts`
#  alternatively make `class timeseriesPlot(Axes):` then ax.plot_ts()

import itertools as itt

import numpy as np

import matplotlib as mpl

mpl.use('Qt5Agg')
# from matplotlib import rcParams

import matplotlib.pyplot as plt
# import colormaps as cmaps
# plt.register_cmap(name='viridis', cmap=cmaps.viridis)

from recipes.array import ndgrid
from recipes.dict import AttrDict

# from recipes.string import minlogfmt

from IPython import embed

# from dataclasses import dataclass

from attr import attrs, attrib as attr

from recipes.introspection.utils import get_module_name
import logging

logger = logging.getLogger(get_module_name(__file__))

# Set parameter defaults
default_cmap = 'nipy_spectral'
default_figsize = (14, 8)

defaults = AttrDict(
        labels=None,
        title='',
        #
        relative_time=False,
        timescale='s',
        start=None,

        # NOTE: this might get messy! Consider setting up the axes outside
        #  and always passing?
        ax=None,
        axlabels=(('t (s)', ''),
                  # bottom and top x-axes       # TODO: axes_labels
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

        whitespace=0.025)  # TODO: x, y, upper, lower

# Default options for plotting related stuff
default_opts = AttrDict(
        errorbar=dict(fmt='o',
                      # TODO: sampled lcs from distribution implied by
                      #  errorbars ?  simulate_samples
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

allowed_kws = list(defaults.keys())
allowed_kws.extend(default_opts.keys())


def _set_defaults(props, defaults):
    for k, v in defaults.items():
        props.setdefault(k, v)


def check_kws(kws):
    # these are AttrDict!
    opts = defaults.copy()
    dopts = default_opts.copy()

    invalid = set(allowed_kws) - set(kws.keys())
    if len(invalid):
        raise KeyError('Invalid keyword{}: {}.\n'
                       'Only the following keywords are recognised: {}'
                       ''.format('s' if len(invalid) > 1 else '',
                                 invalid, allowed_kws))

    for key, val in kws.items():
        # deal with keyword args for which values are dict
        if key in dopts:
            dopts[key].update(val)
    opts.update(kws)
    return opts, dopts


def get_data(data):
    """parse data arguments"""

    # signals only
    if len(data) == 1:
        # Assume here each row gives individual signal for a TS
        signal = data[0]
        times = []  # No time given, plot by array index
        errors = []  # No errors given

    # times & signals given
    elif len(data) == 2:  # No errors given
        times, signal = data
        errors = []

    # times, signals, errors given
    elif len(data) == 3:
        times, signal, errors = data

    else:
        raise ValueError('Invalid number of arguments: %i' % len(data))

    return times, signal, errors


def sanitize_data(t, signal, stddev, show_errors, relative_time):
    # catch in case for some reason the user provides a sequence of
    # empty error sequences
    # note ax.errorbar borks for empty error sequences

    if show_errors:
        if len(stddev) == 0:
            logger.warning('Ignoring empty uncertainties')
            stddev = None
        elif len(stddev) != len(signal):
            raise ValueError('Unequal number of points between data and '
                             'uncertainty arrays')
    else:
        logger.warning('Ignoring uncertainties since `show_errors = False`')
        stddev = None

    # plot by frame index if no time
    if t is None:
        t = np.arange(len(signal))
    elif relative_time:
        t -= t[0]

    # mask nans
    signal = np.ma.masked_where(np.isnan(signal), signal)
    stddev = np.ma.masked_where(np.isnan(stddev), stddev)
    # NOTE: masked array behaves badly in mpl < 1.5.
    # see: https://github.com/matplotlib/matplotlib/issues/5016/
    # signal = signal.filled(np.nan)
    return t, signal, stddev


def get_line_colours(n, colours, cmap):
    # Ensure we plot with unique colours

    # `cmap` superceeds `colours` kw
    prop_cycle_too_small = len(mpl.rcParams['axes.prop_cycle']) < n
    if (cmap is not None) or ((len(colours) == 0) and prop_cycle_too_small):
        cm = plt.get_cmap(cmap)
        colours = cm(np.linspace(0, 1, n))  # linear colour map for ts

    elif len(colours) < n:
        logger.warning('Colour sequence has too few colours (<%i).  Colours '
                       'will repeat')
    return colours


from obstools.phot.utils import duplicate_if_scalar


def get_axes_limits(data, whitespace, offsets=None):
    """Axes limits"""

    x, y, u = data
    xf, yf = duplicate_if_scalar(whitespace)  # Fractional white space in figure
    xl, xu = axes_limit_from_data(x, xf)
    yl, yu = axes_limit_from_data(y, yf, u)

    if offsets is not None:
        yl += min(offsets)
        yu += max(offsets)

    return (xl, xu), (yl, yu)


def axes_limit_from_data(x, whitespace, *e, ignore_large_errors=3.):
    """
    Return suggested axis limits based on the extrema of x, and the desired
    fractional whitespace on plot.
    whitespace  - can be number, or (bottom, top) tuple
    e - uncertainty
        can be either single array of same shape as x, or 2 arrays (δx+, δx-)
    """
    if np.size(e) == 0:
        el = eu = 0
    elif len(e) == 1:
        el = eu = e[0] / 2
    elif len(e) == 2:
        el, eu = e
    else:  # basically ignoring any weird e values here
        raise ValueError('Optional stddev (errorbar) values should be tuple '
                         'of size 1, 2 or 3.')

    # sanitize
    x = np.ma.MaskedArray(x, np.isnan(x) | np.isinf(x))
    xl, xu = x.min(), x.max()
    xd = xu - xl
    ed = eu - el

    # ignore errorbars > than `ignore_large_errors` * data peak to peak
    if np.any(ed > ignore_large_errors * xd):
        xu += eu
        xl -= el

    wxd = np.multiply(whitespace, xd)
    return xl - wxd[0], xu + wxd[1]


def get_axes_labels(axes_labels):
    if len(axes_labels) == 0:
        xlabels, ylabel = default_opts.axlabels
    elif len(axes_labels) == 2:
        xlabels, ylabel = axes_labels
        if isinstance(xlabels, str):
            xlabels = (xlabels, '')
    else:
        raise ValueError

    return xlabels, ylabel


def uncertainty_contours(ax, t, signal, stddev, styles, **kws):
    # NOTE: interpret uncertainties as stddev of distribution
    from tsa.tsa import smoother

    # preserve colour cycle
    sigma = 3
    c, = ax.plot(t, smoother(signal + sigma * stddev), **kws)
    colour = styles.errorbar['color'] = c.get_color()
    return ax.plot(t, smoother(signal - sigma * stddev), colour, **kws)


def plot(*data, **kws):
    """
    Plot light curve(s)

    Parameters
    ----------
    data        :   tuple of array-likes
        (signal,)   -   in which case t is implicitly the integers up to
                        len(signal)
        (t, signal) -   in which case uncertainty is ignored
        (t, signal, uncertainty)
    t           :   (optional; array-like or dict)
        time sequence of corresponding data
    signal      :   (array-like or dict)
        time series data values
    uncertainty :   (optional; array-like or dict)
        standard deviation uncertainty associated with signal

    """
    # FIXME: get this to work with astropy time objects
    # TODO: docstring
    # TODO: astropy.units ??
    # TODO: max points = 1e4 ??

    # Check keyword argument validity
    kws, styles = check_kws(kws)
    show_hist = kws.pop('show_hist', bool(len(kws.get('hist', {}))))

    #
    tsp = TimeSeriesPlot(kws, styles)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # check for structured data (dict keyed on labels and containing data)
    labels = kws.labels
    l = [isinstance(d, dict) for d in data]
    if any(l):
        dgen = (list(d.values()) if ll else d for d, ll in zip(data, l))
        keys = [tuple(data[i].keys()) for i in np.where(l)[0]]
        data = tuple(dgen)
        if len(keys) > 1:
            assert keys[0] == keys[1], "dict keys don't match"
        if labels is None:
            labels = keys[0]

    # # parse data args
    times, signals, errors = data = get_data(data)
    n = len(signals)

    #
    # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # # check labels
    # if labels is None:
    #     labels = []
    # if isinstance(labels, str):
    #     labels = [labels]  # check if single label given
    # labels = labels[:N]

    colours = get_line_colours(n, kws.colours, kws.cmap)
    fig, ax = tsp.setup_figure(kws.ax, colours)

    # Do the plotting
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for i, (t, signal, stddev, label) in enumerate(itt.zip_longest(
            times, signals, errors, labels, fill_value=None)):
        # zip_longest in case errors or times are empty sequences
        tsp.plot_ts(ax, t, signal, stddev, label, kws.show_errors)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    tsp.set_labels(ax, kws.title, *get_axes_labels(kws.axlabels),
                   kws.twinx, kws.relative_time)
    xlims, ylims = tsp.set_axes_limits(data, kws.whitespace,
                                       (kws.xscale, kws.yscale))
    ax.set(xlims=xlims, ylim=ylims)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Setup plots for canvas interaction
    if kws.draggable and not show_hist:
        # FIXME: maybe warn if both draggable and show_hist
        # make the artists draggable
        from graphical.draggables import DraggableErrorbar
        plots = DraggableErrorbar(tsp.art, offsets=kws.offsets,
                                  linked=tsp._linked,
                                  **tsp.styles.legend)
        # TODO: legend with linked plots!

    else:
        ax.legend(tsp.art, labels, **styles.legend)
        # self._make_legend(ax, tsp.art, labels)

    return tsp


@attrs
class TimeSeriesPlot(object):
    kws = attr({})
    styles = attr({})

    fig = attr(None)
    ax = attr(None)
    art = attr(())

    times = attr(())
    signals = attr(())
    errors = attr(())

    hist = attr([])
    hax = attr(None)

    mask_shown = attr(False, init=False)  # , repr=False
    _linked = attr([], init=False)
    _proxies = attr([], init=False)

    # show_hist = attr(False)

    # def __init__(self):

    def plot_ts(self, ax, t, signal, stddev, label, show_errors, show_masked,
                show_hist, relative_time, styles):

        # TODO maxpoints = 1e4  opt

        if (stddev is not None) & (show_errors == 'contour'):
            uncertainty_contours(ax, t, signal, stddev, styles, lw=1)
            # TODO: or simulated_samples?

        # main plot
        ebar_art = ax.errorbar(
                *sanitize_data(t, signal, stddev, show_errors, relative_time),
                label=label, **styles.errorbar)
        self.art.append(ebar_art)

        # plot masked values with different style if requested
        if show_masked:
            col = ebar_art[0].get_color()
            msk_art = self.plot_masked_points(t, signal, col, show_masked)
            if msk_art:
                self.art.append(msk_art)
                self._linked.append((ebar_art, msk_art))

        # Histogram
        if show_hist:
            self.plot_histogram(signal, **styles.hist)

        return ebar_art

    def plot_masked_points(self, t, signal, colour, how):
        # Get / Plot GTIs

        # valid = self._original_sizes[i]
        unmasked = signal.copy()  # [:valid]
        # tum = t[:valid]

        # msk_art = None
        # if how == 'span':
        #     self.plot_masked_intervals(ax, t, unmasked.mask)

        if how == 'x':
            # invert mask
            unmasked.mask = ~unmasked.mask
            # NOTE: using errorbar here so we can easily convert to
            #  DraggableErrorbarContainers
            # FIXME: Can fix this once DraggableLines are supported
            # TODO: manage opts in styles.....
            ebar_msk = self.ax.errorbar(t, unmasked, color=colour,
                                        marker='x', ls='None', alpha=0.7,
                                        label='_nolegend_')

        else:
            raise NotImplementedError

    def plot_histogram(self, signal, **props):
        #
        self.hist.append(
                self.hax.hist(np.ma.compress(signal), **props)
        )
        self.hax.grid(True)

    def setup_figure(self, ax, colours, show_hist):
        """Setup figure geometry"""

        # FIXME:  leave space on the right of figure to display offsets
        fig, ax = self.get_axes(ax)

        # Add subplot for histogram
        if show_hist:
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(ax)
            self.hax = divider.append_axes('right', size='25%', pad=0.,
                                           sharey=ax)
            self.hax.grid()
            self.hax.yaxis.tick_right()

        # NOTE: #mpl >1.4 only
        if len(colours):
            from cycler import cycler
            ccyc = cycler('color', colours)
            ax.set_prop_cycle(ccyc)
            if show_hist:
                self.hax.set_prop_cycle(ccyc)

        ax.grid(b=True, which='both')
        self.fig = fig
        self.ax = ax
        return fig, ax

    def get_axes(self, ax, figsize=default_figsize, twinx=None,
                 timescale=None, start=None):

        if ax is not None:
            return ax.figure, ax

        # default layout for pretty figures
        left, bottom, right, top = [0.025, 0.01, 0.97, .98]

        if not twinx:
            fig, ax = plt.subplots(figsize=figsize)

        elif twinx == 'sexa':
            # need extra space for the top tick labels
            top = .94

            # TODO: a factory that gets the appropriate class and keyword
            from .dualaxes import DateTimeDualAxes
            # from astropy import units as u
            # from astropy.coordinates.angles import Angle

            # TODO: support for longer scale time JD, MJD, etc...
            # h,m,s = Angle((self.t0 * u.Unit(self.timescale)).to(u.h)).hms
            # #start time in (h,m,s)
            # hms = int(h), int(m), int(round(s))Ep
            # #datetime hates floats#

            # FIXME: self.start might be None
            # if self.start is None:

            # ymd = tuple(map(int, self.start.split('-')))
            # start = ymd + hms

            fig = plt.figure(figsize=figsize)
            ax = DateTimeDualAxes(fig, 1, 1, 1,
                                  timescale=timescale,
                                  start=start)
            ax.setup_ticks()
            fig.add_subplot(ax)

        else:
            # TODO: centralise error handling
            raise NotImplementedError(
                    'Option %s not understood for `twinx`' % twinx)

        fig.tight_layout(rect=[left, bottom, right, top])
        return fig, ax

    def set_axes_limits(self, data, whitespace, scales, offsets):
        """Axes limits"""

        x, y, u = data
        (xl, xu), (yl, yu) = get_axes_limits(data, whitespace, offsets)

        x_scale, y_scale = scales
        if (y_scale == 'log') and (yl <= 0):
            yl = np.sort(y[y > 0])[0]  # nonzeromin
            # TODO: WARN
            # TODO: indicate more data points with arrows????????????
            #       : Would be cool if done while hovering mouse on legend
            # yl = max(y.min(), nonzeromin)

        if (x_scale == 'log') and (xl <= 0):
            xl = np.sort(x[x > 0])[0]  # nonzeromin
            # xl = max(x.min(), nonzeromin)

        return (xl, xu), (yl, yu)

    def set_labels(self, title, xlabels, ylabel, twinx, relative_time, t0=''):
        """axis title + labels"""
        ax = self.ax
        title_text = ax.set_title(title, fontweight='bold')

        xlb, xlt = xlabels
        ax.set_xlabel(xlb)
        if twinx:
            # make space for the tick labels
            title_text.set_position((0.5, 1.09))
            if xlt:
                ax.parasite.set_xlabel(xlt)
        ax.set_ylabel(ylabel)

        # display time offset
        if relative_time:
            ax.xoffsetText = ax.text(1, ax.xaxis.labelpad,
                                     '[{:+.1f}]'.format(t0),
                                     ha='right',
                                     transform=ax.xaxis.label.get_transform())


def convert_mask_to_intervals(a, mask=None):
    """Return index tuples of contiguous masked values."""
    if mask is None:
        mask = a.mask
        # NOTE: If a is a masked array, this function returns masked values!!!

    if ~np.any(mask):
        return ()

    from recipes.iter import interleave
    w, = np.where(mask)
    l1 = w - np.roll(w, 1) > 1
    l2 = np.roll(w, -1) - w > 1
    idx = [w[0]] + interleave(w[l2], w[l1]) + [w[-1]]
    return a[idx].reshape(-1, 2)


# ******************************************************************************
# NOTE: you can probs use the std plt.subplots machinery if you register your
#  axes classes

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


def phase_time_plot(P, toff=0, **figkws):
    from matplotlib.transforms import Affine2D
    from graphical.dualaxes import DualAxes
    fig = plt.figure(**figkws)

    aux_trans = Affine2D().translate(-toff, 0).scale(1 / P)
    ax = DualAxes(fig, 1, 1, 1, aux_trans=aux_trans)
    ax.setup_ticks()
    ax.set_xlabel('Orbital Phase')
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

    # embed()

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

    r = Rectangle((0, 0), 1, 1, fc='grey',
                  ec='none')  # rectangle proxy art for legend.
    leg = ax.legend((plm, plmn, r), ('mean', 'extrema', r'$1\sigma$'))

    ax.figure.tight_layout()
    # return fig


# class TSplotter(object):


# TODO: evolve to multiprocessed TS plotter.

# TODO: Keyword translation!

# def __call__(self, *data, **kws):  # def __init__(self):
#
#
# # alias the call method
# plot = __call__

# def _get_twin(self, fig):
# if self.twin ==
# 'sexa':DateTimeDualAxes
#
# else:

# def plot_masked_intervals(self, ax, t, mask):
#     """
#     Highlight the masked values within the time series with a span across
#      the axis
#      """
#     spans = convert_mask_to_intervals(t, mask)
#     for s in spans:
#         ax.axvspan(*s, **self.dopts.spans)
#
#     self.mask_shown = True
#     # bool(bti)
#     # #just so we don't make a legend entry for this if it's empty


# def _make_legend(self, ax, plots, labels):
#     """Legend"""
#
#     # print( labels, '!'*10 )
#
#     if len(labels):
#         if self.mask_shown:
#             from matplotlib.patches import Rectangle
#             span_label = self.span_props.pop('label')
#             r = Rectangle((0, 0), 1, 1,
#                           **self.span_props)  # span proxy artist for legend
#
#             plots += [r]
#             labels += [span_label]
#
#         ax.legend(plots, labels, **self.dopts.legend)


if __name__ == '__main__':
    """tests"""

    # TODO: move to tests directory

    from motley.profiler import profile

    profiler = profile()


    @profile.histogram
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
