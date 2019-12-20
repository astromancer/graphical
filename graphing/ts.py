"""
Versatile functions for plotting time-series data
"""

# TODO:
#  alternatively make `class timeseriesPlot(Axes):` then ax.plot_ts()
# NOTE: you can probs use the std plt.subplots machinery if you register your
#  axes classes

import itertools as itt

import numpy as np
from numpy.lib.stride_tricks import as_strided

import matplotlib as mpl

mpl.use('Qt5Agg')
# from matplotlib import rcParams

import matplotlib.pyplot as plt
# import colormaps as cmaps
# plt.register_cmap(name='viridis', cmap=cmaps.viridis)

from recipes.array import ndgrid
from recipes.containers.dict_ import AttrDict

# from recipes.string import minlogfmt

# from IPython import embed

# from dataclasses import dataclass

from attr import attrs, attrib as attr  # , astuple

from recipes.introspection.utils import get_module_name
import logging

logger = logging.getLogger(get_module_name(__file__))

N_MAX_TS_SAFETY = 50

# Set parameter defaults
default_cmap = 'nipy_spectral'
default_figsize = (14, 8)

defaults = AttrDict(
        labels=(),
        title='',
        #
        relative_time=False,
        timescale='s',
        start=None,

        # NOTE: this might get messy! Consider setting up the axes outside
        #  and always passing?
        ax=None,
        axes_labels=(('t (s)', ''),  # bottom and top x-axes
                     'Counts'),
        # TODO: axes_label_position: (left, right, center, <,>,^)
        twinx=None,  # Top x-axis display format
        xscale='linear',
        yscale='linear',
        plims=((0, 1),  # x-axis
               (-0.1, 1.01)),  # y-axis
        #
        colours=None,
        cmap=None,  # TODO: colormap
        #
        show_errors='bar',
        show_masked=False,
        show_hist=False,

        draggable=False,
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

from .dualaxes import DateTimeDualAxes

TWIN_AXES_CLASSES = {'sexa': DateTimeDualAxes}

# TODO: indicate more data points with arrows????????????
#       : Would be cool if done while hovering mouse on legend

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

from math import floor, ceil


def get_percentile(data, p):
    a = abs(p)
    s = [-1, 1][p > 0]
    r, q = divmod(a, 1)
    c = abs(float(p > 1) - s * q) * 100
    d = 0
    if c > 0:
        d = np.percentile(np.ma.compressed(data), c)

    mn, mx, = data.min(), data.max()
    p1 = int(p > 1)
    s2 = 1 if 0 < p < 1 else -1

    # m = p1 - s * ceil(a) + 1
    # n = p1 + s * floor(a)
    # for x in 'psrqcmn':
    #     print(f'{x} = {eval(x):<4g}', end='\t')
    # print()

    return (p1 - s * ceil(a) + 1) * mn + (p1 + s * floor(a)) * mx + s2 * d
    # print('p = ', p, 'lim', l, 'expected', expect)


def get_percentile_limits(data, e=(), plims=(-0.05, 1.05)):
    """
    Return suggested axis limits based on the extrema of x, optional errorbars,
    and the desired fractional whitespace on axes.

    x: array-like
        data on display
    e - uncertainty (stddev, measurement errors)
        can be either single array of same shape as x, or 2 arrays (δx+, δx-)
    plims: 2-tuple
        Data limits expressed as percentiles of the data distribution.
        0 corresponds to the 0th percentile, 1 to the 100th percentile.
        numbers outside range (0, 1) are allowed in which case they will be
        interpreted as distances from the 0th and 100th percentile
        respectively and the unit distance is the data peak-to-peak width.
    """

    x = get_data_pm_1sigma(data, e)
    lims = np.empty(2, data.dtype)
    for i, (x, p) in enumerate(zip(x, plims)):
        lims[i] = get_percentile(x, p)

    return lims


@attrs
class DataPercentileAxesLimits(object):
    lower = attr(-0.05)
    upper = attr(+1.05)

    def get(self, data, e=()):
        return get_percentile_limits(data, e, (self.lower, self.upper))


def _set_defaults(props, defaults):
    for k, v in defaults.items():
        props.setdefault(k, v)


def check_kws(kws):
    # these are AttrDict!
    opts = defaults.copy()
    dopts = default_opts.copy()

    invalid = set(kws.keys()) - set(allowed_kws)
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


# def atleast_2d(x):
#     x = np.ma.atleast_2d(x)
#     if x.mask is False:


def get_data(data):
    """parse data arguments"""

    # signals only
    if len(data) == 1:
        # Assume here each row gives individual signal for a TS
        signals = np.ma.atleast_2d(np.ma.squeeze(data))
        # No times, plot by array index
        times, y_err, x_err = np.full((3, len(signals)), None)

        # times & signals given
    elif len(data) == 2:
        # times, signals = data
        times, signals = np.atleast_2d(*data)
        # signals = np.ma.atleast_2d(np.ma.squeeze(data[1])) #
        # No errors given
        y_err, x_err = np.full((2, len(signals)), None)

    # times, signals, y-errors given
    elif len(data) == 3:
        # times, signals, errors = data
        times, signals, y_err = np.ma.atleast_2d(*data)
        x_err = np.full(len(signals), None)

    # times, signals, y-errors, x-errors given
    elif len(data) == 4:
        times, signals, y_err, x_err = np.ma.atleast_2d(*data)

    else:
        raise ValueError('Invalid number of arguments: %i' % len(data))

    # NOTE: if signals has non-uniform length, these will be arrays with
    #  dtype object

    # safety breakout for erroneous arguments that can trigger very slow
    # plotting loop
    n = len(signals)
    if n > N_MAX_TS_SAFETY:
        raise TooManyToPlot(
                'Received %i time series to plot. Safety limit is '
                'currently set to %i' % (n, N_MAX_TS_SAFETY))

    # duplicate times if multivariate implied (without duplicating memory!!)
    # NOTE: this does not work if the array is a column of a recarray
    # if (len(times) == 1) & (n > 1):
    #     times = as_strided(times, (n, times.size), (0, times.itemsize))
    if (len(times) == 1) & (n > 1):
        times = itt.repeat(times[0], n)

    return times, signals, y_err, x_err


def sanitize_data(t, signal, y_err, x_err, show_errors, relative_time):
    # catch in case for some reason the user provides a sequence of
    # empty error sequences
    # note ax.errorbar borks for empty error sequences
    n = len(signal)
    stddevs = []
    for yx, std in zip('yx', (y_err, x_err)):
        if std is not None:
            if show_errors:
                size = np.size(std)
                if size == 0:
                    logger.warning('Ignoring empty uncertainties in {xy}.')
                    std = None
                elif size != n:
                    raise ValueError(f'Unequal number of points between data '
                                     f'({n}) and {yx}-stddev arrays ({size}).')
                else:
                    std = np.ma.masked_where(np.isnan(std), std)

                # check that errors are not all masked. This sometimes happens
                # when data is read into fields where uncertainties are expected
                if std.mask.all():
                    logger.warning(f'All uncertainties in {yx} are masked.  '
                                   f'Ignoring.')
                    std = None

            else:
                logger.warning(f'Ignoring uncertainties in {yx} since '
                               '`show_errors = False`.')
                std = None
        # aggregate
        stddevs.append(std)

    # plot by frame index if no time
    if (t is None) or (len(t) == 0):
        t = np.arange(len(signal))
    else:
        if len(t) != len(signal):
            raise ValueError('Unequal number of points between data and time '
                             'arrays.')
        # Adjust start time
        if relative_time:
            t = t - t[0]

    # mask nans
    signal = np.ma.MaskedArray(signal, ~np.isfinite(signal))
    return (t, signal) + tuple(stddevs)


def get_line_colours(n, colours, cmap):
    # Ensure we plot with unique colours

    # rules here are:
    # `cmap` always used if given
    # `colours` always used if given, except if `cmap` given
    #   warning emitted if too few colours - colour sequence will repeat
    too_few_colours = len(mpl.rcParams['axes.prop_cycle']) < n
    if (cmap is not None) or ((colours is None) and too_few_colours):
        cm = plt.get_cmap(cmap)
        colours = cm(np.linspace(0, 1, n))  # linear colour map for ts

    elif (colours is not None) and (len(colours) < n):
        logger.warning('Colour sequence has too few colours (%i < %i). Colours '
                       'will repeat' % (len(colours), n))
    return colours


from recipes.misc import duplicate_if_scalar


# def get_axes_limits(data, whitespace, offsets=None):
#     """Axes limits"""
#
#     x, y, u = data
#     xf, yf = duplicate_if_scalar(whitespace)  # Fractional white space in figure
#     xl, xu = axes_limit_from_data(x, xf)
#     yl, yu = axes_limit_from_data(y, yf, u)
#
#     if offsets is not None:
#         yl += min(offsets)
#         yu += max(offsets)
#
#     return (xl, xu), (yl, yu)

def get_data_pm_1sigma(x, e=()):  # , sigma_e_cut=3.
    #
    if e is None:
        return x, x

    n = len(e)
    if n == 0:
        return x, x
    elif n == 2:
        return x - e[0], x + e[1]
    else:
        return x - e, x + e


def get_axes_labels(axes_labels):
    if (axes_labels is None) or (len(axes_labels) == 0):
        return defaults.axes_labels

    if len(axes_labels) == 2:
        xlabels, ylabel = axes_labels

        if isinstance(xlabels, str):
            xlabels = (xlabels, '')
    else:
        raise ValueError('Invalid axes labels')

    return xlabels, ylabel


def uncertainty_contours(ax, t, signal, stddev, styles, **kws):
    # NOTE: interpret uncertainties as stddev of distribution
    from tsa.smoothing import smoother

    # preserve colour cycle
    sigma = 3
    c, = ax.plot(t, smoother(signal + sigma * stddev), **kws)
    colour = styles.errorbar['color'] = c.get_color()
    return ax.plot(t, smoother(signal - sigma * stddev), colour, **kws)


class TooManyToPlot(Exception):
    pass


# TODO: evolve to multiprocessed TS plotter.

# TODO: Keyword translation?

@attrs
class TimeSeriesPlot(object):
    kws = attr({})
    styles = attr({})

    fig = attr(None)
    ax = attr(None)
    art = attr([])

    hist = attr([])
    hax = attr(None)

    # mask_shown = attr(False, init=False)  # , repr=False
    _linked = attr([], init=False)

    # _proxies = attr([], init=False)

    # axes limits
    plims = attr((None, None))
    x_lim = attr(np.array([np.inf, -np.inf]), init=False)
    y_lim = attr(np.array([np.inf, -np.inf]), init=False)

    zorder0 = attr(10, init=False, repr=False)

    @classmethod
    def plot(cls, *data, **kws):
        """
        Plot light curve(s)

        Parameters
        ----------
        data:   tuple of array-likes
            (signal,)   -   in which case t is implicitly the integers up to
                            len(signal)
            (t, signal) -   in which case uncertainty is ignored
            (t, signal, uncertainty)
            Where:
                t:   (optional; array-like or dict)
                    time sequence of corresponding data
                signal:   (array-like or dict)
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
        tsp = cls(kws, styles)

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

        # parse data args
        times, signals, y_err, x_err = get_data(data)
        n = len(signals)

        # print(np.shape(times), np.shape(signals),
        #       np.shape(y_err), np.shape(x_err))

        # from IPython import embed
        # embed()

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # # check labels
        if labels is None:
            labels = []
        elif isinstance(labels, str):
            labels = [labels]  # check if single label given
        elif len(labels) > n:
            raise ValueError('Bad labels')

        # tuple, dict, array...

        colours = get_line_colours(n, kws.colours, kws.cmap)
        fig, ax = tsp.setup_figure(kws.ax, colours, show_hist)

        # todo ; move to init
        tsp.plims = kws.plims

        # print('before zip:', len(times), len(signals), len(errors))
        # Do the plotting
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # noinspection NonAsciiCharacters
        for i, (x, y, σy, σx, label) in enumerate(itt.zip_longest(
                times, signals, y_err, x_err, labels)):
            # print(np.shape(x), np.shape(y), np.shape(y_err), np.shape(x_err))

            # zip_longest in case errors or times are empty sequences
            tsp.plot_ts(ax, x, y, σy, σx, label, kws.show_errors,
                        kws.show_masked, show_hist, kws.relative_time, styles)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # add text labels
        tsp.set_labels(kws.title, kws.axes_labels,
                       kws.twinx, kws.relative_time)

        # for lim in (tsp.x_lim, tsp.y_lim):
        #     lim += np.multiply([-1, 1], (np.ptp(lim) * kws.whitespace / 2))

        # set auto-scale limits
        # print('setting lims: ', tsp.y_lim)

        if np.isfinite(tsp.x_lim).all():
            ax.set_xlim(tsp.x_lim)
        if np.isfinite(tsp.y_lim).all():
            ax.set_ylim(tsp.y_lim)

        # xlim=tsp.x_lim, ylim=tsp.y_lim,
        ax.set(xscale=tsp.kws.xscale, yscale=tsp.kws.yscale)

        # tsp.set_axes_limits(data, kws.whitespace, (kws.xscale, kws.yscale),
        #                     kws.offsets)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Setup plots for canvas interaction
        if kws.draggable and not show_hist:
            # FIXME: maybe warn if both draggable and show_hist
            # make the artists draggable
            from graphing.draggable import DraggableErrorbar
            plots = DraggableErrorbar(tsp.art, offsets=kws.offsets,
                                      linked=tsp._linked,
                                      **tsp.styles.legend)
            # TODO: legend with linked plots!

        else:
            ax.legend(tsp.art, labels, **styles.legend)
            # self._make_legend(ax, tsp.art, labels)

        return tsp

    @classmethod
    def loglog(cls, *data, **kws):
        kws.setdefault('xscale', 'log')
        kws.setdefault('yscale', 'log')
        return cls.plot(*data, **kws)

    def plot_ts(self, ax, x, y, y_err, x_err, label, show_errors,
                show_masked, show_hist, relative_time, styles):

        # TODO maxpoints = 1e4  opt

        if (y_err is not None) & (show_errors == 'contour'):
            uncertainty_contours(ax, x, y, y_err, styles, lw=1)
            # TODO: or simulated_samples?

        # NOTE: masked array behaves badly in mpl < 1.5.
        # see: https://github.com/matplotlib/matplotlib/issues/5016/
        # x = x.filled(np.nan)

        # main plot
        x, y, y_err, x_err = data = sanitize_data(x, y, y_err, x_err,
                                                  show_errors,
                                                  relative_time)
        # print(np.shape(x), np.shape(y), np.shape(y_err), np.shape(x_err))


        ebar_art = ax.errorbar(*data, label=label, zorder=self.zorder0,
                               **styles.errorbar)
        self.art.append(ebar_art)

        # set axes limits
        x_lims = get_percentile_limits(x, x_err, self.plims[0])
        y_lims = get_percentile_limits(y, y_err, self.plims[1])
        # print('ylims', y_lims)
        # if not np.isfinite([x_lims, y_lims]).all():
        #     print('NON FINITE!')
        #     from IPython import embed
        #     embed()

        for lims, xy in zip((x_lims, y_lims), 'xy'):
            l, u = getattr(self, f'{xy}_lim')
            new_lims = np.array([min(lims[0], l), max(lims[1], u)])

            scale = self.kws[f'{xy}scale']
            if scale == 'log':
                neg = ([x, y][xy == 'y'] <= 0)
                # if neg.any():
                #     logger.warning(
                #             'Requested logarithmic scale, but data contains '
                #             'negative points. Switching to symmetric log '
                #             'scale')
                #     self.kws[f'{xy}scale'] = 'symlog'
                if new_lims[0] <= 0:  # FIXME: both could be smaller than 0
                    logger.warning(
                            'Requested negative limits on log scaled axis. '
                            'Using smallest positive data element as lower '
                            'limit instead.')
                    new_lims[0] = y[~neg].min()
            # print('new', new_lims)
            # set new limits
            setattr(self, f'{xy}_lim', new_lims)

        # print('YLIMS', y_lims)
        # print('lims', 'x', self.x_lim, 'y', self.y_lim)

        # plot masked values with different style if requested
        if show_masked:
            col = ebar_art[0].get_color()
            msk_art = self.plot_masked_points(x, y, col, show_masked)
            if msk_art:
                self.art.append(msk_art)
                self._linked.append((ebar_art, msk_art))

        # Histogram
        if show_hist:
            self.plot_histogram(y, **styles.hist)

        self.zorder0 = 1

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
            return self.ax.errorbar(t, unmasked, color=colour,
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
        if colours is not None:
            from cycler import cycler
            ccyc = cycler('color', colours)
            ax.set_prop_cycle(ccyc)
            if show_hist:
                self.hax.set_prop_cycle(ccyc)

        ax.grid(b=True)  # which='both'
        self.fig = fig
        self.ax = ax
        return fig, ax

    def get_axes(self, ax, figsize=default_figsize, twinx=None, **kws):

        if ax is not None:
            return ax.figure, ax

        # get axes with parasite (sic)
        if twinx is not None:
            axes_cls = TWIN_AXES_CLASSES.get(twinx)
            if axes_cls:
                # make twin
                fig = plt.figure(figsize=figsize)
                ax = axes_cls(fig, 1, 1, 1, **kws)
                ax.setup_ticks()
                fig.add_subplot(ax)
                return fig, ax

            #
            logger.warning('Option %r not understood for argument `twinx`'
                           'ignoring' % twinx)

        return plt.subplots(figsize=figsize)

        # default layout for pretty figures
        # left, bottom, right, top = [0.025, 0.01, 0.97, .98]
        # fig.tight_layout(rect=rect)
        # return fig, ax

    def set_labels(self, title, axes_labels, twinx, relative_time, t0=''):
        """axis title + labels"""
        ax = self.ax
        title_text = ax.set_title(title, fontweight='bold')

        xlabels, ylabel = get_axes_labels(axes_labels)
        xlb, xlt = xlabels
        ax.set_xlabel(xlb)
        ax.set_ylabel(ylabel)

        if twinx:
            # make space for the tick labels
            title_text.set_position((0.5, 1.09))
            if xlt:
                ax.parasite.set_xlabel(xlt)

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
    from graphing.dualaxes import DualAxes
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


plot = TimeSeriesPlot.plot

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
#
#
# if __name__ == '__main__':
#     # generate some data
#     n = 250
#     np.random.seed(666)
#     t = np.linspace(0, 2 * np.pi, n)
#     y = [3 * np.sin(3 * t),
#          # np.cos(10*t),
#          np.cos(10 * np.sqrt(t))]
#     e = np.random.randn(len(y), n)
#     m = np.random.rand(len(y), n) > 0.8
#
#     kws = {}
#
#     # case 1:    bare minimum
#     print('CASE1')
#     tsp = plot(y[0], **kws)
#     #
#     # # case 2:    multiple series, no time
#     print('CASE2')
#     tsp = plot(y, **kws)
#     #
#     # # case 3:    multivariate time series
#     print('CASE3')
#     tsp = plot(t, y, **kws)
#
#     # # case 4:    full args
#     print('CASE4')
#     tsp = plot(t, y, e, **kws)
#     #
#     # # case 5: masked data
#     print('CASE5')
#     ym = np.ma.array(y, mask=m)
#     tsp = plot(t, ym, e,
#                show_masked='x',
#                **kws)
#
#     plt.show()
