# import numpy as np
# import matplotlib.pyplot as plt

# from matplotlib import scale as mscale
# from matplotlib import ticker
#
# from .ticks import locator_transform_factory, SwitchLogFormatter
from .dualaxes import TimeFreqDualAxes2 as PeriodFrequencyDual
# NOTE: Major rework of this class needed
from .ts import TimeSeriesPlot, defaults, default_opts, TWIN_AXES_CLASSES

# from IPython import embed

# TODO: loglog + all the fancy ticks
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
defaults = defaults.copy()
default_opts = default_opts.copy()

defaults.axes_labels = (('Frequency (Hz)', 'Period (s)'),  # bottom and top
                        'Power')
default_opts.errorbar = dict(fmt='-',
                             capsize=0)

TWIN_AXES_CLASSES['period'] = PeriodFrequencyDual
# FIXME: rename PeriodFrequencyDual


class PeriodogramPlot(TimeSeriesPlot):

    def get_axes(self, ax, figsize=(14, 8), twinx='period', **kws):
        kws.setdefault('xax', 'f')
        return TimeSeriesPlot.get_axes(self, ax, figsize, twinx, **kws)

    # def loglog(self, *data, **kws):
    #
    #     kws.setdefault('xscale', 'log')
    #     kws.setdefault('yscale', 'log')
    #     return self.plot_ts(*data, **kws)


        # ax = kws.get(ax, )
        #
        # # WARNING: this will change the parasite axes scale, locators, etc..
        # ax.set_xscale('log')
        # ax.set_yscale('log')  # FIXME: figure out how to avoid locators reset

        # TODO: do this automatically when switching scales
        # NOTE: see if you can implement as a scale
        #  via `scale.set_default_locators_and_formatters`
        # OR these should be overwritten in TimeFreqDualAxes2????
        # ax.xaxis._scale = mscale.scale_factory('log', ax.xaxis)
        # ax.yaxis._scale = mscale.scale_factory('log', ax.yaxis)



        # # x axis
        # fmt = SwitchLogFormatter(3, '\cdot')
        # ax.xaxis.set_major_formatter(fmt)
        # # twin axis major
        # MajLoc = locator_transform_factory(ax.xaxis.major.locator,
        #                                    ax.transAux._x)
        # ax.parasite.xaxis.set_major_locator(MajLoc())
        # majForm = SwitchLogFormatter(3, '\cdot')
        # ax.parasite.xaxis.set_major_formatter(majForm)
        #
        # # suppress: UserWarning: AutoMinorLocator does not work with
        # # logarithmic scale
        # # FIXME: TimeFreqDualAxes2 class
        # ax.parasite.xaxis.set_minor_locator(ticker.NullLocator())
        #
        # return res


    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # def _set_labels(self, ax, title, xlabels, ylabel):
    # '''axis title + labels'''

    # if not self.xax.lower().startswith('f'):
    #     xlabels = xlabels[::-1]

    # ax.set_ylabel( r'$\Theta^{-1}$' )


plot = PeriodogramPlot.plot
loglog = PeriodogramPlot.loglog