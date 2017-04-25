# import numpy as np
import matplotlib.pyplot as plt

# from matplotlib import scale as mscale
from matplotlib import ticker

from .ticks import locator_transform_factory, SwitchLogFormatter
from .dualaxes import TimeFreqDualAxes, TimeFreqDualAxes2
from .ts import TSplotter

# from IPython import embed


#****************************************************************************************************
class SpecPlotter(TSplotter):

    # TODO: loglog + all the fancy ticks
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    defaults = TSplotter.defaults.copy()
    default_opts = TSplotter.default_opts.copy()

    defaults.axlabels = (('Frequency (Hz)', 'Period (s)'),  # bottom and top
                         'Power')
    default_opts.errorbar = dict(fmt='-',
                                 capsize=0)

    xax = 'f'

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def get_axes(self, ax):
        try:  #FIXME: a factory that gets the appropriate class and keywords will obviate this try
            return super().get_axes(ax)
        except NotImplementedError:
            fig = plt.figure(figsize=self.figsize)
            ax = TimeFreqDualAxes2(fig, 1, 1, 1, xax=self.xax)   #NOTE: Major rework of this class needed
            fig.add_subplot(ax)
            ax.setup_ticks()

        return fig, ax

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # def setup_figure_geometry(self, ax, colours):
    #
    #     fig = plt.figure(figsize=(18, 8))
    #     ax = TimeFreqDualAxes2(fig, 1, 1, 1, xax=self.xax)   #NOTE: Major rework of this class needed
    #     fig.add_subplot(ax)
    #
    #     ax.setup_ticks()
    #
    #     ax.grid(b=True, which='both')
    #
    #     # NOTE: #mpl >1.4 only
    #     if len(colours):
    #         from cycler import cycler
    #         ccyc = cycler('color', colours)
    #         ax.set_prop_cycle(ccyc)
    #
    #     rect = left, bottom, right, top = [0.025, 0.01, 0.97, .98]
    #     fig.tight_layout(rect=[left, bottom, right, top])
    #
    #     return fig, ax

     # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # def _set_labels(self, ax, title, xlabels, ylabel):
    '''axis title + labels'''

    # if not self.xax.lower().startswith('f'):
    #     xlabels = xlabels[::-1]



    # ax.set_ylabel( r'$\Theta^{-1}$' )

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def loglog(self, *data, **kws):

        kws.setdefault('xscale', 'log')
        kws.setdefault('yscale', 'log')
        res = fig, plots, *stuff = self.plot(*data, **kws)
        ax = fig.axes[0]

        ax.set_xscale('log')        #WARNING: this will change the parasite axes scale, locators, etc..
        ax.set_yscale('log')        #FIXME: figure out how to avoid locators reset

        # TODO: do this automatically when switching scales
        # NOTE: see if you can implement as a scale via `scale.set_default_locators_and_formatters`
        # OR these should be overwritten in TimeFreqDualAxes2????
        # ax.xaxis._scale = mscale.scale_factory('log', ax.xaxis)
        # ax.yaxis._scale = mscale.scale_factory('log', ax.yaxis)

        return ax

        # x axis
        fmt = SwitchLogFormatter(3, '\cdot')
        ax.xaxis.set_major_formatter(fmt)
        # twin axis major
        MajLoc = locator_transform_factory(ax.xaxis.major.locator, ax.transAux._x)
        ax.parasite.xaxis.set_major_locator(MajLoc())
        majForm = SwitchLogFormatter(3, '\cdot')
        ax.parasite.xaxis.set_major_formatter(majForm)

        # suppress: UserWarning: AutoMinorLocator does not work with logarithmic scale
        # FIXME: TimeFreqDualAxes2 class
        ax.parasite.xaxis.set_minor_locator(ticker.NullLocator())


        return res



#****************************************************************************************************


# splot = Splot()


if __name__ =='__main__':
    pass