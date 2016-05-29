import numpy as np
import matplotlib.pyplot as plt

from .dualaxes import TimeFreqDualAxes
from .lc import LCplot

from IPython import embed

        
#****************************************************************************************************
class Splot(LCplot):
    
    #FIXME:  period axes does not display correctly when plotting from 0 on frequency axis....
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    _default_errorbar = dict(fmt='-', 
                             capsize=0)
    
    xax = 'f'
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def setup_figure_geometry(self, ax, colours): 
        fig = plt.figure( figsize=(18,8) )
        ax = TimeFreqDualAxes(fig, 1, 1, 1, xax=self.xax)
        fig.add_subplot(ax)
        ax.setup_ticks()
        
        ax.grid( b=True, which='both' )
        
        #NOTE: #mpl >1.4 only
        if len(colours):
            from cycler import cycler
            ccyc = cycler('color', colours)
            ax.set_prop_cycle(ccyc)      
                
        rect = left, bottom, right, top = [0.025, 0.01, 0.97, .98]
        fig.tight_layout( rect=[left, bottom, right, top] )
        
        return fig, ax
        
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _set_labels(self, ax, title, axlabels):
        '''axis title + labels'''
        ax.set_title( title )
        
        if len(axlabels) == 0:
            xlabels, ylabel = (('Frequency (Hz)', 'Period (s)'),  #bottom and top
                              'Power' )
        if len(axlabels)==2:
            xlabels, ylabel  = axlabels
        
        if not self.xax.lower().startswith( 'f' ):
            xlabels = xlabels[::-1]
            
        ax.set_xlabel( xlabels[0] )
        ax.parasite.set_xlabel( xlabels[1] )
        ax.set_ylabel( ylabel )
        
        #ax.set_ylabel( r'$\Theta^{-1}$' )

#****************************************************************************************************


splot = Splot()


if __name__ =='__main__':
    pass