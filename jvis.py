import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, AutoMinorLocator
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import AxesGrid

import colormaps as cmaps
import numpy as np

plt.register_cmap(name='viridis', cmap=cmaps.viridis)

#****************************************************************************************************    
class ImaginaryScalarFormatter(ScalarFormatter):
    def __call__(self, x, pos=None):
        s = ScalarFormatter.__call__(self, x, pos)
        return s+'$i$'


#****************************************************************************************************    
class Jvis():
    #TODO: 3D axes spines
    #TODO: auto redraw of 3D axes to display correctly....
    #TODO: matched rotation between 3D axes!!!!
    #TODO: arg(z) colors
    #TODO: function box intersection
    
    '''visualise a complex values function'''
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __call__(self, func, **kws):
        
        cmap = 'viridis'
        
        fig = self.setup()
        
        #Get the data
        xspan = x0, x1 = -5, 5
        yspan = y0, y1 = -5, 5
        xres, yres = 50j, 50j
        maxz = 20
        X, Y = np.mgrid[x0:x1:xres, y0:y1:yres]
        C = X + 1j*Y
        Z = func(C)
        #Z[Z>maxz] = np.nan
        data = Zr, Zi, Za = np.real(Z), np.imag(Z), np.abs(Z)
        
        options = dict(cmap=cmap, 
                    lw=0.25,
                    edgecolors='grey',
                    rstride=1, cstride=1)
        
        #plot the data
        for i, z in enumerate(data):
            ax = self.grid_3D[i]
            ax.plot_surface(X, Y, z, **options)
            
            axi = self.grid_images[i]
            im = axi.imshow(z,
                            cmap=cmap,
                            #interpolation='cubic', 
                            origin='llc', 
                            extent=[x0,x1,y0,y1])

        self.grid_images.cbar_axes[0].colorbar(im)

        fig.tight_layout()
    
        return fig
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def setup(self):
        '''setup figure for plotting - create axes etc.'''
        #Create 3D axes grid
        fig = plt.figure(figsize=(15,15))
        self.grid_3D = AxesGrid( fig, 211, # similar to subplot(211)
                                            nrows_ncols = (1, 3),
                                            axes_pad = -0.5,
                                            label_mode = None,          #This is necessary to avoid AxesGrid._tick_only throwing
                                            #share_all = True,
                                            axes_class=(Axes3D,{}) )

        #Create the plot grid for the images
        self.grid_images = AxesGrid( fig, 212, # similar to subplot(212)
                                    nrows_ncols = (1, 3),
                                    axes_pad = 0.1,
                                    label_mode = "L",           #THIS DOESN'T FUCKING WORK!
                                    #share_all = True,
                                    cbar_location="right",
                                    cbar_mode="edge",
                                    cbar_size="5%",
                                    cbar_pad="0%"  )
        
        titles = [r'$\Re[Z]$', r'$\Im[Z]$', '$|Z|$']
        for i, (ax3, axi) in enumerate(zip(self.grid_3D, self.grid_images)):
            #setup 3d axes
            ax3.set_title(titles[i],
                         y=1.1,
                         fontdict=dict(fontweight='heavy',
                                       fontsize='xx-large'))
                                #verticalalignment='top'))
            ax3.xaxis.set_minor_locator(AutoMinorLocator())
            ax3.yaxis.set_minor_locator(AutoMinorLocator())
            ax3.yaxis.set_major_formatter(ImaginaryScalarFormatter())
            ax3.set_axis_bgcolor('None')
        
            #setup image axes
            axi.xaxis.set_minor_locator(AutoMinorLocator())
            axi.yaxis.set_minor_locator(AutoMinorLocator())
            axi.yaxis.set_major_formatter(ImaginaryScalarFormatter())
        
        return fig
        
jvis = Jvis()
    
    
if __name__ =='__main__':
    jvis(np.cos)
    plt.show()
    

