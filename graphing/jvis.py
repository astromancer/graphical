
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, AutoMinorLocator
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import AxesGrid

# matplotlib <1.5
# import colormaps as cmaps
# plt.register_cmap(name='viridis', cmap=cmaps.viridis)

#****************************************************************************************************
class ImaginaryScalarFormatter(ScalarFormatter):
    def __call__(self, x, pos=None):
        s = ScalarFormatter.__call__(self, x, pos)
        return s+'$i$'

#TODO: modular surfaces

#****************************************************************************************************
class Jvis():
    #TODO: 3D axes spines
    #TODO: auto redraw of 3D axes to display correctly....
    #TODO: matched rotation between 3D axes!!!!
    #FIXME / BUGREPORT: zooming shifts all 3 axes???????
    #TODO: arg(z) colors
    #TODO: function box intersection

    '''visualise a complex values function'''
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __init__(self, func):
        #TODO: check if callable
        self.func = func

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def setup(self):
        '''setup figure for plotting - create axes etc.'''
        # Create 3D axes grid
        fig = plt.figure(figsize=(15, 10))
        self.grid_3D = AxesGrid(fig, 211,  # similar to subplot(211)
                                nrows_ncols=(1, 3),
                                axes_pad=-0.5,
                                label_mode=None,  # This is necessary to avoid AxesGrid._tick_only throwing
                                # share_all = True,
                                axes_class=(Axes3D, {}))

        # Create the plot grid for the images
        self.grid_images = AxesGrid(fig, 212,  # similar to subplot(212)
                                    nrows_ncols=(1, 3),
                                    axes_pad=0.1,
                                    label_mode="L",  # THIS DOESN'T FUCKING WORK!
                                    # share_all = True,
                                    cbar_location="right",
                                    cbar_mode="edge",
                                    cbar_size="5%",
                                    cbar_pad="0%")

        titles = [r'$\Re[Z]$', r'$\Im[Z]$', '$|Z|$']
        for i, (ax3, axi) in enumerate(zip(self.grid_3D, self.grid_images)):
            # setup 3d axes
            ax3.set_title(titles[i],
                          y=1.1,
                          fontdict=dict(fontweight='heavy',
                                        fontsize='xx-large'))
            # verticalalignment='top'))
            ax3.xaxis.set_minor_locator(AutoMinorLocator())
            ax3.yaxis.set_minor_locator(AutoMinorLocator())
            ax3.yaxis.set_major_formatter(ImaginaryScalarFormatter())
            ax3.set_facecolor('None')

            # setup image axes
            axi.xaxis.set_minor_locator(AutoMinorLocator())
            axi.yaxis.set_minor_locator(AutoMinorLocator())
            axi.yaxis.set_major_formatter(ImaginaryScalarFormatter())

        return fig

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def plot3(self, extent=None, maxz=None, **kws):

        fig = self.setup()

        # Generate the data
        e = 5 if extent is None else extent
        if np.isscalar(e):
            xspan = x0, x1 = -e, e
            yspan = y0, y1 = -e, e
        else:
            e = np.ravel(e)
            if e.size == 2:
                x0, x1 = y0, y1 = np.sort(e)
            elif e.size == 4:
                x0, x1 = np.sort(e[:2])
                y0, y1 = np.sort(e[2:])
            else:
                raise ValueError('Incorrect shape for extent')

        xres, yres = 50j, 50j

        X, Y = np.mgrid[x0:x1:xres, y0:y1:yres]
        C = X + 1j*Y
        Z = self.func(C)

        # limit maximum
        # maxz = maxz or 200
        # Z[(Z > maxz) | (-maxz < Z)] = np.nan
        data = Zr, Zi, Za = np.real(Z), np.imag(Z), np.abs(Z)

        surface_opts = dict(cmap='viridis',
                            edgecolors='grey',
                            lw=0.25,
                            rstride=1,
                            cstride=1)
        surface_opts.update(kws)

        #plot the data
        for i, z in enumerate(data):
            ax = self.grid_3D[i]
            ax.plot_surface(X, Y, z, **surface_opts)

            axi = self.grid_images[i]
            im = axi.imshow(z,
                            cmap=surface_opts['cmap'],
                            #interpolation='cubic',
                            origin='llc',
                            extent=[x0,x1,y0,y1])
            ax.view_init(30, 135)

        self.grid_images.cbar_axes[0].colorbar(im)

        #fig.tight_layout()

        return fig

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # def




# jvis = Jvis()


if __name__ =='__main__':
    jvis(np.cos)
    plt.show()


