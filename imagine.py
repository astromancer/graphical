import traceback
from collections import Callable
import numpy as np

import matplotlib
matplotlib.use('qt5agg')
import matplotlib.pylab as plt
from matplotlib.widgets import Slider #AxesWidget, 
#from matplotlib.patches import FancyArrow, Circle
#from matplotlib.transforms import Affine2D
#from matplotlib.transforms import blended_transform_factory as btf

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from mpl_toolkits.axes_grid1 import AxesGrid

#from .interactive import ConnectionMixin

#from draggables.machinery import DragMachinery
from .sliders import AxesSliders

from decor import expose

#from recipes.iter import grouper

#from PyQt4.QtCore import pyqtRemoveInputHook, pyqtRestoreInputHook
#from IPython import embed
#pyqtRemoveInputHook()
#embed()
#pyqtRestoreInputHook()


#class ColourSliders():
    ##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #def __init__(self, ax, x1, x2, slide_on='y', axpos=0., **kwargs):
        #''' '''
        ##self.setup_axes(ax, axpos)
        
        #markers = ax.plot(axpos, x1, 'b>', 
                          #axpos, np.mean((x1,x2)), 'go', 
                          #axpos, x2, 'r<',
                          #ms=7.5,
                          #clip_on=False, zorder=3)
        
        #self.knobs = DragMachinery(markers, annotate=False) #TODO: inherit???
        #self.max_knob, self.centre_knob, self.min_knob = self.knobs
        
        ##FIXME: knobs don't update immediately when shifting center knob - i guess this could be desirable
        ##TODO: options for plot_on_motion / plot_on_release behaviour
        ##self.centre_knob.on_changed(self.center_shift)
        
        #self.min_knob.on_changed(self.recentre0)
        #self.max_knob.on_changed(self.recentre1)
        ##self.min_knob.on_changed(lambda o: self.recentre(-o))
        ##FIXME: update centre knob when shifting others...
        
        #self.knobs.connect()
    
    
    ##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #def setup_axes(self, ax, axpos):
        #''' '''
        ##hide axis patch
        #ax.patch.set_visible(0)
        ##setup ticks
        #self.ax.tick_params(axis=self.slide_on, direction='inout', which='both')
        #axsli, axoth = [ax.xaxis, ax.yaxis][::self._order]
        #axoth.set_ticks([])
        #which_spine = 'right' if self._index else 'bottom'
        #axsli.set_ticks_position(which_spine)
        ##hide the axis spines
        #ax.spines[which_spine].set_position(('axes', axpos))
        #for where, spine in ax.spines.items():
            #if where != which_spine:
                #spine.set_visible(0)
    
    ##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #def get_vals(self):
        #return tuple(k.get_ydata()[0] + d.offset 
                        #for i, (k,d) in enumerate(self.knobs.draggables.items()) 
                            #if not i==1)
        ##SAY WHAAAAT???
    
    ##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #def on_changed(self, func):
        #for knob in (self.min_knob, self.max_knob):
            #knob.on_changed(func)
    
    ##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ##@expose.args()
    #def recentre0(self, offset):
        ##momentarily disable observers to reposition center knob
        ##offset = (self.max_knob.offset + self.min_knob.offset) / 2
        #print('offset', offset)
        #self.centre_knob.shift(offset/2, observers_active=False)
        #self.centre_knob.offset = offset/2
        
    #def recentre1(self, offset):
        #self.recentre0(-offset)
    
    ##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ##@expose.args()
    ##def center_shift(self, offset):
        ##for knob in (self.min_knob, self.max_knob):
            ##knob.shift(offset)
            ##knob.offset = offset
    
    
        
        
   
    
#TODO checkout APLPY?????????
from mpl_toolkits.axes_grid1 import make_axes_locatable
#****************************************************************************************************
class ImageDisplay(object):
    #TODO: Option for data histogram on slider bar!!!
    #TODO: Choose figure geometry based on data shape
    #TODO: connect histogram range with colorbar and drop colorbar ticklabels + shift hist ticklabels to outside?
    
    #FIXME: sliders disappear behind histogram on blitting
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    SliderClass = AxesSliders#ColourSliders #AxesSliders
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __init__(self, ax, data, *args, **kwargs):
        ''' '''
        #self.sscale = kwargs.pop('sscale', 'linear')
        title = kwargs.pop('title', None)
        self.has_hist = kwargs.pop('hist', True)
        self.use_blit = kwargs.pop('use_blit', True)
        
        self.data = data = np.atleast_2d(data)
        self.ax = ax
        self.ax.format_coord = self.cooDisplayFormatter
        
        #set the clims vmin and vmax in kwargs according to requested autoscaling
        kwargs = self.update_autoscale_limits(data, **kwargs)
        
        #set the axes title if given
        if not title is None:
            ax.set_title(title)
        
        #use imshow to do the plotting
        self.imgplt = ax.imshow(data, *args, **kwargs)
        
        #create the colourbar and the AxesSliders
        self.divider = make_axes_locatable(ax)
        
        self.Createcolorbar()
        
        #if self.has_hist:
            #self.CreateHistogram()
        
        self.CreateSliders()
        
        if self.use_blit:
            self.imgplt.set_animated(True)
            self.sliders.ax.set_animated(True)
        
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def update_autoscale_limits(self, data, **kwargs):
        '''
        Update keyword dictionary with vmin, vmax for image autoscaling.  
        Remove other unnecessary keywords.
        
        Parameters
        ----------
        data            :       array-like
            data to use for deciding colour limits
        
        Keywords
        --------
        autoscale       :       str - {'percentile', 'zscale'}; 
                                default 'percentile'
            which algorithm to use
        
        other keywords specify options for specific algorithm.
        
        Returns
        -------
        dict of modified keywords
        '''
        #if both vmin and vmax are given, use those as limits
        if ('vmin' in kwargs) and ('vmax' in kwargs):
            kwargs.pop('clims', None) 
            return kwargs       #NOTE: might stil need to pop cruft as below
        
        #clims means the same as vmin, vmax
        clims = kwargs.pop('clims', None) 
        if clims is None:
            autoscale = kwargs.pop('autoscale', 'percentile')
        
            if autoscale.startswith('p'):
                pmin = kwargs.pop( 'pmin', 2.25 )
                pmax =  kwargs.pop( 'pmax', 99.75 )
                plims = kwargs.pop( 'plims', (pmin, pmax) )
                clims = np.percentile( data, plims  )
            
            elif autoscale.startswith('z'):
                from zscale import zrange
                contrast    = kwargs.pop('contrast', 1/100 )
                sigma_clip  = kwargs.pop('sigma_clip', 3.5)
                maxiter     = kwargs.pop('maxiter', 10)
                num_points  = kwargs.pop('num_points', 1000)
                clims = zrange( data,
                                contrast=contrast,
                                sigma_clip=sigma_clip,
                                maxiter=maxiter,
                                num_points=num_points )
            else:
                clims = data.min(), data.max()
        
        #set the computed/provided clim values 
        #setdefaults ==> if either vmin or vmax provided they will supercede
        kwargs.setdefault('vmin', clims[0])
        kwargs.setdefault('vmax', clims[1])
        return kwargs
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def get_autoscale_limits(self, data, **kwargs):
        '''Get autoscale limits for data'''
        kws = self.update_autoscale_limits(data, **kwargs)
        return kws['vmin'],  kws['vmax']
        
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def Createcolorbar(self):
        self.cax = self.divider.append_axes('right', size=0.2, pad=0)
        self.cbar = self.ax.figure.colorbar( self.imgplt, cax=self.cax)
        
        if self.use_blit:
            self.cax.set_animated(True)
        
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def CreateSliders(self):
        
        sax = self.divider.append_axes('right', size=1, pad=0.5)
        sax.set_ylim((self.data.min(), self.data.max()))
        #sax.set_yscale( self.sscale )
        
        #self.sliders = type('null', (), {})()
        #self.sliders.ax = sax
        
        if self.has_hist:
            self.CreateHistogram(sax)
        
        self.sliders = self.SliderClass(sax, *self.imgplt.get_clim(), slide_on='y')
        #self.sliders.drawon = False
        #self.sliders.knob[1].
        self.sliders.min_slide.on_changed(self.set_clim)
        self.sliders.max_slide.on_changed(self.set_clim)
        
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def CreateHistogram(self, ax):
        #FIXME
        '''histogram data on slider axis'''
        #from matplotlib.collections import PatchCollection
        
        h = ax.hist(self.data.ravel(), bins=100,
                    orientation='horizontal', log=True)
        self.hvals, self.bin_edges, self.patches = h
        
        #TODO: use PatchCollection?????
        if self.use_blit:
            for p in self.patches:
                p.set_animated(True)
            
        clims = self.imgplt.get_clim()
        self.hup(clims)
        
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def hup(self, clims):    
        for i, (p, c) in enumerate(zip(self.patches, self.get_hcol(clims))):
            p.set_fc(c)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def get_hcol(self, clims):
        
        cmap = self.imgplt.get_cmap()
        
        vm = np.ma.masked_outside(self.bin_edges, *clims)
        colours = cmap((vm - vm.min())/vm.max())
        colours[vm.mask, :3] = 0.25
        colours[vm.mask, -1] = 1
        
        return colours
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def set_clim(self, xydata):
        #clims = xy[self._index]
        #print('clims', clims)
        
        clims = np.array(self.sliders.positions) #NOTE: positions will only be current upon release
        clims[self.sliders.index()] = xydata[self.sliders._index]
        self.imgplt.set_clim(clims)
        
        if not self.has_hist:
            return self.imgplt #COLOURBAR ticklabels??
        
        self.hup(clims)
        return self.imgplt, self.patches #COLOURBAR ticklabels??
        
         #COLOURBAR??
    
        #fig = self.ax.figure
        
        #self.imgplt.draw(fig._cachedRenderer)
        
        #print('DRAW')
        #fig.canvas.draw()
        #self.draw_blit()
        #self.background = ax.figure.canvas.copy_from_bbox(ax.bbox)
        #self.imgplt.figure.canvas.draw()        #TODO: BLIT!!!!!
        #print('FOO!')
        
        #TODO: figure out how to blit properly in an interactive session
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #def draw_blit(self):
        #print('blitt!!')
        ##FIXME!
        #fig = self.ax.figure
        #fig.canvas.restore_region(self.background)
        
        #self.ax.draw_artist(self.imgplt)
        #self.ax.draw_artist(self.sliders.selection) #FIXME: does this redraw all 3 sliders when i'ts the center knob?
        
        #fig.canvas.blit(self.ax.bbox)
        #fig.canvas.blit(self.sax.bbox)

   
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def cooDisplayFormatter(self, x, y):
        col, row = int(x+0.5), int(y+0.5)
        Nrows, Ncols = self.data.shape
        if col>=0 and col<Ncols and row>=0 and row<Nrows:
            z = self.data[row, col]
            return 'x=%1.3f,\ty=%1.3f,\tz=%1.3f'%(x, y, z)
        else:
            return 'x=%1.3f, y=%1.3f'%(x, y)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def connect(self):
        self.sliders.connect()
    
    
#****************************************************************************************************
class CubeDisplayBase(ImageDisplay):
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __init__(self, ax, data, coords=None, **kwargs):
        '''
        Image display for 3D data. Implements frame slider and image scroll.  
        Optionally also displays apertures if coordinates provided.
        
        subclasses must implement set_frame, get_frame methods
        
        Parameters
        ----------
        ax      :       Axes object
            Axes on which to display
        data    :       array-like
            initial display data
        coords  :       optional, np.ndarray
            coordinates of apertures to display.  This must be an np.ndarray with
            shape (k, N, 2) where k is the number of apertures per frame, and N 
            is the number of frames
        
        kwargs are passed directly to ImageDisplay.
        '''
        #setup image display
        self.autoscale = kwargs.pop('autoscale', 'percentile') #TODO: move up??
        ImageDisplay.__init__(self, ax, data, **kwargs)
        
        #self.coords = coords
        
        #setup frame slider
        self._frame = 0
        self.fsax = self.divider.append_axes('bottom', size=0.2, pad=0.25)
        #TODO: elliminated this SHIT Slider class!!!
        self.frame_slider = Slider(self.fsax, 'frame', 0, len(self), valfmt='%d')
        self.frame_slider.on_changed(self.set_frame)
        if self.use_blit:
            self.frame_slider.drawon = False   
            
        #save background for blitting
        fig = ax.figure
        self.background = fig.canvas.copy_from_bbox(ax.bbox)

        #enable frame scroll
        fig.canvas.mpl_connect('scroll_event', self._scroll)
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #@property
    #def has_coords(self):
        #return self.coords is not None
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _needs_drawing(self):
        #NOTE: this method is a temp hack to return the artists that need to be 
        #drawn when the frame is changed (for blitting). This is in place while
        #the base class is being refined.
        #TODO: proper observers as modelled on draggables.machinery
        
        needs_drawing = [self.imgplt]
        if self.has_hist:
            needs_drawing.extend(self.patches)      #TODO: PatchCollection...
        
        if self.autoscale:
            needs_drawing.extend(self.sliders.sliders)
            
            
        ##[#self.imgplt.colorbar, #self.sliders.centre_knob])
        
        
        return needs_drawing
        
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def get_data(self, i):
        return self.data[i]
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def get_frame(self):
        return self._frame

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #@expose.args()
    def set_frame(self, i, draw=False):
        '''Set frame data. draw if requested '''
        i %= len(self)          #wrap around! (eg. scroll past end ==> go to beginning)
        i = int(round(i, 0))    #make sure we have an int
        self._frame = i
        
        data = self.get_data(i)
        
        #ImageDisplay.draw_blit??
        #set the slider axis limits
        dmin, dmax = data.min(), data.max()
        self.sliders.ax.set_ylim(dmin, dmax)
        self.sliders.valmin, self.sliders.valmax = dmin, dmax
        #needs_drawing.append()???
        
        #set the image data
        self.imgplt.set_data(data)
        #needs_drawing = [self.imgplt]
        
        
        if self.autoscale:
            #set the slider positiions / color limits
            vmin, vmax = self.get_autoscale_limits(data, autoscale=self.autoscale)
            self.imgplt.set_clim(vmin, vmax)
            self.sliders.set_positions((vmin, vmax))
            
        
        #TODO: update hisogram values etc...
        
        #ImageDisplay.draw_blit??
        if draw:
            needs_drawing = self._needs_drawing()
            self.draw_blit(needs_drawing)

    frame = property(get_frame, set_frame)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _scroll(self, event):
        self.frame += [-1, +1][event.button == 'up']
        self.frame_slider.set_val(self.frame)
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #@expose.args()
    def draw_blit(self, artists):
        
        #print('draw_blit')
        
        fig = self.ax.figure
        fig.canvas.restore_region(self.background)
        
        for art in artists:
            try:
                self.ax.draw_artist(art)
            except Exception as err:
                print('drawing FAILED', art)
                traceback.print_exc()
                
        
        fig.canvas.blit(fig.bbox)
        
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def cooDisplayFormatter(self, x, y):
        s = ImageDisplay.cooDisplayFormatter(self, x,y)
        return 'frame %d: %s'%(self.frame, s)


    
#****************************************************************************************************
class ImageCubeDisplay(CubeDisplayBase):
    #TODO: frame switch buttons;
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __init__(self, ax, data, coords=None, **kwargs):
        self.data = data = np.atleast_3d(data)
        CubeDisplayBase.__init__(self, ax, data[0], **kwargs)
        
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __len__(self):
        return len(self.data)
    


#****************************************************************************************************
class ImageCubeDisplayA(ImageCubeDisplay):
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    aperture_properties = dict(ec='m', lw=1, animated=True, picker=False,
                               widths=7.5, heights=7.5)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __init__(self, ax, data, ap_data_dict={}, **kwargs):
        '''with Apertures
        ap_data_dict is dict keyed on aperture aperture_properties, values of which
        are either array-like sequence of values indeces corresponing to frame number, 
        or callables that generate values.
        '''
        CubeDisplayBase.__init__(self, ax, data, **kwargs)
        
        #create apertures if coordinates provided
        self.aps = None
        self.ap_data = ap_data_dict
        if self.ap_data:
            from obstools.aps import ApertureCollection
            #add apertures to axes.  will not display yet as coordinates not set
            props = ImageCubeDisplayA.aperture_properties
            self.aps = ApertureCollection(**props)
            
            self.aps.axadd(ax)
            
            #TODO: check data
            
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def set_frame(self, i, draw=True):
        
        super().set_frame(i, False)
        needs_drawing = self._needs_drawing()
        
        if self.aps is not None:
            for attr, item in self.ap_data.items():
                if isinstance(item, Callable):
                    vals = item(i)
                else:
                    vals = item[i]
                    
                setattr(self.aps, attr, vals)
            needs_drawing.append(self.aps)
        
        self.draw_blit(needs_drawing)
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #def update(self, draggable, xydata):
        #'''draw all artists that where changed by the motion'''
        #artists = filter(None, 
                         #(func(xydata) for cid, func in draggable.observers.items()))    #six.iteritems(self.observers):
        #if self._draw_on:
            #if self._use_blit:
                #self.canvas.restore_region(self.background)
                #for art in flatiter((artists, draggable.ref_art)): #WARNING: will flatten containers etc
                    #art.draw(self.canvas.renderer)
                #self.canvas.blit(self.figure.bbox)
            #else:
                #self.canvas.draw()
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #def draw_blit(self, artists):
        #fig = self.ax.figure
        #fig.canvas.restore_region(self.background)
        
        #for art in artists:
            #self.ax.draw_artist(art)
        
        #fig.canvas.blit(fig.bbox)
        

#****************************************************************************************************
class ImageCubeDisplayX(ImageCubeDisplay):
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    marker_properties = dict(c='r', marker='x', alpha=.3, ls='none')
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __init__(self, ax, data, coords=None, **kwargs):
        CubeDisplayBase.__init__(self, ax, data, coords=None, **kwargs)
        
        if self.has_coords:
            self.marks = ax.plot([],[], **self.marker_properties)
            





#****************************************************************************************************
from fastfits import FITSFrame
from recipes.iter import interleave
class FITSCubeDisplay(ImageCubeDisplayA, FITSFrame):
    #FIXME: switching with slider messes up the aperture indexes
    #TODO: frame switch buttons; 
    #TODO: option to set clim from first frame??
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #DisplayClass = ImageCubeDisplayA
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __init__(self, ax, filename, ap_data_dict={}, **kws):
        ''' '''
        #setup data access
        FITSFrame.__init__(self, filename)
        
        sx = self.ishape
        extent = interleave((0,)*len(sx), sx)
        ImageCubeDisplayA.__init__(self, ax, [[0]], 
                                    ap_data_dict,
                                    origin='llc',
                                    extent=extent,
                                    **kws)
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __len__(self):
        return FITSFrame.__len__(self)
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def get_data(self, i):
        return self[i]
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def get_frame(self):
        return self._frame

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #def set_frame(self, i):
        
        #self._frame = i
        #data = self[int(i%len(self))]
        #if self.autoscale:
            ##set the slider axis limits
            #dmin, dmax = data.min(), data.max()
            #self.sliders.ax.set_ylim(dmin, dmax)
            #self.sliders.valmin, self.sliders.valmax = dmin, dmax
            
            ##set the slider positiions / color limits
            #vmin, vmax = self.get_autoscale_limits(data, autoscale=self.autoscale)
            #self.imgplt.set_data(data)
            #self.imgplt.set_clim(vmin, vmax)
            #self.sliders.set_positions((vmin, vmax))
        
        ##update the apertures if needed
        #if self.has_coords:
             #self.aps.coords = self.coords[:, i, :]
        
        ##TODO: BLIT!!
        #self.ax.figure.canvas.draw()
        ##self.draw_blit([self.imgplt, self.aps])
        

    #frame = property(get_frame, set_frame)


#class FITSCubeDisplay(ImageCubeDisplay, 



#class MultiImageCubeDisplay():
    #def __init__(self, *args, **kwargs):
        
        #assert not len(args)%2
        #self.axes, self.data = grouper(args, 2)
        
        #for 
        #super().__init__(self.axes[-1], self.data[-1], **kwargs)
        
        
    
    ##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #def get_frame(self):
        #return self._frame
    
    #def set_frame(self, f):
        
        #f %= len(self.data[0])  #wrap around!
        #for 
        #self.imgplt.set_data(self.data[f])
        #self.ax.figure.canvas.draw()
        #self._frame = f

    #frame = property(get_frame, set_frame)





#****************************************************************************************************
class Compare3DImage():
    #TODO: profile & speed up!
    #TODO: link viewing angles!!!!!!!!!
    #MODE = 'update'
    '''Class for plotting image data for comparison'''
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #@profile()
    def __init__(self, fig=None, *args, **kws):
        '''args=X, Y, Z, data'''
        
        self.titles = kws.get('titles', ['Data', 'Fit', 'Residual'])
        self.setup_figure(fig)
        if len(args):
            #X, Y, Z, data
            self.update(*args)
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #@unhookPyQt
    def setup_figure(self, fig=None):
        #TODO: Option for colorbars
        '''
        Initialize grid of 2x3 subplots. Top 3 are 3D wireframe, bottom 3 are colour images of 
        data, fit, residual.
        '''
        ##### Plots for current fit #####
        self.fig = fig = fig or plt.figure( figsize=(16,12),)
                                            #tight_layout=True )
        self.plots, self.images = [], []
        #TODO:  Include info as text in figure??????
        
        self.setup_3D_axes(fig)
        self.setup_image_axes(fig)
        
        fig.set_tight_layout(True)
        #fig.suptitle( 'PSF Fitting' )                   #TODO:  Does not display correctlt with tight layout
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def setup_3D_axes(self, fig):
        #Create the plot grid for the 3D plots
        self.grid_3D = AxesGrid(fig, 211, # similar to subplot(211)
                                nrows_ncols = (1, 3),
                                axes_pad = -0.2,
                                label_mode = None,          #This is necessary to avoid AxesGrid._tick_only throwing
                                share_all = True,
                                axes_class=(Axes3D,{}) )
        
        for ax, title in zip(self.grid_3D, self.titles):
            #pl = ax.plot_wireframe( [],[],[] )     #since matplotlib 1.5 can no longer initialize this way
            pl = Line3DCollection([])
            ax.add_collection(pl)
                
            #set title to display above axes
            title = ax.set_title( title, {'fontweight':'bold'} )
            x,y = title.get_position()
            title.set_position( (x, 1.0) )
            ax.set_facecolor('None')
            #ax.patch.set_linewidth( 1 )
            #ax.patch.set_edgecolor( 'k' )
            self.plots.append( pl )
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def setup_image_axes(self, fig):
        #Create the plot grid for the images
        self.grid_images = AxesGrid(fig, 212, # similar to subplot(212)
                                    nrows_ncols = (1, 3),
                                    axes_pad = 0.1,
                                    label_mode = "L",           #THIS DOESN'T FUCKING WORK!
                                    #share_all = True,
                                    cbar_location="right",
                                    cbar_mode="each",
                                    cbar_size="7.5%",
                                    cbar_pad="0%"  )
        
        for i, (ax, cax) in enumerate(zip(self.grid_images, self.grid_images.cbar_axes)):
            im = ax.imshow( np.zeros((1,1)), origin='lower' )
            cbar = cax.colorbar(im)
            #make the colorbar ticks look nice
            c = 'orangered' # > '0.85'
            cax.axes.tick_params(axis='y', 
                                 pad=-7,
                                 direction='in',
                                 length=3,
                                 colors=c,
                                 labelsize='x-small')
            #make the colorbar spine invisible
            cax.spines['left'].set_visible(False)
            #for w in ('top', 'bottom', 'right'):
            cax.spines['right'].set_color(c)
            
            for t in cax.axes.yaxis.get_ticklabels():
                t.set_weight('bold')
                t.set_ha('center')
                t.set_va('center')
                t.set_rotation(90)
                
            #if i>1:
                #ax.set_yticklabels( [] )       #FIXME:  This kills all ticklabels
            self.images.append(im)
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @staticmethod
    def make_segments(X, Y, Z):
        '''Update segments of wireframe plots.'''
        xlines = np.r_['-1,3,0', X, Y, Z]
        ylines = xlines.transpose(1,0,2)        #swap x-y axes
        return list(xlines) + list(ylines)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def update(self, X, Y, Z, data):
        '''update plots with new data.'''
        res = data - Z
        plots, images = self.plots, self.images
        
        plots[0].set_segments( self.make_segments(X,Y,data) )
        plots[1].set_segments( self.make_segments(X,Y,Z) )
        plots[2].set_segments( self.make_segments(X,Y,res) )
        images[0].set_data( data )
        images[1].set_data( Z )
        images[2].set_data( res )
        
        zlims = [Z.min(), Z.max()]
        rlims = [res.min(), res.max()]
        #plims = 0.25, 99.75                             #percentiles
        #clims = np.percentile( data, plims )            #colour limits for data
        #rlims = np.percentile( res, plims )             #colour limits for residuals
        for i, pl in enumerate( plots ):
            ax = pl.axes
            ax.set_zlim( zlims if (i+1)%3 else rlims )
        ax.set_xlim( [X[0,0],X[0,-1]] ) 
        ax.set_ylim( [Y[0,0],Y[-1,0]] )
        
        for i,im in enumerate(images):
            ax = im.axes
            im.set_clim( zlims if (i+1)%3 else rlims )
            #artificially set axes limits --> applies to all since share_all=True in constuctor
            im.set_extent( [X[0,0], X[0,-1], Y[0,0], Y[-1,0]] )
            
        #self.fig.canvas.draw()
        #TODO: SAVE FIGURES.................



#****************************************************************************************************        
class Compare3DContours(Compare3DImage):
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def setup_image_axes(self, fig):
        #Create the plot grid for the contour plots
        self.grid_contours = AxesGrid(fig, 212, # similar to subplot(211)
                                        nrows_ncols = (1, 3),
                                        axes_pad = 0.2,
                                        label_mode = 'L',          #This is necessary to avoid AxesGrid._tick_only throwing
                                        share_all = True)
        
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def update(self, X, Y, Z, data):
        '''update plots with new data.'''
        res = data - Z
        plots, images = self.plots, self.images
        
        plots[0].set_segments( self.make_segments(X,Y,data) )
        plots[1].set_segments( self.make_segments(X,Y,Z) )
        plots[2].set_segments( self.make_segments(X,Y,res) )
        #images[0].set_data( data )
        #images[1].set_data( Z )
        #images[2].set_data( res )
        
        for ax, z in zip(self.grid_contours, (data, Z, res)):
            cs = ax.contour(X, Y, z)
            ax.clabel(cs, inline=1, fontsize=7) #manual=manual_locations
        
        zlims = [Z.min(), Z.max()]
        rlims = [res.min(), res.max()]
        #plims = 0.25, 99.75                             #percentiles
        #clims = np.percentile( data, plims )            #colour limits for data
        #rlims = np.percentile( res, plims )             #colour limits for residuals
        for i, pl in enumerate( plots ):
            ax = pl.axes
            ax.set_zlim( zlims if (i+1)%3 else rlims )
        ax.set_xlim( [X[0,0],X[0,-1]] ) 
        ax.set_ylim( [Y[0,0],Y[-1,0]] )
        
        #for i,im in enumerate(images):
            #ax = im.axes
            #im.set_clim( zlims if (i+1)%3 else rlims )
            ##artificially set axes limits --> applies to all since share_all=True in constuctor
            #im.set_extent( [X[0,0], X[0,-1], Y[0,0], Y[-1,0]] )
            
        #self.fig.canvas.draw()






#====================================================================================================
def supershow(ax, data, *args, **kwargs):
    ss = ImageDisplay(ax, data, *args, **kwargs)
    ss.sliders.connect()
    return ss.imgplt
    

#from misc import make_ipshell
#ipshell = make_ipshell()
#ipshell()
if __name__=='__main__':
    import pylab as plt
                
    
    fig, ax = plt.subplots(figsize=(18,8))
    data = np.random.random((100,100))
    supershow(ax, data)
    
    
    #fig, ax = plt.subplots(1,1, figsize=(2.5, 10), tight_layout=True)
    #ax.set_ylim(0, 250)
    #sliders = AxesSliders(ax, 0.2, 0.7, slide_on='y')
    #sliders.connect()
    
    
    plt.show()


#class Imager(object):

    #def __init__(self, ax, z, x, y):
        #self.ax = ax
        #self.x  = x
        #self.y  = y
        #self.z  = z
        #self.dx = self.x[1] - self.x[0]
        #self.dy = self.y[1] - self.y[0]
        #self.numrows, self.numcols = self.z.shape
        #self.ax.format_coord = self.format_coord
    
    #def format_coord(self, x, y):
        #col = int(x/self.dx+0.5)
        #row = int(y/self.dy+0.5)
        ##print "Nx, Nf = ", len(self.x), len(self.y), "    x, y =", x, y, "    dx, dy =", self.dx, self.dy, "    col, row =", col, row
        #xyz_str = ''
        #if (col>=0 and col<self.numcols and row>=0 and row<self.numrows):
            #zij = self.z[row,col]
            ##print "zij =", zij, '  |zij| =', abs(zij)
            #if (np.iscomplex(zij)):
                #amp, phs = abs(zij), np.angle(zij) / np.pi
                #signz = '+' if (zij.imag >= 0.0) else '-'
                #xyz_str = 'x=' + str('%.4g' % x) + ', y=' + str('%.4g' % y) + ',' \
                            #+ ' z=(' + str('%.4g' % zij.real) + signz + str('%.4g' % abs(zij.imag)) + 'j)' \
                            #+ '=' + str('%.4g' % amp) + r'*exp{' + str('%.4g' % phs) + u' π j})'
            #else:
                #xyz_str = 'x=' + str('%.4g' % x) + ', y=' + str('%.4g' % y) + ', z=' + str('%.4g' % zij)
        #else:
            #xyz_str = 'x=%1.4f, y=%1.4f'%(x, y)
        #return xyz_str
        
    

#def supershow(ax, x, y, z, *args, **kwargs):
    
    #assert len(x) == z.shape[1]
    #assert len(y) == z.shape[0]
    
    #dx = x[1] - x[0]
    #dy = y[1] - y[0]
    #zabs = abs(z) if np.iscomplex(z).any() else z
    
    ## Use this to center pixel around (x,y) values
    #extent = (x[0]-dx/2.0, x[-1]+dx/2.0, y[0]-dy/2.0, y[-1]+dy/2.0)
    
    #im = ax.imshow(zabs, extent = extent, *args, **kwargs)
    #imager = Imager(ax, z, x, y)
    #ax.set_xlim((x[0], x[-1]))
    #ax.set_ylim((y[0], y[-1]))
    
    #return im
    
    
