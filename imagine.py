import numpy as np

import matplotlib
matplotlib.use('qt4agg')
from matplotlib.widgets import AxesWidget, Slider
from matplotlib.patches import FancyArrow, Circle
from matplotlib.transforms import Affine2D
from matplotlib.transforms import blended_transform_factory as btf

#from recipes.iter import grouper

from PyQt4.QtCore import pyqtRemoveInputHook, pyqtRestoreInputHook
from IPython import embed
#pyqtRemoveInputHook()
#embed()
#pyqtRestoreInputHook()


def picker(artist, event):
    #print(vars(event))
    mouse_position = (event.xdata, event.ydata)
    if None in mouse_position:
        return False, {}
    
    ax = artist.axes
    _data2ax_trans = ax.transData + ax.transAxes.inverted()
    mouse_ax_pos = _data2ax_trans.transform(mouse_position)
    
    centre = artist.get_centre()
   
    prox = np.linalg.norm(mouse_ax_pos - centre)
    hit = prox < 0.5
    print( 'mouse_position, mouse_ax_pos,  centre' )
    print( mouse_position, mouse_ax_pos, centre )       
    print( 'prox', prox )
    print( 'hit', hit )
    return hit, {}

#****************************************************************************************************
class AxesSliders(AxesWidget):
    #TODO:  USE DraggableLine to achieve what this class is doing????!!!!!!!!!
    #TODO:  DISPLAY ARROWS SIZES CORRECTLY -- INDEPENDENT OF AXES SCALing (LOG etc)!
    #TODO:  OPTION FOR LOGARITHMIC SLIDER AXIS
    #FIXME: scale independent picking....
    marker_size = 10
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __init__(self, ax, x1, x2, slide_on='x', valfmt='%1.2f',
                    closedmin=True, closedmax=True, dragging=True, **kwargs):
        
        AxesWidget.__init__(self, ax)           #NOTE: sets self.ax
        
        self.slide_on = slide_on
        self._index = int(slide_on == 'y')
        self._order = -1 if self._index else 1
        
        self.setup_axes(ax)
        #ax2data = ax.transAxes + ax.transData.inverted()
        
        
        #initial values of SliderArrows
        self._original_position = [x1, x2]
        self.valmin, self.valmax = ax.viewLim.get_points()[:, self._index]
        
        #transform position of sliders form data to axes coordinates
        self.positions = np.array([x1, x2]) #
        val_ax = [x1, x2] / (self.valmax - self.valmin)

        self.dragging = dragging
        #create sliders & add to axis
        
        #TODO: shift markers to point on axis
        self.axpos = axpos = 0.5
        ms = self.marker_size
        coo = [x1, self.axpos][::self._order]
        self.min_slider, = ax.plot(*coo, 
                                    marker='>', ms=ms, mfc='b', 
                                    picker=15)
        coo = [x2, self.axpos][::self._order]
        self.max_slider, = ax.plot(*coo, 
                                    marker='<', ms=ms, mfc='r', 
                                    picker=15)
        self.sliders = [self.min_slider, self.max_slider]
        #self.valfmt = valfmt
        
        ax.set_navigate(False)
        
        #self.label = ax.text(-0.02, 0.5, label, transform=ax.transAxes,
                             #verticalalignment='center',
                             #horizontalalignment='right')

        #self.valtext = ax.text(1.02, 0.5, valfmt % x1,
        #                       transform=ax.transAxes,
         #                      verticalalignment='center',
          #                     horizontalalignment='left')

        self.cnt = 0
        self.observers = {}

        self.closedmin = closedmin
        self.closedmax = closedmax
        self.selection = None
        self.which_active = None
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def setup_axes(self, ax):
        ''' '''
        #hide axis patch
        ax.patch.set_visible(0)
        #setup ticks
        self.ax.tick_params(axis=self.slide_on, direction='inout', which='both')
        axsli, axoth = [ax.xaxis, ax.yaxis][::self._order]
        axoth.set_ticks([])
        which_spine = 'right' if self._index else 'bottom'
        axsli.set_ticks_position(which_spine)
        #hide the axis spines
        ax.spines[which_spine].set_position(('axes', 0.5))
        for where, spine in ax.spines.items():
            if where != which_spine:
                spine.set_visible(0)
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _on_pick(self, event):
        #print( 'picked:', event.artist, id(event.artist) )
        #print( 'event.artist in self.sliders', event.artist in self.sliders )
        if event.artist in self.sliders:
            self.selection = event.artist
            self.which_active = self.sliders.index(self.selection)
            
            #TODO: connect motion event here
            
            #self._orig_pos = event.artist.get_val()
            #print( 'which active:', self.which_active )

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _update(self, event):
        """update the slider position"""
        #print(11111)
        if self.ignore(event):
            return
        #print(2*11111)
        if event.button != 1:
            return
         
        #print(3*11111)
        if self.selection is None:
            return
        #print(2*22222)
        #motion_notify_event handled below
        self._on_motion(event)
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _on_release(self, event):
        #print( 'releasing..', event.name )
        #event.canvas.release_mouse(self.ax)
        self.selection = None
        self.which_active = None
        
        #TODO: disconnect motion event here
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _on_motion(self, event):
        ''' '''
        val = [event.xdata, event.ydata][self._index]
        val = self.validate(self.selection, val)
        
        #print( val, 'validated'  )
        
        if val:
            #print( '!' *10 )
            self.set_val(val)
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def validate(self, slider, val):
        '''check if new slider position value is OK'''
        
        if val is None:     #out of axis
            return

        if val <= self.valmin:
            if not self.closedmin:
                return
            val = self.valmin
        
        elif val >= self.valmax:
            if not self.closedmax:
                return
            val = self.valmax
        
        min_pos, max_pos = self.positions
        if (((slider is self.min_slider) and (val >= max_pos)) or 
            ((slider is self.max_slider) and (val <= min_pos))):
            return
        
        return val
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #def update_slider(self, val):
        #self.selection.set_val(val - self._orig_pos)
        
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def set_val(self, val):
        #FIXME!
        '''set value of active slider '''
        #self.selection.set_val(self.data2ax(val))       #convert to axis_coordinates
        xy = [val, self.axpos][::self._order]
        self.selection.set_data(xy)
        
        #self._orig_pos = self.selection.get_val()
        #self.valtext.set_text(self.valfmt % val)
        
        if self.drawon:
            self.ax.figure.canvas.draw()
        
        self.positions[self.which_active] = val
        
        if not self.eventson:
            return
        
        for cid, func in self.observers.items():
            func(self.positions)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def set_positions(self, values):
        #FIXME!
        mem = self.selection
        for slider, val in zip(self.sliders, values):
            xy = [val, self.axpos][::self._order]
            
            sval = self.validate(slider, val)
            if sval:
                self.selection = slider
                self.which_active = self.sliders.index(self.selection)
                self.set_val(sval)
           
        self.selection = mem
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def on_changed(self, func):
        """
        When the slider value is changed, call *func* with the new
        slider position

        A connection id is returned which can be used to disconnect
        """
        cid = self.cnt
        self.observers[cid] = func
        self.cnt += 1
        return cid
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def connect(self):
        '''connect events'''
        self.connect_event('pick_event', self._on_pick)
        #self.connect_event('button_press_event', self.)
        self.connect_event('button_release_event', self._on_release)
        if self.dragging:
            self.connect_event('motion_notify_event', self._update)
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def disconnect(self, cid):
        """remove the observer with connection id *cid*"""
        try:
            del self.observers[cid]
        except KeyError:
            pass

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def reset(self):
        """reset the slider to the initial value if needed"""
        if (self.val != self.valinit):
            self.set_val(self.valinit)


class ColourSliders(AxesSliders):
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __init__(self, ax, x1, x2, slide_on='x', valfmt='%1.2f',
                 closedmin=True, closedmax=True, dragging=True, **kwargs):
        ''' '''
        super().__init__(ax, x1, x2, slide_on, valfmt,
                         closedmin, closedmax, dragging, **kwargs)
        
        #data2fig_trans = self.ax.transData + self.ax.transFigure.inverted()
        #data2fig_trans.transform
        pos = self.positions.mean()
        coo = [pos, self.axpos][::self._order]
        self.centre_knob, = ax.plot(*coo, marker='o', ms=self.marker_size, mfc='g', 
                                    picker=10)
        self.centre_knob.set_clip_on(False)
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def set_positions(self, values):
        AxesSliders.set_positions(self, values)
        xy = [self.positions.mean(), self.axpos][::self._order]
        self.centre_knob.set_data(xy)
        
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _on_pick(self, event):
        if event.artist is self.centre_knob:
            self.selection = event.artist
        else:
            super()._on_pick(event)
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _on_motion(self, event):
        #FIXME!
        val = [event.xdata, event.ydata][self._index]
        
        if val is None:     #out of axis
            return
        
        
        _drawon = self.drawon
        if self.selection == self.centre_knob:
            self.drawon = False
            current = self.centre_knob.get_xydata()[0, self._index]
            offset = val - current
            #update positions of sliders
            #HACK!!
            for slider, sval in zip(self.sliders, self.positions):
                sval = self.validate(slider, sval + offset)
                if sval:
                    self.selection = slider
                    self.which_active = self.sliders.index(self.selection)
                    self.set_val(sval)
                else:
                    self.selection = self.centre_knob
                    return
                
            xy = [self.positions.mean(), self.axpos][::self._order]
            self.centre_knob.set_data(xy)
            
            self.selection = self.centre_knob
            
        else:
            super()._on_motion(event)
            
            xy = [self.positions.mean(), self.axpos][::self._order]
            self.centre_knob.set_data(xy)
            
        #TODO: BLIT:
        self.ax.figure.canvas.draw()
        self.drawon = _drawon
            
            
    
    

            
#TODO checkout APLPY?????????
from mpl_toolkits.axes_grid1 import make_axes_locatable
#****************************************************************************************************
class ImageDisplay(object):
    #TODO: Option for data histogram on slider bar!!!
    #TODO: Choose figure geometry based on data shape
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    SliderClass = ColourSliders #AxesSliders
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __init__(self, ax, data, *args, **kwargs):
        ''' '''
        #self.sscale = kwargs.pop('sscale', 'linear')
        self.data = data = np.atleast_2d(data)
        self.ax = ax
        self.ax.format_coord = self.cooDisplayFormatter
        
        #set the clims vmin and vmax in kwargs according to requested autoscaling
        kwargs = self.update_autoscale_limits(data, **kwargs)
        
        #set the axes title if given
        title = kwargs.pop('title', None )
        if not title is None:
            ax.set_title( title )
        
        #use imshow to do the plotting
        self.imgplt = ax.imshow(data, *args, **kwargs)
        
        #create the colourbar and the AxesSliders
        self.divider = make_axes_locatable(ax)
        self.Createcolorbar()
        self.CreateSliders()
    
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
        self.ax.figure.colorbar( self.imgplt, cax=self.cax)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def CreateSliders(self):
        sax = self.divider.append_axes('right', size=0.2, pad=0.5)
        sax.set_ylim((self.data.min(), self.data.max()))
        #sax.set_yscale( self.sscale )

        self.sliders = self.SliderClass(sax, *self.imgplt.get_clim(), slide_on='y' )
        self.sliders.drawon = False
        self.sliders.on_changed(self.set_clim)
   
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def set_clim(self, clims):
        self.imgplt.set_clim(clims)
        #self.draw_blit()
        #self.background = ax.figure.canvas.copy_from_bbox(ax.bbox)
        self.imgplt.figure.canvas.draw()        #TODO: BLIT!!!!!
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def draw_blit(self):
        #FIXME!
        fig = self.ax.figure
        fig.canvas.restore_region(self.background)
        
        self.ax.draw_artist(self.imgplt)
        self.ax.draw_artist(self.sliders.selection)
        
        fig.canvas.blit(self.ax.bbox)
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

    
#****************************************************************************************************
class CubeDisplayBase(ImageDisplay):
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    aperture_properties = dict(radii=10, ec='m', lw=2)
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
        
        #setup frame slider
        self._frame = 0
        self.fsax = self.divider.append_axes('bottom', size=0.2, pad=0.25)
        self.frame_slider = Slider(self.fsax, 'frame', 0, len(self), valfmt='%d')
        self.frame_slider.on_changed(self.set_frame)
        
        #create apertures if coordinates provided
        self.aps = None
        self.coords = coords
        self.has_coords = not coords is None
        if self.has_coords:
            from ApertureCollections import ApertureCollection
            #add apertures to axes.  will not display yet as coordinates not set
            self.aps = ApertureCollection( **self.aperture_properties )
            self.aps.axadd(ax)
            
        #save background for blitting
        fig = ax.figure
        self.background = fig.canvas.copy_from_bbox(ax.bbox)

        #enable frame scroll
        fig.canvas.mpl_connect('scroll_event', self._scroll)
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _scroll(self, event):
        self.frame += [-1, +1][event.button == 'up']
        self.frame_slider.set_val(self.frame)
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def draw_blit(self, artists):
        fig = self.ax.figure
        fig.canvas.restore_region(self.background)
        
        for art in artists:
            self.ax.draw_artist(art)
        
        fig.canvas.blit(fig.bbox) #FIXME: what about colorbar?
        
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
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def get_frame(self):
        return self._frame

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def set_frame(self, i):
        i %= len(self)  #wrap around!
        self.draw_blit(self.data[i])
        
        self._frame = i

    frame = property(get_frame, set_frame)



#****************************************************************************************************
from fastfits import FITSFrame
from recipes.iter import interleave
class FITSCubeDisplay(CubeDisplayBase, FITSFrame):
    #TODO: frame switch buttons; 
    #TODO: option to set clim from first frame??
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __init__(self, ax, filename, coords=None, **kws):
        ''' '''
        #setup data access
        FITSFrame.__init__(self, filename)
        
        sx = self.shape
        extent = interleave((0,)*len(sx), sx)
        CubeDisplayBase.__init__(self, ax, [[0]], 
                                  coords,
                                  origin='llc',
                                  extent=extent,
                                  **kws)
        
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def get_frame(self):
        return self._frame

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def set_frame(self, i):
        
        self._frame = i
        data = self[int(i%len(self))]
        if self.autoscale:
            #set the slider axis limits
            dmin, dmax = data.min(), data.max()
            self.sliders.ax.set_ylim(dmin, dmax)
            self.sliders.valmin, self.sliders.valmax = dmin, dmax
            
            #set the slider positiions / color limits
            vmin, vmax = self.get_autoscale_limits(data, autoscale=self.autoscale)
            self.imgplt.set_data(data)
            self.imgplt.set_clim(vmin, vmax)
            self.sliders.set_positions((vmin, vmax))
        
        #update the apertures if needed
        if self.has_coords:
             self.aps.coords = self.coords[:, i, :]
        
        #TODO: BLIT!!
        self.ax.figure.canvas.draw()
        #self.draw_blit([self.imgplt, self.aps])
        

    frame = property(get_frame, set_frame)


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
    
    
