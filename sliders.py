import operator

import numpy as np #NOTE: might be unnecessary

from matplotlib.widgets import AxesWidget, Slider
from matplotlib.patches import Circle

from matplotlib.transforms import Affine2D
from matplotlib.transforms import blended_transform_factory as btf


from .interactive import ConnectionMixin, mpl_connect
from draggables.machinery import DraggableBase, DragMachinery

from decor import expose

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
#class Slider(DraggableBase):
    ##def __init__(self):
        
    
    #def get_val(self):
        #y, = self.ref_art.get_data()[1] #TODO: slide_on x?
        #return float(y + self.offset)  #NOTE the value of offset is not updated during drag
    
    #def validate(self, val):
        #'''validate in data coordinates'''
        #current = self.get_val()
        

    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #def bind(self, interval):
        #i0, i1 = sorted(interval)
        #lambda o: i0 <= o <= i1
        
        ##self.validation()
    
    #def unbind(self, interval):
        #pass
    
    
#class SlideMachinery(DragMachinery):
    #@staticmethod
    #def artist_factory(art, offset, annotate, **kws):
        #return art, Slider(art, offset, annotate, **kws)
    
    
#****************************************************************************************************
class AxesSliders(DragMachinery): #ConnectionMixin, AxesWidget, #NOTE: ConnectionMixin renders AxesWidget somewhat redundant
    #TODO:  OPTION FOR LOGARITHMIC SLIDER AXIS
    #FIXME: fast dragging past limits ==> set slider value to limit!?
    #TODO: sliders in axes coordinates... will make it easier to plot othes artists on the same axes
    #machinery = SlideMachinery
    marker_size = 10
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __init__(self, ax, x1, x2, bounds=None, slide_on='x', valfmt='%1.2f',
                    closedmin=True, closedmax=True, dragging=True, use_blit=True, **kwargs):
        
        self.ax = ax
        #AxesWidget.__init__(self, ax)          #NOTE: sets self.ax, self.canvas etc
        
        self.slide_on = slide_on
        self._index = int(slide_on == 'y')
        self._locked = 'yx'[self._index]
        self._order = slice(None, None, -1 if self._index else 1)
        
        self.axpos = ax.viewLim.get_points()[int(not self._index), 0]#1.0    #set the axis spine position #TODO: more descriptive name for this thing
        #print(ax.get_xlim(), ax.get_ylim())
        self.setup_axes(ax)
        
        #ax2data = ax.transAxes + ax.transData.inverted()
        
        
        #initial values
        #self.positions  = np.array([x1, x2]) #x1, x2 = vals
        self._original_position = np.array([x1, x2])
        
        #set the bounds #IS THIS EXPLICITLY NECESSARY??
        self._bounds = bounds
        if bounds is not None:
            self._bounds = sorted(bounds)
            
        
        #transform position of sliders form data to axes coordinates
        #val_ax = [x1, x2] / (self.valmax - self.valmin)

        self.dragging = dragging
        #create sliders & add to axis
        
        ms = self.marker_size
        coo1 = [x1, self.axpos][self._order]
        coo2 = [x2, self.axpos][self._order]
        
        #transform = btf(*[ax.transAxes, ax.transData][self._order])
        
        markers = ax.plot(*coo1, 'b>',
                          *coo2, 'r<',                      #note: python3.5 syntax only
                          ms=ms, picker=15, clip_on=False,)  #plot to outside axes
                          #transform=transform)

        #self.sliders = [self.min_slider, self.max_slider]
        #self.valfmt = valfmt
        
        DragMachinery.__init__(self, markers, annotate=False)
        self.sliders = self.min_slide, self.max_slide = list(self)
        for slide in self.sliders:
            slide.lock(self._locked)
        
        opg = operator.ge if closedmin else operator.gt
        self.min_slide.validation(lambda xy: opg(xy[self._index], self.boundmin))
        self.min_slide.validation(lambda xy: xy[self._index] < self.positions[1])
        
        opl = operator.le if closedmax else operator.lt
        self.max_slide.validation(lambda xy: opl(xy[self._index], self.boundmax))
        self.max_slide.validation(lambda xy: xy[self._index] > self.positions[0])
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def get_positions(self):
        return self._original_position + self.offsets[:, self._index]
    positions = property(get_positions)
    
    
    def set_positions(self, values):
        #mem = self.selection
        
        #current = self.offsets[:, self._index]
        offsets = values - self._original_position
        current = self.offsets[:]
        current[:, self._index] = offsets
        
        for slider, off in zip(self.sliders, current):
            slider.shift(off)
            slider.offset = off
            
            
            
            
            #sval = self.validate(slider, val)
            #if sval:
                #self.selection = slider
                #self.which_active = self.sliders.index(self.selection)
                #self.set_val(sval)
           
        #self.selection = mem
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def get_bounds(self):
        #let slider limits be set by axes limits.
        if self._bounds is None:
            return self.ax.viewLim.get_points()[:, self._index]
        return self._bounds
    
    def set_bounds(self, val):
        self._bounds = val
    
    bounds = property(get_bounds, set_bounds)
    
    def get_boundmax(self):
        return self.bounds[1]
    boundmax = property(get_boundmax)
    
    def get_boundmin(self):
        return self.bounds[0]
    boundmin = property(get_boundmin)
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #def _active(self):
        #return self.sliders.index(self.selection)
    #which_active = property(
    
    def index(self):
        return [slide.ref_art for slide in self.sliders].index(self.selection)
        
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #def get_valmin(self):
        #return self.bounds[0]
    #valmin = property(get_valmin)
    
    #def get_valmax(self):
        #return self.bounds[1]
    #valmax = property(get_valmax)
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def setup_axes(self, ax):
        ''' '''
        ax.set_navigate(False)      #turn off nav for this axis
        #ax.patch.set_visible(0)     #hide axis patch
        #setup ticks
        #self.ax.tick_params(axis=self.slide_on, direction='inout', which='both')
        #axsli, axoth = [ax.xaxis, ax.yaxis][self._order]
        #axoth.set_ticks([])
        #which_spine = 'right' if self._index else 'bottom'
        #axsli.set_ticks_position(which_spine)
        #hide the axis spines
        #ax.spines[which_spine].set_position(('axes', self.axpos))
        #for where, spine in ax.spines.items():
            #if where != which_spine:
                #spine.set_visible(0)
    
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #@mpl_connect('pick event')
    #def _on_pick(self, event):
        ##print( 'picked:', event.artist, id(event.artist) )
        ##print( 'event.artist in self.sliders', event.artist in self.sliders )
        #if event.artist in self.sliders:
            #self.selection = event.artist
            #self.which_active = self.sliders.index(self.selection)
            
            ##TODO: connect motion event here
            
            ##self._orig_pos = event.artist.get_val()
            ##print( 'which active:', self.which_active )

    ##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #@mpl_connect('motion_notify_event')
    #def _update(self, event):
        #"""update the slider position"""
        #if self.ignore(event):
            #return
        
        #if event.button != 1:
            #return
         
        #if self.selection is None:
            #return
        
        ##motion_notify_event handled below
        #self._on_motion(event)
    
    ##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #def _on_motion(self, event):
        #''' '''
        #val = [event.xdata, event.ydata][self._index]
        #val = self.validate(self.knobs.selection, val)
        
        ##print( val, 'validated'  )
        
        #if val:
            ##print( '!' *10 )
            #self.set_val(val)
    
    ##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #def validate(self, slider, val):
        #'''check if new slider position value is OK'''
        
        #if val is None:     #out of axis
            #return

        #if val <= self.valmin:
            #if not self.closedmin:
                #return
            #val = self.valmin
        
        #elif val >= self.valmax:
            #if not self.closedmax:
                #return
            #val = self.valmax
        
        #min_pos, max_pos = self.positions
        #if (((slider is self.min_slider) and (val >= max_pos)) or 
            #((slider is self.max_slider) and (val <= min_pos))):
            #return
        
        #return val
    
    ##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #def _on_release(self, event):
        ##print( 'releasing..', event.name )
        ##event.canvas.release_mouse(self.ax)
        #self.selection = None
        #self.which_active = None
        
        ##TODO: disconnect motion event here
    
    
    ##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ##def update_slider(self, val):
        ##self.selection.set_val(val - self._orig_pos)
        
    ##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #def set_val(self, val):
        ##print('AxesSliders.set_val')
        ##FIXME!
        #'''set value of active slider '''
        ##self.selection.set_val(self.data2ax(val))       #convert to axis_coordinates
        #xy = [val, self.axpos][self._order]
        #self.selection.set_data(xy)
        
        ##self._orig_pos = self.selection.get_val()
        ##self.valtext.set_text(self.valfmt % val)
        
        #if self.drawon:
            #print('DRAW!')
            #self.ax.figure.canvas.draw()
        
        #self.positions[self.which_active] = val
        
        #if not self.eventson:
            #return
        
        #for cid, func in self.observers.items():
            #func(self.positions)

    ##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #def set_positions(self, values):
        ##FIXME!
        #mem = self.selection
        #for slider, val in zip(self.sliders, values):
            #xy = [val, self.axpos][self._order]
            
            #sval = self.validate(slider, val)
            #if sval:
                #self.selection = slider
                #self.which_active = self.sliders.index(self.selection)
                #self.set_val(sval)
           
        #self.selection = mem
    
    ##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #def on_changed(self, func):
        #"""
        #When the slider value is changed, call *func* with the new
        #slider position

        #A connection id is returned which can be used to disconnect
        #"""
        #cid = self.cnt
        #self.observers[cid] = func
        #self.cnt += 1
        #return cid
    
    ##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #def connect(self):
        #'''connect events'''
        #self.connect_event('pick_event', self._on_pick)
        ##self.connect_event('button_press_event', self.)
        #self.connect_event('button_release_event', self._on_release)
        #if self.dragging:
            #self.connect_event('motion_notify_event', self._update)
    
    ##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #def disconnect(self, cid):
        #"""remove the observer with connection id *cid*"""
        #try:
            #del self.observers[cid]
        #except KeyError:
            #pass

    ##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #def reset(self):
        #"""reset the slider to the initial value if needed"""
        #if (self.val != self.valinit):
            #self.set_val(self.valinit)


#SlideMachinery

class ColourSliders(AxesSliders):
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #machinery = 
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __init__(self, ax, x1, x2, bounds=None, slide_on='x', valfmt='%1.2f',
                 closedmin=True, closedmax=True, dragging=True, **kwargs):
        ''' '''
        AxesSliders.__init__(self, ax, x1, x2, bounds, slide_on, valfmt,
                         closedmin, closedmax, dragging, **kwargs)
        
        #data2fig_trans = self.ax.transData + self.ax.transFigure.inverted()
        #data2fig_trans.transform
        pos = sum(self.positions) / 2.  #midpoint of sliders
        
        coo = [pos, self.axpos][self._order]
        marker, = ax.plot(*coo, 'go', 
                        ms=self.marker_size, 
                        picker=10, clip_on=False)
        
        self._original_position = np.r_[self.positions, pos]
        self.centre_knob = self.add(marker,  annotate=False)
        self.centre_knob.lock(self._locked)
                
        self.centre_knob.validation(lambda xy: 
            self.positions[0] < xy[self._index] < self.positions[1])
        
        
        #self.centre_knob.on_changed(self.centre_shift)
        
        #self.centre_knob.on_changed(self.max_slide.validate_and_shift)
        
        #self.centre_knob.set_clip_on(False)
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def centre_shift(self, xydata):
        
        #cdat = xydata#[self._index]
        mmo = np.ptp(self.positions) / 2
        #minmax = np.add(cdat, self.positions)
        
        delta = xydata - self.ref_point
        #print('xydata', xydata)
        #print('delta', delta)
        #self.positions
        #self.canvas.restore_region(self.background)
        
        for slide in (self.min_slide, self.max_slide):
            #print('minmax', xydata + mmo)
            if slide.validate(xydata + mmo):
                #print('mmo', mmo)
                #print('slide.offset', slide.offset)
                #print()
                slide.shift(slide.offset + delta)
                slide.draw()
            #print()
        
        self.canvas.blit(self.figure.bbox)
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #@mpl_connect('pick_event')
    #def on_pick(self, event): 
        #'''Pick event handler.'''
        
        
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @mpl_connect('button_release_event')
    def on_release(self, event):
                
        if event.button != 1:
            return
        
        #AxesSliders.on_release(self, event)
        
        if self.selection == self.centre_knob.ref_art:
            print('centre release')
            for slide in self.sliders:
                print('sliding away', self.delta)
                slide.shift(slide.offset + self.delta)
                slide.offset = self.delta
                slide.ref_art.set_animated(False)       #FIXME: This precludes setting by an external method without auto draw by mpl.....
                slide.draw()
            
            #self.canvas.blit(self.figure.bbox)
            #self.canvas.draw()
            
        #for drag in self:
                #drag.ref_art.set_animated(True)
        #self._draw_on = False
        #AxesSliders.on_release(self, event)
        #self._draw_on = True
        #self.canvas.draw()
        #self.canvas.blit(self.figure.bbox)
    
    ##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #def set_positions(self, values):
        #AxesSliders.set_positions(self, values)
        #xy = [self.positions.mean(), self.axpos][self._order]
        #self.centre_knob.set_data(xy)
        
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @mpl_connect('pick_event')
    def on_pick(self, event):
        print('picking')
        if event.artist is self.centre_knob.ref_art:
            for drag in self.sliders:
                drag.ref_art.set_animated(True)
            #self.canvas.draw()
            #self.background = self.canvas.copy_from_bbox(self.figure.bbox)
        
        #self._draw_on = False
        AxesSliders.on_pick(self, event)
        #self._draw_on = True
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def on_motion(self, event):
        
        if event.button != 1:
            return
        
        #_drawon = self.drawon
        if self.selection == self.centre_knob.ref_art:
            #print('GOGOGO')
            print('moving')
            self.canvas.restore_region(self.background)
            self._draw_on = False
            AxesSliders.on_motion(self, event)
            
            #offset = val - current
            #update positions of sliders
            #HACK!!
            for slide, val in zip(self.sliders, self.positions):
                if slide.validate(val + self.delta):
                    slide.shift(slide.offset + self.delta)
                    #TODO: observers for this one
                    
                    slide.draw()
            
            self.centre_knob.draw()
            #xy = [self.positions.mean(), self.axpos][self._order]
            #self.selection = self.centre_knob
            self._draw_on = True
            
        else:
            AxesSliders.on_motion(self, event)
            
            xy = [self.positions.mean(), self.axpos][self._order]
            #self.centre_knob.set_data(xy)
            
        self.canvas.blit(self.figure.bbox)
        
        #TODO: BLIT:
        #self.ax.figure.canvas.draw()
        #self.drawon = _drawon