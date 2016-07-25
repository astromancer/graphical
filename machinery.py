import numpy as np

#import itertools as itt
from collections import OrderedDict
#from matplotlib.offsetbox import DraggableBase

from matplotlib.container import ErrorbarContainer
from matplotlib.lines import Line2D

from matplotlib.transforms import blended_transform_factory as btf
from matplotlib.transforms import Affine2D

from grafico.interactive import ConnectionMixin, mpl_connect
from recipes.iter import flatiter
from decor import expose

#====================================================================================================
def draggable_artist_factory(art, offset, annotate, **kws):
    if isinstance(art, ErrorbarContainer):
        from draggables.errorbars import DraggableErrorbarContainer
        draggable = DraggableErrorbarContainer(art, 
                                               offset=offset,
                                               annotate=annotate,
                                               **kws)
        markers, _, _  = draggable
        return markers, draggable #map Line2D to DraggableErrorbarContainer. The picker unavoidably returns the markers.
    
    if isinstance(art, Line2D):
        #from draggables.lines import DraggableLine
        return art, DraggableBase(art, offset, annotate, **kws)

#====================================================================================================
@expose.args()
def fpicker(artist, event):
    ''' an artist picker that works for clicks outside the axes'''
    #print('line picker')
    
    if event.button != 1:           #otherwise intended reset will select
        print('wrong button asshole!')
        return False, {}
    
    props = {}
    
    transformed_path = artist._get_transformed_path()
    path, affine = transformed_path.get_transformed_path_and_affine()
    path = affine.transform_path(path)
    xy = path.vertices
    xt = xy[:, 0]
    yt = xy[:, 1]
    
    # Convert pick radius from points to pixels
    pixels = artist.figure.dpi / 72. * artist.pickradius
    
    xd, yd = xt - event.x, yt - event.y
    prox = xd ** 2 + yd ** 2           #distance of click from points in pixels (display coords)
    picked = prox - pixels**2 < 0
    
    #print(picked, '!!')
    
    return picked.any(), props


#****************************************************************************************************
class IndexableOrderedDict(OrderedDict):
    def __missing__(self, key):
        if isinstance(key, int):
            return self[list(self.keys())[key]]
        else:
            return OrderedDict.__missing__(self, key)


#****************************************************************************************************
class DraggableBase():
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    annotation_format = '[%+3.2f]'
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #@classmethod
    #def draggable_artist_factory(cls, art, offset, annotate, **kws):
        #if isinstance(art, ErrorbarContainer):
            #from draggables.errorbars import DraggableErrorbarContainer
            #draggable = DraggableErrorbarContainer(art, 
                                                #offset=offset,
                                                #annotate=annotate,
                                                #**kws)
            #markers, _, _  = draggable
            #return markers, draggable #map Line2D to DraggableErrorbarContainer. The picker unavoidably returns the markers.
        
        #if isinstance(art, Line2D):
            ##from draggables.lines import DraggableLine
            #return art, cls(art, 
                            #offset=offset, 
                            #annotate=annotate,
                            #**kws)
        
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __init__(self, artist, offset=(0.,0.), annotate=True, haunted=False, **kws):
        ''' '''
        
        #Line2D.__init__(self, *line.get_data())
        #self.update_from(line)
        self.ref_art = artist
        
        self.offset = np.array(offset)
        self.ref_point = offset
        self.annotated = annotate
        #haunted = 
        
        self._original_transform = artist.get_transform()
        
        #make the lines pickable
        if not artist.get_picker():
            artist.set_picker(fpicker)
        
        #
        #self.ref_art.set_animated(True)
        #self._draw_on = True
        
        #Manage with ConnectionMixin?
        self.observers = {}
        self.validators = {}
        #self.observers_active = True
        self.cnt = 0
        self.vnt = 0
        
        #Initialize offset texts
        ax = artist.axes
        if self.annotated:
            self.text_trans = btf(ax.transAxes, ax.transData)
            self.ytxt = artist.get_ydata().mean()
            self.annotation = ax.text(1.005, self.ytxt, '')
            self.on_changed(self.shift_text)
        
        if haunted:
            self.haunt()
        
        #self._locked = np.zeros(2, bool)
        self._locked_on = np.empty(2)
        self._locked_on.fill(None)
        
        #shift to the given offset (default 0)
        self.shift(self.offset)
        
        
        
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __str__(self):
        return 'Draggable' + str(self.ref_art) #TODO if bounded show bounds
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def save_offset(self):
        self.ref_point = self.offset
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #def update_offset(self, offset):
        
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def validation(self, func):
        '''add validator'''
        vid = self.vnt
        self.validators[vid] = func
        self.vnt += 1
        return vid
    
    #@expose.returns()
    #@expose.args(pre='='*50, post='*'*50)
    def validate(self, xydata):
        '''check validity'''
        for cid, func in self.validators.items():    #six.iteritems(self.observers):
            #print(func, offset)
            if ~func(xydata):
                return False
        else:
            return True
        
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def shift(self, offset, observers_active=True):
        #TODO: should this method handle data coordinates??
        '''Shift the artist by an offset'''
        offset = np.where(np.isnan(self._locked_on), offset, self._locked_on)
        
        #add the offset with transform
        offset_trans = Affine2D().translate(*offset)
        trans = offset_trans + self._original_transform
        self.ref_art.set_transform(trans)
        
        #x, y = self.ref_art.get_data()
        #if observers_active:
            #for cid, func in self.observers.items():    #six.iteritems(self.observers):
                #func(offset) #OR pass data coordinates here??
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #def update(self, delta):
        #'''shift and draw'''
        #self.shift(delta)
        #artists = filter(None, 
                        #(func(xydata) for cid, func in self.observers.items()))    #six.iteritems(self.observers):
        #if self._draw_on:
            #fig = self.ref_art.figure
            #canvas = fig.canvas
            #if self._use_blit:
                #canvas.restore_region(self.background)
                #for art in flatiter((artists, self.ref_art)): #WARNING: will flatten containers etc
                    #art.draw(canvas.renderer)
                #canvas.blit(fig.bbox)
            #else:
                #canvas.draw()
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #def validate_and_shift(self, delta, xydata):
            
        #if not self.validate(xydata):
            ##print('NOT VALID', self.offset)
            #return
        
        #self.shift(delta)
        ##self.offset = delta
        ##self.canvas.restore_region(self.background)
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #NOTE: this function can be avoided if you make a DraggableText?
    def shift_text(self, offset):
        '''Shift the annotation by an offset'''
        #offset = val - self.ytxt
        offset_trans = Affine2D().translate(*offset)
        trans = offset_trans + self._original_transform
        
        txt = self.annotation
        txt.set_transform(offset_trans + self.text_trans)
        txt.set_text(self.annotation_format % offset)
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def lock(self, which):
        ix = 'xy'.index(which.lower())
        self._locked_on[ix] = self.offset[ix]
    
    def free(self, which):
        ix = 'xy'.index(which.lower())
        self._locked_on[ix] = None
    
    def lock_x(self):
        self.lock('x')
    
    def free_x(self):
        self.free('x')
    
    def lock_y(self):
       self.lock('y')
    
    def free_y(self):
        self.free('y')
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def on_changed(self, func):
        """
        When the artist is dragged, call *func* with the new coordinate position

        A connection id is returned which can be used to disconnect
        """
        cid = self.cnt
        self.observers[cid] = func
        self.cnt += 1
        return cid
     
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def haunt(self):
        ''' '''
        #create a ghost artist of the same class
        self.ghost = self.__class__(self.ref_art, offset=self.offset,
                                            annotate=self.annotated)
        
        self.ghost.set_alpha(0.2)
        self.ghost.set_visible(False)
        ax.add_artist(self.ghost)
        
        #NOTE: can avoid this if statement by subclassing...
        if self.annotated:
            self.ghost.annotation.set_alpha(0.2)
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #TODO: inherit from the artist to avoid redefining these methods????
    def set_transform(self, trans):
        self.art.set_transform(trans)
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def set_visible(self, vis):
        self.art.set_visible(vis)
        
        #NOTE: can avoid this if statement by subclassing...
        if self.annotated:
            self.annotation.set_visible(vis)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #def draw(self):
        #renderer = self.ref_art.figure.canvas.renderer
        #self.ref_art.draw(renderer)
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #def get_xdata()
    #def get_ydata() etc...
    


#****************************************************************************************************
class DragMachinery(ConnectionMixin): 
    #TODO: haunt, link, lockx/y 
    #TODO: one way links
    #TODO: #incorp best methods from mpl.DraggableBase
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    supported_artists = [Line2D, ErrorbarContainer]
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @staticmethod
    def artist_factory(art, offset, annotate, **kws):
        return draggable_artist_factory(art, offset, annotate, **kws)
    
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __init__(self, plots, offsets=None, annotate=True, linked=None, haunted=False,
                 auto_legend=True, use_blit=True, **legendkw):
        self.selection          = None
        self.ref_point       = None
        
        if offsets is None:
            offsets = np.zeros((len(plots),2))
        self._original_offsets = offsets                #remember for reset
       
        self.delta              = np.zeros(2)           #in case of pick without motion
        
        self.ax = ax            = plots[0].axes
        self.figure             = ax.figure
        
        #initialize mapping
        self.draggables = IndexableOrderedDict() 
        
        #initialize auto-connect
        ConnectionMixin.__init__(self, ax.figure)
        
        #flag for blitting behaviour
        self._draw_on = True
        self._use_blit = use_blit and self.canvas.supports_blit
        
        #esure linked argument is a nested list
        if linked is None:
            linked = []         #lists as default args hide errors
        elif len(linked) and isinstance(linked[0], self.supported_artists): #HACK
            linked = [linked]   #np.atleast_2d does not work because it unpacks the ErrorbarContainer
        else:
            linked = list(linked)
            
        _friends = [f for _, *friends in linked for f in friends]
        
        #build the draggable objects
        for plot, offset in zip(plots, offsets):
            #only annotate the "first" linked artist
            ann = annotate and (not plot in _friends)
            
            key, drg = self.artist_factory(plot, 
                                            offset=offset,
                                            annotate=ann,
                                            haunted=haunted)
            self.draggables[key] = drg
        
        #TODO:
        ##enable legend picking
        #self.legend = None
        #if legendkw or auto_legend:
            #self.legend = DynamicLegend(ax, plots, legendkw)
            #self.legend.connect()
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __getitem__(self, key):
        '''hack for quick indexing'''       #OR inherit from dict????
        return self.draggables[key]
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def get_offsets(self):
        return np.array([drag.offset for drag in self])
    
    offsets = property(get_offsets)
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def add(self, artist, offset=(0,0), annotate=True, haunted=False): #TODO: linked=None,
        '''add a draggable artist'''
        
        key, drg = self.artist_factory(artist, 
                                        offset=offset,
                                        annotate=annotate,
                                        haunted=haunted)
        self.draggables[key] = drg
        self._original_offsets = np.r_['0,2', self._original_offsets, offset]
        
        return drg
        
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def reset(self):
        '''reset the plot positions to zero offset'''
        #print('resetting!')
        for draggable, off in zip(self.draggables.values(), self._original_offsets):
            draggable.shift(off)
            draggable.offset = off
        
        self.canvas.draw()
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @mpl_connect('button_press_event')
    def on_click(self, event):
        
        #print( 'on_click', repr(self.selection ))
        
        '''reset plot on middle mouse'''
        if event.button == 2:
            self.reset()
        else:
            return
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @mpl_connect('pick_event')
    def on_pick(self, event): 
        '''Pick event handler.'''
        
        if event.artist in self.draggables:
            
            #print('picked!!!!!')
            
            ax = self.ax
            self.selection = event.artist
            draggable = self.draggables[self.selection]
            
            xydisp = event.mouseevent.x, event.mouseevent.y     #xy in display coordinates
            xy = ax.transData.inverted().transform(xydisp)      #xy in data coordinates
            
            #connect motion_notify_event for dragging the selected artist
            self.add_connection('motion_notify_event', self.on_motion)
            
            #TODO: option for connecting other methods here
            
            if self._use_blit:
                #print('BURP!!!')
                #setup for blitting
                self.selection.set_animated(True)
                self.canvas.draw()                  #NECESSARY??
                self.background = self.canvas.copy_from_bbox(self.figure.bbox)  #save the background for blitting
                #draggable.draw()                   #NECESSARY??
                #self.canvas.blit(self.figure.bbox) #NECESSARY??
            
            #draggable.save_offset()
            self.ref_point = np.subtract(xy, draggable.offset) #current_offset = draggable.offset
 
            #make the ghost artists visible
            #for link in draggable.linked:
                ##set the ghost vis same as the original vis for linked
                #for l, g in zip(link.get_children(), 
                                #link.ghost.get_children()):
                    #g.set_visible(l.get_visible())
                    #print( 'visible!', g, g.get_visible() )
        
        #print('picked', repr(self.selection), event.artist)
    
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def on_motion(self, event):
        #TODO: pull in draggableBase on_motion_blit
        #if we want the artist to respond to events only inside the axes - may not be desirable
        #if event.inaxes != self.ax:
            #return
        
        if event.button != 1:
            return
        
        if self.selection:
            draggable = self.draggables[self.selection]
            
            xydisp = event.x, event.y
            xydata = x, y = self.ax.transData.inverted().transform(xydisp) #xydata = 
            delta = xydata - self.ref_point     #from original data
            
            if not draggable.validate(xydata):
                return
            
            #print('delta', delta)
            self.delta = delta
            draggable.shift(self.delta)
            self.update(draggable, xydata) #update_offset?
           
                
            #for link in draggable.linked:
                #link.ghost.shift(delta)
                
                #print( [(ch.get_visible(), ch.get_alpha()) for ch in link.ghost.get_children()] )
                
                #link.ghost.draw(self.canvas.renderer)
            #print('...')
            
            #self.canvas.blit(self.figure.bbox)  #self.ax.bbox??
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @mpl_connect('button_release_event')
    def on_release(self, event):
        
        #print( 'release top', repr(self.selection ))
        
        if event.button != 1:
            return
        
        if self.selection:
            #Remove dragging method for selected artist
            self.remove_connection( 'motion_notify_event' )
            
            #xydisp = event.x, event.y  #NOTE: may be far outside allowed range
            #xydata = x, y = self.ax.transData.inverted().transform(xydisp) #xydata = 
            #delta = xydata - self.ref_point     #from original data
            #TODO: fix to end of range value??
            
            xydata = self.delta + self.ref_point
            
            draggable = self.draggables[self.selection]
            draggable.shift(self.delta)
            self.update(draggable, xydata)
            draggable.offset = self.delta
            self.selection.set_animated(False)
            
            #save offset
            #for linked in draggable.linked:
                #linked.shift( self.delta )
                #linked.offset = self.delta
                #linked.ghost.set_visible(False)

            #do_legend()
            ##self.canvas.draw()
            #if self._draw_on:
                #if self._use_blit:
                    #self.canvas.restore_region(self.background)
                    #draggable.draw()
                    #self.canvas.blit(self.figure.bbox)
                    #self.selection.set_animated(False)
                #else:
                    #self.canvas.draw()
            
        self.selection = None
        #self.delta.fill(0)
        #print( 'release', repr(self.selection ))
        #print()    
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #def get_draw_list(self, draggable):
        
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def update(self, draggable, xydata):
        '''draw all artists that where changed by the motion'''
        artists = filter(None, 
                          (func(xydata) for cid, func in draggable.observers.items()))    #six.iteritems(self.observers):
        if self._draw_on:
            if self._use_blit:
                self.canvas.restore_region(self.background)
                for art in flatiter((artists, draggable.ref_art)): #WARNING: will flatten containers etc
                    art.draw(self.canvas.renderer)
                self.canvas.blit(self.figure.bbox)
            else:
                self.canvas.draw()
    