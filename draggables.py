import numpy as np
from copy import copy

import collections as coll

from matplotlib.lines import Line2D
import matplotlib.collections as mcoll
from matplotlib.container import ErrorbarContainer
from matplotlib.legend_handler import HandlerErrorbar
from matplotlib.transforms import blended_transform_factory as btf
from matplotlib.transforms import Affine2D

from superplot.misc import ConnectionMixin, mpl_connect

import itertools as itt
from magic.iter import grouper, partition
from magic.list import flatten

#from PyQt4.QtCore import pyqtRemoveInputHook, pyqtRestoreInputHook
#from decor import unhookPyQt, expose

#from IPython import embed
#from pprint import pprint

#TODO: Convert to general class for draggable artists.  see: matplotlib.offsetbox.DraggableBase
from matplotlib.offsetbox import DraggableBase

class Foo( DraggableBase ):
    artist_picker = 5
    
    def __init__(self, ref_artist, use_blit=False, free_axis='both'):
        DraggableBase.__init__(self, ref_artist, use_blit=False)
        self._original_transform = ref_artist.get_transform
        
    def save_offset(self):
        art = self.ref_artist
        self.ghost = ghost = copy(art)
        art.set_alpha( 0.2 )
        ax = art.get_axes()
        ax.add_artist( ghost )
    
    def update_offset(self, dx, dy):
        print( dx, dy )
        trans_offset = ScaledTranslation(dx, dy, IdentityTransform())
        trans = ax.transData + trans_offset
        
        self.ref_artist.set_transform( trans )
        

    def finalize_offset(self):
        #print( vars(self) )
        self.ghost.remove()
        self.ref_artist.set_alpha( 1.0 )
        self.canvas.draw()



#****************************************************************************************************
def is_line(o):
    return isinstance(o, Line2D)

def flatten_nested_dict(d):
    items = []
    for v in d.values():
        if isinstance(v, coll.MutableMapping):
            items.extend( flatten_nested_dict(v) )
        else:
            items.append(v)
    return items



#****************************************************************************************************
class DraggableLine( object ): #TODO: ELLIMINATE THIS CLASS
    '''
    Class which takes Line2D objects and makes them draggable on the figure canvas.  Also allows  
    toggling line visibility by selecting on the legend.
    '''

    _default_legend = dict( loc         =       'upper right',
                            fancybox    =       True,
                            framealpha  =       0.25 )
    
    #TODO:  INHERIT THIS METHOD
    @staticmethod
    def _set_defaults(props, defaults):
        for k,v in defaults.items():
            props.setdefault(k,v)
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __init__(self, plots, offsets=None, annotation=[], **legendkw):
        '''
        Parameters
        ----------
        plots : list of Line2D objects
        offsers : sequence of floats
            Initial offsets to use for Line2D objects
        legendkw : dict, optional
            Keywords passed directly to ax.legend
        '''
        self.selection          = None
        self.select_point       = None
        self.plots = plots      = flatten(plots)
        self._original_y        = oy = {}
        
        offsets                 = offsets if not offsets is None else np.zeros(len(plots))
        self.offsets            = {}
        self.tmp_offset         = 0                             #in case of pick without motion
        
        self.ax = ax            = self.plots[0].get_axes()
        self.fig                = ax.figure
        self.annotation         = ann = {}
        self.connections        = []
        
        
        text_trans = btf( ax.transAxes, ax.transData )
        txtx = 1.005
        #Save copy of original data
        for i, art in enumerate(flatten(plots)):
            oy[art] = y = art.get_ydata()
            
            #make the lines pickable
            if not art.get_picker():
                art.set_picker(5)
            
            #Set the initial offsets (if given)
            self.offsets[art] = offset = offsets[i]
            art.set_ydata( y + offset )
            
            #Initialize texts
            ann[art] = ax.text( txtx, y[-1]+ offset, '[%+g]'%offset, transform=text_trans )
        
        ax.relim()
        ax.autoscale_view()
        
        self.enable_legend_picking( ax, plots, legendkw )
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def enable_legend_picking(self, ax, plots, legendkw):
        #enable legend picking
        self._set_defaults( legendkw, self._default_legend )
        autolegend = legendkw.pop('autolegend', True)
        handles, labels = ax.get_legend_handles_labels()
        
        if len(handles) == 0:
            if autolegend:
                #Auto-generate labels
                labels = []
                for i, pl in enumerate(plots):
                    lbl = self.labelmap[type(pl)] + str(i)
                    
                    pl.set_label(lbl)
                    labels.append( lbl )
            else:
                return
            
        leg = ax.legend( plots, labels, **legendkw )
        
        self.leg_map = {}
        for art, origart in zip(leg.legendHandles, plots): #get_lines()
            #for art in flatten(legart):
            art.set_pickradius( 10 )                     # 10 pts tolerance
            art.set_picker( self.legend_picker )         
            self.leg_map[art] = origart
        
            
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def on_click(self, event):
        '''reset plot on middle mouse'''
        if event.button==2:
            self.reset()
        else:
            return
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def reset(self):
        '''reset the plot positions to zero offset'''
        for pl in self.plots:
            self.shift( 0, 'original', self.draggables[pl] )
        
        self.fig.canvas.draw()
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def on_pick(self, event):
        #print( 'pick' )
        
        if event.artist in self.leg_map:
            self.toggle_vis( event )
            #self.selection = self.leg_map[event.artist]
        
        #print(  event.artist )
        #print( event.artist in self.leg_map )
        elif event.artist in self._original_y:
            self.selection = sel = event.artist
            xy = event.mouseevent.xdata, event.mouseevent.ydata
            
            self.select_point = np.subtract( xy, current_offset )
            self.ghost = ghost = copy( sel )
            
            sel.axes.add_artist( ghost )
            ghost.set_alpha( 0.2 )
            ghost.set_visible( 0 )
            
            txt = self.annotation[sel]
            self.ghost_txt = ghost_txt = copy( txt )
            ghost_txt.set_alpha( 0.2 )
            sel.axes.add_artist( ghost_txt )
            
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def toggle_vis(self, event):
        # on the pick event, find the orig line corresponding to the
        # legend proxy line, and toggle the visibility
        legline = event.artist
        origline = self.leg_map[legline]
        vis = not origline.get_visible()
        origline.set_visible(vis)
        # Change the alpha on the line in the legend so we can see what lines
        # have been toggled
        alpha = 1. if vis else .2
        legline.set_alpha( alpha )
        
        self.fig.canvas.draw()
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def on_motion(self, event):

        if event.inaxes != self.ax:
            return
        
        if self.selection:
            ghost = self.ghost
            self.tmp_offset = tmp_offset = event.ydata - self.select_point[1]   #from original data
            
            y = self._original_y[self.yidx]
            ghost.set_ydata( y + tmp_offset )
            
            if not ghost.get_visible( ):
                ghost.set_visible( 1 )
            
            ghost_txt = self.ghost_txt
            ghost_txt.set_y( y[-1]+tmp_offset )
            ghost_txt.set_text( '[%+3.2f]'%(tmp_offset) ) 
            
            self.fig.canvas.draw()
            
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def shift(self, artist, offset):
        
        oy = self._original_y[artist]
        artist.set_ydata( oy + offset )
        
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def on_release(self, event):
        #print( 'release' )
        selection = self.selection
        if selection:
            #print( ghost )
            
            y_orig = self._original_y[selection]
            ghost_txt = self.ghost_txt
            offset = self.offsets[selection] = self.tmp_offset
            
            #selection.set_ydata( y_orig + offset )
            
            #selection.set_label( 'uno, [+%3.2f]'%offset )
            
            txt = self.annotation[selection]
            txt.set_y( ghost_txt.get_position()[1] )
            txt.set_text( '[%+3.2f]'%offset ) 
            
            self.ghost.remove()
            ghost_txt.set_visible(0)      #HACK!
            #print( ax.texts )
            
            #do_legend()
            self.fig.canvas.draw()
            
        self.selection = None
        self.yidx = None
        
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def connect(self):
        self.fig.canvas.mpl_connect('pick_event', self.on_pick)
        #self.fig.canvas.mpl_connect('button_press_event', self.on_pick)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)


#****************************************************************************************************
#Alias
class DraggableLines( DraggableLine ):
    pass


        
#****************************************************************************************************
class ReorderedErrorbarHandler(HandlerErrorbar):
    '''
    Sub-class the standard errorbar handler to make the legends pickable.
    We pick on the markers (Line2D) of the ErrorbarContainer.  We know that the
    markers will always be created, even if not displayed.  The bar stems and
    caps may not be created.  We therefore re-order the ErrorbarContainer 
    where it is created by the legend handler by intercepting the create_artist 
    method
    '''
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __init__(self, xerr_size=0.5, yerr_size=None,
                 marker_pad=0.3, numpoints=None, **kw):
        
        HandlerErrorbar.__init__(self, xerr_size, yerr_size,
                                 marker_pad, numpoints, **kw)
        self.eventson = True
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #@expose.args()
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        """
        x, y, w, h in display coordinate w/ default dpi (72)
        fontsize in points
        """
        xdescent, ydescent, width, height = self.adjust_drawing_area(
                                                     legend, orig_handle,
                                                     handlebox.xdescent, 
                                                     handlebox.ydescent,
                                                     handlebox.width, 
                                                     handlebox.height,
                                                     fontsize)
        
        artists = self.create_artists(legend, orig_handle,
                                      xdescent, ydescent, width, height,
                                      fontsize, handlebox.get_transform())
        
        container = NamedErrorbarContainer(artists[:-1], #NOTE: we are completely ignoring legline here
                                           orig_handle.has_xerr, 
                                           orig_handle.has_yerr)
        
        # create_artists will return a list of artists.
        for art in container.get_children():
            handlebox.add_artist(art)
        
        #these become the legendHandles
        return container
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize,
                       trans):
        #from matplotlib.collections import LineCollection
        
        #  call the parent class function
        artists = HandlerErrorbar.create_artists(self, legend, orig_handle,
                                                 xdescent, ydescent, 
                                                 width, height, fontsize,
                                                 trans)
        
        #Identify the artists. just so we know what each is
        #NOTE: The order here is different to that in ErrorbarContainer, so we re-order
        barlinecols, rest =  partition( is_line, artists )
        barlinecols = tuple(barlinecols)
        *caplines, legline, legline_marker = rest
        
        xerr_size, yerr_size = self.get_err_size(legend, xdescent, ydescent,
                                                 width, height, fontsize)       #NOTE: second time calling this (already done in HandlerErrorbar.create_artists
        
        legline_marker.set_pickradius( xerr_size*1.25 )
        legline_marker.set_picker( ErrorbarPicker() )
        
        return [legline_marker, caplines, barlinecols, legline]


#****************************************************************************************************
class ErrorbarPicker():
    '''Hadles picking artists in the legend'''
    parts = ('markers', 'stems', 'caps')

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @staticmethod
    def __call__(artist, event):
        '''
        Hack the picker to emulate true artist picking in the legend. Specific 
        implementation for errorbar plots. Pretent that the points / bars / caps
        are selected, based on the radius from the central point. Pass the 
        "part" description str as a property to the event from where it can be used 
        by the on_pick method.
        '''
        props = {}

        # Convert points to pixels
        transformed_path = artist._get_transformed_path()
        path, affine = transformed_path.get_transformed_path_and_affine()
        path = affine.transform_path(path)
        xy = path.vertices
        xt = xy[:, 0]
        yt = xy[:, 1]
        
        # Convert pick radius from points to pixels
        pixels = artist.figure.dpi / 72. * artist.pickradius

        # Split the pick area in 3
        Rpix = np.c_[1/3:1:3j] * pixels   #radii of concentric circles 
        
        xd, yd = xt - event.x, yt - event.y
        prox = xd ** 2 + yd ** 2           #distance of click from points in pixels
        c = prox - Rpix**2                 #2D array. columns rep. points in line; rows rep. distance 
        picked = np.any( c<0 )
        
        if picked:
            #part index (which part of the errorbar container) and point index (which point along the line)
            #NOTE: the indeces here are wrt the re-ordered container. We therefore
            partix, pointix = np.unravel_index( abs(c).argmin(), c.shape )
            props['part'] = ErrorbarPicker.parts[partix]
            props['partxy'] = 'yx'[np.argmin(np.abs((xd, yd)))]
           
        return picked, props

#****************************************************************************************************
class DynamicLegend(ConnectionMixin):
    '''
    Enables toggling marker / bar / cap visibility by selecting on the legend.
    '''
    _default_legend = dict( fancybox=True,
                            framealpha=0.5,
                            handler_map = {ErrorbarContainer: ReorderedErrorbarHandler(numpoints=1)} )

    labelmap = { ErrorbarContainer : '_errorbar', Line2D : '_line' }
    
   #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __init__(self, ax, plots, legendkw):
        '''enable legend picking'''
        
        #initialize auto-connect
        ConnectionMixin.__init__(self, ax.figure)
        
        handles, labels = ax.get_legend_handles_labels()
        
        #Auto-generate labels
        if len(handles) == 0:
            labels = []
            for i, pl in enumerate(plots):
                lbl = self.labelmap[type(pl)] + str(i)
                
                pl.set_label(lbl)
                labels.append( lbl )
        
        #update default legend props with user specified props
        lkw = self._default_legend
        lkw.update(legendkw)
        
        #create the legend
        self.legend = ax.legend( plots, labels, **lkw )
        
        #create mapping between the picked legend artists (markers), and the 
        #original (axes) artists
        self.to_orig = {}
        self.to_leg = {}
        self.to_handle = {}
    
        #enable legend picking by setting the picker method
        for handel, origart in zip(self.legend.legendHandles, plots): #get_lines()
            self.to_orig[handel.markers] = NamedErrorbarContainer(origart)
            self.to_leg[handel.markers] = handel
            self.to_handle[origart[0]] = handel
         
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @mpl_connect('pick_event')
    def on_pick(self, event): 
        '''Pick event handler.'''
        if event.artist in self.to_orig:
            self.toggle_vis( event )

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #@unhookPyQt
    def toggle_vis(self, event):
        ''' 
        on the pick event, find the orig line corresponding to the
        legend proxy line, and toggle the visibility.
        '''
        def get_part(mapping, event):
            part = getattr(mapping[event.artist], event.part)
            
            if event.part in ('stems','caps'):
                artists = getattr(part, event.partxy)
            else:
                artists = part
            
            yield from flatten([artists])
            
        for art in get_part(self.to_orig, event):
            vis = not art.get_visible()
            art.set_visible(vis)
        
        for art in get_part(self.to_leg, event):
            art.set_alpha( 1.0 if vis else 0.2 )
        
        #FIXME UnboundLocalError: local variable 'vis' referenced before assignment
        
        self.canvas.draw()



#######################################################################################################################
def get_xy(container, has_yerr):
    '''
    Return the constituents of an ErrorbarContainer object as dictionary keyed 
    on 'x' and 'y'.
    '''
    markers, caps, stems = container
    order = reversed if has_yerr else lambda x: x
    def getter(item, g):
        return coll.defaultdict(list,
                                itt.zip_longest(order('xy'), 
                                                grouper(order(item), g),
                                                fillvalue=())
                                )
    _stems = getter(stems, 1)
    _caps = getter(caps, 2)
    return markers, _stems, _caps

#_NamedErrorbarContainer = coll.namedtuple('_NamedErrorbarContainer', 
                                           #'markers stems caps')
class NamedErrorbarContainer(ErrorbarContainer,
                             coll.namedtuple('_NamedErrorbarContainer', 
                                             'markers stems caps')):
    '''
    Wrapped ErrorbarContainer that identifies constituents explicitly and
    allows dictionary like item access.
    '''
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __new__(cls, container, has_xerr=False, has_yerr=False, **kws):
        if isinstance(container, list):
            if len(container) != 3:
                container = cls._partition(container)
            
        markers, caps, stems = get_xy(container, has_yerr)

        stems = coll.namedtuple('Stems', 'x y')(**stems)
        caps = coll.namedtuple('Caps', 'x y')(**caps)

        return super().__new__(cls, (markers, caps, stems))

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @staticmethod
    def _partition(artists):
        ''' '''
        stems, rest = partition(is_line, artists)
        markers, *caps = rest
        return markers, caps, tuple(stems)

    def partition(self):
        return self._partition(self.get_children())
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __init__(self, container, has_xerr=False, has_yerr=False, **kws):
        ''' '''
        if isinstance(container, ErrorbarContainer):
           ErrorbarContainer.__init__(self, container, 
                                       container.has_xerr, container.has_yerr,
                                        label=container.get_label())
        else:
            
            ErrorbarContainer.__init__(self, container, has_xerr, has_yerr, **kws)

        

#****************************************************************************************************
class DraggableErrorbarContainer(NamedErrorbarContainer):
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    annotation_format = '[%+3.2f]'
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __init__(self, container, has_xerr=False, has_yerr=False, **kws):
        ''' '''
        self.offset = kws.pop('offset', 0)
        
        NamedErrorbarContainer.__init__(self, container, has_xerr, has_yerr, **kws)
        
        #by default the container is self-linked
        self.linked = {self}
        
        #Save copy of original transform
        markers = self[0]
        self._original_transform = markers.get_transform()
        
        #make the lines pickable
        if not markers.get_picker():
            markers.set_picker(5)
        
        #Initialize offset texts
        ax = markers.axes
        self.text_trans = btf(ax.transAxes, ax.transData)
        ytxt = markers.get_ydata().mean()
        self.annotation = ax.text( 1.005, ytxt, '' )
                                    #transform=self.text_trans )
        
        #shift to the given offset (default 0)
        self.shift(self.offset)
        
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __repr__(self):
        return "<%s object of %d artists>" % (self.__class__.__name__, len(self))
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def set_transform(self, trans):
        for art in self.get_children():         #flatten list of containers into list of individual artists
            art.set_transform(trans)
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def set_visible(self, vis):
        for art in self.get_children():
            art.set_visible(vis)
            
        self.annotation.set_visible(vis)
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def haunt(self):
        ''' '''
        #embed()
        ghost_artists = self._partition(map(copy, self.get_children()))
        container = ErrorbarContainer(ghost_artists, 
                                      self.has_xerr, self.has_yerr,
                                      label = self._label)
        self.ghost = DraggableErrorbarContainer(container, offset=self.offset)
        
        ax = self[0].axes
        self.ghost.annotation.set_alpha( 0.2 )
        for g in self.ghost.get_children():
            g.set_alpha( 0.2 )
            g.set_visible( 0 )
            ax.add_artist(g)
        
        #.add_container( self.ghost )
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def shift(self, offset):
        '''Shift the data by offset by setting the transform '''
        #add the offset to the y coordinate
        offset_trans = Affine2D().translate(0, offset)
        trans = offset_trans + self._original_transform
        self.set_transform(trans)
        
        txt = self.annotation
        txt.set_transform(offset_trans + self.text_trans)
        txt.set_text( self.annotation_format % offset )
        

#****************************************************************************************************
class DraggableErrorbar(ConnectionMixin):  #TODO: rename!
    #TODO:      Use offsetbox????
    #TODO:      BLITTING
    '''
    Class which takes ErrorbarCollection objects and makes them draggable on the figure canvas.
    '''
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __init__(self, plots, offsets=None, linked=[], **legendkw):
        '''
        Parameters
        ----------
        plots : list of plot objects
            List of (optionally nested) Line2D objects or ErrorbarContainers
        offsets : sequence of floats
            Initial offsets to use for Line2D objects
        autolabel : boolean
            Whether labels should be generated for the plot objects if plot objects are unlabeled
        
        Keywords
        --------
        legendkw : dict, optional
            Keywords passed directly to axes.legend
        '''
        self.selection          = None
        self.select_point       = None
        
        offsets                 = offsets       or      np.zeros(len(plots))
        #self.offsets            = {}                    #map pickable artist to offset value for that artist
        self.tmp_offset         = 0                     #in case of pick without motion
        
        #self.plots              = plots #= flatten(plots)
        self.ax = ax            = plots[0][0].get_axes()
        #self.fig                = ax.figure
        
        #initialize mapping
        self.draggables = {}
        
        #initialize auto-connect
        ConnectionMixin.__init__(self, ax.figure)
        
        #build the draggable objects
        for plot, offset in zip(plots, offsets):
            #if isinstance(plot, ErrorbarContainer):    #NOTE: will need to check this when generalizing
            markers, _, _  = draggable = DraggableErrorbarContainer(plot, 
                                                                    offset=offset )
            self.draggables[markers] = draggable   #maps Line2D to DraggableErrorbarContainer. The picker returns the markers
            draggable.haunt()
        
        #establish links
        if isinstance(linked[0], ErrorbarContainer):
            linked = [linked]   #np.atleast_2d does not work because it unpacks the ErrorbarContainer
                      
        for links in linked:
            linked = {self.draggables[m] for m, _, _ in links}   #set of linked DraggableErrorbarContainers
            for drag in linked:
                drag.linked = linked

        #enable legend picking
        self.legend = DynamicLegend(ax, plots, legendkw)
        self.legend.connect()
        
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def reset(self):
        '''reset the plot positions to zero offset'''
        for draggable in self.draggables.values():
            draggable.shift(0)  #NOTE: should this be the original offsets??
        
        self.canvas.draw()
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @mpl_connect('button_press_event')
    def on_click(self, event):
        '''reset plot on middle mouse'''
        if event.button == 2:
            self.reset()
        else:
            return
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @mpl_connect('pick_event')
    def on_pick(self, event): 
        '''Pick event handler.  On '''
        
        if event.artist in self.draggables:
            ax = self.ax
            xy = event.mouseevent.xdata, event.mouseevent.ydata
            self.selection = event.artist
            
            #connect motion_notify_event for dragging the selected artist
            self.add_connection( 'motion_notify_event', self.on_motion )
            
            draggable = self.draggables[self.selection]
            self.select_point = np.subtract( xy, draggable.offset ) #current_offset = draggable.offset
 
            #make the ghost artists visible
            for link in draggable.linked:
                #set the ghost vis same as the original vis for linked
                for l, g in zip(link.get_children(), 
                                link.ghost.get_children()):
                    g.set_visible(l.get_visible())
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def on_motion(self, event):
        
        if event.inaxes != self.ax:
            return
        
        if self.selection:
            self.tmp_offset = tmp_offset = event.ydata - self.select_point[1]   #from original data
            
            draggable = self.draggables[self.selection]
            for linked in draggable.linked:
                linked.ghost.shift( tmp_offset )
            
            
            self.canvas.draw()
            
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @mpl_connect('button_release_event')
    def on_release(self, event):
        #print( 'release' )
        if event.button != 1:
            return
        
        if self.selection:
            #Remove dragging method for selected artist
            self.remove_connection( 'motion_notify_event' )
            
            draggable = self.draggables[self.selection]
            for linked in draggable.linked:
                linked.shift( self.tmp_offset )
                linked.offset = self.tmp_offset
                linked.ghost.set_visible(False)

            #do_legend()
            self.canvas.draw()
            
        self.selection = None
    

#######################################################################################################################
#Alias
#****************************************************************************************************
class DraggableErrorbars(DraggableErrorbar):
    pass

#######################################################################################################################

#The code below serves as a test
if __name__ == '__main__':
    import numpy as np
    import pylab as plt

    fig, ax = plt.subplots( figsize=(18,8) )
    N = 100
    x = np.arange(N)
    y0 = np.random.randn(N)
    y1, y2, y3  = y0 + np.c_[[5, 10, -10]]
    y0err, y1err = np.random.randn(2, N) / np.c_[[5, 2]]
    y2err, y3err = None, None
    x0err, x1err, x2err, x3err  = np.random.randn(N), None, None, np.random.randn(N)*8
    
    
    plots = [ ax.errorbar( x, y0, y0err, x0err, fmt='go', label='foo' ),
              ax.errorbar( x, y1, y1err, x1err, fmt='mp', label='bar' ),
              ax.errorbar( x, y2, y2err, x2err, fmt='cd', label='baz' ),
              ax.errorbar( x, y3, y3err, x3err, fmt='r*', ms=25, mew=2, 
                          mec='c', label='linked to baz' ) ]
    
    d = DraggableErrorbar( plots, linked=plots[-2:] )
    d.connect()
    plt.show()
