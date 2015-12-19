import numpy as np
from copy import copy

import collections as coll#import defaultdict, OrderedDict

from matplotlib.lines import Line2D
import matplotlib.collections as mcoll
#from matplotlib.collections import LineCollection
from matplotlib.container import Container, ErrorbarContainer
from matplotlib.legend_handler import HandlerErrorbar
from matplotlib.transforms import blended_transform_factory as btf
from matplotlib.transforms import Affine2D

from PyQt4.QtCore import pyqtRemoveInputHook, pyqtRestoreInputHook

#from decor import unhookPyQt, expose
from superplot.misc import ConnectionMixin, mpl_connect

from magic.iter import partition, grouper
from magic.list import flatten
from magic.dict import InvertibleDict

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
class DraggableLine( object ):
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
            self.shift( 0, 'original', self.linemap[pl] )
        
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


        
#######################################################################################################################
class NamedErrorbarContainer(coll.UserDict, ErrorbarContainer):
    '''
    Wrapped ErrorbarContainer that identifies constituents explicitly and
    allows dictionary like item access.
    '''
    def __init__(self, container):
        ErrorbarContainer.__init__(self, list(container), 
                                   container.has_xerr, container.has_yerr,
                                   label=container._label)
        
        markers, caps, stems = container
        
        #WARNING: this will not work for data containing only x-errors!
        _caps = coll.defaultdict( list, zip('yx', grouper(caps[::-1], 2)) )
        _stems = coll.defaultdict( list, zip('yx', stems[::-1]) )
        
        coll.UserDict.__init__(self, zip(('markers', 'caps', 'stems'),
                                          (markers, _caps, _stems)) )
    


#legend_artists = {}
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
        global legend_artists                   #NOTE: ugly hack!!
        
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
        
        # create_artists will return a list of artists.
        for art in flatten(flatten_nested_dict(artists)):
            handlebox.add_artist(art)
        
        # only one artist is added to the legend artist list. We make this the 
        # markers which we want to pick on
        return artists#['markers']
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize,
                       trans):

        plotlines, caplines, barlinecols = orig_handle

        xdata, xdata_marker = self.get_xdata(legend, xdescent, ydescent,
                                             width, height, fontsize)

        ydata = ((height - ydescent) / 2.) * np.ones(xdata.shape, float)
        legline = Line2D(xdata, ydata)


        xdata_marker = np.asarray(xdata_marker)
        ydata_marker = np.asarray(ydata[:len(xdata_marker)])

        xerr_size, yerr_size = self.get_err_size(legend, xdescent, ydescent,
                                                 width, height, fontsize)
        
        legline_marker = Line2D(xdata_marker, ydata_marker)

        # when plotlines are None (only errorbars are drawn), we just
        # make legline invisible.
        if plotlines is None:
            legline.set_visible(False)
            legline_marker.set_visible(False)
        else:
            self.update_prop(legline, plotlines, legend)

            legline.set_drawstyle('default')
            legline.set_marker('None')

            self.update_prop(legline_marker, plotlines, legend)
            legline_marker.set_linestyle('None')

            if legend.markerscale != 1:
                newsz = legline_marker.get_markersize() * legend.markerscale
                legline_marker.set_markersize(newsz)
            
            legline_marker.set_pickradius( xerr_size*1.25 )
            legline_marker.set_picker( ErrorbarPicker() )
            
        handle_barlinecols = coll.defaultdict(list)
        handle_caplines = coll.defaultdict(list)
        
        def make_errorbars(x_or_y):
            if x_or_y == 'x':
                xers, yers, marker = xerr_size, 0, "|"
            else:
                xers, yers, marker = 0, yerr_size, "_"
            
            verts = [ ((x - xers, y - yers), (x + xers, y + yers))
                      for x, y in zip(xdata_marker, ydata_marker)]
            coll = mcoll.LineCollection(verts)
            self.update_prop(coll, barlinecols[0], legend)
            handle_barlinecols[x_or_y].append(coll)

            if caplines:
                capline_left = Line2D(xdata_marker - xers, ydata_marker - yers)
                capline_right = Line2D(xdata_marker + xers, ydata_marker + yers)
                self.update_prop(capline_left, caplines[0], legend)
                self.update_prop(capline_right, caplines[0], legend)
                capline_left.set_marker(marker)
                capline_right.set_marker(marker)

                handle_caplines[x_or_y].extend((capline_left, capline_right))
        
        if orig_handle.has_xerr:
            make_errorbars('x')
    
        if orig_handle.has_yerr:
            make_errorbars('y')
        
        artists = coll.OrderedDict()
        artists['stems'] = handle_barlinecols
        artists['caps']  = handle_caplines
        artists['line'] = legline
        artists['markers'] = legline_marker
        
        for artist in flatten(flatten_nested_dict(artists)):
            artist.set_transform(trans)
        
        return artists


#****************************************************************************************************
class ErrorbarPicker():
    parts = ('markers', 'stems', 'caps')
    #parts = InvertibleDict({ 0 : 'markers',
                             #1 : 'stems',
                             #2 : 'caps'       })
    #iparts = parts.inverse()
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @staticmethod
    def __call__(artist, event):
        '''
        Hack the picker to emulate true artist picking in the legend. Specific 
        implementation for errorbar plots. Pretent that the points / bars / caps
        are selected, based on the radius from the central point. Pass the 
        "part" description str as a property to the event from where it can be picked 
        up by the on_pick method.
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
    Enables toggling point / bar / barcap visibility by selecting on the legend.
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
    
        #enable legend picking by setting the picker method
        for handel, origart in zip(self.legend.legendHandles, plots): #get_lines()
            self.to_orig[handel['markers']] = NamedErrorbarContainer(origart)
            self.to_leg[handel['markers']] = handel
         
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
            part = mapping[event.artist][event.part]
            
            if event.part in ('stems','caps'):
                q = part[event.partxy]
            else:
                q = part
            
            yield from flatten([q])
            
        for art in get_part(self.to_orig, event):
            vis = not art.get_visible()
            art.set_visible(vis)
        
        for art in get_part(self.to_leg, event):
            art.set_alpha( 1.0 if vis else 0.2 )
        
        self.canvas.draw()



#****************************************************************************************************
class DraggableErrorbarContainer(ErrorbarContainer):
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    annotation_format = '[%+3.2f]'
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __init__(self, container, has_xerr=False, has_yerr=False, **kws):
        ''' '''
        self.offset = kws.pop('offset', 0)
        
        if isinstance(container, ErrorbarContainer):
            ErrorbarContainer.__init__(self, container.get_children(), 
                                        container.has_xerr, container.has_yerr,
                                        label=container._label)
        else:
            ErrorbarContainer.__init__(self, container, has_xerr, has_yerr, **kws)
        
        markers = self[0]
        
        #Save copy of original transform
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
        
        self.shift(self.offset)
        
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
        self.ghost = DraggableErrorbarContainer( list(map(copy, self.get_children())), 
                                                 self.has_xerr, self.has_yerr,
                                                 label = self._label,
                                                 offset = self.offset )
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
class DraggableErrorbar(ConnectionMixin):  
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
        self.fig                = ax.figure
        
        self.has_linked         = bool(len(linked))
        self.linked             = list(map(set, linked))
        
        
        #initialize auto-connect
        ConnectionMixin.__init__(self, ax.figure)
        
        #create mapping between ErrorbarContainer and Line2D representing data points
        self.linemap = linemap = InvertibleDict()
        
        #self.plots = []
        for plot, offset in zip(plots, offsets):
            #if isinstance(plot, ErrorbarContainer):
            draggable = DraggableErrorbarContainer( plot, 
                                                    offset=offset )
            linemap[draggable[0]] = draggable   #maps ErrorbarContainer to Line2D of points. This is the artist that the picker returns  NOTE: linemap here could also just be a function that returns the zeroth element
            draggable.haunt()

        self.invlinemap = linemap.inverse()
        
        legend = DynamicLegend(ax, plots, legendkw)
        legend.connect()
        
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def reset(self):
        '''reset the plot positions to zero offset'''
        for draggable in self.linemap.values():
            draggable.shift(0)  #NOTE: should this be the original offsets??
        
        self.fig.canvas.draw()
    
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
        
        if event.artist in self.linemap:
            ax = self.ax
            xy = event.mouseevent.xdata, event.mouseevent.ydata
            self.selection = sel = event.artist
            
            #connect motion_notify_event for dragging the selected artist
            self.add_connection( 'motion_notify_event', self.on_motion )
            
            draggable = self.linemap[sel]
            current_offset = draggable.offset
            self.select_point = np.subtract( xy, current_offset )
            
            draggable.ghost.set_visible(True)
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def on_motion(self, event):
        
        if event.inaxes != self.ax:
            return
        
        sel = self.selection
        if self.selection:
            self.tmp_offset = tmp_offset = event.ydata - self.select_point[1]   #from original data
            
            draggable = self.linemap[sel]
            draggable.ghost.shift( tmp_offset )
            
            self.fig.canvas.draw()
            
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @mpl_connect('button_release_event')
    def on_release(self, event):
        #print( 'release' )
        if event.button == 1:
            sel = self.selection
            if sel:
                
                #Remove dragging method for selected artist
                self.remove_connection( 'motion_notify_event' )
                
                #oy = self._original_y[sel]
                draggable = self.linemap[sel]
                draggable.shift( self.tmp_offset )
                draggable.offset = self.tmp_offset
                draggable.ghost.set_visible(False)
 
                #do_legend()
                self.fig.canvas.draw()
            
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

    fig, ax = plt.subplots( )
    N = 100
    x = np.arange(N)
    y0 = np.random.randn(N)
    y1 = y0 + 5
    y2 = y0 + 10
    y0err, y1err = np.random.randn(2, N) / np.r_['0,2,0',[5,2]]
    y2err = None
    x0err, x1err, x2err  = np.random.randn(N), None, None
    
    
    plots = [ ax.errorbar( x, y0, y0err, x0err, fmt='go', label='foo' ),
              ax.errorbar( x, y1, y1err, x1err, fmt='mp', label='bar' ),
              ax.errorbar( x, y2, y2err, x2err, fmt='cd', label='baz' ) ]
    
    d = DraggableErrorbar( plots )
    d.connect()
    plt.show()
