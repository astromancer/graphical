import numpy as np
from copy import copy#, deepcopy
from misc import flatten, partition, nthzip, invertdict #, lmap

from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from matplotlib.container import Container, ErrorbarContainer
from matplotlib.legend_handler import HandlerErrorbar
from matplotlib.transforms import blended_transform_factory as btf

from PyQt4.QtCore import pyqtRemoveInputHook, pyqtRestoreInputHook
#from misc import make_ipshell
#from qtconsole import qtshell

#ipshell = make_ipshell()

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
#from decor import print_args
  
legend_artists = []
#****************************************************************************************************
class ReorderedErrorbarHandler(HandlerErrorbar):
    '''
    Sub-class the standard errorbar handler to make the legends pickable.
    '''
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __init__(self, xerr_size=0.5, yerr_size=None,
                 marker_pad=0.3, numpoints=None, **kw):
        
        HandlerErrorbar.__init__(self, xerr_size, yerr_size, marker_pad, numpoints, **kw)
        self.eventson = True
    
    #@print_args()
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def legend_artist(self, legend, orig_handle,
                 fontsize,
                 handlebox):
        """
        x, y, w, h in display coordinate w/ default dpi (72)
        fontsize in points
        """
        global legend_artists                   #ugly hack!!
        
        xdescent, ydescent, width, height = self.adjust_drawing_area(
                                                     legend, orig_handle,
                                                     handlebox.xdescent, handlebox.ydescent,
                                                     handlebox.width, handlebox.height,
                                                     fontsize)
        
        artists = self.create_artists(legend, orig_handle,
                                      xdescent, ydescent, width, height,
                                      fontsize, handlebox.get_transform())
        
        legend_artists.append( artists )
        # create_artists will return a list of artists.
        for a in flatten(artists):
            handlebox.add_artist(a)
        
        #self.handlebox = handlebox
        #artists = artists[-1:] + artists[:-1]
        return artists[0]
        
        #return HandlerErrorbar.__call__(self, *args, **kwargs):
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def create_artists(self, *args, **kwargs):
        #from matplotlib.collections import LineCollection
        
        #  call the parent class function
        artists = HandlerErrorbar.create_artists(self, *args, **kwargs)
        
        #Identify the artists. just so we know what each is
        is_line = lambda o: isinstance(o, Line2D)
        barlinecols, rest =  partition( is_line, artists )
        barlinecols = tuple(barlinecols)
        *caplines, legline, legline_marker = rest
        
        # re-order the artist list, only the first artist is added to the
        # legend artist list, this is the one that corresponds to the markers 
        # which we want to pick on
        return [legline_marker, barlinecols, caplines, legline]


#****************************************************************************************************
class DraggableErrorbar( object ):  
    #TODO:      Use offsetbox????
    #TODO:      BLITTING
    '''
    Class which takes ErrorbarCollection objects and makes them draggable on the figure canvas.
    Also enables toggling point / bar / barcap visibility by selecting on the legend.
    '''
    labelmap = { ErrorbarContainer : '_errorbar', Line2D : '_line' }
    
    _default_legend = dict( fancybox=True,
                            framealpha=0.25,
                            handler_map = {ErrorbarContainer: ReorderedErrorbarHandler(numpoints=2)} )
    
    #TODO:  INHERIT THIS METHOD
    @staticmethod
    def _set_defaults(props, defaults):
        for k,v in defaults.items():
            props.setdefault(k,v)
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __init__(self, plots, offsets=None, **legendkw):
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
        self.plots              = plots #= flatten(plots)
        offsets                 = offsets       or      np.zeros(len(plots))
        self.offsets            = {}                    #map pickable artist to offset value for that artist
        self.tmp_offset         = 0                             #in case of pick without motion
        self.ax = ax            = plots[0][0].get_axes()
        self.fig                = ax.figure
        self.connections        = []
        
        #Save copy of original data
        self._original_y = oy = {}
        for art in flatten(plots):
            if isinstance(art, Line2D):
                oy[art] = art.get_ydata()
            elif isinstance(art, LineCollection):
                oy[art] = art.get_segments()
        
        #create mapping between ErrorbarContainer and Line2D representing data points
        self.linemap = linemap = {}
        self.annotation = ann = {}
        text_trans = btf( ax.transAxes, ax.transData )
        
        for i, pl in enumerate(plots):
            if isinstance(pl, ErrorbarContainer):
                points, caps, bars = pl
                linemap[pl] = points            #maps ErrorbarContainer to Line2D of points. This is the artist that the picker returns  NOTE: linemap here could also just be a function that returns the zeroth element
            
            #create the offset map
            self.offsets[points] = offsets[i]
            
            #make the lines pickable
            if not points.get_picker():
                points.set_picker(5)
            
            #from IPython import embed
            #embed()
            #for cap in caps:
                #cap.set_picker(None)
            
            #Initialize texts
            ytxt = np.mean( oy[points] )
            text = ax.text( 1.005, ytxt, '[%+3.2f]'%offsets[i], transform=text_trans )
            ann[points] = text
                    
        self.invlinemap = invertdict( linemap )
        
        self.enable_legend_picking(ax, plots, legendkw)
        
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def enable_legend_picking(self, ax, plots, legendkw):
        #enable legend picking
        self._set_defaults( legendkw, self._default_legend )
        handles, labels = ax.get_legend_handles_labels()
        
        if len(handles) == 0:
            #Auto-generate labels
            labels = []
            for i, pl in enumerate(plots):
                lbl = self.labelmap[type(pl)] + str(i)
                
                pl.set_label(lbl)
                labels.append( lbl )
        
        #legendkw.setdefault('upper right')
        leg = ax.legend( plots, labels, **legendkw )
        
        self.leg_map = {}
        for art, origart in zip(leg.legendHandles, plots): #get_lines()
            #for art in flatten(legart):
            art.set_pickradius( 10 )                     # 10 pts tolerance
            art.set_picker( self.legend_picker )         
            #lines, caps, bars = origart
            #Here we re-order the ErrorbarContainer (again!?) so we can directly index the list returned by leg_map 
            self.leg_map[art] = origart         #lines, bars, caps
        
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #@print_args()
    def legend_picker(self, artist, event):
        '''
        Hack the picker to emulate true artist picking in the legend. Specific implementation 
        for errorbar plots.  Pretent that the points / bars / caps are selected, based on the
        radius from the central point. Pass the "selection" index as property to the event from
        where it can be picked up by the on_pick method.'''
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
        # Split the pick are in 3
        Rpix = np.atleast_2d(1/3 * np.arange(1,4) * pixels).T   #radii of concentric circles 
        
        d = (xt - event.x) ** 2 + (yt - event.y) ** 2           #distance of click from points in pixels
        c = d - Rpix**2                                         #2D array. columns rep. points in line; rows rep. distance 
        picked = np.any( c<0 )
        
        if picked:
            ix = np.where(c<0)
            bix, pix = ix[0][0], ix[1][0]       #bar index (which part of the errorbar container) and point index (which point along the line)
            props['part'] = bix
        
        return picked, props
        
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def reset(self):
        '''reset the plot positions to zero offset'''
        for pl in self.plots:
            self.shift( 0, 'original', self.linemap[pl] )
        
        self.fig.canvas.draw()
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def on_click(self, event):
        '''reset plot on middle mouse'''
        if event.button==2:
            self.reset()
        else:
            return
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def on_pick(self, event): 
        '''Pick event handler.  On '''
        #print( 'pick' )
        #print(  event.artist )
        
        if event.artist in self.leg_map:
            #print( 'LEGEND!!!\n' )
            #print( ' event.part', event.part )
            self.toggle_vis( event )
            #print()
            #self.selection = self.leg_map[event.artist]
        
        #print(  event.artist )
        #print( event.artist in self.leg_map )
        elif event.artist in self.offsets:
            #print( 'AXES' )
            ax = self.ax
            xy = event.mouseevent.xdata, event.mouseevent.ydata
            self.selection = sel = event.artist
            
             #TODO:  CONNECT THE DRAG METHODS HERE!
            
            
            try:
                current_offset = self.offsets[sel]
            except Exception as err:
                print( 'EXCEPTION CAUGHT!!!' )
                print( err )
                from IPython import embed
                pyqtRemoveInputHook()
                embed()
                pyqtRestoreInputHook()

            
            
            
            #current_offset = self.offsets[sel]
            self.select_point = np.subtract( xy, current_offset )
            self.ghost = ghost = [copy(g) for g in flatten(self.invlinemap[sel])]
        
            for art in ghost:
                if isinstance(art, Line2D):
                    ax.add_artist( art )                       #NOTE: ax.add_container( ghost )?
                elif isinstance(art, LineCollection):
                    ax.add_collection( art )
                
                art.set_alpha( 0.2 )
                art.set_visible( 0 )
            
            txt = self.annotation[sel]
            self.ghost_txt = ghost_txt = copy( txt )
            ghost_txt.set_alpha( 0.2 )
            sel.axes.add_artist( ghost_txt )
            #print( '!!!' )
            
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def toggle_vis(self, event):
        # on the pick event, find the orig line corresponding to the
        # legend proxy line, and toggle the visibility
        
        legline = event.artist
        orig_art = self.leg_map[legline][event.part]
        
        #legline.set_alpha(1.0)
        
        #print(  'legline', legline, id(legline), legline.get_alpha() )
        #print( '--> ART!' )
        #print( vars(event) )
        
        for art in flatten([orig_art]):
            #print( art )
            vis = not art.get_visible()
            art.set_visible(vis)
        #print('vis', vis)
        #print( 'legline', legline, legline.get_alpha() )
        #print()
        
        #from qtconsole import qtshell
        #pyqtRemoveInputHook()
        #v = globals()
        #v.update( locals() )
        #qtshell( v )
        #pyqtRestoreInputHook()
        
        
        # Change the alpha on the line in the legend so we can see what lines have been toggle
        #alpha = 1.0 if vis else 0.2
        ix = nthzip( 0, *legend_artists ).index( legline )
        legart = legend_artists[ix][event.part]
        print( 'legart', ix, event.part, legart )
        for art in flatten([legart]):
            
            art.set_alpha( 1.0 if vis else 0.2 )
            
        
        #legline_marker, barlinecols, caplines, legline = legend_artists[ix]
        
        self.fig.canvas.draw()
    
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def shift(self, offset, which='ghost', selection=None):

        oy = self._original_y
        sel = selection or self.selection
        pl = self.invlinemap[sel]
        
        is_line = lambda o: isinstance(o, Line2D)               #TODO Make golbal
        obars, olines =  map( tuple, partition( is_line, flatten(pl) ) )  #obars are LineCollection; olines are Line2D
        gbars, glines =  map( tuple, partition( is_line, self.ghost ) )
        for o,g in zip(olines, glines):
            L = g if which.startswith('g') else o
            L.set_ydata( oy[o] + offset )
        
        for o,g in zip(obars, gbars):
            s = np.array( oy[o] )
            s[...,1] += offset                           #add the offset to the y coordinate
            b = g if which.startswith('g') else o
            b.set_segments( s )
        
        ytxt = np.mean( oy[sel] )
        txt = self.ghost_txt if which.startswith('g') else self.annotation[sel]
        txt.set_y( ytxt + offset )
        txt.set_text( '[%+3.2f]'%(offset) ) 
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def on_motion(self, event):
        
        if event.inaxes != self.ax:
            return
        
        if self.selection:
            self.tmp_offset = tmp_offset = event.ydata - self.select_point[1]   #from original data
            
            try:
                self.shift( tmp_offset )
            except Exception as err:
                pyqtRemoveInputHook()
                from IPython import embed
                embed()
                pyqtRestoreInputHook()

            for g in self.ghost:
                if not g.get_visible( ):
                    g.set_visible( 1 )
            
            self.fig.canvas.draw()
            
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def on_release(self, event):
        #print( 'release' )
        if event.button==1:
            sel = self.selection
            if sel:
                oy = self._original_y[sel]
                offset = self.offsets[sel] = self.tmp_offset
                
                self.shift( offset, 'original' )
                #self.offsets[sel] = 
                
                
                [g.remove() for g in flatten(self.ghost)]
                self.ghost_txt.set_visible(0)      #HACK!  ax.text.remove throws NotImplementedError
                
                #do_legend()
                self.fig.canvas.draw()
            
        self.selection = None
        
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def connect(self):
        mpl_connect = self.fig.canvas.mpl_connect
        cids = self.connections
        cids.append( mpl_connect( 'pick_event', self.on_pick ) )#self.event_handler
        cids.append( mpl_connect( 'button_press_event', self.on_click ) )
        cids.append( mpl_connect( 'button_release_event', self.on_release ) )
        cids.append( mpl_connect( 'motion_notify_event', self.on_motion ) )
        
    
    #def disconnect(self):
        #for cid in self.connections:
            #self.fig.canvas.mpl_disconnect
            
        #self.fig.canvas.mpl_connect( 'pick_event', self.on_pick ) #self.event_handler
        ##self.fig.canvas.mpl_connect( 'button_press_event', self.pick_point )
        #self.fig.canvas.mpl_connect( 'button_release_event', self.on_release )
        #self.fig.canvas.mpl_connect( 'motion_notify_event', self.on_motion )    

#######################################################################################################################
#Alias
#****************************************************************************************************
class DraggableErrorbars( DraggableErrorbar ):
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
              ax.errorbar( x, y2, y2err, x2err, fmt='mp', label='bar' ) ]
    
    d = DraggableErrorbar( plots )
    d.connect()
    plt.show()
