



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