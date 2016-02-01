from recipes.meta import flaggerFactory

######################################################################################################
ConnectionManager, mpl_connect = flaggerFactory( collection='_connections' )

class ConnectionMixin( ConnectionManager ):
    '''Mixin for connecting the decorated methods to the figure canvas'''
    def __init__(self, fig):
        ConnectionManager.__init__(self)
        self.canvas = fig.canvas
    
    #=============================================================================================== 
    def add_connection(self, name, method):
        self.connections[name] = self.canvas.mpl_connect(name, method)
    
    #=============================================================================================== 
    def remove_connection(self, name):
        self.canvas.mpl_disconnect( self.connections[name] )
        self.connections.pop( name )
    
    #=============================================================================================== 
    def connect(self):
        '''connect the flagged methods to the canvas'''
        self.connections = {}                   #connection ids
        for (name,), method in self._connections.items():  #TODO: map??
            self.add_connection( name, method )
                
    #=============================================================================================== 
    def disconnect(self):
        '''
        Disconnect from figure canvas.
        '''
        for name, cid in self.connections.items():
            self.canvas.mpl_disconnect( cid )
        print('Disconnected from figure {}'.format(self.figure.canvas) )