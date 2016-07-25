#The code below serves as a test
if __name__ == '__main__':
    import numpy as np
    import pylab as plt
    
    from draggables.machinery import DragMachinery
    from draggables.errorbar import DraggableErrorbar
    from draggables.lines import DragggableLine
    
    
    #Line2D test
    x = 0 #np.arange(N)
    y0 = 0.5    #np.random.randn(N)
    
    fig, ax = plt.subplots( figsize=(18,8) )
    pl = ax.plot(x, y0, 'r>', clip_on=False)
    ax.set_xlim(0, 0.5)
    
    
    d = DragMachinery(pl)
    d.connect()
    plt.show(block=False)
    
    
    #Errorbar test
    #fig, ax = plt.subplots( figsize=(18,8) )
    #N = 100
    #x = np.arange(N)
    #y0 = np.random.randn(N)
    #y1, y2, y3  = y0 + np.c_[[5, 10, -10]]
    #y0err, y1err = np.random.randn(2, N) / np.c_[[5, 2]]
    #y2err, y3err = None, None
    #x0err, x1err, x2err, x3err  = np.random.randn(N), None, None, np.random.randn(N)*8
    
    
    #plots = [ ax.errorbar( x, y0, y0err, x0err, fmt='go', label='foo' ),
            #ax.errorbar( x, y1, y1err, x1err, fmt='mp', label='bar' ),
            #ax.errorbar( x, y2, y2err, x2err, fmt='cd', label='baz' ),
            #ax.errorbar( x, y3, y3err, x3err, fmt='r*', ms=25, mew=2, 
                        #mec='c', label='linked to baz' ) ]
    
    #d = DraggableErrorbar( plots, linked=plots[-2:] )
    #d.connect()
    #plt.show()