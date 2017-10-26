#The code below serves as a test
if __name__ == '__main__':

    #NOTE: speed of blitting highly dependent on the size of the figure.  For large figures, can be nearly
    #NOTE  as slaw as redraw.

    import sys
    import argparse

    import numpy as np
    import pylab as plt

    from draggables.machinery import DragMachinery
    # from draggables.errorbar import DraggableErrorbar
    # from draggables.lines import DragggableLine

    parser = argparse.ArgumentParser(
        description='Tests for interactive draggable artists in matplotlib'
    )
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--blit', action='store_true', default=True)
    args = parser.parse_args(sys.argv[1:])
    #
    # from IPython import embed
    # embed()

    if args.profile:
        import inspect
        from decor.profiler import HLineProfiler
        profiler = HLineProfiler()

        for name, method in inspect.getmembers(DragMachinery, predicate=inspect.isfunction):
            profiler.add_function(method)
        profiler.enable_by_count()

    if args.debug:
        from decor.utils import decorateAll
        from decor import expose
        DragMachinery = decorateAll(expose.args())(DragMachinery)

    # from IPython import embed
    # embed()

    # Line2D test
    x = 0 # np.arange(N)
    y0 = 0.5    # np.random.randn(N)

    fig, ax = plt.subplots(figsize=(8,5))
    pl = ax.plot(x, y0, 'r>', clip_on=False)
    ax.set_xlim(0, 0.5)

    d = DragMachinery(pl, use_blit=args.blit)
    d.connect()
    plt.show(block=True)

    # from IPython import embed
    # embed()
    if args.profile:
        profiler.print_stats()
        profiler.rank_functions()

    # Line2D test 2
    # fig, ax = plt.subplots(figsize=(16, 4))
    # x1, y1 = np.arange(0,10), np.zeros(10)
    # x2, y2 = np.arange(0,10), np.ones(10)
    # plots = ax.plot(x1, y1, 'r-', x2, y2, 'b-')

    # machine = DragMachinery(plots, use_blit=args.blit)
    # machine.connect()
    # plt.show()      #if not is_interactive() ???



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