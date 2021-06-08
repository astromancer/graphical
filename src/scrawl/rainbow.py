from matplotlib.transforms import offset_copy


def rainbow_text(ax, x, y, ls, lc, **kw):
    """
    Take a list of strings ``ls`` and colors ``lc`` and place them next to each
    other, with text ls[i] being shown in color lc[i].

    This example shows how to do both vertical and horizontal text, and will
    pass all keyword arguments to plt.text, so you can set the font size,
    family, etc.
    """
    # TODO
    trans = ax.transAxes
    # horizontal version
    for i, (s, c) in enumerate(zip(ls, lc)):
        text = ax.text(x, y, '{}{}'.format(' ' * bool(i), s),
                       color=c, transform=trans, **kw)
        text.draw(ax.figure.canvas.get_renderer())
        ex = text.get_window_extent()
        trans = offset_copy(text._transform, x=ex.width, units='dots')

    # vertical version
#     for s,c in zip(ls,lc):
#         text = plt.text(x,y," "+s+" ",color=c, transform=trans,
#                 rotation=90, va='bottom', ha='center', **kw)
#         text.draw(fig.canvas.get_renderer())
#         ex = text.get_window_extent()
#         t = transforms.offset_copy(text._transform, y=ex.height, units='dots')

# plt.figure()
# rainbow_text(0.5,0.5, "all unicorns poop rainbows ! ! !".split(),
#         ['red', 'orange', 'brown', 'green', 'blue', 'purple', 'black'],
#         size=40)

# NOTE: This works only for ps backend
# from matplotlib import rc
# rc('text',usetex=True)
# rc('text.latex', preamble=r'\usepackage{color}')
# ax.set_xlabel(r'$\textcolor{green}{}$ / UTC (h)')
# rainbow_text(ax, 0.5, -.08, ['SAST','/ UTC'], ['g', 'k'], fontweight='bold')
