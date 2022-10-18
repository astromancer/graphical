

from .callbacks import CallbackManager, mpl_connect


class DynamicLegend(CallbackManager):
    # TODO: subclass Legend??
    '''
    Enables toggling marker / bar / cap visibility by selecting on the legend.
    '''
    
    _default_legend = dict(fancybox=True,
                           framealpha=0.5,
                           #                            handler_map={ErrorbarContainer:
                           #                                             ReorderedErrorbarHandler(numpoints=1)}
                           )

    #     label_map = {ErrorbarContainer: 'errorbar{}',
    #                  Line2D: 'line{}'}

    
    def __init__(self, ax, plots, legendkw={}):
        '''enable legend picking'''

        # initialize auto-connect
        CallbackManager.__init__(self, ax.figure)

        # Auto-generate labels
        # NOTE: This needs to be done to enable legend picking. if the artists
        # are unlabeled, no legend will be created and we therefor cannot pick them!
        #         i = 0
        #         for plot in plots:
        #             if not plot.get_label():
        # lbl = self.label_map[type(plot)].format(i)
        # plot.set_label(lbl)
        # i += 1

        # update default legend props with user specified props
        lkw = self._default_legend
        lkw.update(legendkw)

        # create the legend

        # print('PING!!'*10 )
        # embed()

        # self.legend = ax.legend( plots, labels, **lkw )
        self.legend = ax.legend(**lkw)

        if self.legend:  # if no labels --> no legend, and we are done!
            # create mapping between the picked legend artists (markers), and the
            # original (axes) artists
            self.to_orig = {}
            # self.to_leg = {}
            # self.to_handle = {}

            # enable legend picking by setting the picker method
            for handel, origart in zip(self.legend.legendHandles, plots):  # get_lines()
                handel.set_pickradius(10)
                self.to_orig[handel] = origart
                # self.to_leg[handel] = handel
                # self.to_handle[origart[0]] = handel

        # self.connect()

    @mpl_connect('pick_event')
    def on_pick(self, event):
        '''Pick event handler.'''
        # print('RARARARA', event.artist)
        if event.artist in self.to_orig:
            self.toggle_vis(event)

    # @unhookPyQt
    def toggle_vis(self, event):
        '''
        on the pick event, find the orig line corresponding to the
        legend proxy line, and toggle the visibility.
        '''
        # Toggle vis of axes artists
        art = self.to_orig[event.artist]
        vis = not art.get_visible()
        art.set_visible(vis)

        # set alpha of legend artist
        art = event.artist
        art.set_alpha(1.0 if vis else 0.2)

        # TODO: BLIT
        self.canvas.draw()


# class DynamicLegend(CallbackManager):  # TODO: move to seperate script....
#     # TODO: subclass Legend??
#     '''
#     Enables toggling marker / bar / cap visibility by selecting on the legend.
#     '''
#     # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#     _default_legend = dict(fancybox=True,
#                            framealpha=0.5,
#                            handler_map={ErrorbarContainer:
#                                             ReorderedErrorbarHandler(numpoints=1)})
#     label_map = {ErrorbarContainer: 'errorbar{}',
#                  Line2D: 'line{}'}
#
#     # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#     def __init__(self, ax, plots, legendkw={}):
#         '''enable legend picking'''
#
#         # initialize auto-connect
#         CallbackManager.__init__(self, ax.figure)
#
#         # Auto-generate labels
#         # NOTE: This needs to be done to enable legend picking. if the artists
#         # are unlabeled, no legend will be created and we therefor cannot pick them!
#         i = 0
#         for plot in plots:
#             if not plot.get_label():
#                 lbl = self.label_map[type(plot)].format(i)
#                 plot.set_label(lbl)
#                 i += 1
#
#         # update default legend props with user specified props
#         lkw = self._default_legend
#         lkw.update(legendkw)
#
#         # create the legend
#
#         # print('PING!!'*10 )
#         # embed()
#
#         # self.legend = ax.legend( plots, labels, **lkw )
#         self.legend = ax.legend(**lkw)
#
#         if self.legend:  # if no labels --> no legend, and we are done!
#             # create mapping between the picked legend artists (markers), and the
#             # original (axes) artists
#             self.to_orig = {}
#             self.to_leg = {}
#             self.to_handle = {}
#
#             # enable legend picking by setting the picker method
#             for handel, origart in zip(self.legend.legendHandles, plots):  # get_lines()
#                 self.to_orig[handel.markers] = NamedErrorbarContainer(origart)
#                 self.to_leg[handel.markers] = handel
#                 self.to_handle[origart[0]] = handel
#
#     # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#     @mpl_connect('pick_event')
#     def on_pick(self, event):
#         '''Pick event handler.'''
#         if event.artist in self.to_orig:
#             self.toggle_vis(event)
#
#     # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#     # @unhookPyQt
#     def toggle_vis(self, event):
#         '''
#         on the pick event, find the orig line corresponding to the
#         legend proxy line, and toggle the visibility.
#         '''
#
#         def get_part(mapping, event):
#             part = getattr(mapping[event.artist], event.part)
#
#             if event.part in ('stems', 'caps'):
#                 artists = getattr(part, event.partxy)
#             else:
#                 artists = part
#
#             yield from flatten([artists])
#
#         for art in get_part(self.to_orig, event):
#             vis = not art.get_visible()
#             art.set_visible(vis)
#
#         for art in get_part(self.to_leg, event):
#             art.set_alpha(1.0 if vis else 0.2)
#
#         # FIXME UnboundLocalError: local variable 'vis' referenced before assignment
#         # TODO: BLIT
#         self.canvas.draw()
