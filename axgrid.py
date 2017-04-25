import inspect

import matplotlib.pyplot as plt


# def make_fig(sigmas, ncol=2):
#     from mpl_toolkits.axes_grid1 import AxesGrid
#     nrow = sum(divmod(len(sigmas), ncol))
#
#     fig = plt.figure(figsize=(8, 14))
#     axes = AxesGrid(fig, 111, nrows_ncols=(nrow, ncol),
#                     label_mode="L",
#                     share_all=True)
#     return fig, axes


def make_fig(sigmas, ncol=2, progression='clockwise'):
    nrow, left = divmod(len(sigmas), ncol)
    fig, axes = plt.subplots(nrow, ncol,
                             figsize=(10.4, 9.5),
                             sharex='col', sharey=True,  # 'row', 'col' #!!!
                             gridspec_kw=dict(hspace=0.01,
                                              wspace=0.0,
                                              left=0.07,
                                              right=0.99,
                                              bottom=0.07,
                                              top=0.95),
                             # tight_layout=True,  # triggers warning
                             )
    return fig, axes


class get_index():
    def __init__(self, nrow, progression='clockwise'):
        progression = progression.lower()
        self.progression = progression
        self.nrow = nrow
        self.method = None
        for name, meth in inspect.getmembers(self, inspect.ismethod):
            # print(name, '_%s' % progression[0])
            if name.startswith('_%s' % progression[0]):
                self.method = meth
                break
        if self.method is None:
            raise ValueError('unknown progression %s' % progression)

    def __getitem__(self, i):
        return self.method(i)

    def _lr(self, i):
        return divmod(i, self.nrow)  # for left-right ordering

    def _ud(self, i):
        return divmod(i, self.nrow)[::-1]  # for up-down ordering

    def _cw(self, i):
        r, c = self._ud(i)
        if i >= self.nrow:
            r = self.nrow - r - 1
        return r, c
