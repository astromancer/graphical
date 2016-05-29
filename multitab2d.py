import sys

import matplotlib
matplotlib.use('QT4Agg')
from matplotlib.backends.qt_compat import QtGui, QtCore
from matplotlib.backends.backend_qt4 import (FigureCanvasQT as FigureCanvas,
                                            NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure
from matplotlib import pyplot as plt

#from decor import expose, profile

import numpy as np

#****************************************************************************************************
class MplMultiTab2D(QtGui.QMainWindow):
    '''Combination tabs display matplotlib canvas'''
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __init__(self, figures=[], labels=[], shape=None, title=None):
        #TODO: ND
        ''' '''
        super().__init__()
        #self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle(title or self.__class__.__name__)
        
        self.figures = np.array(figures).reshape(shape)
        self.labels = labels    #np.array(labels).reshape(self.figures.shape)
        
        #create main widget
        self.main_widget = QtGui.QWidget(self)
        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)
        
        #Create the navigation toolbar stack
        self.toolstack = QtGui.QStackedWidget(self.main_widget)
        
        #stack switches display for central pannel
        self.stack = QtGui.QStackedWidget(self.main_widget)
        
        #create the tab bars
        self.tabbars = []
        for loc, lbls in zip((QtGui.QTabBar.RoundedWest, QtGui.QTabBar.RoundedNorth), 
                             self.labels):
            tabs = self._create_tabs(loc, lbls)
            tabs.currentChanged.connect(self.tab_change)
            self.tabbars.append(tabs)
        self.tabsWest, self.tabsNorth = self.tabbars
        
        #define layout
        grid = QtGui.QGridLayout(self.main_widget)
        #grid.setSpacing(10)
        
        #add widgets to layout
        grid.addWidget(self.toolstack, 0, 0, 1, 2)
        grid.addWidget(self.tabsNorth, 1, 1, 1, 1, QtCore.Qt.AlignLeft)
        grid.addWidget(self.tabsWest, 2, 0, 1, 1, QtCore.Qt.AlignTop)
        grid.addWidget(self.stack, 2, 1)
        
        #add canvasses to stack
        for row in self.figures:
            for fig in row:
                #print(self._rows*i + j)
                #print(fig)
                canvas = fig.canvas
                navtool = NavigationToolbar(canvas, self.toolstack)
                self.stack.addWidget(canvas)
                self.toolstack.addWidget(navtool)
                
                plt.close()
                
        #self.show()
        
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _create_tabs(self, loc, labels):
        '''create the tab bars at location with labels'''
        tabs = QtGui.QTabBar(self.main_widget)
        tabs.setShape(loc)
        for i, d in enumerate(labels):
            tabs.addTab(d)
        
        return tabs
        
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def tab_change(self):
        '''Called upon change of tab'''
        i, j = self.tabsWest.currentIndex(), self.tabsNorth.currentIndex()
        _rows, _ = self.figures.shape
        #print( 'shape:', self.figures.shape )
        #print(i,j, _rows*j + i)
        #print()
        #print(self.stack.currentWidget())
        self.stack.setCurrentIndex(_rows*j + i)
        self.toolstack.setCurrentIndex(_rows*j + i)
        
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def save_plots(self, filenames, path=''):
        import os
        #file_choices = "PNG (*.png)|*.png"
        #path = str(QFileDialog.getSaveFileName(self, 
                        #'Save file', '', 
                        #file_choices))
        path = os.path.realpath(path) + os.path.sep
        ntabs = len(self.canvases)
        if isinstance(filenames, str):
            if not '{' in filenames:            #no format str specifier
                filenames = filenames + '{}'    #append numerical
            filenames = [filenames.format(i) for i in range(ntabs)]
          
        for i, canvas in enumerate(self.canvases):
            filename = filenames[i]
            root, name = os.path.split(filename)
            if root:
                savename = filename
            else:
                savename = os.path.join(path, filename)
            
            canvas.figure.savefig(savename)

if __name__ == '__main__':
    #FIXME: doesn't seem to work!!
    #import numpy as np
    from matplotlib import cm
    #Example use
    r, c, N = 4, 3, 100
    colours = iter(cm.spectral(np.linspace(0,1,r*c)))
    figures = []
    row_labels, col_labels = [], []
    for i in range(r):
        for j in range(c):
            fig, ax = plt.subplots()
            ax.plot( np.random.randn(N), color=next(colours) )
            figures.append( fig )
    
    row_labels = ['Row %i'%i for i in range(r)]
    col_labels = ['Col %i'%i for i in range(c)]
    labels = row_labels, col_labels
    
    app = QtGui.QApplication(sys.argv)
    ui = MplMultiTab2D(figures, labels, shape=(r,c))
    ui.show()
    sys.exit(app.exec_())