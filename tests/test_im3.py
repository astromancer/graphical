# third-party
import cmasher
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger
from scipy.stats import multivariate_normal

# local
from scrawl.image import Image3D


logger.enable('recipes.oo.meta.tagger')
# logger.enable('scrawl.moves.callbacks')
logger.enable('scrawl')
# logger.enable('scrawl.depth.bar3d')


res = 2**4
sl = slice(-3, 3, complex(res))
Y, X = np.mgrid[sl, sl]
grid = np.array([X, Y])


mu = (0, 0)
covm = np.array([[0.8,  0.3],
                 [0.3,  0.5]])
rv = multivariate_normal(mu, covm)

Zg = rv.pdf(grid.transpose(1, 2, 0)).T


im = Image3D(Zg)

# fig, ax = plt.subplots(subplot_kw={'projection':'3d'})
# bars = Bar3D(ax, X, Y, Zg, cmap=cmr.ocean, zaxis_cbar=True)


plt.show()
