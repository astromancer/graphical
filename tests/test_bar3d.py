
# third-party
import numpy as np
import cmasher as cmr
import matplotlib.pyplot as plt
from loguru import logger
from scipy.stats import multivariate_normal

# local
from scrawl.depth.bar3d import Bar3DCollection


logger.enable('recipes')
logger.enable('scrawl')


res = 2**3
sl = slice(-3, 3, complex(res))
x, y = xy = np.array(np.mgrid[sl, sl][::-1])
# bars width and breadth
dxy = dx = dy = 0.8


mu = (0, 0)
covm = np.array([[0.8,  0.3],
                 [0.3,  0.5]])
rv = multivariate_normal(mu, covm)
z = rv.pdf(xy.transpose(1, 2, 0)).T

# sm = ScalarMappable(None, 'cmr.emergency_r')
# color = sm.to_rgba(z).reshape(-1, 4)

# xyz = np.array([x, y, z])
# for i, (xx, yy, dz, c) in enumerate(zip(*xyz.reshape(3, -1), color)):
#     bar = ax.bar3d(xx, yy, 0, dx, dy, dz, color=c)

#im = Image3D(Zg)
#im = ImageDisplay(Zg)


fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
#i = [15], [14]
bars = Bar3DCollection(x, y, z, cmap=cmr.ocean)
ax.add_collection(bars)
ax.set(xlabel='x', ylabel='y')

ax.auto_scale_xyz((x.min(), (x + dx).max()),
                  (y.min(), (y + dy).max()),
                  (z.min(), z.max()),
                  False)

plt.show()
