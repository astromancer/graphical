
import logging

import numpy as np
import matplotlib.pyplot as plt

from grafico.sliders import ColourSliders

root = logging.getLogger()
# root.setLevel(logging.DEBUG)
root.setLevel(logging.WARNING)

fig, ax = plt.subplots()
ax.set_xlim(0, 1e4)
x0, x1 = 0.25, 0.55

#createHistogram(ax, np.random.rand(int(1e4)))
sliders = ColourSliders(ax, x0, x1, 'y')#
sliders.connect()

print('!' * 88)
print(str(sliders.lower), str(sliders.upper))
print('!' * 88)

plt.show()