# third-party
import numpy as np

# local
from scrawl.image import ImageDisplay


data = np.random.random((100, 100))
ImageDisplay(data)

# TESTS:
# all zero data

# fig, ax = plt.subplots(1,1, figsize=(2.5, 10), tight_layout=True)
# ax.set_ylim(0, 250)
# sliders = AxesSliders(ax, 0.2, 0.7, slide_axis='y')
# sliders.connect()

# plt.show()
