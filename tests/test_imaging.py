
# third-party
import numpy as np

# local
from scrawl.image import ImageDisplay, hist
from loguru import logger


# ---------------------------------------------------------------------------- #
logger.enable('scrawl')

# ---------------------------------------------------------------------------- #
im = ImageDisplay(np.random.randn(100, 100))

# TESTS:
# all zero data
# cbar, hist, sliders
# 0, 0, 0
# 0, 1, 0
# 1, 0, 0
# 1, 0, 1
# 1, 1, 0
# 1, 1, 1



if __name__ == '__main__':
    import matplotlib.pyplot as plt

    plt.show()
