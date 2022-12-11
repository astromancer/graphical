
# third-party
import numpy as np
from loguru import logger

# local
from scrawl.video import VideoDisplay, ImageDisplay


logger.enable('scrawl')

# ---------------------------------------------------------------------------- #

vid = VideoDisplay(np.random.randn(10, 12, 12))


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    plt.show()
