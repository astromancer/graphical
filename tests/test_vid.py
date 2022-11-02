# third-party
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger

# local
from scrawl.video import VideoDisplay, ImageDisplay


logger.enable('scrawl')

# imd = ImageDisplay(np.random.randn(12, 12))
vid = VideoDisplay(np.random.randn(10, 12, 12))


if __name__ == '__main__':
    plt.show()
