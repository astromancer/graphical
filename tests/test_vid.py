
# third-party
import numpy as np
from loguru import logger

# local
from scrawl.video import VideoDisplay


# ---------------------------------------------------------------------------- #

def test_vid():
    vid = VideoDisplay(np.random.randn(10, 12, 12))
