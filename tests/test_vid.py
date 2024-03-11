
# third-party
import numpy as np

# local
from scrawl.video import VideoDisplay


# ---------------------------------------------------------------------------- #

def test_vid():
    vid = VideoDisplay(np.random.randn(10, 12, 12))
