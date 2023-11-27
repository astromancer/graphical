
# third-party
import numpy as np

# local
from scrawl.image import ImageDisplay


# TODO TESTS:
# all zero data
# cbar, hist, sliders
# 0, 0, 0
# 0, 1, 0
# 1, 0, 0
# 1, 0, 1
# 1, 1, 0
# 1, 1, 1


# ---------------------------------------------------------------------------- #
def test_display():
    im = ImageDisplay(np.random.randn(100, 100))
