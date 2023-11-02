# third-party
import numpy as np

# local
from recipes.testing import expected
from scrawl.utils import percentile


# ---------------------------------------------------------------------------- #
np.random.seed(27189)
data = np.random.randn(1000)
mn, mx = data.min(), data.max()
p20, p30, p50, p80 = np.percentile(data, (20, 30, 50, 80))
ptp = mx - mn
md = p50 - mn


@expected({
    -250:   mn - (2 * ptp + md),      #: +4 * mn - 2 * mx - p50
    -200:   mn - (2 * ptp),           #: +3 * mn - 2 * mx
    -150:   mn - (1 * ptp + md),      #: +3 * mn - 1 * mx - p50
    -120:   mn - ptp - (p20 - mn),    #: +3 * mn - 1 * mx - p20
    -100:   mn - (1 * ptp),           #: +2 * mn - 1 * mx
    -50:    mn - md,                  #: +2 * mn - 0 * mx - p50
    +0:     mn,                       #: +1 * mn - 0 * mx
    +30:    p30,                      #: +0 * mn - 0 * mx + p30
    +50:    p50,                      #: +0 * mn - 0 * mx + p50
    +100:   mx,                       #: +0 * mn + 1 * mx
    +120:   mx + (mx - p80),          #: +0 * mn + 2 * mx - p80
    +150:   mx + (mx - p50),          #: +0 * mn + 2 * mx - p50
    +270:   mx + ptp + (mx - p30)     #: -1 * mn + 3 * mx - p30
},
    data=data
)
def test_plims(p, data, expected):
    z = percentile(data, p)
    tolerance = 1e-6
    assert (z - expected) < tolerance, (
        f'Expected percentile value of {expected:.3f} '
        f'does not match computed {z:.3f}.'
    )
