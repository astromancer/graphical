
# third-party
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# local
from scrawl import density


def test_scatter():

    fig, ax = plt.subplots()
    rv = multivariate_normal((0, 0),
                             [[0.8,  0.3],
                              [0.3,  0.5]])
    n = 10_000
    hvals, poly, points = density.scatter_map(ax, rv.rvs(n),
                                              min_count=7,
                                              density_kws={'alpha': 0.5})


if __name__ == '__main__':
    test_scatter()
    plt.show()
