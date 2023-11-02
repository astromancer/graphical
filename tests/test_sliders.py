
# third-party
import matplotlib.pyplot as plt

# local
from scrawl.sliders import RangeSliders


if __name__ == '__main__':

    fig, ax = plt.subplots()
    ax.set_xlim(0, 1e4)
    x0, x1 = 0.25, 0.55

    #createHistogram(ax, np.random.rand(int(1e4)))
    sliders = RangeSliders(ax, (x0, x1), 'y')
    sliders.connect()

    # print('!' * 88)
    # print(str(sliders.lower), str(sliders.upper))
    # print('!' * 88)

    plt.show()
