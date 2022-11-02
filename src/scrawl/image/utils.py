"""
Image plotting utilities.
"""

# third-party
import numpy as np
from loguru import logger

FIGSIZE_MIN_INCHES = (5, 5)


def _sanitize_data(data):
    """
    Removes nans and masked elements
    Returns flattened array
    """
    if np.ma.is_masked(data):
        data = data[~data.mask]
    return np.asarray(data[~np.isnan(data)])


def move_axes(ax, x, y):
    """Move the axis in the figure by x, y"""
    l, b, w, h = ax.get_position(True).bounds
    ax.set_position((l + x, b + y, w, h))


def get_screen_size_inches():
    """
    Use QT to get the size of the primary screen in inches.

    Returns
    -------
    size_inches: list

    """
    import sys
    from PyQt5.QtWidgets import QApplication, QDesktopWidget

    # Note the check on QApplication already running and not executing the exit
    #  statement at the end.
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    else:
        logger.debug('Retrieving screen size for existing QApplication '
                     f'instance: {app}')

    # TODO: find out on which screen the focus is

    w = QDesktopWidget()
    s = w.screen()
    size_inches = [s.width() / s.physicalDpiX(), s.height() / s.physicalDpiY()]
    logger.info('Screen size is: {}', size_inches)
    w.close()
    return size_inches

    # screens = app.screens()
    # size_inches = np.empty((len(screens), 2))
    # for i, s in enumerate(screens):
    #     g = s.geometry()
    #     size_inches[i] = np.divide(
    #             [g.height(), g.width()], s.physicalDotsPerInch()
    #     )
    # app.exec_()
    # return size_inches


def guess_figsize(image, fill_factor=0.75, max_pixel_size=0.2,
                  min_size_inches=FIGSIZE_MIN_INCHES):
    """
    Make an educated guess of the size of the figure needed to display the
    image data.

    Parameters
    ----------
    image: np.ndarray
        Sample image
    fill_factor: float
        Maximal fraction of screen size allowed in any direction
    min_size: 2-tuple
        Minimum allowed size (width, height) in inches
    max_pixel_size: float
        Maximum allowed pixel size

    Returns
    -------
    size: tuple
        Size (width, height) of the figure in inches
    """

    # Sizes reported by mpl figures seem about half the actual size on screen
    shape = np.array(np.shape(image)[::-1])
    return _guess_figsize(shape, fill_factor, max_pixel_size, min_size_inches)


def _guess_figsize(image_shape, fill_factor=0.75, max_pixel_size=0.2,
                   min_size=FIGSIZE_MIN_INCHES):

    # screen dimensions
    screen_size = np.array(get_screen_size_inches())

    # change order of image dimensions since opposite order of screen
    max_size = np.multiply(image_shape, max_pixel_size)

    # get upper limit for fig size based on screen and data and fill factor
    max_size = np.min([max_size, screen_size * fill_factor], 0)

    # get size from data
    aspect = image_shape / image_shape.max()
    size = max_size[aspect == 1] * aspect

    # enlarge =
    size *= max(np.max(min_size / size), 1)

    logger.debug('Guessed figure size: ({:.1f}, {:.1f})', *size)
    return size


def get_clim(data, plims=(0.25, 99.75)):
    """
    Get colour scale limits for data.
    """
    from .utils import get_percentile_limits

    if np.all(np.ma.getmask(data)):
        return None, None

    clims = get_percentile_limits(_sanitize_data(data), plims)

    bad_clims = (clims[0] == clims[1])
    if bad_clims:
        logger.warning('Ignoring bad colour interval: ({:.1f}, {:.1f}). ', *clims)
        return None, None

    return clims


def set_clim_connected(x, y, artist, sliders):
    artist.set_clim(*sliders.positions)
    return artist
