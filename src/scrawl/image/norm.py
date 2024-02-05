
# third-party
from astropy.visualization.stretch import BaseStretch
from astropy.visualization.interval import BaseInterval
from astropy.visualization.mpl_normalize import ImageNormalize

# local
from recipes.oo.utils import subclasses

# relative
from .utils import _sanitize_data


def get_norm(image, interval, stretch):
    """

    Parameters
    ----------
    image
    interval
    stretch

    Returns
    -------

    """
    # choose colour interval algorithm based on data type
    if image.dtype.kind == 'i' and image.ptp() < 1000:   # integer array
        interval = 'minmax'

    # determine colour transform from `interval` and `stretch`
    if isinstance(interval, str):
        interval = interval,
    interval = Interval.from_name(*interval)
    #
    if isinstance(stretch, str):
        stretch = stretch,
    stretch = Stretch.from_name(*stretch)

    # Create an ImageNormalize object
    return ImageNormalize(image, interval, stretch=stretch)


class FromNameMixin:
    @classmethod
    def from_name(cls, method, *args, **kws):
        """
        Construct derived subtype from `method` string and `kws`
        """

        from recipes import iter as itr

        if not isinstance(method, str):
            raise TypeError('method should be a string.')

        allowed_names = set()
        for sub in subclasses(cls.__bases__[0]):
            name = sub.__name__
            if name.lower().startswith(method.lower()):
                break
            else:
                allowed_names.add(name)

        else:
            raise ValueError(f'Unrecognized method {method!r}. Please use one'
                             f' of the following {tuple(allowed_names)}.')

        return sub(*args, **kws)


class Interval(BaseInterval, FromNameMixin):
    def get_limits(self, values):
        print('hi')  # FIXME: this is missed
        return BaseInterval.get_limits(self, _sanitize_data(values))


class Stretch(BaseStretch, FromNameMixin):
    pass


# class ImageNormalize(mpl_normalize.ImageNormalize):
#
#     # FIXME: ImageNormalize fills masked arrays with vmax instead of removing
# them.
#     # this skews the colour distribution.  TODO: report bug
#
#     def __init__(self, data=None, *args, **kws):
#         if data is not None:
#             data = _sanitize_data(data)
#
#         mpl_normalize.ImageNormalize.__init__(self, data, *args, **kws)
#
#     def __call__(self, values, clip=None):
#         return mpl_normalize.Normalize.__call__(
#                 self, _sanitize_data(values), clip)

