

from matplotlib.transforms import blended_transform_factory as btf
from matplotlib.transforms import Affine2D


def line_picker(artist, event):
    ''' an artist picker that works for clicks outside the axes'''
    #print('line picker')
    if event.button != 1:  # otherwise intended reset will select
        #print('wrong button asshole!')
        return False, {}

    props = {}

    transformed_path = artist._get_transformed_path()
    path, affine = transformed_path.get_transformed_path_and_affine()
    path = affine.transform_path(path)
    xy = path.vertices
    xt = xy[:, 0]
    yt = xy[:, 1]

    # Convert pick radius from points to pixels
    pixels = artist.figure.dpi / 72. * artist.pickradius

    xd, yd = xt - event.x, yt - event.y
    prox = xd ** 2 + yd ** 2  # distance of click from points in pixels (display coords)
    picked = prox - pixels**2 < 0

    #print(picked, '!!')

    return picked.any(), props


class MovableLine():

    annotation_format = '[%+3.2f]'

    def __init__(self, line, **kws):
        ''' '''

        #Line2D.__init__(self, *line.get_data())
        # self.update_from(line)
        self.line = line

        self.offset = kws.pop('offset', 0)
        self.annotated = kws.pop('annotate', True)

        self._original_transform = line.get_transform()

        # make the lines pickable
        if not line.get_picker():
            line.set_picker(line_picker)

        # Initialize offset texts
        ax = line.axes
        self.text_trans = btf(ax.transAxes, ax.transData)
        ytxt = line.get_ydata().mean()
        self.annotation = ax.text(1.005, ytxt, '')
        # transform=self.text_trans )

        # shift to the given offset (default 0)
        # self.shift(self.offset)

    def shift(self, offset):
        '''Shift the data by offset by setting the transform '''
        # add the offset to the y coordinate
        offset_trans = Affine2D().translate(0, offset)
        trans = offset_trans + self._original_transform
        self.line.set_transform(trans)

        # NOTE: can avoid this if statement by subclassing...
        if self.annotated:
            txt = self.annotation
            txt.set_transform(offset_trans + self.text_trans)
            txt.set_text(self.annotation_format % offset)


# class MovableLine(MotionInterface):
