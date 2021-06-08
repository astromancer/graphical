from matplotlib.transforms import Transform


class ReciprocalTransform(Transform):
    input_dims = 1
    output_dims = 1
    is_separable = False
    has_inverse = True

    def __init__(self, thresh=1e-6):
        Transform.__init__(self)
        self.thresh = thresh

    def transform_non_affine(self, x):
        mask = abs(x) < self.thresh
        # x[mask] = np.ma.masked#self.thresh
        # if any(mask):
        #     x = np.ma.masked_where(mask`, x)
        # return 1. / xm
        # else:
        return 1. / x

    def inverted(self):
        return ReciprocalTransform()