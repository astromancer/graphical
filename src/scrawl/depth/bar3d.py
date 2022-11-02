"""
3D bar plots that actually render correctly.
"""

# std
import numbers

# third-party
import numpy as np
from loguru import logger
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, PolyCollection

# relative
from ..moves import mpl_connect
from ..moves.machinery import CanvasBlitHelper, Observers
from . import camera
from .zaxis_cbar import ZAxisCbar


# ---------------------------------------------------------------------------- #
# shape (6, 4, 3)
# All faces are oriented facing outwards - when viewed from the
# outside, their vertices are in a counterclockwise ordering.
CUBOID = np.array([
    # -z
    (
        (0, 0, 0),
        (0, 1, 0),
        (1, 1, 0),
        (1, 0, 0),
    ),
    # +z
    (
        (0, 0, 1),
        (1, 0, 1),
        (1, 1, 1),
        (0, 1, 1),
    ),
    # -y
    (
        (0, 0, 0),
        (1, 0, 0),
        (1, 0, 1),
        (0, 0, 1),
    ),
    # +y
    (
        (0, 1, 0),
        (0, 1, 1),
        (1, 1, 1),
        (1, 1, 0),
    ),
    # -x
    (
        (0, 0, 0),
        (0, 0, 1),
        (0, 1, 1),
        (0, 1, 0),
    ),
    # +x
    (
        (1, 0, 0),
        (1, 1, 0),
        (1, 1, 1),
        (1, 0, 1),
    ),
])


CAMERA_VIEW_QUADRANT_TO_CUBE_FACE_ZORDER = {
    #          -z, +z, -y, +y, -x, +x
    #           0,  1,  2,  3,  4,  5

    # viewing | cube face
    # quadrant| indices                 | face name
    0:         (5, 0, 4, 1, 3, 2),      # '-y', '-x'
    1:         (5, 0, 4, 1, 2, 3),      # '-y', '+x'
    2:         (5, 0, 1, 4, 2, 3),      # '+y', '+x'
    3:         (5, 0, 1, 4, 3, 2)       # '+y', '-x'
}


# ---------------------------------------------------------------------------- #
# TODO: back panels don't need to be drawn if alpha == 1


def get_cube_face_zorder(ax):
    # -z, +z, -y, +y, -x, +x
    # 0,  1,   2,  3,  4,  5

    view_quadrant = int((ax.azim % 360) // 90)
    idx = CAMERA_VIEW_QUADRANT_TO_CUBE_FACE_ZORDER[view_quadrant]
    order = np.array(idx)

    if (ax.elev % 180) > 90:
        order[:2] = order[1::-1]

    logger.trace('Panel draw order quadrant {}:\n{}\n{}', view_quadrant,  order,
                 list(np.take(['-z', '+z', '-y', '+y', '-x', '+x'],
                              order)))

    return order


def is_one_colour(c):
    return (isinstance(c, (str, numbers.Real)) or np.size(c) in {3, 4})


class Bar3DCollection(Poly3DCollection):

    def __init__(self, x, y, z, dxy=0.8, shade=True, **kws):
        #
        assert 0 < dxy <= 1

        self.xyz = np.atleast_3d([x, y, z])
        self.dxy = dxy = float(dxy)

        # bar width and breadth
        d = [dxy, dxy]
        for i, s in enumerate(self.xyz.shape[1:]):
            d[i], =  dxy * (np.array([1]) if s == 1 else
                            np.diff(self.xyz[(i, *(0, np.s_[:2])[::(1, -1)[i]])]))
        dx, dy = d
        assert (dx != 0) & (dy != 0)

        # Shade faces by angle to light source
        self._shade = bool(shade)
        self._original_alpha = kws.pop('alpha', None)

        # rectangle polygon vertices
        verts = self._compute_verts()

        # init Poly3DCollection
        print(kws)
        Poly3DCollection.__init__(self, verts, **kws)  # facecolor=facecolors,
        self.set_array(z.ravel())

    @property
    def xy(self):
        return self.xyz[:2]

    @property
    def z(self):
        return self.xyz[1]

    def set_data(self, xyz):
        self.xyz = np.atleast_3d(xyz)
        super().set_verts(self._compute_verts())

        # if self._face_is_mapped or self._edge_is_mapped:
        #     self.set_array(np.tile(self.z, (6,1,1)))

        self.do_3d_projection()

    # def _compute_verts(self):
    #     # indexed by [bar, face, vertex, coord]
    #     shape = (*self.xyz.shape[1:], 1, 1, 3)
    #     xyz = np.zeros(shape)
    #     xyz[..., 0, 0, :2] = np.moveaxis(self.xy, 0, -1)
    #     dxyz = np.full(shape, self.dxy)
    #     dxyz[..., 0, 0, -1] = self.z

    #     verts = xyz + dxyz * CUBOID
    #     # collapse all but the last two axes
    #     return verts.reshape((-1, 4, 3))

    def _compute_verts(self, xyz=None):
        # indexed by [bar, face, vertex, coord]
        if xyz is None:
            xyz = self.xyz

        # indexed by [bar, face, vertex, coord]
        x, y, dz = xyz
        dx = dy = np.full(x.shape, self.dxy)
        # handle each coordinate separately
        polys = np.empty(x.shape + CUBOID.shape)
        for i, p, dp in [(0, x, dx), (1, y, dy), (2, np.zeros_like(x), dz)]:
            p = p[..., np.newaxis, np.newaxis]
            dp = dp[..., np.newaxis, np.newaxis]
            polys[..., i] = p + dp * CUBOID[..., i]

        # collapse the first two axes
        return polys.reshape((-1,) + polys.shape[-2:])

    def do_3d_projection(self):
        """
        Perform the 3D projection for this object.
        """
        if self._A is not None:
            # force update of color mapping because we re-order them
            # below.  If we do not do this here, the 2D draw will call
            # this, but we will never port the color mapped values back
            # to the 3D versions.
            #
            # We hold the 3D versions in a fixed order (the order the user
            # passed in) and sort the 2D version by view depth.
            self.update_scalarmappable()
            if self._face_is_mapped:
                self._facecolor3d = self._facecolors
            if self._edge_is_mapped:
                self._edgecolor3d = self._edgecolors

        txs, tys, tzs = proj3d._proj_transform_vec(self._vec, self.axes.M)
        xyzlist = [(txs[sl], tys[sl], tzs[sl]) for sl in self._segslices]

        # get panel facecolors
        cface, cedge = self._resolve_colors(xyzlist)

        if xyzlist:
            zorder = self._compute_zorder()

            z_segments_2d = sorted(
                ((zo, np.column_stack([xs, ys]), fc, ec, idx)
                 for idx, (zo, (xs, ys, _), fc, ec)
                 in enumerate(zip(zorder, xyzlist, cface, cedge))),
                key=lambda x: x[0], reverse=True)

            _, segments_2d, self._facecolors2d, self._edgecolors2d, idxs = \
                zip(*z_segments_2d)
        else:
            segments_2d = []
            self._facecolors2d = np.empty((0, 4))
            self._edgecolors2d = np.empty((0, 4))
            idxs = []

        if self._codes3d is None:
            PolyCollection.set_verts(self, segments_2d, self._closed)
        else:
            codes = [self._codes3d[idx] for idx in idxs]
            PolyCollection.set_verts_and_codes(self, segments_2d, codes)

        if len(self._edgecolor3d) != len(cface):
            self._edgecolors2d = self._edgecolor3d

        # Return zorder value
        if self._sort_zpos is not None:
            zvec = np.array([[0], [0], [self._sort_zpos], [1]])
            ztrans = proj3d._proj_transform_vec(zvec, self.axes.M)
            return ztrans[2][0]
        elif tzs.size > 0:
            # FIXME: Some results still don't look quite right.
            #        In particular, examine contourf3d_demo2.py
            #        with az = -54 and elev = -45.
            return np.min(tzs)
        else:
            return np.nan

    def _resolve_colors(self, xyzlist):
        # This extra fuss is to re-order face / edge colors
        cface = self._facecolor3d
        cedge = self._edgecolor3d
        if len(cface) * 6 == len(xyzlist):
            cface = cface.repeat(6, axis=0)
            if self._shade:
                verts = self._compute_verts()
                ax = self.axes
                normals = ax._generate_normals(verts)
                cface = ax._shade_colors(cface, normals)
            if self._original_alpha:
                cface[:, -1] = self._original_alpha

        if len(cface) != len(xyzlist):
            cface = cface.repeat(len(xyzlist), axis=0)

        if len(cedge) != len(xyzlist):
            cedge = cface if len(cedge) == 0 else cedge.repeat(len(xyzlist), axis=0)

        return cface, cedge

    def _compute_zorder(self):
        # sort by depth (furthest drawn first)
        zorder = camera.distance(self.axes, *self.xy)
        zorder = (zorder - zorder.min()) / zorder.ptp()
        zorder = zorder.ravel() * len(zorder)
        panel_order = get_cube_face_zorder(self.axes)
        zorder = (zorder[..., None] + panel_order / 6).ravel()
        return zorder


class Bar3D(CanvasBlitHelper):  # Bar3DGrid
    """
    3D bar plot that renders correctly for different viewing angles.
    """
    #  TODO inherit from collection ??

    @classmethod
    def from_image(cls, ax, image, dxy=0.8, **kws):
        origin = np.array(image.get_extent())[[0, 2]]
        y, x = np.indices(image.shape) + origin[:, None, None]
        cmap = image.get_cmap()
        return cls(ax, x, y, image.get_array(), dxy,
                   color=cmap(image.norm(image)),
                   **kws)

    def __init__(self, ax, x, y, z, dxy=0.8, cmap=None, shade=True,
                 zaxis_cbar=False, **kws):
        #
        assert ax.name == '3d'
        assert 0 < dxy <= 1

        had_data = ax.has_data()

        self.ax = self.axes = ax
        self.bars = Bar3DCollection(x, y, z, dxy, shade, cmap=cmap, **kws)
        ax.add_collection(self.bars)

        ax.auto_scale_xyz((x.min(), (x + x[0, :2].ptp()).max()),
                          (y.min(), (y + y[:2, 0].ptp()).max()),
                          (z.min(), z.max()),
                          had_data)

        self.cbar = None
        if zaxis_cbar:
            clim = list(self.bars.get_clim())
            if clim[0] is None:
                clim[0] = z.min()
            if clim[1] is None:
                clim[1] = z.max()

            self.cbar = ZAxisCbar(self.axes, cmap, )

        CanvasBlitHelper.__init__(self, (self.bars, self.cbar.line),
                                  connect=True)

        self.on_rotate = Observers()

    def set_cmap(self, cmap):
        self.bars.set_cmap(cmap)
        self.cbar.line.set_cmap(cmap)

    @mpl_connect('motion_notify_event')
    def on_rotate(self, event):  # sourcery skip: de-morgan
        # Redo zorder when rotating axes
        if ((event.inaxes is not self.ax) or
                (self.ax.button_pressed not in self.ax._rotate_btn)):
            return

        logger.debug('Rotating axes: {}', self.ax)
        self.on_rotate(event.x, event.y)


# alias
bar3d = Bar3d = Bar3D
