"""
3D prism plots (bars and hexagons).
"""

# std
import warnings

# third-party
import numpy as np
import more_itertools as mit
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import (Poly3DCollection, PolyCollection,
                                        _generate_normals, _shade_colors)

# local
from recipes.config import ConfigNode
from recipes.functionals import is_none
from recipes.containers import duplicate_if_scalar

# relative
from ..moves.machinery import CanvasBlitHelper, Observers
from . import camera
from .zaxis_cbar import ZAxisCbar


# ---------------------------------------------------------------------------- #
# module config
CONFIG = ConfigNode.load_module(__file__)


# ---------------------------------------------------------------------------- #
# chosen for backwards-compatibility
CLASSIC_LIGHTSOURCE = LightSource(azdeg=225, altdeg=19.4712)

# Unit cube
# All faces are oriented facing outwards - when viewed from the
# outside, their vertices are in a counterclockwise ordering.
# shape (6, 4, 3)
# panel order:  -x, -y, +x, +y, -z, +z
CUBOID = np.array([
    # -x
    (
        (0, 0, 0),
        (0, 0, 1),
        (0, 1, 1),
        (0, 1, 0),
    ),
    # -y
    (
        (0, 0, 0),
        (1, 0, 0),
        (1, 0, 1),
        (0, 0, 1),
    ),
    # +x
    (
        (1, 0, 0),
        (1, 1, 0),
        (1, 1, 1),
        (1, 0, 1),
    ),
    # +y
    (
        (0, 1, 0),
        (0, 1, 1),
        (1, 1, 1),
        (1, 1, 0),
    ),
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

])


# Base hexagon for creating prisms (HexBar3DCollection).
# sides are ordered anti-clockwise from left: ['W', 'SW', 'SE', 'E', 'NE', 'NW']
# autopep8: off
HEXAGON = np.array([
    [-2,  1],
    [-2, -1],
    [ 0, -2],
    [ 2, -1],
    [ 2,  1],
    [ 0,  2]
]) / 4
# autopep8: on


# ---------------------------------------------------------------------------- #


class Bar3DCollection(Poly3DCollection):
    """
    Bars (rectangular prisms) with constant square cross section, bases located
    on z-plane at *z0*, arranged in a regular grid at *x*, *y* locations and
    with height *z - z0*.
    """

    _n_faces = 6

    def __init__(self, x, y, z, dxy=CONFIG.dxy, z0=0,
                 shade=True, lightsource=None, cmap=CONFIG.cmap, **kws):
        #
        x, y, z, z0 = np.ma.atleast_1d(x, y, z, z0)
        assert x.shape == y.shape == z.shape

        # array for bar positions, height (x, y, z)
        self._xyz = np.empty((3, *x.shape))
        for i, p in enumerate((x, y, z)):
            if p is not None:
                self._xyz[i] = p

        # bar width and breadth
        self.dxy = dxy
        self.dx, self.dy = self._resolve_dx_dy(dxy)

        if z0 is not None:
            self.z0 = float(z0)

        # Shade faces by angle to light source
        self._original_alpha = kws.pop('alpha', 1)
        self._shade = bool(shade)
        if lightsource is None:
            # chosen for backwards-compatibility
            lightsource = CLASSIC_LIGHTSOURCE
        else:
            assert isinstance(lightsource, LightSource)
        self._lightsource = lightsource

        COLOR_KWS = {'color', 'facecolor', 'facecolors'}
        if cmap is not None:
            if (ckw := COLOR_KWS.intersection(kws)):
                warnings.warn(f'Ignoring cmap since {ckw!r} provided.')
            else:
                kws.update(cmap=cmap)

        # init Poly3DCollection
        #                               rectangle side panel vertices
        Poly3DCollection.__init__(self, self._compute_verts(), **kws)

        if cmap:
            self.set_array(self.z.ravel())

    def _resolve_dx_dy(self, dxy):

        d = list(duplicate_if_scalar(dxy))

        for i, xy in enumerate(self.xy):
            # if dxy a number -> use it directly else if str,
            # scale dxy to data step.
            # get x/y step along axis -1/-2 (x/y considered constant along axis
            # -2/-1)
            data_step = _get_grid_step(xy, -i - 1) if isinstance(d[i], str) else 1
            d[i] = float(d[i]) * data_step

        dx, dy = d
        assert (dx != 0)
        assert (dy != 0)

        return dx, dy

    @property
    def x(self):
        return self._xyz[0]

    @x.setter
    def x(self, x):
        self.set_data(x=x)

    @property
    def y(self):
        return self._xyz[1]

    @y.setter
    def y(self, y):
        self.set_data(y=y)

    @property
    def xy(self):
        return self._xyz[:2]

    @property
    def z(self):
        return self._xyz[2]

    def set_z(self, z, z0=None, clim=None):
        self.set_data(z=z, z0=z0, clim=clim)

    def set_z0(self, z0):
        self.z0 = float(z0)
        super().set_verts(self._compute_verts())

    def set_data(self, x=None, y=None, z=None, z0=None, clim=None):
        # self._xyz = np.atleast_3d(xyz)
        assert not all(map(is_none, (x, y, z, z0)))

        if (x is not None) or (y is not None):
            self._resolve_dx_dy(self.dxy)

        for i, p in enumerate((x, y, z)):
            if p is not None:
                self._xyz[i] = p

        if z0 is not None:
            self.z0 = float(z0)

        # compute points
        super().set_verts(self._compute_verts())
        self.set_array(z := self.z.ravel())

        if clim is None or clim is True:
            clim = (z.min(), z.max())

        if clim is not False:
            self.set_clim(*clim)

        if not self.axes:
            return

        if self.axes.M is not None:
            self.do_3d_projection()

    def _compute_verts(self):

        x, y = self.xy
        z = np.full(x.shape, self.z0)

        # indexed by [bar, face, vertex, axis]
        xyz = np.expand_dims(np.moveaxis([x, y, z], 0, -1), (-2, -3))
        dxyz = np.empty_like(xyz)
        dxyz[..., :2] = np.array([[[self.dx]], [[self.dy]]]).T
        dxyz[..., 2] = np.array(self.z - self.z0)[..., np.newaxis, np.newaxis]
        polys = xyz + dxyz * CUBOID[np.newaxis, :]  # (n, 6, 4, 3)

        # collapse the first two axes
        return polys.reshape((-1, 4, 3))  # *polys.shape[-2:]

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
        cface, cedge = self._compute_colors(xyzlist, self._lightsource)

        if xyzlist:
            zorder = self._compute_zorder()
            occluded = np.isnan(zorder)

            z_segments_2d = sorted(
                ((zo, np.column_stack([xs, ys]), fc, ec, idx)
                 for idx, (ok, zo, (xs, ys, _), fc, ec)
                 in enumerate(zip(~occluded, zorder, xyzlist, cface, cedge))
                 if ok),
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

        return np.min(tzs) if tzs.size > 0 else np.nan

    def _compute_colors(self, xyzlist, lightsource):
        # This extra fuss is to re-order face / edge colors
        cface = self._facecolor3d
        cedge = self._edgecolor3d
        n, nc = len(xyzlist), len(cface)

        if (nc == 1) or (nc * self._n_faces == n):
            cface = cface.repeat(n // nc, axis=0)
            if self._shade:
                verts = self._compute_verts()
                normals = _generate_normals(verts)
                cface = _shade_colors(cface, normals, lightsource)

            if self._original_alpha is not None:
                cface[:, -1] = self._original_alpha

        if len(cface) != n:
            raise ValueError
            # cface = cface.repeat(n, axis=0)

        if len(cedge) != n:
            cedge = cface if len(cedge) == 0 else cedge.repeat(n, axis=0)

        return cface, cedge

    def _compute_zorder(self):
        # sort by depth (furthest drawn first)
        zorder = camera.distance(self.axes, *self.xy)
        zorder = (zorder - zorder.min()) / (np.ptp(zorder) or 1)
        zorder = zorder.ravel() * len(zorder)
        face_zorder = get_prism_face_zorder(self.axes,
                                            self._original_alpha == 1,
                                            self._n_faces - 2)
        return (zorder[..., None] + face_zorder).ravel()


class HexBar3DCollection(Bar3DCollection):
    """
    Hexagonal prisms with uniform cross section, bases located on z-plane at *z0*,
    aranged in a regular grid at *x*, *y* locations and height *z - z0*.
    """
    _n_faces = 8

    def _compute_verts(self):
        new = np.newaxis
        # scale the base hexagon
        hexagon = np.array([self.dx, self.dy]).T * HEXAGON
        xy_pairs = np.moveaxis([hexagon, np.roll(hexagon, -1, 0)], 0, 1)
        xy_sides = xy_pairs[np.newaxis] + self.xy[:, new, new].T  # (n,6,2,2)

        # sides (rectangle faces)
        # Array of vertices of the faces composing the prism moving counter
        # clockwise when looking from above starting at west (-x) facing panel.
        # Vertex sequence is counter-clockwise when viewed from outside.
        # shape:     (n, [m ...], 6,    4,      3)
        # indexed by [bars ... ,  face, vertex, axis]
        data_shape = np.shape(self.z)
        shape = (*data_shape, 6, 2, 1)
        z0 = np.full(shape, self.z0)
        z1 = self.z0 + (self.z * np.ones(shape[::-1])).T
        sides = np.concatenate(
            [np.concatenate([xy_sides, z0], -1),
             np.concatenate([xy_sides, z1], -1)[..., ::-1, :]],
            axis=-2)  # (n, [m ...], 6, 4, 3)

        # endcaps (hexagons) # (n, [m ...], 6, 3)
        xy_ends = (self.xy[..., new] + hexagon.T[:, new])
        z0 = self.z0 * np.ones((1, *data_shape, 6))
        z1 = z0 + self.z[new, ..., new]
        base = np.moveaxis(np.vstack([xy_ends, z0]), 0, -1)
        top = np.moveaxis(np.vstack([xy_ends, z1]), 0, -1)

        # get list of arrays of polygon vertices
        verts = []
        for s, b, t in zip(sides, base, top):
            verts.extend([*s, b, t])

        return verts


# ---------------------------------------------------------------------------- #
def _get_grid_step(x, axis=0):

    # deal with singular dimension (this ignores axis param)
    if x.ndim == 1:
        if d := next(filter(None, map(np.diff, mit.pairwise(x))), None):
            return d

    if x.shape[axis % x.ndim] == 1:
        return 1

    key = [0] * x.ndim
    key[axis] = np.s_[:2]
    return np.diff(x[tuple(key)]).item()


def get_prism_face_zorder(ax, mask_occluded=True, nfaces=4):
    # compute panel draw sequence based on camera position

    # these index positions are determined by the order of the faces returned
    # by `_compute_verts`

    # horizontal faces
    base, top = nfaces, nfaces + 1
    if ax.elev < 0:
        # flip order if viewed from below
        base, top = top, base

    # vertical faces
    angle = 360 / nfaces
    zero = -angle / 2  # starting point of the first vertical face
    flip = (np.abs(ax.elev) % 180 > 90)
    sector = (((ax.azim - zero + 180 * flip) % 360) / angle) % nfaces

    # get indices for panels in draw order
    first = int(sector)
    second = (first + 1) % nfaces
    third = (first + nfaces - 1) % nfaces
    if (sector - first) < 0.5:
        second, third = third, second

    # get indices for panels in plot order
    sequence = [base, first, second, third]
    sequence = [*sequence, *np.setdiff1d(np.arange(nfaces), sequence), top]

    # reverse the panel sequence if elevation has flipped the axes by 180 multiple
    if np.abs(ax.elev) % 360 > 180:
        sequence = sequence[::-1]

    # normalize zorder to < 1
    zorder = np.argsort(sequence) / len(sequence)

    if mask_occluded:
        #  we don't need to draw back panels since they are behind others
        zorder[zorder < 0.5] = np.nan

    # # This order is determined by the ordering of `CUBOID` and `HEXAGON` globals
    # names = {4: ['+x', '+y',  '-x', '-y', '-z', '+z'],
    #          6: ['W', 'SW', 'SE', 'E', 'NE', 'NW', 'BASE', 'TOP']}[nfaces]
    # print('',
    #       f'Panel draw sequence ({ax.azim = :}, {ax.elev = :}):',
    #       f'{sector = :}',
    #       f'{sequence = :}',
    #       f'names = {list(np.take(names, sequence))}',
    #       f'{zorder = :}',
    #       f'zorder = {pformat(dict(zip(*cosort(zorder, names)[::-1])))}',
    #       sep='\n')

    return zorder


# ---------------------------------------------------------------------------- #

PRISM_WORKERS = dict(
    rect=Bar3DCollection,
    hex=HexBar3DCollection
)


class Bar3D(CanvasBlitHelper):  # Bar3DGrid
    """
    3D bar plot blit and cmap support.
    """

    # @classmethod
    # def from_image(cls, ax, image, dxy=CONFIG.dxy, **kws):
    #     # from matplotlib image
    #     origin = np.array(image.get_extent())[[0, 2]]
    #     y, x = np.indices(image.shape) + origin[:, None, None]
    #     cmap = image.get_cmap()
    #     return cls(ax, x, y, image.get_array(), dxy, cmap=cmap, **kws)

    @classmethod
    def hex(cls, ax,  x, y, z, dxy=CONFIG.dxy, **kws):
        return cls(ax, x, y, z, dxy, tessellation='hex', **kws)

    def __init__(self, ax, x, y, z, dxy=CONFIG.dxy, cmap=CONFIG.cmap,
                 zaxis_cbar=False, tessellation='rect', **kws):
        #
        assert ax.name == '3d'

        self.ax = self.axes = ax
        had_data = ax.has_data()

        # init Bar3DCollection
        Bar3D = PRISM_WORKERS[tessellation]
        self.bars = bars = Bar3D(x, y, z, dxy, cmap=cmap, **kws)
        ax.add_collection(bars)

        viewlim = np.array([(np.min(x), np.max(np.add(x, bars.dx))),
                            (np.min(y), np.max(np.add(y, bars.dy))),
                            (bars.z0, np.max(np.add(bars.z0, z)))])
        if tessellation == 'hex':
            viewlim[:2, 0] = viewlim[:2, 0] - np.array([bars.dx, bars.dy] / 2).T

        ax.auto_scale_xyz(*viewlim, had_data)

        self.cbar = None
        if zaxis_cbar:
            clim = list(self.bars.get_clim())
            if clim[0] is None:
                clim[0] = z.min()
            if clim[1] is None:
                clim[1] = z.max()

            self.cbar = ZAxisCbar(self.axes, cmap, )

        super().__init__((self.bars, getattr(self.cbar, 'line', None)),
                         connect=True)

        self.on_rotate = Observers()

    def set_cmap(self, cmap):
        self.bars.set_cmap(cmap)
        self.cbar.line.set_cmap(cmap)

    def set_z(self, z, clim=None, zlim=None):

        self.bars.set_z(z, clim=clim)

        if zlim is False:
            return

        if zlim is None or zlim is True:
            zlim = clim

        if zlim is False:
            return

        if zlim is None or zlim is True:
            zlim = self.bars.get_clim()

        self.axes.set_zlim(*zlim)

        # update zaxis cbar
        self.cbar.xyz[2] = np.linspace(*zlim)
        self.cbar.update()

    # @mpl_connect('motion_notify_event')
    # def _on_rotate(self, event):  # sourcery skip: de-morgan
    #     # Redo zorder when rotating axes
    #     if ((event.inaxes is not self.ax) or
    #             (self.ax.button_pressed not in self.ax._rotate_btn)):
    #         return

    #     logger.debug('Rotating axes: {}.', self.ax)
    #     self.on_rotate(event.x, event.y)


# alias
bar3d = Bar3d = Prisms = Bar3D
