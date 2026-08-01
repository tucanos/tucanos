import unittest

import numpy as np

from . import BoundaryMesh2d, Idx, Mesh2d
from .curving import HAVE_FERREUS_RBF, p1_to_p2, p2_to_p1
from .mesh import get_square


def circle_boundary(center, radius, n, tag=1):
    theta = 2.0 * np.pi * np.linspace(0, 1, n, endpoint=False)
    coords = center + radius * np.stack([np.cos(theta), np.sin(theta)], axis=-1)
    edgs = np.stack([np.arange(n), (np.arange(n) + 1) % n], axis=-1).astype(Idx)
    etags = np.full(n, tag, dtype=np.int16)
    return BoundaryMesh2d(
        coords, edgs, etags, np.zeros([0, 1], dtype=Idx), np.zeros(0, dtype=np.int16)
    )


def square_boundary_geometry(sagitta, n=60):
    """Boundary geometry matching get_square()'s 4 tags (bottom=1, right=2,
    top=3, left=4): the bottom edge gently bulges outward by `sagitta`
    (a scale representative of a CAD-faceting correction, not a large shape
    change), the other 3 sides are exact straight-line matches of the mesh's
    own edges, so their projection is a no-op."""
    x = np.linspace(0.0, 1.0, n + 1)
    bottom = np.stack([x, -sagitta * 4.0 * x * (1.0 - x)], axis=-1)
    right = np.array([[1.0, 0.0], [1.0, 1.0]])
    top = np.array([[1.0, 1.0], [0.0, 1.0]])
    left = np.array([[0.0, 1.0], [0.0, 0.0]])

    coords = np.vstack([bottom, right, top, left])
    offsets = np.cumsum(
        [0, bottom.shape[0], right.shape[0], top.shape[0], left.shape[0]]
    )

    def segments(lo, hi, tag):
        idx = np.arange(lo, hi - 1)
        edgs = np.stack([idx, idx + 1], axis=-1).astype(Idx)
        etags = np.full(edgs.shape[0], tag, dtype=np.int16)
        return edgs, etags

    edgs_tags = [
        segments(offsets[i], offsets[i + 1], tag) for i, tag in enumerate([1, 2, 3, 4])
    ]
    edgs = np.vstack([e for e, _ in edgs_tags])
    etags = np.concatenate([t for _, t in edgs_tags])

    geom_bdy = BoundaryMesh2d(
        coords, edgs, etags, np.zeros([0, 1], dtype=Idx), np.zeros(0, dtype=np.int16)
    )
    geom_tag_names = {"bottom": 1, "right": 2, "top": 3, "left": 4}
    return geom_bdy, geom_tag_names


class TestCurving(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import logging

        logging.disable(logging.CRITICAL)

    def square_mesh_uniform_tag(self):
        # get_square() tags its 4 boundary edges separately (1..4); use a
        # single tag for the whole boundary instead, since project() panics
        # (uncatchable) if any tag *value* present on the mesh has no
        # counterpart in the geometry, matching how the validated
        # production case (a geometry file generated to cover every mesh
        # boundary tag) avoids that constraint.
        coords, elems, etags, faces, ftags = get_square(two_tags=False)
        ftags = np.ones_like(ftags)
        msh = Mesh2d(coords, elems, etags, faces, ftags).split().split()
        msh.fix()
        return msh

    def square_mesh_four_tags(self):
        coords, elems, etags, faces, ftags = get_square(two_tags=False)
        msh = Mesh2d(coords, elems, etags, faces, ftags).split().split()
        msh.fix()
        return msh

    def test_p2_to_p1_roundtrip_is_exact(self):
        msh = self.square_mesh_uniform_tag()
        qm = msh.to_quadratic()
        p1 = p2_to_p1(qm)

        self.assertEqual(p1.n_verts(), msh.n_verts())
        self.assertEqual(p1.get_elems().shape, msh.get_elems().shape)
        np.testing.assert_allclose(
            np.sort(p1.get_verts(), axis=0), np.sort(msh.get_verts(), axis=0)
        )

    def test_p1_to_p2_no_matching_tag_raises(self):
        msh = self.square_mesh_uniform_tag()
        # geometry's tag *value* (1) matches the mesh's, so project() itself
        # doesn't panic, but its *name* doesn't match tag_names, so no
        # control points are selected: the intended graceful failure mode.
        geom_bdy = circle_boundary(np.array([0.5, 0.5]), 0.5**0.5, 20)
        geom_tag_names = {"unrelated_name": 1}
        if not HAVE_FERREUS_RBF:
            with self.assertRaises(ImportError):
                p1_to_p2(msh, (geom_bdy, geom_tag_names), {1: "circle"})
            return
        with self.assertRaises(RuntimeError):
            p1_to_p2(msh, (geom_bdy, geom_tag_names), {1: "circle"})

    @unittest.skipUnless(HAVE_FERREUS_RBF, "ferreus_rbf not installed")
    def test_p1_to_p2_gentle_curving_stays_valid(self):
        msh = self.square_mesh_four_tags()
        sagitta = 0.02
        geom = square_boundary_geometry(sagitta)
        tag_names = {1: "bottom", 2: "right", 3: "top", 4: "left"}

        qm = p1_to_p2(msh, geom, tag_names)

        distortion = qm.distortion()
        self.assertTrue(
            np.all(distortion > 0), f"min distortion = {distortion.min():.4g}"
        )

        # the bottom edge should have bulged outward (y < 0 near its middle;
        # its endpoints are shared with the untouched left/right edges and
        # the bulge is 0 there by construction, so only check the extremum)
        verts = qm.get_verts()
        faces_p2 = qm.get_faces()
        ftags_p2 = qm.get_ftags()
        bottom_y = verts[np.unique(faces_p2[ftags_p2 == 1])][:, 1]
        self.assertLess(bottom_y.min(), -1e-4)
        self.assertLess(
            -bottom_y.min(), 2 * sagitta
        )  # ...but only by about `sagitta`, not further

        # ...while the other 3 sides, whose geometry exactly matches the
        # mesh's own straight edges, should barely have moved from their
        # pre-curving (straight-sided, to_quadratic()-only) position: a
        # small residual is expected, since the RBF fit is only accurate to
        # ~1e-3 relative (see p1_to_p2's docstring), not exact.
        straight_verts = msh.to_quadratic().get_verts()
        for tag in (2, 3, 4):
            idx = np.unique(faces_p2[ftags_p2 == tag])
            np.testing.assert_allclose(verts[idx], straight_verts[idx], atol=5e-4)
