import os
import tempfile
import unittest

import numpy as np

from . import BoundaryMesh2d, Idx, QuadraticBoundaryMesh2d, load_cgns, write_cgns
from .cgns_io import HAVE_CGNS


@unittest.skipIf(not HAVE_CGNS, "pycgns not available")
class TestCgnsIo2d(unittest.TestCase):
    def test_roundtrip_boundary_mesh_2d(self):
        coords = np.array(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float64
        )
        edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=Idx)
        etags = np.array([1, 1, 2, 2], dtype=np.int16)
        faces = np.empty((0, 1), dtype=Idx)
        ftags = np.empty(0, dtype=np.int16)
        msh = BoundaryMesh2d(coords, edges, etags, faces, ftags)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "square.cgns")
            write_cgns(msh, path, {"a": 1, "b": 2})
            back, names = load_cgns(path)

            self.assertEqual(names, {"a": 1, "b": 2})
            self.assertTrue(np.allclose(back.get_verts(), coords))
            self.assertTrue(np.array_equal(back.get_elems(), edges))
            self.assertTrue(np.array_equal(back.get_etags(), etags))

    def test_roundtrip_quadratic_boundary_mesh_2d(self):
        # Same square, but each edge is a BAR_3 bowed outward from its
        # linear midpoint. If write_cgns or load_cgns silently dropped the
        # curvature (e.g. read back as straight BAR_2, or swapped the
        # corners and mid-node), the checks below would catch it.
        corners = np.array(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float64
        )
        mids = np.array(
            [[0.5, -0.1], [1.1, 0.5], [0.5, 1.1], [-0.1, 0.5]], dtype=np.float64
        )
        coords = np.vstack([corners, mids])
        edges = np.array([[0, 1, 4], [1, 2, 5], [2, 3, 6], [3, 0, 7]], dtype=Idx)
        etags = np.array([1, 1, 2, 2], dtype=np.int16)
        faces = np.empty((0, 1), dtype=Idx)
        ftags = np.empty(0, dtype=np.int16)
        msh = QuadraticBoundaryMesh2d(coords, edges, etags, faces, ftags)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "curved_square.cgns")
            write_cgns(msh, path, {"a": 1, "b": 2})
            back, names = load_cgns(path)

            self.assertIsInstance(back, QuadraticBoundaryMesh2d)
            self.assertEqual(names, {"a": 1, "b": 2})
            self.assertTrue(np.allclose(back.get_verts(), coords))
            self.assertTrue(np.array_equal(back.get_elems(), edges))
            self.assertTrue(np.array_equal(back.get_etags(), etags))


if __name__ == "__main__":
    unittest.main()
