import unittest
import numpy as np
from . import Mesh2d, DualType, DualMesh2d


class TestExtrudedMeshes(unittest.TestCase):
    def test_mesh(self):
        nx = 10
        ny = 15
        msh = Mesh2d.rectangle_mesh(np.linspace(0, 1, nx), np.linspace(0, 2, ny))
        extruded = msh.extrude(1.0)
        msh2 = extruded.to_mesh2d()

        msh.check_equals(msh2, 1e-12)

    def test_dual(self):
        nx = 10
        ny = 15
        msh = Mesh2d.rectangle_mesh(np.linspace(0, 1, nx), np.linspace(0, 2, ny))
        dual = DualMesh2d(msh, DualType.Median)
        extruded = dual.extrude(1.0)

        self.assertEqual(extruded.n_elems(), msh.n_verts())
        (ptr, conn, orient) = extruded.get_elems()
        self.assertEqual(ptr.size, extruded.n_elems() + 1)
        self.assertEqual(ptr.min(), 0)
        self.assertEqual(ptr.max(), conn.size)
