import unittest
import numpy as np
from . import Mesh2d, Mesh3d

class TestMeshes(unittest.TestCase):

    def test_2d(self):

        nx = 10
        ny = 15
        msh = Mesh2d.rectangle_mesh(np.linspace(0, 1, nx), np.linspace(0, 2, ny))
        self.assertEqual(msh.n_verts(), nx * ny)
        self.assertEqual(msh.n_elems(), 2 * (nx - 1) * (ny- 1))
        self.assertEqual(msh.n_faces(), 2 * ((nx - 1)  +  (ny - 1)))

        bdy, _ = msh.boundary()
        bdy.fix()
        self.assertEqual(bdy.n_verts(), 2 * (nx + ny - 2))
        self.assertEqual(bdy.n_elems(), 2 * ((nx - 1)  +  (ny - 1)))
        self.assertEqual(bdy.n_faces(), 4)

    def test_3d(self):

        nx = 10
        ny = 15
        nz = 20
        msh = Mesh3d.box_mesh(np.linspace(0, 1, nx), np.linspace(0, 2, ny), np.linspace(0, 3, nz))
        self.assertEqual(msh.n_verts(), nx * ny * nz)
        self.assertEqual(msh.n_elems(), 6 * (nx - 1) * (ny- 1) * (nz - 1))
        self.assertEqual(msh.n_faces(), 4 * ((nx - 1) *(ny - 1) + (nx - 1) *(nz - 1) + (nz - 1) *(ny - 1)))

        bdy, _ = msh.boundary()
        bdy.fix()
        self.assertEqual(bdy.n_verts(), 2 * (nx * ny + nx * (nz - 2) + (ny - 2) * (nz - 2)))
        self.assertEqual(bdy.n_elems(), msh.n_faces())
        self.assertEqual(bdy.n_faces(), 4 * ((nx - 1) + (ny - 1) + (nz - 1)))

