import unittest
import numpy as np
from . import Mesh2d, Mesh3d, DualType, DualMesh2d, DualMesh3d

class TestMeshes(unittest.TestCase):

    def test_2d_median(self):

        nx = 10
        ny = 15
        msh = Mesh2d.rectangle_mesh(np.linspace(0, 1, nx), np.linspace(0, 2, ny))
        self.assertEqual(msh.n_verts(), nx * ny)
        self.assertEqual(msh.n_elems(), 2 * (nx - 1) * (ny- 1))
        self.assertEqual(msh.n_faces(), 2 * ((nx - 1)  +  (ny - 1)))
        
        dual = DualMesh2d(msh, DualType.Median)
        self.assertEqual(dual.n_elems(), msh.n_verts())
        
        dual_bdy, _ = dual.boundary()
        dual_bdy.fix()
        self.assertEqual(dual_bdy.n_verts(), 2 * (nx + ny - 2) + 2 * ((nx - 1)  +  (ny - 1)))
        self.assertEqual(dual_bdy.n_elems(), 4 * ((nx - 1)  +  (ny - 1)))
        self.assertEqual(dual_bdy.n_faces(), 4)

    def test_2d_barth(self):

        nx = 10
        ny = 15
        msh = Mesh2d.rectangle_mesh(np.linspace(0, 1, nx), np.linspace(0, 2, ny))
        self.assertEqual(msh.n_verts(), nx * ny)
        self.assertEqual(msh.n_elems(), 2 * (nx - 1) * (ny- 1))
        self.assertEqual(msh.n_faces(), 2 * ((nx - 1)  +  (ny - 1)))
        
        dual = DualMesh2d(msh, DualType.Barth)
        self.assertEqual(dual.n_elems(), msh.n_verts())
        
        dual_bdy, _ = dual.boundary()
        dual_bdy.fix()
        self.assertEqual(dual_bdy.n_verts(), 2 * (nx + ny - 2) + 2 * ((nx - 1)  +  (ny - 1)))
        self.assertEqual(dual_bdy.n_elems(), 4 * ((nx - 1)  +  (ny - 1)))
        self.assertEqual(dual_bdy.n_faces(), 4)

    def test_3d(self):

        nx = 10
        ny = 15
        nz = 20
        msh = Mesh3d.box_mesh(np.linspace(0, 1, nx), np.linspace(0, 2, ny), np.linspace(0, 3, nz))
        self.assertEqual(msh.n_verts(), nx * ny * nz)
        self.assertEqual(msh.n_elems(), 6 * (nx - 1) * (ny- 1) * (nz - 1))
        self.assertEqual(msh.n_faces(), 4 * ((nx - 1) *(ny - 1) + (nx - 1) *(nz - 1) + (nz - 1) *(ny - 1)))

        dual = DualMesh3d(msh, DualType.Median)
        self.assertEqual(dual.n_elems(), msh.n_verts())
        
        bdy, _ = dual.boundary()
        bdy.fix()
        self.assertEqual(bdy.n_elems(), 6 * msh.n_faces())
        self.assertEqual(bdy.n_faces(), 2*4 * ((nx - 1) + (ny - 1) + (nz - 1)))

