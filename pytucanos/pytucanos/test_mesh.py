import os
import unittest
import numpy as np

try:
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy  # type: ignore

    HAVE_VTK = True
except ImportError:
    HAVE_VTK = False

from . import Mesh2d, Mesh3d, BoundaryMesh3d, Idx
from .mesh import get_cube, get_square


class TestMeshes(unittest.TestCase):
    def test_2d(self):
        nx = 10
        ny = 15
        msh = Mesh2d.rectangle_mesh(np.linspace(0, 1, nx), np.linspace(0, 2, ny))
        self.assertEqual(msh.n_verts(), nx * ny)
        self.assertEqual(msh.n_elems(), 2 * (nx - 1) * (ny - 1))
        self.assertEqual(msh.n_faces(), 2 * ((nx - 1) + (ny - 1)))

        bdy, _ = msh.boundary()
        bdy.fix()
        self.assertEqual(bdy.n_verts(), 2 * (nx + ny - 2))
        self.assertEqual(bdy.n_elems(), 2 * ((nx - 1) + (ny - 1)))
        self.assertEqual(bdy.n_faces(), 4)

        self.assertEqual(msh.get_elems().dtype, Idx)
        self.assertEqual(msh.get_faces().dtype, Idx)

    @unittest.skipIf(not HAVE_VTK, "vtk not available")
    def test_2d_vtk(self):
        nx = 10
        ny = 15
        msh = Mesh2d.rectangle_mesh(np.linspace(0, 1, nx), np.linspace(0, 2, ny))
        msh.write_vtk("mesh_2d.vtu")

        reader = vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName("mesh_2d.vtu")
        reader.Update()

        data = reader.GetOutput()
        self.assertEqual(msh.n_verts(), data.GetNumberOfPoints())
        self.assertEqual(msh.n_elems(), data.GetNumberOfCells())

        alg = vtk.vtkCellSizeFilter()
        alg.SetInputDataObject(data)
        alg.Update()
        data = alg.GetOutput()
        cell_data = data.GetCellData()
        self.assertEqual(cell_data.GetArrayName(3), "Area")
        vol = vtk_to_numpy(cell_data.GetArray(3))
        self.assertAlmostEqual(vol.sum(), 2)

    def test_3d(self):
        nx = 10
        ny = 15
        nz = 20
        msh = Mesh3d.box_mesh(
            np.linspace(0, 1, nx), np.linspace(0, 2, ny), np.linspace(0, 3, nz)
        )
        self.assertEqual(msh.n_verts(), nx * ny * nz)
        self.assertEqual(msh.n_elems(), 6 * (nx - 1) * (ny - 1) * (nz - 1))
        self.assertEqual(
            msh.n_faces(),
            4 * ((nx - 1) * (ny - 1) + (nx - 1) * (nz - 1) + (nz - 1) * (ny - 1)),
        )

        bdy, _ = msh.boundary()
        bdy.fix()
        self.assertEqual(
            bdy.n_verts(), 2 * (nx * ny + nx * (nz - 2) + (ny - 2) * (nz - 2))
        )
        self.assertEqual(bdy.n_elems(), msh.n_faces())
        self.assertEqual(bdy.n_faces(), 4 * ((nx - 1) + (ny - 1) + (nz - 1)))

    @unittest.skipIf(not HAVE_VTK, "vtk not available")
    def test_3d_vtk(self):
        nx = 10
        ny = 15
        nz = 20
        msh = Mesh3d.box_mesh(
            np.linspace(0, 1, nx), np.linspace(0, 2, ny), np.linspace(0, 3, nz)
        )
        msh.write_vtk("mesh_3d.vtu")

        reader = vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName("mesh_3d.vtu")
        reader.Update()

        data = reader.GetOutput()
        self.assertEqual(msh.n_verts(), data.GetNumberOfPoints())
        self.assertEqual(msh.n_elems(), data.GetNumberOfCells())

        # Invalid volumes in vtk for non-convex polyhedra
        # alg = vtk.vtkCellSizeFilter()
        # alg.SetInputDataObject(data)
        # alg.Update()
        # data = alg.GetOutput()
        # cell_data = data.GetCellData()
        # self.assertEqual(cell_data.GetArrayName(4), "Volume")
        # vol = vtk_to_numpy(cell_data.GetArray(4))
        # self.assertAlmostEqual(vol.sum(), 6)

    def test_init_2d_fail(self):
        coords, elems, etags, faces, ftags = get_square()

        with self.assertRaises(ValueError):
            _msh = Mesh3d(coords, elems, etags, faces, ftags)

    def test_init_3d_fail(self):
        coords, elems, etags, faces, ftags = get_cube()

        with self.assertRaises(ValueError):
            _msh = BoundaryMesh3d(coords, elems, etags, faces, ftags)

    def test_init_2d(self):
        coords, elems, etags, faces, ftags = get_square()
        msh = Mesh2d(coords, elems, etags, faces, ftags)

        self.assertEqual(msh.n_verts(), coords.shape[0])
        self.assertEqual(msh.n_elems(), elems.shape[0])
        self.assertEqual(msh.n_faces(), faces.shape[0])
        self.assertTrue(np.allclose(msh.get_verts(), coords))
        self.assertTrue(np.allclose(msh.get_elems(), elems))
        self.assertTrue(np.allclose(msh.get_etags(), etags))
        self.assertTrue(np.allclose(msh.get_faces(), faces))
        self.assertTrue(np.allclose(msh.get_ftags(), ftags))

    def test_meshb_2d(self):
        coords, elems, etags, faces, ftags = get_square()
        msh = Mesh2d(coords, elems, etags, faces, ftags)

        msh.write_meshb("tmp.meshb")

        msh2 = Mesh2d.from_meshb("tmp.meshb")

        self.assertEqual(msh.n_verts(), msh2.n_verts())
        self.assertTrue(np.allclose(msh.get_verts(), msh2.get_verts()))

        self.assertEqual(msh.n_elems(), msh2.n_elems())
        self.assertTrue(np.allclose(msh.get_elems(), msh2.get_elems()))
        self.assertTrue(np.allclose(msh.get_etags(), msh2.get_etags()))

        self.assertEqual(msh.n_faces(), msh2.n_faces())
        self.assertTrue(np.allclose(msh.get_faces(), msh2.get_faces()))
        self.assertTrue(np.allclose(msh.get_ftags(), msh2.get_ftags()))

        os.remove("tmp.meshb")

    def test_init_3d(self):
        coords, elems, etags, faces, ftags = get_cube()
        msh = Mesh3d(coords, elems, etags, faces, ftags)

        self.assertEqual(msh.n_verts(), coords.shape[0])
        self.assertEqual(msh.n_elems(), elems.shape[0])
        self.assertEqual(msh.n_faces(), faces.shape[0])
        self.assertTrue(np.allclose(msh.get_verts(), coords))
        self.assertTrue(np.allclose(msh.get_elems(), elems))
        self.assertTrue(np.allclose(msh.get_etags(), etags))
        self.assertTrue(np.allclose(msh.get_faces(), faces))
        self.assertTrue(np.allclose(msh.get_ftags(), ftags))

    def test_split_2d(self):
        coords, elems, etags, faces, ftags = get_square()
        msh = Mesh2d(coords, elems, etags, faces, ftags)
        msh = msh.split().split()

        self.assertEqual(msh.n_elems(), 4**2 * elems.shape[0])
        self.assertEqual(msh.n_faces(), 2**2 * faces.shape[0])

    def test_split_3d(self):
        coords, elems, etags, faces, ftags = get_cube()
        msh = Mesh3d(coords, elems, etags, faces, ftags)
        msh = msh.split().split()

        self.assertEqual(msh.n_elems(), 8**2 * elems.shape[0])
        self.assertEqual(msh.n_faces(), 4**2 * faces.shape[0])

    def test_boundary_2d(self):
        coords, elems, etags, faces, ftags = get_square()
        msh = Mesh2d(coords, elems, etags, faces, ftags)
        msh = msh.split().split()

        bdy, _ = msh.boundary()

        self.assertEqual(bdy.n_elems(), 2**2 * faces.shape[0])
        self.assertEqual(bdy.n_faces(), 0)

    def test_boundary_3d(self):
        coords, elems, etags, faces, ftags = get_cube()
        msh = Mesh3d(coords, elems, etags, faces, ftags)
        msh = msh.split().split()

        bdy, _ = msh.boundary()

        self.assertEqual(bdy.n_elems(), 4**2 * faces.shape[0])
        self.assertEqual(bdy.n_faces(), 0)

    def test_hilbert_2d(self):
        coords, elems, etags, faces, ftags = get_square()
        msh = Mesh2d(coords, elems, etags, faces, ftags)
        msh = msh.split().split().split().split().split()

        elems = msh.get_elems()
        mini = elems.min(axis=1)
        maxi = elems.max(axis=1)
        before = (maxi - mini).mean()

        msh, _, _, _ = msh.reorder_hilbert()

        elems = msh.get_elems()
        mini = elems.min(axis=1)
        maxi = elems.max(axis=1)
        after = (maxi - mini).mean()

        self.assertTrue(after < 0.11 * before)

    def test_hilbert_3d(self):
        coords, elems, etags, faces, ftags = get_cube()
        msh = Mesh3d(coords, elems, etags, faces, ftags)
        msh = msh.split().split().split()

        elems = msh.get_elems()
        mini = elems.min(axis=1)
        maxi = elems.max(axis=1)
        before = (maxi - mini).mean()

        msh, _, _, _ = msh.reorder_hilbert()

        elems = msh.get_elems()
        mini = elems.min(axis=1)
        maxi = elems.max(axis=1)
        after = (maxi - mini).mean()

        self.assertTrue(after < 0.5 * before)

    def test_boundary_faces_2d(self):
        coords, elems, etags, faces, ftags = get_square()
        faces = faces[:-1, :]
        ftags = ftags[:-1]

        msh = Mesh2d(coords, elems, etags, faces, ftags)
        msh = msh.split().split().split().split().split()

        faces_before, ftags_before = msh.get_faces(), msh.get_ftags()
        mask = ftags_before < 5
        faces_before = faces_before[mask, :]
        ftags_before = ftags_before[mask]
        idx = np.lexsort(faces_before.T)
        faces_before = faces_before[idx, :]
        ftags_before = ftags_before[idx]

        (bdy, ifc) = msh.fix()
        msh.check()
        self.assertEqual(len(bdy), 0)
        self.assertEqual(len(ifc), 1)

        faces_after, ftags_after = msh.get_faces(), msh.get_ftags()
        mask = ftags_after < 5
        faces_after = faces_after[mask, :]
        ftags_after = ftags_after[mask]
        idx = np.lexsort(faces_after.T)
        faces_after = faces_after[idx, :]
        ftags_after = ftags_after[idx]

        self.assertTrue(np.array_equal(faces_before, faces_after))
        self.assertTrue(np.array_equal(ftags_before, ftags_after))

    def test_boundary_faces_2d_2(self):
        coords, elems, etags, faces, ftags = get_square()
        faces = faces[:-1, :]
        ftags = ftags[:-1]
        faces[0, :] = faces[0, ::-1]
        msh = Mesh2d(coords, elems, etags, faces, ftags)
        msh = msh.split().split().split().split().split()

        faces_before, ftags_before = msh.get_faces(), msh.get_ftags()
        idx = np.lexsort(faces_before.T)
        faces_before = faces_before[idx, :]
        ftags_before = ftags_before[idx]

        (bdy, ifc) = msh.fix()
        msh.check()
        self.assertEqual(len(bdy), 0)
        self.assertEqual(len(ifc), 1)

        faces_after, ftags_after = msh.get_faces(), msh.get_ftags()
        idx = np.lexsort(faces_after.T)
        faces_after = faces_after[idx, :]
        ftags_after = ftags_after[idx]

        self.assertFalse(np.array_equal(faces_before, faces_after))

    def test_boundary_faces_2d_3(self):
        coords, elems, etags, faces, ftags = get_square()
        faces = faces[:2, :]
        ftags = ftags[:2]

        msh = Mesh2d(coords, elems, etags, faces, ftags)
        msh = msh.split().split().split().split().split()

        (bdy, ifc) = msh.fix()
        msh.check()

        self.assertEqual(len(bdy), 1)
        self.assertEqual(len(ifc), 1)
        tag = bdy[2]
        self.assertEqual((msh.get_ftags() == tag).sum(), 2 * 2**5)

        _faces_after, ftags_after = msh.get_faces(), msh.get_ftags()
        self.assertTrue(np.array_equal(np.unique(ftags_after), [1, 2, 3, 4]))
        self.assertEqual(np.nonzero(ftags_after == 3)[0].size, 2 * 2**5)

    def test_boundary_faces_3d(self):
        coords, elems, etags, faces, ftags = get_cube()
        msh = Mesh3d(coords, elems, etags, faces, ftags)
        msh = msh.split().split().split()

        faces_before, ftags_before = msh.get_faces(), msh.get_ftags()
        idx = np.lexsort(faces_before.T)
        faces_before = faces_before[idx, :]
        ftags_before = ftags_before[idx]

        (bdy, ifc) = msh.fix()
        self.assertEqual(len(bdy), 0)
        self.assertEqual(len(ifc), 0)

    def test_boundary_faces_3d_2(self):
        coords, elems, etags, faces, ftags = get_cube()
        faces[0, :] = faces[0, ::-1]
        msh = Mesh3d(coords, elems, etags, faces, ftags)
        msh = msh.split().split().split()

        faces_before, ftags_before = msh.get_faces(), msh.get_ftags()
        idx = np.lexsort(faces_before.T)
        faces_before = faces_before[idx, :]
        ftags_before = ftags_before[idx]

        (bdy, ifc) = msh.fix()
        msh.check()

        self.assertEqual(len(bdy), 0)
        self.assertEqual(len(ifc), 0)

    def test_boundary_faces_3d_3(self):
        coords, elems, etags, faces, ftags = get_cube()
        faces = faces[:2, :]
        ftags = ftags[:2]
        msh = Mesh3d(coords, elems, etags, faces, ftags)
        msh = msh.split().split().split()

        (bdy, ifc) = msh.fix()
        msh.check()

        self.assertEqual(len(bdy), 1)
        self.assertEqual(len(ifc), 0)
        tag = bdy[1]
        self.assertEqual((msh.get_ftags() == tag).sum(), 10 * 4**3)

    def test_fields_2d(self):
        coords, elems, etags, faces, ftags = get_square()
        msh = Mesh2d(coords, elems, etags, faces, ftags)
        msh = msh.split().split().split()

        x, y = msh.get_verts().T
        f = (x + y).reshape((-1, 1))

        elems = msh.get_elems()
        f_e = f[elems].mean(axis=1)

        with self.assertRaises(ValueError):
            msh.vertex_data_to_elem_data(f_e)

        msh.vertex_data_to_elem_data(f)

        with self.assertRaises(ValueError):
            msh.elem_data_to_vertex_data(f)

        msh.elem_data_to_vertex_data(f_e)

    def test_meshb_3d(self):
        coords, elems, etags, faces, ftags = get_cube()
        msh = Mesh3d(coords, elems, etags, faces, ftags)

        msh.write_meshb("tmp.meshb")

        msh2 = Mesh3d.from_meshb("tmp.meshb")

        self.assertEqual(msh.n_verts(), msh2.n_verts())
        self.assertTrue(np.allclose(msh.get_verts(), msh2.get_verts()))

        self.assertEqual(msh.n_elems(), msh2.n_elems())
        self.assertTrue(np.allclose(msh.get_elems(), msh2.get_elems()))
        self.assertTrue(np.allclose(msh.get_etags(), msh2.get_etags()))

        self.assertEqual(msh.n_faces(), msh2.n_faces())
        self.assertTrue(np.allclose(msh.get_faces(), msh2.get_faces()))
        self.assertTrue(np.allclose(msh.get_ftags(), msh2.get_ftags()))

        os.remove("tmp.meshb")
