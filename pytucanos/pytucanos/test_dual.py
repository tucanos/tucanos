import unittest
import numpy as np

try:
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy  # type: ignore

    HAVE_VTK = True
except ImportError:
    HAVE_VTK = False

from . import Mesh2d, Mesh3d, DualType, DualMesh2d, DualMesh3d, USE_32BIT_INTS, Idx


class TestMeshes(unittest.TestCase):
    def test_2d_median(self):
        nx = 10
        ny = 15
        msh = Mesh2d.rectangle_mesh(np.linspace(0, 1, nx), np.linspace(0, 2, ny))
        self.assertEqual(msh.n_verts(), nx * ny)
        self.assertEqual(msh.n_elems(), 2 * (nx - 1) * (ny - 1))
        self.assertEqual(msh.n_faces(), 2 * ((nx - 1) + (ny - 1)))

        dual = DualMesh2d(msh, DualType.Median)
        self.assertEqual(dual.n_elems(), msh.n_verts())

        dual_bdy, _ = dual.boundary()
        dual_bdy.fix()
        self.assertEqual(
            dual_bdy.n_verts(), 2 * (nx + ny - 2) + 2 * ((nx - 1) + (ny - 1))
        )
        self.assertEqual(dual_bdy.n_elems(), 4 * ((nx - 1) + (ny - 1)))
        self.assertEqual(dual_bdy.n_faces(), 4)

    def test_2d_barth(self):
        nx = 10
        ny = 15
        msh = Mesh2d.rectangle_mesh(np.linspace(0, 1, nx), np.linspace(0, 2, ny))
        self.assertEqual(msh.n_verts(), nx * ny)
        self.assertEqual(msh.n_elems(), 2 * (nx - 1) * (ny - 1))
        self.assertEqual(msh.n_faces(), 2 * ((nx - 1) + (ny - 1)))

        dual = DualMesh2d(msh, DualType.Barth)
        self.assertEqual(dual.n_elems(), msh.n_verts())

        dual_bdy, _ = dual.boundary()
        dual_bdy.fix()
        self.assertEqual(
            dual_bdy.n_verts(), 2 * (nx + ny - 2) + 2 * ((nx - 1) + (ny - 1))
        )
        self.assertEqual(dual_bdy.n_elems(), 4 * ((nx - 1) + (ny - 1)))
        self.assertEqual(dual_bdy.n_faces(), 4)

        if USE_32BIT_INTS:
            self.assertEqual(dual.get_elems()[0].dtype, np.uint32)
            self.assertEqual(dual.get_faces()[0].dtype, np.uint32)
        else:
            self.assertEqual(dual.get_elems()[0].dtype, Idx)
            self.assertEqual(dual.get_faces()[0].dtype, Idx)

    @unittest.skipIf(not HAVE_VTK, "vtk not available")
    def test_2d_vtk(self):
        nx = 10
        ny = 15
        msh = Mesh2d.rectangle_mesh(np.linspace(0, 1, nx), np.linspace(0, 2, ny))
        dual = DualMesh2d(msh, DualType.Barth)

        dual.write_vtk("dual_2d.vtu")

        reader = vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName("dual_2d.vtu")
        reader.Update()

        data = reader.GetOutput()
        self.assertEqual(dual.n_verts(), data.GetNumberOfPoints())
        self.assertEqual(dual.n_elems(), data.GetNumberOfCells())

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

        dual = DualMesh3d(msh, DualType.Median)
        self.assertEqual(dual.n_elems(), msh.n_verts())

        bdy, _ = dual.boundary()
        bdy.fix()
        self.assertEqual(bdy.n_elems(), 6 * msh.n_faces())
        self.assertEqual(bdy.n_faces(), 2 * 4 * ((nx - 1) + (ny - 1) + (nz - 1)))

    @unittest.skipIf(not HAVE_VTK, "vtk not available")
    def test_3d_vtk(self):
        nx = 5
        ny = 6
        nz = 7
        msh = Mesh3d.box_mesh(
            np.linspace(0, 1, nx), np.linspace(0, 2, ny), np.linspace(0, 3, nz)
        )
        dual = DualMesh3d(msh, DualType.Barth)

        dual.write_vtk("dual_3d.vtu")

        reader = vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName("dual_3d.vtu")
        reader.Update()

        data = reader.GetOutput()
        self.assertEqual(dual.n_verts(), data.GetNumberOfPoints())
        self.assertEqual(dual.n_elems(), data.GetNumberOfCells())

        # Warning: only works for convex cells -> ok with Barth
        alg = vtk.vtkCellSizeFilter()
        alg.SetInputDataObject(data)
        alg.Update()
        data = alg.GetOutput()
        cell_data = data.GetCellData()
        self.assertEqual(cell_data.GetArrayName(4), "Volume")
        vol = vtk_to_numpy(cell_data.GetArray(4))
        self.assertAlmostEqual(vol.sum(), 6)
