import unittest
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy  # type: ignore
from . import Mesh2d, Mesh3d


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
