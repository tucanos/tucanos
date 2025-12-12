import numpy as np
import unittest
from . import (
    Mesh2d,
    Mesh3d,
    BoundaryMesh2d,
    LinearGeometry2d,
    LinearGeometry3d,
    Idx,
    QuadraticBoundaryMesh3d,
    QuadraticGeometry3d,
)
from .mesh import get_cube, get_square


class TestGeometry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import logging

        logging.disable(logging.CRITICAL)

    def test_init_2d(self):
        coords, elems, etags, faces, ftags = get_square()
        msh = Mesh2d(coords, elems, etags, faces, ftags)
        msh.fix()
        bdy, _ = msh.boundary()
        _geom = LinearGeometry2d(bdy)

    def test_init_3d(self):
        coords, elems, etags, faces, ftags = get_cube()
        msh = Mesh3d(coords, elems, etags, faces, ftags)
        msh.fix()
        bdy, _ = msh.boundary()
        bdy.fix()
        geom = LinearGeometry3d(bdy)
        geom.set_topo_map(msh)

    def test_curvature_3d(self):
        coords, elems, etags, faces, ftags = get_cube()
        msh = Mesh3d(coords, elems, etags, faces, ftags)
        msh.fix()
        bdy, _ = msh.boundary()
        bdy.fix()
        geom = LinearGeometry3d(bdy)
        geom.set_topo_map(msh)

    def test_project_2d(self):
        coords, elems, etags, faces, ftags = get_square()
        msh = Mesh2d(coords, elems, etags, faces, ftags).split()
        msh.fix()

        m = 20
        n = 4 * m
        theta = 2.0 * np.pi * np.linspace(0, 1, n + 1) - 3.0 * np.pi / 4.0
        edgs = np.stack([np.arange(0, n), np.arange(1, n + 1)], axis=-1).astype(Idx)
        edgs[-1, -1] = 0
        etags = np.ones(n, dtype=np.int16)
        etags[:m] = 1
        etags[m : 2 * m] = 2
        etags[2 * m : 3 * m] = 3
        etags[3 * m : 4 * m] = 4

        edgs = np.vstack(
            [
                edgs,
                [
                    [0, 2 * m],
                ],
            ],
        ).astype(Idx)
        etags = np.append(etags, 5).astype(np.int16)

        x = 0.5 + 0.5**0.5 * np.cos(theta)
        y = 0.5 + 0.5**0.5 * np.sin(theta)
        coords = np.stack([x, y], axis=-1)
        geom = BoundaryMesh2d(
            coords,
            edgs,
            etags,
            np.zeros([0, 1], dtype=Idx),
            np.zeros(0, dtype=np.int16),
        )

        msh.write_vtk("msh.vtu")
        geom.write_vtk("geom.vtu")

        geom = LinearGeometry2d(geom)

        new_coords = geom.project(msh)
        new_coords -= 0.5
        r = np.linalg.norm(new_coords, axis=1)
        flg = np.logical_and(r > 0.5**0.5 - 0.01, r < 0.5**0.5 + 0.01)
        (ok,) = np.nonzero(~flg)
        self.assertEqual(ok.size, 1)

    def test_quadratic_3d(self):
        msh = Mesh3d.ball_mesh(1.0, 3)
        n = msh.n_verts()
        msh = msh.split()
        bdy = QuadraticBoundaryMesh3d.sphere_mesh(1.0, 3)

        bdy.fix()
        geom = QuadraticGeometry3d(bdy)
        geom.set_topo_map(msh)

        coords = msh.get_verts()
        new_coords = geom.project(msh)
        d = np.linalg.norm(coords - new_coords, axis=1)
        self.assertLess(d[:n].max(), 1e-12)
        self.assertLess(d[n:].max(), 1.5e-2)
