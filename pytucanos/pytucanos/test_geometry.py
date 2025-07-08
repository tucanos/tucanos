import numpy as np
import unittest
from .mesh import Mesh22, get_square, Mesh33, get_cube, Mesh21
from .geometry import LinearGeometry2d, LinearGeometry3d


class TestGeometry3d(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import logging

        logging.disable(logging.CRITICAL)

    def test_init_2d(self):
        coords, elems, etags, faces, ftags = get_square()
        msh = Mesh22(coords, elems, etags, faces, ftags)
        msh.compute_topology()
        _geom = LinearGeometry2d(msh)

    def test_init_3d(self):
        coords, elems, etags, faces, ftags = get_cube()
        msh = Mesh33(coords, elems, etags, faces, ftags)
        msh.compute_topology()
        _geom = LinearGeometry3d(msh)

    def test_curvature_3d(self):
        coords, elems, etags, faces, ftags = get_cube()
        msh = Mesh33(coords, elems, etags, faces, ftags)
        msh.compute_topology()
        geom = LinearGeometry3d(msh)
        geom.compute_curvature()

    def test_project_2d(self):
        coords, elems, etags, faces, ftags = get_square()
        msh = Mesh22(coords, elems, etags, faces, ftags).split()
        msh.compute_topology()

        m = 20
        n = 4 * m
        theta = 2.0 * np.pi * np.linspace(0, 1, n + 1) - 3.0 * np.pi / 4.0
        edgs = np.stack([np.arange(0, n), np.arange(1, n + 1)], axis=-1).astype(
            np.uint32
        )
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
        ).astype(np.uint32)
        etags = np.append(etags, 5).astype(np.int16)

        x = 0.5 + 0.5**0.5 * np.cos(theta)
        y = 0.5 + 0.5**0.5 * np.sin(theta)
        coords = np.stack([x, y], axis=-1)
        geom = Mesh21(
            coords,
            edgs,
            etags,
            np.zeros([0, 1], dtype=np.uint32),
            np.zeros(0, dtype=np.int16),
        )

        msh.write_vtk("msh.vtu")
        geom.write_vtk("geom.vtu")

        geom = LinearGeometry2d(msh, geom)

        new_coords = geom.project(msh)
        new_coords -= 0.5
        r = np.linalg.norm(new_coords, axis=1)
        flg = np.logical_and(r > 0.5**0.5 - 0.01, r < 0.5**0.5 + 0.01)
        (ok,) = np.nonzero(~flg)
        self.assertEqual(ok.size, 1)
