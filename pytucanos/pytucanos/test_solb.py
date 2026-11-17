import os
import unittest

import numpy as np

from . import Mesh2d, Mesh3d
from .mesh import get_cube, get_square


class TestField(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import logging

        logging.disable(logging.CRITICAL)

    def test_2d_scalar(self):
        coords, elems, etags, faces, ftags = get_square()
        msh = Mesh2d(coords, elems, etags, faces, ftags)
        f = np.random.rand(msh.n_verts(), 1)
        msh.write_solb("tmp.solb", f)
        g = Mesh2d.read_solb("tmp.solb")
        self.assertTrue(np.allclose(f, g))

        os.remove("tmp.solb")

    def test_2d_fail(self):
        coords, elems, etags, faces, ftags = get_square()
        msh = Mesh2d(coords, elems, etags, faces, ftags)

        # Fortran-ordered arrays are rejected
        f = np.asfortranarray(np.random.rand(msh.n_verts(), 2))
        with self.assertRaises(ValueError):
            msh.write_solb("tmp.solb", f)

        # unsupported number of components
        f = np.random.rand(msh.n_verts(), 4)
        with self.assertRaises(RuntimeError):
            msh.write_solb("tmp.solb", f)

        # dimension mismatch
        coords, elems, etags, faces, ftags = get_cube()
        msh = Mesh3d(coords, elems, etags, faces, ftags)
        msh.write_solb("tmp.solb", np.random.rand(msh.n_verts(), 1))
        with self.assertRaises(RuntimeError):
            Mesh2d.read_solb("tmp.solb")

        os.remove("tmp.solb")

    def test_2d_vector(self):
        coords, elems, etags, faces, ftags = get_square()
        msh = Mesh2d(coords, elems, etags, faces, ftags)
        f = np.random.rand(msh.n_verts(), 2)
        msh.write_solb("tmp.solb", f)
        g = Mesh2d.read_solb("tmp.solb")
        self.assertTrue(np.allclose(f, g))

        os.remove("tmp.solb")

    def test_2d_tensor(self):
        coords, elems, etags, faces, ftags = get_square()
        msh = Mesh2d(coords, elems, etags, faces, ftags)
        f = np.random.rand(msh.n_verts(), 3)
        msh.write_solb("tmp.solb", f)
        g = Mesh2d.read_solb("tmp.solb")
        self.assertTrue(np.allclose(f, g))

        os.remove("tmp.solb")

    def test_3d_scalar(self):
        coords, elems, etags, faces, ftags = get_cube()
        msh = Mesh3d(coords, elems, etags, faces, ftags)
        f = np.random.rand(msh.n_verts(), 1)
        msh.write_solb("tmp.solb", f)
        g = Mesh3d.read_solb("tmp.solb")
        self.assertTrue(np.allclose(f, g))

        os.remove("tmp.solb")

    def test_3d_vector(self):
        coords, elems, etags, faces, ftags = get_cube()
        msh = Mesh3d(coords, elems, etags, faces, ftags)
        f = np.random.rand(msh.n_verts(), 3)
        msh.write_solb("tmp.solb", f)
        g = Mesh3d.read_solb("tmp.solb")
        self.assertTrue(np.allclose(f, g))

        os.remove("tmp.solb")

    def test_3d_tensor(self):
        coords, elems, etags, faces, ftags = get_cube()
        msh = Mesh3d(coords, elems, etags, faces, ftags)
        f = np.random.rand(msh.n_verts(), 6)
        msh.write_solb("tmp.solb", f)
        g = Mesh3d.read_solb("tmp.solb")
        self.assertTrue(np.allclose(f, g))

        os.remove("tmp.solb")
