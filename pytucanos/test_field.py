import os
import numpy as np
import unittest
from .mesh import Mesh22, get_square, Mesh33, get_cube
from . import HAVE_MESHB
if HAVE_MESHB:
    from .field import read_solb


@unittest.skipUnless(HAVE_MESHB, "The libMeshb interface is not available")
class TestField(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import logging

        logging.disable(logging.CRITICAL)

    def test_2d_scalar(self):

        coords, elems, etags, faces, ftags = get_square()
        msh = Mesh22(coords, elems, etags, faces, ftags)
        f = np.random.rand(msh.n_verts(), 1)
        msh.write_solb("tmp.solb", f)
        g = read_solb("tmp.solb")
        self.assertTrue(np.allclose(f, g))

        os.remove("tmp.solb")

    def test_2d_vector(self):

        coords, elems, etags, faces, ftags = get_square()
        msh = Mesh22(coords, elems, etags, faces, ftags)
        f = np.random.rand(msh.n_verts(), 2)
        msh.write_solb("tmp.solb", f)
        g = read_solb("tmp.solb")
        self.assertTrue(np.allclose(f, g))

        os.remove("tmp.solb")

    def test_2d_tensor(self):

        coords, elems, etags, faces, ftags = get_square()
        msh = Mesh22(coords, elems, etags, faces, ftags)
        f = np.random.rand(msh.n_verts(), 3)
        msh.write_solb("tmp.solb", f)
        g = read_solb("tmp.solb")
        self.assertTrue(np.allclose(f, g))

        os.remove("tmp.solb")

    def test_3d_scalar(self):

        coords, elems, etags, faces, ftags = get_cube()
        msh = Mesh33(coords, elems, etags, faces, ftags)
        f = np.random.rand(msh.n_verts(), 1)
        msh.write_solb("tmp.solb", f)
        g = read_solb("tmp.solb")
        self.assertTrue(np.allclose(f, g))

        os.remove("tmp.solb")

    def test_3d_vector(self):

        coords, elems, etags, faces, ftags = get_cube()
        msh = Mesh33(coords, elems, etags, faces, ftags)
        f = np.random.rand(msh.n_verts(), 3)
        msh.write_solb("tmp.solb", f)
        g = read_solb("tmp.solb")
        self.assertTrue(np.allclose(f, g))

        os.remove("tmp.solb")

    def test_3d_tensor(self):

        coords, elems, etags, faces, ftags = get_cube()
        msh = Mesh33(coords, elems, etags, faces, ftags)
        f = np.random.rand(msh.n_verts(), 6)
        msh.write_solb("tmp.solb", f)
        g = read_solb("tmp.solb")
        self.assertTrue(np.allclose(f, g))

        os.remove("tmp.solb")
