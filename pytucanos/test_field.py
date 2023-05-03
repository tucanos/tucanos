import os
import numpy as np
import unittest
from .field import read_solb, write_solb
from . import HAVE_MESHB


@unittest.skipUnless(HAVE_MESHB, "The libMeshb interface is not available")
class TestField(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import logging

        logging.disable(logging.CRITICAL)

    def test_2d_scalar(self):

        f = np.random.rand(10, 1)

        write_solb("tmp.solb", 2, f)
        g = read_solb("tmp.solb")
        self.assertTrue(np.allclose(f, g))

        os.remove("tmp.solb")

    def test_2d_vector(self):

        f = np.random.rand(10, 2)

        write_solb("tmp.solb", 2, f)
        g = read_solb("tmp.solb")
        self.assertTrue(np.allclose(f, g))

        os.remove("tmp.solb")

    def test_2d_tensor(self):

        f = np.random.rand(10, 3)

        write_solb("tmp.solb", 2, f)
        g = read_solb("tmp.solb")
        self.assertTrue(np.allclose(f, g))

        os.remove("tmp.solb")

    def test_3d_scalar(self):

        f = np.random.rand(10, 1)

        write_solb("tmp.solb", 3, f)
        g = read_solb("tmp.solb")
        self.assertTrue(np.allclose(f, g))

        os.remove("tmp.solb")

    def test_3d_vector(self):

        f = np.random.rand(10, 3)

        write_solb("tmp.solb", 3, f)
        g = read_solb("tmp.solb")
        self.assertTrue(np.allclose(f, g))

        os.remove("tmp.solb")

    def test_3d_tensor(self):

        f = np.random.rand(10, 6)

        write_solb("tmp.solb", 3, f)
        g = read_solb("tmp.solb")
        self.assertTrue(np.allclose(f, g))

        os.remove("tmp.solb")
