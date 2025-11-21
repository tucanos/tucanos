import numpy as np
import unittest
from .mesh import get_square, get_cube
from . import Mesh2d, Mesh3d, autotag, transfer_tags


class TestMeshes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import logging

        logging.disable(logging.CRITICAL)

    def test_autotag_2d(self):
        coords, elems, etags, faces, ftags = get_square(two_tags=False)
        ftags[:] = 1

        msh = Mesh2d(coords, elems, etags, faces, ftags)
        msh.fix()
        msh = msh.split().split().split()

        bdy, _ = msh.boundary()

        autotag(bdy, 30.0)

        tags = bdy.get_etags()
        vals, counts = np.unique(tags, return_counts=True)
        vals.sort()
        self.assertTrue(np.array_equal(vals, [1, 2, 3, 4]))
        self.assertTrue(np.array_equal(counts, [8, 8, 8, 8]))

        transfer_tags(bdy, msh)

        tags = msh.get_ftags()
        vals, counts = np.unique(tags, return_counts=True)
        vals.sort()
        self.assertTrue(np.array_equal(vals, [1, 2, 3, 4]))
        self.assertTrue(np.array_equal(counts, [8, 8, 8, 8]))

    def test_autotag_3d(self):
        coords, elems, etags, faces, ftags = get_cube()
        ftags[:] = 1

        msh = Mesh3d(coords, elems, etags, faces, ftags)
        msh.fix()
        msh = msh.split()

        bdy, _ = msh.boundary()
        bdy = bdy.split().split()

        autotag(bdy, 30.0)

        tags = bdy.get_etags()
        vals, counts = np.unique(tags, return_counts=True)
        vals.sort()
        self.assertTrue(np.array_equal(vals, [1, 2, 3, 4, 5, 6]))
        self.assertTrue(np.array_equal(counts, [128, 128, 128, 128, 128, 128]))

        transfer_tags(bdy, msh)

        tags = msh.get_ftags()
        vals, counts = np.unique(tags, return_counts=True)
        vals.sort()
        self.assertTrue(np.array_equal(vals, [1, 2, 3, 4, 5, 6]))
        self.assertTrue(np.array_equal(counts, [8, 8, 8, 8, 8, 8]))
