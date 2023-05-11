import os
import numpy as np
import unittest
from .mesh import Mesh22, get_square, Mesh33, get_cube
from .geometry import LinearGeometry2d, LinearGeometry3d


class TestGeometry3d(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import logging

        logging.disable(logging.CRITICAL)

    def test_init_2d(self):

        coords, elems, etags, faces, ftags = get_square()
        msh = Mesh22(coords, elems, etags, faces, ftags)
        geom = LinearGeometry2d(msh)

    def test_init_3d(self):

        coords, elems, etags, faces, ftags = get_cube()
        msh = Mesh33(coords, elems, etags, faces, ftags)
        geom = LinearGeometry3d(msh)

    def test_curvature_3d(self):

        coords, elems, etags, faces, ftags = get_cube()
        msh = Mesh33(coords, elems, etags, faces, ftags)
        geom = LinearGeometry3d(msh)
        geom.compute_curvature()
