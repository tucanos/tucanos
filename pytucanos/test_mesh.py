import os
import numpy as np
import unittest
from .mesh import (
    Mesh22,
    Mesh32,
    Mesh33,
    get_square,
    get_cube,
)
from . import HAVE_MESHB


class TestMeshes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import logging

        logging.disable(logging.CRITICAL)

    def test_init_2d_fail(self):
        coords, elems, etags, faces, ftags = get_square()

        with self.assertRaises(ValueError):
            msh = Mesh33(coords, elems, etags, faces, ftags)

    def test_init_3d_fail(self):
        coords, elems, etags, faces, ftags = get_cube()

        with self.assertRaises(ValueError):
            msh = Mesh32(coords, elems, etags, faces, ftags)

    def test_init_2d(self):
        coords, elems, etags, faces, ftags = get_square()
        msh = Mesh22(coords, elems, etags, faces, ftags)

        self.assertEqual(msh.n_verts(), coords.shape[0])
        self.assertEqual(msh.n_elems(), elems.shape[0])
        self.assertEqual(msh.n_faces(), faces.shape[0])
        self.assertTrue(np.allclose(msh.vol(), 1.0))
        self.assertTrue(np.allclose(msh.get_coords(), coords))
        self.assertTrue(np.allclose(msh.get_elems(), elems))
        self.assertTrue(np.allclose(msh.get_etags(), etags))
        self.assertTrue(np.allclose(msh.get_faces(), faces))
        self.assertTrue(np.allclose(msh.get_ftags(), ftags))

    @unittest.skipUnless(HAVE_MESHB, "The libMeshb interface is not available")
    def test_meshb_2d(self):
        coords, elems, etags, faces, ftags = get_square()
        msh = Mesh22(coords, elems, etags, faces, ftags)

        msh.write_meshb("tmp.meshb")

        msh2 = Mesh22.from_meshb("tmp.meshb")

        self.assertEqual(msh.n_verts(), msh2.n_verts())
        self.assertTrue(np.allclose(msh.get_coords(), msh2.get_coords()))

        self.assertEqual(msh.n_elems(), msh2.n_elems())
        self.assertTrue(np.allclose(msh.get_elems(), msh2.get_elems()))
        self.assertTrue(np.allclose(msh.get_etags(), msh2.get_etags()))

        self.assertEqual(msh.n_faces(), msh2.n_faces())
        self.assertTrue(np.allclose(msh.get_faces(), msh2.get_faces()))
        self.assertTrue(np.allclose(msh.get_ftags(), msh2.get_ftags()))

        os.remove("tmp.meshb")

    def test_init_3d(self):
        coords, elems, etags, faces, ftags = get_cube()
        msh = Mesh33(coords, elems, etags, faces, ftags)

        self.assertEqual(msh.n_verts(), coords.shape[0])
        self.assertEqual(msh.n_elems(), elems.shape[0])
        self.assertEqual(msh.n_faces(), faces.shape[0])
        self.assertTrue(np.allclose(msh.vol(), 1.0))
        self.assertTrue(np.allclose(msh.get_coords(), coords))
        self.assertTrue(np.allclose(msh.get_elems(), elems))
        self.assertTrue(np.allclose(msh.get_etags(), etags))
        self.assertTrue(np.allclose(msh.get_faces(), faces))
        self.assertTrue(np.allclose(msh.get_ftags(), ftags))

    def test_split_2d(self):
        coords, elems, etags, faces, ftags = get_square()
        msh = Mesh22(coords, elems, etags, faces, ftags)
        msh = msh.split().split()

        self.assertEqual(msh.n_elems(), 4**2 * elems.shape[0])
        self.assertEqual(msh.n_faces(), 2**2 * faces.shape[0])
        self.assertTrue(np.allclose(msh.vol(), 1.0))

    def test_split_3d(self):
        coords, elems, etags, faces, ftags = get_cube()
        msh = Mesh33(coords, elems, etags, faces, ftags)
        msh = msh.split().split()

        self.assertEqual(msh.n_elems(), 8**2 * elems.shape[0])
        self.assertEqual(msh.n_faces(), 4**2 * faces.shape[0])
        self.assertTrue(np.allclose(msh.vol(), 1.0))

    def test_boundary_2d(self):
        coords, elems, etags, faces, ftags = get_square()
        msh = Mesh22(coords, elems, etags, faces, ftags)
        msh = msh.split().split()

        bdy, _ = msh.boundary()

        self.assertEqual(bdy.n_elems(), 2**2 * faces.shape[0])
        self.assertEqual(bdy.n_faces(), 0)
        self.assertTrue(np.allclose(bdy.vol(), 4.0 + 2**0.5))

    def test_boundary_3d(self):
        coords, elems, etags, faces, ftags = get_cube()
        msh = Mesh33(coords, elems, etags, faces, ftags)
        msh = msh.split().split()

        bdy, _ = msh.boundary()

        self.assertEqual(bdy.n_elems(), 4**2 * faces.shape[0])
        self.assertEqual(bdy.n_faces(), 0)
        self.assertTrue(np.allclose(bdy.vol(), 6.0))

    def test_hilbert_2d(self):
        coords, elems, etags, faces, ftags = get_square()
        msh = Mesh22(coords, elems, etags, faces, ftags)
        msh = msh.split().split().split().split().split()

        elems = msh.get_elems()
        mini = elems.min(axis=1)
        maxi = elems.max(axis=1)
        before = (maxi - mini).mean()

        msh.reorder_hilbert()

        elems = msh.get_elems()
        mini = elems.min(axis=1)
        maxi = elems.max(axis=1)
        after = (maxi - mini).mean()

        self.assertTrue(after < 0.11 * before)

    def test_hilbert_3d(self):
        coords, elems, etags, faces, ftags = get_cube()
        msh = Mesh33(coords, elems, etags, faces, ftags)
        msh = msh.split().split().split()

        elems = msh.get_elems()
        mini = elems.min(axis=1)
        maxi = elems.max(axis=1)
        before = (maxi - mini).mean()

        msh.reorder_hilbert()

        elems = msh.get_elems()
        mini = elems.min(axis=1)
        maxi = elems.max(axis=1)
        after = (maxi - mini).mean()

        self.assertTrue(after < 0.5 * before)

    def test_boundary_faces_2d(self):
        coords, elems, etags, faces, ftags = get_square()
        msh = Mesh22(coords, elems, etags, faces, ftags)
        msh = msh.split().split().split().split().split()

        faces_before, ftags_before = msh.get_faces(), msh.get_ftags()
        mask = ftags_before < 5
        faces_before = faces_before[mask, :]
        ftags_before = ftags_before[mask]
        idx = np.lexsort(faces_before.T)
        faces_before = faces_before[idx, :]
        ftags_before = ftags_before[idx]

        tag = msh.add_boundary_faces()
        self.assertEqual((msh.get_ftags() == tag).sum(), 0)

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
        faces[0, :] = faces[0, ::-1]
        msh = Mesh22(coords, elems, etags, faces, ftags)
        msh = msh.split().split().split().split().split()

        faces_before, ftags_before = msh.get_faces(), msh.get_ftags()
        idx = np.lexsort(faces_before.T)
        faces_before = faces_before[idx, :]
        ftags_before = ftags_before[idx]

        tag = msh.add_boundary_faces()
        self.assertEqual((msh.get_ftags() == tag).sum(), 0)

        faces_after, ftags_after = msh.get_faces(), msh.get_ftags()
        idx = np.lexsort(faces_after.T)
        faces_after = faces_after[idx, :]
        ftags_after = ftags_after[idx]

        self.assertFalse(np.array_equal(faces_before, faces_after))

    def test_boundary_faces_2d_3(self):
        coords, elems, etags, faces, ftags = get_square()
        faces = faces[:2, :]
        ftags = ftags[:2]

        msh = Mesh22(coords, elems, etags, faces, ftags)
        msh = msh.split().split().split().split().split()

        tag = msh.add_boundary_faces()
        self.assertEqual((msh.get_ftags() == tag).sum(), 2 * 2**5)

        faces_after, ftags_after = msh.get_faces(), msh.get_ftags()
        self.assertTrue(np.array_equal(np.unique(ftags_after), [1, 2, 3, 4]))
        self.assertEqual(np.nonzero(ftags_after == 3)[0].size, 2 * 2**5)

    def test_boundary_faces_3d(self):
        coords, elems, etags, faces, ftags = get_cube()
        msh = Mesh33(coords, elems, etags, faces, ftags)
        msh = msh.split().split().split()

        faces_before, ftags_before = msh.get_faces(), msh.get_ftags()
        idx = np.lexsort(faces_before.T)
        faces_before = faces_before[idx, :]
        ftags_before = ftags_before[idx]

        tag = msh.add_boundary_faces()
        self.assertEqual((msh.get_ftags() == tag).sum(), 0)

    def test_boundary_faces_3d_2(self):
        coords, elems, etags, faces, ftags = get_cube()
        faces[0, :] = faces[0, ::-1]
        msh = Mesh33(coords, elems, etags, faces, ftags)
        msh = msh.split().split().split()

        faces_before, ftags_before = msh.get_faces(), msh.get_ftags()
        idx = np.lexsort(faces_before.T)
        faces_before = faces_before[idx, :]
        ftags_before = ftags_before[idx]

        tag = msh.add_boundary_faces()
        self.assertEqual((msh.get_ftags() == tag).sum(), 0)

    def test_boundary_faces_3d_3(self):
        coords, elems, etags, faces, ftags = get_cube()
        faces = faces[:2, :]
        ftags = ftags[:2]
        msh = Mesh33(coords, elems, etags, faces, ftags)
        msh = msh.split().split().split()

        tag = msh.add_boundary_faces()
        self.assertEqual((msh.get_ftags() == tag).sum(), 10 * 4**3)

    def test_fields_2d(self):
        coords, elems, etags, faces, ftags = get_square()
        msh = Mesh22(coords, elems, etags, faces, ftags)
        msh = msh.split().split().split()

        x, y = msh.get_coords().T
        f = (x + y).reshape((-1, 1))
        g = (x - y).reshape((-1, 1))

        elems = msh.get_elems()
        f_e = f[elems].mean(axis=1)
        g_e = g[elems].mean(axis=1)

        with self.assertRaises(ValueError):
            msh.vertex_data_to_elem_data(f_e)

        msh.vertex_data_to_elem_data(f)

        with self.assertRaises(ValueError):
            msh.elem_data_to_vertex_data(f)

        with self.assertRaises(RuntimeError):
            msh.elem_data_to_vertex_data(f_e)

        msh.compute_vertex_to_elems()
        msh.compute_volumes()

        msh.elem_data_to_vertex_data(f_e)

    @unittest.skipUnless(HAVE_MESHB, "The libMeshb interface is not available")
    def test_meshb_3d(self):
        coords, elems, etags, faces, ftags = get_cube()
        msh = Mesh33(coords, elems, etags, faces, ftags)

        msh.write_meshb("tmp.meshb")

        msh2 = Mesh33.from_meshb("tmp.meshb")

        self.assertEqual(msh.n_verts(), msh2.n_verts())
        self.assertTrue(np.allclose(msh.get_coords(), msh2.get_coords()))

        self.assertEqual(msh.n_elems(), msh2.n_elems())
        self.assertTrue(np.allclose(msh.get_elems(), msh2.get_elems()))
        self.assertTrue(np.allclose(msh.get_etags(), msh2.get_etags()))

        self.assertEqual(msh.n_faces(), msh2.n_faces())
        self.assertTrue(np.allclose(msh.get_faces(), msh2.get_faces()))
        self.assertTrue(np.allclose(msh.get_ftags(), msh2.get_ftags()))

        os.remove("tmp.meshb")

    def test_vols_2d(self):
        coords, elems, etags, faces, ftags = get_square()
        msh = Mesh22(coords, elems, etags, faces, ftags)
        msh = msh.split().split().split()

        vol = msh.vol()
        self.assertTrue(np.allclose(vol, 1.0))

        vols = msh.vols()
        self.assertTrue(np.allclose(vols.sum(), 1.0))
