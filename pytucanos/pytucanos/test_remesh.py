import numpy as np
import unittest
from . import (
    BoundaryMesh2d,
    Mesh2d,
    LinearGeometry2d,
    Remesher2dIso,
    Remesher2dAniso,
    RemesherParams,
    RemeshingStep,
    ParallelRemesher2dIso,
    ParallelRemesher2dAniso,
    ParallelRemesherParams,
    Idx,
)
from .mesh import (
    get_square,
)
from .remesh import (
    update_params,
)


class TestRemesh(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import logging

        logging.disable(logging.DEBUG)

    def test_2d_iso(self):
        coords, elems, etags, faces, ftags = get_square()
        msh = Mesh2d(coords, elems, etags, faces, ftags).split().split()
        bdy, _ = msh.boundary()
        geom = LinearGeometry2d(bdy)

        h = 0.1 * np.ones(msh.n_verts()).reshape((-1, 1))

        remesher = Remesher2dIso(msh, geom, h)
        remesher.remesh(geom, RemesherParams.default())

        msh = remesher.to_mesh()

        self.assertTrue(np.allclose(msh.vol(), 1.0))
        etags = np.unique(msh.get_etags())
        etags.sort()
        self.assertTrue(np.array_equal(etags, [1, 2]))
        ftags = np.unique(msh.get_ftags())
        ftags.sort()
        self.assertTrue(np.array_equal(ftags, [1, 2, 3, 4, 5]))

        self.assertGreater(msh.n_verts(), 100)
        self.assertLess(msh.n_verts(), 200)

    def test_2d_iso_parallel(self):
        coords, elems, etags, faces, ftags = get_square(two_tags=False)
        msh = Mesh2d(coords, elems, etags, faces, ftags).split().split()
        bdy, _ = msh.boundary()
        bdy.fix()
        geom = LinearGeometry2d(bdy)

        h = 0.1 * np.ones(msh.n_verts()).reshape((-1, 1))

        remesher = ParallelRemesher2dIso(msh, 2)
        params = RemesherParams.default()
        parallel_params = ParallelRemesherParams.default()
        parallel_params.max_levels = 2
        parallel_params.min_verts = 0
        (msh, _, _) = remesher.remesh(geom, h, params, parallel_params)

        self.assertTrue(np.allclose(msh.vol(), 1.0))
        etags = np.unique(msh.get_etags())
        etags.sort()
        self.assertTrue(np.array_equal(etags, [1]))
        ftags = np.unique(msh.get_ftags())
        ftags.sort()
        self.assertTrue(np.array_equal(ftags, [1, 2, 3, 4]))

        self.assertGreater(msh.n_verts(), 100)
        self.assertLess(msh.n_verts(), 200)

    def test_2d_iso_circle(self):
        coords, elems, etags, faces, ftags = get_square()
        msh = Mesh2d(coords, elems, etags, faces, ftags)

        n = 3
        theta = 0.25 * np.pi + np.linspace(0, 2 * np.pi, 4 * n + 1)
        r = 0.5 * 2**0.5
        x = 0.5 + r * np.cos(theta)
        y = 0.5 + r * np.sin(theta)
        coords = np.stack([x, y], axis=-1)

        idx = np.arange(4 * n, dtype=Idx)
        elems = np.stack(
            [idx, idx + 1],
            axis=-1,
        )
        elems[-1, 1] = 0
        etags = np.zeros(4 * n, dtype=np.int16)
        etags[0 * n : 1 * n] = 3
        etags[1 * n : 2 * n] = 4
        etags[2 * n : 3 * n] = 1
        etags[3 * n : 4 * n] = 2

        elems = np.vstack(
            [
                elems,
                np.array(
                    [
                        [2 * n, 0],
                    ],
                    dtype=Idx,
                ),
            ]
        )
        etags = np.append(
            etags,
            np.array(
                [5],
                dtype=np.int16,
            ),
        )

        faces = np.zeros((0, 1), dtype=Idx)
        ftags = np.zeros(0, dtype=np.int16)
        bdy = BoundaryMesh2d(coords, elems, etags, faces, ftags)
        geom = LinearGeometry2d(bdy)

        h = 0.1 * np.ones(msh.n_verts()).reshape((-1, 1))

        steps = RemesherParams.default().steps
        steps = 10 * steps[:5] + steps[-2:]
        params = RemesherParams(steps, False)
        params = update_params(params, RemeshingStep.Split, "min_q_rel", 0.5)

        remesher = Remesher2dIso(msh, geom, h)
        remesher.remesh(geom, params)
        msh = remesher.to_mesh()

        self.assertGreater(msh.vol(), 0.9 * 0.5 * np.pi)
        self.assertLess(msh.vol(), 1.1 * 0.5 * np.pi)

        etags = np.unique(msh.get_etags())
        etags.sort()
        self.assertTrue(np.array_equal(etags, [1, 2]))
        ftags = np.unique(msh.get_ftags())
        ftags.sort()
        self.assertTrue(np.array_equal(ftags, [1, 2, 3, 4, 5]))

        self.assertGreater(msh.n_verts(), 100 * msh.vol())
        self.assertLess(msh.n_verts(), 200 * msh.vol())

    def test_2d_aniso(self):
        coords, elems, etags, faces, ftags = get_square()
        msh = Mesh2d(coords, elems, etags, faces, ftags).split().split()

        for _ in range(4):
            bdy, _ = msh.boundary()
            geom = LinearGeometry2d(bdy)
            hx = 0.3
            hy = 0.03
            m = np.zeros((msh.n_verts(), 3))
            m[:, 0] = 1.0 / hx**2
            m[:, 1] = 1.0 / hy**2

            remesher = Remesher2dAniso(msh, geom, m)
            # steps = RemesherParams.default().steps
            # steps = 4 * steps[:5] + steps[-2:]
            # params = RemesherParams(steps, False)
            # # params = update_params(params, RemeshingStep.Split, "min_l_abs", 0.1)
            # params = update_params(params, RemeshingStep.Split, "min_q_rel_bdy", 0.5)
            remesher.remesh(geom, RemesherParams.default())
            msh = remesher.to_mesh()

        q = remesher.qualities()
        self.assertGreater(q.min(), 0.11)

        self.assertTrue(np.allclose(msh.vol(), 1.0))
        etags = np.unique(msh.get_etags())
        etags.sort()
        self.assertTrue(np.array_equal(etags, [1, 2]))
        ftags = np.unique(msh.get_ftags())
        ftags.sort()
        self.assertTrue(np.array_equal(ftags, [1, 2, 3, 4, 5]))

        c = remesher.complexity()
        self.assertGreater(msh.n_verts(), 0.5 * c)
        self.assertLess(msh.n_verts(), 2.5 * c)

        self.assertTrue(np.allclose(c, 4.0 / 3.0**0.5 / (0.3 * 0.03)))

        self.assertGreater(msh.n_verts(), 150)
        self.assertLess(msh.n_verts(), 300)

    def test_2d_aniso_parallel(self):
        coords, elems, etags, faces, ftags = get_square(two_tags=False)
        msh = Mesh2d(coords, elems, etags, faces, ftags).split().split()
        bdy, _ = msh.boundary()
        geom = LinearGeometry2d(bdy)

        hx = 0.3
        hy = 0.03
        for _ in range(4):
            m = np.zeros((msh.n_verts(), 3))
            m[:, 0] = 1.0 / hx**2
            m[:, 1] = 1.0 / hy**2

            remesher = ParallelRemesher2dAniso(msh, 2)
            params = update_params(
                RemesherParams.default(), RemeshingStep.Split, "min_q_abs", 0.1
            )
            parallel_params = ParallelRemesherParams.default()
            (msh, _, _) = remesher.remesh(geom, m, params, parallel_params)

        self.assertTrue(np.allclose(msh.vol(), 1.0))
        etags = np.unique(msh.get_etags())
        etags.sort()
        self.assertTrue(np.array_equal(etags, [1]))
        ftags = np.unique(msh.get_ftags())
        ftags.sort()
        self.assertTrue(np.array_equal(ftags, [1, 2, 3, 4]))

        self.assertGreater(msh.n_verts(), 150)
        self.assertLess(msh.n_verts(), 300)
