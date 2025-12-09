//! Tetrahedron meshes in 3d
use crate::{
    Vert3d,
    mesh::{GenericMesh, Hexahedron, Mesh, Quadrangle, Tetrahedron, Triangle, elements::Idx},
};

/// Create a `Mesh<3, Tetrahedron<_>>` of a `lx` by `ly` by `lz` box by splitting a `nx` by `ny` by `nz`
/// uniform structured grid
#[must_use]
pub fn box_mesh<M: Mesh<3, C = Tetrahedron<impl Idx>>>(
    lx: f64,
    nx: usize,
    ly: f64,
    ny: usize,
    lz: f64,
    nz: usize,
) -> M {
    let dx = lx / (nx as f64 - 1.);
    let x_1d = (0..nx).map(|i| i as f64 * dx).collect::<Vec<_>>();

    let dy = ly / (ny as f64 - 1.);
    let y_1d = (0..ny).map(|i| i as f64 * dy).collect::<Vec<_>>();

    let dz = lz / (nz as f64 - 1.);
    let z_1d = (0..nz).map(|i| i as f64 * dz).collect::<Vec<_>>();

    nonuniform_box_mesh(&x_1d, &y_1d, &z_1d)
}

/// Create a `Mesh<2, Tetrahedron<_>>` of box by splitting a structured grid`
#[must_use]
pub fn nonuniform_box_mesh<M: Mesh<3, C = Tetrahedron<impl Idx>>>(
    x: &[f64],
    y: &[f64],
    z: &[f64],
) -> M {
    let nx = x.len();
    let ny = y.len();
    let nz = z.len();

    let idx = |i, j, k| i + j * nx + k * nx * ny;

    let mut verts = vec![Vert3d::zeros(); nx * ny * nz];
    for (i, &x) in x.iter().enumerate() {
        for (j, &y) in y.iter().enumerate() {
            for (k, &z) in z.iter().enumerate() {
                verts[idx(i, j, k)] = Vert3d::new(x, y, z);
            }
        }
    }

    let mut hexas = Vec::with_capacity((nx - 1) * (ny - 1) * (nz - 1));
    let mut etags = Vec::with_capacity(hexas.capacity());
    for i in 0..nx - 1 {
        for j in 0..ny - 1 {
            for k in 0..nz - 1 {
                hexas.push(Hexahedron::new([
                    idx(i, j, k),
                    idx(i + 1, j, k),
                    idx(i + 1, j + 1, k),
                    idx(i, j + 1, k),
                    idx(i, j, k + 1),
                    idx(i + 1, j, k + 1),
                    idx(i + 1, j + 1, k + 1),
                    idx(i, j + 1, k + 1),
                ]));
                etags.push(1);
            }
        }
    }

    let mut quads = Vec::with_capacity(
        2 * (nx - 1) * (ny - 1) + 2 * (nx - 1) * (nz - 1) + 2 * (nz - 1) * (ny - 1),
    );
    let mut ftags = Vec::with_capacity(quads.capacity());

    for i in 0..nx - 1 {
        for j in 0..ny - 1 {
            let k = 0;
            quads.push(Quadrangle::new(
                idx(i, j, k),
                idx(i, j + 1, k),
                idx(i + 1, j + 1, k),
                idx(i + 1, j, k),
            ));
            ftags.push(1);
            let k = nz - 1;
            quads.push(Quadrangle::new(
                idx(i, j, k),
                idx(i + 1, j, k),
                idx(i + 1, j + 1, k),
                idx(i, j + 1, k),
            ));
            ftags.push(2);
        }
    }

    for i in 0..nx - 1 {
        for k in 0..nz - 1 {
            let j = 0;
            quads.push(Quadrangle::new(
                idx(i, j, k),
                idx(i + 1, j, k),
                idx(i + 1, j, k + 1),
                idx(i, j, k + 1),
            ));
            ftags.push(3);
            let j = ny - 1;
            quads.push(Quadrangle::new(
                idx(i, j, k),
                idx(i, j, k + 1),
                idx(i + 1, j, k + 1),
                idx(i + 1, j, k),
            ));
            ftags.push(4);
        }
    }

    for j in 0..ny - 1 {
        for k in 0..nz - 1 {
            let i = 0;
            quads.push(Quadrangle::new(
                idx(i, j, k),
                idx(i, j, k + 1),
                idx(i, j + 1, k + 1),
                idx(i, j + 1, k),
            ));
            ftags.push(5);
            let i = nx - 1;
            quads.push(Quadrangle::new(
                idx(i, j, k),
                idx(i, j + 1, k),
                idx(i, j + 1, k + 1),
                idx(i, j, k + 1),
            ));
            ftags.push(6);
        }
    }

    let mut res = M::empty();
    res.add_verts(verts.iter().copied());
    res.add_hexahedra(hexas.iter().copied(), etags.iter().copied());
    res.add_quadrangles(quads.iter().copied(), ftags.iter().copied());
    // let faces = res.compute_faces();
    // res.fix_orientation(&faces);
    res
}

/// Create a `Mesh<3, Tetrahedron<_>>` of a ball
#[must_use]
pub fn ball_mesh<M: Mesh<3, C = Tetrahedron<impl Idx>>>(r: f64, n: usize) -> M {
    let mut res = M::empty();

    res.add_verts(
        [
            (0., 0., 1.),
            (1., 0., 0.),
            (0., 1., 0.),
            (-1., 0., 0.),
            (0., -1., 0.),
            (0., 0., -1.),
            (0., 0., 0.),
        ]
        .into_iter()
        .map(|(x, y, z)| Vert3d::new(x, y, z)),
    );

    res.add_elems_and_tags(
        [
            (6, 0, 1, 2),
            (6, 0, 2, 3),
            (6, 0, 3, 4),
            (6, 0, 4, 1),
            (6, 5, 2, 1),
            (6, 5, 3, 2),
            (6, 5, 4, 3),
            (6, 5, 1, 4),
        ]
        .into_iter()
        .map(|(a, b, c, d)| (Tetrahedron::new(a, b, c, d), 1)),
    );

    res.add_faces_and_tags(
        [
            (0, 1, 2),
            (0, 2, 3),
            (0, 3, 4),
            (0, 4, 1),
            (5, 2, 1),
            (5, 3, 2),
            (5, 4, 3),
            (5, 1, 4),
        ]
        .into_iter()
        .map(|(a, b, c)| (Triangle::new(a, b, c), 1)),
    );

    for _ in 0..n {
        res = res.split();
        let flg = res.boundary_flag();

        for (x, is_boundary) in res.verts_mut().zip(flg) {
            if is_boundary {
                *x *= r / x.norm();
            }
        }
    }

    res
}

/// Tetrahedron mesh in 3d
pub type Mesh3d = GenericMesh<3, Tetrahedron<usize>>;

#[cfg(test)]
mod tests {
    use crate::{
        Vert3d, assert_delta,
        mesh::{
            BoundaryMesh3d, GSimplex, GradientMethod, Mesh, Mesh3d, Simplex, bandwidth, box_mesh,
            mesh_3d::ball_mesh,
            partition::{HilbertPartitioner, KMeansPartitioner3d, RCMPartitioner},
        },
    };
    use rayon::iter::ParallelIterator;
    use std::f64::consts::PI;

    #[test]
    fn test_box() {
        let msh = box_mesh::<Mesh3d>(1.0, 2, 1.0, 2, 1.0, 2).random_shuffle();

        let faces = msh.all_faces();
        msh.check(&faces).unwrap();

        let vol = msh.gelems().map(|ge| ge.vol()).sum::<f64>();
        assert_delta!(vol, 1.0, 1e-12);
    }

    #[test]
    fn test_gradient_linear() {
        let grad = Vert3d::new(9.8, 7.6, 5.4);
        let msh = box_mesh::<Mesh3d>(1.0, 10, 1.0, 15, 1.0, 20).random_shuffle();
        let f = msh
            .par_verts()
            .map(|v| grad[0] * v[0] + grad[1] * v[1] + grad[2] * v[2])
            .collect::<Vec<_>>();

        for method in [
            GradientMethod::LinearLeastSquares(1),
            GradientMethod::LinearLeastSquares(2),
            GradientMethod::QuadraticLeastSquares(1),
            GradientMethod::L2Projection,
        ] {
            let gradient = msh.gradient(method, &f);

            for x in gradient.chunks(3) {
                let x = Vert3d::from_row_slice(x);
                let err = (x - grad).norm();
                assert!(err < 1e-10, "{method:?}, {x:?}");
            }
        }
    }

    fn run_gradient(method: GradientMethod, n: u32, shake: bool) -> f64 {
        let n = 2_usize.pow(n) + 1;
        let mut mesh = box_mesh::<Mesh3d>(1.0, n, 1.0, n, 1.0, n).random_shuffle();
        if shake {
            mesh.random_shake(0.1);
        }
        let f = mesh.verts().map(|p| p[0] * p[1] * p[2]).collect::<Vec<_>>();
        let grad = mesh
            .verts()
            .map(|p| Vert3d::new(p[1] * p[2], p[0] * p[2], p[0] * p[1]))
            .collect::<Vec<_>>();
        let res = mesh
            .gradient(method, &f)
            .chunks(3)
            .map(Vert3d::from_column_slice)
            .collect::<Vec<_>>();
        let err = grad
            .iter()
            .zip(res.iter())
            .map(|(x, y)| (x - y).norm())
            .collect::<Vec<_>>();

        mesh.norm(&err)
    }

    #[test]
    fn test_gradient() {
        for method in [
            GradientMethod::LinearLeastSquares(1),
            GradientMethod::LinearLeastSquares(2),
            GradientMethod::QuadraticLeastSquares(1),
            GradientMethod::L2Projection,
        ] {
            let mut prev = f64::MAX;
            for n in 2..5 {
                let nrm = run_gradient(method, n, false);
                assert!(nrm < 0.5 * prev, "{method:?}, {nrm:.2e} {prev:.2e}");
                prev = nrm;
            }
        }
    }

    #[test]
    fn test_gradient_shake() {
        for method in [
            GradientMethod::LinearLeastSquares(1),
            GradientMethod::LinearLeastSquares(2),
            GradientMethod::QuadraticLeastSquares(1),
            GradientMethod::L2Projection,
        ] {
            let mut prev = f64::MAX;
            for n in 2..5 {
                let nrm = run_gradient(method, n, true);
                assert!(nrm < 0.5 * prev, "{method:?}, {nrm:.2e} {prev:.2e}");
                prev = nrm;
            }
        }
    }

    #[test]
    fn test_hessian_quadratic() {
        let mesh = box_mesh::<Mesh3d>(1.0, 10, 1.0, 15, 1.0, 20).random_shuffle();
        let flg = mesh.boundary_flag();
        let v2v = mesh.vertex_to_vertices();

        let f: Vec<_> = mesh
            .verts()
            .map(|p| {
                p[0] * p[0]
                    + 2.0 * p[1] * p[1]
                    + 3.0 * p[2] * p[2]
                    + 4.0 * p[0] * p[1]
                    + 5.0 * p[1] * p[2]
                    + 6.0 * p[0] * p[2]
            })
            .collect();

        for method in [
            GradientMethod::QuadraticLeastSquares(1),
            GradientMethod::L2Projection,
        ] {
            let res = mesh.hessian(method, &f);
            for i_vert in 0..mesh.n_verts() {
                if matches!(method, GradientMethod::L2Projection)
                    && v2v.row(i_vert).iter().any(|&j| flg[j])
                {
                    // l2proj is not correct at the boundaries
                    continue;
                }
                assert!(f64::abs(res[6 * i_vert] - 2.) < 1e-10);
                assert!(f64::abs(res[6 * i_vert + 1] - 4.) < 1e-10);
                assert!(f64::abs(res[6 * i_vert + 2] - 6.) < 1e-10);
                assert!(f64::abs(res[6 * i_vert + 3] - 4.) < 1e-10);
                assert!(f64::abs(res[6 * i_vert + 4] - 5.) < 1e-10);
                assert!(f64::abs(res[6 * i_vert + 5] - 6.) < 1e-10);
            }
        }
    }

    fn run_hessian(method: GradientMethod, n: u32, shake: bool) -> f64 {
        let n = 2_usize.pow(n) + 1;
        let mut mesh = box_mesh::<Mesh3d>(1.0, n, 1.0, n, 1.0, n).random_shuffle();
        if shake {
            mesh.random_shake(0.1);
        }

        let f: Vec<_> = mesh
            .verts()
            .map(|p| {
                let x = p[0];
                let y = p[1];
                let z = p[2];
                x * x * y * z + 2.0 * x * y * y * z + 3.0 * x * y * z * z
            })
            .collect();
        let hess = mesh
            .verts()
            .map(|p| {
                let x = p[0];
                let y = p[1];
                let z = p[2];
                [
                    2.0 * y * z,
                    4.0 * x * z,
                    6.0 * x * y,
                    z * (2.0 * x + 4.0 * y + 3.0 * z),
                    x * (x + 4.0 * y + 6.0 * z),
                    2.0 * y * (x + y + 3.0 * z),
                ]
            })
            .collect::<Vec<_>>();
        let res = mesh.hessian(method, &f);

        let err = hess
            .iter()
            .zip(res.chunks(6))
            .map(|(x, y)| {
                ((x[0] - y[0]).powi(2)
                    + (x[1] - y[1]).powi(2)
                    + (x[2] - y[2]).powi(2)
                    + (x[3] - y[3]).powi(2)
                    + (x[4] - y[4]).powi(2)
                    + (x[5] - y[5]).powi(2))
                .sqrt()
            })
            .collect::<Vec<_>>();

        mesh.norm(&err)
    }

    #[test]
    fn test_hessian() {
        let mut prev = f64::MAX;
        for n in 2..5 {
            let nrm = run_hessian(GradientMethod::QuadraticLeastSquares(1), n, false);
            assert!(nrm < 0.5 * prev);
            prev = nrm;
        }
    }

    #[test]
    fn test_hessian_l2proj() {
        // WARNING: l2proj hessian does not converge
        let mut prev = f64::MAX;
        for n in 2..5 {
            let nrm = run_hessian(GradientMethod::L2Projection, n, false);
            assert!(nrm < prev, "{nrm:.2e} {prev:.2e}");
            prev = nrm;
        }
    }

    #[test]
    fn test_hessian_shake() {
        let mut prev = f64::MAX;
        for n in 2..5 {
            let nrm = run_hessian(GradientMethod::QuadraticLeastSquares(1), n, true);
            assert!(nrm < 0.5 * prev);
            prev = nrm;
        }
    }

    #[test]
    fn test_hessian_l2proj_shake() {
        // WARNING: l2proj hessian does not converge
        let mut prev = f64::MAX;
        for n in 2..5 {
            let nrm = run_hessian(GradientMethod::L2Projection, n, true);
            assert!(nrm < prev, "{nrm:.2e} {prev:.2e}");
            prev = nrm;
        }
    }

    #[test]
    fn test_smooth() {
        let mesh = box_mesh::<Mesh3d>(1.0, 9, 1.0, 9, 1.0, 9).random_shuffle();

        let f: Vec<_> = mesh
            .verts()
            .map(|p| p[0] + 2.0 * p[1] + 3.0 * p[2])
            .collect();
        let res = mesh.smooth(GradientMethod::LinearLeastSquares(2), &f);
        for i_vert in 0..mesh.n_verts() {
            assert!(f64::abs(res[i_vert] - f[i_vert]) < 1e-10);
        }

        let f: Vec<_> = mesh.verts().map(|p| p[0] * p[1] * p[2]).collect();
        let res = mesh.smooth(GradientMethod::LinearLeastSquares(2), &f);
        for i_vert in 0..mesh.n_verts() {
            assert!(f64::abs(res[i_vert] - f[i_vert]) < 2e-2);
        }
    }

    #[test]
    fn test_integrate() {
        let msh = box_mesh::<Mesh3d>(1.0, 10, 2.0, 15, 1.0, 20).random_shuffle();

        let vol = msh.par_gelems().map(|ge| ge.vol()).sum::<f64>();
        assert_delta!(vol, 2.0, 1e-12);

        let f = msh.par_verts().map(|v| v[0]).collect::<Vec<_>>();

        let val = msh.integrate(&f, |_| 1.0);
        assert_delta!(val, 2.0, 1e-12);

        let val = msh.integrate(&f, |x| x);
        assert_delta!(val, 1.0, 1e-12);

        let nrm = msh.norm(&f);
        let nrm_ref = (2.0_f64 / 3.0).sqrt();
        assert_delta!(nrm, nrm_ref, 1e-12);
    }

    #[test]
    fn test_meshb() {
        let msh = box_mesh::<Mesh3d>(1.0, 10, 1.0, 10, 1.0, 10);
        let fname = "box3d.meshb";
        msh.write_meshb(fname).unwrap();
        let new_msh = Mesh3d::from_meshb(fname).unwrap();

        msh.check_equals(&new_msh, 1e-12).unwrap();

        std::fs::remove_file(fname).unwrap();
    }

    #[test]
    fn test_rcm() {
        let msh = box_mesh::<Mesh3d>(1.0, 20, 1.0, 20, 1.0, 20).random_shuffle();
        let avg_bandwidth = bandwidth(msh.elems()).1;
        assert!(avg_bandwidth > 1000.0);

        let (msh_rcm, vert_ids, elem_ids, face_ids) = msh.reorder_rcm();
        let avg_bandwidth_rcm = bandwidth(msh_rcm.elems()).1;

        assert!(avg_bandwidth_rcm < 320.0);

        for (i, v) in msh_rcm.verts().enumerate() {
            let other = msh.vert(vert_ids[i]);
            assert!((v - other).norm() < 1e-12);
        }

        for (i, v) in msh_rcm.gelems().enumerate() {
            let v = v.center();
            let other = msh.gelem(&msh.elem(elem_ids[i]));
            let other = other.center();
            assert!((v - other).norm() < 1e-12);
        }

        for (i, tag) in msh_rcm.etags().enumerate() {
            let other = msh.etag(elem_ids[i]);
            assert_eq!(tag, other);
        }

        for (i, v) in msh_rcm.gfaces().enumerate() {
            let v = v.center();
            let other = msh.gface(&msh.face(face_ids[i]));
            let other = other.center();
            assert!((v - other).norm() < 1e-12);
        }

        for (i, tag) in msh_rcm.ftags().enumerate() {
            let other = msh.ftag(face_ids[i]);
            assert_eq!(tag, other);
        }

        msh_rcm.check(&msh_rcm.all_faces()).unwrap();
    }

    #[test]
    fn test_part_rcm() {
        let mut msh = box_mesh::<Mesh3d>(1.0, 20, 1.0, 20, 1.0, 20).random_shuffle();
        let (quality, imbalance) = msh.partition::<RCMPartitioner>(4, None).unwrap();

        assert!(quality < 0.045);
        assert!(imbalance < 0.0002);

        for i in 0..4 {
            let part = msh.get_partition(i).mesh;
            let cc = part.vertex_to_vertices().connected_components().unwrap();
            let n_cc = cc.iter().copied().max().unwrap() + 1;
            assert_eq!(n_cc, 1);
        }
    }

    #[test]
    fn test_hilbert() {
        let msh = box_mesh::<Mesh3d>(1.0, 20, 1.0, 20, 1.0, 20).random_shuffle();
        let avg_bandwidth = bandwidth(msh.elems()).1;
        assert!(avg_bandwidth > 1000.0);

        let (msh_hilbert, vert_ids, elem_ids, face_ids) = msh.reorder_hilbert();
        let avg_bandwidth_hilbert = bandwidth(msh_hilbert.elems()).1;
        assert!(avg_bandwidth_hilbert < 440.0);

        for (i, v) in msh_hilbert.verts().enumerate() {
            let other = msh.vert(vert_ids[i]);
            assert!((v - other).norm() < 1e-12);
        }

        for (i, v) in msh_hilbert.gelems().enumerate() {
            let v = v.center();
            let other = msh.gelem(&msh.elem(elem_ids[i]));
            let other = other.center();
            assert!((v - other).norm() < 1e-12);
        }

        for (i, tag) in msh_hilbert.etags().enumerate() {
            let other = msh.etag(elem_ids[i]);
            assert_eq!(tag, other);
        }

        for (i, v) in msh_hilbert.gfaces().enumerate() {
            let v = v.center();
            let other = msh.gface(&msh.face(face_ids[i]));
            let other = other.center();
            assert!((v - other).norm() < 1e-12);
        }

        for (i, tag) in msh_hilbert.ftags().enumerate() {
            let other = msh.ftag(face_ids[i]);
            assert_eq!(tag, other);
        }

        msh_hilbert.check(&msh_hilbert.all_faces()).unwrap();
    }

    #[test]
    fn test_part_hilbert() {
        let mut msh = box_mesh::<Mesh3d>(1.0, 20, 1.0, 20, 1.0, 20).random_shuffle();
        let (quality, imbalance) = msh.partition::<HilbertPartitioner>(4, None).unwrap();

        assert!(quality < 0.04);
        assert!(imbalance < 0.0002);

        for i in 0..4 {
            let part = msh.get_partition(i).mesh;
            let cc = part.vertex_to_vertices().connected_components().unwrap();
            let n_cc = cc.iter().copied().max().unwrap() + 1;
            assert_eq!(n_cc, 1);
        }
    }

    #[test]
    fn test_part_kmeans() {
        let mut msh = box_mesh::<Mesh3d>(1.0, 6, 1.0, 5, 1.0, 5).random_shuffle();
        let (quality, imbalance) = msh.partition::<KMeansPartitioner3d>(4, None).unwrap();

        assert!(quality < 0.11);
        assert!(imbalance < 0.04);

        for i in 0..4 {
            let part = msh.get_partition(i).mesh;
            let cc = part.vertex_to_vertices().connected_components().unwrap();
            let n_cc = cc.iter().copied().max().unwrap() + 1;
            assert_eq!(n_cc, 1);
        }
    }

    #[test]
    fn test_split() {
        let msh = box_mesh::<Mesh3d>(1.0, 2, 1.0, 2, 1.0, 2).random_shuffle();

        let msh = msh.split();
        assert_eq!(msh.n_verts(), 27);
        assert_eq!(msh.n_faces(), 12 * 4);
        assert_eq!(msh.n_elems(), 6 * 8);

        let (bdy, _): (BoundaryMesh3d, _) = msh.boundary();
        let area = bdy.gelems().map(|ge| ge.vol()).sum::<f64>();
        assert_delta!(area, 6.0, 1e-10);

        let vol = msh.gelems().map(|ge| ge.vol()).sum::<f64>();
        assert_delta!(vol, 1.0, 1e-10);
    }

    #[test]
    fn test_skewness_3d() {
        let mesh = box_mesh::<Mesh3d>(1.0, 3, 1.0, 3, 1.0, 3).random_shuffle();

        let all_faces = mesh.all_faces();
        let count = mesh
            .face_skewnesses(&all_faces)
            .map(|(_, _, s)| assert!(s < 0.5, "{s}"))
            .count();
        assert_eq!(count, 72);
    }

    #[test]
    fn test_edge_ratio_3d() {
        let mesh = box_mesh::<Mesh3d>(1.0, 3, 1.0, 3, 1.0, 3).random_shuffle();

        let count = mesh
            .edge_length_ratios()
            .map(|s| assert!(s < 3.0_f64.sqrt() + 1e-6))
            .count();
        assert_eq!(count, 48);
    }

    #[test]
    fn test_gamma_3d() {
        let mesh = box_mesh::<Mesh3d>(1.0, 3, 1.0, 3, 1.0, 3).random_shuffle();

        let (gamma_min, gamma_max) = mesh
            .elem_gammas()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |a, b| {
                (a.0.min(b), a.1.max(b))
            });
        assert!((gamma_min - 0.717).abs() < 1e-3);
        assert!((gamma_max - 0.717).abs() < 1e-3);
    }

    #[test]
    fn test_add_3d() {
        let mut mesh1 = box_mesh::<Mesh3d>(1.0, 3, 1.0, 3, 1.0, 3).random_shuffle();
        assert_eq!(mesh1.n_tagged_faces(6), 8);
        assert_eq!(mesh1.n_tagged_faces(11), 0);

        let mut mesh2 = box_mesh::<Mesh3d>(1.0, 3, 1.0, 3, 1.0, 3).random_shuffle();
        mesh2
            .verts_mut()
            .for_each(|x| *x += Vert3d::new(1.0, 0.5, 0.5));
        mesh2.ftags_mut().for_each(|t| *t += 10);

        mesh1.add(&mesh2, |_| true, |_| true, Some(1e-12));

        assert_eq!(mesh1.n_verts(), 2 * mesh2.n_verts() - 4);
        assert_eq!(mesh1.n_tagged_faces(6), 8);
        assert_eq!(mesh1.n_tagged_faces(11), 8);
    }

    #[test]
    fn test_ball() {
        let r = 1.234;
        let msh: Mesh3d = ball_mesh(r, 6);
        let surf: BoundaryMesh3d = msh.boundary().0;
        assert_delta!(msh.vol(), 4.0 / 3.0 * PI * r.powi(3), 0.003);
        assert_delta!(surf.vol(), 4.0 * PI * r.powi(2), 0.004);
    }

    #[test]
    fn test_shake() {
        let mut mesh = box_mesh::<Mesh3d>(1.0, 3, 1.0, 3, 1.0, 3).random_shuffle();
        mesh.random_shake(0.1);
        mesh.check(&mesh.all_faces()).unwrap();

        assert_delta!(mesh.vol(), 1.0, 1e-12);

        let mut l = mesh
            .edges()
            .keys()
            .map(|e| (mesh.vert(e.get(0)) - mesh.vert(e.get(1))).norm())
            .collect::<Vec<_>>();
        l.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for tmp in l.windows(2) {
            assert!(tmp[1] > tmp[0] + 1e-8);
        }
    }

    #[test]
    fn test_shake_ball() {
        let mut mesh = ball_mesh::<Mesh3d>(1.0, 3).random_shuffle();
        assert_delta!(mesh.vol(), 4.0 / 3.0 * PI, 0.1);

        mesh.random_shake(0.1);
        mesh.check(&mesh.all_faces()).unwrap();

        assert_delta!(mesh.vol(), 4.0 / 3.0 * PI, 0.21);

        let mut l = mesh
            .edges()
            .keys()
            .map(|e| (mesh.vert(e.get(0)) - mesh.vert(e.get(1))).norm())
            .collect::<Vec<_>>();
        l.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for tmp in l.windows(2) {
            assert!(tmp[1] > tmp[0] + 1e-10);
        }
    }
}
