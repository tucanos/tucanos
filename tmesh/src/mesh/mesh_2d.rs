//! Triangle meshes in 2d
use crate::{
    Vert2d,
    mesh::{
        Edge, GenericMesh, Mesh, Quadrangle, QuadraticTriangle, Triangle, elements::Idx,
        to_quadratic::to_quadratic_triangle_mesh,
    },
};

/// Create a `Mesh<2, C=Triangle<_>>` of a `lx` by `ly` rectangle by splitting a `nx` by `ny`
/// uniform structured grid
#[must_use]
pub fn rectangle_mesh<M: Mesh<2, C = Triangle<impl Idx>>>(
    lx: f64,
    nx: usize,
    ly: f64,
    ny: usize,
) -> M {
    let dx = lx / (nx as f64 - 1.);
    let x_1d = (0..nx).map(|i| i as f64 * dx).collect::<Vec<_>>();

    let dy = ly / (ny as f64 - 1.);
    let y_1d = (0..ny).map(|i| i as f64 * dy).collect::<Vec<_>>();

    nonuniform_rectangle_mesh(&x_1d, &y_1d)
}

/// Create a `Mesh<2, C=Triangle<_>>` of rectangle by splitting a structured grid
#[must_use]
pub fn nonuniform_rectangle_mesh<M: Mesh<2, C = Triangle<impl Idx>>>(x: &[f64], y: &[f64]) -> M {
    let nx = x.len();
    let ny = y.len();

    let idx = |i, j| i + j * nx;

    let mut verts = vec![Vert2d::zeros(); nx * ny];
    for (i, &x) in x.iter().enumerate() {
        for (j, &y) in y.iter().enumerate() {
            verts[idx(i, j)] = Vert2d::new(x, y);
        }
    }

    let mut quads = Vec::with_capacity((nx - 1) * (ny - 1));
    let mut etags = Vec::with_capacity((nx - 1) * (ny - 1));
    for i in 0..nx - 1 {
        for j in 0..ny - 1 {
            quads.push(Quadrangle::new(
                idx(i, j),
                idx(i + 1, j),
                idx(i + 1, j + 1),
                idx(i, j + 1),
            ));
            etags.push(1);
        }
    }

    let mut faces = Vec::with_capacity(2 * (nx - 1 + ny - 1));
    let mut ftags = Vec::with_capacity(2 * (nx - 1 + ny - 1));

    for i in 0..nx - 1 {
        faces.push(Edge::new(idx(i, 0), idx(i + 1, 0)));
        ftags.push(1);
        faces.push(Edge::new(idx(i + 1, ny - 1), idx(i, ny - 1)));
        ftags.push(3);
    }

    for j in 0..ny - 1 {
        faces.push(Edge::new(idx(nx - 1, j), idx(nx - 1, j + 1)));
        ftags.push(2);
        faces.push(Edge::new(idx(0, j + 1), idx(0, j)));
        ftags.push(4);
    }

    let mut res = M::empty();
    res.add_verts(verts.iter().copied());
    res.add_quadrangles(quads.iter().copied(), etags.iter().copied());
    res.add_faces(faces.iter().copied(), ftags.iter().copied());
    let faces = res.all_faces();
    res.fix_faces_orientation(&faces);
    res
}

/// Create a `Mesh<2, C=Triangle<_>>` of circle by splitting square and projecting the vertices
#[must_use]
pub fn disk_mesh<M: Mesh<2, C = Triangle<impl Idx>>>(n: usize) -> M {
    let mut msh: M = rectangle_mesh(0.5_f64.sqrt(), 2, 0.5_f64.sqrt(), 2);
    msh.verts_mut().for_each(|v| {
        *v -= 0.5_f64.sqrt() * Vert2d::new(0.5, 0.5);
    });

    for _ in 0..n {
        msh = msh.split();
        let flg = msh.boundary_flag();
        for (v, f) in msh.verts_mut().zip(flg) {
            if f {
                *v *= 0.5 / v.norm();
            }
        }
    }

    msh
}

/// Create a `Mesh<2, C=QuadraticTriangle<_>>` of circle by splitting square and projecting the vertices
#[must_use]
pub fn quadratic_disk_mesh<M: Mesh<2, C = QuadraticTriangle<impl Idx>>>(n: usize) -> M {
    let msh = disk_mesh::<GenericMesh<2, Triangle<usize>>>(n);
    let mut msh: M = to_quadratic_triangle_mesh(&msh);
    let flg = msh.boundary_flag();
    for (v, f) in msh.verts_mut().zip(flg) {
        if f {
            *v *= 0.5 / v.norm();
        }
    }

    msh
}
/// Triangle mesh in 2d
pub type Mesh2d = GenericMesh<2, Triangle<usize>>;
/// Quadratic triangle mesh in 2d
pub type QuadraticMesh2d = GenericMesh<2, QuadraticTriangle<usize>>;

#[cfg(test)]
mod tests {
    use crate::{
        Vert2d, assert_delta,
        mesh::{
            AdativeBoundsQuadraticTriangle, BoundaryMesh2d, Edge, GSimplex, GradientMethod, Mesh,
            Mesh2d, QuadraticMesh2d, bandwidth, disk_mesh, quadratic_disk_mesh, rectangle_mesh,
        },
    };
    use rayon::iter::ParallelIterator;

    #[test]
    fn test_2d_simple_1() {
        let msh = rectangle_mesh::<Mesh2d>(1.0, 2, 1.0, 2);

        let faces = msh.all_faces();
        msh.check(&faces).unwrap();

        let edgs = msh.edges();
        assert_eq!(edgs.len(), 5, "{edgs:?}");
        assert!(edgs.contains_key(&Edge::new(0, 1)));
        assert!(edgs.contains_key(&Edge::new(2, 3)));
        assert!(edgs.contains_key(&Edge::new(0, 2)));
        assert!(edgs.contains_key(&Edge::new(1, 3)));
        assert!(edgs.contains_key(&Edge::new(0, 3)));

        let faces = msh.all_faces();
        assert_eq!(faces.len(), 5);
        assert!(faces.contains_key(&Edge::new(0, 1)));
        assert!(faces.contains_key(&Edge::new(2, 3)));
        assert!(faces.contains_key(&Edge::new(0, 2)));
        assert!(faces.contains_key(&Edge::new(1, 3)));
        assert!(faces.contains_key(&Edge::new(0, 3)));
    }

    #[test]
    fn test_2d_simple_2() {
        let msh = rectangle_mesh::<Mesh2d>(1.0, 3, 1.0, 2);

        let faces = msh.all_faces();
        msh.check(&faces).unwrap();

        let edgs = msh.edges();
        assert_eq!(edgs.len(), 9);
        assert!(edgs.contains_key(&Edge::new(0, 1)));
        assert!(edgs.contains_key(&Edge::new(1, 2)));
        assert!(edgs.contains_key(&Edge::new(3, 4)));
        assert!(edgs.contains_key(&Edge::new(4, 5)));
        assert!(edgs.contains_key(&Edge::new(0, 3)));
        assert!(edgs.contains_key(&Edge::new(1, 4)));
        assert!(edgs.contains_key(&Edge::new(2, 5)));
        assert!(edgs.contains_key(&Edge::new(0, 4)));
        assert!(edgs.contains_key(&Edge::new(1, 5)));
    }

    #[test]
    fn test_2d_rect() {
        let msh = rectangle_mesh::<Mesh2d>(1.0, 10, 2.0, 15).random_shuffle();

        let faces = msh.all_faces();
        msh.check(&faces).unwrap();

        let edgs = msh.edges();
        assert_eq!(edgs.len(), 9 * 15 + 10 * 14 + 9 * 14);

        let vol = msh.gelems().map(|ge| ge.vol()).sum::<f64>();
        assert!((vol - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_gradient_linear() {
        let grad = Vert2d::new(9.8, 7.6);
        let msh = rectangle_mesh::<Mesh2d>(1.0, 10, 1.0, 10).random_shuffle();
        let f = msh
            .par_verts()
            .map(|v| grad[0] * v[0] + grad[1] * v[1])
            .collect::<Vec<_>>();

        for method in [
            GradientMethod::LinearLeastSquares(1),
            GradientMethod::LinearLeastSquares(2),
            GradientMethod::QuadraticLeastSquares(1),
            GradientMethod::L2Projection,
        ] {
            let gradient = msh.gradient(method, &f);

            for x in gradient.chunks(2) {
                let x = Vert2d::from_row_slice(x);
                let err = (x - grad).norm();
                assert!(err < 1e-10, "{method:?}, {x:?}");
            }
        }
    }

    fn run_gradient(method: GradientMethod, n: u32) -> f64 {
        let n = 2_usize.pow(n) + 1;
        let mesh = rectangle_mesh::<Mesh2d>(1.0, n, 1.0, n).random_shuffle();

        let f = mesh.verts().map(|p| p[0] * p[1]).collect::<Vec<_>>();
        let grad = mesh
            .verts()
            .map(|p| Vert2d::new(p[1], p[0]))
            .collect::<Vec<_>>();
        let res = mesh
            .gradient(method, &f)
            .chunks(2)
            .map(Vert2d::from_column_slice)
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
            for n in 3..7 {
                let nrm = run_gradient(method, n);
                assert!(nrm < 0.5 * prev, "{method:?}, {nrm:.2e} {prev:.2e}");
                prev = nrm;
            }
        }
    }

    #[test]
    fn test_hessian_quadratic() {
        let mesh = rectangle_mesh::<Mesh2d>(1.0, 10, 1.0, 10).random_shuffle();
        let flg = mesh.boundary_flag();
        let v2v = mesh.vertex_to_vertices();

        let f: Vec<_> = mesh
            .verts()
            .map(|p| p[0] * p[0] + 2.0 * p[1] * p[1] + 3.0 * p[0] * p[1])
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
                assert!(
                    f64::abs(res[3 * i_vert] - 2.) < 1e-10,
                    "{method:?}, {:?}",
                    &res[3 * i_vert..3 * i_vert + 3]
                );
                assert!(
                    f64::abs(res[3 * i_vert + 1] - 4.) < 1e-10,
                    "{method:?}, {:?}",
                    &res[3 * i_vert..3 * i_vert + 3]
                );
                assert!(
                    f64::abs(res[3 * i_vert + 2] - 3.) < 1e-10,
                    "{method:?}, {:?}",
                    &res[3 * i_vert..3 * i_vert + 3]
                );
            }
        }
    }

    fn run_hessian(method: GradientMethod, n: u32) -> f64 {
        let n = 2_usize.pow(n) + 1;
        let mesh = rectangle_mesh::<Mesh2d>(1.0, n, 1.0, n).random_shuffle();

        let f: Vec<_> = mesh
            .verts()
            .map(|p| p[0] * p[0] * p[1] + 2.0 * p[0] * p[1] * p[1])
            .collect();
        let hess = mesh
            .verts()
            .map(|p| [2.0 * p[1], 4.0 * p[0], 2.0 * p[0] + 4.0 * p[1]])
            .collect::<Vec<_>>();
        let res = mesh.hessian(method, &f);

        let err = hess
            .iter()
            .zip(res.chunks(3))
            .map(|(x, y)| {
                ((x[0] - y[0]).powi(2) + (x[1] - y[1]).powi(2) + (x[2] - y[2]).powi(2)).sqrt()
            })
            .collect::<Vec<_>>();

        mesh.norm(&err)
    }

    #[test]
    fn test_hessian() {
        let mut prev = f64::MAX;
        for n in 3..7 {
            let nrm = run_hessian(GradientMethod::QuadraticLeastSquares(1), n);
            assert!(nrm < 0.5 * prev, "{nrm:.2e} {prev:.2e}");
            prev = nrm;
        }
    }

    #[test]
    fn test_hessian_l2proj() {
        // WARNING: l2proj hessian does not converge
        let mut prev = f64::MAX;
        for n in 3..7 {
            let nrm = run_hessian(GradientMethod::L2Projection, n);
            assert!(nrm < prev, "{nrm:.2e} {prev:.2e}");
            prev = nrm;
        }
    }

    #[test]
    fn test_smooth() {
        let mesh = rectangle_mesh::<Mesh2d>(1.0, 9, 1.0, 9).random_shuffle();

        let f: Vec<_> = mesh.verts().map(|p| p[0] + 2.0 * p[1]).collect();
        let res = mesh.smooth(GradientMethod::LinearLeastSquares(2), &f);
        for i_vert in 0..mesh.n_verts() {
            assert!(f64::abs(res[i_vert] - f[i_vert]) < 1e-10);
        }

        let f: Vec<_> = mesh.verts().map(|p| p[0] * p[1]).collect();
        let res = mesh.smooth(GradientMethod::LinearLeastSquares(2), &f);
        for i_vert in 0..mesh.n_verts() {
            assert!(f64::abs(res[i_vert] - f[i_vert]) < 1e-2);
        }
    }

    #[test]
    fn test_integrate() {
        let msh = rectangle_mesh::<Mesh2d>(1.0, 10, 2.0, 15).random_shuffle();

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
        let msh: Mesh2d = rectangle_mesh::<Mesh2d>(1.0, 100, 1.0, 100);
        let fname = "rect2d.meshb";
        msh.write_meshb(fname).unwrap();
        let new_msh = Mesh2d::from_meshb(fname).unwrap();

        msh.check_equals(&new_msh, 1e-12).unwrap();

        std::fs::remove_file(fname).unwrap();
    }

    #[test]
    fn test_rcm() {
        let msh = rectangle_mesh::<Mesh2d>(1.0, 100, 1.0, 100).random_shuffle();
        let avg_bandwidth = bandwidth(msh.elems()).1;
        assert!(avg_bandwidth > 1000.0);

        let (msh_rcm, vert_ids, elem_ids, face_ids) = msh.reorder_rcm();
        let avg_bandwidth_rcm = bandwidth(msh_rcm.elems()).1;

        assert!(avg_bandwidth_rcm < 80.0);

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
    fn test_split() {
        let msh = rectangle_mesh::<Mesh2d>(1.0, 2, 1.0, 2).random_shuffle();

        let msh = msh.split();
        assert_eq!(msh.n_verts(), 9);
        assert_eq!(msh.n_faces(), 8);
        assert_eq!(msh.n_elems(), 8);

        let (bdy, _): (BoundaryMesh2d, _) = msh.boundary();
        let area = bdy.gelems().map(|ge| ge.vol()).sum::<f64>();
        assert_delta!(area, 4.0, 1e-10);

        let vol = msh.gelems().map(|ge| ge.vol()).sum::<f64>();
        assert_delta!(vol, 1.0, 1e-10);
    }

    #[test]
    fn test_skewness_2d() {
        let mesh = rectangle_mesh::<Mesh2d>(1.0, 3, 1.0, 3).random_shuffle();

        let all_faces = mesh.all_faces();
        let count = mesh
            .face_skewnesses(&all_faces)
            .map(|(_, _, s)| assert!(s.abs() < 1e-3))
            .count();
        assert_eq!(count, 8);
    }

    #[test]
    fn test_edge_ratio_2d() {
        let mesh = rectangle_mesh::<Mesh2d>(1.0, 3, 1.0, 3).random_shuffle();

        let count = mesh
            .edge_length_ratios()
            .map(|s| assert!((s - std::f64::consts::SQRT_2) < 1e-6))
            .count();
        assert_eq!(count, 8);
    }

    #[test]
    fn test_gamma_2d() {
        let mesh = rectangle_mesh::<Mesh2d>(1.0, 3, 1.0, 3).random_shuffle();

        let count = mesh
            .elem_gammas()
            .map(|s| assert!((s - 0.8284).abs() < 1e-4))
            .count();
        assert_eq!(count, 8);
    }

    #[test]
    fn test_disk() {
        let msh = disk_mesh::<Mesh2d>(3);

        let faces = msh.all_faces();
        msh.check(&faces).unwrap();

        let vol = msh.gelems().map(|ge| ge.vol()).sum::<f64>();
        assert_delta!(vol, 0.25 * std::f64::consts::PI, 6e-3);

        let msh = quadratic_disk_mesh::<QuadraticMesh2d>(3);
        // msh.write_meshb("qdisk.mesh").unwrap();

        let faces = msh.all_faces();
        msh.check(&faces).unwrap();
        assert_delta!(msh.vol(), 0.25 * std::f64::consts::PI, 3e-6);

        let d = AdativeBoundsQuadraticTriangle::element_distortion(&msh);
        let dmax = d.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        // Value computed with gmsh
        assert_delta!(dmax, 1.0 / 0.00122, 0.1);

        // let mut writer = crate::io::VTUFile::from_mesh(&msh);
        // writer.add_cell_data("distorsion", 1, d.iter().copied());
        // writer.export("quadratic_disk.vtu").unwrap();
    }
}
