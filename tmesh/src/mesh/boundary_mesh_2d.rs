//! Boundary of `Mesh2d`
use std::{f64::consts::PI, iter::once};

use crate::{
    Vert2d, Vertex,
    mesh::{Edge, GenericMesh, Idx, Mesh, QuadraticEdge, Simplex},
};

/// Edge mesh in 2d
pub type BoundaryMesh2d = GenericMesh<2, Edge<usize>>;
pub type QuadraticBoundaryMesh2d = GenericMesh<2, QuadraticEdge<usize>>;

/// Create a `Mesh<2, Edge<_>>` of a circle
#[must_use]
pub fn circle_mesh<M: Mesh<2, Edge<impl Idx>>>(r: f64, n: usize) -> M {
    let dtheta = 2.0 * PI / n as f64;

    let mut res = M::empty();
    res.add_verts(
        (0..n).map(|i| r * Vert2d::new((i as f64 * dtheta).cos(), (i as f64 * dtheta).sin())),
    );
    res.add_elems((0..n).map(|i| Edge::new(i, (i + 1) % n)), (0..n).map(|_| 1));

    res
}

/// Create a `Mesh<2, QuadraticEdge<_>>` of a circle
#[must_use]
pub fn quadratic_circle_mesh<M: Mesh<2, QuadraticEdge<impl Idx>>>(r: f64, n: usize) -> M {
    let mut res: M = to_quadratic_edge_mesh(&circle_mesh::<BoundaryMesh2d>(r, n));
    res.verts_mut().for_each(|x| *x *= r / x.norm());

    res
}

#[must_use]
pub fn to_quadratic_edge_mesh<const D: usize, T: Idx, M: Mesh<D, QuadraticEdge<T>>, T2: Idx>(
    msh: &impl Mesh<D, Edge<T2>>,
) -> M {
    let edges = msh.edges();

    let mut res = M::empty();
    res.add_verts(msh.verts());
    let mut new_verts = vec![Vertex::zeros(); edges.len()];
    for (&e, &i) in &edges {
        new_verts[i] = 0.5 * (msh.vert(e.get(0)) + msh.vert(e.get(1)));
    }
    res.add_verts(new_verts.iter().copied());

    for (e, t) in msh.elems().zip(msh.etags()) {
        res.add_elems(
            once(QuadraticEdge::new(
                e.get(0),
                e.get(1),
                edges.get(&e.sorted()).unwrap() + msh.n_verts(),
            )),
            once(t),
        );
    }
    res
}

#[cfg(test)]
mod tests {
    use std::f64::consts::PI;

    use crate::{
        assert_delta,
        mesh::{
            GSimplex, Mesh, Mesh2d, QuadraticBoundaryMesh2d,
            boundary_mesh_2d::{circle_mesh, quadratic_circle_mesh},
            rectangle_mesh,
        },
    };
    use rayon::iter::ParallelIterator;

    use super::BoundaryMesh2d;

    #[test]
    fn test_rectangle() {
        let msh = rectangle_mesh::<Mesh2d>(1.0, 10, 2.0, 20);

        let (mut bdy, ids): (BoundaryMesh2d, _) = msh.boundary();

        let faces = bdy.all_faces();
        let tags = bdy.tag_internal_faces(&faces);
        assert_eq!(tags.len(), 4);
        bdy.check(&faces).unwrap();

        assert_eq!(bdy.n_verts(), 2 * 10 + 2 * 20 - 4);
        assert_eq!(bdy.n_elems(), 2 * 9 + 2 * 19);

        for (i, &j) in ids.iter().enumerate() {
            let pi = bdy.vert(i);
            let pj = msh.vert(j);
            let d = (pj - pi).norm();
            assert!(d < 1e-12);
        }
    }

    #[test]
    fn test_integrate() {
        let msh = rectangle_mesh::<Mesh2d>(1.0, 10, 2.0, 15);

        let f = msh.par_verts().map(|v| v[0]).collect::<Vec<_>>();

        let tag = 1;
        let (bdy, ids): (BoundaryMesh2d, _) = msh.extract_faces(|t| t == tag);
        let f_bdy = ids.iter().map(|&i| f[i]).collect::<Vec<_>>();

        let val = bdy.integrate(&f_bdy, |_| 1.0);
        assert_delta!(val, 1.0, 1e-12);

        let val = bdy.integrate(&f_bdy, |x| x);
        assert_delta!(val, 0.5, 1e-12);

        let nrm = bdy.norm(&f_bdy);
        assert_delta!(nrm, 1.0 / 3.0_f64.sqrt(), 1e-12);
    }

    #[test]
    fn test_circle() {
        let n = 10;

        let msh: BoundaryMesh2d = circle_mesh(1.0, 2 * n);

        let qmsh: QuadraticBoundaryMesh2d = quadratic_circle_mesh(1.0, n);

        assert_delta!(msh.vol(), 2.0 * PI, 0.03);
        assert_delta!(qmsh.vol(), 2.0 * PI, 0.001);

        let d = msh
            .gelems()
            .map(|ge| (ge.center().norm() - 1.0).abs())
            .fold(0.0, f64::max);
        let qd = qmsh
            .gelems()
            .map(|ge| (ge.center().norm() - 1.0).abs())
            .fold(0.0, f64::max);

        assert_delta!(d, 0.0, 0.02);
        assert_delta!(qd, 0.0, 1e-12);
    }
}
