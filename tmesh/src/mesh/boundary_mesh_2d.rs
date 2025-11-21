//! Boundary of `Mesh2d`
use crate::mesh::{Edge, GenericMesh};

/// Edge mesh in 2d
pub type BoundaryMesh2d = GenericMesh<2, Edge<usize>>;

#[cfg(test)]
mod tests {
    use crate::{
        assert_delta,
        mesh::{Mesh, Mesh2d, rectangle_mesh},
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
}
