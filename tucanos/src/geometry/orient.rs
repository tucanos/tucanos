use crate::mesh::{SimplexMesh, SubSimplexMesh};
use log::{debug, warn};
use rustc_hash::FxHashSet;
use std::f64::consts::PI;
use tmesh::{
    mesh::{GSimplex, Idx, Mesh, MutMesh, Simplex},
    spatialindex::ObjectIndex,
};

/// Reorder a surface mesh that provides a representation of the geometry of the boundary of a
/// volume mesh such that boundary faces are oriented outwards.
pub fn orient_geometry<T: Idx, const D: usize, C: Simplex<T>>(
    mesh: &SimplexMesh<T, D, C>,
    stl_mesh: &mut SimplexMesh<T, D, C::FACE>,
) -> (T, f64) {
    debug!("Orient the boundary mesh");

    let (bdy, _) = mesh.boundary::<SimplexMesh<T, D, C::FACE>>();
    let tags = bdy.etags().collect::<FxHashSet<_>>();

    let n_elems = stl_mesh.n_elems();
    let mut new_elems = stl_mesh.elems().collect::<Vec<_>>();
    let mut dmin = 1.0;
    let mut n_inverted = T::ZERO;

    for tag in tags {
        let bdy = SubSimplexMesh::new(&bdy, |t| t == tag);
        let tree = ObjectIndex::new(&bdy.mesh);
        for i in 0..n_elems.try_into().unwrap() {
            let t = stl_mesh.etag(i.try_into().unwrap());
            let e = stl_mesh.elem(i.try_into().unwrap());
            if t == tag {
                let ge = stl_mesh.gelem(&e);
                let c = ge.center();
                let n = ge.normal();
                let i_face_mesh = tree.nearest_elem(&c);
                let i_face_mesh = bdy.parent_elem_ids[i_face_mesh].try_into().unwrap();
                let f_mesh = mesh.face(i_face_mesh);
                let t_mesh = mesh.ftag(i_face_mesh);
                assert_eq!(t, t_mesh);
                let gf_mesh = mesh.gface(&f_mesh);
                let n_mesh = gf_mesh.normal();
                let mut d = n.dot(&n_mesh);
                if d < 0.0 {
                    new_elems[i as usize].invert();
                    d = -d;
                    n_inverted += T::ONE;
                }
                dmin = f64::min(dmin, d);
            }
        }
    }

    stl_mesh
        .elems_mut()
        .zip(new_elems)
        .for_each(|(e0, e1)| *e0 = e1);

    if n_inverted > T::ZERO {
        warn!("{n_inverted} / {} faces reoriented", stl_mesh.n_elems());
    }
    (n_inverted, f64::acos(dmin) * 180. / PI)
}

#[cfg(test)]
mod tests {
    use std::fs::remove_file;

    use tmesh::mesh::{Mesh, MutMesh, Triangle, read_stl};

    use super::orient_geometry;
    use crate::{
        Result,
        mesh::{
            SimplexMesh,
            test_meshes::{test_mesh_3d, write_stl_file},
        },
    };

    #[test]
    fn test_stl() -> Result<()> {
        write_stl_file("cube.stl")?;
        let geom: SimplexMesh<u32, 3, Triangle<u32>> = read_stl("cube.stl")?;
        remove_file("cube.stl")?;

        let v: f64 = geom.vol();
        assert!(f64::abs(v - 6.0) < 1e-10);

        Ok(())
    }

    #[test]
    fn test_reorient() -> Result<()> {
        let mut mesh = test_mesh_3d().split().split();
        mesh.fix();
        mesh.ftags_mut().for_each(|t| *t = 1);

        write_stl_file("cube1.stl")?;
        let mut geom: SimplexMesh<u32, 3, Triangle<u32>> = read_stl("cube1.stl")?;
        remove_file("cube1.stl")?;

        let v: f64 = geom.vol();
        assert!(f64::abs(v - 6.0) < 1e-10);

        geom.elems_mut().enumerate().for_each(|(i, e)| {
            if i % 2 == 0 {
                *e = Triangle::from([e[0], e[2], e[1]]);
            }
        });

        let (n, angle) = orient_geometry(&mesh, &mut geom);

        assert_eq!(n, 6);
        assert!(f64::abs(angle - 0.0) < 1e-10);

        let v: f64 = geom.vol();
        assert!(f64::abs(v - 6.0) < 1e-10);
        Ok(())
    }
}
