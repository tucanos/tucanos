use log::{debug, warn};
use rustc_hash::FxHashSet;
use std::f64::consts::PI;
use tmesh::{
    mesh::{GSimplex, GenericMesh, Mesh, Simplex, SubMesh},
    spatialindex::ObjectIndex,
};

/// Reorder a surface mesh that provides a representation of the geometry of the boundary of a
/// volume mesh such that boundary faces are oriented outwards.
pub fn orient_geometry<const D: usize, M: Mesh<D>, M2: Mesh<D, C = <M::C as Simplex>::FACE>>(
    mesh: &M,
    stl_mesh: &mut M2,
) -> (usize, f64) {
    debug!("Orient the boundary mesh");

    let (bdy, _) = mesh.boundary::<GenericMesh<D, <M::C as Simplex>::FACE>>();
    let tags = bdy.etags().collect::<FxHashSet<_>>();

    let mut new_elems = stl_mesh.elems().collect::<Vec<_>>();
    let mut dmin = 1.0;
    let mut n_inverted = 0;

    for tag in tags {
        let bdy = SubMesh::new(&bdy, |t| t == tag);
        let tree = ObjectIndex::new(&bdy.mesh);
        for (i, (e, t)) in stl_mesh.elems().zip(stl_mesh.etags()).enumerate() {
            if t == tag {
                let ge = stl_mesh.gelem(&e);
                let c = ge.center();
                let n = ge.normal().normalize();
                let i_face_mesh = tree.nearest_elem(&c);
                let i_face_mesh = bdy.parent_elem_ids[i_face_mesh];
                let f_mesh = mesh.face(i_face_mesh);
                let t_mesh = mesh.ftag(i_face_mesh);
                assert_eq!(t, t_mesh);
                let gf_mesh = mesh.gface(&f_mesh);
                let n_mesh = gf_mesh.normal().normalize();
                let mut d = n.dot(&n_mesh);
                if d < 0.0 {
                    new_elems[i].invert();
                    d = -d;
                    n_inverted += 1;
                }
                dmin = f64::min(dmin, d);
            }
        }
    }

    stl_mesh
        .elems_mut()
        .zip(new_elems)
        .for_each(|(e0, e1)| *e0 = e1);

    if n_inverted > 0 {
        warn!("{n_inverted} / {} faces reoriented", stl_mesh.n_elems());
    }
    (n_inverted, f64::acos(dmin) * 180. / PI)
}

#[cfg(test)]
mod tests {
    use std::fs::remove_file;

    use tmesh::mesh::{BoundaryMesh3d, Mesh, Mesh3d, Simplex, box_mesh, read_stl};

    use super::orient_geometry;
    use crate::{Result, mesh::test_meshes::write_stl_file};

    #[test]
    fn test_stl() -> Result<()> {
        write_stl_file("cube.stl")?;
        let geom: BoundaryMesh3d = read_stl("cube.stl")?;
        remove_file("cube.stl")?;

        let v: f64 = geom.vol();
        assert!(f64::abs(v - 6.0) < 1e-10);

        Ok(())
    }

    #[test]
    fn test_reorient() -> Result<()> {
        let mut mesh: Mesh3d = box_mesh(1.0, 5, 1.0, 5, 1.0, 5);
        mesh.fix().unwrap();
        mesh.ftags_mut().for_each(|t| *t = 1);

        write_stl_file("cube1.stl")?;
        let mut geom: BoundaryMesh3d = read_stl("cube1.stl")?;
        remove_file("cube1.stl")?;

        let v: f64 = geom.vol();
        assert!(f64::abs(v - 6.0) < 1e-10);

        geom.elems_mut().enumerate().for_each(|(i, e)| {
            if i % 2 == 0 {
                e.invert();
            }
        });

        let (n, angle) = orient_geometry(&mesh, &mut geom);

        assert_eq!(n, 6);
        assert!(f64::abs(angle - 0.0) < 1e-10, "angle={angle}");

        let v: f64 = geom.vol();
        assert!(f64::abs(v - 6.0) < 1e-10);
        Ok(())
    }
}
