use crate::{
    geom_elems::GElem,
    mesh::SimplexMesh,
    octree::Octree,
    topo_elems::{Elem, Triangle},
    Idx, Mesh,
};
use log::{info, warn};
use std::{f64::consts::PI, fs::OpenOptions};

/// Read a .stl file (ascii or binary) and return a new SimplexMesh<3, Triangle>
#[must_use]
pub fn read_stl(file_name: &str) -> SimplexMesh<3, Triangle> {
    info!("Read {file_name}");

    let mut file = OpenOptions::new().read(true).open(file_name).unwrap();
    let stl = stl_io::read_stl(&mut file).unwrap();

    let mut coords = Vec::with_capacity(3 * stl.vertices.len());
    coords.extend(
        stl.vertices
            .iter()
            .flat_map(|v| (0..3).map(|i| f64::from(v[i]))),
    );

    let mut elems = Vec::with_capacity(3 * stl.faces.len());
    elems.extend(
        stl.faces
            .iter()
            .flat_map(|v| (0..3).map(|i| v.vertices[i] as Idx)),
    );
    let etags = vec![1; stl.faces.len()];
    let faces = Vec::new();
    let ftags = Vec::new();

    SimplexMesh::<3, Triangle>::new(coords, elems, etags, faces, ftags)
}

/// Reorder a surface mesh that provides a representation of the geometry of the boundary of a
/// volume mesh such that boundary faces are oriented outwards.
/// TODO: find a better name!
pub fn orient_stl<const D: usize, E: Elem>(
    mesh: &SimplexMesh<D, E>,
    stl_mesh: &mut SimplexMesh<D, E::Face>,
) -> (Idx, f64) {
    info!("Orient the boundary mesh");

    let (bdy, _) = mesh.boundary();
    let tree = Octree::new(&bdy);

    let mut dmin = 1.0;
    let mut n_inverted = 0;
    for i_face in 0..stl_mesh.n_elems() {
        let c = stl_mesh.elem_center(i_face);
        let gf = stl_mesh.gelem(i_face);
        let n = gf.normal();
        let i_face_mesh = tree.nearest(&c);
        let gf_mesh = mesh.gface(i_face_mesh);
        let n_mesh = gf_mesh.normal();
        let mut d = n.dot(&n_mesh);
        if d < 0.0 {
            stl_mesh.elems.swap(
                (E::Face::N_VERTS * i_face) as usize,
                (E::Face::N_VERTS * i_face + 1) as usize,
            );
            d = -d;
            n_inverted += 1;
        }
        dmin = f64::min(dmin, d);
    }
    if n_inverted > 0 {
        warn!("{} faces reoriented", n_inverted);
    }
    (n_inverted, f64::acos(dmin) * 180. / PI)
}

#[cfg(test)]
mod tests {
    use std::fs::remove_file;

    use super::orient_stl;
    use crate::{
        mesh_stl::read_stl,
        test_meshes::{test_mesh_3d, write_stl_file},
        Result,
    };

    #[test]
    fn test_stl() -> Result<()> {
        write_stl_file("cube.stl")?;
        let geom = read_stl("cube.stl");
        remove_file("cube.stl")?;

        let v: f64 = geom.elem_vols().sum();
        assert!(f64::abs(v - 6.0) < 1e-10);

        Ok(())
    }

    #[test]
    fn test_reorient() -> Result<()> {
        let mut mesh = test_mesh_3d().split().split();
        mesh.add_boundary_faces();

        write_stl_file("cube1.stl")?;
        let mut geom = read_stl("cube1.stl");
        remove_file("cube1.stl")?;
        geom.elems.swap(4, 5);

        let (n, angle) = orient_stl(&mesh, &mut geom);

        assert_eq!(n, 1);
        assert!(f64::abs(angle - 0.0) < 1e-10);

        let v: f64 = geom.elem_vols().sum();
        assert!(f64::abs(v - 6.0) < 1e-10);
        Ok(())
    }
}
