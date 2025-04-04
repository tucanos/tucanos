use crate::{
    mesh::{
        geom_elems::GElem, Elem, Point, QuadraticMesh, QuadraticTriangle, SimplexMesh, Triangle,
    },
    spatialindex::{DefaultObjectIndex, ObjectIndex},
    Idx,
};
use log::{debug, warn};
use std::{f64::consts::PI, fs::OpenOptions};

/// Read a .stl file (ascii or binary) and return a new SimplexMesh<3, Triangle>
#[must_use]
pub fn read_stl(file_name: &str) -> SimplexMesh<3, Triangle> {
    debug!("Read {file_name}");

    let mut file = OpenOptions::new().read(true).open(file_name).unwrap();
    let stl = stl_io::read_stl(&mut file).unwrap();

    let mut verts = Vec::with_capacity(stl.vertices.len());
    verts.extend(
        stl.vertices
            .iter()
            .map(|v| Point::<3>::new(f64::from(v[0]), f64::from(v[1]), f64::from(v[2]))),
    );

    let mut elems = Vec::with_capacity(3 * stl.faces.len());
    elems.extend(stl.faces.iter().map(|v| {
        Triangle::new(
            v.vertices[0] as Idx,
            v.vertices[1] as Idx,
            v.vertices[2] as Idx,
        )
    }));
    let etags = vec![1; stl.faces.len()];
    let faces = Vec::new();
    let ftags = Vec::new();

    SimplexMesh::<3, Triangle>::new(verts, elems, etags, faces, ftags)
}

/// Read a .stl file (ascii or binary) and return a new QuadraticMesh<QuadraticTriangle>
#[must_use]
pub fn read_stl_quadratic(file_name: &str) -> QuadraticMesh<QuadraticTriangle> {
    debug!("Read {file_name}");

    // Ouvrir le fichier STL
    let mut file = OpenOptions::new().read(true).open(file_name).unwrap();
    let stl = stl_io::read_stl(&mut file).unwrap();

    // Lire les sommets
    let mut verts = Vec::with_capacity(stl.vertices.len());
    verts.extend(
        stl.vertices
            .iter()
            .map(|v| Point::<3>::new(f64::from(v[0]), f64::from(v[1]), f64::from(v[2]))),
    );

    // Lire les triangles quadratiques
    let mut tris = Vec::with_capacity(stl.faces.len());
    tris.extend(stl.faces.iter().map(|face| {
        // Assurez-vous que chaque face contient 6 sommets
        assert_eq!(face.vertices.len(), 6, "Each triangle must have 6 vertices");

        // Créer un `QuadraticTriangle` avec les 6 sommets
        QuadraticTriangle::new(
            face.vertices[0] as Idx, // Premier sommet
            face.vertices[1] as Idx, // Deuxième sommet
            face.vertices[2] as Idx, // Troisième sommet
            face.vertices[3] as Idx, // Quatrième sommet (milieu de l'arête 0-1)
            face.vertices[4] as Idx, // Cinquième sommet (milieu de l'arête 1-2)
            face.vertices[5] as Idx, // Sixième sommet (milieu de l'arête 2-0)
        )
    }));

    // Tags pour les triangles et les arêtes
    let tri_tags = vec![1; stl.faces.len()];
    let edgs = Vec::new();
    let edg_tags = Vec::new();

    // Créer et retourner le maillage quadratique
    QuadraticMesh::<QuadraticTriangle>::new(verts, tris, tri_tags, edgs, edg_tags)
}

/// Reorder a surface mesh that provides a representation of the geometry of the boundary of a
/// volume mesh such that boundary faces are oriented outwards.
/// TODO: find a better name!
pub fn orient_stl<const D: usize, E: Elem>(
    mesh: &SimplexMesh<D, E>,
    stl_mesh: &mut SimplexMesh<D, E::Face>,
) -> (Idx, f64) {
    debug!("Orient the boundary mesh");

    let (bdy, _) = mesh.boundary();
    let tree = <DefaultObjectIndex<D> as ObjectIndex<D>>::new(&bdy);
    let mut dmin = 1.0;
    let mut n_inverted = 0;
    let mut new_elems = Vec::with_capacity(stl_mesh.n_elems() as usize);
    for e in stl_mesh.elems() {
        let ge = stl_mesh.gelem(e);
        let c = ge.center();
        let n = ge.normal();
        let i_face_mesh = tree.nearest_elem(&c);
        let f_mesh = mesh.face(i_face_mesh);
        let gf_mesh = mesh.gface(f_mesh);
        let n_mesh = gf_mesh.normal();
        let mut d = n.dot(&n_mesh);
        let mut new_e = e;
        if d < 0.0 {
            new_e[0] = e[1];
            new_e[1] = e[0];
            d = -d;
            n_inverted += 1;
        }
        dmin = f64::min(dmin, d);
        new_elems.push(new_e);
    }

    stl_mesh
        .mut_elems()
        .zip(new_elems)
        .for_each(|(e0, e1)| *e0 = e1);

    if n_inverted > 0 {
        warn!("{} faces reoriented", n_inverted);
    }
    (n_inverted, f64::acos(dmin) * 180. / PI)
}

#[cfg(test)]
mod tests {
    use std::fs::remove_file;

    use super::{orient_stl, read_stl_quadratic};
    use crate::{
        mesh::io::read_stl,
        mesh::test_meshes::{test_mesh_3d, test_mesh_2d_quadratic, write_stl_file},
        mesh::{Triangle, QuadraticTriangle, QuadraticEdge},
        Result,
    };
    use std::fs::File;

    #[test]
    fn test_stl() -> Result<()> {
        write_stl_file("cube.stl")?;
        let geom = read_stl("cube.stl");
        remove_file("cube.stl")?;

        let v: f64 = geom.vol();
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

        let v: f64 = geom.vol();
        assert!(f64::abs(v - 6.0) < 1e-10);

        geom.mut_elems().enumerate().for_each(|(i, e)| {
            if i % 2 == 0 {
                *e = Triangle::new(e[0], e[2], e[1]);
            }
        });

        let (n, angle) = orient_stl(&mesh, &mut geom);

        assert_eq!(n, 6);
        assert!(f64::abs(angle - 0.0) < 1e-10);

        let v: f64 = geom.vol();
        assert!(f64::abs(v - 6.0) < 1e-10);
        Ok(())
    }

    #[test]
    fn test_read_stl_quadratic() {
        // Create a temporary STL file from `test_mesh_2d_quadratic`
        let mesh = test_mesh_2d_quadratic();
        let mut file = File::create("test_quadratic.stl").unwrap();
    

        // Read the STL file back into a quadratic mesh
        let read_mesh = read_stl_quadratic("test_quadratic.stl");

        // Verify the mesh properties
        assert_eq!(read_mesh.n_verts(), mesh.n_verts());
        assert_eq!(read_mesh.n_tris(), mesh.n_tris());
        assert_eq!(read_mesh.n_edges(), mesh.n_edges());

        assert_eq!(read_mesh.vert(0), mesh.vert(0));
        assert_eq!(read_mesh.tri(0), QuadraticTriangle::new(0, 1, 2, 3, 4, 5));
        assert_eq!(read_mesh.edge(0), QuadraticEdge::new(0, 1, 3));

        // Clean up the temporary file
        std::fs::remove_file("test_quadratic.stl").unwrap();
    }
}
