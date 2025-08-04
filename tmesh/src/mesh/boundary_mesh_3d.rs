//! Boundary of `Mesh3d`
use crate::{
    Result, Vert3d,
    mesh::{GenericMesh, Mesh},
};
use std::fs::OpenOptions;

/// Triangle mesh in 3d
pub type BoundaryMesh3d = GenericMesh<3, 3, 2>;

/// Read a stl file
pub fn read_stl<M: Mesh<3, 3, 2>>(file_name: &str) -> Result<M> {
    let mut file = OpenOptions::new().read(true).open(file_name).unwrap();
    let stl = stl_io::read_stl(&mut file).unwrap();

    let mut verts = Vec::with_capacity(stl.vertices.len());
    verts.extend(
        stl.vertices
            .iter()
            .map(|v| Vert3d::new(f64::from(v[0]), f64::from(v[1]), f64::from(v[2]))),
    );

    let mut elems = Vec::with_capacity(3 * stl.faces.len());
    elems.extend(stl.faces.iter().map(|v| v.vertices));
    let etags = vec![1; stl.faces.len()];
    let faces = Vec::new();
    let ftags = Vec::new();

    Ok(M::new(&verts, &elems, &etags, &faces, &ftags))
}

#[cfg(test)]
mod tests {
    use crate::{
        Vert3d, assert_delta,
        mesh::{BoundaryMesh3d, Mesh, Mesh3d, Simplex, Triangle, box_mesh},
    };
    use rayon::iter::ParallelIterator;

    #[test]
    fn test_box() {
        let msh = box_mesh::<Mesh3d>(1.0, 10, 2.0, 15, 1.0, 20);

        let (mut bdy, _): (BoundaryMesh3d, _) = msh.boundary();

        let faces = bdy.all_faces();
        let tags = bdy.tag_internal_faces(&faces);
        assert_eq!(tags.len(), 12);
        bdy.check(&faces).unwrap();

        let vol = bdy.gelems().map(|ge| Triangle::vol(&ge)).sum::<f64>();
        assert_delta!(vol, 10.0, 1e-12);
    }

    #[test]
    fn test_integrate() {
        let v0 = Vert3d::new(0.0, 0.0, 1.0);
        let v1 = Vert3d::new(0.5, 0.0, 1.0);
        let v2 = Vert3d::new(0.0, 0.5, 1.0);
        let ge = [v0, v1, v2];
        assert_delta!(Triangle::vol(&ge), 0.125, 1e-12);
        let ge = [v1, v0, v2];
        assert_delta!(Triangle::vol(&ge), 0.125, 1e-12);

        let msh = box_mesh::<Mesh3d>(1.0, 10, 2.0, 15, 1.0, 20);

        let f = msh.par_verts().map(|v| v[0]).collect::<Vec<_>>();

        let tag = 3;
        let (bdy, ids): (BoundaryMesh3d, _) = msh.extract_faces(|t| t == tag);
        let f_bdy = ids.iter().map(|&i| f[i]).collect::<Vec<_>>();

        let val = bdy.integrate(&f_bdy, |_| 1.0);
        assert_delta!(val, 1.0, 1e-12);

        let val = bdy.integrate(&f_bdy, |x| x);
        assert_delta!(val, 0.5, 1e-12);

        let nrm = bdy.norm(&f_bdy);
        assert_delta!(nrm, 1.0 / 3.0_f64.sqrt(), 1e-12);
    }
}
