//! Boundary of `Mesh3d`
use crate::{mesh::Mesh, Edge, Tag, Triangle, Vert3d};

/// Triangle mesh in 3d
pub struct BoundaryMesh3d {
    verts: Vec<Vert3d>,
    elems: Vec<Triangle>,
    etags: Vec<Tag>,
    faces: Vec<Edge>,
    ftags: Vec<Tag>,
}

impl BoundaryMesh3d {
    /// Create a new mesh from coordinates, connectivities and tags
    pub fn new(
        verts: Vec<Vert3d>,
        elems: Vec<Triangle>,
        etags: Vec<Tag>,
        faces: Vec<Edge>,
        ftags: Vec<Tag>,
    ) -> Self {
        Self {
            verts,
            elems,
            etags,
            faces,
            ftags,
        }
    }
}

impl Mesh<3, 3, 2> for BoundaryMesh3d {
    fn empty() -> Self {
        Self {
            verts: Vec::new(),
            elems: Vec::new(),
            etags: Vec::new(),
            faces: Vec::new(),
            ftags: Vec::new(),
        }
    }

    fn n_verts(&self) -> usize {
        self.verts.len()
    }

    fn vert(&self, i: usize) -> &Vert3d {
        &self.verts[i]
    }

    fn add_verts<I: ExactSizeIterator<Item = Vert3d>>(&mut self, v: I) {
        self.verts.extend(v);
    }

    fn n_elems(&self) -> usize {
        self.elems.len()
    }

    fn elem(&self, i: usize) -> &Triangle {
        &self.elems[i]
    }

    fn etag(&self, i: usize) -> Tag {
        self.etags[i]
    }

    fn add_elems<I1: ExactSizeIterator<Item = Triangle>, I2: ExactSizeIterator<Item = Tag>>(
        &mut self,
        elems: I1,
        etags: I2,
    ) {
        self.elems.extend(elems);
        self.etags.extend(etags);
    }

    fn clear_elems(&mut self) {
        self.elems.clear();
        self.etags.clear();
    }

    fn add_elems_and_tags<I: ExactSizeIterator<Item = (Triangle, Tag)>>(
        &mut self,
        elems_and_tags: I,
    ) {
        self.elems.reserve(elems_and_tags.len());
        self.etags.reserve(elems_and_tags.len());
        for (e, t) in elems_and_tags {
            self.elems.push(e);
            self.etags.push(t);
        }
    }

    fn invert_elem(&mut self, i: usize) {
        let e = self.elems[i];
        self.elems[i] = [e[1], e[0], e[2]];
    }

    fn n_faces(&self) -> usize {
        self.faces.len()
    }

    fn face(&self, i: usize) -> &Edge {
        &self.faces[i]
    }

    fn ftag(&self, i: usize) -> Tag {
        self.ftags[i]
    }

    fn add_faces<I1: ExactSizeIterator<Item = Edge>, I2: ExactSizeIterator<Item = Tag>>(
        &mut self,
        faces: I1,
        ftags: I2,
    ) {
        self.faces.extend(faces);
        self.ftags.extend(ftags);
    }

    fn clear_faces(&mut self) {
        self.faces.clear();
        self.ftags.clear();
    }

    fn add_faces_and_tags<I: ExactSizeIterator<Item = (Edge, Tag)>>(&mut self, faces_and_tags: I) {
        self.faces.reserve(faces_and_tags.len());
        self.ftags.reserve(faces_and_tags.len());
        for (e, t) in faces_and_tags {
            self.faces.push(e);
            self.ftags.push(t);
        }
    }

    fn invert_face(&mut self, _i: usize) {
        unreachable!("No normal for BoundaryMesh3d")
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        assert_delta,
        boundary_mesh_3d::BoundaryMesh3d,
        mesh::Mesh,
        mesh_3d::{box_mesh, Mesh3d},
        simplices::Simplex,
        Triangle, Vert3d,
    };
    use rayon::iter::ParallelIterator;

    #[test]
    fn test_box() {
        let msh = box_mesh::<Mesh3d>(1.0, 10, 2.0, 15, 1.0, 20);

        let (mut bdy, _): (BoundaryMesh3d, _) = msh.boundary();

        let faces = bdy.compute_faces();
        let tags = bdy.tag_internal_faces(&faces);
        assert_eq!(tags.len(), 12);
        bdy.check(&faces).unwrap();

        let vol = bdy.gelems().map(Triangle::vol).sum::<f64>();
        assert_delta!(vol, 10.0, 1e-12);
    }

    #[test]
    fn test_integrate() {
        let v0 = Vert3d::new(0.0, 0.0, 1.0);
        let v1 = Vert3d::new(0.5, 0.0, 1.0);
        let v2 = Vert3d::new(0.0, 0.5, 1.0);
        let ge = [&v0, &v1, &v2];
        assert_delta!(Triangle::vol(ge), 0.125, 1e-12);
        let ge = [&v1, &v0, &v2];
        assert_delta!(Triangle::vol(ge), 0.125, 1e-12);

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
