//! Triangle meshes in 2d
use crate::{mesh::Mesh, Edge, Tag, Triangle, Vert2d};

/// Create a `Mesh<2, 3, 2>` of a `lx` by `ly` rectangle by splitting a `nx` by `ny`
/// uniform structured grid
pub fn rectangle_mesh<M: Mesh<2, 3, 2>>(lx: f64, nx: usize, ly: f64, ny: usize) -> M {
    let dx = lx / (nx as f64 - 1.);
    let x_1d = (0..nx).map(|i| i as f64 * dx).collect::<Vec<_>>();

    let dy = ly / (ny as f64 - 1.);
    let y_1d = (0..ny).map(|i| i as f64 * dy).collect::<Vec<_>>();

    nonuniform_rectangle_mesh(&x_1d, &y_1d)
}

/// Create a `Mesh<2, 3, 2>` of rectangle by splitting a structured grid
pub fn nonuniform_rectangle_mesh<M: Mesh<2, 3, 2>>(x: &[f64], y: &[f64]) -> M {
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
            quads.push([idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)]);
            etags.push(1);
        }
    }

    let mut faces = Vec::with_capacity(2 * (nx - 1 + ny - 1));
    let mut ftags = Vec::with_capacity(2 * (nx - 1 + ny - 1));

    for i in 0..nx - 1 {
        faces.push([idx(i, 0), idx(i + 1, 0)]);
        ftags.push(1);
        faces.push([idx(i + 1, ny - 1), idx(i, ny - 1)]);
        ftags.push(3);
    }

    for j in 0..ny - 1 {
        faces.push([idx(nx - 1, j), idx(nx - 1, j + 1)]);
        ftags.push(2);
        faces.push([idx(0, j + 1), idx(0, j)]);
        ftags.push(4);
    }

    let mut res = M::empty();
    res.add_verts(verts.iter().cloned());
    res.add_quadrangles(quads.iter().cloned(), etags.iter().cloned());
    res.add_faces(faces.iter().cloned(), ftags.iter().cloned());
    let faces = res.compute_faces();
    res.fix_orientation(&faces);
    res
}

/// Triangle mesh in 2d
pub struct Mesh2d {
    verts: Vec<Vert2d>,
    elems: Vec<Triangle>,
    etags: Vec<Tag>,
    faces: Vec<Edge>,
    ftags: Vec<Tag>,
}

impl Mesh2d {
    /// Create a new mesh from coordinates, connectivities and tags
    pub fn new(
        verts: Vec<Vert2d>,
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

impl Mesh<2, 3, 2> for Mesh2d {
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

    fn vert(&self, i: usize) -> &Vert2d {
        &self.verts[i]
    }

    fn add_verts<I: ExactSizeIterator<Item = Vert2d>>(&mut self, v: I) {
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

    fn invert_face(&mut self, i: usize) {
        let f = self.faces[i];
        self.faces[i] = [f[1], f[0]];
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        assert_delta,
        mesh::{bandwidth, cell_center, Mesh},
        mesh_2d::{rectangle_mesh, Mesh2d},
        simplices::Simplex,
        Triangle, Vert2d,
    };
    use rayon::iter::ParallelIterator;

    #[test]
    fn test_2d_simple_1() {
        let msh = rectangle_mesh::<Mesh2d>(1.0, 2, 1.0, 2);

        let faces = msh.compute_faces();
        msh.check(&faces).unwrap();

        let edgs = msh.compute_edges();
        assert_eq!(edgs.len(), 5, "{edgs:?}");
        assert!(edgs.contains_key(&[0, 1]));
        assert!(edgs.contains_key(&[2, 3]));
        assert!(edgs.contains_key(&[0, 2]));
        assert!(edgs.contains_key(&[1, 3]));
        assert!(edgs.contains_key(&[0, 3]));

        let faces = msh.compute_faces();
        assert_eq!(faces.len(), 5);
        assert!(faces.contains_key(&[0, 1]));
        assert!(faces.contains_key(&[2, 3]));
        assert!(faces.contains_key(&[0, 2]));
        assert!(faces.contains_key(&[1, 3]));
        assert!(faces.contains_key(&[0, 3]));
    }

    #[test]
    fn test_2d_simple_2() {
        let msh = rectangle_mesh::<Mesh2d>(1.0, 3, 1.0, 2);

        let faces = msh.compute_faces();
        msh.check(&faces).unwrap();

        let edgs = msh.compute_edges();
        assert_eq!(edgs.len(), 9);
        assert!(edgs.contains_key(&[0, 1]));
        assert!(edgs.contains_key(&[1, 2]));
        assert!(edgs.contains_key(&[3, 4]));
        assert!(edgs.contains_key(&[4, 5]));
        assert!(edgs.contains_key(&[0, 3]));
        assert!(edgs.contains_key(&[1, 4]));
        assert!(edgs.contains_key(&[2, 5]));
        assert!(edgs.contains_key(&[0, 4]));
        assert!(edgs.contains_key(&[1, 5]));
    }

    #[test]
    fn test_2d_rect() {
        let msh = rectangle_mesh::<Mesh2d>(1.0, 10, 2.0, 15).random_shuffle();

        let faces = msh.compute_faces();
        msh.check(&faces).unwrap();

        let edgs = msh.compute_edges();
        assert_eq!(edgs.len(), 9 * 15 + 10 * 14 + 9 * 14);

        let vol = msh.gelems().map(Triangle::vol).sum::<f64>();
        assert!((vol - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_gradient() {
        let grad = Vert2d::new(9.8, 7.6);
        let msh = rectangle_mesh::<Mesh2d>(1.0, 10, 1.0, 10).random_shuffle();
        let f = msh
            .par_verts()
            .map(|v| grad[0] * v[0] + grad[1] * v[1])
            .collect::<Vec<_>>();
        let v2v = msh.compute_vertex_to_vertices();
        let gradient = msh.gradient(&v2v, 1, &f).collect::<Vec<_>>();

        for &x in &gradient {
            let err = (x - grad).norm();
            assert!(err < 1e-10, "{x:?}");
        }
    }

    #[test]
    fn test_integrate() {
        let v0 = Vert2d::new(0.0, 0.0);
        let v1 = Vert2d::new(1.0, 0.0);
        let v2 = Vert2d::new(0.0, 1.0);
        let ge = [&v0, &v1, &v2];
        assert_delta!(Triangle::vol(ge), 0.5, 1e-12);
        let ge = [&v0, &v2, &v1];
        assert_delta!(Triangle::vol(ge), -0.5, 1e-12);

        let msh = rectangle_mesh::<Mesh2d>(1.0, 10, 2.0, 15).random_shuffle();

        let vol = msh.par_gelems().map(Triangle::vol).sum::<f64>();
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
        let avg_bandwidth = bandwidth(msh.elems().cloned()).1;
        assert!(avg_bandwidth > 1000.0);

        let (msh_rcm, vert_ids, elem_ids, face_ids) = msh.reorder_rcm();
        let avg_bandwidth_rcm = bandwidth(msh_rcm.elems().cloned()).1;

        assert!(avg_bandwidth_rcm < 80.0);

        for (i, v) in msh_rcm.verts().enumerate() {
            let other = msh.vert(vert_ids[i]);
            assert!((v - other).norm() < 1e-12);
        }

        for (i, v) in msh_rcm.gelems().enumerate() {
            let v = cell_center(v);
            let other = msh.gelem(msh.elem(elem_ids[i]));
            let other = cell_center(other);
            assert!((v - other).norm() < 1e-12);
        }

        for (i, tag) in msh_rcm.etags().enumerate() {
            let other = msh.etag(elem_ids[i]);
            assert_eq!(tag, other);
        }

        for (i, v) in msh_rcm.gfaces().enumerate() {
            let v = cell_center(v);
            let other = msh.gface(msh.face(face_ids[i]));
            let other = cell_center(other);
            assert!((v - other).norm() < 1e-12);
        }

        for (i, tag) in msh_rcm.ftags().enumerate() {
            let other = msh.ftag(face_ids[i]);
            assert_eq!(tag, other);
        }

        msh_rcm.check(&msh_rcm.compute_faces()).unwrap();
    }
}
