use crate::{mesh::Mesh, Edge, Tag, Triangle, Vert2d};

pub fn rectangle_mesh<M: Mesh<2, 3, 2>>(lx: f64, nx: usize, ly: f64, ny: usize) -> M {
    let dx = lx / (nx as f64 - 1.);
    let x_1d = (0..nx).map(|i| i as f64 * dx).collect::<Vec<_>>();

    let dy = ly / (ny as f64 - 1.);
    let y_1d = (0..ny).map(|i| i as f64 * dy).collect::<Vec<_>>();

    nonuniform_rectangle_mesh(&x_1d, &y_1d)
}

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

    let mut elems = Vec::with_capacity(2 * (nx - 1) * (ny - 1));
    let mut etags = Vec::with_capacity(2 * (nx - 1) * (ny - 1));
    for i in 0..nx - 1 {
        for j in 0..ny - 1 {
            elems.push([idx(i, j), idx(i + 1, j), idx(i + 1, j + 1)]);
            etags.push(1);
            elems.push([idx(i, j), idx(i + 1, j + 1), idx(i, j + 1)]);
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
    res.add_elems(elems.iter().cloned(), etags.iter().cloned());
    res.add_faces(faces.iter().cloned(), ftags.iter().cloned());
    let faces = res.compute_faces();
    res.fix_orientation(&faces);
    res
}

pub struct Mesh2d {
    verts: Vec<Vert2d>,
    elems: Vec<Triangle>,
    etags: Vec<Tag>,
    faces: Vec<Edge>,
    ftags: Vec<Tag>,
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

    fn vol(v: [&Vert2d; 3]) -> f64 {
        let e1 = v[1] - v[0];
        let e2 = v[2] - v[0];

        0.5 * (e1[0] * e2[1] - e1[1] * e2[0])
    }

    fn normal(v: [&Vert2d; 2]) -> Vert2d {
        Vert2d::new(v[1][1] - v[0][1], v[0][0] - v[1][0])
    }

    fn radius(v: [&Vert2d; 3]) -> f64 {
        let a = (v[2] - v[1]).norm();
        let b = (v[2] - v[0]).norm();
        let c = (v[1] - v[0]).norm();
        let s = 0.5 * (a + b + c);
        ((s - a) * (s - b) * (s - c) / s).sqrt()
    }

    fn elem_to_faces() -> Vec<Edge> {
        vec![[0, 1], [1, 2], [2, 0]]
    }

    fn elem_to_edges() -> Vec<Edge> {
        vec![[0, 1], [1, 2], [2, 0]]
    }

    fn quadrature(&self) -> (Vec<f64>, Vec<Vec<f64>>) {
        let weights = vec![1. / 3., 1. / 3., 1. / 3.];
        let pts = vec![
            vec![2. / 3., 1. / 6.],
            vec![1. / 6., 2. / 3.],
            vec![1. / 6., 1. / 6.],
        ];
        (weights, pts)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        assert_delta,
        mesh::{bandwidth, cell_center, Mesh},
        mesh_2d::{rectangle_mesh, Mesh2d},
        Vert2d,
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
        let msh = rectangle_mesh::<Mesh2d>(1.0, 10, 2.0, 15);

        let faces = msh.compute_faces();
        msh.check(&faces).unwrap();

        let edgs = msh.compute_edges();
        assert_eq!(edgs.len(), 9 * 15 + 10 * 14 + 9 * 14);

        let vol = msh.seq_gelems().map(Mesh2d::vol).sum::<f64>();
        assert!((vol - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_gradient() {
        let grad = Vert2d::new(9.8, 7.6);
        let msh = rectangle_mesh::<Mesh2d>(1.0, 10, 1.0, 10);
        let f = msh
            .verts()
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
        assert_delta!(Mesh2d::vol(ge), 0.5, 1e-12);
        let ge = [&v0, &v2, &v1];
        assert_delta!(Mesh2d::vol(ge), -0.5, 1e-12);

        let msh = rectangle_mesh::<Mesh2d>(1.0, 10, 2.0, 15);

        let vol = msh.gelems().map(Mesh2d::vol).sum::<f64>();
        assert_delta!(vol, 2.0, 1e-12);

        let f = msh.verts().map(|v| v[0]).collect::<Vec<_>>();

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

        for (v0, v1) in msh.seq_verts().zip(new_msh.seq_verts()) {
            assert!((v0 - v1).norm() < 1e-12);
        }

        for (e0, e1) in msh.seq_elems().zip(new_msh.seq_elems()) {
            assert_eq!(e0, e1);
        }

        for (t0, t1) in msh.seq_etags().zip(new_msh.seq_etags()) {
            assert_eq!(t0, t1);
        }

        for (e0, e1) in msh.seq_faces().zip(new_msh.seq_faces()) {
            assert_eq!(e0, e1);
        }

        for (t0, t1) in msh.seq_ftags().zip(new_msh.seq_ftags()) {
            assert_eq!(t0, t1);
        }

        std::fs::remove_file(fname).unwrap();
    }

    #[test]
    fn test_rcm() {
        let msh: Mesh2d = rectangle_mesh::<Mesh2d>(1.0, 100, 1.0, 100);
        let avg_bandwidth = bandwidth(msh.seq_elems().cloned()).1;

        let (msh_rcm, vert_ids, elem_ids, face_ids) = msh.reorder_rcm();
        let avg_bandwidth_rcm = bandwidth(msh_rcm.seq_elems().cloned()).1;

        assert!(avg_bandwidth_rcm < avg_bandwidth);

        for (i, v) in msh_rcm.seq_verts().enumerate() {
            let other = msh.vert(vert_ids[i]);
            assert!((v - other).norm() < 1e-12);
        }

        for (i, v) in msh_rcm.seq_gelems().enumerate() {
            let v = cell_center(v);
            let other = msh.gelem(msh.elem(elem_ids[i]));
            let other = cell_center(other);
            assert!((v - other).norm() < 1e-12);
        }

        for (i, tag) in msh_rcm.seq_etags().enumerate() {
            let other = msh.etag(elem_ids[i]);
            assert_eq!(tag, other);
        }

        for (i, v) in msh_rcm.seq_gfaces().enumerate() {
            let v = cell_center(v);
            let other = msh.gface(msh.face(face_ids[i]));
            let other = cell_center(other);
            assert!((v - other).norm() < 1e-12);
        }

        for (i, tag) in msh_rcm.seq_ftags().enumerate() {
            let other = msh.ftag(face_ids[i]);
            assert_eq!(tag, other);
        }

        msh_rcm.check(&msh_rcm.compute_faces()).unwrap();
    }
}
