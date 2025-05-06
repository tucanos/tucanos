use crate::{mesh::Mesh, Tag, Tetrahedron, Triangle, Vert3d};

pub fn box_mesh<M: Mesh<3, 4, 3>>(lx: f64, nx: usize, ly: f64, ny: usize, lz: f64, nz: usize) -> M {
    let dx = lx / (nx as f64 - 1.);
    let x_1d = (0..nx).map(|i| i as f64 * dx).collect::<Vec<_>>();

    let dy = ly / (ny as f64 - 1.);
    let y_1d = (0..ny).map(|i| i as f64 * dy).collect::<Vec<_>>();

    let dz = lz / (nz as f64 - 1.);
    let z_1d = (0..nz).map(|i| i as f64 * dz).collect::<Vec<_>>();

    nonuniform_box_mesh(&x_1d, &y_1d, &z_1d)
}

pub fn nonuniform_box_mesh<M: Mesh<3, 4, 3>>(x: &[f64], y: &[f64], z: &[f64]) -> M {
    let nx = x.len();
    let ny = y.len();
    let nz = z.len();

    let idx = |i, j, k| i + j * nx + k * nx * ny;

    let mut verts = vec![Vert3d::zeros(); nx * ny * nz];
    for (i, &x) in x.iter().enumerate() {
        for (j, &y) in y.iter().enumerate() {
            for (k, &z) in z.iter().enumerate() {
                verts[idx(i, j, k)] = Vert3d::new(x, y, z);
            }
        }
    }

    let mut hexas = Vec::with_capacity((nx - 1) * (ny - 1) * (nz - 1));
    let mut etags = Vec::with_capacity(hexas.capacity());
    for i in 0..nx - 1 {
        for j in 0..ny - 1 {
            for k in 0..nz - 1 {
                hexas.push([
                    idx(i, j, k),
                    idx(i + 1, j, k),
                    idx(i + 1, j + 1, k),
                    idx(i, j + 1, k),
                    idx(i, j, k + 1),
                    idx(i + 1, j, k + 1),
                    idx(i + 1, j + 1, k + 1),
                    idx(i, j + 1, k + 1),
                ]);
                etags.push(1);
            }
        }
    }

    let mut quads = Vec::with_capacity(
        2 * (nx - 1) * (ny - 1) + 2 * (nx - 1) * (nz - 1) + 2 * (nz - 1) * (ny - 1),
    );
    let mut ftags = Vec::with_capacity(quads.capacity());

    for i in 0..nx - 1 {
        for j in 0..ny - 1 {
            let k = 0;
            quads.push([
                idx(i, j, k),
                idx(i, j + 1, k),
                idx(i + 1, j + 1, k),
                idx(i + 1, j, k),
            ]);
            ftags.push(1);
            let k = nz - 1;
            quads.push([
                idx(i, j, k),
                idx(i + 1, j, k),
                idx(i + 1, j + 1, k),
                idx(i, j + 1, k),
            ]);
            ftags.push(2);
        }
    }

    for i in 0..nx - 1 {
        for k in 0..nz - 1 {
            let j = 0;
            quads.push([
                idx(i, j, k),
                idx(i + 1, j, k),
                idx(i + 1, j, k + 1),
                idx(i, j, k + 1),
            ]);
            ftags.push(3);
            let j = ny - 1;
            quads.push([
                idx(i, j, k),
                idx(i, j, k + 1),
                idx(i + 1, j, k + 1),
                idx(i + 1, j, k),
            ]);
            ftags.push(4);
        }
    }

    for j in 0..ny - 1 {
        for k in 0..nz - 1 {
            let i = 0;
            quads.push([
                idx(i, j, k),
                idx(i, j, k + 1),
                idx(i, j + 1, k + 1),
                idx(i, j + 1, k),
            ]);
            ftags.push(5);
            let i = nx - 1;
            quads.push([
                idx(i, j, k),
                idx(i, j + 1, k),
                idx(i, j + 1, k + 1),
                idx(i, j, k + 1),
            ]);
            ftags.push(6);
        }
    }

    let mut res = M::empty();
    res.add_verts(verts.iter().cloned());
    res.add_hexahedra(hexas.iter().cloned(), etags.iter().cloned());
    res.add_quadrangles(quads.iter().cloned(), ftags.iter().cloned());
    // let faces = res.compute_faces();
    // res.fix_orientation(&faces);
    res
}

pub struct Mesh3d {
    verts: Vec<Vert3d>,
    elems: Vec<Tetrahedron>,
    etags: Vec<Tag>,
    faces: Vec<Triangle>,
    ftags: Vec<Tag>,
}

impl Mesh3d {
    pub fn new(
        verts: Vec<Vert3d>,
        elems: Vec<Tetrahedron>,
        etags: Vec<Tag>,
        faces: Vec<Triangle>,
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

impl Mesh<3, 4, 3> for Mesh3d {
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

    fn elem(&self, i: usize) -> &Tetrahedron {
        &self.elems[i]
    }

    fn etag(&self, i: usize) -> Tag {
        self.etags[i]
    }

    fn add_elems<I1: ExactSizeIterator<Item = Tetrahedron>, I2: ExactSizeIterator<Item = Tag>>(
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

    fn add_elems_and_tags<I: ExactSizeIterator<Item = (Tetrahedron, Tag)>>(
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
        self.elems[i] = [e[1], e[0], e[2], e[3]];
    }

    fn n_faces(&self) -> usize {
        self.faces.len()
    }

    fn face(&self, i: usize) -> &Triangle {
        &self.faces[i]
    }

    fn ftag(&self, i: usize) -> Tag {
        self.ftags[i]
    }

    fn add_faces<I1: ExactSizeIterator<Item = Triangle>, I2: ExactSizeIterator<Item = Tag>>(
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

    fn add_faces_and_tags<I: ExactSizeIterator<Item = (Triangle, Tag)>>(
        &mut self,
        faces_and_tags: I,
    ) {
        self.faces.reserve(faces_and_tags.len());
        self.ftags.reserve(faces_and_tags.len());
        for (e, t) in faces_and_tags {
            self.faces.push(e);
            self.ftags.push(t);
        }
    }

    fn invert_face(&mut self, i: usize) {
        let f = self.faces[i];
        self.faces[i] = [f[1], f[0], f[2]];
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        assert_delta,
        mesh::{bandwidth, cell_center, Mesh},
        mesh_3d::{box_mesh, Mesh3d},
        simplices::Simplex,
        Tetrahedron, Vert3d,
    };
    use rayon::iter::ParallelIterator;

    #[test]
    fn test_box() {
        let msh = box_mesh::<Mesh3d>(1.0, 2, 1.0, 2, 1.0, 2).random_shuffle();

        let faces = msh.compute_faces();
        msh.check(&faces).unwrap();

        let vol = msh.seq_gelems().map(Tetrahedron::vol).sum::<f64>();
        assert_delta!(vol, 1.0, 1e-12);
    }

    #[test]
    fn test_gradient() {
        let grad = Vert3d::new(9.8, 7.6, 5.4);
        let msh = box_mesh::<Mesh3d>(1.0, 10, 1.0, 15, 1.0, 20).random_shuffle();
        let f = msh
            .verts()
            .map(|v| grad[0] * v[0] + grad[1] * v[1] + grad[2] * v[2])
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
        let v0 = Vert3d::new(0.0, 0.0, 0.0);
        let v1 = Vert3d::new(1.0, 0.0, 0.0);
        let v2 = Vert3d::new(0.0, 1.0, 0.0);
        let v3 = Vert3d::new(0.0, 0.0, 1.0);
        let ge = [&v0, &v1, &v2, &v3];
        assert_delta!(Tetrahedron::vol(ge), 1.0 / 6.0, 1e-12);
        let ge = [&v0, &v2, &v1, &v3];
        assert_delta!(Tetrahedron::vol(ge), -1.0 / 6.0, 1e-12);

        let msh = box_mesh::<Mesh3d>(1.0, 10, 2.0, 15, 1.0, 20).random_shuffle();

        let vol = msh.gelems().map(Tetrahedron::vol).sum::<f64>();
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
        let msh = box_mesh::<Mesh3d>(1.0, 10, 1.0, 10, 1.0, 10);
        let fname = "box3d.meshb";
        msh.write_meshb(fname).unwrap();
        let new_msh = Mesh3d::from_meshb(fname).unwrap();

        msh.check_equals(&new_msh, 1e-12).unwrap();

        std::fs::remove_file(fname).unwrap();
    }

    #[test]
    fn test_rcm() {
        let msh = box_mesh::<Mesh3d>(1.0, 20, 1.0, 20, 1.0, 20).random_shuffle();
        let avg_bandwidth = bandwidth(msh.seq_elems().cloned()).1;
        assert!(avg_bandwidth > 1000.0);

        let (msh_rcm, vert_ids, elem_ids, face_ids) = msh.reorder_rcm();
        let avg_bandwidth_rcm = bandwidth(msh_rcm.seq_elems().cloned()).1;

        assert!(avg_bandwidth_rcm < 320.0);

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
