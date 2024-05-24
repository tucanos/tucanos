use crate::{
    mesh::{Point, SimplexMesh},
    topo_elems::Elem,
    Idx,
};
use lindel::Lineariseable;
use log::debug;

fn get_indices<F: Fn(usize) -> usize>(n: usize, comp: F) -> Vec<Idx> {
    let mut indices = Vec::with_capacity(n);
    indices.extend(0..n);
    indices.sort_by_key(|i| comp(*i));
    let mut new_indices = vec![0; n];
    for i in 0..n {
        new_indices[indices[i]] = i as Idx;
    }
    new_indices
}

pub fn hilbert_indices<const D: usize, I: ExactSizeIterator<Item = Point<D>>>(
    bb: (Point<D>, Point<D>),
    verts: I,
) -> Vec<Idx> {
    // bounding box
    let (mini, maxi) = bb;

    let n = verts.len();
    // Hilbert index
    let order = 16;
    let scale = usize::pow(2, order) as f64 - 1.0;
    let hilbert = |x: Point<D>| {
        let mut tmp = [0; 3];
        for j in 0..D {
            tmp[j] = (scale * (x[j] - mini[j]) / (maxi[j] - mini[j])).round() as u16;
        }
        tmp.hilbert_index() as usize
    };

    let hilbert_ids = verts.map(hilbert).collect::<Vec<_>>();

    get_indices(n, |i| hilbert_ids[i])
}

impl<const D: usize, E: Elem> SimplexMesh<D, E> {
    /// Reorder the mesh vertices
    /// Vertex data is updated accordingly, but edges, vertex-to-vertex connections
    /// and vertex volumes are reset to None
    /// TODO: update the fields instead of setting them to None and avoid recomputing
    pub fn reorder_vertices(&mut self, new_indices: &[Idx]) {
        debug!("Reordering the vertices");

        let n = self.n_verts() as usize;
        assert_eq!(new_indices.len(), n);

        let mut new_verts = vec![Point::<D>::zeros(); self.n_verts() as usize];
        for (i_old, &i_new) in new_indices.iter().enumerate() {
            new_verts[i_new as usize] = self.vert(i_old as Idx);
        }
        self.mut_verts()
            .zip(new_verts.iter())
            .for_each(|(p0, p1)| *p0 = *p1);

        self.mut_elems()
            .for_each(|e| *e = E::from_iter(e.iter().map(|&i| new_indices[i as usize])));
        self.mut_faces()
            .for_each(|f| *f = E::Face::from_iter(f.iter().map(|&i| new_indices[i as usize])));

        self.clear_vertex_to_elems();
        self.clear_vertex_to_vertices();
        self.clear_edges();
        self.clear_volumes();
    }

    /// Reorder the mesh elements
    /// Element data is updated accordingly, face-to-element and element-to-element connections
    /// and element volumes are reset to None
    /// TODO: update the fields instead of setting them to None and avoid recomputing
    pub fn reorder_elems(&mut self, new_indices: &[Idx]) {
        debug!("Reordering the elements");
        let n = self.n_elems() as usize;
        assert_eq!(new_indices.len(), n);

        let mut elems = vec![E::default(); n];
        let mut etags = vec![0; n];
        for (i_old, i_new) in new_indices.iter().copied().enumerate() {
            elems[i_new as usize] = self.elem(i_old as Idx);
            etags[i_new as usize] = self.etag(i_old as Idx);
        }
        self.mut_elems()
            .zip(elems.iter())
            .for_each(|(e0, e1)| *e0 = *e1);
        self.mut_etags()
            .zip(etags.iter())
            .for_each(|(t0, t1)| *t0 = *t1);

        self.clear_face_to_elems();
        self.clear_elem_to_elems();
        self.clear_volumes();
    }

    /// Reorder the mesh faces
    /// Element data is updated accordingly, face-to-element connections are reset to None
    /// TODO: update the fields instead of setting them to None and avoid recomputing
    pub fn reorder_faces(&mut self, new_indices: &[Idx]) {
        debug!("Reordering the faces");
        let n = self.n_faces() as usize;
        assert_eq!(new_indices.len(), n);

        let mut faces = vec![E::Face::default(); self.faces().len()];
        let mut ftags = vec![0; n];
        for (i_old, i_new) in new_indices.iter().copied().enumerate() {
            faces[i_new as usize] = self.face(i_old as Idx);
            ftags[i_new as usize] = self.ftag(i_old as Idx);
        }
        self.mut_faces()
            .zip(faces.iter())
            .for_each(|(f0, f1)| *f0 = *f1);
        self.mut_ftags()
            .zip(ftags.iter())
            .for_each(|(t0, t1)| *t0 = *t1);

        self.clear_face_to_elems();
    }

    /// Reorder the vertices, elements and faces using a Hilbert SFC
    /// Elements and faces are renumbered using their minumim vertex Id and not the
    /// coordinate of their centers
    pub fn reorder_hilbert(&mut self) -> (Vec<Idx>, Vec<Idx>, Vec<Idx>) {
        debug!("Reordering the vertices / elements / faces (Hilbert)");

        let new_vert_indices = hilbert_indices(self.bounding_box(), self.verts());
        self.reorder_vertices(&new_vert_indices);

        // // Sort the elems
        // let hilbert_e = |i: usize| hilbert(self.elem_center(i as Idx).as_slice()) as usize;

        // let new_indices = get_indices(self.n_elems() as usize, hilbert_e);
        // self.reorder_elems(&new_indices);

        // // Sort the faces
        // let hilbert_f = |i: usize| hilbert(self.face_center(i as Idx).as_slice()) as usize;

        // let new_indices = get_indices(self.n_faces() as usize, hilbert_e);
        // self.reorder_faces(&new_indices);

        let (new_elem_indices, new_face_indices) = self.reorder_elems_and_faces();

        (new_vert_indices, new_elem_indices, new_face_indices)
    }

    /// Reorder faces and elements to have increasing minimum vertex indices
    pub fn reorder_elems_and_faces(&mut self) -> (Vec<Idx>, Vec<Idx>) {
        debug!("Reordering the elements / faces based on their minimum vertex Id");
        let new_elem_indices = get_indices(self.n_elems() as usize, |i| {
            *self.elem(i as Idx).iter().min().unwrap() as usize
        });
        self.reorder_elems(&new_elem_indices);

        let new_face_indices = get_indices(self.n_faces() as usize, |i| {
            *self.face(i as Idx).iter().min().unwrap() as usize
        });
        self.reorder_faces(&new_face_indices);

        (new_elem_indices, new_face_indices)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        geom_elems::GElem,
        mesh::SimplexMesh,
        test_meshes::{test_mesh_2d, test_mesh_3d},
        topo_elems::Elem,
        Idx,
    };
    use rand::{prelude::SliceRandom, rngs::StdRng, SeedableRng};

    #[test]
    fn test_reorder_verts_2d() {
        let mut mesh = test_mesh_2d().split().split().split();

        let f: Vec<f64> = mesh.verts().map(|p| p[0]).collect();

        // Random reordering
        let mut new_vert_indices: Vec<Idx> = (0..mesh.n_verts()).collect();
        let mut rng = StdRng::seed_from_u64(123);
        new_vert_indices.shuffle(&mut rng);

        mesh.reorder_vertices(&new_vert_indices);
        let mut f_new = vec![0.0; mesh.n_verts() as usize];
        for (i_old, i_new) in new_vert_indices.iter().copied().enumerate() {
            f_new[i_new as usize] = f[i_old];
        }

        let v: f64 = mesh.vol();
        assert!(f64::abs(v - 1.0) < 1e-10);

        for (a, b) in mesh.verts().map(|p| p[0]).zip(f_new.iter().copied()) {
            assert!(f64::abs(b - a) < 1e-10);
        }
    }

    #[test]
    fn test_reorder_verts_3d() {
        let mut mesh = test_mesh_3d().split().split().split();

        let f: Vec<f64> = mesh.verts().map(|p| p[0]).collect();

        // Random reordering
        let mut new_vert_indices: Vec<Idx> = (0..mesh.n_verts()).collect();
        let mut rng = StdRng::seed_from_u64(123);
        new_vert_indices.shuffle(&mut rng);

        mesh.reorder_vertices(&new_vert_indices);

        let v: f64 = mesh.vol();
        assert!(f64::abs(v - 1.0) < 1e-10);

        let mut f_new = vec![0.0; mesh.n_verts() as usize];
        for (i_old, i_new) in new_vert_indices.iter().copied().enumerate() {
            f_new[i_new as usize] = f[i_old];
        }
        for (a, b) in mesh.verts().map(|p| p[0]).zip(f_new.iter().copied()) {
            assert!(f64::abs(b - a) < 1e-10);
        }
    }

    #[test]
    fn test_reorder_elems_2d() {
        let mut mesh = test_mesh_2d().split().split().split();

        let f: Vec<f64> = mesh.gelems().map(|ge| ge.center()[0]).collect();

        // Random reordering
        let mut new_elem_indices: Vec<Idx> = (0..mesh.n_elems()).collect();
        let mut rng = StdRng::seed_from_u64(123);
        new_elem_indices.shuffle(&mut rng);

        mesh.reorder_elems(&new_elem_indices);

        let v: f64 = mesh.vol();
        assert!(f64::abs(v - 1.0) < 1e-10);

        let mut f_new = vec![0.0; mesh.n_elems() as usize];
        for (i_old, i_new) in new_elem_indices.iter().copied().enumerate() {
            f_new[i_new as usize] = f[i_old];
        }
        for (a, b) in mesh
            .gelems()
            .map(|ge| ge.center()[0])
            .zip(f_new.iter().copied())
        {
            assert!(f64::abs(b - a) < 1e-10);
        }
    }

    #[test]
    fn test_reorder_elems_3d() {
        let mut mesh = test_mesh_3d().split().split().split();

        let f: Vec<f64> = mesh.gelems().map(|ge| ge.center()[0]).collect();

        // Random reordering
        let mut new_elem_indices: Vec<Idx> = (0..mesh.n_elems()).collect();
        let mut rng = StdRng::seed_from_u64(123);
        new_elem_indices.shuffle(&mut rng);

        mesh.reorder_elems(&new_elem_indices);

        let v: f64 = mesh.vol();
        assert!(f64::abs(v - 1.0) < 1e-10);

        let mut f_new = vec![0.0; mesh.n_elems() as usize];
        for (i_old, i_new) in new_elem_indices.iter().copied().enumerate() {
            f_new[i_new as usize] = f[i_old];
        }

        for (a, b) in mesh
            .gelems()
            .map(|ge| ge.center()[0])
            .zip(f_new.iter().copied())
        {
            assert!(f64::abs(b - a) < 1e-10);
        }
    }

    #[test]
    fn test_reorder_faces_2d() {
        let mesh = test_mesh_2d();
        let mut mesh = mesh.split().split().split();
        let (bdy_tags, _) = mesh.add_boundary_faces();
        assert!(bdy_tags.is_empty());

        // Random reordering
        let mut new_face_indices: Vec<Idx> = (0..mesh.n_faces()).collect();
        let mut rng = StdRng::seed_from_u64(123);
        new_face_indices.shuffle(&mut rng);

        mesh.reorder_faces(&new_face_indices);
        let (bdy_tags, _) = mesh.add_boundary_faces();
        assert!(bdy_tags.is_empty());
    }

    #[test]
    fn test_reorder_faces_3d() {
        let mesh = test_mesh_3d();
        let mut mesh = mesh.split().split().split();
        let (bdy_tags, _) = mesh.add_boundary_faces();
        assert!(bdy_tags.is_empty());

        // Random reordering
        let mut new_face_indices: Vec<Idx> = (0..mesh.n_faces()).collect();
        let mut rng = StdRng::seed_from_u64(123);
        new_face_indices.shuffle(&mut rng);

        mesh.reorder_faces(&new_face_indices);
        let (bdy_tags, _) = mesh.add_boundary_faces();
        assert!(bdy_tags.is_empty());
    }

    #[test]
    fn test_elem_to_elems_2d() {
        let mut mesh = test_mesh_2d().split();
        mesh.compute_elem_to_elems();

        let e2e = mesh.get_elem_to_elems().unwrap();

        assert_eq!(e2e.row(0), [2, 4]);
        assert_eq!(e2e.row(1), [2]);
        assert_eq!(e2e.row(2), [0, 1, 3]);
        assert_eq!(e2e.row(3), [2, 5]);
        assert_eq!(e2e.row(4), [0, 6]);
        assert_eq!(e2e.row(5), [3, 6]);
        assert_eq!(e2e.row(6), [4, 5, 7]);
        assert_eq!(e2e.row(7), [6]);
    }

    fn mean_bandwidth_e2v<const D: usize, E: Elem>(mesh: &SimplexMesh<D, E>) -> f64 {
        let mut mean = 0;
        for i_elem in 0..mesh.n_elems() {
            let e = mesh.elem(i_elem);
            mean += e.iter().max().unwrap() - e.iter().min().unwrap();
        }
        f64::from(mean) / f64::from(mesh.n_elems())
    }

    fn mean_bandwidth_e2e<const D: usize, E: Elem>(mesh: &SimplexMesh<D, E>) -> f64 {
        let e2e = mesh.get_elem_to_elems().unwrap();
        let mut mean = 0;
        for i_elem in 0..mesh.n_elems() {
            let n = e2e.row(i_elem);
            mean += n
                .iter()
                .map(|i| i32::abs(*i as i32 - i_elem as i32))
                .max()
                .unwrap();
        }
        f64::from(mean) / f64::from(mesh.n_elems())
    }

    #[test]
    fn test_hilbert_2d() {
        let mut mesh = test_mesh_2d().split().split().split().split().split();
        mesh.compute_elem_to_elems();

        let mean_e2v_before = mean_bandwidth_e2v(&mesh);
        let mean_e2e_before = mean_bandwidth_e2e(&mesh);

        mesh.reorder_hilbert();

        mesh.compute_elem_to_elems();
        let mean_e2v_after = mean_bandwidth_e2v(&mesh);
        let mean_e2e_after = mean_bandwidth_e2e(&mesh);

        assert!(mean_e2v_after < 0.11 * mean_e2v_before);
        assert!(mean_e2e_after < 1.3 * mean_e2e_before);
    }

    #[test]
    fn test_hilbert_3d() {
        let mut mesh = test_mesh_3d().split().split().split();
        mesh.compute_elem_to_elems();

        let mean_e2v_before = mean_bandwidth_e2v(&mesh);
        let mean_e2e_before = mean_bandwidth_e2e(&mesh);

        mesh.reorder_hilbert();

        mesh.compute_elem_to_elems();
        let mean_e2v_after = mean_bandwidth_e2v(&mesh);
        let mean_e2e_after = mean_bandwidth_e2e(&mesh);

        assert!(mean_e2v_after < 0.5 * mean_e2v_before);
        assert!(mean_e2e_after < 1.1 * mean_e2e_before);
    }
}
