use crate::{mesh::SimplexMesh, topo_elems::Elem, Idx, Mesh};
use lindel::Lineariseable;
use log::{debug, info};

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

impl<const D: usize, E: Elem> SimplexMesh<D, E> {
    /// Reorder the mesh vertices
    /// Vertex data is updated accordingly, but edges, vertex-to-vertex connections
    /// and vertex volumes are reset to None
    /// TODO: update the fields instead of setting them to None and avoid recomputing
    pub fn reorder_vertices(&mut self, new_indices: &[Idx]) {
        debug!("Reordering the vertices");

        let n = self.n_verts() as usize;
        assert_eq!(new_indices.len(), n);

        let mut coords = vec![0.0; self.coords.len()];
        for (i_old, i_new) in new_indices.iter().copied().enumerate() {
            for j in 0..D {
                coords[D * i_new as usize + j] = self.coords[D * i_old + j];
            }
        }
        self.coords = coords;

        self.elems
            .iter_mut()
            .for_each(|i| *i = new_indices[*i as usize]);
        self.faces
            .iter_mut()
            .for_each(|i| *i = new_indices[*i as usize]);

        self.vertex_to_elems = None;
        self.vertex_to_vertices = None;
        self.edges = None;
        self.vert_vol = None;
    }

    /// Reorder the mesh elements
    /// Element data is updated accordingly, face-to-element and element-to-element connections
    /// and element volumes are reset to None
    /// TODO: update the fields instead of setting them to None and avoid recomputing
    pub fn reorder_elems(&mut self, new_indices: &[Idx]) {
        debug!("Reordering the elements");
        let n = self.n_elems() as usize;
        assert_eq!(new_indices.len(), n);

        let mut elems = vec![0; self.elems.len()];
        let mut etags = vec![0; n];
        for (i_old, i_new) in new_indices.iter().copied().enumerate() {
            for j in 0..E::N_VERTS as usize {
                elems[(E::N_VERTS * i_new) as usize + j] =
                    self.elems[E::N_VERTS as usize * i_old + j];
            }
            etags[i_new as usize] = self.etags[i_old];
        }
        self.elems = elems;
        self.etags = etags;

        self.clear_face_to_elems();
        self.elem_to_elems = None;
        self.elem_vol = None;
    }

    /// Reorder the mesh faces
    /// Element data is updated accordingly, face-to-element connections are reset to None
    /// TODO: update the fields instead of setting them to None and avoid recomputing
    pub fn reorder_faces(&mut self, new_indices: &[Idx]) {
        debug!("Reordering the faces");
        let n = self.n_faces() as usize;
        assert_eq!(new_indices.len(), n);

        let mut faces = vec![0; self.faces.len()];
        let mut ftags = vec![0; n];
        for (i_old, i_new) in new_indices.iter().copied().enumerate() {
            for j in 0..E::Face::N_VERTS as usize {
                faces[(E::Face::N_VERTS * i_new) as usize + j] =
                    self.faces[E::Face::N_VERTS as usize * i_old + j];
            }
            ftags[i_new as usize] = self.ftags[i_old];
        }
        self.faces = faces;
        self.ftags = ftags;

        self.clear_face_to_elems();
    }

    /// Reorder the vertices, elements and faces using a Hilbert SFC
    /// Elements and faces are renumbered using their minumim vertex Id and not the
    /// coordinate of their centers
    pub fn reorder_hilbert(&mut self) -> (Vec<Idx>, Vec<Idx>, Vec<Idx>) {
        info!("Reordering the vertices / elements / faces (Hilbert)");
        // bounding box
        let mut mini = [0.; D];
        let mut maxi = [0.; D];

        for p in self.verts() {
            for j in 0..D {
                mini[j] = f64::min(mini[j], p[j]);
                maxi[j] = f64::max(maxi[j], p[j]);
            }
        }

        // Hilbert index
        let order = 16;
        let scale = usize::pow(2, order) as f64 - 1.0;
        let hilbert = |x: &[f64]| {
            let mut tmp = [0; 3];
            for j in 0..D {
                tmp[j] = (scale * (x[j] - mini[j]) / (maxi[j] - mini[j])).round() as u16;
            }
            tmp.hilbert_index()
        };

        // Sort the vertices
        let hilbert_v = |i: usize| {
            let start = D * i;
            let end = start + D;
            hilbert(&self.coords[start..end]) as usize
        };

        let new_vert_indices = get_indices(self.n_verts() as usize, hilbert_v);
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
        mesh::SimplexMesh,
        test_meshes::{test_mesh_2d, test_mesh_3d},
        topo_elems::Elem,
        Idx, Mesh, Result,
    };
    use rand::{prelude::SliceRandom, rngs::StdRng, SeedableRng};

    #[test]
    fn test_reorder_verts_2d() -> Result<()> {
        let mut mesh = test_mesh_2d().split().split().split();

        let f: Vec<f64> = mesh.coords.iter().step_by(2).copied().collect();

        // Random reordering
        let mut new_vert_indices: Vec<Idx> = (0..mesh.n_verts()).collect();
        let mut rng = StdRng::seed_from_u64(123);
        new_vert_indices.shuffle(&mut rng);

        mesh.reorder_vertices(&new_vert_indices);
        let mut f_new = vec![0.0; mesh.n_verts() as usize];
        for (i_old, i_new) in new_vert_indices.iter().copied().enumerate() {
            f_new[i_new as usize] = f[i_old];
        }

        let v: f64 = mesh.elem_vols().sum();
        assert!(f64::abs(v - 1.0) < 1e-10);

        for (a, b) in mesh
            .coords
            .iter()
            .step_by(2)
            .copied()
            .zip(f_new.iter().copied())
        {
            assert!(f64::abs(b - a) < 1e-10);
        }

        Ok(())
    }

    #[test]
    fn test_reorder_verts_3d() -> Result<()> {
        let mut mesh = test_mesh_3d().split().split().split();

        let f: Vec<f64> = mesh.coords.iter().step_by(3).copied().collect();

        // Random reordering
        let mut new_vert_indices: Vec<Idx> = (0..mesh.n_verts()).collect();
        let mut rng = StdRng::seed_from_u64(123);
        new_vert_indices.shuffle(&mut rng);

        mesh.reorder_vertices(&new_vert_indices);

        let v: f64 = mesh.elem_vols().sum();
        assert!(f64::abs(v - 1.0) < 1e-10);

        let mut f_new = vec![0.0; mesh.n_verts() as usize];
        for (i_old, i_new) in new_vert_indices.iter().copied().enumerate() {
            f_new[i_new as usize] = f[i_old];
        }
        for (a, b) in mesh
            .coords
            .iter()
            .step_by(3)
            .copied()
            .zip(f_new.iter().copied())
        {
            assert!(f64::abs(b - a) < 1e-10);
        }

        Ok(())
    }

    #[test]
    fn test_reorder_elems_2d() -> Result<()> {
        let mut mesh = test_mesh_2d().split().split().split();

        let f: Vec<f64> = (0..(mesh.n_elems() as Idx))
            .map(|i| mesh.elem_center(i)[0])
            .collect();

        // Random reordering
        let mut new_elem_indices: Vec<Idx> = (0..mesh.n_elems()).collect();
        let mut rng = StdRng::seed_from_u64(123);
        new_elem_indices.shuffle(&mut rng);

        mesh.reorder_elems(&new_elem_indices);

        let v: f64 = mesh.elem_vols().sum();
        assert!(f64::abs(v - 1.0) < 1e-10);

        let mut f_new = vec![0.0; mesh.n_elems() as usize];
        for (i_old, i_new) in new_elem_indices.iter().copied().enumerate() {
            f_new[i_new as usize] = f[i_old];
        }
        for (a, b) in (0..(mesh.n_elems() as Idx))
            .map(|i| mesh.elem_center(i)[0])
            .zip(f_new.iter().copied())
        {
            assert!(f64::abs(b - a) < 1e-10);
        }

        Ok(())
    }

    #[test]
    fn test_reorder_elems_3d() -> Result<()> {
        let mut mesh = test_mesh_3d().split().split().split();

        let f: Vec<f64> = (0..(mesh.n_elems() as Idx))
            .map(|i| mesh.elem_center(i)[0])
            .collect();

        // Random reordering
        let mut new_elem_indices: Vec<Idx> = (0..mesh.n_elems()).collect();
        let mut rng = StdRng::seed_from_u64(123);
        new_elem_indices.shuffle(&mut rng);

        mesh.reorder_elems(&new_elem_indices);

        let v: f64 = mesh.elem_vols().sum();
        assert!(f64::abs(v - 1.0) < 1e-10);

        let mut f_new = vec![0.0; mesh.n_elems() as usize];
        for (i_old, i_new) in new_elem_indices.iter().copied().enumerate() {
            f_new[i_new as usize] = f[i_old];
        }

        for (a, b) in (0..(mesh.n_elems() as Idx))
            .map(|i| mesh.elem_center(i)[0])
            .zip(f_new.iter().copied())
        {
            assert!(f64::abs(b - a) < 1e-10);
        }

        Ok(())
    }

    #[test]
    fn test_reorder_faces_2d() {
        let mesh = test_mesh_2d();
        let mut mesh = mesh.split().split().split();
        let (bdy_tag, _) = mesh.add_boundary_faces();
        assert_eq!(mesh.n_tagged_faces(bdy_tag), 0);

        // Random reordering
        let mut new_face_indices: Vec<Idx> = (0..mesh.n_faces()).collect();
        let mut rng = StdRng::seed_from_u64(123);
        new_face_indices.shuffle(&mut rng);

        mesh.reorder_faces(&new_face_indices);
        let (bdy_tag, _) = mesh.add_boundary_faces();
        assert_eq!(mesh.n_tagged_faces(bdy_tag), 0);
    }

    #[test]
    fn test_reorder_faces_3d() {
        let mesh = test_mesh_3d();
        let mut mesh = mesh.split().split().split();
        let (bdy_tag, _) = mesh.add_boundary_faces();
        assert_eq!(mesh.n_tagged_faces(bdy_tag), 0);

        // Random reordering
        let mut new_face_indices: Vec<Idx> = (0..mesh.n_faces()).collect();
        let mut rng = StdRng::seed_from_u64(123);
        new_face_indices.shuffle(&mut rng);

        mesh.reorder_faces(&new_face_indices);
        let (bdy_tag, _) = mesh.add_boundary_faces();
        assert_eq!(mesh.n_tagged_faces(bdy_tag), 0);
    }

    #[test]
    fn test_elem_to_elems_2d() {
        let mut mesh = test_mesh_2d().split();
        mesh.compute_elem_to_elems();

        let e2e = mesh.elem_to_elems.as_ref().unwrap();

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
        let e2e = mesh.elem_to_elems.as_ref().unwrap();
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
