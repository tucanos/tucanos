//! General polyline, polygon and polyhedral meshes
use crate::{
    Error, Result, Tag, Vertex,
    io::VTUFile,
    mesh::{Edge, GSimplex, Mesh, Simplex, Triangle},
};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use rustc_hash::{FxBuildHasher, FxHashMap, FxHashSet};

/// Polymesh type
#[derive(Debug, Clone, Copy)]
pub enum PolyMeshType {
    /// Polylines
    Polylines,
    /// Polygons
    Polygons,
    /// Polyhedra
    Polyhedra,
}

/// Polymesh type
#[derive(Debug, Clone, Copy)]
pub enum PolyFaceType {
    /// General,
    General,
    /// Simplices
    Simplices,
}

/// Polylines, polygons or polyhedra meshes in D dimensions
///   - faces are represented by the indices of their vertices (oriented)
///   - elements are indices of their faces and flags indicating if the face is oriented
///     outwards or inwards
pub trait PolyMesh<const D: usize>: Sync + Sized {
    /// Element type
    fn poly_type(&self) -> PolyMeshType;

    /// Face type
    fn face_type(&self) -> PolyFaceType;

    /// Number of vertices
    fn n_verts(&self) -> usize;

    /// Get the `i`th vertex
    fn vert(&self, i: usize) -> Vertex<D>;

    /// Parallel iterator over the mesh vertices
    fn par_verts(&self) -> impl IndexedParallelIterator<Item = Vertex<D>> + Clone + '_ {
        (0..self.n_verts()).into_par_iter().map(|i| self.vert(i))
    }

    /// Sequential iterator over the mesh vertices
    fn verts(&self) -> impl ExactSizeIterator<Item = Vertex<D>> + Clone + '_ {
        (0..self.n_verts()).map(|i| self.vert(i))
    }

    /// Number of elements
    fn n_elems(&self) -> usize;

    /// Get the `i`th face
    fn elem(&self, i: usize) -> impl ExactSizeIterator<Item = (usize, bool)> + Clone + Send;

    /// Number of vertices in the `i`the element
    fn elem_n_verts(&self, i: usize) -> usize {
        let mut verts = FxHashSet::with_hasher(FxBuildHasher);
        for (j, _) in self.elem(i) {
            for k in self.face(j) {
                verts.insert(k);
            }
        }
        verts.len()
    }

    /// Parallel iterator over the mesh elements
    fn par_elems(
        &self,
    ) -> impl IndexedParallelIterator<
        Item = impl ExactSizeIterator<Item = (usize, bool)> + Clone + Send,
    > + Clone
    + '_ {
        (0..self.n_elems()).into_par_iter().map(|i| self.elem(i))
    }

    /// Sequantial iterator over the mesh elements
    fn elems(
        &self,
    ) -> impl ExactSizeIterator<Item = impl ExactSizeIterator<Item = (usize, bool)> + Clone + Send>
    + Clone
    + '_ {
        (0..self.n_elems()).map(|i| self.elem(i))
    }

    /// Get the tag of the `i`the element
    fn etag(&self, i: usize) -> Tag;

    /// Parallel iterator over the element tags
    fn par_etags(&self) -> impl IndexedParallelIterator<Item = Tag> + Clone + '_ {
        (0..self.n_elems()).into_par_iter().map(|i| self.etag(i))
    }

    /// Sequential iterator over the element tags
    fn etags(&self) -> impl ExactSizeIterator<Item = Tag> + Clone + '_ {
        (0..self.n_elems()).map(|i| self.etag(i))
    }

    /// Number of faces
    fn n_faces(&self) -> usize;

    /// Get the `i`the face
    fn face(&self, i: usize) -> impl ExactSizeIterator<Item = usize> + Clone + Send;

    /// Parallel iterator over the faces
    fn par_faces(
        &self,
    ) -> impl IndexedParallelIterator<Item = impl ExactSizeIterator<Item = usize> + Clone + Send>
    + Clone
    + '_ {
        (0..self.n_faces()).into_par_iter().map(|i| self.face(i))
    }

    /// Sequential iterator over the faces
    fn faces(
        &self,
    ) -> impl ExactSizeIterator<Item = impl ExactSizeIterator<Item = usize> + Clone + Send> + Clone + '_
    {
        (0..self.n_faces()).map(|i| self.face(i))
    }

    /// Get the tag of the `i`th face
    fn ftag(&self, i: usize) -> Tag;

    /// Parallel iterator over the face tags
    fn par_ftags(&self) -> impl IndexedParallelIterator<Item = Tag> + Clone + '_ {
        (0..self.n_faces()).into_par_iter().map(|i| self.ftag(i))
    }

    /// Sequential iterator over the face tags
    fn ftags(&self) -> impl ExactSizeIterator<Item = Tag> + Clone + '_ {
        (0..self.n_faces()).map(|i| self.ftag(i))
    }

    /// Export the mesh to a `.vtu` file
    fn write_vtk(&self, file_name: &str) -> std::io::Result<()> {
        VTUFile::from_poly_mesh(self).export(file_name)
    }

    fn elem_gfaces_c<C: Simplex>(
        &self,
        i: usize,
    ) -> Option<impl ExactSizeIterator<Item = C::GEOM<D>> + '_>;

    /// Get the centroid of the `i`th cell
    fn elem_center_c<C: Simplex>(&self, i: usize) -> Vertex<D> {
        let mut vol = 0.0;
        let mut first_moment = Vertex::<D>::zeros();

        self.elem_gfaces_c::<C>(i)
            .expect("Faces are not simplices")
            .for_each(|gf| {
                let c = gf.center();
                let flux = c.dot(&gf.normal(None));
                vol += flux / D as f64;
                first_moment += c * (flux / (D as f64 + 1.0));
            });

        assert!(vol > f64::EPSILON, "Element {i} has zero volume");
        first_moment / vol
    }

    /// Check if vertex `v` is in the `i`th element by computing the winding number
    fn is_vertex_in_elem_c<C: Simplex>(&self, v: &Vertex<D>, i: usize) -> bool {
        let tol = 1e-12;

        if D == 2 {
            let mut angle_sum = 0.0;

            for gf in self.elem_gfaces_c::<C>(i).expect("Faces are not simplices") {
                let mut pts = gf.into_iter();
                let p0 = pts.next().unwrap();
                let p1 = pts.next().unwrap();

                let a = p0 - *v;
                let b = p1 - *v;

                let la = a.norm();
                let lb = b.norm();
                if la <= tol || lb <= tol {
                    return true;
                }

                let edge = p1 - p0;
                let edge_len = edge.norm();
                if edge_len > tol {
                    let t = (v - p0).dot(&edge) / edge.norm_squared();
                    if (-tol..=1.0 + tol).contains(&t) {
                        let proj = p0 + t * edge;
                        if (*v - proj).norm() <= tol * edge_len {
                            return true;
                        }
                    }
                }

                let cross = a[0] * b[1] - a[1] * b[0];
                let dot = a.dot(&b);
                angle_sum += cross.atan2(dot);
            }

            return angle_sum.abs() > std::f64::consts::PI;
        }

        if D == 3 {
            let mut solid_angle_sum = 0.0;

            for gf in self.elem_gfaces_c::<C>(i).expect("Faces are not simplices") {
                let mut pts = gf.into_iter();
                let p0 = pts.next().unwrap();
                let p1 = pts.next().unwrap();
                let p2 = pts.next().unwrap();

                let a = p0 - *v;
                let b = p1 - *v;
                let c = p2 - *v;

                let la = a.norm();
                let lb = b.norm();
                let lc = c.norm();
                if la <= tol || lb <= tol || lc <= tol {
                    return true;
                }

                // Van Oosterom-Strackee solid angle of oriented triangle (p0,p1,p2) at v
                let det = a[0] * (b[1] * c[2] - b[2] * c[1])
                    + a[1] * (b[2] * c[0] - b[0] * c[2])
                    + a[2] * (b[0] * c[1] - b[1] * c[0]);
                let den = la * lb * lc + a.dot(&b) * lc + b.dot(&c) * la + c.dot(&a) * lb;
                solid_angle_sum += 2.0 * det.atan2(den);
            }

            return solid_angle_sum.abs() > 2.0 * std::f64::consts::PI;
        }

        unreachable!("is_vertex_in_elem is only implemented for D=2 and D=3");
    }

    /// Get the volume of the `i`th cell
    fn vol_c<C: Simplex>(&self, i: usize) -> f64 {
        self.elem_gfaces_c::<C>(i)
            .expect("Faces are not simplices")
            .map(|gf| gf.center().dot(&gf.normal(None)))
            .sum::<f64>()
            / D as f64
    }

    /// Parallel iterator over cell volumes
    fn par_vols_c<C: Simplex>(&self) -> impl IndexedParallelIterator<Item = f64> + '_ {
        (0..self.n_elems())
            .into_par_iter()
            .map(|i| self.vol_c::<C>(i))
    }

    /// Sequential iterator over cell volumes
    fn vols_c<C: Simplex>(&self) -> impl ExactSizeIterator<Item = f64> + '_ {
        (0..self.n_elems()).map(|i| self.vol_c::<C>(i))
    }

    /// Check if polygonal cell `i` is closed
    fn is_closed_c<C: Simplex>(&self, i: usize) -> bool {
        let mut res = [0.0; D];

        self.elem_gfaces_c::<C>(i)
            .expect("Faces are not simplices")
            .for_each(|gf| {
                let n = gf.normal(None);
                res.iter_mut().zip(n.iter()).for_each(|(x, y)| *x += y);
            });
        res.iter().map(|x| x.abs()).sum::<f64>() < 1e-10
    }

    /// Check the validity of the dual mesh
    ///  - consistent number of faces and faces tags
    ///  - consistent face to vertex connectivity
    ///  - consistent element to face connectivity
    ///  - closed elements
    ///  - unique faces
    fn check_c<C: Simplex>(&self) -> Result<()> {
        // lengths
        if self.par_faces().len() != self.par_ftags().len() {
            return Err(Error::from("Inconsistent sizes (faces)"));
        }

        // indices
        if self.par_faces().any(|mut f| f.any(|i| i >= self.n_verts())) {
            return Err(Error::from("Inconsistent indices (faces)"));
        }

        // faces
        if self
            .par_elems()
            .any(|mut e| e.any(|x| x.0 >= self.n_faces()))
        {
            return Err(Error::from("Inconsistent indices (elems)"));
        }

        // closed elements
        for (i, (e, v)) in self.elems().zip(self.vols_c::<C>()).enumerate() {
            if v < f64::EPSILON {
                let e = e.collect::<Vec<_>>();
                return Err(Error::from(&format!(
                    "Element {i} invalid: vol={v} < 0  ({e:?})"
                )));
            }
            if !self.is_closed_c::<C>(i) {
                let e = e.collect::<Vec<_>>();
                return Err(Error::from(&format!("Element {i} not closed ({e:?})")));
            }
        }

        // all faces appear only once
        let mut faces = FxHashMap::with_hasher(FxBuildHasher);
        for (i, f) in self.faces().enumerate() {
            assert_eq!(f.len(), C::N_VERTS);
            let res = C::from_iter(f.clone()).sorted();
            if let std::collections::hash_map::Entry::Vacant(e) = faces.entry(res) {
                e.insert(i);
            } else {
                let j = *faces.get(&res).unwrap();
                let f = f.collect::<Vec<_>>();
                let f_j = self.face(j).collect::<Vec<_>>();
                return Err(Error::from(&format!(
                    "Face {i} ({f:?}) = face {j} ({f_j:?})"
                )));
            }
        }

        // faces appear at most once in 1 or 2 elements
        let mut flg = vec![0; self.n_faces()];
        for (i_elem, e) in self.elems().enumerate() {
            let mut tmp = FxHashSet::with_capacity_and_hasher(e.len(), FxBuildHasher);
            for (i, sgn) in e {
                if tmp.contains(&i) {
                    return Err(Error::from(&format!(
                        "Face {i} appears multiple times in element {i_elem}"
                    )));
                }
                tmp.insert(i);
                if flg[i] == 0 {
                    if sgn {
                        flg[i] = 1;
                    } else {
                        flg[i] = -1;
                    }
                } else if flg[i] == 1 {
                    if sgn {
                        return Err(Error::from(&format!("Face {i} appears twice with True")));
                    }
                    flg[i] = 2;
                } else if flg[i] == -1 {
                    if !sgn {
                        return Err(Error::from(&format!("Face {i} appears twice with False")));
                    }
                    flg[i] = 2;
                } else if flg[i] == 2 {
                    return Err(Error::from(&format!("Face {i} appears 3 times")));
                } else {
                    unreachable!()
                }
            }
        }

        Ok(())
    }

    /// Get the dimension of the elements
    fn elem_dim(&self) -> usize {
        match self.poly_type() {
            PolyMeshType::Polylines => 1,
            PolyMeshType::Polygons => 2,
            PolyMeshType::Polyhedra => 3,
        }
    }

    fn check(&self) -> Result<()> {
        if matches!(self.face_type(), PolyFaceType::Simplices) {
            match self.elem_dim() {
                2 => self.check_c::<Edge<usize>>(),
                3 => self.check_c::<Triangle<usize>>(),
                _ => unimplemented!(),
            }
        } else {
            Err(Error::from("Check only available with simplex faces"))
        }
    }
}

/// General `PolyMesh<D>`
pub struct SimplePolyMesh<const D: usize> {
    poly_type: PolyMeshType,
    face_type: PolyFaceType,
    verts: Vec<Vertex<D>>,
    face_to_node_ptr: Vec<usize>,
    face_to_node: Vec<usize>,
    ftags: Vec<Tag>,
    elem_to_face_ptr: Vec<usize>,
    elem_to_face: Vec<(usize, bool)>,
    etags: Vec<Tag>,
}

impl<const D: usize> SimplePolyMesh<D> {
    /// Create a new mesh from coordinates, connectivities and tags
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub const fn new(
        poly_type: PolyMeshType,
        face_type: PolyFaceType,
        verts: Vec<Vertex<D>>,
        face_to_node_ptr: Vec<usize>,
        face_to_node: Vec<usize>,
        ftags: Vec<Tag>,
        elem_to_face_ptr: Vec<usize>,
        elem_to_face: Vec<(usize, bool)>,
        etags: Vec<Tag>,
    ) -> Self {
        Self {
            poly_type,
            face_type,
            verts,
            face_to_node_ptr,
            face_to_node,
            ftags,
            elem_to_face_ptr,
            elem_to_face,
            etags,
        }
    }

    /// Create an empty mesh with a given element type
    #[must_use]
    pub fn empty(poly_type: PolyMeshType, face_type: PolyFaceType) -> Self {
        Self {
            poly_type,
            face_type,
            verts: Vec::new(),
            face_to_node_ptr: vec![0],
            face_to_node: Vec::new(),
            ftags: Vec::new(),
            elem_to_face_ptr: vec![0],
            elem_to_face: Vec::new(),
            etags: Vec::new(),
        }
    }

    /// Insert a vertex and return its index
    pub fn insert_vert(&mut self, v: Vertex<D>) -> usize {
        self.verts.push(v);
        self.verts.len() - 1
    }

    /// Insert a face and return its index
    pub fn insert_face(&mut self, f: &[usize], t: Tag) -> usize {
        if matches!(self.face_type, PolyFaceType::Simplices) {
            assert_eq!(f.len(), self.elem_dim());
        }
        self.face_to_node.extend(f);
        self.face_to_node_ptr.push(self.face_to_node.len());
        self.ftags.push(t);
        self.ftags.len() - 1
    }

    /// Insert an element and return its index
    pub fn insert_elem<I: IntoIterator<Item = (usize, bool)>>(&mut self, e: I, t: Tag) -> usize {
        self.elem_to_face.extend(e);
        self.elem_to_face_ptr.push(self.elem_to_face.len());
        self.etags.push(t);
        self.etags.len() - 1
    }
    /// Helper: Analyze mesh to group faces based on which elements they connect.
    fn group_connected_faces<T: PolyMesh<D>>(
        mesh: &T,
    ) -> FxHashMap<[usize; 2], Vec<(usize, bool)>> {
        // simplify faces
        let mut face2elems = vec![[usize::MAX, usize::MAX]; mesh.n_faces()];
        for (i_elem, elem) in mesh.elems().enumerate() {
            for (i_face, orient) in elem {
                if orient {
                    assert_eq!(face2elems[i_face][0], usize::MAX);
                    face2elems[i_face][0] = i_elem;
                } else {
                    assert_eq!(face2elems[i_face][1], usize::MAX);
                    face2elems[i_face][1] = i_elem;
                }
            }
        }

        let mut face_groups =
            FxHashMap::<[usize; 2], Vec<(usize, bool)>>::with_hasher(FxBuildHasher);
        for (i_face, elems) in face2elems.iter().enumerate() {
            if let Some(x) = face_groups.get_mut(elems) {
                x.push((i_face, true));
            } else if let Some(x) = face_groups.get_mut(&[elems[1], elems[0]]) {
                x.push((i_face, false));
            } else {
                face_groups.insert(*elems, vec![(i_face, true)]);
            }
        }
        face_groups
    }

    /// Helper: Reconstruct the element connectivity pointers.
    fn build_element_connectivity(
        n_elems: usize,
        new_faces_elems: &[[usize; 2]],
    ) -> (Vec<usize>, Vec<(usize, bool)>) {
        let mut new_elems_ptr = vec![0; n_elems + 1];
        for &[i0, i1] in new_faces_elems {
            new_elems_ptr[i0 + 1] += 1;
            if i1 != usize::MAX {
                new_elems_ptr[i1 + 1] += 1;
            }
        }
        for i in 0..n_elems {
            new_elems_ptr[i + 1] += new_elems_ptr[i];
        }
        let mut new_elems = vec![(usize::MAX, false); *new_elems_ptr.last().unwrap()];
        for (i_face, &[i0, i1]) in new_faces_elems.iter().enumerate() {
            let mut ok = false;
            for v in new_elems
                .iter_mut()
                .take(new_elems_ptr[i0 + 1])
                .skip(new_elems_ptr[i0])
            {
                if v.0 == usize::MAX {
                    *v = (i_face, true);
                    ok = true;
                    break;
                }
            }
            assert!(ok);
            if i1 != usize::MAX {
                let mut ok = false;
                for v in new_elems
                    .iter_mut()
                    .take(new_elems_ptr[i1 + 1])
                    .skip(new_elems_ptr[i1])
                {
                    if v.0 == usize::MAX {
                        *v = (i_face, false);
                        ok = true;
                        break;
                    }
                }
                assert!(ok);
            }
        }
        (new_elems_ptr, new_elems)
    }

    /// Simplify a `PolyMesh<D>` by merging its faces connecting the same two elements
    pub fn simplify<T: PolyMesh<D>>(mesh: &T, simplify: bool) -> Self {
        let poly_type = mesh.poly_type();
        let face_groups = Self::group_connected_faces(mesh);
        let mut new_faces = Vec::new();
        let mut new_faces_ptr = Vec::new();
        new_faces_ptr.push(0);
        let mut new_faces_elems = Vec::new();
        let mut new_ftags = Vec::new();
        for (elems, faces) in face_groups {
            assert_ne!(elems[0], usize::MAX);
            if elems[1] != usize::MAX && simplify {
                // copy faces to reorient if needed
                let mut tmp_ptr = Vec::new();
                tmp_ptr.push(0);
                let mut tmp = Vec::new();
                for &(i_face, orient) in &faces {
                    if orient {
                        tmp.extend(mesh.face(i_face));
                    } else {
                        let mut f = mesh.face(i_face).collect::<Vec<_>>();
                        f.reverse();
                        tmp.extend_from_slice(&f);
                    }
                    tmp_ptr.push(tmp.len());
                }
                let n_faces = faces.len();
                let faces = (0..n_faces)
                    .map(|i| {
                        let start = tmp_ptr[i];
                        let end = tmp_ptr[i + 1];
                        &tmp[start..end]
                    })
                    .collect::<Vec<_>>();
                match poly_type {
                    PolyMeshType::Polylines => todo!(),
                    PolyMeshType::Polygons => {
                        let polygons = merge_polylines(&faces);
                        assert_eq!(polygons.len(), 1);

                        new_faces.extend_from_slice(&polygons[0]);
                    }
                    PolyMeshType::Polyhedra => {
                        let polylines = merge_polygons(&faces);
                        assert_eq!(polylines.len(), 1);

                        new_faces.extend_from_slice(&polylines[0]);
                    }
                }

                new_faces_ptr.push(new_faces.len());
                new_faces_elems.push(elems);
                new_ftags.push(0);
            } else {
                for (i_face, _) in faces {
                    new_faces.extend(mesh.face(i_face));
                    new_faces_ptr.push(new_faces.len());
                    new_faces_elems.push(elems);
                    new_ftags.push(mesh.ftag(i_face));
                }
            }
        }
        let (new_elems_ptr, new_elems) =
            Self::build_element_connectivity(mesh.n_elems(), &new_faces_elems);
        let mut res = Self {
            poly_type: mesh.poly_type(),
            face_type: PolyFaceType::General,
            verts: mesh.verts().collect(),
            face_to_node_ptr: new_faces_ptr,
            face_to_node: new_faces,
            ftags: new_ftags,
            elem_to_face_ptr: new_elems_ptr,
            elem_to_face: new_elems,
            etags: mesh.etags().collect(),
        };

        // unused vertices
        let mut new_ids = vec![usize::MAX; mesh.n_verts()];
        let mut new_verts = Vec::new();
        let mut next = 0;
        for face in res.faces() {
            for i in face {
                if new_ids[i] == usize::MAX {
                    new_ids[i] = next;
                    new_verts.push(mesh.vert(i));
                    next += 1;
                }
            }
        }
        res.face_to_node.iter_mut().for_each(|i| *i = new_ids[*i]);
        res.verts = new_verts;

        res
    }

    /// PolyMesh representation of a `Mesh<D>`
    pub fn from_mesh<M: Mesh<D>>(mesh: &M) -> Self {
        let poly_type = match <M::C as Simplex>::DIM {
            3 => PolyMeshType::Polyhedra,
            2 => PolyMeshType::Polygons,
            1 => PolyMeshType::Polylines,
            _ => unimplemented!(),
        };
        let all_faces = mesh.all_faces();
        let tagged_faces = mesh
            .par_faces()
            .zip(mesh.par_ftags())
            .map(|(f, t)| (f.sorted(), t))
            .collect::<FxHashMap<_, _>>();

        let mut elem_to_face_ptr = vec![0; mesh.n_elems() + 1];
        let mut elem_to_face =
            vec![(usize::MAX, true); mesh.n_elems() * <M::C as Simplex>::N_VERTS];
        let mut face_to_node_ptr = vec![0; all_faces.len() + 1];
        let mut face_to_node = vec![0; all_faces.len() * <M::C as Simplex>::FACE::N_VERTS];
        let mut ftags = vec![0; all_faces.len()];

        for i_elem in 0..mesh.n_elems() {
            elem_to_face_ptr[i_elem + 1] = <M::C as Simplex>::N_VERTS * (i_elem + 1);
        }
        for (f, &(i_face, i0, i1)) in &all_faces {
            face_to_node_ptr[i_face + 1] = <M::C as Simplex>::FACE::N_VERTS * (i_face + 1);
            for k in 0..<M::C as Simplex>::FACE::N_VERTS {
                face_to_node[<M::C as Simplex>::FACE::N_VERTS * i_face + k] = f.get(k);
            }
            if let Some(i0) = i0 {
                let mut ok = false;
                for v in elem_to_face
                    .iter_mut()
                    .take(elem_to_face_ptr[i0 + 1])
                    .skip(elem_to_face_ptr[i0])
                {
                    if v.0 == usize::MAX {
                        *v = (i_face, true);
                        ok = true;
                        break;
                    }
                }
                assert!(ok);
            }
            if let Some(i1) = i1 {
                let mut ok = false;
                for v in elem_to_face
                    .iter_mut()
                    .take(elem_to_face_ptr[i1 + 1])
                    .skip(elem_to_face_ptr[i1])
                {
                    if v.0 == usize::MAX {
                        *v = (i_face, false);
                        ok = true;
                        break;
                    }
                }
                assert!(ok);
            }
            if i0.is_none() && i1.is_none() {
                let f = f.sorted();
                ftags[i_face] = *tagged_faces.get(&f).unwrap();
            } else {
                ftags[i_face] = 0;
            }
        }

        Self {
            poly_type,
            face_type: PolyFaceType::Simplices,
            verts: mesh.par_verts().collect(),
            face_to_node_ptr,
            face_to_node,
            ftags,
            elem_to_face_ptr,
            elem_to_face,
            etags: mesh.par_etags().collect(),
        }
    }
}

impl<const D: usize> PolyMesh<D> for SimplePolyMesh<D> {
    fn poly_type(&self) -> PolyMeshType {
        self.poly_type
    }

    fn face_type(&self) -> PolyFaceType {
        self.face_type
    }

    fn n_verts(&self) -> usize {
        self.verts.len()
    }

    fn vert(&self, i: usize) -> Vertex<D> {
        self.verts[i]
    }

    fn n_elems(&self) -> usize {
        self.elem_to_face_ptr.len() - 1
    }

    fn elem(&self, i: usize) -> impl ExactSizeIterator<Item = (usize, bool)> + Clone + Send {
        let start = self.elem_to_face_ptr[i];
        let end = self.elem_to_face_ptr[i + 1];
        self.elem_to_face[start..end].iter().copied()
    }

    fn etag(&self, i: usize) -> Tag {
        self.etags[i]
    }

    fn n_faces(&self) -> usize {
        self.face_to_node_ptr.len() - 1
    }

    fn face(&self, i: usize) -> impl ExactSizeIterator<Item = usize> + Clone + Send {
        let start = self.face_to_node_ptr[i];
        let end = self.face_to_node_ptr[i + 1];
        self.face_to_node[start..end].iter().copied()
    }

    fn ftag(&self, i: usize) -> Tag {
        self.ftags[i]
    }

    fn elem_gfaces_c<C: Simplex>(
        &self,
        i: usize,
    ) -> Option<impl ExactSizeIterator<Item = C::GEOM<D>> + '_> {
        if matches!(self.face_type, PolyFaceType::Simplices) && C::DIM == self.elem_dim() - 1 {
            Some(self.elem(i).map(|(i_face, orient)| {
                let mut f = C::from_iter(self.face(i_face));
                if !orient {
                    f.invert();
                }
                C::GEOM::from_iter(f.into_iter().map(|i| self.vert(i)))
            }))
        } else {
            None
        }
    }
}

pub fn merge_polylines(polylines: &[&[usize]]) -> Vec<Vec<usize>> {
    let mut res = Vec::new();
    let (polyline, next) = try_merge_polylines(polylines);
    res.push(polyline);
    if !next.is_empty() {
        res.append(&mut merge_polylines(&next));
    }
    res
}

fn try_merge_polylines<'a>(polylines: &[&'a [usize]]) -> (Vec<usize>, Vec<&'a [usize]>) {
    let mut mask = vec![true; polylines.len()];
    let mut connectivity = Vec::with_capacity(polylines.len());

    // add the first edge
    connectivity.extend(polylines[0].iter().copied());
    mask[0] = false;

    loop {
        let mut added_one = false;
        // add at the end?
        let last = *connectivity.last().unwrap();
        for (edg, m) in polylines.iter().zip(mask.iter_mut()) {
            assert!(edg.len() > 1);
            if *m && edg[0] == last {
                *m = false;
                connectivity.extend(edg.iter().copied().skip(1));
                added_one = true;
                break;
            }
        }
        // add at the start?
        let first = *connectivity.first().unwrap();
        for (&edg, m) in polylines.iter().zip(mask.iter_mut()) {
            assert!(edg.len() > 1);
            if *m && edg[edg.len() - 1] == first {
                *m = false;
                let mut tmp = edg.to_vec();
                tmp.extend(connectivity.iter().copied().skip(1));
                connectivity = tmp;
                added_one = true;
                break;
            }
        }
        if !added_one {
            break;
        }
    }

    let unmerged = polylines
        .iter()
        .zip(mask.iter())
        .filter(|&(_, m)| *m)
        .map(|(&e, _)| e)
        .collect::<Vec<_>>();
    (connectivity, unmerged)
}

fn polygon_edges(face: &[usize]) -> impl ExactSizeIterator<Item = [usize; 2]> + '_ {
    let n_edgs = face.len();
    (0..n_edgs).map(move |i_edg| {
        if i_edg == n_edgs - 1 {
            [face[n_edgs - 1], face[0]]
        } else {
            [face[i_edg], face[i_edg + 1]]
        }
    })
}

pub fn merge_polygons(polylines: &[&[usize]]) -> Vec<Vec<usize>> {
    let mut res = Vec::new();
    let (polyline, next) = try_merge_polygons(polylines);
    res.push(polyline);
    if !next.is_empty() {
        res.append(&mut merge_polygons(&next));
    }
    res
}

fn try_merge_two_polygons(p0: &[usize], p1: &[usize]) -> Result<Vec<usize>> {
    for (i0, e0) in polygon_edges(p0).enumerate() {
        for (i1, e1) in polygon_edges(p1).enumerate() {
            if e0[0] == e1[1] && e0[1] == e1[0] {
                let mut p0 = p0.to_vec();
                let mut p1 = p1.to_vec();
                p0.rotate_left(i0 + 1);
                p1.rotate_left(i1 + 1);
                loop {
                    if p0.len() < 3 || p1.len() < 3 {
                        break;
                    }
                    let n0 = p0.len();
                    let n1 = p1.len();
                    if p0[n0 - 2] == p1[1] {
                        p0.pop().unwrap();
                        p1.remove(0);
                    } else if p0[1] == p1[n1 - 2] {
                        p0.remove(0);
                        p1.pop().unwrap();
                    } else {
                        break;
                    }
                }
                let (mut polyline, tmp) = try_merge_polylines(&[&p0, &p1]);
                assert!(tmp.is_empty());
                let n = polyline.len();
                assert_eq!(polyline[0], polyline[n - 1]);
                polyline.pop().unwrap();
                return Ok(polyline);
            }
        }
    }
    Err(Error::from("unable to merge faces"))
}

fn try_merge_polygons<'a>(polygons: &[&'a [usize]]) -> (Vec<usize>, Vec<&'a [usize]>) {
    let mut mask = vec![true; polygons.len()];
    let mut connectivity = Vec::with_capacity(polygons.len());

    // add the first polygon
    connectivity.extend(polygons[0].iter().copied());
    mask[0] = false;

    loop {
        let mut added_one = false;
        for (face, m) in polygons.iter().zip(mask.iter_mut()) {
            if !*m {
                continue;
            }
            if let Ok(res) = try_merge_two_polygons(&connectivity, face) {
                // check that no edge apears more than once
                let tmp = polygon_edges(&res)
                    .map(|e| if e[0] < e[1] { e } else { [e[1], e[0]] })
                    .collect::<FxHashSet<_>>();
                if tmp.len() == res.len() {
                    connectivity = res;
                    added_one = true;
                    *m = false;
                    break;
                }
            }
        }
        if !added_one {
            break;
        }
    }
    let unmerged = polygons
        .iter()
        .zip(mask.iter())
        .filter(|&(_, m)| *m)
        .map(|(&e, _)| e)
        .collect::<Vec<_>>();
    (connectivity, unmerged)
}

#[cfg(test)]
mod tests {
    use crate::{
        dual::merge_polygons,
        mesh::{Mesh3d, box_mesh},
    };

    use super::{PolyMesh, SimplePolyMesh, merge_polylines};

    #[test]
    fn test_merge_polylines() {
        let polyline: [&[usize]; 4] = [&[0, 1], &[3, 4, 5], &[2, 3], &[1, 2]];
        let res = merge_polylines(&polyline);
        assert_eq!(res.len(), 1);
        assert_eq!(res[0], [0, 1, 2, 3, 4, 5]);

        let polyline: [&[usize]; 4] = [&[3, 4, 5], &[2, 3], &[1, 2], &[0, 1]];
        let res = merge_polylines(&polyline);
        assert_eq!(res.len(), 1);
        assert_eq!(res[0], [0, 1, 2, 3, 4, 5]);

        let polyline: [&[usize]; 4] = [&[3, 4, 5], &[2, 3], &[1, 2], &[1, 0]];
        let res = merge_polylines(&polyline);
        assert_eq!(res.len(), 2);
        assert_eq!(res[0], [1, 2, 3, 4, 5]);
        assert_eq!(res[1], [1, 0]);

        let polyline: [&[usize]; 4] = [&[3, 4, 5], &[2, 3], &[2, 1], &[0, 1]];
        let res = merge_polylines(&polyline);
        assert_eq!(res.len(), 3);
        assert_eq!(res[0], [2, 3, 4, 5]);
        assert_eq!(res[1], [2, 1]);
        assert_eq!(res[2], [0, 1]);
    }

    #[test]
    fn test_merge_polygons() {
        let polygons: [&[usize]; 5] = [&[0, 1, 2], &[1, 3, 2], &[2, 3, 4], &[5, 2, 4], &[0, 2, 5]];
        let res = merge_polygons(&polygons);
        assert_eq!(res.len(), 1);
        assert_eq!(res[0], [0, 1, 3, 4, 5]);

        let polygons: [&[usize]; 5] = [&[0, 1, 2], &[2, 3, 4], &[1, 3, 2], &[5, 2, 4], &[0, 2, 5]];
        let res = merge_polygons(&polygons);
        assert_eq!(res.len(), 1);
        assert_eq!(res[0], [0, 1, 3, 4, 5]);

        let polygons: [&[usize]; 5] = [&[0, 1, 2], &[3, 2, 4], &[1, 3, 2], &[5, 2, 4], &[0, 2, 5]];
        let res = merge_polygons(&polygons);
        assert_eq!(res.len(), 2);
        assert_eq!(res[0], [5, 0, 1, 3, 2, 4]);
        assert_eq!(res[1], [3, 2, 4]);
    }

    #[test]
    fn test_merge_polygons_2() {
        let polygons: [&[usize]; 5] = [
            &[0, 1, 7],
            &[1, 2, 3],
            &[3, 4, 5],
            &[5, 6, 7],
            &[7, 1, 3, 5],
        ];
        let res = merge_polygons(&polygons);
        assert_eq!(res.len(), 1);
        assert_eq!(res[0], [7, 0, 1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_merge_polygons_3() {
        let polygons: [&[usize]; 8] = [
            &[0, 3, 1],
            &[3, 4, 1],
            &[1, 4, 2],
            &[4, 5, 2],
            &[0, 9, 6, 3],
            &[3, 6, 7, 4],
            &[7, 8, 5, 4],
            &[9, 10, 8, 7, 6],
        ];
        let res = merge_polygons(&polygons);
        assert_eq!(res.len(), 1);
        assert_eq!(res[0], [8, 5, 2, 1, 0, 9, 10]);

        let polygons: [&[usize]; 8] = [
            &[0, 3, 1],
            &[3, 4, 1],
            &[1, 4, 2],
            &[4, 5, 2],
            &[0, 9, 6, 3],
            &[9, 10, 8, 7, 6],
            &[7, 8, 5, 4],
            &[3, 6, 7, 4],
        ];
        let res = merge_polygons(&polygons);
        assert_eq!(res.len(), 1);
        assert_eq!(res[0], [5, 2, 1, 0, 9, 10, 8]);
    }

    #[test]
    fn test_vtk_polyhedra() {
        let msh: Mesh3d = box_mesh(1.0, 2, 2.0, 2, 3.0, 2);
        let poly = SimplePolyMesh::from_mesh(&msh);
        poly.write_vtk("poly3d.vtu").unwrap();
    }
}
