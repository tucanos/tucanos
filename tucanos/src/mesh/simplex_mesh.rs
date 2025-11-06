use super::{topology::Topology, vector::Vector};
use crate::{Dim, Error, Result, Tag, TopoTag};
use log::{debug, warn};
use nalgebra::SVector;
use rayon::{
    prelude::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator},
    slice::ParallelSliceMut,
};
use rustc_hash::FxHashMap;
use std::marker::PhantomData;
use tmesh::{
    Vertex,
    graph::CSRGraph,
    mesh::{Edge, GSimplex, Idx, Mesh, MutMesh, Simplex, SubMesh, get_face_to_elem},
};

/// A mesh containing a single type of elements in D-dimensions
#[derive(Debug, Default)]
pub struct SimplexMesh<T: Idx, const D: usize, C: Simplex<T>> {
    /// Coordinates of the vertices (length = D * # of vertices)
    pub(super) verts: Vector<Vertex<D>>,
    /// Element connectivity (length = # of vertices per element * # of elements)
    pub(super) elems: Vector<C>,
    /// Element tags (length = # of elements)
    pub(super) etags: Vector<Tag>,
    /// Face connectivity (length = # of vertices per face * # of faces)
    pub(super) faces: Vector<C::FACE>,
    /// Faces tags (length = # of faces)
    pub(super) ftags: Vector<Tag>,
    point: PhantomData<SVector<f64, D>>,
    elem: PhantomData<C>,
    /// Face to element connectitivity stored as a HashMap taking the face vertices (sorted) and returning
    /// a vector of element Ids
    faces_to_elems: Option<FxHashMap<C::FACE, tmesh::mesh::twovec::Vec<T>>>,
    /// Vertex-to-element connectivity stored in CSR format
    vertex_to_elems: Option<CSRGraph<T>>,
    /// Element-to-element connectivity stored in CSR format
    elem_to_elems: Option<CSRGraph<T>>,
    /// Edges (length = # of edges)
    edges: Option<Vec<Edge<T>>>,
    /// Vertex-to-vertex (~edges) connectivity stored in CSR format
    vertex_to_vertices: Option<CSRGraph<T>>,
    /// Element volumes (length = # of elements)
    elem_vol: Option<Vec<f64>>,
    /// Vertex volumes (length = # of vertices)
    /// The volume of a vertex in the weighted average of the neighboring element volumes
    /// It can be seen as the volume of a dual cell
    /// sum(elem_vol) = sum(vert_vol)
    vert_vol: Option<Vec<f64>>,
    /// Topology
    topo: Option<Topology>,
    /// Vertex tags
    vtags: Option<Vec<TopoTag>>,
}

impl<T: Idx, const D: usize, C: Simplex<T>> Clone for SimplexMesh<T, D, C> {
    fn clone(&self) -> Self {
        let mut res = Self::new_with_vector(
            self.verts.clone(),
            self.elems.clone(),
            self.etags.clone(),
            self.faces.clone(),
            self.ftags.clone(),
        );
        if let Some(topo) = self.topo.as_ref() {
            res.topo = Some(topo.clone());
        }
        if let Some(vtags) = self.vtags.as_ref() {
            res.vtags = Some(vtags.clone());
        }
        res
    }
}

impl<T: Idx, const D: usize, C: Simplex<T>> SimplexMesh<T, D, C> {
    /// Create a new `SimplexMesh`. The extra connectivity information is not built
    #[must_use]
    pub fn new_with_vector(
        verts: Vector<Vertex<D>>,
        elems: Vector<C>,
        etags: Vector<Tag>,
        faces: Vector<C::FACE>,
        ftags: Vector<Tag>,
    ) -> Self {
        debug!(
            "Create a SimplexMesh with {} {}D vertices / {} elems / {} faces",
            verts.len(),
            D,
            elems.len(),
            faces.len(),
        );
        Self {
            verts,
            elems,
            etags,
            faces,
            ftags,
            point: PhantomData,
            elem: PhantomData,
            faces_to_elems: None,
            vertex_to_elems: None,
            elem_to_elems: None,
            edges: None,
            vertex_to_vertices: None,
            elem_vol: None,
            vert_vol: None,
            topo: None,
            vtags: None,
        }
    }

    #[must_use]
    pub fn new_with_vec(
        verts: Vec<Vertex<D>>,
        elems: Vec<C>,
        etags: Vec<Tag>,
        faces: Vec<C::FACE>,
        ftags: Vec<Tag>,
    ) -> Self {
        Self::new_with_vector(
            verts.into(),
            elems.into(),
            etags.into(),
            faces.into(),
            ftags.into(),
        )
    }

    /// Get the total volume of a mesh
    #[must_use]
    pub fn vol(&self) -> f64 {
        self.gelems().map(|ge| ge.vol()).sum()
    }

    /// Compute the face-to-element connectivity
    pub fn compute_face_to_elems(&mut self) -> &FxHashMap<C::FACE, tmesh::mesh::twovec::Vec<T>> {
        debug!("Compute the face to element connectivity");
        if self.faces_to_elems.is_none() {
            self.faces_to_elems = Some(get_face_to_elem(self.elems()));
        } else {
            warn!("Face to element connectivity already computed");
        }
        self.faces_to_elems.as_ref().unwrap()
    }

    /// Clear the face-to-element connectivity
    pub fn clear_face_to_elems(&mut self) {
        debug!("Delete the face to element connectivity");
        self.faces_to_elems = None;
    }

    /// Get the face-to-element connectivity
    pub fn get_face_to_elems(
        &self,
    ) -> Result<&FxHashMap<<C as Simplex<T>>::FACE, tmesh::mesh::twovec::Vec<T>>> {
        if self.faces_to_elems.is_none() {
            Err(Error::from("Face to element connectivity not computed"))
        } else {
            Ok(self.faces_to_elems.as_ref().unwrap())
        }
    }

    /// Compute the vertex-to-element connectivity
    pub fn compute_vertex_to_elems(&mut self) -> &CSRGraph<T> {
        debug!("Compute the vertex to element connectivity");

        if self.vertex_to_elems.is_none() {
            self.vertex_to_elems = Some(CSRGraph::transpose(self.elems.iter(), None));
        } else {
            warn!("Vertex to element connectivity already computed");
        }
        self.vertex_to_elems.as_ref().unwrap()
    }

    /// Clear the vertex-to-element connectivity
    pub fn clear_vertex_to_elems(&mut self) {
        debug!("Delete the vertex to element connectivity");
        self.vertex_to_elems = None;
    }

    /// Get the vertex-to-element connectivity
    pub fn get_vertex_to_elems(&self) -> Result<&CSRGraph<T>> {
        if self.vertex_to_elems.is_none() {
            Err(Error::from("Vertex to element connectivity not computed"))
        } else {
            Ok(self.vertex_to_elems.as_ref().unwrap())
        }
    }

    /// Compute the element-to-element connectivity
    /// face-to-element connectivity is computed if not available
    pub fn compute_elem_to_elems(&mut self) -> &CSRGraph<T> {
        debug!("Compute the element to element connectivity");
        if self.elem_to_elems.is_none() {
            if self.faces_to_elems.is_none() {
                self.compute_face_to_elems();
            }
            let f2e = self.faces_to_elems.as_ref().unwrap();

            let mut g = Vec::new();
            for val in f2e.values() {
                for (i, i_elem) in val.iter().copied().enumerate() {
                    for j_elem in val.iter().skip(i + 1).copied() {
                        g.push([i_elem, j_elem]);
                    }
                }
            }
            self.elem_to_elems = Some(CSRGraph::from_edges(g.iter().copied(), None));
        } else {
            warn!("Element to element connectivity already computed");
        }
        self.elem_to_elems.as_ref().unwrap()
    }

    /// Clear the element-to-element connectivity
    pub fn clear_elem_to_elems(&mut self) {
        debug!("Delete the element to element connectivity");
        self.elem_to_elems = None;
    }

    /// Get the element-to-element connectivity
    pub fn get_elem_to_elems(&self) -> Result<&CSRGraph<T>> {
        if self.elem_to_elems.is_none() {
            Err(Error::from("Element to element connectivity not computed"))
        } else {
            Ok(self.elem_to_elems.as_ref().unwrap())
        }
    }

    /// Compute the edges
    pub fn compute_edges(&mut self) -> &Vec<Edge<T>> {
        debug!("Compute the edges");
        if self.edges.is_none() {
            self.edges = Some(self.edges().keys().copied().collect());
        } else {
            warn!("Edges already computed");
        }
        self.edges.as_ref().unwrap()
    }

    /// Clear the edges
    pub fn clear_edges(&mut self) {
        debug!("Delete the edges");
        self.edges = None;
    }

    /// Get the the edges
    pub fn get_edges(&self) -> Result<&[Edge<T>]> {
        if self.edges.is_none() {
            Err(Error::from("Edges not computed"))
        } else {
            Ok(self.edges.as_ref().unwrap())
        }
    }

    /// Compute the vertex-to-vertex connectivity
    /// Edges are computed if not available
    pub fn compute_vertex_to_vertices(&mut self) -> &CSRGraph<T> {
        debug!("Compute the vertex to vertex connectivity");
        if self.vertex_to_vertices.is_none() {
            if self.edges.is_none() {
                self.compute_edges();
            }
            self.vertex_to_vertices = Some(CSRGraph::from_edges(
                self.edges.as_ref().unwrap().iter().map(|e| [e[0], e[1]]),
                None,
            ));
        } else {
            warn!("Vertex to vertex connectivity already computed");
        }
        self.vertex_to_vertices.as_ref().unwrap()
    }

    /// Clear the vertex-to-vertex connectivity
    pub fn clear_vertex_to_vertices(&mut self) {
        debug!("Delete the vertex to vertex connectivity");
        self.vertex_to_vertices = None;
    }

    /// Get the vertex-to-vertes connectivity
    pub fn get_vertex_to_vertices(&self) -> Result<&CSRGraph<T>> {
        if self.vertex_to_vertices.is_none() {
            Err(Error::from("Vertex to vertex connectivity not computed"))
        } else {
            Ok(self.vertex_to_vertices.as_ref().unwrap())
        }
    }

    /// Compute the volume and vertex volumes
    pub fn compute_volumes(&mut self) -> (&Vec<f64>, &Vec<f64>) {
        debug!("Compute the vertex & element volumes");
        if self.elem_vol.is_none() {
            let mut elem_vol = vec![0.0; self.n_elems().try_into().unwrap()];
            let mut node_vol = vec![0.0; self.n_verts().try_into().unwrap()];
            let fac = 1.0 / C::N_VERTS as f64;
            for (i_elem, e) in self.elems().enumerate() {
                let v = self.gelem(&e).vol();
                elem_vol[i_elem] = v;
                for i in e.into_iter() {
                    node_vol[i.try_into().unwrap()] += fac * v;
                }
            }
            self.elem_vol = Some(elem_vol);
            self.vert_vol = Some(node_vol);
        } else {
            warn!("Volumes already computed");
        }
        (
            self.elem_vol.as_ref().unwrap(),
            self.vert_vol.as_ref().unwrap(),
        )
    }

    /// Clear the volume and vertex volumes
    pub fn clear_volumes(&mut self) {
        debug!("Delete the vertex & element volumes");
        self.elem_vol = None;
        self.vert_vol = None;
    }

    /// Get the vertex volumes
    pub fn get_vertex_volumes(&self) -> Result<&[f64]> {
        if self.vert_vol.is_none() {
            Err(Error::from("Volumes not computed"))
        } else {
            Ok(self.vert_vol.as_ref().unwrap())
        }
    }

    /// Get the element volumes
    pub fn get_elem_volumes(&self) -> Result<&[f64]> {
        if self.elem_vol.is_none() {
            Err(Error::from("Volumes not computed"))
        } else {
            Ok(self.elem_vol.as_ref().unwrap())
        }
    }

    /// Convert a field defined at the element centers (P0) to a field defined at the vertices (P1)
    /// using a weighted average. For metric fields, use `elem_data_to_vertex_data_metric`
    /// vertex-to-element connectivity and volumes are required
    pub fn elem_data_to_vertex_data(&self, v: &[f64]) -> Result<Vec<f64>> {
        debug!("Convert element data to vertex data");
        if self.vertex_to_elems.is_none() {
            return Err(Error::from("vertex to element connectivity not computed"));
        }
        if self.elem_vol.is_none() {
            return Err(Error::from("element volumes not computed"));
        }
        if self.vert_vol.is_none() {
            return Err(Error::from("node volumes not computed"));
        }

        let n_elems = self.n_elems().try_into().unwrap();
        let n_verts = self.n_verts().try_into().unwrap();
        assert_eq!(v.len() % n_elems, 0);

        let n_comp = v.len() / n_elems;

        let mut res = vec![0.; n_comp * n_verts];

        let v2e = self.vertex_to_elems.as_ref().unwrap();
        let elem_vol = self.elem_vol.as_ref().unwrap();
        let node_vol = self.vert_vol.as_ref().unwrap();

        res.par_chunks_mut(n_comp)
            .enumerate()
            .for_each(|(i_vert, vals)| {
                for i_elem in v2e.row(i_vert.try_into().unwrap()).iter().copied() {
                    let w = elem_vol[i_elem.try_into().unwrap()] / C::N_VERTS as f64;
                    for i_comp in 0..n_comp {
                        vals[i_comp] += w * v[n_comp * i_elem.try_into().unwrap() + i_comp];
                    }
                }
                let w = node_vol[i_vert];
                for v in vals.iter_mut() {
                    *v /= w;
                }
            });

        Ok(res)
    }

    /// Convert a field defined at the vertices (P1) to a field defined at the element centers (P0)
    /// For metric fields, use `elem_data_to_vertex_data_metric`
    pub fn vertex_data_to_elem_data(&self, v: &[f64]) -> Result<Vec<f64>> {
        debug!("Convert vertex data to element data");
        let n_elems = self.n_elems().try_into().unwrap();
        let n_verts = self.n_verts().try_into().unwrap();
        assert_eq!(v.len() % n_verts, 0);

        let n_comp = v.len() / n_verts;

        let mut res = vec![0.; n_comp * n_elems];

        let f = 1. / C::N_VERTS as f64;
        res.par_chunks_mut(n_comp)
            .zip(self.par_elems())
            .for_each(|(vals, e)| {
                for i_comp in 0..n_comp {
                    for i_vert in e.into_iter() {
                        vals[i_comp] += f * v[n_comp * i_vert.try_into().unwrap() + i_comp];
                    }
                }
            });

        Ok(res)
    }

    /// Compute the mesh topology
    pub fn compute_topology(&mut self) -> &Topology {
        if self.topo.is_none() {
            let mut topo = Topology::new(C::DIM as Dim);
            let vtags = topo.update_from_elems_and_faces(
                &self.elems,
                &self.etags,
                &self.faces,
                &self.ftags,
            );
            self.topo = Some(topo);
            self.vtags = Some(vtags);
        } else {
            warn!("Topology already computed");
        }
        self.topo.as_ref().unwrap()
    }

    /// Compute the mesh topology but updating an existing one
    pub fn compute_topology_from(&mut self, mut topo: Topology) {
        if self.topo.is_none() {
            let vtags = topo.update_from_elems_and_faces(
                &self.elems,
                &self.etags,
                &self.faces,
                &self.ftags,
            );
            self.topo = Some(topo);
            self.vtags = Some(vtags);
        } else {
            warn!("Topology already computed");
        }
    }

    /// Clear the mesh topology
    pub fn clear_topology(&mut self) {
        self.topo = None;
        self.vtags = None;
    }

    /// Get the topology
    pub fn get_topology(&self) -> Result<&Topology> {
        if self.topo.is_none() {
            Err(Error::from("Topology not computed"))
        } else {
            Ok(self.topo.as_ref().unwrap())
        }
    }

    /// Get the vertex tags
    pub fn get_vertex_tags(&self) -> Result<&[TopoTag]> {
        if self.topo.is_none() {
            Err(Error::from("Topology not computed"))
        } else {
            Ok(self.vtags.as_ref().unwrap())
        }
    }

    pub fn clear_all(&mut self) {
        self.faces_to_elems = None;
        self.vertex_to_elems = None;
        self.elem_to_elems = None;
        self.edges = None;
        self.vertex_to_vertices = None;
        self.elem_vol = None;
        self.vert_vol = None;
        self.topo = None;
        self.vtags = None;
    }

    /// Modify the face tags
    pub fn update_face_tags<F: FnMut(Tag) -> Tag>(&mut self, mut new_ftags: F) {
        self.ftags
            .as_std_mut()
            .iter_mut()
            .for_each(|t| *t = new_ftags(*t));
    }
}

impl<T: Idx, const D: usize, C: Simplex<T>> Mesh<T, D, C> for SimplexMesh<T, D, C> {
    fn new(
        verts: &[Vertex<D>],
        elems: &[C],
        etags: &[Tag],
        faces: &[C::FACE],
        ftags: &[Tag],
    ) -> Self {
        Self::new_with_vector(
            verts.to_vec().into(),
            elems.to_vec().into(),
            etags.to_vec().into(),
            faces.to_vec().into(),
            ftags.to_vec().into(),
        )
    }

    fn empty() -> Self {
        Self {
            verts: Vec::new().into(),
            elems: Vec::new().into(),
            etags: Vec::new().into(),
            faces: Vec::new().into(),
            ftags: Vec::new().into(),
            point: PhantomData,
            elem: PhantomData,
            faces_to_elems: None,
            vertex_to_elems: None,
            elem_to_elems: None,
            edges: None,
            vertex_to_vertices: None,
            elem_vol: None,
            vert_vol: None,
            topo: None,
            vtags: None,
        }
    }

    /// Get the number of vertices
    fn n_verts(&self) -> T {
        self.verts.len().try_into().unwrap()
    }

    /// Get the number or elements
    fn n_elems(&self) -> T {
        self.elems.len().try_into().unwrap()
    }

    /// Get the number of faces
    fn n_faces(&self) -> T {
        self.faces.len().try_into().unwrap()
    }

    /// Get the i-th vertex
    fn vert(&self, idx: T) -> Vertex<D> {
        self.verts.index(idx.try_into().unwrap())
    }

    /// Get an iterator through the vertices
    fn verts(&self) -> impl ExactSizeIterator<Item = Vertex<D>> + Clone + '_ {
        self.verts.iter()
    }

    /// Get a parallel iterator through the vertices
    fn par_verts(&self) -> impl IndexedParallelIterator<Item = Vertex<D>> + Clone + '_ {
        (0..self.n_verts().try_into().unwrap())
            .into_par_iter()
            .map(|i| self.vert(i.try_into().unwrap()))
    }

    /// Get the i-th element
    fn elem(&self, idx: T) -> C {
        self.elems.index(idx.try_into().unwrap())
    }

    /// Get an iterator through the elements
    fn elems(&self) -> impl ExactSizeIterator<Item = C> + Clone + '_ {
        self.elems.iter()
    }

    /// Get a parallel iterator through the elements
    fn par_elems(&self) -> impl IndexedParallelIterator<Item = C> + Clone + '_ {
        (0..self.n_elems().try_into().unwrap())
            .into_par_iter()
            .map(|i| self.elem(i.try_into().unwrap()))
    }

    /// Get the i-th element tag
    fn etag(&self, idx: T) -> Tag {
        self.etags.index(idx.try_into().unwrap())
    }

    /// Get an iterator through the elements tags
    fn etags(&self) -> impl ExactSizeIterator<Item = Tag> + Clone + '_ {
        self.etags.iter()
    }

    /// Get a parallel iterator through the elements tags
    fn par_etags(&self) -> impl IndexedParallelIterator<Item = Tag> + Clone + '_ {
        (0..self.n_elems().try_into().unwrap())
            .into_par_iter()
            .map(|i| self.etag(i.try_into().unwrap()))
    }

    fn gelem(&self, e: &C) -> C::GEOM<D> {
        C::GEOM::from_iter(e.into_iter().map(|i| self.vert(i)))
    }

    /// Get an iterator through the geometric elements
    fn gelems(&self) -> impl ExactSizeIterator<Item = C::GEOM<D>> + Clone + '_ {
        self.elems().map(|e| self.gelem(&e))
    }

    /// Get a parallel iterator through the geometric elements
    fn par_gelems(&self) -> impl IndexedParallelIterator<Item = C::GEOM<D>> + Clone + '_ {
        self.par_elems().map(|e| self.gelem(&e))
    }

    /// Get the i-th face
    fn face(&self, idx: T) -> C::FACE {
        self.faces.index(idx.try_into().unwrap())
    }

    /// Get an iterator through the faces
    fn faces(&self) -> impl ExactSizeIterator<Item = C::FACE> + Clone + '_ {
        self.faces.iter()
    }

    /// Get a parallel iterator through the faces
    fn par_faces(&self) -> impl IndexedParallelIterator<Item = C::FACE> + Clone + '_ {
        (0..self.n_faces().try_into().unwrap())
            .into_par_iter()
            .map(|i| self.face(i.try_into().unwrap()))
    }

    /// Get the i-th face tag
    fn ftag(&self, idx: T) -> Tag {
        self.ftags.index(idx.try_into().unwrap())
    }

    /// Get an iterator through the face tags
    fn ftags(&self) -> impl ExactSizeIterator<Item = Tag> + Clone + '_ {
        self.ftags.iter()
    }

    /// Get a parallel iterator through the face tags
    fn par_ftags(&self) -> impl IndexedParallelIterator<Item = Tag> + Clone + '_ {
        (0..self.n_faces().try_into().unwrap())
            .into_par_iter()
            .map(|i| self.ftag(i.try_into().unwrap()))
    }

    fn gface(&self, f: &C::FACE) -> <C::FACE as Simplex<T>>::GEOM<D> {
        <C::FACE as Simplex<T>>::GEOM::from_iter(f.into_iter().map(|i| self.vert(i)))
    }

    /// Get an iterator through the geometric faces
    fn gfaces(
        &self,
    ) -> impl ExactSizeIterator<Item = <C::FACE as Simplex<T>>::GEOM<D>> + Clone + '_ {
        self.faces().map(|f| self.gface(&f))
    }

    /// Get an iterator through the geometric faces
    fn par_gfaces(
        &self,
    ) -> impl IndexedParallelIterator<Item = <C::FACE as Simplex<T>>::GEOM<D>> + Clone + '_ {
        self.par_faces().map(|f| self.gface(&f))
    }

    fn add_verts<I: ExactSizeIterator<Item = Vertex<D>>>(&mut self, v: I) {
        self.verts.extend(v);
    }

    fn invert_elem(&mut self, i: T) {
        self.elems.index_mut(i.try_into().unwrap()).invert();
    }

    fn add_elems<I1: ExactSizeIterator<Item = C>, I2: ExactSizeIterator<Item = Tag>>(
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

    fn add_elems_and_tags<I: ExactSizeIterator<Item = (C, Tag)>>(&mut self, elems_and_tags: I) {
        self.elems.reserve(elems_and_tags.len());
        self.etags.reserve(elems_and_tags.len());
        for (e, t) in elems_and_tags {
            self.elems.push(e);
            self.etags.push(t);
        }
    }

    fn invert_face(&mut self, i: T) {
        self.faces.index_mut(i.try_into().unwrap()).invert();
    }

    fn add_faces<
        I1: ExactSizeIterator<Item = <C as Simplex<T>>::FACE>,
        I2: ExactSizeIterator<Item = Tag>,
    >(
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

    fn add_faces_and_tags<I: ExactSizeIterator<Item = (<C as Simplex<T>>::FACE, Tag)>>(
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

    fn set_etags<I: ExactSizeIterator<Item = Tag>>(&mut self, tags: I) {
        self.etags
            .as_std_mut()
            .iter_mut()
            .zip(tags)
            .for_each(|(x, y)| *x = y);
    }
}

impl<T: Idx, const D: usize, C: Simplex<T>> MutMesh<T, D, C> for SimplexMesh<T, D, C> {
    /// Get an iterator through the vertices
    fn verts_mut(&mut self) -> impl ExactSizeIterator<Item = &mut Vertex<D>> + '_ {
        self.verts.as_std_mut().iter_mut()
    }

    /// Get an iterator through the elements
    fn elems_mut<'a>(&mut self) -> impl ExactSizeIterator<Item = &mut C> + '_
    where
        C: 'a,
    {
        self.elems.as_std_mut().iter_mut()
    }

    /// Get an iterator through the elements tags
    fn etags_mut(&mut self) -> impl ExactSizeIterator<Item = &mut Tag> + '_ {
        self.etags.as_std_mut().iter_mut()
    }

    /// Get an iterator through the faces
    fn faces_mut<'a>(&mut self) -> impl ExactSizeIterator<Item = &mut C::FACE> + '_
    where
        C: 'a,
    {
        self.faces.as_std_mut().iter_mut()
    }

    /// Get an iterator through the face tags
    fn ftags_mut(&mut self) -> impl ExactSizeIterator<Item = &mut Tag> + '_ {
        self.ftags.as_std_mut().iter_mut()
    }
}

pub type SubSimplexMesh<T: Idx, const D: usize, C: Simplex<T>> =
    SubMesh<T, D, C, SimplexMesh<T, D, C>>;

#[cfg(test)]
mod tests {

    use crate::{
        Result,
        mesh::{
            SubSimplexMesh,
            test_meshes::{test_mesh_2d, test_mesh_3d},
        },
    };
    use tmesh::mesh::{Edge, GSimplex, Mesh, Triangle};

    #[test]
    fn test_2d() {
        let mesh = test_mesh_2d();

        assert_eq!(mesh.n_verts(), 4);
        assert_eq!(mesh.n_faces(), 4);
        assert_eq!(mesh.n_elems(), 2);

        assert_eq!(mesh.elems.index(1), Triangle::from([0, 2, 3]));
        assert_eq!(mesh.faces.index(1), Edge::from([1, 2]));

        let v: f64 = mesh.gelems().map(|ge| ge.vol()).sum();
        assert!((v - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_3d() {
        let mesh = test_mesh_3d();

        assert_eq!(mesh.n_verts(), 8);
        assert_eq!(mesh.n_faces(), 12);
        assert_eq!(mesh.n_elems(), 5);

        let v: f64 = mesh.gelems().map(|ge| ge.vol()).sum();
        assert!(f64::abs(v - 1.0) < 1e-12);
    }

    #[test]
    fn test_cell_to_node() -> Result<()> {
        let mesh = test_mesh_3d();
        let mut mesh = mesh.split().split().split();

        let n_elems: usize = mesh.n_elems().try_into().unwrap();
        let n_verts: usize = mesh.n_verts().try_into().unwrap();
        let mut v_e = Vec::with_capacity(3 * n_elems);

        for c in mesh.gelems().map(|ge| ge.center()) {
            v_e.push(c[0] + c[1]);
            v_e.push(c[0] * c[1]);
            v_e.push(c[0] - c[1]);
        }
        let v_v = mesh.elem_data_to_vertex_data(&v_e);
        assert!(v_v.is_err());

        mesh.compute_vertex_to_elems();
        let v_v = mesh.elem_data_to_vertex_data(&v_e);
        assert!(v_v.is_err());

        mesh.compute_volumes();
        let v_v = mesh.elem_data_to_vertex_data(&v_e)?;
        assert_eq!(v_v.len(), 3 * n_verts);

        for (i_vert, pt) in mesh.verts().enumerate() {
            assert!(f64::abs(v_v[3 * i_vert] - (pt[0] + pt[1])) < 0.15);
            assert!(f64::abs(v_v[3 * i_vert + 1] - (pt[0] * pt[1])) < 0.15);
            assert!(f64::abs(v_v[3 * i_vert + 2] - (pt[0] - pt[1])) < 0.15);
        }

        Ok(())
    }

    #[test]
    fn test_vertex_to_elem() {
        let mesh = test_mesh_3d();
        let mesh = mesh.split().split().split();

        let n_elems: usize = mesh.n_elems().try_into().unwrap();
        let mut v_v = Vec::with_capacity(3 * n_elems);

        for p in mesh.verts() {
            v_v.push(p[0] + p[1]);
            v_v.push(p[0] * p[1]);
            v_v.push(p[0] - p[1]);
        }

        let v_e = mesh.vertex_data_to_elem_data(&v_v).unwrap();
        assert_eq!(v_e.len(), 3 * n_elems);

        for (i_elem, pt) in mesh.gelems().map(|ge| ge.center()).enumerate() {
            assert!(f64::abs(v_e[3 * i_elem] - (pt[0] + pt[1])) < 1e-10);
            assert!(f64::abs(v_e[3 * i_elem + 1] - (pt[0] * pt[1])) < 0.1);
            assert!(f64::abs(v_e[3 * i_elem + 2] - (pt[0] - pt[1])) < 1e-10);
        }
    }

    #[test]
    fn test_extract_2d() {
        let mesh = test_mesh_2d().split();

        let sub_mesh = SubSimplexMesh::new(&mesh, |t| t == 1);
        let smesh = sub_mesh.mesh;
        let ids = sub_mesh.parent_vert_ids;
        assert_eq!(smesh.n_verts(), 6);
        assert_eq!(smesh.n_elems(), 4);
        assert_eq!(smesh.n_faces(), 4);
        assert!(f64::abs(smesh.gelems().map(|ge| ge.vol()).sum::<f64>() - 0.5) < 1e-10);

        assert_eq!(ids.len(), 6);
        for i in ids {
            let p = mesh.verts.index(i);
            assert!(p[0] - p[1] > -1e-10);
        }
    }
}
