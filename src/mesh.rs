use crate::{
    geom_elems::GElem,
    graph::{reindex, CSRGraph},
    metric::IsoMetric,
    spatialindex::{self, DefaultObjectIndex, DefaultPointIndex, ObjectIndex, PointIndex},
    topo_elems::{get_face_to_elem, Elem},
    topology::Topology,
    twovec, Error, Idx, Result, Tag, TopoTag,
};
use log::{debug, info, warn};
use nalgebra::SVector;
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::{HashMap, HashSet};
use std::hash::BuildHasherDefault;
use std::marker::PhantomData;

/// A mesh containing a single type of elements in D-dimensions
#[derive(Debug, Default)]
pub struct SimplexMesh<const D: usize, E: Elem> {
    /// Coordinates of the vertices (length = D * # of vertices)
    verts: Vec<Point<D>>,
    /// Element connectivity (length = # of vertices per element * # of elements)
    elems: Vec<E>,
    /// Element tags (length = # of elements)
    etags: Vec<Tag>,
    /// Face connectivity (length = # of vertices per face * # of faces)
    faces: Vec<E::Face>,
    /// Faces tags (length = # of faces)
    ftags: Vec<Tag>,
    point: PhantomData<SVector<f64, D>>,
    elem: PhantomData<E>,
    /// Face to element connectitivity stored as a HashMap taking the face vertices (sorted) and returning
    /// a vector of element Ids
    faces_to_elems: Option<FxHashMap<E::Face, twovec::Vec<u32>>>,
    /// Vertex-to-element connectivity stored in CSR format
    vertex_to_elems: Option<CSRGraph>,
    /// Element-to-element connectivity stored in CSR format
    elem_to_elems: Option<CSRGraph>,
    /// Edges (length = # of edges)
    edges: Option<Vec<[Idx; 2]>>,
    /// Vertex-to-vertex (~edges) connectivity stored in CSR format
    vertex_to_vertices: Option<CSRGraph>,
    /// Element volumes (length = # of elements)
    elem_vol: Option<Vec<f64>>,
    /// Vertex volumes (length = # of vertices)
    /// The volume of a vertex in the weighted average of the neighboring element volumes
    /// It can be seen as the volume of a dual cell
    /// sum(elem_vol) = sum(vert_vol)
    vert_vol: Option<Vec<f64>>,
    /// Octree
    tree: Option<spatialindex::DefaultObjectIndex>,
    /// Topology
    topo: Option<Topology>,
    /// Vertex tags
    vtags: Option<Vec<TopoTag>>,
}

impl<const D: usize, E: Elem> Clone for SimplexMesh<D, E> {
    fn clone(&self) -> Self {
        Self::new(
            self.verts.clone(),
            self.elems.clone(),
            self.etags.clone(),
            self.faces.clone(),
            self.ftags.clone(),
        )
    }
}

pub struct SubSimplexMesh<const D: usize, E: Elem> {
    pub mesh: SimplexMesh<D, E>,
    pub parent_vert_ids: Vec<Idx>,
    pub parent_elem_ids: Vec<Idx>,
    pub parent_face_ids: Vec<Idx>,
}

pub type Point<const D: usize> = SVector<f64, D>;

impl<const D: usize, E: Elem> SimplexMesh<D, E> {
    /// Create a new `SimplexMesh`. The extra connectivity information is not built
    #[must_use]
    pub fn new(
        verts: Vec<Point<D>>,
        elems: Vec<E>,
        etags: Vec<Tag>,
        faces: Vec<E::Face>,
        ftags: Vec<Tag>,
    ) -> Self {
        info!(
            "Create a SimplexMesh with {} {}D vertices / {} {} / {} {}",
            verts.len(),
            D,
            elems.len() / E::N_VERTS as usize,
            E::NAME,
            faces.len() / E::Face::N_VERTS as usize,
            E::Face::NAME
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
            tree: None,
            topo: None,
            vtags: None,
        }
    }

    #[must_use]
    pub fn empty() -> Self {
        Self {
            verts: Vec::new(),
            elems: Vec::new(),
            etags: Vec::new(),
            faces: Vec::new(),
            ftags: Vec::new(),
            point: PhantomData,
            elem: PhantomData,
            faces_to_elems: None,
            vertex_to_elems: None,
            elem_to_elems: None,
            edges: None,
            vertex_to_vertices: None,
            elem_vol: None,
            vert_vol: None,
            tree: None,
            topo: None,
            vtags: None,
        }
    }

    /// Get the number of vertices
    #[must_use]
    pub fn n_verts(&self) -> Idx {
        self.verts.len() as Idx
    }

    /// Get the number or elements
    #[must_use]
    pub fn n_elems(&self) -> Idx {
        self.elems.len() as Idx
    }

    /// Get the number of faces
    #[must_use]
    pub fn n_faces(&self) -> Idx {
        self.faces.len() as Idx
    }

    /// Get the i-th vertex
    #[must_use]
    pub fn vert(&self, idx: Idx) -> Point<D> {
        self.verts[idx as usize]
    }

    /// Get an iterator through the vertices
    #[must_use]
    pub fn verts(&self) -> impl ExactSizeIterator<Item = Point<D>> + '_ {
        self.verts.iter().copied()
    }

    /// Get an iterator through the vertices
    pub fn mut_verts(&mut self) -> impl ExactSizeIterator<Item = &mut Point<D>> + '_ {
        self.verts.iter_mut()
    }

    /// Get the i-th element
    #[must_use]
    pub fn elem(&self, idx: Idx) -> E {
        self.elems[idx as usize]
    }

    /// Get an iterator through the elements
    #[must_use]
    pub fn elems(&self) -> impl ExactSizeIterator<Item = E> + '_ {
        self.elems.iter().copied()
    }

    /// Get an iterator through the elements
    pub fn mut_elems(&mut self) -> impl ExactSizeIterator<Item = &mut E> + '_ {
        self.elems.iter_mut()
    }

    /// Get the i-th element tag
    #[must_use]
    pub fn etag(&self, idx: Idx) -> Tag {
        self.etags[idx as usize]
    }

    /// Get an iterator through the elements tags
    #[must_use]
    pub fn etags(&self) -> impl ExactSizeIterator<Item = Tag> + '_ {
        self.etags.iter().copied()
    }

    /// Get an iterator through the elements tags
    pub fn mut_etags(&mut self) -> impl ExactSizeIterator<Item = &mut Tag> + '_ {
        self.etags.iter_mut()
    }

    pub fn gelem(&self, e: E) -> E::Geom<D, IsoMetric<D>> {
        E::Geom::from_verts(
            e.iter()
                .map(|&i| (self.verts[i as usize], IsoMetric::<D>::from(1.0))),
        )
    }

    /// Get an iterator through the geometric elements
    #[must_use]
    pub fn gelems(&self) -> impl ExactSizeIterator<Item = E::Geom<D, IsoMetric<D>>> + '_ {
        self.elems().map(|e| self.gelem(e))
    }

    /// Get the i-th face
    #[must_use]
    pub fn face(&self, idx: Idx) -> E::Face {
        self.faces[idx as usize]
    }

    /// Get an iterator through the faces
    #[must_use]
    pub fn faces(&self) -> impl ExactSizeIterator<Item = E::Face> + '_ {
        self.faces.iter().copied()
    }

    /// Get an iterator through the faces
    pub fn mut_faces(&mut self) -> impl ExactSizeIterator<Item = &mut E::Face> + '_ {
        self.faces.iter_mut()
    }

    /// Get the i-th face tag
    #[must_use]
    pub fn ftag(&self, idx: Idx) -> Tag {
        self.ftags[idx as usize]
    }

    /// Get an iterator through the face tags
    #[must_use]
    pub fn ftags(&self) -> impl ExactSizeIterator<Item = Tag> + '_ {
        self.ftags.iter().copied()
    }

    /// Get an iterator through the face tags
    pub fn mut_ftags(&mut self) -> impl ExactSizeIterator<Item = &mut Tag> + '_ {
        self.ftags.iter_mut()
    }

    pub fn gface(&self, f: E::Face) -> <E::Face as Elem>::Geom<D, IsoMetric<D>> {
        <E::Face as Elem>::Geom::from_verts(
            f.iter()
                .map(|&i| (self.verts[i as usize], IsoMetric::<D>::from(1.0))),
        )
    }

    /// Get an iterator through the geometric elements
    pub fn gfaces(&self) -> impl Iterator<Item = <E::Face as Elem>::Geom<D, IsoMetric<D>>> + '_ {
        self.faces().map(|f| self.gface(f))
    }

    /// Get the total volume of a mesh
    #[must_use]
    pub fn vol(&self) -> f64 {
        self.gelems().map(|ge| ge.vol()).sum()
    }

    /// Compute the face-to-element connectivity
    pub fn compute_face_to_elems(&mut self) {
        debug!("Compute the face to element connectivity");
        if self.faces_to_elems.is_none() {
            self.faces_to_elems = Some(get_face_to_elem(self.elems()));
        } else {
            warn!("Face to element connectivity already computed");
        }
    }

    /// Clear the face-to-element connectivity
    pub fn clear_face_to_elems(&mut self) {
        debug!("Delete the face to element connectivity");
        self.faces_to_elems = None;
    }

    /// Get the face-to-element connectivity
    pub fn get_face_to_elems(&self) -> Result<&FxHashMap<<E as Elem>::Face, twovec::Vec<u32>>> {
        if self.faces_to_elems.is_none() {
            Err(Error::from("Face to element connectivity not computed"))
        } else {
            Ok(self.faces_to_elems.as_ref().unwrap())
        }
    }

    /// Compute the vertex-to-element connectivity
    pub fn compute_vertex_to_elems(&mut self) {
        debug!("Compute the vertex to element connectivity");

        if self.vertex_to_elems.is_none() {
            self.vertex_to_elems = Some(CSRGraph::transpose(&self.elems));
        } else {
            warn!("Vertex to element connectivity already computed");
        }
    }

    /// Clear the vertex-to-element connectivity
    pub fn clear_vertex_to_elems(&mut self) {
        debug!("Delete the vertex to element connectivity");
        self.vertex_to_elems = None;
    }

    /// Get the vertex-to-element connectivity
    pub fn get_vertex_to_elems(&self) -> Result<&CSRGraph> {
        if self.vertex_to_elems.is_none() {
            Err(Error::from("Vertex to element connectivity not computed"))
        } else {
            Ok(self.vertex_to_elems.as_ref().unwrap())
        }
    }

    /// Compute the element-to-element connectivity
    /// face-to-element connectivity is computed if not available
    pub fn compute_elem_to_elems(&mut self) {
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
            self.elem_to_elems = Some(CSRGraph::new(&g));
        } else {
            warn!("Element to element connectivity already computed");
        }
    }

    /// Clear the element-to-element connectivity
    pub fn clear_elem_to_elems(&mut self) {
        debug!("Delete the element to element connectivity");
        self.elem_to_elems = None;
    }

    /// Get the element-to-element connectivity
    pub fn get_elem_to_elems(&self) -> Result<&CSRGraph> {
        if self.elem_to_elems.is_none() {
            Err(Error::from("Element to element connectivity not computed"))
        } else {
            Ok(self.elem_to_elems.as_ref().unwrap())
        }
    }

    /// Compute the edges
    pub fn compute_edges(&mut self) {
        debug!("Compute the edges");
        if self.edges.is_none() {
            let mut edgs = FxHashSet::with_hasher(BuildHasherDefault::default());
            for e in self.elems() {
                for i_edg in 0..E::N_EDGES {
                    let mut edg = e.edge(i_edg);
                    edg.sort_unstable();
                    edgs.insert(edg);
                }
            }
            let edgs: Vec<_> = edgs.iter().copied().collect();
            self.edges = Some(edgs);
        } else {
            warn!("Edges already computed");
        }
    }

    /// Clear the edges
    pub fn clear_edges(&mut self) {
        debug!("Delete the edges");
        self.edges = None;
    }

    /// Get the the edges
    pub fn get_edges(&self) -> Result<&[[Idx; 2]]> {
        if self.edges.is_none() {
            Err(Error::from("Edges not computed"))
        } else {
            Ok(self.edges.as_ref().unwrap())
        }
    }

    /// Compute the vertex-to-vertex connectivity
    /// Edges are computed if not available
    pub fn compute_vertex_to_vertices(&mut self) {
        debug!("Compute the vertex to vertex connectivity");
        if self.vertex_to_vertices.is_none() {
            if self.edges.is_none() {
                self.compute_edges();
            }
            self.vertex_to_vertices = Some(CSRGraph::new(self.edges.as_ref().unwrap()));
        } else {
            warn!("Vertex to vertex connectivity already computed");
        }
    }

    /// Clear the vertex-to-vertex connectivity
    pub fn clear_vertex_to_vertices(&mut self) {
        debug!("Delete the vertex to vertex connectivity");
        self.vertex_to_vertices = None;
    }

    /// Get the vertex-to-vertes connectivity
    pub fn get_vertex_to_vertices(&self) -> Result<&CSRGraph> {
        if self.vertex_to_vertices.is_none() {
            Err(Error::from("Vertex to vertex connectivity not computed"))
        } else {
            Ok(self.vertex_to_vertices.as_ref().unwrap())
        }
    }

    /// Compute the volume and vertex volumes
    pub fn compute_volumes(&mut self) {
        debug!("Compute the vertex & element volumes");
        if self.elem_vol.is_none() {
            let mut elem_vol = vec![0.0; self.n_elems() as usize];
            let mut node_vol = vec![0.0; self.n_verts() as usize];
            let fac = 1.0 / f64::from(E::N_VERTS);
            for (i_elem, e) in self.elems().enumerate() {
                let v = self.gelem(e).vol();
                elem_vol[i_elem] = v;
                for i in e.iter().copied() {
                    node_vol[i as usize] += fac * v;
                }
            }
            self.elem_vol = Some(elem_vol);
            self.vert_vol = Some(node_vol);
        } else {
            warn!("Volumes already computed");
        }
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

    /// Compute an octree
    pub fn compute_octree(&mut self) {
        debug!("Compute an octree");
        if self.tree.is_none() {
            self.tree = Some(<DefaultObjectIndex as ObjectIndex<D>>::new(self));
        } else {
            warn!("Octree already computed");
        }
    }

    /// Clear the octree
    pub fn clear_octree(&mut self) {
        debug!("Delete the octree");
        self.tree = None;
    }

    /// Get the octree
    pub fn get_octree(&self) -> Result<&DefaultObjectIndex> {
        if self.tree.is_none() {
            Err(Error::from("Octree not computed"))
        } else {
            Ok(self.tree.as_ref().unwrap())
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

        let n_elems = self.n_elems() as usize;
        let n_verts = self.n_verts() as usize;
        assert_eq!(v.len() % n_elems, 0);

        let n_comp = v.len() / n_elems;

        let mut res = vec![0.; n_comp * n_verts];

        let v2e = self.vertex_to_elems.as_ref().unwrap();
        let elem_vol = self.elem_vol.as_ref().unwrap();
        let node_vol = self.vert_vol.as_ref().unwrap();

        for i_vert in 0..n_verts {
            for i_elem in v2e.row(i_vert as Idx).iter().copied() {
                let w = elem_vol[i_elem as usize] / f64::from(E::N_VERTS);
                for i_comp in 0..n_comp {
                    res[n_comp * i_vert + i_comp] += w * v[n_comp * i_elem as usize + i_comp];
                }
            }
            let w = node_vol[i_vert];
            for i_comp in 0..n_comp {
                res[n_comp * i_vert + i_comp] /= w;
            }
        }

        Ok(res)
    }

    /// Convert a field defined at the vertices (P1) to a field defined at the element centers (P0)
    /// For metric fields, use `elem_data_to_vertex_data_metric`
    pub fn vertex_data_to_elem_data(&self, v: &[f64]) -> Result<Vec<f64>> {
        debug!("Convert vertex data to element data");
        let n_elems = self.n_elems() as usize;
        let n_verts = self.n_verts() as usize;
        assert_eq!(v.len() % n_verts, 0);

        let n_comp = v.len() / n_verts;

        let mut res = vec![0.; n_comp * n_elems];

        let f = 1. / f64::from(E::N_VERTS);
        for (i_elem, e) in self.elems().enumerate() {
            for i_comp in 0..n_comp {
                for i_vert in e.iter().copied() {
                    res[n_comp * i_elem + i_comp] += f * v[n_comp * i_vert as usize + i_comp];
                }
            }
        }

        Ok(res)
    }

    /// Compute and orient the boundary faces
    /// the face-to-element connectivity must be available
    /// internal tagged faces are also returned
    #[allow(clippy::type_complexity)]
    pub fn boundary_faces(&self) -> Result<(Vec<E::Face>, Vec<Tag>, Tag, HashMap<Tag, Vec<Tag>>)> {
        debug!("Compute and order the boundary faces");
        if self.faces_to_elems.is_none() {
            return Err(Error::from("face to element connectivity not computed"));
        }

        let f2e = self.faces_to_elems.as_ref().unwrap();
        let n_bdy = f2e.iter().filter(|(_, v)| v.len() == 1).count();

        let mut tagged_faces: FxHashMap<E::Face, Tag> =
            FxHashMap::with_hasher(BuildHasherDefault::default());
        for (mut face, ftag) in self.faces().zip(self.ftags()) {
            face.sort();
            tagged_faces.insert(face, ftag);
        }

        let mut bdy = Vec::with_capacity(n_bdy);
        let mut bdy_tags = Vec::with_capacity(n_bdy);

        let new_faces_tag = self.ftags.iter().max().unwrap_or(&0) + 1;
        let mut next_internal_tag = new_faces_tag + 1;
        let mut internal_faces_tags: HashMap<Tag, Vec<Tag>> = HashMap::new();

        for (k, v) in f2e {
            if v.len() == 1 {
                // This is a boundary face
                let elem = self.elems[v[0] as usize];
                let mut ok = false;
                for i_face in 0..E::N_FACES {
                    let mut f = elem.face(i_face);
                    f.sort();
                    let is_same = !f.iter().zip(k.iter()).any(|(x, y)| x != y);
                    if is_same {
                        // face k is the i_face-th face of elem: use its orientation
                        let tag = tagged_faces.get(&f).unwrap_or(&new_faces_tag);
                        bdy_tags.push(*tag);
                        let f = elem.face(i_face);
                        bdy.push(f);
                        ok = true;
                        break;
                    }
                }
                assert!(ok);
            } else {
                // TODO: check all internal faces if the elems are tagged differently
                let tag = tagged_faces.get(k);
                if let Some(tag) = tag {
                    // This is a tagged internal face
                    bdy_tags.push(*tag);
                    bdy.push(*k);
                    let mut etags = v
                        .iter()
                        .copied()
                        .map(|i| self.etags[i as usize])
                        .collect::<Vec<_>>();
                    etags.sort_unstable();
                    if let Some(etags_ref) = internal_faces_tags.get(tag) {
                        // Check that the tags are the same
                        let mut is_ok = etags.len() == etags_ref.len();
                        for (t0, t1) in etags.iter().zip(etags_ref.iter()) {
                            is_ok = is_ok && (t0 == t1);
                        }
                        if !is_ok {
                            return Err(Error::from(&format!(
                                "internal faces with tag {tag} belong to {etags:?} and {etags_ref:?}"
                            )));
                        }
                    } else {
                        internal_faces_tags.insert(*tag, etags);
                    }
                } else {
                    let mut etags = v
                        .iter()
                        .copied()
                        .map(|i| self.etags[i as usize])
                        .collect::<Vec<_>>();
                    etags.sort_unstable();
                    if etags.len() > 2 || etags[0] != etags[1] {
                        let mut new_tag = true;
                        for (tag, etags_ref) in &internal_faces_tags {
                            let mut is_same = etags.len() == etags_ref.len();
                            for (t0, t1) in etags.iter().zip(etags_ref.iter()) {
                                is_same = is_same && (t0 == t1);
                            }
                            if is_same {
                                new_tag = false;
                                bdy_tags.push(*tag);
                                bdy.push(*k);
                                break;
                            }
                        }
                        if new_tag {
                            internal_faces_tags.insert(next_internal_tag, etags);
                            bdy_tags.push(next_internal_tag);
                            bdy.push(*k);
                            next_internal_tag += 1;
                        }
                    }
                }
            }
        }

        Ok((bdy, bdy_tags, new_faces_tag, internal_faces_tags))
    }

    /// Add the missing boundary faces and make sure that boundary faces are oriented outwards
    /// If internal faces are present, these are keps
    /// TODO: add the missing internal faces (belonging to 2 elems tagged differently) if needed
    /// TODO: whatto do if > 2 elems?
    pub fn add_boundary_faces(&mut self) -> (Tag, HashMap<Tag, Vec<Tag>>) {
        debug!("Add the missing boundary faces & orient all faces outwards");
        if self.faces_to_elems.is_none() {
            self.compute_face_to_elems();
        }

        let (faces, ftags, new_tag, internal_faces) = self.boundary_faces().unwrap();
        let n_untagged = ftags.iter().filter(|x| **x == new_tag).count();
        if n_untagged > 0 {
            info!("Added {} untagged faces with tag={}", n_untagged, new_tag);
        }
        for (tag, etags) in &internal_faces {
            if *tag > new_tag {
                info!(
                    "Added tag {} to internal faces belonging to elements with tags {:?}",
                    *tag, etags
                );
            }
        }
        self.faces = faces;
        self.ftags = ftags;

        (new_tag, internal_faces)
    }

    /// Return the mesh of the boundary
    /// Vertices are reindexed, and the surface id to volume id is returned
    /// together with the surface mesh
    #[must_use]
    pub fn boundary(&self) -> (SimplexMesh<D, E::Face>, Vec<Idx>) {
        debug!("Extract the mesh boundary");
        if self.faces.is_empty() {
            return (
                SimplexMesh::<D, E::Face>::new(
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                ),
                Vec::new(),
            );
        }
        let (new_faces, indices) = reindex(&self.faces);
        let n_bdy_verts = indices.len();

        let mut verts = vec![Point::<D>::zeros(); n_bdy_verts];
        let mut vert_ids = vec![0; n_bdy_verts];

        for (old, new) in indices {
            verts[new as usize] = self.verts[old as usize];
            vert_ids[new as usize] = old;
        }

        (
            SimplexMesh::<D, E::Face>::new(
                verts,
                new_faces,
                self.ftags.clone(),
                Vec::new(),
                Vec::new(),
            ),
            vert_ids,
        )
    }

    /// Return a bool vector that indicates wether a vertex in on a face
    #[must_use]
    pub fn boundary_flag(&self) -> Vec<bool> {
        let mut res = vec![false; self.n_verts() as usize];
        self.faces().flatten().for_each(|i| res[i as usize] = true);
        res
    }

    /// Extract a sub-mesh containing all the elements with a specific tag
    /// Return the sub-mesh and the indices of the vertices, elements and faces in the
    /// parent mesh
    #[must_use]
    pub fn extract_tag(&self, tag: Tag) -> SubSimplexMesh<D, E> {
        self.extract(|t| t == tag)
    }

    pub fn extract<F>(&self, elem_filter: F) -> SubSimplexMesh<D, E>
    where
        F: FnMut(Tag) -> bool,
    {
        let mut res = Self::empty();
        let (parent_vert_ids, parent_elem_ids, parent_face_ids) =
            res.add(self, elem_filter, |_| true, None::<fn(Tag) -> bool>);

        SubSimplexMesh {
            mesh: res,
            parent_vert_ids,
            parent_elem_ids,
            parent_face_ids,
        }
    }

    pub fn check(&self) -> Result<()> {
        self.check_volumes()?;
        self.check_boundary_faces()?;
        Ok(())
    }

    pub fn check_volumes(&self) -> Result<()> {
        debug!("Check the element volumes");
        for (i_elem, v) in self.gelems().map(|ge| ge.vol()).enumerate() {
            if v < 0.0 {
                return Err(Error::from(&format!(
                    "The volume of element {i_elem} is {v}",
                )));
            }
        }
        Ok(())
    }

    pub fn check_boundary_faces(&self) -> Result<()> {
        debug!("Check the boundary faces");
        if self.faces_to_elems.is_none() {
            return Err(Error::from("face to element connectivity not computed"));
        }

        let mut tagged_faces: FxHashMap<E::Face, Tag> =
            FxHashMap::with_hasher(BuildHasherDefault::default());
        for (mut face, ftag) in self.faces().zip(self.ftags()) {
            face.sort();
            tagged_faces.insert(face, ftag);
        }

        let f2e = self.faces_to_elems.as_ref().unwrap();
        for (f, els) in f2e {
            let tmp = tagged_faces.get(f);
            match els.len() {
                1 => {
                    if tmp.is_none() {
                        return Err(Error::from(&format!(
                            "face {:?} belongs to 1 element ({:?}) but is not tagged",
                            f, self.elems[els[0] as usize]
                        )));
                    }
                }
                2 => {
                    let should_be_tagged =
                        self.etags[els[0] as usize] != self.etags[els[1] as usize];
                    if tmp.is_none() && should_be_tagged {
                        return Err(Error::from(&format!(
                            "face {:?} belongs to 2 element ({:?} / {:?} and {:?}/ {:?})  but is not tagged",
                            f,
                            self.elems[els[0] as usize],
                            self.etags[els[0] as usize],
                            self.elems[els[1] as usize],
                            self.etags[els[1] as usize],
                        )));
                    } else if tmp.is_some() && !should_be_tagged {
                        return Err(Error::from(&format!(
                            "face {:?} belongs to 2 element ({:?} and {:?}) but is tagged {}",
                            f,
                            self.elems[els[0] as usize],
                            self.elems[els[1] as usize],
                            tmp.unwrap()
                        )));
                    }
                }
                _ => todo!(),
            }
        }

        Ok(())
    }

    /// Compute the mesh topology
    pub fn compute_topology(&mut self) {
        let (topo, vtags) = Topology::from_mesh(self);
        if self.topo.is_none() {
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
        self.tree = None;
        self.topo = None;
        self.vtags = None;
    }

    /// Add vertices, elements and faces from another mesh according to their tag
    ///   - only the elements with a tag t such that `element_filter(t)` is true are inserted
    ///   - among the faces belonging to these elements, only those with a tag such that `face_filter` is true are inserted
    ///   - if `face_merge_filter` is not None, vertices for `other` on faces with a tagged t such that `face_merge_filter(t)`is true are merged
    ///      with the closest vertices from `self`
    pub fn add<F1, F2, F3>(
        &mut self,
        other: &Self,
        mut elem_filter: F1,
        mut face_filter: F2,
        face_merge_filter: Option<F3>,
    ) -> (Vec<Idx>, Vec<Idx>, Vec<Idx>)
    where
        F1: FnMut(Tag) -> bool,
        F2: FnMut(Tag) -> bool,
        F3: FnMut(Tag) -> bool,
    {
        self.clear_all();

        let n_verts = self.n_verts();
        let n_verts_other = other.n_verts();
        let mut new_vert_ids = vec![Idx::MAX; n_verts_other as usize];

        for (e, t) in other.elems().zip(other.etags()) {
            if elem_filter(t) {
                e.iter()
                    .for_each(|&i| new_vert_ids[i as usize] = Idx::MAX - 1);
            }
        }

        // If needed, merge some face tags
        if let Some(mut face_merge_filter) = face_merge_filter {
            let merge_face_tags = other
                .ftags()
                .filter(|&t| face_merge_filter(t))
                .collect::<HashSet<_>>();
            let mut flg = vec![false; n_verts_other as usize];
            let (bdy, ids) = self.boundary();
            for tag in merge_face_tags {
                let smsh = bdy.extract_tag(tag);
                if smsh.mesh.n_verts() == 0 {
                    continue;
                }
                println!("merge tag {tag}");
                let tree = <DefaultPointIndex as PointIndex<D>>::new(&smsh.mesh);
                flg.iter_mut().for_each(|x| *x = false);
                other
                    .faces()
                    .zip(other.ftags())
                    .filter(|(_, t)| *t == tag)
                    .flat_map(|(f, _)| f)
                    .for_each(|i| flg[i as usize] = true);
                for i_vert in 0..n_verts_other {
                    if flg[i_vert as usize] && new_vert_ids[i_vert as usize] == Idx::MAX - 1 {
                        let vx = other.verts[i_vert as usize];
                        let i = tree.nearest_vertex(&vx);
                        let i = ids[smsh.parent_vert_ids[i as usize] as usize];
                        new_vert_ids[i_vert as usize] = i;
                    }
                }
                println!("done merge tag {tag}");
            }
        }

        // number & add the new vertices
        let mut next = n_verts;
        let mut added_verts = Vec::new();
        new_vert_ids.iter_mut().enumerate().for_each(|(i, x)| {
            if *x == Idx::MAX - 1 {
                added_verts.push(i as Idx);
                *x = next;
                next += 1;
                self.verts.push(other.verts[i]);
            }
        });

        let mut added_elems = Vec::new();
        for (i, (e, t)) in other
            .elems()
            .zip(other.etags())
            .enumerate()
            .filter(|(_, (_, t))| elem_filter(*t))
        {
            added_elems.push(i as Idx);
            self.elems
                .push(E::from_iter(e.iter().map(|&i| new_vert_ids[i as usize])));
            self.etags.push(t);
        }

        let mut added_faces = Vec::new();
        for (i, (f, t)) in other
            .faces()
            .zip(other.ftags())
            .enumerate()
            .filter(|(_, (_, t))| face_filter(*t))
            .filter(|(_, (f, _))| f.iter().all(|&j| new_vert_ids[j as usize] < Idx::MAX - 1))
        {
            added_faces.push(i as Idx);
            self.faces.push(E::Face::from_iter(
                f.iter().map(|&i| new_vert_ids[i as usize]),
            ));
            self.ftags.push(t);
        }

        (added_verts, added_elems, added_faces)
    }

    /// Remove faces based on their tag
    pub fn remove_faces<F: FnMut(Tag) -> bool>(&mut self, mut face_filter: F) {
        let mut new_faces = Vec::new();
        let mut new_ftags = Vec::new();

        for (f, t) in self
            .faces()
            .zip(self.ftags())
            .filter(|(_, t)| !face_filter(*t))
        {
            new_faces.push(f);
            new_ftags.push(t);
        }
        self.faces = new_faces;
        self.ftags = new_ftags;
    }

    /// Modify the face tags
    pub fn update_face_tags<F: FnMut(Tag) -> Tag>(&mut self, mut new_ftags: F) {
        self.ftags.iter_mut().for_each(|t| *t = new_ftags(*t));
    }

    /// Get the number of faces with a given tag
    #[must_use]
    pub fn n_tagged_faces(&self, tag: Tag) -> Idx {
        self.ftags.iter().filter(|&&t| t == tag).count() as Idx
    }
}

#[cfg(test)]
mod tests {

    use crate::{
        geom_elems::GElem,
        test_meshes::{test_mesh_2d, test_mesh_3d},
        topo_elems::{Edge, Elem, Triangle},
        Result,
    };

    #[test]
    fn test_2d() {
        let mesh = test_mesh_2d();

        assert_eq!(mesh.n_verts(), 4);
        assert_eq!(mesh.n_faces(), 4);
        assert_eq!(mesh.n_elems(), 2);

        assert_eq!(mesh.elems[1], Triangle::from_slice(&[0, 2, 3]));
        assert_eq!(mesh.faces[1], Edge::from_slice(&[1, 2]));

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

        let n_elems = mesh.n_elems() as usize;
        let n_verts = mesh.n_verts() as usize;
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

        let n_elems = mesh.n_elems() as usize;
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

        let sub_mesh = mesh.extract_tag(1);
        let smesh = sub_mesh.mesh;
        let ids = sub_mesh.parent_vert_ids;
        assert_eq!(smesh.n_verts(), 6);
        assert_eq!(smesh.n_elems(), 4);
        assert_eq!(smesh.n_faces(), 4);
        assert!(f64::abs(smesh.gelems().map(|ge| ge.vol()).sum::<f64>() - 0.5) < 1e-10);

        assert_eq!(ids.len(), 6);
        for i in ids {
            let p = mesh.verts[i as usize];
            assert!(p[0] - p[1] > -1e-10);
        }
    }
}
