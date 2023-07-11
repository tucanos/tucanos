use crate::{
    geom_elems::GElem,
    graph::{CSRGraph, Connectivity, ElemGraph, ElemGraphInterface},
    metric::IsoMetric,
    octree::Octree,
    topo_elems::{get_elem, get_face_to_elem, Elem},
    topology::Topology,
    twovec, Error, Idx, Mesh, Result, Tag, TopoTag,
};
use log::{debug, info, warn};
use nalgebra::SVector;
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::{HashMap, HashSet};
use std::hash::BuildHasherDefault;
use std::marker::PhantomData;

/// A mesh containing a single type of elements in D-dimensions
#[derive(Debug, Default, Clone)]
pub struct SimplexMesh<const D: usize, E: Elem> {
    /// Coordinates of the vertices (length = D * # of vertices)
    pub coords: Vec<f64>,
    /// Element connectivity (length = # of vertices per element * # of elements)
    pub elems: Vec<Idx>,
    /// Element tags (length = # of elements)
    pub etags: Vec<Tag>,
    /// Face connectivity (length = # of vertices per face * # of faces)
    pub faces: Vec<Idx>,
    /// Faces tags (length = # of faces)
    pub ftags: Vec<Tag>,
    pub point: PhantomData<SVector<f64, D>>,
    pub elem: PhantomData<E>,
    /// Face to element connectitivity stored as a HashMap taking the face vertices (sorted) and returning
    /// a vector of element Ids
    pub faces_to_elems: Option<FxHashMap<E::Face, twovec::Vec<u32>>>,
    /// Vertex-to-element connectivity stored in CSR format
    pub vertex_to_elems: Option<CSRGraph>,
    /// Element-to-element connectivity stored in CSR format
    pub elem_to_elems: Option<CSRGraph>,
    /// Edges (length = 2 * # of edges)
    pub edges: Option<Vec<Idx>>,
    /// Vertex-to-vertex (~edges) connectivity stored in CSR format
    pub vertex_to_vertices: Option<CSRGraph>,
    /// Element volumes (length = # of elements)
    pub elem_vol: Option<Vec<f64>>,
    /// Vertex volumes (length = # of vertices)
    /// The volume of a vertex in the weighted average of the neighboring element volumes
    /// It can be seen as the volume of a dual cell
    /// sum(elem_vol) = sum(vert_vol)
    pub vert_vol: Option<Vec<f64>>,
    /// Octree
    pub tree: Option<Octree>,
    /// Topology
    pub topo: Option<Topology>,
    /// Vertex tags
    pub vtags: Option<Vec<TopoTag>>,
}

pub struct SubSimplexMesh<const D: usize, E: Elem> {
    pub mesh: SimplexMesh<D, E>,
    pub parent_vert_ids: Vec<Idx>,
    pub parent_elem_ids: Vec<Idx>,
    pub parent_face_ids: Vec<Idx>,
}

pub type Point<const D: usize> = SVector<f64, D>;

#[must_use]
pub fn get_point<const D: usize>(conn: &[f64], idx: Idx) -> Point<D> {
    let start = idx as usize * D;
    let end = start + D;
    return SVector::from_iterator(conn[start..end].iter().copied());
}

impl<const D: usize, E: Elem> Mesh for SimplexMesh<D, E> {
    fn dim(&self) -> Idx {
        D as Idx
    }

    fn cell_dim(&self) -> Idx {
        E::DIM as Idx
    }

    fn n_verts(&self) -> Idx {
        (self.coords.len() / D) as Idx
    }

    fn n_elems(&self) -> Idx {
        self.elems.len() as Idx / E::N_VERTS
    }

    fn n_faces(&self) -> Idx {
        self.faces.len() as Idx / E::Face::N_VERTS
    }
}

impl<const D: usize, E: Elem> SimplexMesh<D, E> {
    /// Create a new `SimplexMesh`. The extra connectivity information is not built
    #[must_use]
    pub fn new(
        coords: Vec<f64>,
        elems: Vec<Idx>,
        etags: Vec<Tag>,
        faces: Vec<Idx>,
        ftags: Vec<Tag>,
    ) -> Self {
        info!(
            "Create a SimplexMesh with {} {}D vertices / {} {} / {} {}",
            coords.len() / D,
            D,
            elems.len() / E::N_VERTS as usize,
            E::NAME,
            faces.len() / E::Face::N_VERTS as usize,
            E::Face::NAME
        );
        Self {
            coords,
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

    pub fn empty() -> Self {
        Self {
            coords: Vec::new(),
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

    /// Get the i-th vertex
    #[must_use]
    pub fn vert(&self, idx: Idx) -> Point<D> {
        get_point(&self.coords, idx)
    }

    /// Get an iterator through the vertices
    pub fn verts(&self) -> impl Iterator<Item = Point<D>> + '_ {
        (0..self.n_verts()).map(|i| self.vert(i))
    }

    /// Get the i-th element
    #[must_use]
    pub fn elem(&self, idx: Idx) -> E {
        get_elem(&self.elems, idx)
    }

    /// Get an iterator through the elements
    pub fn elems(&self) -> impl Iterator<Item = E> + '_ {
        (0..self.n_elems()).map(|i| self.elem(i))
    }

    /// Get an iterator through the element tags
    pub fn etags(&self) -> impl Iterator<Item = Tag> + '_ {
        (0..self.n_elems()).map(|i| self.etags[i as usize])
    }

    /// Get the i-th face
    #[must_use]
    pub fn face(&self, idx: Idx) -> E::Face {
        get_elem(&self.faces, idx)
    }

    /// Get an iterator through the faces
    pub fn faces(&self) -> impl Iterator<Item = E::Face> + '_ {
        (0..self.n_faces()).map(|i| self.face(i))
    }

    /// Get an iterator through the face tags
    pub fn ftags(&self) -> impl Iterator<Item = Tag> + '_ {
        (0..self.n_faces()).map(|i| self.ftags[i as usize])
    }

    /// Get the i-th geometrical element
    /// TODO: the metric information is not used, create a `NoMetric`
    #[must_use]
    pub fn gelem(&self, idx: Idx) -> E::Geom<D, IsoMetric<D>> {
        let e: E = get_elem(&self.elems, idx);
        let metric = IsoMetric::<D>::from(1.0);
        E::Geom::from_verts(e.iter().map(|i| (get_point::<D>(&self.coords, *i), metric)))
    }

    /// Get an iterator through the geometric elements
    pub fn gelems(&self) -> impl Iterator<Item = E::Geom<D, IsoMetric<D>>> + '_ {
        (0..self.n_elems()).map(|i| self.gelem(i))
    }

    /// Get the volume (area in 2D) of the i-th element
    #[must_use]
    pub fn elem_vol(&self, idx: Idx) -> f64 {
        let ge = self.gelem(idx);
        ge.vol()
    }

    /// Get an iterator through the element volumes
    pub fn elem_vols(&self) -> impl Iterator<Item = f64> + '_ {
        (0..self.n_elems()).map(|i| self.elem_vol(i))
    }

    /// Get the center of the i-th element
    #[must_use]
    pub fn elem_center(&self, idx: Idx) -> Point<D> {
        let ge = self.gelem(idx);
        ge.center()
    }

    /// Get an iterator through the element centers
    pub fn elem_centers(&self) -> impl Iterator<Item = Point<D>> + '_ {
        (0..self.n_elems()).map(|i| self.elem_center(i))
    }

    /// Get the total volume of a mesh
    #[must_use]
    pub fn vol(&self) -> f64 {
        (0..self.n_elems()).map(|i| self.elem_vol(i)).sum()
    }

    /// Get the i-th geometrical face
    /// TODO: the metric information is not used, create a `NoMetric`
    #[must_use]
    pub fn gface(&self, idx: Idx) -> <<E as Elem>::Face as Elem>::Geom<D, IsoMetric<D>> {
        let f: E::Face = get_elem(&self.faces, idx);
        let metric = IsoMetric::<D>::from(1.0);
        <<E as Elem>::Face as Elem>::Geom::from_verts(
            f.iter().map(|i| (get_point::<D>(&self.coords, *i), metric)),
        )
    }

    /// Get an iterator through the geometric faces
    pub fn gfaces(
        &self,
    ) -> impl Iterator<Item = <<E as Elem>::Face as Elem>::Geom<D, IsoMetric<D>>> + '_ {
        (0..self.n_faces()).map(|i| self.gface(i))
    }

    /// Get the area (length in 2D) of the i-th face
    #[must_use]
    pub fn face_vol(&self, idx: Idx) -> f64 {
        let gf = self.gface(idx);
        gf.vol()
    }

    /// Get the center of the i-th face
    #[must_use]
    pub fn face_center(&self, idx: Idx) -> Point<D> {
        let gf = self.gface(idx);
        gf.center()
    }

    /// Get an iterator through the face centers
    pub fn face_centers(&self) -> impl Iterator<Item = Point<D>> + '_ {
        (0..self.n_faces()).map(|i| self.face_center(i))
    }

    /// Compute the face-to-element connectivity
    pub fn compute_face_to_elems(&mut self) {
        debug!("Compute the face to element connectivity");
        if self.faces_to_elems.is_none() {
            self.faces_to_elems = Some(get_face_to_elem::<E>(&self.elems));
        } else {
            warn!("Face to element connectivity already computed");
        }
    }

    /// Clear the face-to-element connectivity
    pub fn clear_face_to_elems(&mut self) {
        debug!("Delete the face to element connectivity");
        self.faces_to_elems = None;
    }

    /// Compute the vertex-to-element connectivity
    pub fn compute_vertex_to_elems(&mut self) {
        debug!("Compute the vertex to element connectivity");
        let g = ElemGraphInterface::new(E::N_VERTS, &self.elems);
        if self.vertex_to_elems.is_none() {
            self.vertex_to_elems = Some(CSRGraph::transpose(&g));
        } else {
            warn!("Vertex to element connectivity already computed");
        }
    }

    /// Clear the vertex-to-element connectivity
    pub fn clear_vertex_to_elems(&mut self) {
        debug!("Delete the vertex to element connectivity");
        self.vertex_to_elems = None;
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

            let mut g = ElemGraph::new(2, f2e.len() as Idx);
            for (_, val) in f2e.iter() {
                for (i, i_elem) in val.iter().copied().enumerate() {
                    for j_elem in val.iter().skip(i + 1).copied() {
                        g.add_elem(&[i_elem, j_elem]);
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
            let edgs: Vec<_> = edgs.iter().flatten().copied().collect();
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

    /// Compute the vertex-to-vertex connectivity
    /// Edges are computed if not available
    pub fn compute_vertex_to_vertices(&mut self) {
        debug!("Compute the vertex to vertex connectivity");
        if self.vertex_to_vertices.is_none() {
            if self.edges.is_none() {
                self.compute_edges();
            }
            let g = ElemGraphInterface::new(2, self.edges.as_ref().unwrap());
            self.vertex_to_vertices = Some(CSRGraph::new(&g));
        } else {
            warn!("Vertex to vertex connectivity already computed");
        }
    }

    /// Clear the vertex-to-vertex connectivity
    pub fn clear_vertex_to_vertices(&mut self) {
        debug!("Delete the vertex to vertex connectivity");
        self.vertex_to_vertices = None;
    }

    /// Compute the volume and vertex volumes
    pub fn compute_volumes(&mut self) {
        debug!("Compute the vertex & element volumes");
        if self.elem_vol.is_none() {
            let mut elem_vol = vec![0.0; self.n_elems() as usize];
            let mut node_vol = vec![0.0; self.n_verts() as usize];
            let fac = 1.0 / f64::from(E::N_VERTS);
            for (i_elem, e) in self.elems().enumerate() {
                let v = self.gelem(i_elem as Idx).vol();
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

    /// Compute an octree
    pub fn compute_octree(&mut self) {
        debug!("Compute an octree");
        if self.tree.is_none() {
            self.tree = Some(Octree::new(self));
        } else {
            warn!("Octree already computed");
        }
    }

    /// Clear the octree
    pub fn clear_octree(&mut self) {
        debug!("Delete the octree");
        self.tree = None;
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
    pub fn boundary_faces(&self) -> Result<(Vec<Idx>, Vec<Tag>, Tag, HashMap<Tag, Vec<Tag>>)> {
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

        let mut bdy = Vec::with_capacity(E::Face::N_VERTS as usize * n_bdy);
        let mut bdy_tags = Vec::with_capacity(n_bdy);

        let new_faces_tag = self.ftags.iter().max().unwrap_or(&0) + 1;
        let mut next_internal_tag = new_faces_tag + 1;
        let mut internal_faces_tags: HashMap<Tag, Vec<Tag>> = HashMap::new();

        for (k, v) in f2e.iter() {
            if v.len() == 1 {
                // This is a boundary face
                let elem = self.elem(v[0]);
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
                        bdy.extend(f.iter().copied());
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
                    bdy.extend(k.iter().copied());
                    let mut etags = v
                        .iter()
                        .copied()
                        .map(|i| self.etags[i as usize])
                        .collect::<Vec<_>>();
                    etags.sort();
                    if let Some(etags_ref) = internal_faces_tags.get(tag) {
                        // Check that the tags are the same
                        let mut is_ok = etags.len() == etags_ref.len();
                        for (t0, t1) in etags.iter().zip(etags_ref.iter()) {
                            is_ok = is_ok && (t0 == t1);
                        }
                        if !is_ok {
                            return Err(Error::from(&format!(
                                "internal faces with tag {} belong to {:?} and {:?}",
                                tag, etags, etags_ref
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
                    etags.sort();
                    if etags.len() > 2 || etags[0] != etags[1] {
                        let mut new_tag = true;
                        for (tag, etags_ref) in internal_faces_tags.iter() {
                            let mut is_same = etags.len() == etags_ref.len();
                            for (t0, t1) in etags.iter().zip(etags_ref.iter()) {
                                is_same = is_same && (t0 == t1);
                            }
                            if is_same {
                                new_tag = false;
                                bdy_tags.push(*tag);
                                bdy.extend(k.iter().copied());
                                break;
                            }
                        }
                        if new_tag {
                            internal_faces_tags.insert(next_internal_tag, etags);
                            bdy_tags.push(next_internal_tag);
                            bdy.extend(k.iter().copied());
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
        for (tag, etags) in internal_faces.iter() {
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
        let mut g = ElemGraph::from(1, self.faces.iter().copied());
        let indices = g.reindex();
        let n_bdy_verts = g.max_node() as usize + 1;

        let mut coords = vec![0.0; D * n_bdy_verts];
        let mut vert_ids = vec![0; n_bdy_verts];

        for (old, new) in &indices {
            let pt = self.vert(*old);
            for i in 0..D {
                coords[D * (*new as usize) + i] = pt[i];
            }
            vert_ids[*new as usize] = *old;
        }

        (
            SimplexMesh::<D, E::Face>::new(
                coords,
                g.elems,
                self.ftags.clone(),
                Vec::new(),
                Vec::new(),
            ),
            vert_ids,
        )
    }

    /// Return a bool vector that indicates wether a vertex in on a face
    pub fn boundary_flag(&self) -> Vec<bool> {
        let mut res = vec![false; self.n_verts() as usize];
        self.faces.iter().for_each(|&i| res[i as usize] = true);
        res
    }

    /// Extract a sub-mesh containing all the elements with a specific tag
    /// Return the sub-mesh and the indices of the vertices, elements and faces in the
    /// parent mesh
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
        for (i_elem, v) in self.elem_vols().enumerate() {
            if v < 0.0 {
                return Err(Error::from(&format!(
                    "The volume of element {} is {}",
                    i_elem, v,
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
        for (i_face, ftag) in self.ftags.iter().enumerate() {
            let mut face = self.face(i_face as Idx);
            face.sort();
            tagged_faces.insert(face, *ftag);
        }

        let f2e = self.faces_to_elems.as_ref().unwrap();
        for (f, els) in f2e.iter() {
            let tmp = tagged_faces.get(f);
            match els.len() {
                1 => {
                    if tmp.is_none() {
                        return Err(Error::from(&format!(
                            "face {:?} belongs to 1 element ({:?}) but is not tagged",
                            f,
                            self.elem(els[0])
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
                            self.elem(els[0]),
                            self.etags[els[0] as usize],
                            self.elem(els[1]),
                            self.etags[els[1] as usize],
                        )));
                    } else if tmp.is_some() && !should_be_tagged {
                        return Err(Error::from(&format!(
                            "face {:?} belongs to 2 element ({:?} and {:?}) but is tagged {}",
                            f,
                            self.elem(els[0]),
                            self.elem(els[1]),
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
                let tree = Octree::new(&smsh.mesh);
                flg.iter_mut().for_each(|x| *x = false);
                other
                    .faces()
                    .zip(other.ftags())
                    .filter(|(_, t)| *t == tag)
                    .flat_map(|(f, _)| f)
                    .for_each(|i| flg[i as usize] = true);
                for i_vert in 0..n_verts_other {
                    if flg[i_vert as usize] && new_vert_ids[i_vert as usize] == Idx::MAX - 1 {
                        let vx = other.vert(i_vert);
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
                for j in 0..D {
                    self.coords.push(other.coords[D * i + j])
                }
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
                .extend(e.iter().map(|&i| new_vert_ids[i as usize]));
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
            self.faces
                .extend(f.iter().map(|&i| new_vert_ids[i as usize]));
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
            new_faces.extend(f.iter().copied());
            new_ftags.push(t);
        }
        self.faces = new_faces;
        self.ftags = new_ftags;
    }

    /// Modify the face tags
    pub fn update_face_tags<F: FnMut(Tag) -> Tag>(&mut self, mut new_ftags: F) {
        self.ftags.iter_mut().for_each(|mut t| *t = new_ftags(*t));
    }

    /// Get the number of faces with a given tag
    pub fn n_tagged_faces(&self, tag: Tag) -> Idx {
        self.ftags.iter().filter(|&&t| t == tag).count() as Idx
    }
}

#[cfg(test)]
mod tests {

    use crate::{
        test_meshes::{test_mesh_2d, test_mesh_3d},
        topo_elems::{Edge, Elem, Triangle},
        Mesh, Result,
    };

    #[test]
    fn test_2d() {
        let mesh = test_mesh_2d();

        assert_eq!(mesh.n_verts(), 4);
        assert_eq!(mesh.n_faces(), 4);
        assert_eq!(mesh.n_elems(), 2);

        assert_eq!(mesh.elem(1), Triangle::from_slice(&[0, 2, 3]));
        assert_eq!(mesh.face(1), Edge::from_slice(&[1, 2]));

        let v: f64 = mesh.elem_vols().sum();
        assert!((v - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_3d() {
        let mesh = test_mesh_3d();

        assert_eq!(mesh.n_verts(), 8);
        assert_eq!(mesh.n_faces(), 12);
        assert_eq!(mesh.n_elems(), 5);

        let v: f64 = mesh.elem_vols().sum();
        assert!(f64::abs(v - 1.0) < 1e-12);
    }

    #[test]
    fn test_cell_to_node() -> Result<()> {
        let mesh = test_mesh_3d();
        let mut mesh = mesh.split().split().split();

        let n_elems = mesh.n_elems() as usize;
        let n_verts = mesh.n_verts() as usize;
        let mut v_e = Vec::with_capacity(3 * n_elems);

        for c in mesh.elem_centers() {
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

        for (i_elem, pt) in mesh.elem_centers().enumerate() {
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
        assert!(f64::abs(smesh.vol() - 0.5) < 1e-10);

        assert_eq!(ids.len(), 6);
        for i in ids {
            let p = mesh.vert(i);
            assert!(p[0] - p[1] > -1e-10);
        }
    }
}
