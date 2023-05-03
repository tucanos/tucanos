use crate::geom_elems::GElem;
use crate::graph::{CSRGraph, Connectivity, ElemGraph, ElemGraphInterface};
use crate::metric::IsoMetric;
use crate::octree::Octree;
use crate::topo_elems::{get_elem, get_face_to_elem, Elem};
use crate::twovec;
use crate::{Error, Idx, Mesh, Result, Tag};
use log::{debug, info, warn};
use nalgebra::SVector;
use rustc_hash::{FxHashMap, FxHashSet};
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
    faces_to_elems: Option<FxHashMap<E::Face, twovec::Vec<u32>>>,
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
    // Octree
    pub tree: Option<Octree>,
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
        self.faces_to_elems = Some(get_face_to_elem::<E>(&self.elems));
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
        self.vertex_to_elems = Some(CSRGraph::transpose(&g));
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
    }

    /// Clear the element-to-element connectivity
    pub fn clear_elem_to_elems(&mut self) {
        debug!("Delete the element to element connectivity");
        self.elem_to_elems = None;
    }

    /// Compute the edges
    pub fn compute_edges(&mut self) {
        debug!("Compute the edges");
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
        if self.edges.is_none() {
            self.compute_edges();
        }
        let g = ElemGraphInterface::new(2, self.edges.as_ref().unwrap());
        self.vertex_to_vertices = Some(CSRGraph::new(&g));
    }

    /// Clear the vertex-to-vertex connectivity
    pub fn clear_vertex_to_vertices(&mut self) {
        debug!("Delete the vertex to vertex connectivity");
        self.vertex_to_vertices = None;
    }

    /// Compute the volume and vertex volumes
    pub fn compute_volumes(&mut self) {
        debug!("Compute the vertex & element volumes");
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

        self.tree = Some(Octree::new(self));
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
    /// TODO: rename & find internal faces
    pub fn boundary_faces(&self) -> Result<(Vec<Idx>, Vec<Tag>, Tag)> {
        debug!("Compute and order the boundary faces");
        if self.faces_to_elems.is_none() {
            return Err(Error::from("face to element connectivity not computed"));
        }

        let f2e = self.faces_to_elems.as_ref().unwrap();
        let n_bdy = f2e.iter().filter(|(_, v)| v.len() == 1).count();

        let mut tagged_faces: FxHashMap<E::Face, Tag> =
            FxHashMap::with_hasher(BuildHasherDefault::default());
        for (i_face, ftag) in self.ftags.iter().enumerate() {
            let mut face = self.face(i_face as Idx);
            face.sort();
            tagged_faces.insert(face, *ftag);
        }

        let mut bdy = Vec::with_capacity(E::Face::N_VERTS as usize * n_bdy);
        let mut bdy_tags = Vec::with_capacity(n_bdy);

        let new_faces_tag = self.ftags.iter().max().unwrap() + 1;

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
                }
            }
        }

        Ok((bdy, bdy_tags, new_faces_tag))
    }

    /// Add the missing boundary faces and make sure that boundary faces are oriented outwards
    /// If internal faces are present, these are keps
    /// TODO: add the missing internal faces (belonging to 2 elems tagged differently) if needed
    /// TODO: whatto do if > 2 elems?
    pub fn add_boundary_faces(&mut self) -> Idx {
        debug!("Add the missing boundary faces & orient all faces outwards");
        if self.faces_to_elems.is_none() {
            self.compute_face_to_elems();
        }

        let (faces, ftags, new_tag) = self.boundary_faces().unwrap();
        let n_untagged = ftags.iter().filter(|x| **x == new_tag).count();
        if n_untagged > 0 {
            warn!("Added {} untagged faces with tag={}", n_untagged, new_tag);
        }
        self.faces = faces;
        self.ftags = ftags;
        n_untagged as Idx
    }

    /// Return the mesh of the boundary
    /// Vertices are reindexed
    #[must_use]
    pub fn boundary(&self) -> SimplexMesh<D, E::Face> {
        debug!("Extract the mesh boundary");
        let mut g = ElemGraph::from(1, self.faces.iter().copied());
        let indices = g.reindex();
        let mut coords = Vec::new();
        coords.resize(D * (g.max_node() as usize + 1), 0.);

        for (old, new) in &indices {
            let pt = self.vert(*old);
            for i in 0..D {
                coords[D * (*new as usize) + i] = pt[i];
            }
        }

        SimplexMesh::<D, E::Face>::new(coords, g.elems, self.ftags.clone(), Vec::new(), Vec::new())
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
}
