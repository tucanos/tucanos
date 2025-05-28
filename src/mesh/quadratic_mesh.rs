use super::geom_quadratic_elems::GQuadraticElem;
use super::graph::{reindexq, CSRGraph};
use super::topology::Topology;
use super::twovec;
use super::vector::VectorQuadratic;
use crate::mesh::topo_elems::Triangle;
use crate::mesh::topo_elems_quadratic::{QuadraticEdge, QuadraticElem, QuadraticTriangle};
use crate::mesh::{get_face_to_elem_quadratic, SimplexMesh};
use crate::metric::IsoMetric;
use crate::spatialindex::{DefaultObjectIndex, DefaultPointIndex, ObjectIndex, PointIndex};
use crate::Dim;
use crate::{mesh::Point, Idx, Tag};
use crate::{Error, Result, TopoTag};
use log::{debug, warn};
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::HashMap;
#[derive(Debug, Clone)]

pub struct QuadraticMesh<QE: QuadraticElem> {
    verts: VectorQuadratic<Point<3>>,
    tris: VectorQuadratic<QE>,
    tri_tags: VectorQuadratic<Tag>,
    edgs: VectorQuadratic<QE::Face>,
    edg_tags: VectorQuadratic<Tag>,
    topo: Option<Topology>,
    vtags: Option<Vec<TopoTag>>,
    faces_to_elems: Option<FxHashMap<QE::Face, twovec::Vec<u32>>>,
    /// Vertex-to-element connectivity stored in CSR format
    vertex_to_elems: Option<CSRGraph>,
    /// Element-to-element connectivity stored in CSR format
    elem_to_elems: Option<CSRGraph>,
}

pub struct SubQuadraticMesh<QE: QuadraticElem> {
    pub mesh: QuadraticMesh<QE>,
    pub parent_vert_ids: Vec<Idx>,
    pub parent_elem_ids: Vec<Idx>,
    pub parent_face_ids: Vec<Idx>,
}

pub struct QuadraticMesh2 {
    verts: Vec<Point<3>>,
    tris: Vec<[Idx; 6]>,
    tri_tags: Vec<Tag>,
    edgs: Vec<[Idx; 3]>,
    edg_tags: Vec<Tag>,
}

impl<QE: QuadraticElem> QuadraticMesh<QE> {
    #[must_use]
    pub fn new_with_vector(
        verts: VectorQuadratic<Point<3>>,
        tris: VectorQuadratic<QE>,
        tri_tags: VectorQuadratic<Tag>,
        edgs: VectorQuadratic<QE::Face>,
        edg_tags: VectorQuadratic<Tag>,
    ) -> Self {
        debug!(
            "Create a Quadratic mesh with {} {}D vertices / {} {} / {} {}",
            verts.len(),
            3,
            tris.len(),
            QE::NAME,
            edgs.len(),
            QE::Face::NAME
        );
        Self {
            verts,
            tris,
            tri_tags,
            edgs,
            edg_tags,
            topo: None,
            vtags: None,
            faces_to_elems: None,
            vertex_to_elems: None,
            elem_to_elems: None,
        }
    }

    /// Create a new `QuadraticMesh`. The extra connectivity information is not built.
    #[must_use]
    pub fn new(
        verts: Vec<Point<3>>,
        tris: Vec<QE>,
        tri_tags: Vec<Tag>,
        edgs: Vec<QE::Face>,
        edg_tags: Vec<Tag>,
    ) -> Self {
        Self::new_with_vector(
            verts.into(),
            tris.into(),
            tri_tags.into(),
            edgs.into(),
            edg_tags.into(),
        )
    }

    #[must_use]
    pub fn empty() -> Self {
        Self {
            verts: Vec::new().into(),
            tris: Vec::new().into(),
            tri_tags: Vec::new().into(),
            edgs: Vec::new().into(),
            edg_tags: Vec::new().into(),
            topo: None,
            vtags: None,
            faces_to_elems: None,
            vertex_to_elems: None,
            elem_to_elems: None,
        }
    }

    /// Get the number of vertices
    #[must_use]
    pub fn n_verts(&self) -> Idx {
        self.verts.len() as Idx
    }

    /// Get the number of edges
    #[must_use]
    pub fn n_edges(&self) -> Idx {
        self.edgs.len() as Idx
    }

    /// Get the number of triangles
    #[must_use]
    pub fn n_tris(&self) -> Idx {
        self.tris.len() as Idx
    }

    /// Get the i-th vertex
    #[must_use]
    pub fn vert(&self, idx: Idx) -> Point<3> {
        self.verts.index(idx)
    }

    /// Get the i-th edge
    #[must_use]
    pub fn edge(&self, idx: Idx) -> QE::Face {
        self.edgs.index(idx)
    }

    /// Get the i-th triangle
    #[must_use]
    pub fn tri(&self, idx: Idx) -> QE {
        self.tris.index(idx)
    }

    /// Get the i-th edge tag
    #[must_use]
    pub fn edgetag(&self, idx: Idx) -> Tag {
        self.edg_tags.index(idx)
    }

    /// Get the i-th triangle tag
    #[must_use]
    pub fn tritag(&self, idx: Idx) -> Tag {
        self.tri_tags.index(idx)
    }

    /// Get an iterator through the triangles
    #[must_use]
    pub fn tris(&self) -> impl ExactSizeIterator<Item = QE> + '_ {
        self.tris.iter()
    }

    /// Get an iterator through the vertices
    #[must_use]
    pub fn edges(&self) -> impl ExactSizeIterator<Item = QE::Face> + '_ {
        self.edgs.iter()
    }

    /// Get an iterator through the vertices
    #[must_use]
    pub fn verts(&self) -> impl ExactSizeIterator<Item = Point<3>> + '_ {
        self.verts.iter()
    }

    /// Get an iterator through the triangle tags
    #[must_use]
    pub fn tritags(&self) -> impl ExactSizeIterator<Item = Tag> + '_ {
        self.tri_tags.iter()
    }

    /// Get an iterator through the edge tags
    #[must_use]
    pub fn edgtags(&self) -> impl ExactSizeIterator<Item = Tag> + '_ {
        self.edg_tags.iter()
    }

    pub fn gelem(&self, e: QE) -> QE::GeomQuadratic<IsoMetric<3>> {
        QE::GeomQuadratic::from_verts(
            e.iter()
                .map(|&i| (self.verts.index(i), IsoMetric::<3>::from(1.0))),
        )
    }

    /// Get an iterator through the geometric elements
    #[must_use]
    pub fn gelems(&self) -> impl ExactSizeIterator<Item = QE::GeomQuadratic<IsoMetric<3>>> + '_ {
        self.tris().map(|e| self.gelem(e))
    }

    pub fn gface(&self, f: QE::Face) -> <QE::Face as QuadraticElem>::GeomQuadratic<IsoMetric<3>> {
        <QE::Face as QuadraticElem>::GeomQuadratic::from_verts(
            f.iter()
                .map(|&i| (self.verts.index(i), IsoMetric::<3>::from(1.0))),
        )
    }

    /// Get an iterator through the geometric faces
    pub fn gfaces(
        &self,
    ) -> impl Iterator<Item = <QE::Face as QuadraticElem>::GeomQuadratic<IsoMetric<3>>> + '_ {
        self.edges().map(|f| self.gface(f))
    }

    /// Compute an octree to locate elements
    #[must_use]
    pub fn compute_elem_tree(&self) -> DefaultObjectIndex<3> {
        debug!("Compute the element octree");
        <DefaultObjectIndex<3> as ObjectIndex<3>>::newquadratic(self)
    }

    /// Get the vertex tags
    pub fn get_vertex_tags(&self) -> Result<&[TopoTag]> {
        if self.topo.is_none() {
            Err(Error::from("Topology not computed"))
        } else {
            Ok(self.vtags.as_ref().unwrap())
        }
    }

    /// Get an iterator through the vertices
    pub fn mut_verts(&mut self) -> impl ExactSizeIterator<Item = &mut Point<3>> + '_ {
        self.verts.as_std_mut().iter_mut()
    }

    /// Get the face-to-element connectivity
    pub fn get_face_to_elems(
        &self,
    ) -> Result<&FxHashMap<<QE as QuadraticElem>::Face, twovec::Vec<u32>>> {
        if self.faces_to_elems.is_none() {
            Err(Error::from("Face to element connectivity not computed"))
        } else {
            Ok(self.faces_to_elems.as_ref().unwrap())
        }
    }

    /// Clear the face-to-element connectivity
    pub fn clear_face_to_elems(&mut self) {
        debug!("Delete the face to element connectivity");
        self.faces_to_elems = None;
    }

    /// Compute the face-to-element connectivity
    pub fn compute_face_to_elems(&mut self) -> &FxHashMap<QE::Face, twovec::Vec<u32>> {
        debug!("Compute the face to element connectivity");
        if self.faces_to_elems.is_none() {
            self.faces_to_elems = Some(get_face_to_elem_quadratic(self.tris()));
        } else {
            warn!("Face to element connectivity already computed");
        }
        self.faces_to_elems.as_ref().unwrap()
    }

    /// Compute the element-to-element connectivity
    /// face-to-element connectivity is computed if not available
    pub fn compute_elem_to_elems(&mut self) -> &CSRGraph {
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
        self.elem_to_elems.as_ref().unwrap()
    }

    /// Clear the element-to-element connectivity
    pub fn clear_elem_to_elems(&mut self) {
        debug!("Delete the element to element connectivity");
        self.elem_to_elems = None;
    }

    /// Compute an octree to locate elements
    #[must_use]
    pub fn compute_vert_tree(&self) -> DefaultPointIndex<3> {
        debug!("Compute the vertex octree");
        <DefaultPointIndex<3> as PointIndex<3>>::newquadratic(self)
    }

    #[allow(clippy::type_complexity)]
    #[allow(clippy::too_many_lines)]
    pub fn boundary_faces(
        &self,
    ) -> Result<(
        Vec<QE::Face>,
        Vec<Tag>,
        HashMap<Tag, Tag>,
        HashMap<Tag, Vec<Tag>>,
    )> {
        debug!("Compute and order the boundary faces");
        if self.faces_to_elems.is_none() {
            return Err(Error::from("face to element connectivity not computed"));
        }

        let f2e = self.faces_to_elems.as_ref().unwrap();
        let n_bdy = f2e.iter().filter(|(_, v)| v.len() == 1).count();

        let mut tagged_faces: FxHashMap<QE::Face, Tag> = FxHashMap::default();
        for (mut face, ftag) in self.edges().zip(self.edgtags()) {
            face.sort();
            tagged_faces.insert(face, ftag);
        }

        let mut bdy = Vec::with_capacity(n_bdy);
        let mut bdy_tags = Vec::with_capacity(n_bdy);

        let n_elem_tags = self.tritags().collect::<FxHashSet<_>>().len();

        let mut next_boundary_tag = self.tri_tags.iter().max().unwrap_or(0) + 1;
        let mut boundary_faces_tags: HashMap<Tag, Tag> = HashMap::new();
        let mut next_internal_tag = next_boundary_tag + n_elem_tags as Tag;
        let mut internal_faces_tags: HashMap<Tag, Vec<Tag>> = HashMap::new();

        for (k, v) in f2e {
            if v.len() == 1 {
                // This is a boundary face
                let elem = self.tris.index(v[0]);
                let mut ok = false;
                for i_face in 0..QE::N_FACES {
                    let mut f = elem.face(i_face);
                    f.sort();
                    let is_same = !f.iter().zip(k.iter()).any(|(x, y)| x != y);
                    if is_same {
                        // face k is the i_face-th face of elem: use its orientation
                        #[allow(clippy::option_if_let_else)]
                        let tag = if let Some(tag) = tagged_faces.get(&f) {
                            *tag
                        } else {
                            let etag = self.tri_tags.index(v[0]);
                            if let Some(tag) = boundary_faces_tags.get(&etag) {
                                *tag
                            } else {
                                boundary_faces_tags.insert(etag, next_boundary_tag);
                                next_boundary_tag += 1;
                                next_boundary_tag - 1
                            }
                        };
                        let f = elem.face(i_face);
                        bdy.push(f);
                        bdy_tags.push(tag);
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
                        .map(|i| self.tri_tags.index(i))
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
                        .map(|i| self.tri_tags.index(i))
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

        Ok((bdy, bdy_tags, boundary_faces_tags, internal_faces_tags))
    }

    pub fn add_boundary_faces(&mut self) -> (HashMap<Tag, Tag>, HashMap<Tag, Vec<Tag>>) {
        debug!("Add the missing boundary faces & orient all faces outwards");
        if self.faces_to_elems.is_none() {
            self.compute_face_to_elems();
        }

        let (faces, ftags, boundary_faces, internal_faces) = self.boundary_faces().unwrap();
        self.edgs = faces.into();
        self.edg_tags = ftags.into();

        (boundary_faces, internal_faces)
    }

    #[must_use]
    pub fn boundary(&self) -> (QuadraticMesh<QE::Face>, Vec<Idx>) {
        debug!("Extract the mesh boundary");
        if self.edgs.is_empty() {
            return (
                QuadraticMesh::<QE::Face>::new(
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                ),
                Vec::new(),
            );
        }
        let (new_faces, indices) = reindexq(&self.edgs);
        let n_bdy_verts = indices.len();

        let mut verts = vec![Point::<3>::zeros(); n_bdy_verts];
        let mut vert_ids = vec![0; n_bdy_verts];

        for (old, new) in indices {
            verts[new as usize] = self.verts.index(old);
            vert_ids[new as usize] = old;
        }

        (
            QuadraticMesh::<QE::Face>::new_with_vector(
                verts.into(),
                new_faces.into(),
                self.edg_tags.clone(),
                Vec::new().into(),
                Vec::new().into(),
            ),
            vert_ids,
        )
    }

    /// Compute the mesh topology
    pub fn compute_topology(&mut self) -> &Topology {
        if self.topo.is_none() {
            let mut topo = Topology::new(QE::DIM as Dim);
            let vtags = topo.update_from_elems_and_faces_quadratic(
                &self.tris,
                &self.tri_tags,
                &self.edgs,
                &self.edg_tags,
            );
            self.topo = Some(topo);
            self.vtags = Some(vtags);
        } else {
            warn!("Topology already computed");
        }
        self.topo.as_ref().unwrap()
    }

    /// Get the topology
    pub fn get_topology(&self) -> Result<&Topology> {
        if self.topo.is_none() {
            Err(Error::from("Topology not computed"))
        } else {
            Ok(self.topo.as_ref().unwrap())
        }
    }

    #[must_use]
    pub fn extract_tag(&self, tag: Tag) -> SubQuadraticMesh<QE> {
        self.extract(|t| t == tag)
    }

    pub fn extract<F>(&self, elem_filter: F) -> SubQuadraticMesh<QE>
    where
        F: FnMut(Tag) -> bool,
    {
        let mut res = Self::empty();
        let (parent_vert_ids, parent_elem_ids, parent_face_ids) =
            res.add(self, elem_filter, |_| true, None);

        SubQuadraticMesh {
            mesh: res,
            parent_vert_ids,
            parent_elem_ids,
            parent_face_ids,
        }
    }
    pub fn clear_all(&mut self) {
        self.faces_to_elems = None;
        self.vertex_to_elems = None;
        self.elem_to_elems = None;
        self.topo = None;
        self.vtags = None;
    }

    pub fn add<F1, F2>(
        &mut self,
        other: &Self,
        mut elem_filter: F1,
        mut face_filter: F2,
        merge_tol: Option<f64>,
    ) -> (Vec<Idx>, Vec<Idx>, Vec<Idx>)
    where
        F1: FnMut(Tag) -> bool,
        F2: FnMut(Tag) -> bool,
    {
        self.clear_all();

        let n_verts = self.n_verts();
        let n_verts_other = other.n_verts();
        let mut new_vert_ids = vec![Idx::MAX; n_verts_other as usize];

        for (e, t) in other.tris().zip(other.tritags()) {
            if elem_filter(t) {
                e.iter()
                    .for_each(|&i| new_vert_ids[i as usize] = Idx::MAX - 1);
            }
        }

        // If needed, merge vertices
        if let Some(merge_tol) = merge_tol {
            let (bdy, ids) = self.boundary();
            let (obdy, oids) = other.boundary();
            if bdy.n_verts() > 0 && obdy.n_verts() > 0 {
                let tree = obdy.compute_vert_tree();
                bdy.verts().enumerate().for_each(|(i_self, vx)| {
                    let (i_other, _) = tree.nearest_vert(&vx);
                    let i_self = ids[i_self];
                    let i_other = oids[i_other as usize];
                    if (vx - other.vert(i_other)).norm() < merge_tol {
                        new_vert_ids[i_other as usize] = i_self as Idx;
                    }
                });
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
                self.verts.as_std_mut().push(other.verts.index(i as Idx));
            }
        });

        let mut added_elems = Vec::new();
        // keep track of the possible new faces
        let mut all_added_faces = FxHashSet::default();
        for (i, (e, t)) in other
            .tris()
            .zip(other.tritags())
            .enumerate()
            .filter(|(_, (_, t))| elem_filter(*t))
        {
            added_elems.push(i as Idx);
            let elem = QE::from_iter(e.iter().map(|&i| new_vert_ids[i as usize]));
            self.tris.as_std_mut().push(elem);
            self.tri_tags.as_std_mut().push(t);
            for i_face in 0..QE::N_FACES {
                let f = elem.face(i_face);
                all_added_faces.insert(f.sorted());
            }
        }

        let mut added_faces = Vec::new();
        for (i, (f, t)) in other
            .edges()
            .zip(other.edgtags())
            .enumerate()
            .filter(|(_, (_, t))| face_filter(*t))
            .map(|(i, (f, t))| {
                (
                    i,
                    (
                        QE::Face::from_iter(f.iter().map(|&i| new_vert_ids[i as usize])),
                        t,
                    ),
                )
            })
            .filter(|(_, (f, _))| all_added_faces.contains(&f.sorted()))
        {
            added_faces.push(i as Idx);
            self.edgs.as_std_mut().push(f);
            self.edg_tags.as_std_mut().push(t);
        }

        (added_verts, added_elems, added_faces)
    }
}

impl QuadraticMesh<QuadraticTriangle> {
    /// Create a QuadraticMesh from a SimplexMesh by adding midpoints on each edge
    #[must_use]
    pub fn from_simplex_mesh(mesh: &SimplexMesh<3, Triangle>) -> Self {
        let mut verts: Vec<Point<3>> = mesh.verts().collect::<Vec<_>>();
        let mut tris = Vec::new();
        let mut tri_tags: Vec<Tag> = Vec::new();
        let mut edgs: Vec<QuadraticEdge> = Vec::new();
        let mut edg_tags: Vec<Tag> = Vec::new();

        let mut edge_to_midpoint = std::collections::HashMap::new();

        for (tri, tag) in mesh.elems().zip(mesh.etags()) {
            let mut quadratic_tri = [0; 6];
            for i in 0..3 {
                quadratic_tri[i] = tri[i];
            }

            for i in 0..3 {
                let edge = if tri[i] < tri[(i + 1) % 3] {
                    (tri[i], tri[(i + 1) % 3])
                } else {
                    (tri[(i + 1) % 3], tri[i])
                };

                let midpoint_idx = *edge_to_midpoint.entry(edge).or_insert_with(|| {
                    let midpoint = (mesh.vert(edge.0) + mesh.vert(edge.1)) / 2.0;
                    verts.push(midpoint);
                    (verts.len() - 1) as Idx
                });

                quadratic_tri[3 + i] = midpoint_idx;
            }

            tris.push(QuadraticTriangle::from_slice(&quadratic_tri));
            tri_tags.push(tag);
        }

        for (edge, &midpoint_idx) in &edge_to_midpoint {
            edgs.push(QuadraticEdge::new(edge.0, edge.1, midpoint_idx));
            edg_tags.push(0); // Default tag for edges
        }

        Self::new(verts, tris, tri_tags, edgs, edg_tags)
    }
}

#[cfg(test)]
mod tests {
    use crate::mesh::test_meshes::test_mesh_2d_quadratic;
    use crate::mesh::{
        Point, QuadraticEdge, QuadraticMesh, QuadraticTriangle, SimplexMesh, Triangle,
    };

    #[test]
    fn test_2d_quadratic() {
        let mesh = test_mesh_2d_quadratic();

        assert_eq!(mesh.n_verts(), 6);
        assert_eq!(mesh.n_tris(), 1);
        assert_eq!(mesh.n_edges(), 3);

        assert_eq!(mesh.vert(0), Point::<3>::new(0., 0., 0.));
        assert_eq!(mesh.tri(0), QuadraticTriangle::new(0, 1, 2, 3, 4, 5));
        assert_eq!(mesh.edge(0), QuadraticEdge::new(0, 1, 3));
    }

    #[test]
    fn test_from_simplex_mesh() {
        // Create a SimplexMesh with a single triangle
        let verts = vec![
            Point::<3>::new(0.0, 0.0, 0.0),
            Point::<3>::new(1.0, 0.0, 0.0),
            Point::<3>::new(0.0, 1.0, 0.0),
        ];
        let tris = vec![Triangle::new(0, 1, 2)];
        let tri_tags = vec![1];
        let simplex_mesh =
            SimplexMesh::<3, Triangle>::new(verts.clone(), tris, tri_tags, vec![], vec![]);

        // Convert to QuadraticMesh
        let quadratic_mesh = QuadraticMesh::<QuadraticTriangle>::from_simplex_mesh(&simplex_mesh);
        // Check vertices
        assert_eq!(quadratic_mesh.n_verts(), 6);
        assert_eq!(quadratic_mesh.vert(0), verts[0]);
        assert_eq!(quadratic_mesh.vert(1), verts[1]);
        assert_eq!(quadratic_mesh.vert(2), verts[2]);

        assert_eq!(
            quadratic_mesh.vert(3),
            Point::<3>::new(0.5, 0.0, 0.0) // Midpoint of edge (0, 1)
        );

        assert_eq!(
            quadratic_mesh.vert(4),
            Point::<3>::new(0.5, 0.5, 0.0) // Midpoint of edge (1, 2)
        );

        assert_eq!(
            quadratic_mesh.vert(5),
            Point::<3>::new(0.0, 0.5, 0.0) // Midpoint of edge (2, 0)
        );

        // Check triangles
        assert_eq!(quadratic_mesh.n_tris(), 1);
        assert_eq!(
            quadratic_mesh.tri(0),
            QuadraticTriangle::new(0, 1, 2, 3, 4, 5)
        );

        // Check edges
        assert_eq!(quadratic_mesh.n_edges(), 3);
        // assert_eq!(quadratic_mesh.edge(0), QuadraticEdge::new(1, 2, 4));
        // // assert_eq!(quadratic_mesh.edge(1), QuadraticEdge::new(1, 2, 4));
        // // assert_eq!(quadratic_mesh.edge(2), QuadraticEdge::new(0, 2, 5));
    }
}
