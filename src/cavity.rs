use crate::{
    geom_elems::GElem,
    geometry::Geometry,
    mesh::{Point, SimplexMesh},
    metric::Metric,
    remesher::Remesher,
    topo_elems::Elem,
    topology::Topology,
    Dim, Idx, TopoTag,
};
use core::fmt;
use log::trace;
use rustc_hash::FxHashMap;
use std::{
    array,
    cmp::{min, Ordering},
    hash::BuildHasherDefault,
};

#[derive(Debug, Clone, Copy)]
pub enum CavityType<const D: usize> {
    No,
    Vertex(Idx),
    Edge([Idx; 2]),
    Face([Idx; D]),
}

/// Local cavity built from a mesh entity (vertex or edge)
/// Vertices and elements are copied from the original mesh and stored using a local numbering
///
#[derive(Debug)]
pub struct Cavity<const D: usize, E: Elem, M: Metric<D>> {
    /// Conversion from local to global vertex indices
    pub local2global: Vec<Idx>,
    /// Coordinates of the vertices
    pub points: Vec<Point<D>>,
    /// Metric field at the vertices
    pub metrics: Vec<M>,
    /// TopoTag of the vertices
    pub tags: Vec<TopoTag>,
    /// Elements stored using the local vertex numbering
    pub elems: Vec<E>,
    /// Global element IDs
    pub global_elem_ids: Vec<Idx>,
    /// Faces shared by the cavity with the rest of the mesh. The faces lying on the mesh boundary are not included.
    pub faces: Vec<E::Face>,
    /// Cavity type (vertex / edge)
    pub ctype: CavityType<D>,
    /// Minimum element quality in the cavity
    pub q_min: f64,
}

impl<const D: usize, E: Elem, M: Metric<D>> Cavity<D, E, M> {
    /// Create a new (empty) cavity
    pub fn new() -> Self {
        Self {
            local2global: Vec::new(),
            points: Vec::new(),
            metrics: Vec::new(),
            tags: Vec::new(),
            elems: Vec::new(),
            global_elem_ids: Vec::new(),
            faces: Vec::new(),
            ctype: CavityType::No,
            q_min: -1.0,
        }
    }

    /// Clear the cavity data
    pub fn clear(&mut self) {
        self.local2global.clear();
        self.points.clear();
        self.metrics.clear();
        self.tags.clear();
        self.elems.clear();
        self.global_elem_ids.clear();
        self.faces.clear();
        self.ctype = CavityType::No;
        self.q_min = -1.0;
    }

    /// Get the local vertex index from a global vertex index
    pub fn get_local_index(&self, i: Idx) -> Option<Idx> {
        let res = self
            .local2global
            .iter()
            .enumerate()
            .find(|(_x, y)| **y == i);
        if let Some((i, _j)) = res {
            Some(i as Idx)
        } else {
            None
        }
    }

    /// Intersect two sorted slices
    fn intersection(a: &[Idx], b: &[Idx]) -> Vec<Idx> {
        let mut result = Vec::with_capacity(min(a.len(), b.len()));
        let mut i = 0;
        let mut j = 0;
        while i < a.len() && j < b.len() {
            match a[i].cmp(&b[j]) {
                Ordering::Equal => {
                    result.push(a[i]);
                    i += 1;
                    j += 1;
                }
                Ordering::Less => i += 1,
                Ordering::Greater => j += 1,
            }
        }
        result
    }

    pub fn init_from_edge(&mut self, edg: [Idx; 2], r: &Remesher<D, E, M>) {
        let global_elems = Self::intersection(r.vertex_elements(edg[0]), r.vertex_elements(edg[1]));
        debug_assert!(!global_elems.is_empty());
        self.compute(r, &global_elems, CavityType::Edge(edg));
    }

    pub fn init_from_vertex(&mut self, i: Idx, r: &Remesher<D, E, M>) {
        self.compute(r, r.vertex_elements(i), CavityType::Vertex(i));
    }

    pub fn init_from_face(&mut self, face: &E::Face, r: &Remesher<D, E, M>) {
        let global_elems = face
            .iter()
            .map(|&x| r.vertex_elements(x).to_vec())
            .reduce(|a, b| Self::intersection(&a, &b))
            .unwrap();
        debug_assert!(
            !global_elems.is_empty(),
            "Cannot create cavity for face {:?} with these elements: {:?}",
            face,
            face.iter()
                .map(|&x| (
                    x,
                    r.vertex_elements(x)
                        .iter()
                        .map(|&i| r.get_elem(i).unwrap().el)
                        .collect::<Vec<_>>()
                ))
                .collect::<Vec<_>>()
        );
        self.compute(
            r,
            &global_elems,
            CavityType::Face(array::from_fn(|i| face[i])),
        );
    }

    /// Return the coordinate and the metric of the barycenter of the points used
    /// to generate this cavity (ex: 2 points after using `init_from_edge`)
    pub fn barycenter(&self) -> (Point<D>, M) {
        let local_ids = match &self.ctype {
            CavityType::No => unreachable!(),
            CavityType::Vertex(x) => std::slice::from_ref(x),
            CavityType::Edge(x) => x.as_slice(),
            CavityType::Face(x) => x.as_slice(),
        };
        let scale = 1. / local_ids.len() as f64;
        (
            local_ids
                .iter()
                .map(|&i| self.points[i as usize])
                .sum::<Point<D>>()
                * scale,
            M::interpolate(self.metrics.iter().map(|x| (scale, x))),
        )
    }

    /// Build the local cavity from a list of elements
    fn compute(&mut self, r: &Remesher<D, E, M>, global_elems: &[Idx], x: CavityType<D>) {
        self.clear();
        self.q_min = f64::INFINITY;

        // Compute local2global & get the elements
        let mut local = E::default();
        for i_global in global_elems.iter().copied() {
            self.global_elem_ids.push(i_global);
            let e = r.get_elem(i_global).unwrap();
            self.q_min = self.q_min.min(e.q);
            for (i, j) in e.el.iter().enumerate() {
                let idx = self.get_local_index(*j);
                if let Some(idx) = idx {
                    local[i] = idx;
                } else {
                    // new vertex
                    self.local2global.push(*j);
                    local[i] = self.local2global.len() as Idx - 1;
                    let pt = r.get_vertex(*j).unwrap();
                    self.points.push(pt.0);
                    self.metrics.push(pt.2);
                    self.tags.push(pt.1);
                }
            }
            self.elems.push(local);
        }
        trace!("Cavity: elems & vertices built");

        self.ctype = match x {
            CavityType::Vertex(i) => {
                let i: [Idx; 1] = self.compute_boundary_faces(std::iter::once(i));
                CavityType::Vertex(i[0])
            }
            CavityType::Edge(edg) => CavityType::Edge(self.compute_boundary_faces(edg.into_iter())),
            CavityType::Face(f) => CavityType::Face(self.compute_boundary_faces(f.into_iter())),
            CavityType::No => unreachable!(),
        };
        trace!("Cavity built: {:?}", self);
    }

    /// Get the number of vertices in the cavity
    pub fn n_verts(&self) -> Idx {
        self.points.len() as Idx
    }

    /// Get the number of elements in the cavity
    pub fn n_elems(&self) -> Idx {
        self.elems.len() as Idx
    }

    /// Get the number of faces in the cavity
    pub fn n_faces(&self) -> Idx {
        self.faces.len() as Idx
    }

    /// Get the i-the vertex & the associated tag and metric
    pub fn vert(&self, i: Idx) -> (&Point<D>, TopoTag, &M) {
        (
            &self.points[i as usize],
            self.tags[i as usize],
            &self.metrics[i as usize],
        )
    }

    /// Get the i-th geometrical element
    pub fn gelem(&self, i: Idx) -> E::Geom<D, M> {
        E::Geom::from_verts(
            self.elems[i as usize]
                .iter()
                .map(|x| *x as usize)
                .map(|x| (self.points[x], self.metrics[x])),
        )
    }

    /// Get the i-th geometrical face
    pub fn gface(&self, face: &E::Face) -> <<E as Elem>::Geom<D, M> as GElem<D, M>>::Face {
        <<E as Elem>::Geom<D, M> as GElem<D, M>>::Face::from_verts(
            face.iter()
                .map(|x| *x as usize)
                .map(|x| (self.points[x], self.metrics[x])),
        )
    }

    /// Convert a face from local to global vertex numbering
    pub fn global_face(&self, face: &E::Face) -> E::Face {
        let mut res = E::Face::default();
        for (i, j) in face.iter().enumerate() {
            res[i] = self.local2global[*j as usize];
        }
        res
    }

    fn compute_boundary_faces<const N: usize, I: Iterator<Item = Idx>>(
        &mut self,
        mut point_indices: I,
    ) -> [Idx; N] {
        let local =
            array::from_fn(|_| self.get_local_index(point_indices.next().unwrap()).unwrap());
        self.faces.extend(self.elems.iter().flat_map(|e| {
            (0..E::N_FACES)
                .map(|i| e.face(i))
                // faces which does not contains cavity vertices
                .filter(|f| local.iter().any(|&i| !f.contains_vertex(i)))
        }));
        debug_assert!(!self.faces.is_empty());
        local
    }

    /// Return an iterator through the cavity faces
    pub fn faces(&self) -> impl Iterator<Item = E::Face> + '_ {
        (0..(self.n_faces())).map(|i_face| self.faces[i_face as usize])
    }

    /// Convert the filled cavity to a `SimplexMesh` to export it (for debug)
    #[allow(dead_code)]
    pub fn to_mesh(&self) -> SimplexMesh<D, E> {
        let etags = vec![1; self.elems.len()];
        let faces = Vec::new();
        let ftags = Vec::new();
        SimplexMesh::<D, E>::new(self.points.clone(), self.elems.clone(), etags, faces, ftags)
    }
}

impl<const D: usize, E: Elem, M: Metric<D>> fmt::Display for Cavity<D, E, M> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "Vertices")?;
        for ((p, t), i) in self
            .points
            .iter()
            .zip(self.tags.iter())
            .zip(self.local2global.iter())
        {
            writeln!(f, " {p:?} {t:?} {i}")?;
        }
        writeln!(f, "Elements")?;
        for e in &self.elems {
            writeln!(f, " {e:?}")?;
        }
        writeln!(f, "Faces")?;
        for e in &self.faces {
            writeln!(f, " {e:?}")?;
        }

        writeln!(f, "Type: {:?}", self.ctype)?;

        Ok(())
    }
}

/// Cavity reconstructed either from an existing cavity vertex or from a new one
pub struct FilledCavity<'a, const D: usize, E: Elem, M: Metric<D>> {
    pub cavity: &'a Cavity<D, E, M>,
    pub id: Option<Idx>,
    pub pt: Option<Point<D>>,
    pub m: Option<M>,
}

impl<'a, const D: usize, E: Elem, M: Metric<D>> FilledCavity<'a, D, E, M> {
    /// Construct a `FilledCavity` from a Cavity and one of its vertices
    pub const fn from_cavity_and_vertex_id(cavity: &'a Cavity<D, E, M>, node: Idx) -> Self {
        Self {
            cavity,
            id: Some(node),
            pt: None,
            m: None,
        }
    }

    /// Construct a `FilledCavity` from a Cavity and a new vertex
    pub const fn from_cavity_and_new_vertex(
        cavity: &'a Cavity<D, E, M>,
        pt: &Point<D>,
        m: &M,
    ) -> Self {
        Self {
            cavity,
            id: None,
            pt: Some(*pt),
            m: Some(*m),
        }
    }

    /// Construct a `FilledCavity` from a Cavity when its center is moved
    pub fn from_cavity_and_moved_vertex(cavity: &'a Cavity<D, E, M>, pt: &Point<D>, m: &M) -> Self {
        if let CavityType::Vertex(vx) = cavity.ctype {
            Self {
                cavity,
                id: Some(vx),
                pt: Some(*pt),
                m: Some(*m),
            }
        } else {
            panic!("from_cavity_and_moved_vertex can only be used for vertex cavities")
        }
    }

    /// Return an iterator through the cavity faces
    pub fn faces(&self) -> impl Iterator<Item = E::Face> + '_ {
        self.cavity.faces().filter(|f| {
            if self.id.is_some() && f.contains_vertex(self.id.unwrap()) {
                return false;
            }
            if let CavityType::Edge(edg) = self.cavity.ctype {
                assert!(!f.contains_edge(edg));
            }
            if let CavityType::Vertex(v) = self.cavity.ctype {
                assert!(!f.contains_vertex(v));
            }
            true
        })
    }

    /// Return an iterator through the cavity faces and the corresponding new elements
    pub fn elems_and_faces(&self) -> impl Iterator<Item = (E, E::Face)> + '_ {
        self.faces()
            .map(|f| (E::from_vertex_and_face(self.id.unwrap(), &f), f))
    }

    /// Convert the filled cavity to a `SimplexMesh` to export it (for debug)
    #[allow(dead_code)]
    pub fn to_mesh(&self) -> SimplexMesh<D, E> {
        let mut elems = Vec::new();
        let mut faces = Vec::new();
        for (e, f) in self.elems_and_faces() {
            elems.push(e);
            faces.push(f);
        }
        let etags = vec![1; elems.len()];
        let ftags = vec![1; faces.len()];
        SimplexMesh::<D, E>::new(self.cavity.points.clone(), elems, etags, faces, ftags)
    }

    /// Get the location and metric for the reconstruction vertex
    fn point(&self) -> (&Point<D>, &M) {
        if self.pt.is_some() {
            (self.pt.as_ref().unwrap(), self.m.as_ref().unwrap())
        } else {
            (
                &self.cavity.points[self.id.unwrap() as usize],
                &self.cavity.metrics[self.id.unwrap() as usize],
            )
        }
    }

    pub fn check(&self, l_min: f64, l_max: f64, q_min: f64) -> f64 {
        let (p0, m0) = self.point();

        let mut min_quality = 1.;
        for f in self.faces() {
            for i in f.iter() {
                let pi = &self.cavity.points[*i as usize];
                let mi = &self.cavity.metrics[*i as usize];
                let l = M::edge_length(p0, m0, pi, mi);
                if l < l_min {
                    trace!("cavity check failed: short edge");
                    return -1.0;
                }
                if l > l_max {
                    trace!("cavity check failed: long edge");
                    return -1.0;
                }
            }

            let gf = self.cavity.gface(&f);
            let ge = E::Geom::from_vert_and_face(p0, m0, &gf);

            let q = ge.quality();
            if q < 0.0 {
                trace!("cavity check failed: invalid element");
                return -1.;
            } else if q < q_min {
                trace!("cavity check failed: low quality ({} < {})", q, q_min);
                return -1.;
            }
            min_quality = f64::min(min_quality, q);
        }
        min_quality
    }

    pub fn check_tags(&self, topo: &Topology) -> bool {
        // Check if there may be a face on the boundary
        let mut n = 0;
        for t in &self.cavity.tags {
            if t.0 < (E::DIM as Dim) {
                n += 1;
            }
        }
        if n < E::Face::N_VERTS {
            return true;
        }

        let mut faces = FxHashMap::with_hasher(BuildHasherDefault::default());

        // Build the cavity internal & boundary faces
        for (e, mut f) in self.elems_and_faces() {
            let vtags = e.iter().map(|i| self.cavity.tags[*i as usize]);
            let etag = topo.elem_tag(vtags).unwrap();
            if etag.0 != E::DIM as Dim {
                return false;
            }
            f.sort();
            for j_face in 0..E::N_FACES {
                let mut fj = e.face(j_face);
                fj.sort();
                // if fj != f
                if f.iter().zip(fj.iter()).any(|(x, y)| x != y) {
                    if let Some(val) = faces.get_mut(&fj) {
                        *val += 1;
                    } else {
                        faces.insert(fj, 1);
                    }
                }
            }
        }

        // Check that none of the internal faces are incorectly tagged
        for (fj, n) in &faces {
            let vtags = fj.iter().map(|i| self.cavity.tags[*i as usize]);
            let tag = topo.elem_tag(vtags).unwrap();
            match (tag.0).cmp(&(E::Face::DIM as Dim)) {
                Ordering::Greater => {
                    continue;
                }
                Ordering::Less => {
                    return false;
                }
                Ordering::Equal => {
                    let topo_node = topo.get(tag).unwrap();
                    let mut parents = topo_node.parents.clone();

                    if parents.len() == 1 {
                        // a boundary face shound belong to only one element
                        if *n != 1 {
                            return false;
                        }
                    } else {
                        // otherwise, check that all parent tags are present
                        for (e, _f) in self.elems_and_faces() {
                            let fj_in_e = fj.iter().all(|i| e.contains_vertex(*i));
                            if fj_in_e {
                                let vtags = e.iter().map(|i| self.cavity.tags[*i as usize]);
                                let etag = topo.elem_tag(vtags).unwrap();
                                let _ok = parents.remove(&etag.1);
                            }
                        }
                        if !parents.is_empty() {
                            return false;
                        }
                    }
                }
            }
        }
        true
    }

    /// Check the the angle between the normal of the boundary faces and the normal given by the geometry is smaller than a threshold
    /// This is only required for swaps in 3D
    pub fn check_boundary_normals<G: Geometry<D>>(
        &self,
        topo: &Topology,
        geom: &G,
        threshold_degrees: f64,
    ) -> bool {
        if D >= 3 {
            // when both the edge & reconstruction vertex are on a boundary
            let check = match self.cavity.ctype {
                CavityType::Edge(edg) => {
                    self.cavity.tags[edg[0] as usize].0 < E::DIM as Dim
                        && self.cavity.tags[edg[1] as usize].0 < E::DIM as Dim
                }
                CavityType::Vertex(vx) => self.cavity.tags[vx as usize].0 < E::DIM as Dim,
                CavityType::Face(_) => todo!("Face cavity are never used on a boundary"),
                CavityType::No => unreachable!(),
            };
            let node_on_bdy = self.cavity.tags[self.id.unwrap() as usize].0 < E::DIM as Dim;
            if check && node_on_bdy {
                let (p0, _) = self.point();
                // Check for boundary faces
                for f in self.faces() {
                    for i_edge in 0..E::Face::N_FACES {
                        let e = f.face(i_edge);
                        let new_f = E::Face::from_vertex_and_face(self.id.unwrap(), &e);
                        let vtags = new_f.iter().map(|i| self.cavity.tags[*i as usize]);
                        let ftag = topo.elem_tag(vtags).unwrap();
                        let topo_node = topo.get(ftag).unwrap();
                        let parents = &topo_node.parents;
                        if ftag.0 == E::Face::DIM as Dim && parents.len() == 1 {
                            // boundary face.
                            // let p0 = &self.cavity.points[new_f[0] as usize];
                            let p1 = &self.cavity.points[new_f[1] as usize];
                            let p2 = &self.cavity.points[new_f[2] as usize];
                            let mut n = (p2 - p0).cross(&(p1 - p0));
                            n.normalize_mut();
                            let mut c = (p0 + p1 + p2) / 3.0;
                            let a = geom.angle(&mut c, &n, &ftag);
                            if a > threshold_degrees {
                                return false;
                            }
                        }
                    }
                }
            }
        }
        true
    }
}
