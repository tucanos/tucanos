use crate::{
    Dim, Tag, TopoTag,
    geometry::Geometry,
    mesh::Topology,
    metric::{Metric, MetricElem},
    remesher::{Remesher, sequential::ElemInfo},
};
use core::fmt;
use log::{debug, trace};
use rustc_hash::FxHashSet;
use std::cmp::{Ordering, min};
use tmesh::{
    Vertex,
    mesh::{Edge, GSimplex, GenericMesh, Simplex},
};

#[derive(Debug, Clone, Copy)]
pub(super) enum Seed {
    No,
    Vertex(usize),
    Edge(Edge<usize>),
}

#[derive(Debug)]
#[allow(dead_code)]
pub(super) enum CavityCheckStatus {
    LongEdge(f64),
    ShortEdge(f64),
    Invalid,
    LowQuality(f64),
    Ok(f64),
}

/// Local cavity built from a mesh entity (vertex or edge)
/// Vertices and elements are copied from the original mesh and stored using a local numbering
///
#[derive(Debug)]
pub(super) struct Cavity<const D: usize, C: Simplex, M: Metric<D>> {
    /// Conversion from local to global vertex indices
    pub(super) local2global: Vec<usize>,
    /// Coordinates of the vertices
    pub(super) points: Vec<Vertex<D>>,
    /// Metric field at the vertices
    pub(super) metrics: Vec<M>,
    /// TopoTag of the vertices
    pub(super) tags: Vec<TopoTag>,
    /// Elements stored using the local vertex numbering
    pub(super) elems: Vec<C>,
    /// Element tags
    pub(super) etags: Vec<Tag>,
    /// Global element IDs
    pub(super) global_elem_ids: Vec<usize>,
    /// Faces shared by the cavity with the rest of the mesh.
    /// The faces lying on the mesh boundary are not included.
    /// The tag of the cavity element that contains the face is stored
    pub(super) faces: Vec<(C::FACE, Tag)>,
    /// Tagged faces
    pub(super) tagged_faces: Vec<(C::FACE, Tag)>,
    /// Tagged faces faces
    pub(super) tagged_bdys: Vec<(<C::FACE as Simplex>::FACE, Tag)>,
    /// Tagged faces faces flag (for 2d)
    pub(super) tagged_bdys_flg: Vec<bool>,
    /// From what the cavity was created (vertex, edge or face)
    pub(super) seed: Seed,
    /// Minimum element quality in the cavity
    pub(super) q_min: f64,
    /// Minimum edge length
    pub(super) l_min: f64,
    /// Maximum edge length
    pub(super) l_max: f64,
}

impl<const D: usize, C: Simplex, M: Metric<D>> Cavity<D, C, M> {
    /// Create a new (empty) cavity
    pub const fn new() -> Self {
        Self {
            local2global: Vec::new(),
            points: Vec::new(),
            metrics: Vec::new(),
            tags: Vec::new(),
            elems: Vec::new(),
            etags: Vec::new(),
            global_elem_ids: Vec::new(),
            faces: Vec::new(),
            tagged_faces: Vec::new(),
            tagged_bdys: Vec::new(),
            tagged_bdys_flg: Vec::new(),
            seed: Seed::No,
            q_min: -1.0,
            l_min: -1.0,
            l_max: f64::MAX,
        }
    }

    /// Clear the cavity data
    pub fn clear(&mut self) {
        self.local2global.clear();
        self.points.clear();
        self.metrics.clear();
        self.tags.clear();
        self.elems.clear();
        self.etags.clear();
        self.global_elem_ids.clear();
        self.faces.clear();
        self.tagged_faces.clear();
        self.tagged_bdys.clear();
        self.tagged_bdys_flg.clear();
        self.seed = Seed::No;
        self.q_min = -1.0;
        self.l_min = -1.0;
        self.l_max = f64::MAX;
    }

    /// Get the local vertex index from a global vertex index
    pub fn get_local_index(&self, i: usize) -> Option<usize> {
        let res = self
            .local2global
            .iter()
            .enumerate()
            .find(|(_x, y)| **y == i);
        if let Some((i, _j)) = res {
            Some(i)
        } else {
            None
        }
    }

    /// Intersect two sorted slices
    fn intersection(a: &[usize], b: &[usize]) -> Vec<usize> {
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

    pub fn init_from_edge(&mut self, edg: Edge<usize>, r: &Remesher<D, C, M>) {
        let global_elems =
            Self::intersection(r.vertex_elements(edg.get(0)), r.vertex_elements(edg.get(1)));
        assert!(
            !global_elems.is_empty(),
            "Empty edge cavity: edg = {edg:?} ({})--> elements = {:?} & {:?}",
            r.elem_count(edg),
            r.vertex_elements(edg.get(0)),
            r.vertex_elements(edg.get(1))
        );

        assert!(!global_elems.is_empty());
        self.compute(r, &global_elems, Seed::Edge(edg));
    }

    pub fn init_from_vertex(&mut self, i: usize, r: &Remesher<D, C, M>) {
        let global_elems = r.vertex_elements(i);
        assert!(
            !global_elems.is_empty(),
            "Empty vertex cavity: i = {i:?} (n_verts = {} / n_elems = {})",
            r.n_verts(),
            r.n_elems()
        );

        self.compute(r, global_elems, Seed::Vertex(i));
    }

    /// Return the coordinate and the metric of the barycenter of the points used
    /// to generate this cavity (ex: 2 points after using `init_from_edge`)
    pub fn seed_barycenter(&self) -> (Vertex<D>, M) {
        match &self.seed {
            Seed::No => unreachable!(),
            Seed::Vertex(x) => (self.points[*x], self.metrics[*x]),
            Seed::Edge(x) => (
                x.into_iter().map(|i| self.points[i]).sum::<Vertex<D>>() * 0.5,
                M::interpolate(x.into_iter().map(|i| (0.5, &self.metrics[i]))),
            ),
        }
    }

    fn add_elem(&mut self, r: &Remesher<D, C, M>, e: &ElemInfo<C>) -> C {
        let mut local = C::default();

        self.q_min = self.q_min.min(e.q);
        for (i, j) in e.el.into_iter().enumerate() {
            let idx = self.get_local_index(j);
            if let Some(idx) = idx {
                local.set(i, idx);
            } else {
                // new vertex
                self.local2global.push(j);
                local.set(i, self.local2global.len() - 1);
                let pt = r.get_vertex(j).unwrap();
                self.points.push(pt.0);
                self.metrics.push(pt.2);
                self.tags.push(pt.1);
            }
        }
        for i_edg in 0..C::N_EDGES {
            let edg = local.edge(i_edg);
            let l = M::edge_length(
                &self.points[edg.get(0)],
                &self.metrics[edg.get(0)],
                &self.points[edg.get(1)],
                &self.metrics[edg.get(1)],
            );
            self.l_min = self.l_min.min(l);
            self.l_max = self.l_max.max(l);
        }
        self.elems.push(local);
        self.etags.push(e.tag);

        local
    }

    /// Build the local cavity from a list of elements
    fn compute(&mut self, r: &Remesher<D, C, M>, global_elems: &[usize], x: Seed) {
        self.clear();
        self.q_min = f64::INFINITY;

        // Compute local2global & get the elements
        for i_global in global_elems.iter().copied() {
            self.global_elem_ids.push(i_global);
            let e = r.get_elem(i_global).unwrap();
            self.add_elem(r, &e);
        }
        trace!("Cavity: elems & vertices built");

        self.seed = match x {
            Seed::Vertex(i) => Seed::Vertex(self.get_local_index(i).unwrap()),
            Seed::Edge(edg) => Seed::Edge(Edge::new(
                self.get_local_index(edg.get(0)).unwrap(),
                self.get_local_index(edg.get(1)).unwrap(),
            )),
            Seed::No => unreachable!(),
        };

        self.compute_faces(r);

        trace!("Cavity built: {self:?}");
    }

    fn compute_faces(&mut self, r: &Remesher<D, C, M>) {
        for (face, tag) in self
            .elems
            .iter()
            .zip(self.etags.iter())
            .flat_map(|(e, &t)| (0..C::N_FACES).map(|i| e.face(i)).map(move |f| (f, t)))
        {
            match self.seed {
                Seed::Vertex(i) => {
                    if face.contains(i) {
                        if let Some(face_tag) = r.face_tag(&self.global_elem(&face)) {
                            let sorted = face.sorted();
                            if !self.tagged_faces.iter().any(|(f, _)| f.sorted() == sorted) {
                                self.tagged_faces.push((face, face_tag));
                                for i_bdy in 0..<C::FACE as Simplex>::N_FACES {
                                    let b = face.face(i_bdy);
                                    if !b.contains(i)
                                        && !self.tagged_bdys.iter().any(|(f, _)| f.sorted() == b)
                                    {
                                        self.tagged_bdys.push((b, face_tag));
                                        self.tagged_bdys_flg.push(C::N_VERTS == 3 && i_bdy == 0);
                                    }
                                }
                            }
                        }
                    } else {
                        self.faces.push((face, tag));
                    }
                }
                Seed::Edge(edg) => {
                    if face.contains_edge(&edg) {
                        if let Some(face_tag) = r.face_tag(&self.global_elem(&face)) {
                            let sorted = face.sorted();
                            if !self.tagged_faces.iter().any(|(f, _)| f.sorted() == sorted) {
                                self.tagged_faces.push((face, face_tag));
                                for i_bdy in 0..<C::FACE as Simplex>::N_FACES {
                                    let b = face.face(i_bdy);
                                    if !b.contains_edge(&edg)
                                        && !self.tagged_bdys.iter().any(|(f, _)| f.sorted() == b)
                                    {
                                        self.tagged_bdys.push((b, face_tag));
                                        self.tagged_bdys_flg.push(C::N_VERTS == 3 && i_bdy == 0);
                                    }
                                }
                            }
                        }
                    } else {
                        self.faces.push((face, tag));
                    }
                }
                Seed::No => unreachable!(),
            }
        }

        debug_assert!(!self.faces.is_empty());
    }

    /// Extend the cavity from a face
    pub fn extend(&mut self, r: &Remesher<D, C, M>, mut f: C::FACE, tag: Tag) -> bool {
        if let Some(i_face) = self.faces.iter().position(|&(x, _)| x == f) {
            f.invert();
            trace!("Extend from face {i_face} - {f:?}");
            let mut elems = Self::intersection(
                r.vertex_elements(self.local2global[f.get(0)]),
                r.vertex_elements(self.local2global[f.get(1)]),
            );
            if C::FACE::N_VERTS == 3 {
                elems = Self::intersection(&elems, r.vertex_elements(self.local2global[f.get(2)]));
            }

            match elems.len() {
                1 => {
                    debug!("cannot extend from a boundary face : {i_face} - {f:?}");
                    false
                }
                2 => {
                    let i_elem = if self.global_elem_ids.contains(&elems[0]) {
                        elems[1]
                    } else {
                        elems[0]
                    };
                    let e = r.elems.get(&i_elem).unwrap();
                    assert_eq!(tag, e.tag);
                    self.global_elem_ids.push(i_elem);
                    let e_local = self.add_elem(r, e);
                    trace!("Add elem {i_elem} - {e_local:?} (local)");

                    for i in 0..C::N_FACES {
                        let f_local = e_local.face(i);
                        if let Some(i_face) = self.faces.iter().position(|&(mut f, _)| {
                            f.invert();
                            f.is_same(&f_local)
                        }) {
                            trace!("Remove face {f_local:?} (local)");
                            self.faces.swap_remove(i_face);
                        } else {
                            trace!("Add face {f_local:?} (local)");
                            self.faces.push((f_local, tag));
                        }
                    }
                    true
                }
                _ => unreachable!("face {f:?} belongs to {} elements", elems.len()),
            }
        } else {
            trace!("Face {f:?} has been removed");
            true
        }
    }

    /// Get the number of vertices in the cavity
    pub const fn n_verts(&self) -> usize {
        self.points.len()
    }

    /// Get the number of elements in the cavity
    pub const fn n_elems(&self) -> usize {
        self.elems.len()
    }

    /// Get the i-the vertex & the associated tag and metric
    pub fn vert(&self, i: usize) -> (&Vertex<D>, TopoTag, &M) {
        (&self.points[i], self.tags[i], &self.metrics[i])
    }

    /// Get the i-th geometrical element
    pub fn gelem(&self, i: usize) -> MetricElem<D, C, M> {
        MetricElem::from_iter(
            self.elems[i]
                .into_iter()
                .map(|j| (self.points[j], self.metrics[j])),
        )
    }

    /// Get the i-th geometrical face
    pub fn gface(&self, face: &C::FACE) -> <<C as Simplex>::GEOM<D> as GSimplex<D>>::FACE {
        <<C as Simplex>::GEOM<D> as GSimplex<D>>::FACE::from_iter(
            face.into_iter().map(|x| self.points[x]),
        )
    }

    /// Convert a face from local to global vertex numbering
    pub fn global_elem<C2: Simplex>(&self, face: &C2) -> C2 {
        C2::from_iter(face.into_iter().map(|i| self.local2global[i]))
    }

    /// Return an iterator through the cavity faces
    pub fn faces(&self) -> impl ExactSizeIterator<Item = (C::FACE, Tag)> + '_ {
        self.faces.iter().copied()
    }

    /// Return an iterator through the cavity tagged faces (local indices)
    pub fn tagged_faces(&self) -> impl ExactSizeIterator<Item = (C::FACE, Tag)> + '_ {
        self.tagged_faces.iter().copied()
    }

    /// Return an iterator through the cavity tagged faces (global indices)
    pub fn global_tagged_faces(&self) -> impl ExactSizeIterator<Item = (C::FACE, Tag)> + '_ {
        self.tagged_faces().map(|(f, t)| (self.global_elem(&f), t))
    }

    /// Convert the filled cavity to a `SimplexMesh` to export it (for debug)
    #[allow(dead_code)]
    pub fn to_mesh(&self) -> GenericMesh<D, C> {
        let mut faces = Vec::new();
        let mut ftags = Vec::new();
        for (f, t) in self.faces() {
            faces.push(f);
            ftags.push(-t - 1);
        }
        for (f, t) in self.tagged_faces() {
            faces.push(f);
            ftags.push(t);
        }
        GenericMesh::from_vecs(
            self.points.clone(),
            self.elems.clone(),
            self.etags.clone(),
            faces,
            ftags,
        )
    }

    /// Get the vetices that are internal, i.e. do not belong to any face
    pub fn global_internal_vertices(&self) -> Vec<usize> {
        let bdy_verts = self
            .faces
            .iter()
            .flat_map(|(f, _)| f.into_iter())
            .collect::<FxHashSet<_>>();

        self.local2global
            .iter()
            .enumerate()
            .filter(|(i, _)| {
                let i = *i;
                !bdy_verts.contains(&i)
            })
            .map(|(_, j)| *j)
            .collect()
    }
}

impl<const D: usize, C: Simplex, M: Metric<D>> fmt::Display for Cavity<D, C, M> {
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
        writeln!(f, "Tagged faces")?;
        for e in &self.tagged_faces {
            writeln!(f, " {e:?}")?;
        }
        writeln!(f, "Tagged face boundaries")?;
        for (e, s) in self.tagged_bdys.iter().zip(self.tagged_bdys_flg.iter()) {
            writeln!(f, " {e:?}, sgn = {s}")?;
        }
        writeln!(f, "Type: {:?}", self.seed)?;

        Ok(())
    }
}

/// Filled cavity type
#[derive(Debug, Clone, Copy)]
pub enum FilledCavityType<const D: usize, M: Metric<D>> {
    ExistingVertex(usize),
    MovedVertex((usize, Vertex<D>, M)),
    EdgeCenter((Edge<usize>, Vertex<D>, M)),
}

/// Cavity reconstructed either from an existing cavity vertex or from a new one
pub struct FilledCavity<'a, const D: usize, C: Simplex, M: Metric<D>> {
    pub cavity: &'a Cavity<D, C, M>,
    pub ftype: FilledCavityType<D, M>,
}

impl<'a, const D: usize, C: Simplex, M: Metric<D>> FilledCavity<'a, D, C, M> {
    /// Construct a `FilledCavity`
    pub const fn new(cavity: &'a Cavity<D, C, M>, ftype: FilledCavityType<D, M>) -> Self {
        Self { cavity, ftype }
    }

    pub fn is_same(&self) -> bool {
        if let FilledCavityType::ExistingVertex(i) = self.ftype {
            self.cavity.elems.iter().all(|e| e.contains(i))
        } else {
            false
        }
    }

    /// Return an iterator through the cavity faces
    pub fn faces(&self) -> impl Iterator<Item = (C::FACE, Tag)> + '_ {
        self.cavity.faces().filter(|(f, _)| match self.ftype {
            FilledCavityType::ExistingVertex(i) => !f.contains(i),
            FilledCavityType::MovedVertex((i, _, _)) => !f.contains(i),
            FilledCavityType::EdgeCenter((edg, _, _)) => !f.contains_edge(&edg),
        })
    }

    /// Return the tagged faces
    pub fn tagged_faces_boundary(
        &self,
    ) -> impl Iterator<Item = (<C::FACE as Simplex>::FACE, Tag, bool)> + '_ {
        self.cavity
            .tagged_bdys
            .iter()
            .zip(self.cavity.tagged_bdys_flg.iter())
            .filter(|((b, _), _)| match self.ftype {
                FilledCavityType::ExistingVertex(i) => !b.contains(i),
                FilledCavityType::MovedVertex((i, _, _)) => !b.contains(i),
                FilledCavityType::EdgeCenter((edg, _, _)) => !b.contains_edge(&edg),
            })
            .map(|((b, t), s)| (*b, *t, *s))
    }

    /// Return the tagged faces
    pub fn tagged_faces_boundary_global(
        &self,
    ) -> impl Iterator<Item = (<C::FACE as Simplex>::FACE, Tag)> + '_ {
        self.tagged_faces_boundary()
            .map(|(b, t, _)| (self.cavity.global_elem(&b), t))
    }

    /// Check that the tagged faces are not already present (useful for collapse)
    pub fn check_tagged_faces(&self, r: &Remesher<D, C, M>) -> bool {
        if let FilledCavityType::ExistingVertex(i) = self.ftype {
            let i = self.cavity.local2global[i];
            for (b, _) in self.tagged_faces_boundary_global() {
                let f = C::FACE::from_vertex_and_face(i, &b);
                if r.face_tag(&f).is_some() {
                    return false;
                }
            }
            true
        } else {
            unreachable!();
        }
    }

    /// Convert the filled cavity to a `SimplexMesh` to export it (for debug)
    #[allow(dead_code)]
    pub fn to_mesh(&self) -> GenericMesh<D, C> {
        let mut verts = self.cavity.points.clone();
        if let FilledCavityType::EdgeCenter((_, x, _)) = self.ftype {
            verts.push(x);
        } else if let FilledCavityType::MovedVertex((i, x, _)) = self.ftype {
            verts[i] = x;
        }

        let mut elems = Vec::new();
        let mut faces = Vec::new();
        let mut etags = Vec::new();
        for (f, t) in self.faces() {
            match self.ftype {
                FilledCavityType::ExistingVertex(i) => elems.push(C::from_vertex_and_face(i, &f)),
                FilledCavityType::MovedVertex((i, _, _)) => {
                    elems.push(C::from_vertex_and_face(i, &f));
                }
                FilledCavityType::EdgeCenter(_) => {
                    let i = verts.len() - 1;
                    elems.push(C::from_vertex_and_face(i, &f));
                }
            }
            etags.push(t);
            faces.push(f);
        }
        let ftags = vec![1; faces.len()];
        GenericMesh::from_vecs(verts, elems, etags, faces, ftags)
    }

    /// Get the location and metric for the reconstruction vertex
    fn point(&self) -> (Vertex<D>, M) {
        match self.ftype {
            FilledCavityType::ExistingVertex(i) => (self.cavity.points[i], self.cavity.metrics[i]),
            FilledCavityType::MovedVertex((_, pt, m)) => (pt, m),
            FilledCavityType::EdgeCenter((_, pt, m)) => (pt, m),
        }
    }

    pub fn check(&self, l_min: f64, l_max: f64, q_min: f64) -> CavityCheckStatus {
        let (p0, m0) = self.point();
        let mut min_quality = 1.;
        for (f, _) in self.faces() {
            for i in f {
                let pi = &self.cavity.points[i];
                let mi = &self.cavity.metrics[i];
                let l = M::edge_length(&p0, &m0, pi, mi);
                if l < l_min {
                    trace!("cavity check failed: short edge");
                    return CavityCheckStatus::ShortEdge(l);
                }
                if l > l_max {
                    trace!("cavity check failed: long edge");
                    return CavityCheckStatus::LongEdge(l);
                }
            }

            let gf = self.cavity.gface(&f);
            let ge = C::GEOM::from_vert_and_face(&p0, &gf);
            let mut metrics = <C::GEOM<D> as GSimplex<D>>::ARRAY::<M>::default();
            metrics[0] = m0;
            for (i, j) in f.into_iter().enumerate() {
                metrics[i + 1] = self.cavity.metrics[j];
            }
            let me = MetricElem::<D, C, M>::new(ge, metrics);

            let q = me.quality();
            if q < 0.0 {
                trace!("cavity check failed: invalid element");
                return CavityCheckStatus::Invalid;
            } else if q <= q_min {
                trace!("cavity check failed: low quality ({q} < {q_min})");
                return CavityCheckStatus::LowQuality(q);
            }
            min_quality = f64::min(min_quality, q);
        }
        CavityCheckStatus::Ok(min_quality)
    }

    /// Check the the angle between the normal of the boundary faces and the normal given by the geometry is smaller than a threshold
    /// This is only required for swaps in 3D
    pub fn check_normals<G: Geometry<D>>(
        &self,
        topo: &Topology,
        geom: &G,
        threshold_degrees: f64,
    ) -> bool {
        let (p0, _) = self.point();

        if C::DIM == D {
            for (b, tag, s) in self.tagged_faces_boundary() {
                assert!(
                    tag > 0,
                    "Invalid tag {}\n{:?}\n{}\n{}",
                    tag,
                    self.ftype,
                    self.cavity,
                    topo
                );
                let gb = <<C::FACE as Simplex>::GEOM<D> as GSimplex<D>>::FACE::from_iter(
                    b.into_iter().map(|i| {
                        let (vx, _, _) = self.cavity.vert(i);
                        *vx
                    }),
                );
                let gf = <C::FACE as Simplex>::GEOM::from_vert_and_face(&p0, &gb);
                let center = gf.center();
                let mut normal = gf.normal(None).normalize();
                if s {
                    normal *= -1.0;
                }
                let a = geom.angle(&center, &normal, &(C::DIM as Dim - 1, tag));
                if a > threshold_degrees {
                    return false;
                }
            }
        } else {
            // all the element tags should be equal
            // let etag_min = self.cavity.etags.iter().copied().min().unwrap();
            // let etag_max = self.cavity.etags.iter().copied().max().unwrap();
            // assert_eq!(etag_min, etag_max);
            for (f, tag) in self.faces() {
                let gf = self.cavity.gface(&f);
                let ge = C::GEOM::from_vert_and_face(&p0, &gf);
                let center = ge.center();
                let normal = ge.normal(None).normalize();
                let a = geom.angle(&center, &normal, &(C::DIM as Dim, tag));
                if a > threshold_degrees {
                    return false;
                }
            }
        }

        true
    }
}
