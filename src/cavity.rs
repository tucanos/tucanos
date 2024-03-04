use crate::{
    geom_elems::GElem,
    geometry::Geometry,
    mesh::{Point, SimplexMesh},
    metric::Metric,
    remesher::Remesher,
    topo_elems::Elem,
    Dim, Idx, Tag, TopoTag,
};
use core::fmt;
use log::trace;
use std::cmp::{min, Ordering};

#[derive(Debug, Clone, Copy)]
pub enum Seed {
    No,
    Vertex(Idx),
    Edge([Idx; 2]),
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
    /// Element tags
    pub etags: Vec<Tag>,
    /// Global element IDs
    pub global_elem_ids: Vec<Idx>,
    /// Faces shared by the cavity with the rest of the mesh.
    /// The faces lying on the mesh boundary are not included.
    /// The tag of the cavity element that contains the face is stored
    pub faces: Vec<(E::Face, Tag)>,
    /// Tagged faces
    pub tagged_faces: Vec<(E::Face, Tag)>,
    /// Tagged faces faces
    pub tagged_bdys: Vec<(<E::Face as Elem>::Face, Tag)>,
    /// From what the cavity was created (vertex, edge or face)
    pub seed: Seed,
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
            etags: Vec::new(),
            global_elem_ids: Vec::new(),
            faces: Vec::new(),
            tagged_faces: Vec::new(),
            tagged_bdys: Vec::new(),
            seed: Seed::No,
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
        self.etags.clear();
        self.global_elem_ids.clear();
        self.faces.clear();
        self.tagged_faces.clear();
        self.tagged_bdys.clear();
        self.seed = Seed::No;
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
        assert!(
            !global_elems.is_empty(),
            "Empty edge cavity: edg = {edg:?} ({})--> elements = {:?} & {:?}",
            r.elem_count(edg),
            r.vertex_elements(edg[0]),
            r.vertex_elements(edg[1])
        );

        assert!(!global_elems.is_empty());
        self.compute(r, &global_elems, Seed::Edge(edg));
    }

    pub fn init_from_vertex(&mut self, i: Idx, r: &Remesher<D, E, M>) {
        self.compute(r, r.vertex_elements(i), Seed::Vertex(i));
    }

    /// Return the coordinate and the metric of the barycenter of the points used
    /// to generate this cavity (ex: 2 points after using `init_from_edge`)
    pub fn seed_barycenter(&self) -> (Point<D>, M) {
        let local_ids = match &self.seed {
            Seed::No => unreachable!(),
            Seed::Vertex(x) => std::slice::from_ref(x),
            Seed::Edge(x) => x.as_slice(),
        };
        let scale = 1. / local_ids.len() as f64;
        (
            local_ids
                .iter()
                .map(|&i| self.points[i as usize])
                .sum::<Point<D>>()
                * scale,
            M::interpolate(
                local_ids
                    .iter()
                    .map(|&i| (scale, &self.metrics[i as usize])),
            ),
        )
    }

    /// Build the local cavity from a list of elements
    fn compute(&mut self, r: &Remesher<D, E, M>, global_elems: &[Idx], x: Seed) {
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
            self.etags.push(e.tag);
        }
        trace!("Cavity: elems & vertices built");

        self.seed = match x {
            Seed::Vertex(i) => Seed::Vertex(self.get_local_index(i).unwrap()),
            Seed::Edge(edg) => Seed::Edge([
                self.get_local_index(edg[0]).unwrap(),
                self.get_local_index(edg[1]).unwrap(),
            ]),
            Seed::No => unreachable!(),
        };

        self.compute_faces(r);

        trace!("Cavity built: {:?}", self);
    }

    fn compute_faces(&mut self, r: &Remesher<D, E, M>) {
        for (face, tag) in self
            .elems
            .iter()
            .zip(self.etags.iter())
            .flat_map(|(e, &t)| (0..E::N_FACES).map(|i| e.face(i)).map(move |f| (f, t)))
        {
            match self.seed {
                Seed::Vertex(i) => {
                    if face.contains_vertex(i) {
                        if let Some(face_tag) = r.face_tag(&self.global_elem(&face)) {
                            let sorted = face.sorted();
                            if !self.tagged_faces.iter().any(|(f, _)| f.sorted() == sorted) {
                                self.tagged_faces.push((face, face_tag));
                                for i_bdy in 0..<E::Face as Elem>::N_FACES {
                                    let b = face.face(i_bdy);
                                    if !b.contains_vertex(i)
                                        && !self.tagged_bdys.iter().any(|(f, _)| f.sorted() == b)
                                    {
                                        self.tagged_bdys.push((b, face_tag));
                                    }
                                }
                            }
                        }
                    } else {
                        self.faces.push((face, tag));
                    }
                }
                Seed::Edge(edg) => {
                    if face.contains_edge(edg) {
                        if let Some(face_tag) = r.face_tag(&self.global_elem(&face)) {
                            let sorted = face.sorted();
                            if !self.tagged_faces.iter().any(|(f, _)| f.sorted() == sorted) {
                                self.tagged_faces.push((face, face_tag));
                                for i_bdy in 0..<E::Face as Elem>::N_FACES {
                                    let b = face.face(i_bdy);
                                    if !b.contains_edge(edg)
                                        && !self.tagged_bdys.iter().any(|(f, _)| f.sorted() == b)
                                    {
                                        self.tagged_bdys.push((b, face_tag));
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

    /// Get the number of vertices in the cavity
    pub fn n_verts(&self) -> Idx {
        self.points.len() as Idx
    }

    /// Get the number of elements in the cavity
    pub fn n_elems(&self) -> Idx {
        self.elems.len() as Idx
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
    pub fn global_elem<EE: Elem>(&self, face: &EE) -> EE {
        EE::from_iter(face.iter().map(|&i| self.local2global[i as usize]))
    }

    /// Return an iterator through the cavity faces
    pub fn faces(&self) -> impl Iterator<Item = (E::Face, Tag)> + '_ {
        self.faces.iter().copied()
    }

    /// Return an iterator through the cavity tagged faces (local indices)
    pub fn tagged_faces(&self) -> impl Iterator<Item = (E::Face, Tag)> + '_ {
        self.tagged_faces.iter().copied()
    }

    /// Return an iterator through the cavity tagged faces (global indices)
    pub fn global_tagged_faces(&self) -> impl Iterator<Item = (E::Face, Tag)> + '_ {
        self.tagged_faces().map(|(f, t)| (self.global_elem(&f), t))
    }

    /// Convert the filled cavity to a `SimplexMesh` to export it (for debug)
    #[allow(dead_code)]
    pub fn to_mesh(&self) -> SimplexMesh<D, E> {
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
        SimplexMesh::<D, E>::new(
            self.points.clone(),
            self.elems.clone(),
            self.etags.clone(),
            faces,
            ftags,
        )
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

        writeln!(f, "Type: {:?}", self.seed)?;

        Ok(())
    }
}

/// Filled cavity type
pub enum FilledCavityType<const D: usize, M: Metric<D>> {
    ExistingVertex(Idx),
    MovedVertex((Idx, Point<D>, M)),
    EdgeCenter(([Idx; 2], Point<D>, M)),
}

/// Cavity reconstructed either from an existing cavity vertex or from a new one
pub struct FilledCavity<'a, const D: usize, E: Elem, M: Metric<D>> {
    pub cavity: &'a Cavity<D, E, M>,
    pub ftype: FilledCavityType<D, M>,
}

impl<'a, const D: usize, E: Elem, M: Metric<D>> FilledCavity<'a, D, E, M> {
    /// Construct a `FilledCavity`
    pub const fn new(cavity: &'a Cavity<D, E, M>, ftype: FilledCavityType<D, M>) -> Self {
        Self { cavity, ftype }
    }

    pub fn is_same(&self) -> bool {
        if let FilledCavityType::ExistingVertex(i) = self.ftype {
            self.cavity.elems.iter().all(|e| e.contains_vertex(i))
        } else {
            false
        }
    }

    /// Return an iterator through the cavity faces
    pub fn faces(&self) -> impl Iterator<Item = (E::Face, Tag)> + '_ {
        self.cavity.faces().filter(|(f, _)| {
            // if let FilledCavityType::ExistingVertex(_) = self.ftype {
            //     if let Seed::Edge(edg) = self.cavity.seed {
            //         if f.contains_edge(edg) {
            //             return false;
            //         }
            //     }
            // }

            match self.ftype {
                FilledCavityType::ExistingVertex(i) => !f.contains_vertex(i),
                FilledCavityType::MovedVertex((i, _, _)) => !f.contains_vertex(i),
                FilledCavityType::EdgeCenter((edg, _, _)) => !f.contains_edge(edg),
            }
        })
    }

    /// Return the tagged faces
    pub fn tagged_faces_boundary(
        &self,
    ) -> impl Iterator<Item = (<E::Face as Elem>::Face, Tag)> + '_ {
        self.cavity
            .tagged_bdys
            .iter()
            .filter(|(b, _)| match self.ftype {
                FilledCavityType::ExistingVertex(i) => !b.contains_vertex(i),
                FilledCavityType::MovedVertex((i, _, _)) => !b.contains_vertex(i),
                FilledCavityType::EdgeCenter((edg, _, _)) => !b.contains_edge(edg),
            })
            .map(|(b, t)| (*b, *t))
    }

    /// Return the tagged faces
    pub fn tagged_faces_boundary_global(
        &self,
    ) -> impl Iterator<Item = (<E::Face as Elem>::Face, Tag)> + '_ {
        self.tagged_faces_boundary()
            .map(|(b, t)| (self.cavity.global_elem(&b), t))
    }

    /// Convert the filled cavity to a `SimplexMesh` to export it (for debug)
    #[allow(dead_code)]
    pub fn to_mesh(&self) -> SimplexMesh<D, E> {
        let mut verts = self.cavity.points.clone();
        if let FilledCavityType::EdgeCenter((_, x, _)) = self.ftype {
            verts.push(x);
        } else if let FilledCavityType::MovedVertex((i, x, _)) = self.ftype {
            verts[i as usize] = x;
        }

        let mut elems = Vec::new();
        let mut faces = Vec::new();
        let mut etags = Vec::new();
        for (f, t) in self.faces() {
            match self.ftype {
                FilledCavityType::ExistingVertex(i) => elems.push(E::from_vertex_and_face(i, &f)),
                FilledCavityType::MovedVertex((i, _, _)) => {
                    elems.push(E::from_vertex_and_face(i, &f));
                }
                FilledCavityType::EdgeCenter(_) => {
                    let i = verts.len() as Idx - 1;
                    elems.push(E::from_vertex_and_face(i, &f));
                }
            }
            etags.push(t);
            faces.push(f);
        }
        let ftags = vec![1; faces.len()];
        SimplexMesh::<D, E>::new(verts, elems, etags, faces, ftags)
    }

    /// Get the location and metric for the reconstruction vertex
    fn point(&self) -> (Point<D>, M) {
        match self.ftype {
            FilledCavityType::ExistingVertex(i) => (
                self.cavity.points[i as usize],
                self.cavity.metrics[i as usize],
            ),
            FilledCavityType::MovedVertex((_, pt, m)) => (pt, m),
            FilledCavityType::EdgeCenter((_, pt, m)) => (pt, m),
        }
    }

    pub fn check(&self, l_min: f64, l_max: f64, q_min: f64) -> f64 {
        let (p0, m0) = self.point();
        let mut min_quality = 1.;
        for (f, _) in self.faces() {
            for i in f.iter() {
                let pi = &self.cavity.points[*i as usize];
                let mi = &self.cavity.metrics[*i as usize];
                let l = M::edge_length(&p0, &m0, pi, mi);
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
            let ge = E::Geom::from_vert_and_face(&p0, &m0, &gf);

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

    /// Check the the angle between the normal of the boundary faces and the normal given by the geometry is smaller than a threshold
    /// This is only required for swaps in 3D
    pub fn check_boundary_normals<G: Geometry<D>>(&self, geom: &G, threshold_degrees: f64) -> bool {
        let (p0, m0) = self.point();

        for (b, tag) in self.tagged_faces_boundary() {
            let gb = <<E::Face as Elem>::Geom<D, M> as GElem<D, M>>::Face::from_verts(
                b.iter().map(|&i| {
                    let (vx, _, m) = self.cavity.vert(i);
                    (*vx, *m)
                }),
            );
            let gf = <E::Face as Elem>::Geom::from_vert_and_face(&p0, &m0, &gb);
            let center = gf.center();
            let normal = gf.normal();
            let a = geom.angle(&center, &normal, &(E::DIM as Dim - 1, tag));
            if a > threshold_degrees {
                return false;
            }
        }
        true
    }
}
