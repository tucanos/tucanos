use crate::{
    cavity::{Cavity, CavityType, FilledCavity},
    geom_elems::{AsSliceF64, GElem},
    geometry::Geometry,
    max_iter,
    mesh::{Point, SimplexMesh},
    metric::Metric,
    min_iter, min_max_iter,
    stats::{CollapseStats, InitStats, SmoothStats, SplitStats, Stats, StepStats, SwapStats},
    topo_elems::{get_elem, get_face_to_elem, Elem},
    topology::Topology,
    Dim, Error, Idx, Mesh, Result, Tag, TopoTag,
};
use log::{debug, info, trace, warn};
#[cfg(feature = "nlopt")]
use nlopt::{Algorithm, Nlopt, Target};
use rustc_hash::FxHashMap;
use sorted_vec::SortedVec;
use std::fs::File;
use std::io::Write;
use std::time::Instant;
use std::{
    cmp::{min, Ordering},
    hash::BuildHasherDefault,
};

// /// Get edged indices such that they are sorted by increasing tag dimension and then by
// /// increasing edge length
#[must_use]
pub fn argsort_edges_increasing_length(f: &Vec<(Dim, f64)>) -> Vec<usize> {
    let mut indices = Vec::with_capacity(f.len());
    indices.extend(0..f.len());
    indices.sort_by(|i, j| match f[*j].0.cmp(&f[*i].0) {
        Ordering::Less => Ordering::Less,
        Ordering::Equal => f[*j].1.partial_cmp(&f[*i].1).unwrap(),
        Ordering::Greater => Ordering::Greater,
    });
    indices
}

#[must_use]
pub fn argsort_edges_decreasing_length(f: &Vec<(Dim, f64)>) -> Vec<usize> {
    let mut indices = Vec::with_capacity(f.len());
    indices.extend(0..f.len());
    indices.sort_by(|i, j| match f[*j].0.cmp(&f[*i].0) {
        Ordering::Less => Ordering::Less,
        Ordering::Equal => f[*i].1.partial_cmp(&f[*j].1).unwrap(),
        Ordering::Greater => Ordering::Greater,
    });
    indices
}

/// Vertex information
pub struct VtxInfo<const D: usize, M: Metric<D>> {
    /// Vertex coordinates
    pub vx: Point<D>,
    /// Tag
    pub tag: TopoTag,
    /// Metric
    pub m: M,
    /// Elements containing the vertex
    pub els: sorted_vec::SortedVec<Idx>,
}

/// Element information
#[derive(Clone, Copy)]
pub struct ElemInfo<E: Elem> {
    /// Element connectivity
    pub el: E,
    /// Quality
    pub q: f64,
}

/// Remesher for simplex meshes of elements E in dimension D
pub struct Remesher<const D: usize, E: Elem, M: Metric<D>, G: Geometry<D>> {
    /// The topology information
    topo: Topology,
    /// Vertices
    verts: FxHashMap<Idx, VtxInfo<D, M>>,
    /// Elements
    elems: FxHashMap<Idx, ElemInfo<E>>,
    /// Edges
    edges: FxHashMap<[Idx; 2], i16>,
    /// Next vertex Id
    next_vert: Idx,
    /// Next element Id
    next_elem: Idx,
    /// Geometry
    geom: G,
    /// Statistics
    stats: Vec<StepStats>,
}

enum TrySwapResult {
    QualitySufficient,
    CouldNotSwap,
    CouldSwap,
}

#[derive(Clone, Copy)]
pub enum SmoothingType {
    Laplacian,
    Avro,
    #[cfg(feature = "nlopt")]
    NLOpt,
    Laplacian2,
}

/// Remesher parameters
#[derive(Clone, Copy)]
pub struct RemesherParams {
    /// Number of collapse - split - swap - smooth loops
    pub num_iter: u32,
    /// Perform a first loop targetting only the longest edges
    pub two_steps: bool,
    /// Max. number of loops through the mesh edges during the split step
    pub split_max_iter: u32,
    /// Constraint the length of the newly created edges to be > split_constrain_l/sqrt(2) during split
    pub split_constrain_l: f64,
    /// Constraint the quality of the newly created elements to be > split_constrain_q during split
    pub split_constrain_q: f64,
    /// Max. number of loops through the mesh edges during the collapse step
    pub collapse_max_iter: u32,
    /// Constraint the quality of the newly created elements to be > split_constrain_q during collapse
    pub collapse_constrain_l: f64,
    /// Constraint the length of the newly created edges to be < split_constrain_l * sqrt(2) during collapse
    pub collapse_constrain_q: f64,
    /// Max. number of loops through the mesh edges during the swap step
    pub swap_max_iter: u32,
    /// Constraint the length of the newly created edges to be < split_constrain_l * sqrt(2) and > split_constrain_l/sqrt(2) during swap
    pub swap_constrain_l: f64,
    /// Number of smoothing steps
    pub smooth_iter: u32,
    /// Type of smoothing used
    pub smooth_type: SmoothingType,
    /// Max angle between the normals of the new faces and the geometry (in degrees)
    pub max_angle: f64,
}

impl Default for RemesherParams {
    fn default() -> Self {
        Self {
            num_iter: 2,
            two_steps: false,
            split_max_iter: 2,
            split_constrain_l: 1.0,
            split_constrain_q: 0.75,
            collapse_max_iter: 2,
            collapse_constrain_l: 1.0,
            collapse_constrain_q: 0.75,
            swap_max_iter: 2,
            swap_constrain_l: 0.5,
            smooth_iter: 1,
            smooth_type: SmoothingType::Laplacian,
            max_angle: 20.0,
        }
    }
}

impl<const D: usize, E: Elem, M: Metric<D>, G: Geometry<D>> Remesher<D, E, M, G> {
    /// Initialize the remesher
    pub fn new(mesh: &SimplexMesh<D, E>, m: &[M], geom: G) -> Result<Self> {
        info!(
            "Initialize the remesher with {} {}D vertices / {} {}",
            mesh.n_verts(),
            D,
            mesh.n_elems(),
            E::NAME
        );
        assert_eq!(m.len(), mesh.n_verts() as usize);

        // Compute the topology based on the element and face tags, and return vertex tags
        let (topo, vtag) = Topology::from_mesh(mesh);

        // Check that the geometry and topology are consistent
        geom.check(&topo)?;

        let mut res = Self {
            topo,
            verts: FxHashMap::with_capacity_and_hasher(
                mesh.n_verts() as usize,
                BuildHasherDefault::default(),
            ),
            elems: FxHashMap::with_capacity_and_hasher(
                mesh.n_elems() as usize,
                BuildHasherDefault::default(),
            ),
            edges: FxHashMap::default(),
            next_vert: 0,
            next_elem: 0,
            geom,
            stats: Vec::new(),
        };

        // Insert the vertices (project on the geometry for boundary vertices)
        assert_eq!(mesh.n_verts() as usize, vtag.len());
        let mut dmax = 0.0;
        for (i_vert, (p, tag)) in mesh.verts().zip(vtag.iter()).enumerate() {
            if tag.0 < E::DIM as Dim {
                let mut p_proj = p;
                let d = res.geom.project(&mut p_proj, tag);
                dmax = f64::max(dmax, d);
            }
            res.insert_vertex(p, tag, &m[i_vert]);
        }
        warn!("Max. distance to the geometry: {dmax}");

        assert_eq!(mesh.n_verts(), res.n_verts());

        // Insert the elements
        let n_elems = mesh.n_elems();
        for e in mesh.elems() {
            let tags = e.iter().map(|i| vtag[*i as usize]);
            let etag = res.topo.elem_tag(tags).unwrap();
            assert_eq!(
                etag.0,
                E::DIM as Dim,
                "Todo: fix the element with all the nodes of a surface"
            );
            res.insert_elem(e);
        }
        assert_eq!(n_elems, res.n_elems());

        res.print_stats();

        res.stats.push(StepStats::Init(InitStats::new(&res)));

        Ok(res)
    }

    /// Check that the remesher holds a valid mesh
    pub fn check(&self) -> Result<()> {
        info!("Check the mesh");

        for (i_elem, e) in &self.elems {
            let e = &e.el;
            // Is element-to-vertex and vertex-to-element info consistent?
            for i_vert in e.iter() {
                let res = self.verts.get(i_vert);
                if res.is_none() {
                    return Err(Error::from("Vertex not found"));
                }
                let v2e = &res.unwrap().els;
                if !v2e.iter().any(|x| x == i_elem) {
                    return Err(Error::from("Invalid vertex to element (missing vertex)"));
                }
            }
            // Is the element valid?
            let ge = self.gelem(e);
            if ge.vol() < 0. {
                return Err(Error::from("Negative volume"));
            }
            // Are the edges present?
            for i_edge in 0..E::N_EDGES {
                let mut edg = e.edge(i_edge);
                edg.sort_unstable();
                if self.edges.get(&edg).is_none() {
                    return Err(Error::from("Missing edge"));
                }
            }

            // check that the element tag dimension is E::DIM
            let vtags = e.iter().map(|i| self.verts.get(i).unwrap().tag);
            let etag = self.topo.elem_tag(vtags).unwrap();
            if etag.0 < E::DIM as Dim {
                return Err(Error::from("Invalid element tag"));
            }

            // check that all faces appear once if tagged on a boundary, or twice if tagged in the domain
            for i_face in 0..E::N_FACES {
                let f = e.face(i_face);
                let vtags = f.iter().map(|i| self.verts.get(i).unwrap().tag);
                let ftag = self.topo.elem_tag(vtags).unwrap();

                // filter the elements containing the face
                let mut els = self.verts.get(&f[0]).unwrap().els.iter().filter(|i| {
                    let other = self.elems.get(i).unwrap().el;
                    *i != i_elem && f.iter().all(|j| other.contains_vertex(*j))
                });
                if let Some(other) = els.next() {
                    // At least 3 elements
                    if els.next().is_some() {
                        return Err(Error::from("A face belongs to more than 2 elements"));
                    }
                    let other = self.elems.get(other).unwrap().el;
                    let vtags = other.iter().map(|i| self.verts.get(i).unwrap().tag);
                    let otag = self.topo.elem_tag(vtags).unwrap();
                    if etag.1 != otag.1 && ftag.0 != E::Face::DIM as Dim {
                        return Err(Error::from(
                            "A face belonging to 2 element with different tags is not tagged correctly",
                        ));
                    } else if etag.1 == otag.1 && ftag.0 != E::DIM as Dim {
                        return Err(Error::from(
                            "A face belonging to 2 element with the same tags is not tagged correctly",
                        ));
                    }
                } else if ftag.0 != E::Face::DIM as Dim {
                    return Err(Error::from(
                        "A face belonging to 1 element is not tagged correctly",
                    ));
                }
            }
        }

        for vert in self.verts.values() {
            // Do all element exist ?
            for i_elem in vert.els.iter() {
                if self.elems.get(i_elem).is_none() {
                    return Err(Error::from("Invalid vertex to element (missing element)"));
                }
            }
            // Is the metric valid?
            vert.m.check()?;
        }

        for edg in self.edges.keys() {
            for i in edg {
                let vert = self.verts.get(i);
                if vert.is_none() {
                    return Err(Error::from("Invalid edge (missing vertex)"));
                }
                let v2e = &vert.unwrap().els;
                for i_elem in v2e.iter() {
                    let e = &self.elems.get(i_elem).unwrap().el;
                    if !e.contains_edge(*edg) && vert.is_none() {
                        return Err(Error::from("Invalid edge"));
                    }
                }
            }
        }

        Ok(())
    }

    /// Create a `SimplexMesh`
    pub fn to_mesh(&self, only_bdy_faces: bool) -> SimplexMesh<D, E> {
        info!("Build a mesh");

        let vidx: FxHashMap<Idx, Idx> = self
            .verts
            .iter()
            .enumerate()
            .map(|(i, (k, _v))| (*k, i as Idx))
            .collect();

        let coords: Vec<f64> = self
            .verts
            .iter()
            .flat_map(|(_k, v)| v.vx.iter().copied())
            .collect();

        // Keep the remesher node ids for now to get the tags
        let elems: Vec<Idx> = self.elems.iter().flat_map(|(_k, e)| e.el).collect();

        let mut etags: Vec<Tag> = vec![0; self.n_elems() as usize];
        for (i_elem, val) in etags.iter_mut().enumerate() {
            let elem = get_elem::<E>(&elems, i_elem as Idx);
            let mut vtags = elem.iter().map(|i| self.verts.get(i).unwrap().tag);
            let etag = self.topo.elem_tag(&mut vtags).unwrap();
            if etag.0 < E::DIM as Dim {
                warn!("Element {:?} has tag {:?}", elem, etag);
                let node = self.topo.get(etag).unwrap();
                assert_eq!(node.parents.len(), 1);
                *val = *node.parents.iter().next().unwrap();
            } else {
                *val = etag.1;
            }
        }

        let f2e = get_face_to_elem::<E>(&elems);
        let mut faces = Vec::new();
        let mut ftags = Vec::new();

        for (face, iels) in &f2e {
            let mut vtags = face.iter().map(|i| self.verts.get(i).unwrap().tag);
            let ftag = self.topo.elem_tag(&mut vtags);
            if let Some(ftag) = ftag {
                if ftag.0 == E::Face::DIM as Dim {
                    if iels.len() == 1 {
                        // Orient the face outwards
                        let elem = get_elem::<E>(&elems, iels[0]);
                        let mut ok = false;
                        for i_face in 0..E::N_FACES {
                            let mut f = elem.face(i_face);
                            f.sort();
                            let is_same = !f.iter().zip(face.iter()).any(|(x, y)| x != y);
                            if is_same {
                                let f = elem.face(i_face);
                                for idx in f.iter() {
                                    faces.push(*vidx.get(idx).unwrap());
                                }
                                ftags.push(ftag.1);
                                ok = true;
                                break;
                            }
                        }
                        assert!(ok);
                    } else if !only_bdy_faces {
                        for idx in face.iter() {
                            faces.push(*vidx.get(idx).unwrap());
                        }
                        ftags.push(ftag.1);
                    }
                }
            }
        }

        SimplexMesh::<D, E>::new(
            coords,
            elems.iter().map(|x| *vidx.get(x).unwrap()).collect(),
            etags,
            faces,
            ftags,
        )
    }

    /// Insert a new vertex, and get its index
    pub fn insert_vertex(&mut self, pt: Point<D>, tag: &TopoTag, m: &M) -> Idx {
        self.verts.insert(
            self.next_vert,
            VtxInfo {
                vx: pt,
                tag: *tag,
                m: *m,
                els: SortedVec::default(),
            },
        );
        self.next_vert += 1;
        self.next_vert - 1
    }

    /// Remove a vertex
    pub fn remove_vertex(&mut self, idx: Idx) {
        let vx = self.verts.get(&idx);
        assert!(vx.is_some());
        assert!(vx.unwrap().els.is_empty());
        self.verts.remove(&idx);
    }

    /// Get the coordinates, tag and metric of a vertex
    pub fn get_vertex(&self, i: Idx) -> Option<(Point<D>, TopoTag, M)> {
        self.verts.get(&i).map(|v| (v.vx, v.tag, v.m))
    }

    /// Get the number of vertices
    pub fn n_verts(&self) -> Idx {
        self.verts.len() as Idx
    }

    /// Insert a new element
    pub fn insert_elem(&mut self, el: E) {
        // To remove after debug
        let mut vtags = el.iter().map(|i| self.verts.get(i).unwrap().tag);
        let etag = self.topo.elem_tag(&mut vtags).unwrap();
        assert!(etag.0 >= E::DIM as Dim, "Invalid tag, {etag:?}");
        let ge = self.gelem(&el);
        let q = ge.quality();
        assert!(q > 0.0);
        self.elems.insert(self.next_elem, ElemInfo { el, q });

        // update the vertex-to-element info
        for idx in el.iter() {
            let vx = self.verts.get_mut(idx);
            assert!(vx.is_some());
            vx.unwrap().els.push(self.next_elem);
        }

        // update the edges
        for i_edge in 0..E::N_EDGES {
            let mut edg = el.edge(i_edge);
            edg.sort_unstable();
            let e = self.edges.get_mut(&edg);
            if let Some(e) = e {
                *e += 1;
            } else {
                self.edges.insert(edg, 1);
            }
        }
        self.next_elem += 1;
    }

    /// Remove an element
    pub fn remove_elem(&mut self, idx: Idx) {
        let el = self.elems.get(&idx);
        assert!(el.is_some());
        let el = &el.unwrap().el;

        // update the vertex-to-element info
        for i in el.iter() {
            self.verts.get_mut(i).unwrap().els.remove_item(&idx);
        }

        // update the edges
        for i_edge in 0..E::N_EDGES {
            let mut edg = el.edge(i_edge);
            edg.sort_unstable();
            let e = self.edges.get_mut(&edg).unwrap();
            if *e == 1 {
                self.edges.remove(&edg);
            } else {
                *e -= 1;
            }
        }
        self.elems.remove(&idx);
    }

    /// Get the i-th element
    pub fn get_elem(&self, i: Idx) -> Option<ElemInfo<E>> {
        self.elems.get(&i).copied()
    }

    /// Get the number of elements
    pub fn n_elems(&self) -> Idx {
        self.elems.len() as Idx
    }

    /// Get the number of edges
    pub fn n_edges(&self) -> Idx {
        self.edges.len() as Idx
    }

    /// Get the geometrical element corresponding to elem
    fn gelem(&self, elem: &E) -> E::Geom<D, M> {
        E::Geom::from_verts(elem.iter().map(|j| {
            let pt = self.verts.get(j).unwrap();
            (pt.vx, pt.m)
        }))
    }

    /// Estimate the complexity
    pub fn complexity(&self) -> f64 {
        let mut c = 0.0;
        let weights = vec![1. / f64::from(E::N_VERTS); E::N_VERTS as usize];

        for e in self.elems.values() {
            let ge = self.gelem(&e.el);
            let vol = ge.vol();
            let metrics = e.el.iter().map(|i| &self.verts.get(i).unwrap().m);
            let wm = weights.iter().copied().zip(metrics);
            let m = M::interpolate(wm);
            c += vol / (E::Geom::<D, M>::IDEAL_VOL * m.vol());
        }
        c
    }

    /// Get a vertex cavity
    /// TODO: move as a method of Cavity
    fn compute_cavity_vertex(&self, i: Idx, c: &mut Cavity<D, E, M>) {
        let global_elems = &self.verts.get(&i).unwrap().els;
        c.compute(self, global_elems, CavityType::Vertex(i));
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

    /// Get an edge cavity
    /// TODO: move as a method of Cavity
    fn compute_cavity_edge(&self, edg: [Idx; 2], c: &mut Cavity<D, E, M>) {
        let c0 = self.verts.get(&edg[0]).unwrap().els.as_slice();
        let c1 = self.verts.get(&edg[1]).unwrap().els.as_slice();
        let global_elems = Self::intersection(c0, c1);
        debug_assert!(!global_elems.is_empty());
        c.compute(self, &global_elems, CavityType::Edge(edg));
    }

    /// Compute the length of an edge in metric space
    fn scaled_edge_length(&self, edg: [Idx; 2]) -> f64 {
        let p0 = self.verts.get(&edg[0]).unwrap();
        let p1 = self.verts.get(&edg[1]).unwrap();

        M::edge_length(&p0.vx, &p0.m, &p1.vx, &p1.m)
    }

    /// Compute the length of an edge in metric space and the dimension of its topo entity
    fn dim_and_scaled_edge_length(&self, edg: [Idx; 2]) -> (Dim, f64) {
        let p0 = self.verts.get(&edg[0]).unwrap();
        let p1 = self.verts.get(&edg[1]).unwrap();

        (
            self.topo.parent(p0.tag, p1.tag).unwrap().0,
            M::edge_length(&p0.vx, &p0.m, &p1.vx, &p1.m),
        )
    }

    /// Get the metric at every vertex
    pub fn metric(&self) -> Vec<f64> {
        self.verts.iter().flat_map(|(_k, v)| v.m).collect()
    }

    /// Get an iterator over edge length
    pub fn lengths_iter(&self) -> impl Iterator<Item = f64> + '_ {
        self.edges.keys().map(|k| self.scaled_edge_length(*k))
    }

    /// Get an iterator over edge topo dimension and length
    pub fn dims_and_lengths_iter(&self) -> impl Iterator<Item = (Dim, f64)> + '_ {
        self.edges
            .keys()
            .map(|k| self.dim_and_scaled_edge_length(*k))
    }

    /// Get the edge lengths
    pub fn lengths(&self) -> Vec<f64> {
        self.lengths_iter().collect()
    }

    /// Get an iterator over element qualities
    pub fn qualities_iter(&self) -> impl Iterator<Item = f64> + '_ {
        self.elems.values().map(|v| v.q)
    }

    /// Get the element qualities
    pub fn qualities(&self) -> Vec<f64> {
        self.qualities_iter().collect()
    }

    /// Loop over the edges and split them if
    ///   - their length is larger that `l_0`
    ///   - no edge smaller that `min(constrain_l` * `l_min`, `f64::sqrt(0.5`)) is created
    ///      (`l_min` is the minimum edge length in the mesh)
    ///   - no element with a quality < than `constrain_q` * `q_min` is created
    pub fn split(&mut self, l_0: f64, max_iter: u32, constrain_l: f64, constrain_q: f64) -> u32 {
        info!("Split edges with length > {:?}", l_0);

        let l_min = min_iter(self.lengths_iter());
        let q_min = min_iter(self.qualities_iter());

        let l_min = f64::min(constrain_l * l_min, f64::sqrt(0.5));

        let q_min = constrain_q * q_min;
        debug!("min. allowed quality = {:.2}", q_min);

        let mut n_iter = 0;
        let mut cavity = Cavity::new();
        loop {
            n_iter += 1;

            let mut edges = Vec::with_capacity(self.edges.len());
            edges.extend(self.edges.keys().copied());

            let mut dims_and_lengths = Vec::with_capacity(self.edges.len());
            dims_and_lengths.extend(self.dims_and_lengths_iter());

            // loop through the edges by increasing dimension and decreasing length
            let indices = argsort_edges_increasing_length(&dims_and_lengths);

            let mut n_splits = 0;
            let mut n_fails = 0;
            for i_edge in indices {
                let edg = edges[i_edge];
                let length = dims_and_lengths[i_edge].1;
                if length > f64::sqrt(2.0) {
                    trace!("Try to split edge {:?}, l = {}", edg, length);
                    self.compute_cavity_edge(edg, &mut cavity);
                    // TODO: move to Cavity?
                    let CavityType::Edge(local_edg) = cavity.ctype else { unreachable!() };
                    let i0 = local_edg[0] as usize;
                    let i1 = local_edg[1] as usize;
                    let p0 = &cavity.points[i0];
                    let p1 = &cavity.points[i1];
                    let m0 = &cavity.metrics[i0];
                    let m1 = &cavity.metrics[i1];
                    let t0 = cavity.tags[i0];
                    let t1 = cavity.tags[i1];

                    // edge center
                    let mut new_p = (p0 + p1) * 0.5;
                    // metric
                    let new_m = M::interpolate([(0.5, m0), (0.5, m1)].into_iter());
                    // tag
                    let tag = self.topo.parent(t0, t1).unwrap();
                    // projection if needed
                    if tag.0 < E::DIM as Dim {
                        self.geom.project(&mut new_p, &tag);
                    }

                    let filled_cavity =
                        FilledCavity::from_cavity_and_vertex(&cavity, &new_p, &new_m);

                    if filled_cavity.check(l_min, f64::MAX, q_min) > 0. {
                        trace!("Edge split");
                        for i in &cavity.global_elem_ids {
                            self.remove_elem(*i);
                        }
                        let ip = self.insert_vertex(new_p, &tag, &new_m);
                        for face in cavity.faces() {
                            let f = cavity.global_face(&face);
                            assert!(!f.contains_edge(edg));
                            let e = E::from_vertex_and_face(ip, &f);
                            self.insert_elem(e);
                        }
                        n_splits += 1;
                    } else {
                        n_fails += 1;
                    }
                }
            }

            debug!(
                "Iteration {}: {} edges split ({} failed)",
                n_iter, n_splits, n_fails
            );
            self.stats
                .push(StepStats::Split(SplitStats::new(n_splits, n_fails, self)));

            if n_splits == 0 || n_iter == max_iter {
                return n_iter;
            }
        }
    }

    /// Try to swap an edge if
    ///   - one of the elements in its cavity has a quality < qmin
    ///   - no edge smaller that `l_min` or longer that `l_max` is created
    /// TODO: move to Cavity?
    fn try_swap(
        &mut self,
        edg: [Idx; 2],
        q_min: f64,
        l_min: f64,
        l_max: f64,
        max_angle: f64,
        cavity: &mut Cavity<D, E, M>,
    ) -> TrySwapResult {
        trace!("Try to swap edge {:?}", edg);
        self.compute_cavity_edge(edg, cavity);

        if cavity.global_elem_ids.len() == 1 {
            trace!("Cannot swap, only one adjacent cell");
            return TrySwapResult::QualitySufficient;
        }

        if cavity.q_min > q_min {
            trace!("No need to swap, quality sufficient");
            return TrySwapResult::QualitySufficient;
        }

        let CavityType::Edge(local_edg) = cavity.ctype else { unreachable!() };
        let i0 = local_edg[0] as usize;
        let i1 = local_edg[1] as usize;
        let mut q_ref = cavity.q_min;

        let mut vx = 0;
        let mut succeed = false;

        let etag = self.topo.parent(cavity.tags[i0], cavity.tags[i1]).unwrap();

        for n in 0..cavity.n_verts() {
            if n == i0 as Idx || n == i1 as Idx {
                continue;
            }

            // check topo
            let ptag = self.topo.parent(etag, cavity.tags[n as usize]);
            if ptag.is_none() {
                trace!("Cannot swap, incompatible geometry");
                continue;
            }
            let ptag = ptag.unwrap();
            if ptag.0 != etag.0 || ptag.1 != etag.1 {
                trace!("Cannot swap, incompatible geometry");
                continue;
            }

            let filled_cavity = FilledCavity::from_cavity_and_vertex_id(cavity, n);

            if !filled_cavity.check_tags(&self.topo) {
                trace!("Cannot swap, would create an element/face with an invalid tag");
                continue;
            }

            if !filled_cavity.check_boundary_normals(&self.topo, &self.geom, max_angle) {
                trace!("Cannot swap, would create a non smooth surface");
                continue;
            }

            let min_quality = filled_cavity.check(l_min, l_max, q_ref);
            if min_quality > q_ref {
                trace!("Can swap  from {} : ({} > {})", n, min_quality, q_ref);
                succeed = true;
                q_ref = min_quality;
                vx = n;
            }
        }

        if succeed {
            trace!("Swap from {}", vx);
            for e in &cavity.global_elem_ids {
                self.remove_elem(*e);
            }
            let filled_cavity = FilledCavity::from_cavity_and_vertex_id(cavity, vx);
            for f in filled_cavity.faces() {
                let global_vx = cavity.local2global[vx as usize];
                let f = cavity.global_face(&f);
                assert!(!f.contains_vertex(global_vx));
                assert!(!f.contains_edge(edg));
                let e = E::from_vertex_and_face(global_vx, &f);
                self.insert_elem(e);
            }

            return TrySwapResult::CouldSwap;
        }

        TrySwapResult::CouldNotSwap
    }

    /// Loop over the edges and perform edge swaps if
    ///   - the quality of an adjacent element is < `q_target`
    ///   - no edge smaller that `constrain_l` * `l_min` or larger than lmax / `constrain_l` is created
    ///      (`l_min` / `l_max` are the minimum : maximum edge length in the mesh)
    ///   - the edge swap increases the minimum quality of the adjacent elements
    pub fn swap(&mut self, q_target: f64, max_iter: u32, constrain_l: f64, max_angle: f64) -> u32 {
        info!("Swap edges: target quality = {}", q_target);

        let (lmin, lmax) = min_max_iter(self.lengths_iter());

        let lmin = f64::min(lmin * constrain_l, f64::sqrt(0.5));
        let lmax = f64::max(lmax / constrain_l, f64::sqrt(2.0));
        debug!("min. / max. allowed edge length = {:.2}, {:.2}", lmin, lmax);

        let mut n_iter = 0;
        let mut cavity = Cavity::new();
        loop {
            n_iter += 1;
            let mut edges = Vec::with_capacity(self.edges.len());
            edges.extend(self.edges.keys().copied());

            let mut n_swaps = 0;
            let mut n_fails = 0;
            let mut n_ok = 0;
            for edg in edges {
                let res = self.try_swap(edg, q_target, lmin, lmax, max_angle, &mut cavity);
                match res {
                    TrySwapResult::CouldNotSwap => n_fails += 1,
                    TrySwapResult::CouldSwap => n_swaps += 1,
                    TrySwapResult::QualitySufficient => n_ok += 1,
                }
            }

            debug!(
                "Iteration {}: {} edges swapped ({} failed, {} OK)",
                n_iter, n_swaps, n_fails, n_ok
            );
            self.stats
                .push(StepStats::Swap(SwapStats::new(n_swaps, n_fails, self)));
            if n_swaps == 0 || n_iter == max_iter {
                return n_iter;
            }
        }
    }

    /// Loop over the edges and collapse them if
    ///   - their length is smaller that 1/sqrt(2)
    ///   - no edge larger than sqrt(2.0) / `constrain_l` is created
    ///   - no element with a quality < than `constrain_q` * `q_min` is created
    pub fn collapse(
        &mut self,
        max_iter: u32,
        constrain_l: f64,
        constrain_q: f64,
        max_angle: f64,
    ) -> u32 {
        info!("Collapse elements");

        let l_max = f64::sqrt(2.0) / constrain_l; //crate::max_iter(self.lengths_iter()) / constrain_l;
        debug!("max. allowed length: {:.2}", l_max);
        let q_min = constrain_q * min_iter(self.qualities_iter());
        debug!("min. allowed quality: {:.2}", q_min);

        let mut n_iter = 0;
        let mut cavity = Cavity::new();
        loop {
            n_iter += 1;

            let mut edges = Vec::with_capacity(self.edges.len());
            edges.extend(self.edges.keys().copied());

            let mut dims_and_lengths = Vec::with_capacity(self.edges.len());
            dims_and_lengths.extend(self.dims_and_lengths_iter());

            let indices = argsort_edges_decreasing_length(&dims_and_lengths);

            let mut n_collapses = 0;
            let mut n_fails = 0;
            for i_edge in indices {
                let edg = edges[i_edge];
                let l = dims_and_lengths[i_edge].1;
                if l < f64::sqrt(0.5) {
                    trace!("Try to collapse edgs {:?}", edg);
                    let mut i0 = edg[0];
                    let mut i1 = edg[1];
                    if self.verts.get(&i0).is_none() {
                        trace!("Cannot collapse: vertex deleted");
                        continue;
                    }
                    if self.verts.get(&i1).is_none() {
                        trace!("Cannot collapse: vertex deleted");
                        continue;
                    }

                    let mut topo_0 = self.verts.get(&i0).unwrap().tag;
                    let mut topo_1 = self.verts.get(&i1).unwrap().tag;
                    let tag = self.topo.parent(topo_0, topo_1).unwrap();

                    if topo_0.0 != tag.0 || topo_0.1 != tag.1 {
                        if topo_1.0 == tag.0 || !topo_1.1 == tag.1 {
                            trace!("Swap vertices");
                            std::mem::swap(&mut i1, &mut i0);
                            std::mem::swap(&mut topo_1, &mut topo_0);
                        } else {
                            trace!("Cannot collapse, incompatible geometry");
                            continue;
                        }
                    }

                    self.compute_cavity_vertex(i0, &mut cavity);

                    let local_i1 = cavity
                        .local2global
                        .iter()
                        .copied()
                        .enumerate()
                        .find(|(_x, y)| *y == i1)
                        .unwrap()
                        .0;

                    let filled_cavity =
                        FilledCavity::from_cavity_and_vertex_id(&cavity, local_i1 as Idx);

                    if !filled_cavity.check_tags(&self.topo) {
                        trace!("Cannot collapse, would create an element/face with an invalid tag");
                        continue;
                    }

                    if !filled_cavity.check_boundary_normals(&self.topo, &self.geom, max_angle) {
                        trace!("Cannot collapse, would create a non smooth surface");
                        continue;
                    }

                    // proposition 1?
                    if filled_cavity.check(0.0, l_max, q_min) > 0.0 {
                        trace!("Collapse edge");
                        for i in &cavity.global_elem_ids {
                            self.remove_elem(*i);
                        }

                        self.remove_vertex(i0);

                        for f in filled_cavity.faces() {
                            let f = cavity.global_face(&f);
                            assert!(!f.contains_vertex(i1));
                            let e = E::from_vertex_and_face(i1, &f);
                            self.insert_elem(e);
                        }
                        n_collapses += 1;
                    } else {
                        n_fails += 1;
                    }
                }
            }

            debug!(
                "Iteration {}: {} edges collapsed ({} failed)",
                n_iter, n_collapses, n_fails
            );
            self.stats.push(StepStats::Collapse(CollapseStats::new(
                n_collapses,
                n_fails,
                self,
            )));
            if n_collapses == 0 || n_iter == max_iter {
                return n_iter;
            }
        }
    }

    /// Get the vertices in a vertex cavity usable for smoothing, i.e. with tag that is a children of the cavity vertes
    /// TODO: move to Cavity
    fn get_smoothing_neighbors(&self, cavity: &Cavity<D, E, M>) -> (bool, Vec<Idx>) {
        let mut res = Vec::<Idx>::with_capacity(cavity.n_verts() as usize);
        let CavityType::Vertex(i0) = cavity.ctype else { unreachable!() };

        let m0 = &cavity.metrics[i0 as usize];
        let t0 = cavity.tags[i0 as usize];

        let mut local_minimum = true;
        for i1 in 0..cavity.n_verts() {
            if i1 == i0 {
                continue;
            }
            if res.iter().any(|x| *x == i1) {
                continue;
            }
            // check tag
            let t1 = cavity.tags[i1 as usize];
            let tag = self.topo.parent(t0, t1);
            if tag.is_none() {
                continue;
            }
            let tag = tag.unwrap();
            if t0.0 != tag.0 || t0.1 != tag.1 {
                continue;
            }

            res.push(i1);

            let m1 = &cavity.metrics[i1 as usize];
            if m1.vol() < 1.01 * m0.vol() {
                local_minimum = false;
            }
        }

        (local_minimum, res)
    }

    fn smooth_laplacian(cavity: &Cavity<D, E, M>, neighbors: &[Idx]) -> Point<D> {
        let CavityType::Vertex(i0) = cavity.ctype else { unreachable!() };
        let (p0, _, _) = cavity.vert(i0);
        let mut p0_new = Point::<D>::zeros();
        for i1 in neighbors {
            let p1 = &cavity.points[*i1 as usize];
            let e = p0 - p1;
            p0_new -= e;
        }
        p0_new /= neighbors.len() as f64;
        p0_new += p0;
        p0_new
    }

    fn smooth_laplacian_2(cavity: &Cavity<D, E, M>, neighbors: &[Idx]) -> Point<D> {
        let CavityType::Vertex(i0) = cavity.ctype else { unreachable!() };
        let (p0, _, m0) = cavity.vert(i0);
        let mut p0_new = Point::<D>::zeros();
        let mut w = 0.0;
        for i1 in neighbors {
            let p1 = &cavity.points[*i1 as usize];
            let e = p0 - p1;
            let l = m0.length(&e);
            p0_new += l * (p0 + p1);
            w += l;
        }
        p0_new /= 2.0 * w;
        p0_new
    }

    fn smooth_avro(cavity: &Cavity<D, E, M>, neighbors: &[Idx]) -> Point<D> {
        let CavityType::Vertex(i0) = cavity.ctype else { unreachable!() };
        let (p0, _, m0) = cavity.vert(i0);
        let mut p0_new = Point::<D>::zeros();
        for i1 in neighbors {
            let p1 = &cavity.points[*i1 as usize];
            let e = p0 - p1;
            let omega = 0.2;
            let l = m0.length(&e);
            let l4 = l.powi(4);
            let fac = omega * (1.0 - l4) * f64::exp(-l4) / l;
            p0_new += fac * e;
        }
        p0_new += p0;
        p0_new
    }

    #[cfg(feature = "nlopt")]
    fn smooth_nlopt(cavity: &Cavity<D, E, M>, neighbors: &[Idx]) -> Point<D> {
        let CavityType::Vertex(i0) = cavity.ctype else { unreachable!() };
        let (_, t0, m0) = cavity.vert(i0);
        if t0.0 == E::DIM as Dim {
            let mut p0_new = Point::<D>::zeros();
            let mut qmax = cavity.q_min;
            let gfaces: Vec<_> = cavity.faces().map(|f| cavity.gface(&f)).collect();

            for i_elem in 0..cavity.n_elems() {
                let ge = cavity.gelem(i_elem);

                let n = E::N_VERTS as usize;
                let mut x = vec![0.0; n];

                let func = |x: &[f64], _grad: Option<&mut [f64]>, _params: &mut ()| -> f64 {
                    let p = ge.point(x);
                    let mut q_avg = 0.0;
                    for i_face in 0..cavity.n_faces() {
                        let gf = &gfaces[i_face as usize];
                        let ge1 = E::Geom::from_vert_and_face(&p, m0, gf);
                        q_avg += ge1.quality();
                    }
                    q_avg / cavity.n_faces() as f64
                };

                let mut opt = Nlopt::new(Algorithm::Cobyla, n - 1, func, Target::Maximize, ());
                assert!(opt.set_xtol_rel(1.0e-2).is_ok());
                assert!(opt.set_maxeval(10).is_ok());

                let lb = vec![0.0; n]; // lower bounds
                assert!(opt.set_lower_bounds(&lb).is_ok());
                let ub = vec![1.0; n]; // upper bounds
                assert!(opt.set_upper_bounds(&ub).is_ok());

                let constraint = |x: &[f64], _grad: Option<&mut [f64]>, _param: &mut ()| -> f64 {
                    x.iter().sum::<f64>() - 1.0
                };
                assert!(opt.add_inequality_constraint(constraint, (), 1e-8).is_ok());

                let res = opt.optimize(&mut x);
                trace!("NLOpt: {:?}", res);
                if res.unwrap().1 > qmax {
                    qmax = res.unwrap().1;
                    let mut sum = 0.0;
                    for i in (1..E::N_VERTS as usize).rev() {
                        x[i] = x[i - 1];
                        sum += x[i];
                    }
                    x[0] = 1.0 - sum;
                    p0_new = ge.point(&x);
                    trace!("bcoords = {:?}", x);
                    trace!("p0_new = {:?}", p0_new);
                }
            }
            p0_new
        } else {
            Self::smooth_laplacian(cavity, neighbors)
        }
    }

    /// Perform mesh smoothing
    pub fn smooth(&mut self, stype: SmoothingType, n_iter: u32) {
        info!("Smooth vertices");

        // We modify the vertices while iterating over them so we must copy
        // the keys. Apart from going unsafe the only way to avoid this would be
        // to have one RefCell for each VtxInfo but copying self.verts is cheaper.
        let verts = self.verts.keys().copied().collect::<Vec<_>>();

        let mut cavity = Cavity::new();
        for iter in 0..n_iter {
            let mut n_fails = 0;
            let mut n_min = 0;
            let mut n_smooth = 0;
            for i0 in verts.iter().copied() {
                trace!("Try to smooth vertex {}", i0);

                self.compute_cavity_vertex(i0, &mut cavity);

                let (is_local_minimum, neighbors) = self.get_smoothing_neighbors(&cavity);

                if is_local_minimum {
                    trace!("Won't smooth, local minimum of m");
                    n_min += 1;
                    continue;
                }

                if neighbors.is_empty() {
                    trace!("Cannot smooth, no suitable neighbor");
                    continue;
                }

                let CavityType::Vertex(i0_local) = cavity.ctype else { unreachable!() };
                let p0 = &cavity.points[i0_local as usize];
                let m0 = &cavity.metrics[i0_local as usize];
                let t0 = &cavity.tags[i0_local as usize];

                let mut h0_new = Default::default();
                let mut p0_new = match stype {
                    SmoothingType::Laplacian => Self::smooth_laplacian(&cavity, &neighbors),
                    SmoothingType::Laplacian2 => Self::smooth_laplacian_2(&cavity, &neighbors),
                    SmoothingType::Avro => Self::smooth_avro(&cavity, &neighbors),
                    #[cfg(feature = "nlopt")]
                    SmoothingType::NLOpt => Self::smooth_nlopt(&cavity, &neighbors),
                };

                if t0.0 < E::DIM as Dim {
                    self.geom.project(&mut p0_new, t0);
                }

                trace!(
                    "Smooth, vertex moved by {} -> {:?}",
                    (p0 - p0_new).norm(),
                    p0_new
                );

                let mut q_min_new = f64::INFINITY;
                for f in cavity.faces() {
                    let gf = cavity.gface(&f);
                    let ge1 = E::Geom::from_vert_and_face(&p0_new, m0, &gf);
                    q_min_new = q_min_new.min(ge1.quality());
                }

                if q_min_new < 0.999 * cavity.q_min {
                    n_fails += 1;
                    trace!(
                        "Smooth, quality would decrease: {} < {} ",
                        q_min_new,
                        cavity.q_min
                    );
                    continue;
                }

                // Smoothing is valid, interpolate the metric at the new vertex location
                let mut best = f64::NEG_INFINITY;
                for i_elem in 0..cavity.n_elems() {
                    let ge = cavity.gelem(i_elem);
                    let x = ge.bcoords(&p0_new);
                    let cmin = min_iter(x.as_slice_f64().iter().copied());
                    if cmin > best {
                        let elem = &cavity.elems[i_elem as usize];
                        let metrics = elem.iter().map(|i| &cavity.metrics[*i as usize]);
                        let wm = x.as_slice_f64().iter().copied().zip(metrics);
                        h0_new = M::interpolate(wm);
                        best = cmin;
                        if best > 0.0 {
                            break;
                        }
                    }
                }

                trace!("Smooth, update vertex");
                {
                    let vert = self.verts.get_mut(&i0).unwrap();
                    vert.vx = p0_new;
                    assert!(h0_new.vol() > 0.0);
                    vert.m = h0_new;
                }

                for (i_local, i_global) in cavity.global_elem_ids.iter().enumerate() {
                    // update the quality
                    let ge = cavity.gelem(i_local as Idx); // todo: precompute all ge
                    self.elems.get_mut(i_global).unwrap().q = ge.quality();
                }
                n_smooth += 1;
            }
            debug!(
                "Iteration {}: {} vertices moved, {} fails, {} local minima",
                iter, n_smooth, n_fails, n_min
            );
            self.stats
                .push(StepStats::Smooth(SmoothStats::new(n_fails, self)));
        }
    }

    /// Perform a remeshing iteration
    pub fn remesh(&mut self, params: RemesherParams) {
        info!("Adapt the mesh");
        let now = Instant::now();

        if params.two_steps {
            let l_max = max_iter(self.lengths_iter());
            let l_0 = 0.5 * l_max;
            if l_0 > f64::sqrt(2.0) {
                for _ in 0..params.num_iter {
                    self.collapse(
                        params.collapse_max_iter,
                        params.collapse_constrain_l,
                        params.collapse_constrain_q,
                        params.max_angle,
                    );
                    self.split(
                        l_0,
                        params.split_max_iter,
                        params.split_constrain_l,
                        params.split_constrain_q,
                    );
                    self.swap(
                        0.4,
                        params.swap_max_iter,
                        params.swap_constrain_l,
                        params.max_angle,
                    );
                    self.swap(
                        0.8,
                        params.swap_max_iter,
                        params.swap_constrain_l,
                        params.max_angle,
                    );
                    self.smooth(params.smooth_type, params.smooth_iter);
                }
            }
        }

        for _ in 0..params.num_iter {
            self.collapse(
                params.collapse_max_iter,
                params.collapse_constrain_l,
                params.collapse_constrain_q,
                params.max_angle,
            );
            self.split(
                f64::sqrt(2.0),
                params.split_max_iter,
                params.split_constrain_l,
                params.split_constrain_q,
            );
            self.swap(
                0.4,
                params.swap_max_iter,
                params.swap_constrain_l,
                params.max_angle,
            );
            self.swap(
                0.8,
                params.swap_max_iter,
                params.swap_constrain_l,
                params.max_angle,
            );
            self.smooth(params.smooth_type, params.smooth_iter);
        }

        self.swap(
            0.4,
            params.swap_max_iter,
            params.swap_constrain_l,
            params.max_angle,
        );
        self.swap(
            0.8,
            params.swap_max_iter,
            params.swap_constrain_l,
            params.max_angle,
        );

        info!("Done in {}s", now.elapsed().as_secs_f32());
        self.print_stats();
    }

    /// Print length and quality stats on the mesh / metric
    pub fn print_stats(&self) {
        let stats = Stats::new(self.lengths_iter(), &[f64::sqrt(0.5), f64::sqrt(2.0)]);
        info!("Length: {}", stats);

        let stats = Stats::new(self.qualities_iter(), &[0.4, 0.6, 0.8]);
        info!("Qualities: {}", stats);
    }

    /// Return the stats at each remeshing step as a json string
    pub fn stats_json(&self) -> String {
        serde_json::to_string_pretty(&self.stats).unwrap()
    }

    /// Save the stats at each remeshing step as a json file
    pub fn save_stats(&self, fname: &str) -> Result<()> {
        let mut file = File::create(fname)?;
        writeln!(
            file,
            "{}",
            serde_json::to_string_pretty(&self.stats).unwrap()
        )?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::PI;

    use env_logger::Env;

    use crate::{
        geometry::NoGeometry,
        mesh::{Point, SimplexMesh},
        metric::{AnisoMetric2d, AnisoMetric3d, IsoMetric, Metric},
        remesher::{Remesher, SmoothingType},
        test_meshes::{
            h_2d, h_3d, test_mesh_2d, test_mesh_3d, test_mesh_3d_single_tet, test_mesh_3d_two_tets,
            test_mesh_moon_2d, GeomHalfCircle2d,
        },
        topo_elems::Triangle,
        Idx, Mesh, Result, Tag,
    };

    use super::RemesherParams;

    fn _init_log(level: &str) {
        env_logger::Builder::from_env(Env::default().default_filter_or(level))
            .format_timestamp(None)
            .init();
    }

    #[test]
    fn test_init() -> Result<()> {
        let mesh = test_mesh_2d();
        let h = vec![IsoMetric::<2>::from(1.); mesh.n_verts() as usize];
        let remesher = Remesher::new(&mesh, &h, NoGeometry())?;

        assert_eq!(remesher.n_verts(), 4);
        assert_eq!(remesher.n_elems(), 2);
        assert_eq!(remesher.n_edges(), 5);

        let mesh = remesher.to_mesh(false);
        assert_eq!(mesh.n_verts(), 4);
        assert_eq!(mesh.n_elems(), 2);
        assert_eq!(mesh.n_faces(), 5);

        let mesh = remesher.to_mesh(true);
        assert_eq!(mesh.n_verts(), 4);
        assert_eq!(mesh.n_elems(), 2);
        assert_eq!(mesh.n_faces(), 4);

        for (c, tag) in mesh.elem_centers().zip(mesh.etags()) {
            let mut p = Point::<2>::zeros();
            match tag {
                1 => p.copy_from_slice(&[2. / 3., 1. / 3.]),
                2 => p.copy_from_slice(&[1. / 3., 2. / 3.]),
                _ => p.copy_from_slice(&[-1., -1.]),
            }
            let d = (c - p).norm();
            assert!(d < 1e-8);
        }

        for (c, tag) in mesh.face_centers().zip(mesh.ftags()) {
            let mut p = Point::<2>::zeros();
            match tag {
                1 => p.copy_from_slice(&[0.5, 0.]),
                2 => p.copy_from_slice(&[1., 0.5]),
                3 => p.copy_from_slice(&[0.5, 1.]),
                4 => p.copy_from_slice(&[0., 0.5]),
                _ => p.copy_from_slice(&[-1., -1.]),
            }
            let d = (c - p).norm();
            assert!(d < 1e-8);
        }

        Ok(())
    }

    #[test]
    #[should_panic]
    fn test_remove_vertex_error() {
        let mesh = test_mesh_2d();
        let h = vec![IsoMetric::<2>::from(1.); mesh.n_verts() as usize];

        let mut remesher = Remesher::new(&mesh, &h, NoGeometry()).unwrap();

        remesher.remove_vertex(5);
    }

    #[test]
    #[should_panic]
    fn test_remove_vertex_error_2() {
        let mesh = test_mesh_2d();
        let h = vec![IsoMetric::<2>::from(1.); mesh.n_verts() as usize];

        let mut remesher = Remesher::new(&mesh, &h, NoGeometry()).unwrap();

        remesher.remove_vertex(1);
    }

    #[test]
    fn test_remove_elem() -> Result<()> {
        let mesh = test_mesh_2d();
        let h = vec![IsoMetric::<2>::from(1.); mesh.n_verts() as usize];

        let mut remesher = Remesher::new(&mesh, &h, NoGeometry())?;

        remesher.remove_elem(0);
        remesher.remove_vertex(1);

        assert_eq!(remesher.n_verts(), 3);
        assert_eq!(remesher.n_elems(), 1);
        assert_eq!(remesher.n_edges(), 3);

        Ok(())
    }

    #[test]
    #[should_panic]
    fn test_remove_elem_2() {
        let mesh = test_mesh_2d();
        let h = vec![IsoMetric::<2>::from(1.); mesh.n_verts() as usize];

        let mut remesher = Remesher::new(&mesh, &h, NoGeometry()).unwrap();

        remesher.remove_elem(3);
    }

    #[test]
    fn test_split_2d() -> Result<()> {
        let mesh = test_mesh_2d().split().split();

        let h: Vec<_> = mesh
            .verts()
            .map(|p| IsoMetric::<2>::from(h_2d(&p)))
            .collect();
        let mut remesher = Remesher::new(&mesh, &h, NoGeometry())?;

        let n_iter = remesher.split(f64::sqrt(2.0), 10, 0.5, 0.5);
        assert!(n_iter < 10);

        remesher.check()?;

        let _mesh = remesher.to_mesh(true);

        Ok(())
    }

    #[test]
    fn test_swap_2d() -> Result<()> {
        let mesh = test_mesh_2d().split().split();

        // collapse to lower the quality
        let h = vec![IsoMetric::<2>::from(2.); mesh.n_verts() as usize];
        let mut remesher = Remesher::new(&mesh, &h, NoGeometry())?;
        remesher.check()?;

        let n_iter = remesher.collapse(10, 0.5, 0.75, 20.0);
        assert!(n_iter < 10);
        let mesh = remesher.to_mesh(true);
        assert!(f64::abs(mesh.vol() - 1.) < 1e-12);
        remesher.check()?;

        // swap
        let mesh = remesher.to_mesh(true);
        let h = vec![IsoMetric::<2>::from(2.); mesh.n_verts() as usize];
        let mut remesher = Remesher::new(&mesh, &h, NoGeometry())?;

        let n_iter = remesher.swap(0.8, 10, 0.5, 20.0);
        assert!(n_iter < 10);
        let mesh = remesher.to_mesh(true);
        assert!(f64::abs(mesh.vol() - 1.) < 1e-12);

        remesher.check()?;

        let _mesh = remesher.to_mesh(true);

        Ok(())
    }

    #[test]
    fn test_collapse_2d() -> Result<()> {
        let mesh = test_mesh_2d().split().split();

        let h = vec![IsoMetric::<2>::from(2.); mesh.n_verts() as usize];
        let mut remesher = Remesher::new(&mesh, &h, NoGeometry())?;
        remesher.check()?;

        let n_iter = remesher.collapse(10, 0.5, 0.75, 20.0);
        assert!(n_iter < 10);

        remesher.check()?;

        let _mesh = remesher.to_mesh(true);
        assert!(f64::abs(mesh.vol() - 1.) < 1e-12);

        Ok(())
    }

    #[test]
    fn test_smooth_laplacian_2d() -> Result<()> {
        let coords: Vec<f64> = vec![0., 0., 1., 0., 1., 1., 0., 1., 0.1, 0.1];
        let elems: Vec<Idx> = vec![0, 1, 4, 1, 2, 4, 2, 3, 4, 3, 0, 4];
        let etags: Vec<Tag> = vec![1, 1, 1, 1];
        let faces: Vec<Idx> = vec![0, 1, 1, 2, 2, 3, 3, 0];
        let ftags: Vec<Tag> = vec![1, 2, 3, 4];

        let mesh = SimplexMesh::<2, Triangle>::new(coords, elems, etags, faces, ftags);

        let h = vec![IsoMetric::<2>::from(1.); mesh.n_verts() as usize];
        let mut remesher = Remesher::new(&mesh, &h, NoGeometry())?;

        remesher.smooth(SmoothingType::Laplacian, 1);
        let pt = remesher.verts.get(&4).unwrap().vx;
        let center = Point::<2>::new(0.5, 0.5);
        assert!((pt - center).norm() < 1e-12);

        remesher.check()?;

        let _mesh = remesher.to_mesh(true);

        Ok(())
    }

    #[test]
    #[cfg(feature = "nlopt")]
    fn test_smooth_nlopt_2d() -> Result<()> {
        let coords: Vec<f64> = vec![0., 0., 1., 0., 1., 1., 0., 1., 0.1, 0.1];
        let elems: Vec<Idx> = vec![0, 1, 4, 1, 2, 4, 2, 3, 4, 3, 0, 4];
        let etags: Vec<Tag> = vec![1, 1, 1, 1];
        let faces: Vec<Idx> = vec![0, 1, 1, 2, 2, 3, 3, 0];
        let ftags: Vec<Tag> = vec![1, 2, 3, 4];

        let mesh = SimplexMesh::<2, Triangle>::new(coords, elems, etags, faces, ftags);

        let h = vec![IsoMetric::<2>::from(1.); mesh.n_verts() as usize];
        let mut remesher = Remesher::new(&mesh, &h, NoGeometry())?;

        remesher.smooth(SmoothingType::NLOpt, 1);
        let pt = remesher.verts.get(&4).unwrap().vx;
        let center = Point::<2>::new(0.5, 0.5);
        assert!((pt - center).norm() < 0.05);

        remesher.check()?;

        let _mesh = remesher.to_mesh(true);

        Ok(())
    }

    #[test]
    fn test_smooth_laplacian_2d_aniso() -> Result<()> {
        let coords: Vec<f64> = vec![0., 0., 1., 0., 1., 0.1, 0., 0.1, 0.01, 0.01];
        let elems: Vec<Idx> = vec![0, 1, 4, 1, 2, 4, 2, 3, 4, 3, 0, 4];
        let etags: Vec<Tag> = vec![1, 1, 1, 1];
        let faces: Vec<Idx> = vec![0, 1, 1, 2, 2, 3, 3, 0];
        let ftags: Vec<Tag> = vec![1, 2, 3, 4];

        let mesh = SimplexMesh::<2, Triangle>::new(coords, elems, etags, faces, ftags);

        let h = vec![
            AnisoMetric2d::from_slice(&[1., 100., 0.], 0),
            AnisoMetric2d::from_slice(&[1., 100., 0.], 0),
            AnisoMetric2d::from_slice(&[1., 100., 0.], 0),
            AnisoMetric2d::from_slice(&[1., 100., 0.], 0),
            AnisoMetric2d::from_slice(&[1., 100., 0.], 0),
        ];
        let mut remesher = Remesher::new(&mesh, &h, NoGeometry())?;

        remesher.smooth(SmoothingType::Laplacian, 1);
        let pt = remesher.verts.get(&4).unwrap().vx;
        let center = Point::<2>::new(0.5, 0.05);
        assert!((pt - center).norm() < 1e-12);

        remesher.check()?;

        let _mesh = remesher.to_mesh(true);

        Ok(())
    }

    #[test]
    fn test_smooth_omega_2d() -> Result<()> {
        let coords: Vec<f64> = vec![0., 0., 1., 0., 1., 1., 0., 1., 0.1, 0.1];
        let elems: Vec<Idx> = vec![0, 1, 4, 1, 2, 4, 2, 3, 4, 3, 0, 4];
        let etags: Vec<Tag> = vec![1, 1, 1, 1];
        let faces: Vec<Idx> = vec![0, 1, 1, 2, 2, 3, 3, 0];
        let ftags: Vec<Tag> = vec![1, 2, 3, 4];

        let mesh = SimplexMesh::<2, Triangle>::new(coords, elems, etags, faces, ftags);

        let h = vec![IsoMetric::<2>::from(1.); mesh.n_verts() as usize];
        let mut remesher = Remesher::new(&mesh, &h, NoGeometry())?;

        remesher.smooth(SmoothingType::Avro, 10);
        let pt = remesher.verts.get(&4).unwrap().vx;
        let center = Point::<2>::new(0.5, 0.5);
        assert!((pt - center).norm() < 0.01);

        remesher.check()?;

        let _mesh = remesher.to_mesh(true);

        Ok(())
    }

    #[test]
    fn test_smooth_omega_2d_aniso() -> Result<()> {
        let coords: Vec<f64> = vec![0., 0., 1., 0., 1., 0.1, 0., 0.1, 0.01, 0.01];
        let elems: Vec<Idx> = vec![0, 1, 4, 1, 2, 4, 2, 3, 4, 3, 0, 4];
        let etags: Vec<Tag> = vec![1, 1, 1, 1];
        let faces: Vec<Idx> = vec![0, 1, 1, 2, 2, 3, 3, 0];
        let ftags: Vec<Tag> = vec![1, 2, 3, 4];

        let mesh = SimplexMesh::<2, Triangle>::new(coords, elems, etags, faces, ftags);

        let h = vec![
            AnisoMetric2d::from_slice(&[1., 100., 0.], 0),
            AnisoMetric2d::from_slice(&[1., 100., 0.], 0),
            AnisoMetric2d::from_slice(&[1., 100., 0.], 0),
            AnisoMetric2d::from_slice(&[1., 100., 0.], 0),
            AnisoMetric2d::from_slice(&[1., 100., 0.], 0),
        ];
        let mut remesher = Remesher::new(&mesh, &h, NoGeometry())?;

        remesher.smooth(SmoothingType::Avro, 10);
        let pt = remesher.verts.get(&4).unwrap().vx;
        let center = Point::<2>::new(0.5, 0.05);
        assert!((pt - center).norm() < 0.01);

        remesher.check()?;

        let _mesh = remesher.to_mesh(true);

        Ok(())
    }

    #[test]
    fn test_adapt_2d() -> Result<()> {
        let mesh = test_mesh_2d();

        let mut mesh = mesh.split().split();

        for _iter in 0..2 {
            let h: Vec<_> = (0..mesh.n_verts())
                .map(|i| IsoMetric::<2>::from(h_2d(&mesh.vert(i))))
                .collect();
            let mut remesher = Remesher::new(&mesh, &h, NoGeometry())?;

            remesher.remesh(RemesherParams::default());
            remesher.check()?;

            mesh = remesher.to_mesh(true);
        }

        Ok(())
    }

    #[test]
    fn test_adapt_aniso_2d() -> Result<()> {
        let mut mesh = test_mesh_2d();
        mesh.etags[1] = 1;

        let mfunc = |pt: Point<2>| {
            let y = pt[1];
            let v0 = Point::<2>::new(0.5, 0.);
            let v1 = Point::<2>::new(0., 0.05 + 0.15 * y);
            AnisoMetric2d::from_sizes(&v0, &v1)
        };

        let h: Vec<_> = mesh.verts().map(mfunc).collect();
        let mut remesher = Remesher::new(&mesh, &h, NoGeometry())?;

        remesher.remesh(RemesherParams::default());
        remesher.check()?;

        let mesh = remesher.to_mesh(true);
        assert!(f64::abs(mesh.vol() - 1.0) < 1e-12);

        Ok(())
    }

    #[test]
    fn test_adapt_2d_geom() -> Result<()> {
        let mut mesh = test_mesh_moon_2d();

        let ref_vol = 0.5 * PI - 2.0 * (0.5 * 1.25 * 1.25 * f64::atan2(1., 0.75) - 0.5 * 0.75);

        for _ in 1..4 {
            let h: Vec<_> = mesh
                .verts()
                .map(|p| IsoMetric::<2>::from(h_2d(&p)))
                .collect();
            let mut remesher = Remesher::new(&mesh, &h, GeomHalfCircle2d())?;

            remesher.remesh(RemesherParams::default());
            remesher.check()?;

            mesh = remesher.to_mesh(true);

            let vol = mesh.vol();
            assert!(f64::abs(vol - ref_vol) < 0.05);
        }

        Ok(())
    }

    #[test]
    fn test_split_tet_3d() -> Result<()> {
        let mesh = test_mesh_3d_single_tet();

        let h = vec![IsoMetric::<3>::from(0.25); mesh.n_verts() as usize];
        let mut remesher = Remesher::new(&mesh, &h, NoGeometry())?;
        remesher.check()?;

        let n_iter = remesher.split(f64::sqrt(2.0), 10, 0.5, 0.5);
        assert!(n_iter < 10);
        remesher.check()?;

        let _mesh = remesher.to_mesh(true);

        Ok(())
    }

    #[test]
    fn test_split_3d() -> Result<()> {
        let mesh = test_mesh_3d().split().split();

        let h: Vec<_> = mesh
            .verts()
            .map(|p| IsoMetric::<3>::from(h_3d(&p)))
            .collect();
        let mut remesher = Remesher::new(&mesh, &h, NoGeometry())?;
        remesher.check()?;

        let n_iter = remesher.split(f64::sqrt(2.0), 10, 0.5, 0.5);
        assert!(n_iter < 10);

        remesher.check()?;

        let mesh = remesher.to_mesh(true);
        assert!(f64::abs(mesh.vol() - 1.) < 1e-12);

        Ok(())
    }

    #[test]
    fn test_collapse_3d() -> Result<()> {
        let mesh = test_mesh_3d().split().split();

        let h = vec![IsoMetric::<3>::from(2.); mesh.n_verts() as usize];
        let mut remesher = Remesher::new(&mesh, &h, NoGeometry())?;
        remesher.check()?;

        let n_iter = remesher.collapse(10, 0.5, 0.75, 20.);
        assert!(n_iter < 10);

        remesher.check()?;

        let mesh = remesher.to_mesh(true);
        assert!(f64::abs(mesh.vol() - 1.) < 1e-12);

        Ok(())
    }

    #[test]
    fn test_swap_twotets_3d() -> Result<()> {
        let mesh = test_mesh_3d_two_tets();

        // collapse to lower the quality
        let h = vec![IsoMetric::<3>::from(2.); mesh.n_verts() as usize];
        let mut remesher = Remesher::new(&mesh, &h, NoGeometry())?;

        remesher.check()?;
        let mesh = remesher.to_mesh(true);
        assert!(f64::abs(mesh.vol() - 2.0 * 0.1 / 6.) < 1e-12);

        // swap
        let n_iter = remesher.swap(0.8, 10, 0.5, 20.0);
        assert!(n_iter < 10);
        let mesh = remesher.to_mesh(true);
        assert!(f64::abs(mesh.vol() - 2.0 * 0.1 / 6.) < 1e-12);

        remesher.check()?;

        Ok(())
    }

    #[test]
    fn test_swap_3d() -> Result<()> {
        let mesh = test_mesh_3d().split().split();

        // collapse to lower the quality
        let h = vec![IsoMetric::<3>::from(2.); mesh.n_verts() as usize];
        let mut remesher = Remesher::new(&mesh, &h, NoGeometry())?;
        remesher.check()?;

        let n_iter = remesher.collapse(10, 0.5, 0.75, 20.);
        assert!(n_iter < 10);
        let mesh = remesher.to_mesh(true);
        assert!(f64::abs(mesh.vol() - 1.) < 1e-12);
        remesher.check()?;

        // swap
        let mesh = remesher.to_mesh(true);
        let h = vec![IsoMetric::<3>::from(2.); mesh.n_verts() as usize];
        let mut remesher = Remesher::new(&mesh, &h, NoGeometry())?;

        let n_iter = remesher.swap(0.8, 10, 0.5, 20.0);
        assert!(n_iter < 10);
        let mesh = remesher.to_mesh(true);
        assert!(f64::abs(mesh.vol() - 1.) < 1e-12);

        remesher.check()?;

        let _mesh = remesher.to_mesh(true);

        Ok(())
    }

    #[test]
    fn test_adapt_3d() -> Result<()> {
        let mesh = test_mesh_3d();

        let mesh = mesh.split().split();

        let h: Vec<_> = mesh
            .verts()
            .map(|p| IsoMetric::<3>::from(h_3d(&p)))
            .collect();
        let mut remesher = Remesher::new(&mesh, &h, NoGeometry())?;

        remesher.remesh(RemesherParams::default());

        remesher.check()?;

        let mesh = remesher.to_mesh(true);
        assert!(f64::abs(mesh.vol() - 1.) < 1e-12);

        Ok(())
    }

    #[test]
    fn test_adapt_aniso_3d() -> Result<()> {
        let mesh = test_mesh_3d();

        let mfunc = |_p| {
            let v0 = Point::<3>::new(0.5, 0., 0.);
            let v1 = Point::<3>::new(0.0, 0.5, 0.);
            let v2 = Point::<3>::new(0., 0.0, 0.2);
            AnisoMetric3d::from_sizes(&v0, &v1, &v2)
        };

        let h: Vec<_> = mesh.verts().map(mfunc).collect();
        let mut remesher = Remesher::new(&mesh, &h, NoGeometry())?;

        remesher.remesh(RemesherParams::default());
        remesher.check()?;

        let mesh = remesher.to_mesh(true);
        assert!(f64::abs(mesh.vol() - 1.0) < 1e-12);

        Ok(())
    }

    #[test]
    fn test_complexity_2d() -> Result<()> {
        let mut mesh = test_mesh_2d().split().split();
        mesh.compute_volumes();

        let mfunc = |_p| {
            let v0 = Point::<2>::new(0.1, 0.);
            let v1 = Point::<2>::new(0.0, 0.01);
            AnisoMetric2d::from_sizes(&v0, &v1)
        };
        let c_ref = 4. / f64::sqrt(3.0) / (0.1 * 0.01);

        let h: Vec<_> = mesh.verts().map(mfunc).collect();
        let c = mesh.complexity(h.iter().copied(), 0.0, f64::MAX);
        assert!(f64::abs(c - c_ref) < 1e-6 * c);

        let remesher = Remesher::new(&mesh, &h, NoGeometry())?;

        let c = remesher.complexity();
        assert!(f64::abs(c - c_ref) < 1e-6 * c);

        Ok(())
    }

    #[test]
    fn test_complexity_3d() -> Result<()> {
        let mut mesh = test_mesh_3d().split().split();
        mesh.compute_volumes();

        let mfunc = |_p| {
            let v0 = Point::<3>::new(0.1, 0., 0.);
            let v1 = Point::<3>::new(0.0, 0.01, 0.);
            let v2 = Point::<3>::new(0., 0.0, 2.);
            AnisoMetric3d::from_sizes(&v0, &v1, &v2)
        };
        let c_ref = 6. * f64::sqrt(2.0) / (0.1 * 0.01 * 2.0);

        let h: Vec<_> = mesh.verts().map(mfunc).collect();
        let c = mesh.complexity(h.iter().copied(), 0.0, f64::MAX);
        assert!(f64::abs(c - c_ref) < 1e-6 * c);

        let remesher = Remesher::new(&mesh, &h, NoGeometry())?;

        let c = remesher.complexity();
        assert!(f64::abs(c - c_ref) < 1e-6 * c);

        remesher.check()?;

        let _mesh = remesher.to_mesh(true);

        Ok(())
    }
}
