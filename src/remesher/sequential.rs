use super::cavity::{Cavity, CavityCheckStatus, FilledCavity, FilledCavityType, Seed};
use super::stats::{
    CollapseStats, InitStats, SmoothStats, SplitStats, Stats, StepStats, SwapStats,
};
use crate::{
    geometry::Geometry,
    max_iter,
    mesh::{get_face_to_elem, AsSliceF64, Elem, GElem, Point, SimplexMesh, Topology},
    metric::Metric,
    min_iter, Dim, Error, Idx, Result, Tag, TopoTag,
};
use log::{debug, trace};
#[cfg(feature = "nlopt")]
use nlopt::{Algorithm, Nlopt, Target};
use rustc_hash::FxHashMap;
use sorted_vec::SortedVec;
use std::{cmp::Ordering, fs::File, io::Write, time::Instant};

// /// Get edged indices such that they are sorted by increasing tag dimension and then by
// /// increasing edge length
#[must_use]
pub fn argsort_edges_increasing_length(f: &[(Dim, f64)]) -> Vec<usize> {
    let mut indices = Vec::with_capacity(f.len());
    indices.extend(0..f.len());
    indices.sort_by(|i, j| match f[*i].0.cmp(&f[*j].0) {
        Ordering::Less => Ordering::Less,
        Ordering::Equal => f[*i].1.partial_cmp(&f[*j].1).unwrap(),
        Ordering::Greater => Ordering::Greater,
    });
    indices
}

#[must_use]
pub fn argsort_edges_decreasing_length(f: &[(Dim, f64)]) -> Vec<usize> {
    let mut indices = Vec::with_capacity(f.len());
    indices.extend(0..f.len());
    indices.sort_by(|i, j| match f[*i].0.cmp(&f[*j].0) {
        Ordering::Less => Ordering::Less,
        Ordering::Equal => f[*j].1.partial_cmp(&f[*i].1).unwrap(),
        Ordering::Greater => Ordering::Greater,
    });
    indices
}

/// Vertex information
#[derive(Debug)]
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
    /// Tag
    pub tag: Tag,
    /// Quality
    pub q: f64,
}

/// Remesher for simplex meshes of elements E in dimension D
pub struct Remesher<const D: usize, E: Elem, M: Metric<D>> {
    /// The topology information
    topo: Topology,
    /// Vertices
    verts: FxHashMap<Idx, VtxInfo<D, M>>,
    /// Elements
    elems: FxHashMap<Idx, ElemInfo<E>>,
    /// Edges
    edges: FxHashMap<[Idx; 2], i16>,
    /// Tagged faces
    tagged_faces: FxHashMap<E::Face, Tag>,
    /// Next vertex Id
    next_vert: Idx,
    /// Next element Id
    next_elem: Idx,
    /// Statistics
    stats: Vec<StepStats>,
}

enum TrySwapResult {
    QualitySufficient,
    FixedEdge,
    CouldNotSwap,
    CouldSwap,
}

/// Smoothing methods
/// For all methods except NLOpt, a set of valid neighbors $`N(i)`$ is built as a subset
/// if the neighbors of vertex $`i`$ that are tagged on the same entity of one of its children
/// The new vertex location $`\tilde v_i`$ is then computed as (where $`||v_j - v_i||_M`$ is the
/// edge length in metric space):
///  - for `Laplacian`
/// ```math
/// \tilde v_i = v_i + \sum_{j \in N(i)} (v_j - v_i)
/// ```
///  - for `Laplacian2`
/// ```math
/// \tilde v_i = \frac{\sum_{j \in N(i)} ||v_j - v_i||_M (v_j + v_i)}{2 \sum_{j \in N(i)} ||v_j - v_i||_M}
/// ```
///  - for `Avro`
/// ```math
/// \tilde v_i = v_i + \omega \sum_{j \in N(i)} (1 − ||v_j - v_i||_M^4) \exp(−||v_j - v_i||_M^4)(v_j - v_i)
/// ```
/// - another:
/// ```math
/// \tilde v_i = (1 - \omega) v_i + \omega \frac{\sum_{j \in N(i)} ||v_j - v_i||_M v_j}{\sum_{j \in N(i)} ||v_j - v_i||_M}
/// ```
/// with $`\omega = {1, 1/2, 1/4, ...}`$
#[derive(Clone, Copy)]
pub enum SmoothingType {
    Laplacian,
    Avro,
    #[cfg(feature = "nlopt")]
    NLOpt,
    Laplacian2,
}

/// Remesher parameters
#[derive(Clone)]
pub struct RemesherParams {
    /// Number of collapse - split - swap - smooth loops
    pub num_iter: u32,
    /// Perform a first loop targetting only the longest edges
    pub two_steps: bool,
    /// Max. number of loops through the mesh edges during the split step
    pub split_max_iter: u32,
    /// Constraint the length of the newly created edges to be > split_min_l_rel * min(l) during split
    pub split_min_l_rel: f64,
    /// Constraint the length of the newly created edges to be > split_min_l_abs during split
    pub split_min_l_abs: f64,
    /// Constraint the quality of the newly created elements to be > split_min_q_rel * min(q) during split
    pub split_min_q_rel: f64,
    /// Constraint the quality of the newly created elements to be > split_min_q_abs during split
    pub split_min_q_abs: f64,
    /// Max. number of loops through the mesh edges during the collapse step
    pub collapse_max_iter: u32,
    /// Constraint the length of the newly created edges to be < collapse_max_l_rel * max(l) during collapse
    pub collapse_max_l_rel: f64,
    /// Constraint the length of the newly created edges to be < collapse_max_l_abs during collapse
    pub collapse_max_l_abs: f64,
    /// Constraint the quality of the newly created elements to be > collapse_min_q_rel * min(q) during collapse
    pub collapse_min_q_rel: f64,
    /// Constraint the quality of the newly created elements to be > collapse_min_q_abs during collapse
    pub collapse_min_q_abs: f64,
    /// Max. number of loops through the mesh edges during the swap step
    pub swap_max_iter: u32,
    /// Constraint the length of the newly created edges to be < swap_max_l_rel * max(l) during swap
    pub swap_max_l_rel: f64,
    /// Constraint the length of the newly created edges to be < swap_max_l_abs during swap
    pub swap_max_l_abs: f64,
    /// Constraint the length of the newly created edges to be > swap_min_l_rel * min(l) during swap
    pub swap_min_l_rel: f64,
    /// Constraint the length of the newly created edges to be > swap_min_l_abs during swap
    pub swap_min_l_abs: f64,
    /// Number of smoothing steps
    pub smooth_iter: u32,
    /// Type of smoothing used
    pub smooth_type: SmoothingType,
    /// Smoothing relaxation
    pub smooth_relax: Vec<f64>,
    /// Don't smooth vertices that are a local metric minimum
    pub smooth_keep_local_minima: bool,
    /// Max angle between the normals of the new faces and the geometry (in degrees)
    pub max_angle: f64,
    /// Debug mode
    pub debug: bool,
}

impl Default for RemesherParams {
    fn default() -> Self {
        Self {
            num_iter: 4,
            two_steps: true,
            split_max_iter: 1,
            split_min_l_rel: 1.0,
            split_min_l_abs: 0.75 / f64::sqrt(2.0),
            split_min_q_rel: 1.0,
            split_min_q_abs: 0.5,
            collapse_max_iter: 1,
            collapse_max_l_rel: 1.0,
            collapse_max_l_abs: 1.5 * f64::sqrt(2.0),
            collapse_min_q_rel: 1.0,
            collapse_min_q_abs: 0.5,
            swap_max_iter: 2,
            swap_max_l_rel: 1.5,
            swap_max_l_abs: 1.5 * f64::sqrt(2.0),
            swap_min_l_rel: 0.75,
            swap_min_l_abs: 0.75 / f64::sqrt(2.0),
            smooth_iter: 2,
            smooth_type: SmoothingType::Laplacian,
            smooth_relax: vec![0.5, 0.25, 0.125],
            smooth_keep_local_minima: false,
            max_angle: 20.0,
            debug: false,
        }
    }
}

impl<const D: usize, E: Elem, M: Metric<D>> Remesher<D, E, M> {
    /// Initialize the remesher
    pub fn new<G: Geometry<D>>(mesh: &SimplexMesh<D, E>, m: &[M], geom: &G) -> Result<Self> {
        Self::new_with_iter(mesh, m.iter().copied(), geom)
    }

    pub fn new_with_iter<G: Geometry<D>, IT>(
        mesh: &SimplexMesh<D, E>,
        metric: IT,
        geom: &G,
    ) -> Result<Self>
    where
        IT: Iterator<Item = M> + ExactSizeIterator,
    {
        debug!(
            "Initialize the remesher with {} {D}D vertices / {} {}",
            mesh.n_verts(),
            mesh.n_elems(),
            E::NAME
        );
        assert_eq!(metric.len(), mesh.n_verts() as usize);

        // Get the topology
        let topo = mesh.get_topology()?;
        let vtag = mesh.get_vertex_tags()?;

        // Check that the geometry and topology are consistent
        geom.check(topo)?;
        let (mut verts, mut elems) = (FxHashMap::default(), FxHashMap::default());
        verts.reserve(mesh.n_verts() as usize);
        elems.reserve(mesh.n_elems() as usize);
        let mut res = Self {
            topo: topo.clone(),
            verts,
            elems,
            tagged_faces: FxHashMap::default(),
            edges: FxHashMap::default(),
            next_vert: 0,
            next_elem: 0,
            stats: Vec::new(),
        };

        // Insert the vertices
        assert_eq!(mesh.n_verts() as usize, vtag.len());
        for ((p, tag), m) in mesh.verts().zip(vtag.iter()).zip(metric) {
            res.insert_vertex(p, tag, m);
        }

        assert_eq!(mesh.n_verts(), res.n_verts());

        // Insert the elements
        let n_elems = mesh.n_elems();
        for (e, t) in mesh.elems().zip(mesh.etags()) {
            res.insert_elem(e, t)?;
        }
        assert_eq!(n_elems, res.n_elems());

        // Insert the tagged faces
        mesh.faces()
            .zip(mesh.ftags())
            .for_each(|(f, t)| res.add_tagged_face(f, t).unwrap());

        res.print_stats();
        res.stats.push(StepStats::Init(InitStats::new(&res)));
        Ok(res)
    }

    fn check_vert_to_elems(&self, i_elem: Idx, e: &E) -> Result<()> {
        for i_vert in e.iter() {
            let res = self.verts.get(i_vert);
            if res.is_none() {
                return Err(Error::from("Vertex not found"));
            }
            let v2e = &res.unwrap().els;
            if !v2e.iter().any(|&x| x == i_elem) {
                return Err(Error::from("Invalid vertex to element (missing vertex)"));
            }
        }
        Ok(())
    }

    fn check_elem_volume(&self, e: &E) -> Result<()> {
        let ge = self.gelem(e);
        if ge.vol() < 0. {
            return Err(Error::from("Negative volume"));
        }
        Ok(())
    }

    fn check_edges(&self, e: &E) -> Result<()> {
        for i_edge in 0..E::N_EDGES {
            let mut edg = e.edge(i_edge);
            edg.sort_unstable();
            if !self.edges.contains_key(&edg) {
                return Err(Error::from("Missing edge"));
            }
        }
        Ok(())
    }

    fn check_faces(&self, i_elem: Idx, e: &ElemInfo<E>) -> Result<()> {
        let etag = e.tag;

        for i_face in 0..E::N_FACES {
            let f = e.el.face(i_face);

            // filter the other elements containing the face
            let iels = self
                .verts
                .get(&f[0])
                .unwrap()
                .els
                .iter()
                .filter(|i| {
                    let other = self.elems.get(i).unwrap().el;
                    **i != i_elem && f.iter().all(|j| other.contains_vertex(*j))
                })
                .collect::<Vec<_>>();

            let ftag = self.face_tag(&f);

            if !iels.is_empty() {
                // At least 3 elements
                if iels.len() > 1 {
                    return Err(Error::from("A face belongs to more than 2 elements"));
                }
                let other = self.elems.get(iels[0]).unwrap();
                let otag = other.tag;
                if etag != otag && ftag.is_none() {
                    return Err(Error::from(&format!(
                        "A face ({f:?}:{ftag:?}) belonging to 2 elements ({:?}:{etag:?}, {:?}:{otag:?}) with different tags is not tagged correctly. ", e.el, other.el
                    )));
                } else if etag == otag && ftag.is_some() {
                    return Err(Error::from(&format!(
                        "A face (({f:?}:{ftag:?})) belonging to 2 elements ({:?}:{etag:?}, {:?}:{otag:?}) with the same tags is not tagged \
                        correctly.", e.el, other.el
                    )));
                }
            } else if ftag.is_none() {
                return Err(Error::from(
                    &format!("A face (({f:?}:{ftag:?})) belonging to 1 element ({:?}:{etag:?}) is not tagged correctly", e.el
                )));
            }
        }
        Ok(())
    }

    /// Check that the remesher holds a valid mesh
    pub fn check(&self) -> Result<()> {
        debug!("Check the consistency of the remesher data");

        for (&i_elem, e) in &self.elems {
            // Is element-to-vertex and vertex-to-element info consistent?
            self.check_vert_to_elems(i_elem, &e.el)?;

            // Is the element valid?
            self.check_elem_volume(&e.el)?;

            // Are the edges present?
            self.check_edges(&e.el)?;

            // check that all faces appear once if tagged on a boundary, or twice if tagged in the domain
            self.check_faces(i_elem, e)?;
        }

        for vert in self.verts.values() {
            // Do all element exist ?
            for i_elem in vert.els.iter() {
                if !self.elems.contains_key(i_elem) {
                    return Err(Error::from("Invalid vertex to element (missing element)"));
                }
            }
            // Is the metric valid?
            vert.m.check()?;
        }

        for edg in self.edges.keys() {
            // Check that the edge vertices are ok
            for i in edg {
                // Does the vertex exist ?
                let vert = self.verts.get(i);
                if vert.is_none() {
                    return Err(Error::from("Invalid edge (missing vertex)"));
                }

                // Do all the elements contain the edge ?
                let v2e = &vert.unwrap().els;
                for i_elem in v2e.iter() {
                    let e = &self.elems.get(i_elem).unwrap().el;
                    if !e.contains_edge(*edg) && vert.is_none() {
                        return Err(Error::from("Invalid edge"));
                    }
                }
            }
        }

        for (face, tag) in &self.tagged_faces {
            if face.iter().any(|i| !self.verts.contains_key(i)) {
                return Err(Error::from(&format!(
                    "Invalid tagged face {face:?} - tag={tag}"
                )));
            }
        }

        Ok(())
    }

    /// Create a `SimplexMesh`
    #[must_use]
    pub fn to_mesh(&self, only_bdy_faces: bool) -> SimplexMesh<D, E> {
        debug!("Build a mesh");

        let vidx: FxHashMap<Idx, Idx> = self
            .verts
            .iter()
            .enumerate()
            .map(|(i, (k, _v))| (*k, i as Idx))
            .collect();

        let verts = self.verts.values().map(|v| v.vx).collect();

        let mut elems = Vec::with_capacity(self.n_elems() as usize);
        let mut etags = Vec::with_capacity(self.n_elems() as usize);
        for e in self.elems.values() {
            elems.push(E::from_iter(e.el.iter().map(|&i| *vidx.get(&i).unwrap())));
            etags.push(e.tag);
        }

        let f2e = get_face_to_elem(elems.iter().copied());
        let mut faces = Vec::new();
        let mut ftags = Vec::new();

        for (face, &ftag) in &self.tagged_faces {
            let face = E::Face::from_iter(face.iter().map(|&i| *vidx.get(&i).unwrap()));
            let face = face.sorted();
            let iels = f2e.get(&face).unwrap();
            if iels.len() == 1 {
                // Orient the face outwards
                let elem = elems[iels[0] as usize];
                let mut ok = false;
                for i_face in 0..E::N_FACES {
                    let mut f = elem.face(i_face);
                    f.sort();
                    let is_same = !f.iter().zip(face.iter()).any(|(x, y)| x != y);
                    if is_same {
                        let f = elem.face(i_face);
                        faces.push(f);
                        ftags.push(ftag);
                        ok = true;
                        break;
                    }
                }
                assert!(ok);
            } else if !only_bdy_faces {
                faces.push(face);
                ftags.push(ftag);
            }
        }

        SimplexMesh::<D, E>::new(verts, elems, etags, faces, ftags)
    }

    /// Insert a new vertex, and get its index
    pub fn insert_vertex(&mut self, pt: Point<D>, tag: &TopoTag, m: M) -> Idx {
        self.verts.insert(
            self.next_vert,
            VtxInfo {
                vx: pt,
                tag: *tag,
                m,
                els: SortedVec::default(),
            },
        );
        self.next_vert += 1;
        self.next_vert - 1
    }

    /// Remove a vertex
    pub fn remove_vertex(&mut self, idx: Idx) -> Result<()> {
        let vx = self.verts.get(&idx);
        if vx.is_none() {
            return Err(Error::from("Vertex not present"));
        };
        if !vx.unwrap().els.is_empty() {
            return Err(Error::from("Vertex used"));
        }
        self.verts.remove(&idx);
        Ok(())
    }

    /// Get the coordinates, tag and metric of a vertex
    #[must_use]
    pub fn get_vertex(&self, i: Idx) -> Option<(Point<D>, TopoTag, M)> {
        self.verts.get(&i).map(|v| (v.vx, v.tag, v.m))
    }

    /// Return indices of elements around a `i` vertex
    #[must_use]
    pub fn vertex_elements(&self, i: Idx) -> &[Idx] {
        &self.verts.get(&i).unwrap().els
    }

    /// Get the number of vertices
    #[must_use]
    pub fn n_verts(&self) -> Idx {
        self.verts.len() as Idx
    }

    /// Get the number of elements that contain an edge
    #[must_use]
    pub fn elem_count(&self, edg: [Idx; 2]) -> i16 {
        self.edges.get(&edg).copied().unwrap_or(0)
    }

    /// Insert a new element
    pub fn insert_elem(&mut self, el: E, tag: Tag) -> Result<()> {
        let ge = self.gelem(&el);
        let q = ge.quality();
        if q <= 0.0 {
            return Err(Error::from("Invalid element"));
        }
        self.elems.insert(self.next_elem, ElemInfo { el, tag, q });

        // update the vertex-to-element info
        for idx in el.iter() {
            let vx = self.verts.get_mut(idx);
            if vx.is_none() {
                return Err(Error::from("Element vertex not present"));
            }
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
        Ok(())
    }

    /// Remove an element
    pub fn remove_elem(&mut self, idx: Idx) -> Result<()> {
        let el = self.elems.get(&idx);
        if el.is_none() {
            return Err(Error::from("Element not present"));
        }
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
        Ok(())
    }

    /// Get the i-th element
    #[must_use]
    pub fn get_elem(&self, i: Idx) -> Option<ElemInfo<E>> {
        self.elems.get(&i).copied()
    }

    /// Get the number of elements
    #[must_use]
    pub fn n_elems(&self) -> Idx {
        self.elems.len() as Idx
    }

    /// Get the number of edges
    #[must_use]
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

    /// Return the tag of a face
    pub fn face_tag(&self, face: &E::Face) -> Option<Tag> {
        let face = face.sorted();
        self.tagged_faces.get(&face).copied()
    }

    /// Return the tag of a face
    fn add_tagged_face(&mut self, face: E::Face, tag: Tag) -> Result<()> {
        let face = face.sorted();
        if self.tagged_faces.contains_key(&face) {
            return Err(Error::from("Tagged face already present"));
        }
        if face.iter().any(|i| !self.verts.contains_key(i)) {
            return Err(Error::from("At least a vertex is not in the mesh"));
        }
        self.tagged_faces.insert(face, tag);
        Ok(())
    }

    /// Return the tag of a face
    fn remove_tagged_face(&mut self, face: E::Face) -> Result<()> {
        let face = face.sorted();
        if !self.tagged_faces.contains_key(&face) {
            return Err(Error::from("Tagged face not present"));
        }
        self.tagged_faces.remove(&face).unwrap();
        Ok(())
    }

    /// Estimate the complexity
    #[must_use]
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
    #[must_use]
    pub fn metric(&self) -> Vec<f64> {
        self.verts.iter().flat_map(|(_k, v)| v.m).collect()
    }

    /// Get an iterator over edge length
    pub fn lengths_iter(&self) -> impl Iterator<Item = f64> + '_ {
        self.edges.keys().map(|k| self.scaled_edge_length(*k))
    }

    /// Get an iterator over edge topo dimension and length
    fn dims_and_lengths_iter(&self) -> impl Iterator<Item = (Dim, f64)> + '_ {
        self.edges
            .keys()
            .map(|k| self.dim_and_scaled_edge_length(*k))
    }

    /// Get the edge lengths
    #[must_use]
    pub fn lengths(&self) -> Vec<f64> {
        self.lengths_iter().collect()
    }

    /// Get an iterator over element qualities
    pub fn qualities_iter(&self) -> impl Iterator<Item = f64> + '_ {
        self.elems.values().map(|v| v.q)
    }

    /// Get the metrics
    #[must_use]
    pub fn metrics(&self) -> Vec<M> {
        self.verts.values().map(|v| v.m).collect()
    }

    /// Get the element qualities
    #[must_use]
    pub fn qualities(&self) -> Vec<f64> {
        self.qualities_iter().collect()
    }

    /// Loop over the edges and split them if
    ///   - their length is larger that `l_0`
    ///   - no edge smaller than
    ///       min(1/sqrt(2), max(params.split_min_l_abs, params.collapse_min_l_rel * min(l)))
    ///   - no element with a quality lower than
    ///       max(params.collapse_min_q_abs, params.collapse_min_q_rel * min(q)))
    ///
    /// where min(l) and min(q) as the max edge length and min quality over the entire mesh
    pub fn split<G: Geometry<D>>(
        &mut self,
        l_0: f64,
        params: &RemesherParams,
        geom: &G,
    ) -> Result<u32> {
        debug!("Split edges with length > {:.2e}", l_0);

        let l_min = params.split_min_l_abs;
        debug!("min. allowed length: {:.2}", l_min);
        let q_min = params.split_min_q_abs;
        debug!("min. allowed quality: {:.2}", q_min);

        let mut n_iter = 0;
        let mut cavity = Cavity::new();
        loop {
            n_iter += 1;

            let mut edges = Vec::with_capacity(self.edges.len());
            edges.extend(self.edges.keys().copied());

            let mut dims_and_lengths = Vec::with_capacity(self.edges.len());
            dims_and_lengths.extend(self.dims_and_lengths_iter());

            // loop through the edges by increasing dimension and decreasing length
            let indices = argsort_edges_decreasing_length(&dims_and_lengths);

            let mut n_splits = 0;
            let mut n_fails = 0;
            for i_edge in indices {
                let edg = edges[i_edge];
                let length = dims_and_lengths[i_edge].1;
                if length > l_0 {
                    trace!("Try to split edge {:?}, l = {}", edg, length);
                    cavity.init_from_edge(edg, self);
                    // TODO: move to Cavity?
                    let Seed::Edge(local_edg) = cavity.seed else {
                        unreachable!()
                    };
                    let (mut edge_center, new_metric) = cavity.seed_barycenter();
                    let tag = self
                        .topo
                        .parent(
                            cavity.tags[local_edg[0] as usize],
                            cavity.tags[local_edg[1] as usize],
                        )
                        .unwrap();

                    // tag < 0 on fixed boundaries
                    if tag.1 < 0 {
                        continue;
                    }

                    // projection if needed
                    if tag.0 < E::DIM as Dim {
                        geom.project(&mut edge_center, &tag);
                    }

                    let ftype = FilledCavityType::EdgeCenter((local_edg, edge_center, new_metric));
                    let filled_cavity = FilledCavity::new(&cavity, ftype);

                    // lower the min quality threshold if the min quality in the cavity increases
                    let q_min = q_min.min(cavity.q_min * params.split_min_q_rel);
                    let l_min = l_min.min(cavity.l_min * params.split_min_l_rel);
                    if let CavityCheckStatus::Ok(_) = filled_cavity.check(l_min, f64::MAX, q_min) {
                        trace!("Edge split");
                        for i in &cavity.global_elem_ids {
                            self.remove_elem(*i)?;
                        }
                        let ip = self.insert_vertex(edge_center, &tag, new_metric);
                        for (face, tag) in filled_cavity.faces() {
                            let f = cavity.global_elem(&face);
                            assert!(!f.contains_edge(edg));
                            let e = E::from_vertex_and_face(ip, &f);
                            self.insert_elem(e, tag)?;
                        }
                        for (f, _) in cavity.global_tagged_faces() {
                            self.remove_tagged_face(f)?;
                        }

                        for (b, t) in filled_cavity.tagged_faces_boundary_global() {
                            self.add_tagged_face(E::Face::from_vertex_and_face(ip, &b), t)?;
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

            if n_splits == 0 || n_iter == params.split_max_iter {
                if params.debug {
                    self.check().unwrap();
                }
                return Ok(n_iter);
            }
        }
    }

    /// Try to swap an edge if
    ///   - one of the elements in its cavity has a quality < qmin
    ///   - no edge smaller that `l_min` or longer that `l_max` is created
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_lines)]
    fn try_swap<G: Geometry<D>>(
        &mut self,
        edg: [Idx; 2],
        q_min: f64,
        params: &RemesherParams,
        max_angle: f64,
        cavity: &mut Cavity<D, E, M>,
        geom: &G,
    ) -> Result<TrySwapResult> {
        trace!("Try to swap edge {:?}", edg);
        cavity.init_from_edge(edg, self);
        if cavity.global_elem_ids.len() == 1 {
            trace!("Cannot swap, only one adjacent cell");
            return Ok(TrySwapResult::QualitySufficient);
        }

        if cavity.q_min > q_min {
            trace!("No need to swap, quality sufficient");
            return Ok(TrySwapResult::QualitySufficient);
        }

        let l_min = params
            .swap_min_l_abs
            .min(params.swap_min_l_rel * cavity.l_min);
        let l_max = params
            .swap_max_l_abs
            .min(params.swap_max_l_rel * cavity.l_max);

        trace!(
            "min. / max. allowed edge length = {:.2}, {:.2}",
            l_min,
            l_max
        );

        let Seed::Edge(local_edg) = cavity.seed else {
            unreachable!()
        };
        let local_i0 = local_edg[0] as usize;
        let local_i1 = local_edg[1] as usize;
        let mut q_ref = cavity.q_min;

        let mut vx = 0;
        let mut succeed = false;

        let etag = self
            .topo
            .parent(cavity.tags[local_i0], cavity.tags[local_i1])
            .unwrap();
        // tag < 0 on fixed boundaries
        if etag.1 < 0 {
            return Ok(TrySwapResult::FixedEdge);
        }

        if etag.0 < E::Face::DIM as Dim {
            return Ok(TrySwapResult::CouldNotSwap);
        }

        for n in 0..cavity.n_verts() {
            if n == local_i0 as Idx || n == local_i1 as Idx {
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

            // too difficult otherwise!
            if !cavity.tagged_faces.is_empty() {
                assert!(cavity.tagged_faces.len() == 2);
                if !cavity.tagged_faces().any(|(f, _)| f.contains_vertex(n)) {
                    continue;
                }
            }

            let ftype = FilledCavityType::ExistingVertex(n);
            let filled_cavity = FilledCavity::new(cavity, ftype);

            if filled_cavity.is_same() {
                continue;
            }

            if !filled_cavity.check_boundary_normals(&self.topo, geom, max_angle) {
                trace!("Cannot swap, would create a non smooth surface");
                continue;
            }

            if let CavityCheckStatus::Ok(min_quality) = filled_cavity.check(l_min, l_max, q_ref) {
                trace!("Can swap  from {} : ({} > {})", n, min_quality, q_ref);
                succeed = true;
                q_ref = min_quality;
                vx = n;
            }
        }

        if succeed {
            trace!("Swap from {}", vx);
            let ftype = FilledCavityType::ExistingVertex(vx);
            let filled_cavity = FilledCavity::new(cavity, ftype);
            for e in &cavity.global_elem_ids {
                self.remove_elem(*e)?;
            }
            let global_vx = cavity.local2global[vx as usize];
            for (f, t) in filled_cavity.faces() {
                let f = cavity.global_elem(&f);
                assert!(!f.contains_vertex(global_vx));
                assert!(!f.contains_edge(edg));
                let e = E::from_vertex_and_face(global_vx, &f);
                self.insert_elem(e, t)?;
            }
            for (f, _) in cavity.global_tagged_faces() {
                self.remove_tagged_face(f)?;
            }
            for (b, t) in filled_cavity.tagged_faces_boundary_global() {
                self.add_tagged_face(E::Face::from_vertex_and_face(global_vx, &b), t)?;
            }

            return Ok(TrySwapResult::CouldSwap);
        }

        Ok(TrySwapResult::CouldNotSwap)
    }

    /// Loop over the edges and perform edge swaps if
    ///   - the quality of an adjacent element is < `q_target`
    ///   - no edge smaller than
    ///       min(1/sqrt(2), max(params.swap_min_l_abs, params.cswap_min_l_rel * min(l)))
    ///   - no edge larger than
    ///       max(sqrt(2), min(params.swap_max_l_abs, params.swap_max_l_rel * max(l)))
    ///   - no new boundary face is created if its normal forms an angle > than
    ///       params.max_angle with the normal of the geometry at the face center
    ///   - the edge swap increases the minimum quality of the adjacent elements
    ///
    /// where min(l) and max(l) as the min/max edge length over the entire mesh
    pub fn swap<G: Geometry<D>>(
        &mut self,
        q_target: f64,
        params: &RemesherParams,
        geom: &G,
    ) -> Result<u32> {
        debug!("Swap edges: target quality = {}", q_target);

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
                let res =
                    self.try_swap(edg, q_target, params, params.max_angle, &mut cavity, geom)?;
                match res {
                    TrySwapResult::CouldNotSwap => n_fails += 1,
                    TrySwapResult::CouldSwap => n_swaps += 1,
                    _ => n_ok += 1,
                }
            }

            debug!(
                "Iteration {}: {} edges swapped ({} failed, {} OK)",
                n_iter, n_swaps, n_fails, n_ok
            );
            self.stats
                .push(StepStats::Swap(SwapStats::new(n_swaps, n_fails, self)));
            if n_swaps == 0 || n_iter == params.swap_max_iter {
                if params.debug {
                    self.check().unwrap();
                }
                return Ok(n_iter);
            }
        }
    }

    /// Loop over the edges and collapse them if
    ///   - their length is smaller that 1/sqrt(2)
    ///   - no edge larger than
    ///       max(sqrt(2), min(params.collapse_max_l_abs, params.collapse_max_l_rel * max(l)))
    ///   - no new boundary face is created if its normal forms an angle > than
    ///       params.max_angle with the normal of the geometry at the face center
    ///   - no element with a quality lower than
    ///       max(params.collapse_min_q_abs, params.collapse_min_q_rel * min(q))
    ///
    /// where max(l) and min(q) as the max edge length and min quality over the entire mesh
    #[allow(clippy::too_many_lines)]
    pub fn collapse<G: Geometry<D>>(&mut self, params: &RemesherParams, geom: &G) -> Result<u32> {
        debug!("Collapse elements");

        let l_max = params.collapse_max_l_abs;
        debug!("max. allowed length: {:.2}", l_max);
        let q_min = params.collapse_min_q_abs;
        debug!("min. allowed quality: {:.2}", q_min);

        let mut n_iter = 0;
        let mut cavity = Cavity::new();
        loop {
            n_iter += 1;

            let edges: Vec<_> = self.edges.keys().copied().collect();
            let dims_and_lengths: Vec<_> = self.dims_and_lengths_iter().collect();
            let indices = argsort_edges_increasing_length(&dims_and_lengths);

            let mut n_collapses = 0;
            let mut n_fails = 0;
            for i_edge in indices {
                let edg = edges[i_edge];
                if dims_and_lengths[i_edge].1 < f64::sqrt(0.5) {
                    trace!("Try to collapse edgs {:?}", edg);
                    let (mut i0, mut i1) = edg.into();
                    if !self.verts.contains_key(&i0) {
                        trace!("Cannot collapse: vertex deleted");
                        continue;
                    }
                    if !self.verts.contains_key(&i1) {
                        trace!("Cannot collapse: vertex deleted");
                        continue;
                    }

                    let mut topo_0 = self.verts.get(&i0).unwrap().tag;
                    let mut topo_1 = self.verts.get(&i1).unwrap().tag;
                    // Cannot collapse vertices with entity dim 0
                    if topo_0.0 == 0 && topo_1.0 == 0 {
                        continue;
                    }

                    let tag = self.topo.parent(topo_0, topo_1).unwrap();
                    // tag < 0 on fixed boundaries
                    if tag.1 < 0 {
                        continue;
                    }

                    if topo_0.0 != tag.0 || topo_0.1 != tag.1 {
                        if topo_1.0 == tag.0 && topo_1.1 == tag.1 {
                            trace!("Swap vertices");
                            std::mem::swap(&mut i1, &mut i0);
                            std::mem::swap(&mut topo_1, &mut topo_0);
                        } else {
                            trace!("Cannot collapse, incompatible geometry");
                            continue;
                        }
                    }
                    cavity.init_from_vertex(i0, self);
                    let local_i1 = cavity.get_local_index(i1).unwrap();

                    // too difficult otherwise!
                    if !cavity.tagged_faces.is_empty()
                        && !cavity
                            .tagged_faces()
                            .any(|(f, _)| f.contains_vertex(local_i1))
                    {
                        continue;
                    }

                    let ftype = FilledCavityType::ExistingVertex(local_i1);
                    let filled_cavity = FilledCavity::new(&cavity, ftype);

                    if !filled_cavity.check_tagged_faces(self) {
                        trace!("Cannot collapse, tagged face already present");
                        continue;
                    }

                    if !filled_cavity.check_boundary_normals(&self.topo, geom, params.max_angle) {
                        trace!("Cannot collapse, would create a non smooth surface");
                        continue;
                    }

                    // proposition 1?
                    // lower the min quality threshold if the min quality in the cavity increases
                    let q_min = q_min.min(params.collapse_min_q_rel * cavity.q_min);
                    let l_max = l_max.max(params.collapse_max_l_rel * cavity.l_max);
                    if let CavityCheckStatus::Ok(_) = filled_cavity.check(0.0, l_max, q_min) {
                        trace!("Collapse edge");
                        for i in &cavity.global_elem_ids {
                            self.remove_elem(*i)?;
                        }

                        self.remove_vertex(i0)?;

                        for (f, t) in filled_cavity.faces() {
                            let f = cavity.global_elem(&f);
                            assert!(!f.contains_vertex(i1));
                            self.insert_elem(E::from_vertex_and_face(i1, &f), t)?;
                        }
                        for (f, _) in cavity.global_tagged_faces() {
                            self.remove_tagged_face(f)?;
                        }
                        for (b, t) in filled_cavity.tagged_faces_boundary_global() {
                            assert!(!b.contains_vertex(i1));
                            // self.add_tagged_face(E::Face::from_vertex_and_face(i1, &b), t)?;
                            if self
                                .add_tagged_face(E::Face::from_vertex_and_face(i1, &b), t)
                                .is_err()
                            {
                                filled_cavity.debug();
                                panic!(
                                    "error with face {:?}",
                                    E::Face::from_vertex_and_face(i1, &b)
                                );
                            }
                        }

                        n_collapses += 1;
                    } else {
                        n_fails += 1;
                    }
                }
            }

            debug!(
                "Iteration {}: {n_collapses} edges collapsed, {n_fails} fails",
                n_iter + 1,
            );
            self.stats.push(StepStats::Collapse(CollapseStats::new(
                n_collapses,
                n_fails,
                self,
            )));
            if n_collapses == 0 || n_iter == params.collapse_max_iter {
                if params.debug {
                    self.check().unwrap();
                }
                return Ok(n_iter);
            }
        }
    }

    /// Get the vertices in a vertex cavity usable for smoothing, i.e. with tag that is a children of the cavity vertes
    /// TODO: move to Cavity
    fn get_smoothing_neighbors(&self, cavity: &Cavity<D, E, M>) -> (bool, Vec<Idx>) {
        let mut res = Vec::<Idx>::with_capacity(cavity.n_verts() as usize);
        let Seed::Vertex(i0) = cavity.seed else {
            unreachable!()
        };

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
        let Seed::Vertex(i0) = cavity.seed else {
            unreachable!()
        };
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
        let Seed::Vertex(i0) = cavity.seed else {
            unreachable!()
        };
        let (p0, _, m0) = cavity.vert(i0);
        let mut p0_new = Point::<D>::zeros();
        let mut w = 0.0;
        for i1 in neighbors {
            let p1 = &cavity.points[*i1 as usize];
            let e = p0 - p1;
            let l = m0.length(&e);
            p0_new += l * p1;
            w += l;
        }
        p0_new /= w;
        p0_new
    }

    fn smooth_avro(cavity: &Cavity<D, E, M>, neighbors: &[Idx]) -> Point<D> {
        let Seed::Vertex(i0) = cavity.seed else {
            unreachable!()
        };
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
        let Seed::Vertex(i0) = cavity.seed else {
            unreachable!()
        };
        let (_, t0, m0) = cavity.vert(i0);
        if t0.0 == E::DIM as Dim {
            let mut p0_new = Point::<D>::zeros();
            let mut qmax = cavity.q_min;
            let gfaces: Vec<_> = cavity.faces().map(|(f, _)| cavity.gface(&f)).collect();

            for i_elem in 0..cavity.n_elems() {
                let ge = cavity.gelem(i_elem);

                let n = E::N_VERTS as usize;
                let mut x = vec![0.0; n];

                let func = |x: &[f64], _grad: Option<&mut [f64]>, _params: &mut ()| -> f64 {
                    let p = ge.point(x);
                    let mut q_avg = 0.0;
                    for gf in &gfaces {
                        let ge1 = E::Geom::from_vert_and_face(&p, m0, gf);
                        q_avg += ge1.quality();
                    }
                    q_avg / (gfaces.len() as f64)
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

    fn smooth_iter<G: Geometry<D>>(
        &mut self,
        params: &RemesherParams,
        geom: &G,
        cavity: &mut Cavity<D, E, M>,
        verts: &[Idx],
    ) -> (Idx, Idx, Idx) {
        let (mut n_fails, mut n_min, mut n_smooth) = (0, 0, 0);
        for i0 in verts.iter().copied() {
            trace!("Try to smooth vertex {}", i0);
            cavity.init_from_vertex(i0, self);
            let Seed::Vertex(i0_local) = cavity.seed else {
                unreachable!()
            };
            if cavity.tags[i0_local as usize].0 == 0 {
                continue;
            }

            if cavity.tags[i0_local as usize].1 < 0 {
                continue;
            }

            let (is_local_minimum, neighbors) = self.get_smoothing_neighbors(cavity);

            if params.smooth_keep_local_minima && is_local_minimum {
                trace!("Won't smooth, local minimum of m");
                n_min += 1;
                continue;
            }

            if neighbors.is_empty() {
                trace!("Cannot smooth, no suitable neighbor");
                continue;
            }

            let p0 = &cavity.points[i0_local as usize];
            let m0 = &cavity.metrics[i0_local as usize];
            let t0 = &cavity.tags[i0_local as usize];

            let mut h0_new = Default::default();
            let p0_smoothed = match params.smooth_type {
                SmoothingType::Laplacian => Self::smooth_laplacian(cavity, &neighbors),
                SmoothingType::Laplacian2 => Self::smooth_laplacian_2(cavity, &neighbors),
                SmoothingType::Avro => Self::smooth_avro(cavity, &neighbors),
                #[cfg(feature = "nlopt")]
                SmoothingType::NLOpt => Self::smooth_nlopt(cavity, &neighbors),
            };

            let mut p0_new = Point::<D>::zeros();
            let mut valid = false;

            for omega in params.smooth_relax.iter().copied() {
                p0_new = (1.0 - omega) * p0 + omega * p0_smoothed;

                if t0.0 < E::DIM as Dim {
                    geom.project(&mut p0_new, t0);
                }

                trace!(
                    "Smooth, vertex moved by {} -> {p0_new:?}",
                    (p0 - p0_new).norm()
                );

                let ftype = FilledCavityType::MovedVertex((i0_local, p0_new, *m0));
                let filled_cavity = FilledCavity::new(cavity, ftype);

                if !filled_cavity.check_boundary_normals(&self.topo, geom, params.max_angle) {
                    trace!("Cannot smooth, would create a non smooth surface");
                    continue;
                }

                if let CavityCheckStatus::Ok(_) = filled_cavity.check(0.0, f64::MAX, cavity.q_min) {
                    valid = true;
                    break;
                }
                trace!("Smooth, quality would decrease for omega={}", omega,);
            }

            if !valid {
                n_fails += 1;
                trace!("Smooth, no smoothing is valid");
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

        (n_fails, n_min, n_smooth)
    }
    /// Perform mesh smoothing
    pub fn smooth<G: Geometry<D>>(&mut self, params: &RemesherParams, geom: &G) {
        debug!("Smooth vertices");

        // We modify the vertices while iterating over them so we must copy
        // the keys. Apart from going unsafe the only way to avoid this would be
        // to have one RefCell for each VtxInfo but copying self.verts is cheaper.
        let verts = self.verts.keys().copied().collect::<Vec<_>>();

        let mut cavity = Cavity::new();
        for iter in 0..params.smooth_iter {
            let (n_fails, n_min, n_smooth) = self.smooth_iter(params, geom, &mut cavity, &verts);
            debug!(
                "Iteration {}: {n_smooth} vertices moved, {n_fails} fails, {n_min} local minima",
                iter + 1,
            );
            self.stats
                .push(StepStats::Smooth(SmoothStats::new(n_fails, self)));
        }
        if params.debug {
            self.check().unwrap();
        }
    }

    /// Perform a remeshing iteration
    pub fn remesh<G: Geometry<D>>(&mut self, params: RemesherParams, geom: &G) -> Result<()> {
        debug!("Adapt the mesh");
        let now = Instant::now();

        if params.two_steps {
            let l_max = max_iter(self.lengths_iter());
            if l_max > 2.0 * f64::sqrt(2.0) {
                let l_0 = f64::max(0.5 * l_max, 2.0 * f64::sqrt(2.0));
                debug!("Perform a first step with l_0 = {l_0:.2}");
                let first_step_params = RemesherParams {
                    split_min_q_abs: 0.0,
                    split_min_l_abs: 0.0,
                    ..params.clone()
                };
                for _ in 0..params.num_iter {
                    self.collapse(&first_step_params, geom)?;

                    self.split(l_0, &first_step_params, geom)?;

                    self.swap(0.4, &first_step_params, geom)?;

                    self.swap(0.8, &first_step_params, geom)?;

                    self.smooth(&first_step_params, geom);
                }
            } else {
                debug!("l_max = {l_max}, no first step required");
            }
        }

        for _ in 0..params.num_iter {
            self.collapse(&params, geom)?;

            self.split(f64::sqrt(2.0), &params, geom)?;

            self.swap(0.4, &params, geom)?;

            self.swap(0.8, &params, geom)?;

            self.smooth(&params, geom);
        }

        self.swap(0.4, &params, geom)?;

        self.swap(0.8, &params, geom)?;

        debug!("Done in {}s", now.elapsed().as_secs_f32());
        self.print_stats();
        Ok(())
    }

    /// Print length and quality stats on the mesh / metric
    pub fn print_stats(&self) {
        let stats = Stats::new(self.lengths_iter(), &[f64::sqrt(0.5), f64::sqrt(2.0)]);
        debug!("Length: {}", stats);

        let stats = Stats::new(self.qualities_iter(), &[0.4, 0.6, 0.8]);
        debug!("Qualities: {}", stats);
    }

    /// Return the stats at each remeshing step as a json string
    #[must_use]
    pub fn stats_json(&self) -> String {
        serde_json::to_string_pretty(&self.stats).unwrap()
    }

    /// Return the stats at each remeshing step
    #[must_use]
    pub fn stats(&self) -> &[StepStats] {
        &self.stats
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

    /// Check the edge lengths
    pub fn check_edge_lengths_analytical<F: Fn(&Point<D>) -> M>(&self, f: F) -> (f64, f64, f64) {
        let (mut mini, mut maxi, mut avg) = (f64::MAX, 0.0_f64, 0.0_f64);
        let mut count = 0;
        for [i0, i1] in self.edges.keys() {
            let p0 = &self.verts.get(i0).unwrap().vx;
            let p1 = &self.verts.get(i1).unwrap().vx;
            let m0 = f(p0);
            let m1 = f(p1);
            let l = M::edge_length(p0, &m0, p1, &m1);
            mini = mini.min(l);
            maxi = maxi.max(l);
            avg += l;
            count += 1;
        }
        avg /= f64::from(count);
        (mini, maxi, avg)
    }
}

#[cfg(test)]
mod tests_topo {
    use crate::{
        geometry::NoGeometry,
        mesh::{
            test_meshes::{test_mesh_2d, test_mesh_3d},
            Point,
        },
        metric::IsoMetric,
        remesher::Remesher,
        Tag,
    };
    use rustc_hash::{FxHashMap, FxHashSet};
    use std::collections::hash_map::Entry;

    fn test_topo_2d(etags: [Tag; 2], ftags: [Tag; 4], add_boundary_faces: bool, n_split: i32) {
        let mut mesh = test_mesh_2d();
        mesh.mut_etags().zip(etags).for_each(|(e, t)| *e = t);
        mesh.mut_ftags().zip(ftags).for_each(|(e, t)| *e = t);

        if add_boundary_faces {
            mesh.add_boundary_faces();
        }
        for _ in 0..n_split {
            mesh = mesh.split();
        }

        let bdy = mesh.boundary().0;
        let ftags = mesh.ftags().collect::<FxHashSet<_>>();
        let mut stats = FxHashMap::default();
        for tag in ftags {
            if let Entry::Vacant(v) = stats.entry(tag) {
                let smsh = bdy.extract_tag(tag).mesh;
                let center = smsh.verts().sum::<Point<2>>() / f64::from(smsh.n_verts());
                v.insert((smsh.vol(), center));
            }
        }

        mesh.compute_topology();

        mesh.compute_face_to_elems();
        mesh.check().unwrap();

        let geom = NoGeometry();
        let metric: Vec<_> = mesh.verts().map(|_| IsoMetric::from(0.5)).collect();
        let remesher = Remesher::new(&mesh, &metric, &geom).unwrap();
        remesher.check().unwrap();
        let mut mesh = remesher.to_mesh(false);
        mesh.compute_face_to_elems();
        mesh.check().unwrap();

        let ftags = mesh.ftags().collect::<FxHashSet<_>>();
        let bdy = mesh.boundary().0;
        assert_eq!(ftags.len(), stats.len());
        for (&tag, (vol, center)) in &stats {
            let smsh = bdy.extract_tag(tag).mesh;
            assert!((vol - smsh.vol()).abs() < 1e-10 * vol);
            let new_center = smsh.verts().sum::<Point<2>>() / f64::from(smsh.n_verts());
            assert!((center - new_center).norm() < 1e-10);
        }
    }

    #[test]
    fn test_topo_2d_1() {
        test_topo_2d([1, 1], [1, 2, 3, 4], false, 0);
    }

    #[test]
    fn test_topo_2d_2() {
        test_topo_2d([1, 1], [1, 2, 3, 4], false, 2);
    }

    // Inconsistent face tags (diagonal missing) --> panic
    #[test]
    #[should_panic]
    fn test_topo_2d_3() {
        test_topo_2d([1, 2], [1, 2, 3, 4], false, 0);
    }

    #[test]
    fn test_topo_2d_4() {
        test_topo_2d([1, 2], [1, 2, 3, 4], true, 0);
    }

    #[test]
    fn test_topo_2d_5() {
        test_topo_2d([1, 2], [1, 2, 3, 4], true, 2);
    }

    #[test]
    fn test_topo_2d_6() {
        test_topo_2d([1, 1], [1, 1, 1, 1], false, 0);
    }

    #[test]
    fn test_topo_2d_7() {
        test_topo_2d([1, 1], [1, 1, 1, 1], false, 2);
    }

    #[test]
    fn test_topo_2d_8() {
        test_topo_2d([1, 1], [1, 1, 1, 2], false, 0);
    }
    #[test]
    fn test_topo_2d_9() {
        test_topo_2d([1, 1], [1, 1, 1, 2], false, 2);
    }

    #[test]
    fn test_topo_2d_10() {
        test_topo_2d([1, 1], [1, 2, 1, 2], false, 0);
    }

    #[test]
    fn test_topo_2d_11() {
        test_topo_2d([1, 1], [1, 2, 1, 2], false, 2);
    }

    fn test_topo_3d(ftags: [Tag; 12], n_split: i32) {
        let mut mesh = test_mesh_3d();
        mesh.mut_ftags().zip(ftags).for_each(|(e, t)| *e = t);

        for _ in 0..n_split {
            mesh = mesh.split();
        }
        mesh.compute_topology();

        mesh.compute_face_to_elems();
        mesh.check().unwrap();

        let bdy = mesh.boundary().0;
        let mut stats = FxHashMap::default();
        for tag in ftags {
            if let Entry::Vacant(v) = stats.entry(tag) {
                let smsh = bdy.extract_tag(tag).mesh;
                let center = smsh.verts().sum::<Point<3>>() / f64::from(smsh.n_verts());
                v.insert((smsh.vol(), center));
            }
        }

        let geom = NoGeometry();
        let metric: Vec<_> = mesh.verts().map(|_| IsoMetric::from(0.5)).collect();
        let remesher = Remesher::new(&mesh, &metric, &geom).unwrap();
        remesher.check().unwrap();
        let mut mesh = remesher.to_mesh(false);
        mesh.compute_face_to_elems();
        mesh.check().unwrap();

        let ftags = mesh.ftags().collect::<FxHashSet<_>>();
        let bdy = mesh.boundary().0;
        assert_eq!(ftags.len(), stats.len());
        for (&tag, (vol, center)) in &stats {
            let smsh = bdy.extract_tag(tag).mesh;
            assert!((vol - smsh.vol()).abs() < 1e-10 * vol);
            let new_center = smsh.verts().sum::<Point<3>>() / f64::from(smsh.n_verts());
            assert!((center - new_center).norm() < 1e-10);
        }
    }

    #[test]
    fn test_topo_3d_1() {
        test_topo_3d([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6], 0);
    }

    #[test]
    fn test_topo_3d_2() {
        test_topo_3d([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6], 2);
    }

    #[test]
    fn test_topo_3d_3() {
        test_topo_3d([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 0);
    }

    #[test]
    fn test_topo_3d_4() {
        test_topo_3d([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 2);
    }

    #[test]
    fn test_topo_3d_5() {
        test_topo_3d([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2], 0);
    }

    #[test]
    fn test_topo_3d_6() {
        test_topo_3d([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2], 2);
    }
}

#[cfg(test)]
mod tests {
    use super::RemesherParams;
    use crate::{
        geometry::NoGeometry,
        mesh::{
            test_meshes::{
                h_2d, h_3d, sphere_mesh, test_mesh_2d, test_mesh_3d, test_mesh_3d_single_tet,
                test_mesh_3d_two_tets, test_mesh_moon_2d, GeomHalfCircle2d, SphereGeometry,
            },
            Edge, Elem, GElem, Point, SimplexMesh, Tetrahedron, Triangle,
        },
        metric::{AnisoMetric, AnisoMetric2d, AnisoMetric3d, IsoMetric, Metric},
        remesher::{Remesher, SmoothingType},
        Result,
    };
    use std::f64::consts::PI;

    #[test]
    fn test_init() -> Result<()> {
        let mut mesh = test_mesh_2d();
        mesh.add_boundary_faces();
        mesh.compute_topology();
        let h = vec![IsoMetric::<2>::from(1.); mesh.n_verts() as usize];
        let geom = NoGeometry();
        let remesher = Remesher::new(&mesh, &h, &geom)?;

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

        for (c, tag) in mesh.gelems().map(|ge| ge.center()).zip(mesh.etags()) {
            let mut p = Point::<2>::zeros();
            match tag {
                1 => p.copy_from_slice(&[2. / 3., 1. / 3.]),
                2 => p.copy_from_slice(&[1. / 3., 2. / 3.]),
                _ => unreachable!(),
            }
            let d = (c - p).norm();
            assert!(d < 1e-8);
        }

        for (c, tag) in mesh.gfaces().map(|ge| ge.center()).zip(mesh.ftags()) {
            let mut p = Point::<2>::zeros();
            match tag {
                1 => p.copy_from_slice(&[0.5, 0.]),
                2 => p.copy_from_slice(&[1., 0.5]),
                3 => p.copy_from_slice(&[0.5, 1.]),
                4 => p.copy_from_slice(&[0., 0.5]),
                _ => unreachable!(),
            }
            let d = (c - p).norm();
            assert!(d < 1e-8);
        }

        Ok(())
    }

    #[test]
    #[should_panic]
    fn test_remove_vertex_error() {
        let mut mesh = test_mesh_2d();
        mesh.compute_topology();
        let h = vec![IsoMetric::<2>::from(1.); mesh.n_verts() as usize];

        let mut remesher = Remesher::new(&mesh, &h, &NoGeometry()).unwrap();

        remesher.remove_vertex(5).unwrap();
    }

    #[test]
    #[should_panic]
    fn test_remove_vertex_error_2() {
        let mut mesh = test_mesh_2d();
        mesh.add_boundary_faces();
        mesh.compute_topology();

        let h = vec![IsoMetric::<2>::from(1.); mesh.n_verts() as usize];

        let mut remesher = Remesher::new(&mesh, &h, &NoGeometry()).unwrap();

        remesher.remove_vertex(1).unwrap();
    }

    #[test]
    fn test_remove_elem() -> Result<()> {
        let mut mesh = test_mesh_2d();
        mesh.add_boundary_faces();
        mesh.compute_topology();

        let h = vec![IsoMetric::<2>::from(1.); mesh.n_verts() as usize];

        let geom = NoGeometry();
        let mut remesher = Remesher::new(&mesh, &h, &geom)?;

        remesher.remove_elem(0)?;
        remesher.remove_vertex(1)?;

        assert_eq!(remesher.n_verts(), 3);
        assert_eq!(remesher.n_elems(), 1);
        assert_eq!(remesher.n_edges(), 3);

        Ok(())
    }

    #[test]
    #[should_panic]
    fn test_remove_elem_2() {
        let mut mesh = test_mesh_2d();
        mesh.add_boundary_faces();
        mesh.compute_topology();

        let h = vec![IsoMetric::<2>::from(1.); mesh.n_verts() as usize];

        let mut remesher = Remesher::new(&mesh, &h, &NoGeometry()).unwrap();

        remesher.remove_elem(3).unwrap();
    }

    #[test]
    fn test_split_2d() -> Result<()> {
        let mut mesh = test_mesh_2d().split().split();
        mesh.add_boundary_faces();
        mesh.compute_topology();

        let h: Vec<_> = mesh
            .verts()
            .map(|p| IsoMetric::<2>::from(h_2d(&p)))
            .collect();
        let geom = NoGeometry();
        let mut remesher = Remesher::new(&mesh, &h, &geom)?;

        let params = RemesherParams {
            split_max_iter: 10,
            ..Default::default()
        };
        let n_iter = remesher.split(f64::sqrt(2.0), &params, &geom)?;
        assert!(n_iter < 10);

        remesher.check()?;

        let _mesh = remesher.to_mesh(true);

        Ok(())
    }

    #[test]
    fn test_swap_2d() -> Result<()> {
        let mut mesh = test_mesh_2d().split().split();
        mesh.add_boundary_faces();
        mesh.compute_topology();

        // collapse to lower the quality
        let h = vec![IsoMetric::<2>::from(2.); mesh.n_verts() as usize];
        let geom = NoGeometry();
        let mut remesher = Remesher::new(&mesh, &h, &geom)?;
        remesher.check()?;

        let params = RemesherParams {
            collapse_max_iter: 10,
            swap_max_iter: 10,
            ..Default::default()
        };
        let n_iter = remesher.collapse(&params, &geom)?;
        assert!(n_iter < 10);
        let mesh = remesher.to_mesh(true);
        assert!(f64::abs(mesh.vol() - 1.) < 1e-12);
        remesher.check()?;

        // swap
        let mut mesh = remesher.to_mesh(false);
        mesh.compute_topology();
        let h = vec![IsoMetric::<2>::from(2.); mesh.n_verts() as usize];
        let mut remesher = Remesher::new(&mesh, &h, &geom)?;

        let n_iter = remesher.swap(0.8, &params, &geom)?;
        assert!(n_iter < 10);
        let mesh = remesher.to_mesh(true);
        assert!(f64::abs(mesh.vol() - 1.) < 1e-12);

        remesher.check()?;

        let _mesh = remesher.to_mesh(true);

        Ok(())
    }

    #[test]
    fn test_collapse_2d() -> Result<()> {
        let mut mesh = test_mesh_2d().split().split();
        mesh.add_boundary_faces();
        mesh.compute_topology();

        let h = vec![IsoMetric::<2>::from(2.); mesh.n_verts() as usize];
        let geom = NoGeometry();
        let mut remesher = Remesher::new(&mesh, &h, &geom)?;
        remesher.check()?;

        let params = RemesherParams {
            collapse_max_iter: 10,
            ..Default::default()
        };
        let n_iter = remesher.collapse(&params, &geom)?;
        assert!(n_iter < 10);

        remesher.check()?;

        let _mesh = remesher.to_mesh(true);
        assert!(f64::abs(mesh.vol() - 1.) < 1e-12);

        Ok(())
    }

    #[test]
    fn test_smooth_laplacian_2d() -> Result<()> {
        let coords = vec![
            Point::<2>::new(0., 0.),
            Point::<2>::new(1., 0.),
            Point::<2>::new(1., 1.0),
            Point::<2>::new(0., 1.0),
            Point::<2>::new(0.1, 0.1),
        ];
        let elems = vec![
            Triangle::from_slice(&[0, 1, 4]),
            Triangle::from_slice(&[1, 2, 4]),
            Triangle::from_slice(&[2, 3, 4]),
            Triangle::from_slice(&[3, 0, 4]),
        ];
        let etags = vec![1, 1, 1, 1];
        let faces = vec![
            Edge::from_slice(&[0, 1]),
            Edge::from_slice(&[1, 2]),
            Edge::from_slice(&[2, 3]),
            Edge::from_slice(&[3, 0]),
        ];
        let ftags = vec![1, 2, 3, 4];

        let mut mesh = SimplexMesh::new(coords, elems, etags, faces, ftags);
        mesh.compute_topology();

        let h = vec![IsoMetric::<2>::from(1.); mesh.n_verts() as usize];
        let geom = NoGeometry();
        let mut remesher = Remesher::new(&mesh, &h, &geom)?;

        let params = RemesherParams {
            smooth_type: SmoothingType::Laplacian,
            smooth_iter: 1,
            smooth_relax: vec![1.0],
            ..Default::default()
        };
        remesher.smooth(&params, &geom);
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
        let coords = vec![
            Point::<2>::new(0., 0.),
            Point::<2>::new(1., 0.),
            Point::<2>::new(1., 1.),
            Point::<2>::new(0., 1.),
            Point::<2>::new(0.1, 0.1),
        ];
        let elems = vec![
            Triangle::from_slice(&[0, 1, 4]),
            Triangle::from_slice(&[1, 2, 4]),
            Triangle::from_slice(&[2, 3, 4]),
            Triangle::from_slice(&[3, 0, 4]),
        ];
        let etags = vec![1, 1, 1, 1];
        let faces = vec![
            Edge::from_slice(&[0, 1]),
            Edge::from_slice(&[1, 2]),
            Edge::from_slice(&[2, 3]),
            Edge::from_slice(&[3, 0]),
        ];
        let ftags = vec![1, 2, 3, 4];

        let mut mesh = SimplexMesh::new(coords, elems, etags, faces, ftags);
        mesh.compute_topology();

        let h = vec![IsoMetric::<2>::from(1.); mesh.n_verts() as usize];
        let geom = NoGeometry();
        let mut remesher = Remesher::new(&mesh, &h, &geom)?;

        let params = RemesherParams {
            smooth_type: SmoothingType::NLOpt,
            smooth_iter: 1,
            smooth_relax: vec![1.0],
            ..Default::default()
        };
        remesher.smooth(&params, &geom);
        let pt = remesher.verts.get(&4).unwrap().vx;
        let center = Point::<2>::new(0.5, 0.5);
        assert!((pt - center).norm() < 0.05);

        remesher.check()?;

        let _mesh = remesher.to_mesh(true);

        Ok(())
    }

    #[test]
    fn test_smooth_laplacian_2d_aniso() -> Result<()> {
        let coords = vec![
            Point::<2>::new(0., 0.),
            Point::<2>::new(1., 0.),
            Point::<2>::new(1., 0.1),
            Point::<2>::new(0., 0.1),
            Point::<2>::new(0.01, 0.01),
        ];
        let elems = vec![
            Triangle::from_slice(&[0, 1, 4]),
            Triangle::from_slice(&[1, 2, 4]),
            Triangle::from_slice(&[2, 3, 4]),
            Triangle::from_slice(&[3, 0, 4]),
        ];
        let etags = vec![1, 1, 1, 1];
        let faces = vec![
            Edge::from_slice(&[0, 1]),
            Edge::from_slice(&[1, 2]),
            Edge::from_slice(&[2, 3]),
            Edge::from_slice(&[3, 0]),
        ];
        let ftags = vec![1, 2, 3, 4];

        let mut mesh = SimplexMesh::new(coords, elems, etags, faces, ftags);
        mesh.compute_topology();

        let h = vec![
            AnisoMetric2d::from_slice(&[1., 100., 0.]),
            AnisoMetric2d::from_slice(&[1., 100., 0.]),
            AnisoMetric2d::from_slice(&[1., 100., 0.]),
            AnisoMetric2d::from_slice(&[1., 100., 0.]),
            AnisoMetric2d::from_slice(&[1., 100., 0.]),
        ];
        let geom = NoGeometry();
        let mut remesher = Remesher::new(&mesh, &h, &geom)?;

        let params = RemesherParams {
            smooth_type: SmoothingType::Laplacian,
            smooth_iter: 1,
            smooth_relax: vec![1.0],
            ..Default::default()
        };
        remesher.smooth(&params, &geom);
        let pt = remesher.verts.get(&4).unwrap().vx;
        let center = Point::<2>::new(0.5, 0.05);
        assert!((pt - center).norm() < 1e-12);

        remesher.check()?;

        let _mesh = remesher.to_mesh(true);

        Ok(())
    }

    #[test]
    fn test_smooth_omega_2d() -> Result<()> {
        let coords = vec![
            Point::<2>::new(0., 0.),
            Point::<2>::new(1., 0.),
            Point::<2>::new(1., 1.),
            Point::<2>::new(0., 1.),
            Point::<2>::new(0.1, 0.1),
        ];
        let elems = vec![
            Triangle::from_slice(&[0, 1, 4]),
            Triangle::from_slice(&[1, 2, 4]),
            Triangle::from_slice(&[2, 3, 4]),
            Triangle::from_slice(&[3, 0, 4]),
        ];
        let etags = vec![1, 1, 1, 1];
        let faces = vec![
            Edge::from_slice(&[0, 1]),
            Edge::from_slice(&[1, 2]),
            Edge::from_slice(&[2, 3]),
            Edge::from_slice(&[3, 0]),
        ];
        let ftags = vec![1, 2, 3, 4];

        let mut mesh = SimplexMesh::new(coords, elems, etags, faces, ftags);
        mesh.compute_topology();

        let h = vec![IsoMetric::<2>::from(1.); mesh.n_verts() as usize];
        let geom = NoGeometry();
        let mut remesher = Remesher::new(&mesh, &h, &geom)?;

        let params = RemesherParams {
            smooth_type: SmoothingType::Avro,
            smooth_iter: 10,
            smooth_relax: vec![1.0],
            ..Default::default()
        };
        remesher.smooth(&params, &geom);
        let pt = remesher.verts.get(&4).unwrap().vx;
        let center = Point::<2>::new(0.5, 0.5);
        assert!((pt - center).norm() < 0.01);

        remesher.check()?;

        let _mesh = remesher.to_mesh(true);

        Ok(())
    }

    #[test]
    fn test_smooth_omega_2d_aniso() -> Result<()> {
        let coords = vec![
            Point::<2>::new(0., 0.),
            Point::<2>::new(1., 0.),
            Point::<2>::new(1., 0.1),
            Point::<2>::new(0., 0.1),
            Point::<2>::new(0.01, 0.01),
        ];
        let elems = vec![
            Triangle::from_slice(&[0, 1, 4]),
            Triangle::from_slice(&[1, 2, 4]),
            Triangle::from_slice(&[2, 3, 4]),
            Triangle::from_slice(&[3, 0, 4]),
        ];
        let etags = vec![1, 1, 1, 1];
        let faces = vec![
            Edge::from_slice(&[0, 1]),
            Edge::from_slice(&[1, 2]),
            Edge::from_slice(&[2, 3]),
            Edge::from_slice(&[3, 0]),
        ];
        let ftags = vec![1, 2, 3, 4];

        let mut mesh = SimplexMesh::new(coords, elems, etags, faces, ftags);
        mesh.compute_topology();

        let h = vec![
            AnisoMetric2d::from_slice(&[1., 100., 0.]),
            AnisoMetric2d::from_slice(&[1., 100., 0.]),
            AnisoMetric2d::from_slice(&[1., 100., 0.]),
            AnisoMetric2d::from_slice(&[1., 100., 0.]),
            AnisoMetric2d::from_slice(&[1., 100., 0.]),
        ];
        let geom = NoGeometry();
        let mut remesher = Remesher::new(&mesh, &h, &geom)?;

        let params = RemesherParams {
            smooth_type: SmoothingType::Avro,
            smooth_iter: 10,
            smooth_relax: vec![1.0],
            ..Default::default()
        };
        remesher.smooth(&params, &geom);
        let pt = remesher.verts.get(&4).unwrap().vx;
        let center = Point::<2>::new(0.5, 0.05);
        assert!((pt - center).norm() < 0.01);

        remesher.check()?;

        let _mesh = remesher.to_mesh(true);

        Ok(())
    }

    #[test]
    fn test_adapt_2d() -> Result<()> {
        let mut mesh = test_mesh_2d().split().split();
        mesh.add_boundary_faces();
        mesh.compute_topology();

        for iter in 0..10 {
            let h: Vec<_> = (0..mesh.n_verts())
                .map(|i| IsoMetric::<2>::from(h_2d(&mesh.vert(i))))
                .collect();
            let geom = NoGeometry();
            let mut remesher = Remesher::new(&mesh, &h, &geom)?;

            remesher.remesh(RemesherParams::default(), &geom)?;
            remesher.check()?;

            mesh = remesher.to_mesh(false);
            mesh.compute_topology();

            if iter == 9 {
                let (mini, maxi, _) =
                    remesher.check_edge_lengths_analytical(|x| IsoMetric::<2>::from(h_2d(x)));
                assert!(mini > 0.57, "min. edge length: {mini}");
                assert!(maxi < 1.55, "max. edge length: {maxi}");
            }
        }

        Ok(())
    }

    #[test]
    fn test_adapt_aniso_2d() -> Result<()> {
        let mut mesh = test_mesh_2d();
        mesh.mut_etags().for_each(|t| *t = 1);
        mesh.compute_topology();

        let mfunc = |pt: Point<2>| {
            let y = pt[1];
            let v0 = Point::<2>::new(0.5, 0.);
            let v1 = Point::<2>::new(0., 0.05 + 0.15 * y);
            AnisoMetric2d::from_sizes(&v0, &v1)
        };

        let geom = NoGeometry();
        for iter in 0..3 {
            let h: Vec<_> = mesh.verts().map(mfunc).collect();
            let mut remesher = Remesher::new(&mesh, &h, &geom)?;

            remesher.remesh(RemesherParams::default(), &geom)?;
            remesher.check()?;
            mesh = remesher.to_mesh(true);

            assert!(f64::abs(mesh.vol() - 1.0) < 1e-12, "{} != 1", mesh.vol());
            mesh.compute_topology();

            let (mini, maxi, _) = remesher.check_edge_lengths_analytical(|x| mfunc(*x));
            if iter == 3 {
                assert!(mini > 0.6, "min. edge length: {mini}");
                assert!(maxi < 1.4, "max. edge length: {maxi}");
            }
        }

        Ok(())
    }

    #[test]
    fn test_adapt_2d_geom() -> Result<()> {
        let mut mesh = test_mesh_moon_2d();
        mesh.compute_topology();

        let ref_vol = 0.5 * PI - 2.0 * (0.5 * 1.25 * 1.25 * f64::atan2(1., 0.75) - 0.5 * 0.75);

        for iter in 1..10 {
            let h: Vec<_> = mesh
                .verts()
                .map(|p| IsoMetric::<2>::from(h_2d(&p)))
                .collect();
            let geom = GeomHalfCircle2d();
            let mut remesher = Remesher::new(&mesh, &h, &geom)?;

            let params = RemesherParams {
                split_min_q_abs: 0.4,
                ..RemesherParams::default()
            };
            remesher.remesh(params, &geom)?;
            remesher.check()?;

            mesh = remesher.to_mesh(true);
            mesh.compute_topology();

            let vol = mesh.vol();
            assert!(f64::abs(vol - ref_vol) < 0.05);

            let (mini, maxi, _) =
                remesher.check_edge_lengths_analytical(|x| IsoMetric::<2>::from(h_2d(x)));

            if iter == 9 {
                assert!(mini > 0.5, "min. edge length: {mini}");
                assert!(maxi < 1.41, "max. edge length: {maxi}");
            }
        }

        Ok(())
    }

    #[test]
    fn test_split_tet_3d() -> Result<()> {
        let mut mesh = test_mesh_3d_single_tet();
        mesh.compute_topology();

        let h = vec![IsoMetric::<3>::from(0.25); mesh.n_verts() as usize];
        let geom = NoGeometry();
        let mut remesher = Remesher::new(&mesh, &h, &geom)?;
        remesher.check()?;

        let params = RemesherParams {
            split_max_iter: 10,
            ..Default::default()
        };

        let n_iter = remesher.split(f64::sqrt(2.0), &params, &geom)?;
        assert!(n_iter < 10);
        remesher.check()?;

        let _mesh = remesher.to_mesh(true);

        Ok(())
    }

    #[test]
    fn test_split_3d() -> Result<()> {
        let mut mesh = test_mesh_3d().split().split();
        mesh.compute_topology();

        let h: Vec<_> = mesh
            .verts()
            .map(|p| IsoMetric::<3>::from(h_3d(&p)))
            .collect();
        let geom = NoGeometry();
        let mut remesher = Remesher::new(&mesh, &h, &geom)?;
        remesher.check()?;

        let params = RemesherParams {
            split_max_iter: 10,
            ..Default::default()
        };
        let n_iter = remesher.split(f64::sqrt(2.0), &params, &geom)?;
        assert!(n_iter < 10);

        remesher.check()?;

        let mesh = remesher.to_mesh(true);
        assert!(f64::abs(mesh.vol() - 1.) < 1e-12);

        Ok(())
    }

    #[test]
    fn test_collapse_3d() -> Result<()> {
        let mut mesh = test_mesh_3d().split().split();
        mesh.compute_topology();

        let h = vec![IsoMetric::<3>::from(2.); mesh.n_verts() as usize];
        let geom = NoGeometry();
        let mut remesher = Remesher::new(&mesh, &h, &geom)?;
        remesher.check()?;

        let params = RemesherParams {
            collapse_max_iter: 10,
            ..Default::default()
        };
        let n_iter = remesher.collapse(&params, &geom)?;
        assert!(n_iter < 10);

        remesher.check()?;

        let mesh = remesher.to_mesh(true);
        assert!(f64::abs(mesh.vol() - 1.) < 1e-12);

        Ok(())
    }

    #[test]
    fn test_swap_twotets_3d() -> Result<()> {
        let mut mesh = test_mesh_3d_two_tets();
        mesh.compute_topology();

        // collapse to lower the quality
        let h = vec![IsoMetric::<3>::from(2.); mesh.n_verts() as usize];
        let geom = NoGeometry();
        let mut remesher = Remesher::new(&mesh, &h, &geom)?;

        remesher.check()?;
        let mesh = remesher.to_mesh(true);
        assert!(f64::abs(mesh.vol() - 2.0 * 0.1 / 6.) < 1e-12);

        // swap
        let params = RemesherParams {
            swap_max_iter: 10,
            ..Default::default()
        };

        let n_iter = remesher.swap(0.8, &params, &geom)?;
        assert!(n_iter < 10);
        let mesh = remesher.to_mesh(true);
        assert!(f64::abs(mesh.vol() - 2.0 * 0.1 / 6.) < 1e-12);

        remesher.check()?;

        Ok(())
    }

    #[test]
    fn test_swap_3d() -> Result<()> {
        let mut mesh = test_mesh_3d().split().split();
        mesh.compute_topology();

        // collapse to lower the quality
        let h = vec![IsoMetric::<3>::from(2.); mesh.n_verts() as usize];
        let geom = NoGeometry();
        let mut remesher = Remesher::new(&mesh, &h, &geom)?;
        remesher.check()?;

        let params = RemesherParams {
            collapse_max_iter: 10,
            swap_max_iter: 10,
            ..Default::default()
        };

        let n_iter = remesher.collapse(&params, &geom)?;
        assert!(n_iter < 10);
        let mesh = remesher.to_mesh(true);
        assert!(f64::abs(mesh.vol() - 1.) < 1e-12);
        remesher.check()?;

        // swap
        let mut mesh = remesher.to_mesh(true);
        mesh.compute_topology();

        let h = vec![IsoMetric::<3>::from(2.); mesh.n_verts() as usize];
        let geom = NoGeometry();
        let mut remesher = Remesher::new(&mesh, &h, &geom)?;

        let n_iter = remesher.swap(0.8, &params, &geom)?;
        assert!(n_iter < 10);
        let mesh = remesher.to_mesh(true);
        assert!(f64::abs(mesh.vol() - 1.) < 1e-12);

        remesher.check()?;

        let _mesh = remesher.to_mesh(true);

        Ok(())
    }

    fn get_implied_metric_3d(
        mesh: &SimplexMesh<3, Tetrahedron>,
        pt: Point<3>,
    ) -> Option<AnisoMetric3d> {
        for ge in mesh.gelems() {
            let bcoords = ge.bcoords(&pt);
            if bcoords.min() > -1e-12 {
                return Some(ge.implied_metric());
            }
        }
        None
    }

    #[test]
    fn test_adapt_3d() -> Result<()> {
        let mut mesh = test_mesh_3d().split().split();
        mesh.compute_topology();

        let geom = NoGeometry();

        let pts = [
            Point::<3>::new(0.5, 0.35, 0.35),
            Point::<3>::new(0.0, 0.0, 0.0),
        ];

        for iter in 0..3 {
            let h: Vec<_> = mesh
                .verts()
                .map(|p| IsoMetric::<3>::from(h_3d(&p)))
                .collect();
            let mut remesher = Remesher::new(&mesh, &h, &geom)?;

            remesher.remesh(RemesherParams::default(), &geom)?;

            remesher.check()?;

            mesh = remesher.to_mesh(true);
            mesh.compute_topology();
            assert!(f64::abs(mesh.vol() - 1.) < 1e-12);

            let (mini, maxi, _) =
                remesher.check_edge_lengths_analytical(|x| IsoMetric::<3>::from(h_3d(x)));

            if iter == 2 {
                assert!(mini > 0.34, "min. edge length: {mini}");
                assert!(maxi < 2., "max. edge length: {maxi}");
            }
        }

        for p in &pts {
            let m = get_implied_metric_3d(&mesh, *p).unwrap();
            let (a, b) = m.step(&AnisoMetric3d::from_iso(&IsoMetric::<3>::from(h_3d(p))));
            let c = b.max(1. / a);
            assert!(c < 2.9, "step = {c}");
        }

        Ok(())
    }

    #[test]
    fn test_adapt_aniso_3d() -> Result<()> {
        let mut mesh = test_mesh_3d();
        mesh.compute_topology();

        let mfunc = |_p| {
            let v0 = Point::<3>::new(0.5, 0., 0.);
            let v1 = Point::<3>::new(0.0, 0.5, 0.);
            let v2 = Point::<3>::new(0., 0.0, 0.2);
            AnisoMetric3d::from_sizes(&v0, &v1, &v2)
        };

        let geom = NoGeometry();

        for iter in 0..2 {
            let h: Vec<_> = mesh.verts().map(mfunc).collect();
            let mut remesher = Remesher::new(&mesh, &h, &geom)?;

            let params = RemesherParams {
                split_min_q_abs: 0.4,
                ..RemesherParams::default()
            };
            remesher.remesh(params, &geom)?;
            remesher.check()?;

            mesh = remesher.to_mesh(true);
            mesh.compute_topology();
            assert!(f64::abs(mesh.vol() - 1.0) < 1e-12);

            let (mini, maxi, _) = remesher.check_edge_lengths_analytical(|x| mfunc(*x));

            if iter == 1 {
                assert!(mini > 0.3, "min. edge length: {mini}");
                assert!(maxi < 1.7, "max. edge length: {maxi}");
            }
        }

        Ok(())
    }

    #[test]
    fn test_adapt_aniso_3d_geom() -> Result<()> {
        let mut mesh = sphere_mesh(2);

        let mfunc = |_p| {
            let v0 = Point::<3>::new(0.5, 0., 0.);
            let v1 = Point::<3>::new(0.0, 0.5, 0.);
            let v2 = Point::<3>::new(0., 0.0, 0.1);
            AnisoMetric3d::from_sizes(&v0, &v1, &v2)
        };

        let geom = SphereGeometry;

        let fname = format!("sphere_{}.vtu", 0);
        mesh.write_vtk(&fname, None, None)?;

        for iter in 0..2 {
            let h: Vec<_> = mesh.verts().map(mfunc).collect();
            let mut remesher = Remesher::new(&mesh, &h, &geom)?;

            let params = RemesherParams {
                split_min_q_abs: 0.4,
                ..RemesherParams::default()
            };
            remesher.remesh(params, &geom)?;
            remesher.check()?;

            mesh = remesher.to_mesh(true);
            mesh.compute_topology();

            let (mini, maxi, _) = remesher.check_edge_lengths_analytical(|x| mfunc(*x));

            if iter == 1 {
                assert!(mini > 0.3, "min. edge length: {mini}");
                assert!(maxi < 1.7, "max. edge length: {maxi}");
            }

            let fname = format!("sphere_{}.vtu", iter + 1);
            mesh.write_vtk(&fname, None, None)?;
        }

        Ok(())
    }

    #[test]
    fn test_complexity_2d() -> Result<()> {
        let mut mesh = test_mesh_2d().split().split();
        mesh.add_boundary_faces();
        mesh.compute_topology();
        mesh.compute_volumes();

        let mfunc = |_p| {
            let v0 = Point::<2>::new(0.1, 0.);
            let v1 = Point::<2>::new(0.0, 0.01);
            AnisoMetric2d::from_sizes(&v0, &v1)
        };
        let c_ref = 4. / f64::sqrt(3.0) / (0.1 * 0.01);

        let h: Vec<_> = mesh.verts().map(mfunc).collect();
        let c = mesh.complexity(&h, 0.0, f64::MAX);
        assert!(f64::abs(c - c_ref) < 1e-6 * c);

        let geom = NoGeometry();
        let remesher = Remesher::new(&mesh, &h, &geom)?;

        let c = remesher.complexity();
        assert!(f64::abs(c - c_ref) < 1e-6 * c);

        Ok(())
    }

    #[test]
    fn test_complexity_3d() -> Result<()> {
        let mut mesh = test_mesh_3d().split().split();
        mesh.compute_topology();
        mesh.compute_volumes();

        let mfunc = |_p| {
            let v0 = Point::<3>::new(0.1, 0., 0.);
            let v1 = Point::<3>::new(0.0, 0.01, 0.);
            let v2 = Point::<3>::new(0., 0.0, 2.);
            AnisoMetric3d::from_sizes(&v0, &v1, &v2)
        };
        let c_ref = 6. * f64::sqrt(2.0) / (0.1 * 0.01 * 2.0);

        let h: Vec<_> = mesh.verts().map(mfunc).collect();
        let c = mesh.complexity(&h, 0.0, f64::MAX);
        assert!(f64::abs(c - c_ref) < 1e-6 * c);

        let geom = NoGeometry();
        let remesher = Remesher::new(&mesh, &h, &geom)?;

        let c = remesher.complexity();
        assert!(f64::abs(c - c_ref) < 1e-6 * c);

        remesher.check()?;

        let _mesh = remesher.to_mesh(true);

        Ok(())
    }
}
