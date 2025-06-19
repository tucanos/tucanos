use super::collapse::CollapseParams;
use super::smooth::SmoothParams;
use super::split::SplitParams;
use super::stats::{InitStats, Stats, StepStats};
use super::swap::SwapParams;
use crate::{
    Dim, Error, Idx, Result, Tag, TopoTag,
    geometry::Geometry,
    mesh::{Elem, GElem, Point, SimplexMesh, Topology, get_face_to_elem},
    metric::Metric,
};
use log::debug;
use rustc_hash::FxHashMap;
use sorted_vec::SortedVec;
use std::{cmp::Ordering, fs::File, io::Write, time::Instant};

// /// Get edged indices such that they are sorted by increasing tag dimension and then by
// /// increasing edge length
#[must_use]
pub(super) fn argsort_edges_increasing_length(f: &[(Dim, f64)]) -> Vec<usize> {
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
pub(super) fn argsort_edges_decreasing_length(f: &[(Dim, f64)]) -> Vec<usize> {
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
pub(super) struct VtxInfo<const D: usize, M: Metric<D>> {
    /// Vertex coordinates
    pub(super) vx: Point<D>,
    /// Tag
    pub(super) tag: TopoTag,
    /// Metric
    pub(super) m: M,
    /// Elements containing the vertex
    pub(super) els: sorted_vec::SortedVec<Idx>,
}

/// Element information
#[derive(Clone, Copy)]
pub(super) struct ElemInfo<E: Elem> {
    /// Element connectivity
    pub(super) el: E,
    /// Tag
    pub(super) tag: Tag,
    /// Quality
    pub(super) q: f64,
}

/// Remesher for simplex meshes of elements E in dimension D
pub struct Remesher<const D: usize, E: Elem, M: Metric<D>> {
    /// The topology information
    pub(super) topo: Topology,
    /// Vertices
    pub(super) verts: FxHashMap<Idx, VtxInfo<D, M>>,
    /// Elements
    pub(super) elems: FxHashMap<Idx, ElemInfo<E>>,
    /// Edges
    pub(super) edges: FxHashMap<[Idx; 2], i16>,
    /// Tagged faces
    pub(super) tagged_faces: FxHashMap<E::Face, Tag>,
    /// Next vertex Id
    next_vert: Idx,
    /// Next element Id
    next_elem: Idx,
    /// Statistics
    pub(super) stats: Vec<StepStats>,
}

#[derive(Clone, Debug)]
pub enum RemeshingStep {
    Split(SplitParams),
    Collapse(CollapseParams),
    Swap(SwapParams),
    Smooth(SmoothParams),
}

/// Remesher parameters
#[derive(Clone, Debug)]
pub struct RemesherParams {
    /// Remeshing steps
    pub steps: Vec<RemeshingStep>,
    /// Debug mode
    pub debug: bool,
}

impl Default for RemesherParams {
    fn default() -> Self {
        let mut steps = Vec::new();
        for _ in 0..4 {
            steps.push(RemeshingStep::Collapse(CollapseParams::default()));
            steps.push(RemeshingStep::Split(SplitParams::default()));
            steps.push(RemeshingStep::Swap(SwapParams {
                q: 0.4,
                ..SwapParams::default()
            }));
            steps.push(RemeshingStep::Swap(SwapParams {
                q: 0.8,
                ..SwapParams::default()
            }));
            steps.push(RemeshingStep::Smooth(SmoothParams::default()));
        }
        steps.push(RemeshingStep::Swap(SwapParams {
            q: 0.4,
            ..SwapParams::default()
        }));
        steps.push(RemeshingStep::Swap(SwapParams {
            q: 0.8,
            ..SwapParams::default()
        }));
        Self {
            steps,
            debug: false,
        }
    }
}

impl RemesherParams {
    pub fn set_max_angle(&mut self, angle: f64) {
        for step in &mut self.steps {
            match step {
                RemeshingStep::Split(_) => {}
                RemeshingStep::Collapse(p) => p.max_angle = angle,
                RemeshingStep::Swap(p) => p.max_angle = angle,
                RemeshingStep::Smooth(p) => p.max_angle = angle,
            }
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
            if !v2e.contains(&i_elem) {
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
                        "A face ({f:?}:{ftag:?}) belonging to 2 elements ({:?}:{etag:?}, {:?}:{otag:?}) with different tags is not tagged correctly. ",
                        e.el, other.el
                    )));
                } else if etag == otag && ftag.is_some() {
                    return Err(Error::from(&format!(
                        "A face (({f:?}:{ftag:?})) belonging to 2 elements ({:?}:{etag:?}, {:?}:{otag:?}) with the same tags is not tagged \
                        correctly.",
                        e.el, other.el
                    )));
                }
            } else if ftag.is_none() {
                return Err(Error::from(&format!(
                    "A face (({f:?}:{ftag:?})) belonging to 1 element ({:?}:{etag:?}) is not tagged correctly",
                    e.el
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
            for i_elem in &vert.els {
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
                for i_elem in &vert.unwrap().els {
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
        }
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
            return Err(Error::from(&format!(
                "Invalid element: {el:?} / {ge:?}, quality: {q}"
            )));
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
    pub(super) fn get_elem(&self, i: Idx) -> Option<ElemInfo<E>> {
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
    pub(super) fn add_tagged_face(&mut self, face: E::Face, tag: Tag) -> Result<()> {
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
    pub(super) fn remove_tagged_face(&mut self, face: E::Face) -> Result<()> {
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
    pub(super) fn dims_and_lengths_iter(&self) -> impl Iterator<Item = (Dim, f64)> + '_ {
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

    /// Perform a remeshing iteration
    pub fn remesh<G: Geometry<D>>(&mut self, params: &RemesherParams, geom: &G) -> Result<()> {
        debug!("Adapt the mesh");
        let now = Instant::now();

        for step in &params.steps {
            match step {
                RemeshingStep::Collapse(p) => {
                    let _ = self.collapse(p, geom, params.debug)?;
                }
                RemeshingStep::Split(p) => {
                    let _ = self.split(p, geom, params.debug)?;
                }
                RemeshingStep::Swap(p) => {
                    let _ = self.swap(p, geom, params.debug)?;
                }
                RemeshingStep::Smooth(p) => self.smooth(p, geom, params.debug)?,
            }
        }
        debug!("Done in {}s", now.elapsed().as_secs_f32());
        self.print_stats();
        Ok(())
    }

    /// Print length and quality stats on the mesh / metric
    pub fn print_stats(&self) {
        let stats = Stats::new(self.lengths_iter(), &[f64::sqrt(0.5), f64::sqrt(2.0)]);
        debug!("Length: {stats}");

        let stats = Stats::new(self.qualities_iter(), &[0.4, 0.6, 0.8]);
        debug!("Qualities: {stats}");
    }

    /// Return the stats at each remeshing step as a json string
    #[must_use]
    pub fn stats_json(&self) -> String {
        serde_json::to_string_pretty(&self.stats).unwrap()
    }

    /// Return the stats at each remeshing step
    #[must_use]
    #[allow(clippy::missing_const_for_fn)]
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
        Tag,
        geometry::NoGeometry,
        mesh::{
            Point,
            test_meshes::{test_mesh_2d, test_mesh_3d},
        },
        metric::IsoMetric,
        remesher::Remesher,
    };
    use rustc_hash::{FxHashMap, FxHashSet};
    use std::collections::hash_map::Entry;
    use tmesh::mesh::Mesh;

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
    use tmesh::mesh::Mesh;

    use super::RemesherParams;
    use crate::{
        Result,
        geometry::{LinearGeometry, NoGeometry},
        mesh::{
            Edge, Elem, GElem, Point, SimplexMesh, Tetrahedron, Triangle,
            test_meshes::{
                GeomHalfCircle2d, SphereGeometry, h_2d, h_3d, sphere_mesh, test_mesh_2d,
                test_mesh_3d, test_mesh_3d_single_tet, test_mesh_3d_two_tets, test_mesh_moon_2d,
            },
        },
        metric::{AnisoMetric, AnisoMetric2d, AnisoMetric3d, IsoMetric, Metric},
        remesher::{
            Remesher,
            collapse::CollapseParams,
            sequential::{RemeshingStep, SmoothParams, SplitParams, SwapParams},
            smooth::SmoothingMethod,
        },
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

        let params = SplitParams {
            max_iter: 10,
            ..Default::default()
        };
        let n_iter = remesher.split(&params, &geom, true)?;
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

        let params = CollapseParams {
            max_iter: 10,
            ..Default::default()
        };
        let n_iter = remesher.collapse(&params, &geom, true)?;
        assert!(n_iter < 10);
        let mesh = remesher.to_mesh(true);
        assert!(f64::abs(mesh.vol() - 1.) < 1e-12);
        remesher.check()?;

        // swap
        let mut mesh = remesher.to_mesh(false);
        mesh.compute_topology();
        let h = vec![IsoMetric::<2>::from(2.); mesh.n_verts() as usize];
        let mut remesher = Remesher::new(&mesh, &h, &geom)?;

        let params = SwapParams {
            q: 0.8,
            max_iter: 10,
            ..Default::default()
        };
        let n_iter = remesher.swap(&params, &geom, true)?;
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

        let params = CollapseParams {
            max_iter: 10,
            ..Default::default()
        };
        let n_iter = remesher.collapse(&params, &geom, true)?;
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

        let params = SmoothParams {
            method: SmoothingMethod::Laplacian,
            n_iter: 1,
            relax: vec![1.0],
            ..Default::default()
        };
        remesher.smooth(&params, &geom, true)?;
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

        let params = SmoothParams {
            n_iter: 1,
            method: SmoothingMethod::NLOpt,
            relax: vec![1.0],
            ..Default::default()
        };
        remesher.smooth(&params, &geom, true)?;
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

        let params = SmoothParams {
            method: SmoothingMethod::Laplacian,
            n_iter: 1,
            relax: vec![1.0],
            ..Default::default()
        };
        remesher.smooth(&params, &geom, true)?;
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

        let params = SmoothParams {
            method: SmoothingMethod::Avro,
            n_iter: 10,
            relax: vec![1.0],
            ..Default::default()
        };
        remesher.smooth(&params, &geom, true)?;
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

        let params = SmoothParams {
            method: SmoothingMethod::Avro,
            n_iter: 10,
            relax: vec![1.0],
            ..Default::default()
        };
        remesher.smooth(&params, &geom, true)?;
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

            remesher.remesh(&RemesherParams::default(), &geom)?;
            remesher.check()?;

            mesh = remesher.to_mesh(false);
            mesh.compute_topology();

            if iter == 9 {
                let (mini, maxi, _) =
                    remesher.check_edge_lengths_analytical(|x| IsoMetric::<2>::from(h_2d(x)));
                assert!(mini > 0.43, "min. edge length: {mini}");
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

            remesher.remesh(&RemesherParams::default(), &geom)?;
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

            let mut params = RemesherParams::default();
            if let RemeshingStep::Split(p) = &mut params.steps[1] {
                p.min_q_abs = 0.4;
            } else {
                panic!();
            }
            remesher.remesh(&params, &geom)?;
            remesher.check()?;

            mesh = remesher.to_mesh(true);
            mesh.compute_topology();

            let vol = mesh.vol();
            assert!(f64::abs(vol - ref_vol) < 0.05);

            let (mini, maxi, _) =
                remesher.check_edge_lengths_analytical(|x| IsoMetric::<2>::from(h_2d(x)));

            if iter == 9 {
                assert!(mini > 0.4, "min. edge length: {mini}");
                assert!(maxi < 1.43, "max. edge length: {maxi}");
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

        let params = SplitParams {
            max_iter: 10,
            ..Default::default()
        };

        let n_iter = remesher.split(&params, &geom, true)?;
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

        let params = SplitParams {
            max_iter: 10,
            ..Default::default()
        };
        let n_iter = remesher.split(&params, &geom, true)?;
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

        let params = CollapseParams {
            max_iter: 10,
            ..Default::default()
        };
        let n_iter = remesher.collapse(&params, &geom, true)?;
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
        let params = SwapParams {
            max_iter: 10,
            ..Default::default()
        };

        let n_iter = remesher.swap(&params, &geom, true)?;
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

        let params = CollapseParams {
            max_iter: 10,
            ..Default::default()
        };

        let n_iter = remesher.collapse(&params, &geom, true)?;
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

        let params = SwapParams {
            max_iter: 10,
            ..Default::default()
        };
        let n_iter = remesher.swap(&params, &geom, true)?;
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

            remesher.remesh(&RemesherParams::default(), &geom)?;

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

            let mut params = RemesherParams::default();
            if let RemeshingStep::Split(p) = &mut params.steps[1] {
                p.min_q_abs = 0.4;
            } else {
                panic!();
            }
            remesher.remesh(&params, &geom)?;
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
        let mut mesh = sphere_mesh(3);

        let mfunc = |_p| {
            let v0 = Point::<3>::new(0.5, 0., 0.);
            let v1 = Point::<3>::new(0.0, 0.5, 0.);
            let v2 = Point::<3>::new(0., 0.0, 0.1);
            AnisoMetric3d::from_sizes(&v0, &v1, &v2)
        };

        let geom = SphereGeometry;

        // let fname = format!("sphere_{}.vtu", 0);
        // mesh.write_vtk(&fname, None, None)?;

        for iter in 0..2 {
            let h: Vec<_> = mesh.verts().map(mfunc).collect();
            let mut remesher = Remesher::new(&mesh, &h, &geom)?;

            let mut params = RemesherParams::default();
            for step in &mut params.steps {
                if let RemeshingStep::Split(p) = step {
                    p.min_q_abs = 0.25;
                }
            }
            remesher.remesh(&params, &geom)?;
            remesher.check()?;

            mesh = remesher.to_mesh(true);
            mesh.compute_topology();

            let (mini, maxi, _) = remesher.check_edge_lengths_analytical(|x| mfunc(*x));

            if iter == 1 {
                assert!(mini > 0.24, "min. edge length: {mini}");
                assert!(maxi < 1.7, "max. edge length: {maxi}");
            }

            // let fname = format!("sphere_{}.vtu", iter + 1);
            // mesh.write_vtk(&fname, None, None)?;
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

    #[test]
    fn test_iso3d() -> Result<()> {
        // test used in tucanos-ffi-test
        let verts = [0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1.];
        let elems = [0, 1, 2, 3];
        let faces = [0, 2, 1, 0, 1, 3, 1, 2, 3, 2, 0, 3];
        let metric = [0.1; 4];

        let verts = verts
            .chunks(3)
            .map(Point::<3>::from_column_slice)
            .collect::<Vec<_>>();
        let elems = elems
            .chunks(Tetrahedron::N_VERTS as usize)
            .map(Tetrahedron::from_slice)
            .collect::<Vec<_>>();
        let etags = vec![1; elems.len()];
        let faces = faces
            .chunks(Triangle::N_VERTS as usize)
            .map(Triangle::from_slice)
            .collect::<Vec<_>>();
        let ftags = vec![1, 2, 3, 4];
        let metric = metric
            .iter()
            .map(|&x| IsoMetric::<3>::from(x))
            .collect::<Vec<_>>();
        let mut mesh = SimplexMesh::<3, Tetrahedron>::new(verts, elems, etags, faces, ftags);
        mesh.compute_topology();
        let bdy = mesh.boundary().0;
        let geom = LinearGeometry::new(&mesh, bdy)?;
        let mut remesher = Remesher::new(&mesh, &metric, &geom)?;
        remesher.remesh(&RemesherParams::default(), &geom)?;
        let mesh = remesher.to_mesh(false);
        // mesh.write_meshb("iso3d.meshb")?;
        assert_eq!(mesh.n_verts(), 424);

        Ok(())
    }

    #[test]
    fn test_aniso3d() -> Result<()> {
        // test used in tucanos-ffi-test
        let verts = [0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1.];
        let elems = [0, 1, 2, 3];
        let faces = [0, 2, 1, 0, 1, 3, 1, 2, 3, 2, 0, 3];
        let metric = [[10., 20., 15., 0., 0., 0.]; 4];

        let verts = verts
            .chunks(3)
            .map(Point::<3>::from_column_slice)
            .collect::<Vec<_>>();
        let elems = elems
            .chunks(Tetrahedron::N_VERTS as usize)
            .map(Tetrahedron::from_slice)
            .collect::<Vec<_>>();
        let etags = vec![1; elems.len()];
        let faces = faces
            .chunks(Triangle::N_VERTS as usize)
            .map(Triangle::from_slice)
            .collect::<Vec<_>>();
        let ftags = vec![1, 2, 3, 4];
        let metric = metric
            .iter()
            .map(|x| AnisoMetric3d::from_slice(x))
            .collect::<Vec<_>>();

        let mut mesh = SimplexMesh::<3, Tetrahedron>::new(verts, elems, etags, faces, ftags);
        mesh.compute_topology();
        let bdy = mesh.boundary().0;
        let geom = LinearGeometry::new(&mesh, bdy)?;
        let mut remesher = Remesher::new(&mesh, &metric, &geom)?;
        remesher.remesh(&RemesherParams::default(), &geom)?;
        let mesh = remesher.to_mesh(false);
        // mesh.write_meshb("aniso3d.meshb")?;
        assert_eq!(mesh.n_verts(), 60);

        Ok(())
    }
}
