use super::Remesher;
use crate::{
    Result,
    geometry::Geometry,
    metric::Metric,
    remesher::{
        cavity::{Cavity, CavityCheckStatus, FilledCavity, FilledCavityType, Seed},
        stats::{StepStats, SwapStats},
    },
};
use log::{debug, info, trace};
use tmesh::{
    mesh::{Edge, Simplex},
    trace_if,
};

enum TrySwapResult {
    QualitySufficient,
    FixedEdge,
    CouldNotSwap,
    CouldSwap(usize, f64),
}

#[derive(Clone, Debug)]
pub struct SwapParams {
    /// Quality below which swap is applied
    pub q: f64,
    /// Max. number of loops through the mesh edges during the swap step
    pub max_iter: u32,
    /// Constraint the length of the newly created edges to be < swap_max_l_rel * max(l) during swap
    pub max_l_rel: f64,
    /// Constraint the length of the newly created edges to be < swap_max_l_abs during swap
    pub max_l_abs: f64,
    /// Constraint the length of the newly created edges to be > swap_min_l_rel * min(l) during swap
    pub min_l_rel: f64,
    /// Constraint the length of the newly created edges to be > swap_min_l_abs during swap
    pub min_l_abs: f64,
    /// Max angle between the normals of the new faces and the geometry (in degrees)
    pub max_angle: f64,
    /// Use an alternative algorithm where edges are ordered
    ///
    /// In this case `q` and `max_iter` are ignored
    pub ordered: bool,
    /// Stops the `ordered` algorithm once this improvement factor is reached.
    pub min_improvement_factor: f64,
}

impl Default for SwapParams {
    fn default() -> Self {
        Self {
            q: 0.8,
            max_iter: 2,
            max_l_rel: 1.5,
            max_l_abs: 1.5 * f64::sqrt(2.0),
            min_l_rel: 0.75,
            min_l_abs: 0.75 / f64::sqrt(2.0),
            max_angle: 25.0,
            ordered: false,
            min_improvement_factor: 1.05, // Default to 5%
        }
    }
}

impl<const D: usize, C: Simplex, M: Metric<D>> Remesher<D, C, M> {
    fn perform_swap(
        &mut self,
        cavity: &Cavity<D, C, M>,
        vx: usize,
        edg: &Edge<usize>,
    ) -> Result<()> {
        let ftype = FilledCavityType::ExistingVertex(vx);
        let filled_cavity = FilledCavity::new(cavity, ftype);
        for e in &cavity.global_elem_ids {
            self.remove_elem(*e)?;
        }
        let global_vx = cavity.local2global[vx];
        for (f, t) in filled_cavity.faces() {
            let f = cavity.global_elem(&f);
            assert!(!f.contains(global_vx));
            assert!(!f.contains_edge(edg));
            let e = C::from_vertex_and_face(global_vx, &f);
            self.insert_elem(e, t)?;
        }
        for (f, _) in cavity.global_tagged_faces() {
            self.remove_tagged_face(f)?;
        }
        for (b, t) in filled_cavity.tagged_faces_boundary_global() {
            self.add_tagged_face(C::FACE::from_vertex_and_face(global_vx, &b), t)?;
        }
        Ok(())
    }

    /// Identifies the optimal vertex to use for an edge swap operation.
    ///
    /// A vertex is selected if swapping to it improves element quality and satisfies
    /// geometric constraints. Specifically:
    /// * The current cavity quality must be below `params.q`.
    /// * The new edge length must be within [`l_min`, `l_max`].
    /// * Topology and boundary constraints must be respected.
    fn find_swap_vertex<G: Geometry<D>>(
        &self,
        edg: Edge<usize>,
        params: &SwapParams,
        cavity: &Cavity<D, C, M>,
        geom: &G,
    ) -> TrySwapResult {
        let dbg = self.debug_edge(edg);

        trace_if!(dbg, "Try to swap edge {edg:?}");
        if C::DIM == 2 {
            assert!(cavity.n_elems() <= 2);
        }

        if cavity.global_elem_ids.len() == 1 {
            trace_if!(dbg, "Cannot swap, only one adjacent cell");
            return TrySwapResult::QualitySufficient;
        }

        if cavity.q_min > params.q {
            trace_if!(
                dbg,
                "No need to swap, quality sufficient ({:.2})",
                cavity.q_min
            );
            return TrySwapResult::QualitySufficient;
        }

        let l_min = params.min_l_abs.min(params.min_l_rel * cavity.l_min);
        let l_max = params.max_l_abs.max(params.max_l_rel * cavity.l_max);

        trace_if!(
            dbg,
            "min. / max. cavity edge length = {:.2}, {:.2}",
            cavity.l_min,
            cavity.l_max
        );
        trace_if!(
            dbg,
            "min. / max. allowed edge length = {l_min:.2}, {l_max:.2}"
        );

        let Seed::Edge(local_edg) = cavity.seed else {
            unreachable!()
        };
        let local_i0 = local_edg.get(0);
        let local_i1 = local_edg.get(1);
        let mut q_ref = cavity.q_min;

        let mut vx = None;

        let edge_tag = self
            .topo
            .parent(cavity.tags[local_i0], cavity.tags[local_i1])
            .unwrap();
        // tag < 0 on fixed boundaries
        if edge_tag.1 < 0 {
            trace_if!(dbg, "Cannot swap: fixed edge");
            return TrySwapResult::FixedEdge;
        }

        if edge_tag.0 == 1 {
            trace_if!(dbg, "Cannot swap: tag");
            return TrySwapResult::CouldNotSwap;
        }

        for n in 0..cavity.n_verts() {
            if n == local_i0 || n == local_i1 {
                continue;
            }

            // check topo
            let ptag = self.topo.parent(edge_tag, cavity.tags[n]);
            if ptag.is_none() {
                trace_if!(dbg, "Cannot swap, incompatible geometry");
                continue;
            }
            let ptag = ptag.unwrap();
            if ptag.0 != edge_tag.0 || ptag.1 != edge_tag.1 {
                trace_if!(dbg, "Cannot swap, incompatible geometry");
                continue;
            }

            // too difficult otherwise!
            if !cavity.tagged_faces.is_empty() {
                assert_eq!(
                    cavity.tagged_faces.len(),
                    2,
                    "{cavity:#?}, etag={edge_tag:?}"
                );
                if !cavity.tagged_faces().any(|(f, _)| f.contains(n)) {
                    continue;
                }
            }

            let filled_cavity = FilledCavity::new(cavity, FilledCavityType::ExistingVertex(n));
            if filled_cavity.is_same() {
                continue;
            }

            if !filled_cavity.check_normals(&self.topo, geom, params.max_angle) {
                trace_if!(dbg, "Cannot swap, would create a non smooth surface");
                continue;
            }

            let status = filled_cavity.check(l_min, l_max, q_ref);
            trace_if!(
                dbg,
                "status = {status:?}, l_min = {l_min:.2e}, l_max = {l_max:.2e}, q_min = {q_ref:.2e}"
            );
            // TODO: concider also the LowQuality case because it may improve valencve
            if let CavityCheckStatus::Ok(min_quality) = status {
                trace_if!(dbg, "Can swap from {n}: {min_quality} > {q_ref}");
                q_ref = min_quality;
                vx = Some(n);
            }
        }
        vx.map_or(TrySwapResult::CouldNotSwap, |vx| {
            TrySwapResult::CouldSwap(vx, q_ref)
        })
    }

    /// Loop over the edges and perform edge swaps if
    /// - the quality of an adjacent element is < `q_target`
    /// - no edge smaller than
    ///   min(1/sqrt(2), max(params.swap_min_l_abs, params.cswap_min_l_rel * min(l)))
    /// - no edge larger than
    ///   max(sqrt(2), min(params.swap_max_l_abs, params.swap_max_l_rel * max(l)))
    /// - no new boundary face is created if its normal forms an angle > than
    ///   params.max_angle with the normal of the geometry at the face center
    /// - the edge swap increases the minimum quality of the adjacent elements
    ///
    /// where min(l) and max(l) as the min/max edge length over the entire mesh
    pub fn swap<G: Geometry<D>>(
        &mut self,
        params: &SwapParams,
        geom: &G,
        debug: bool,
    ) -> Result<u32> {
        if params.ordered {
            ordered::Swapper::default().swap(self, params, geom)?;
            if debug {
                self.check().unwrap();
            }
            return Ok(1);
        }
        debug!("Swap edges: target quality = {}", params.q);

        let mut n_iter = 0;
        let mut cavity = Cavity::default();
        loop {
            n_iter += 1;
            let mut edges = Vec::with_capacity(self.edges.len());
            edges.extend(self.edges.keys().copied());

            let mut n_swaps = 0;
            let mut n_fails = 0;
            let mut n_ok = 0;
            for edg in edges {
                cavity.init_from_edge(edg, self);
                match self.find_swap_vertex(edg, params, &cavity, geom) {
                    TrySwapResult::CouldNotSwap => n_fails += 1,
                    TrySwapResult::CouldSwap(vx, _) => {
                        trace_if!(self.debug_edge(edg), "Swap from {vx}");
                        self.perform_swap(&cavity, vx, &edg)?;
                        n_swaps += 1;
                    }
                    _ => n_ok += 1,
                }
            }

            debug!("Iteration {n_iter}: {n_swaps} edges swapped ({n_fails} failed, {n_ok} OK)");
            self.stats
                .push(StepStats::Swap(SwapStats::new(n_swaps, n_fails, self)));
            if n_swaps == 0 || n_iter == params.max_iter {
                if debug {
                    self.check().unwrap();
                }
                return Ok(n_iter);
            }
        }
    }
}

mod ordered {
    use tmesh::mesh::{Edge, Simplex};
    use crate::{
        geometry::Geometry,
        metric::Metric,
        remesher::{
            Remesher, SwapParams,
            cavity::Cavity,
            orderedhashmap,
            stats::{StepStats, SwapStats},
            swap::TrySwapResult,
        },
    };

    trait IdealValence {
        /// The ideal number of incident edges around a vertex
        fn ideal_valence() -> f64;
    }

    impl<T: Idx> IdealValence for tmesh::mesh::Tetrahedron<T> {
        fn ideal_valence() -> f64 {
            let solid_angle = (23. / 27_f64).acos();
            (4. * std::f64::consts::PI) / solid_angle / 2. + 2.
        }
    }

    impl<T: Idx> IdealValence for tmesh::mesh::Triangle<T> {
        fn ideal_valence() -> f64 {
            6.
        }
    }

    #[derive(Default)]
    pub struct Swapper<const D: usize, C: Simplex, M: Metric<D>> {
        /// An OrderedHashMap where the ordered keys are the improvement factor of the quality
        /// when swapping, the hash keys are edges, and the values are the vertices to swap to.
        map: orderedhashmap::OrderedHashMap<orderedhashmap::OrdF64, Edge<usize>, usize>,
        cavity: Cavity<D, C, M>,
        cavity_edges: Vec<Edge<usize>>,
        num_swaps: usize,
        num_fails: usize,
    }

    impl<const D: usize, C: Simplex, M: Metric<D>> Swapper<D, C, M> {
        /// Collect the cavity edges (excluding the seed edge)
        fn collect_cavity_edges(&mut self) {
            self.cavity_edges.clear();
            for f in &self.cavity.faces {
                for edge in f.0.edges() {
                    self.cavity_edges.push(edge.sorted());
                }
            }
            self.cavity_edges.sort_unstable();
            self.cavity_edges.dedup();
            for e in &mut self.cavity_edges {
                *e = self.cavity.global_elem(e);
            }
        }

        /// Returns the target vertex and quality improvement ratio if the edge can be swapped.
        fn new_proposal<G: Geometry<D>>(
            &mut self,
            edge: Edge<usize>,
            remesher: &Remesher<D, C, M>,
            params: &SwapParams,
            geom: &G,
        ) -> Option<(f64, usize)> {
            self.cavity.init_from_edge(edge, remesher);
            let quality_before = self.cavity.q_min;
            // TODO: add valence check. -1 for edge.get(0) and 1
            // +1 for all other vertice so of the cavity
            // alpha * q_after / q_before + beta * sum(abs(vi_after-vopt)) / sum(abs(vi_before-vopt))
            match remesher.find_swap_vertex(edge, params, &self.cavity, geom) {
                TrySwapResult::CouldNotSwap => {
                    self.num_fails += 1;
                    None
                }
                TrySwapResult::CouldSwap(vertex, quality_after) => {
                    self.num_swaps += 1;
                    Some((quality_after / quality_before, vertex))
                }
                _ => None,
            }
        }

        pub fn swap<G: Geometry<D>>(
            mut self,
            remesher: &mut Remesher<D, C, M>,
            params: &SwapParams,
            geom: &G,
        ) -> crate::Result<()> {
            for &edge in remesher.edges.keys() {
                if let Some((improvement_factor, vertex)) =
                    self.new_proposal(edge, remesher, params, geom)
                {
                    self.map.insert(edge.sorted(), improvement_factor, vertex);
                }
            }
            while let Some((edge, vertex, improvement_factor)) = self.map.pop_last::<f64>() {
                if improvement_factor < params.min_improvement_factor {
                    break;
                }
                self.cavity.init_from_edge(edge, remesher);
                remesher.perform_swap(&self.cavity, vertex, &edge)?;
                self.collect_cavity_edges();
                let cavity_edges = std::mem::take(&mut self.cavity_edges);
                for &edge in &cavity_edges {
                    let sw = self.new_proposal(edge, remesher, params, geom);
                    self.map.update(edge.sorted(), sw);
                }
                self.cavity_edges = cavity_edges;
            }
            remesher.stats.push(StepStats::Swap(SwapStats::new(
                self.num_swaps,
                self.num_fails,
                remesher,
            )));
            Ok(())
        }
    }
}
