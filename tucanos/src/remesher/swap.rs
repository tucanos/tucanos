use super::Remesher;
use crate::{
    Dim, Result,
    geometry::Geometry,
    metric::Metric,
    remesher::{
        cavity::{Cavity, CavityCheckStatus, FilledCavity, FilledCavityType, Seed},
        stats::{StepStats, SwapStats},
    },
};
use log::{debug, trace};
use tmesh::mesh::{Edge, Simplex};

enum TrySwapResult {
    QualitySufficient,
    FixedEdge,
    CouldNotSwap,
    CouldSwap,
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
        }
    }
}

impl<const D: usize, C: Simplex, M: Metric<D>> Remesher<D, C, M> {
    /// Try to swap an edge if
    ///   - one of the elements in its cavity has a quality < qmin
    ///   - no edge smaller that `l_min` or longer that `l_max` is created
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_lines)]
    fn try_swap<G: Geometry<D>>(
        &mut self,
        edg: Edge<usize>,
        params: &SwapParams,
        max_angle: f64,
        cavity: &mut Cavity<D, C, M>,
        geom: &G,
    ) -> Result<TrySwapResult> {
        trace!("Try to swap edge {edg:?}");
        cavity.init_from_edge(edg, self);
        if cavity.global_elem_ids.len() == 1 {
            trace!("Cannot swap, only one adjacent cell");
            return Ok(TrySwapResult::QualitySufficient);
        }

        if cavity.q_min > params.q {
            trace!("No need to swap, quality sufficient");
            return Ok(TrySwapResult::QualitySufficient);
        }

        let l_min = params.min_l_abs.min(params.min_l_rel * cavity.l_min);
        let l_max = params.max_l_abs.max(params.max_l_rel * cavity.l_max);

        trace!("min. / max. allowed edge length = {l_min:.2}, {l_max:.2}");

        let Seed::Edge(local_edg) = cavity.seed else {
            unreachable!()
        };
        let local_i0 = local_edg.get(0);
        let local_i1 = local_edg.get(1);
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

        if etag.0 < C::FACE::DIM as Dim {
            return Ok(TrySwapResult::CouldNotSwap);
        }

        for n in 0..cavity.n_verts() {
            if n == local_i0 || n == local_i1 {
                continue;
            }

            // check topo
            let ptag = self.topo.parent(etag, cavity.tags[n]);
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
                if !cavity.tagged_faces().any(|(f, _)| f.contains(n)) {
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
                trace!("Can swap  from {n} : ({min_quality} > {q_ref})");
                succeed = true;
                q_ref = min_quality;
                vx = n;
            }
        }

        if succeed {
            trace!("Swap from {vx}");
            let ftype = FilledCavityType::ExistingVertex(vx);
            let filled_cavity = FilledCavity::new(cavity, ftype);
            for e in &cavity.global_elem_ids {
                self.remove_elem(*e)?;
            }
            let global_vx = cavity.local2global[vx];
            for (f, t) in filled_cavity.faces() {
                let f = cavity.global_elem(&f);
                assert!(!f.contains(global_vx));
                assert!(!f.contains_edge(&edg));
                let e = C::from_vertex_and_face(global_vx, &f);
                self.insert_elem(e, t)?;
            }
            for (f, _) in cavity.global_tagged_faces() {
                self.remove_tagged_face(f)?;
            }
            for (b, t) in filled_cavity.tagged_faces_boundary_global() {
                self.add_tagged_face(C::FACE::from_vertex_and_face(global_vx, &b), t)?;
            }

            return Ok(TrySwapResult::CouldSwap);
        }

        Ok(TrySwapResult::CouldNotSwap)
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
        debug!("Swap edges: target quality = {}", params.q);

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
                let res = self.try_swap(edg, params, params.max_angle, &mut cavity, geom)?;
                match res {
                    TrySwapResult::CouldNotSwap => n_fails += 1,
                    TrySwapResult::CouldSwap => n_swaps += 1,
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
