use std::cmp::Ordering;

use super::Remesher;
use crate::{
    Dim, Result,
    geometry::Geometry,
    metric::Metric,
    remesher::{
        cavity::{Cavity, CavityCheckStatus, FilledCavity, FilledCavityType, Seed},
        stats::{SplitStats, StepStats},
    },
};
use log::{debug, info, trace};
use tmesh::{
    mesh::{Edge, GSimplex, Simplex},
    trace_if,
};

#[derive(Clone, Debug)]
pub struct SplitParams {
    // Length above which split is applied
    pub l: f64,
    /// Max. number of loops through the mesh edges during the split step
    pub max_iter: u32,
    /// Constraint the length of the newly created edges to be > split_min_l_rel * min(l) during split
    pub min_l_rel: f64,
    /// Constraint the length of the newly created edges to be > split_min_l_abs during split
    pub min_l_abs: f64,
    /// Constraint the quality of the newly created elements to be > split_min_q_rel * min(q) during split
    pub min_q_rel: f64,
    /// Constraint the quality of the newly created elements to be > split_min_q_rel * min(q) during split for boundary vertices
    pub min_q_rel_bdy: f64,
    /// Constraint the quality of the newly created elements to be > split_min_q_abs during split
    pub min_q_abs: f64,
    /// Max # of cavity extensions
    pub max_extensions: usize,
    /// Max angle between the normals of the new faces and the geometry (in degrees)
    pub max_angle: f64,
}

impl Default for SplitParams {
    fn default() -> Self {
        Self {
            l: 2.0_f64.sqrt(),
            max_iter: 1,
            min_l_rel: 1.0,
            min_l_abs: 0.75 / f64::sqrt(2.0),
            min_q_rel: 0.8,
            min_q_rel_bdy: 0.5,
            min_q_abs: 0.3,
            max_extensions: 20,
            max_angle: 25.0,
        }
    }
}

#[derive(Default)]
struct NumSplitOps {
    splits: usize,
    fails: usize,
    removed: usize,
    extended: usize,
    extension_failed: usize,
}

impl<const D: usize, C: Simplex, M: Metric<D>> Remesher<D, C, M> {
    fn edge_split(
        &mut self,
        edg: Edge<usize>,
        cavity: &Cavity<D, C, M>,
        filled_cavity: &FilledCavity<D, C, M>,
        edge_center: tmesh::Vertex<D>,
        new_metric: M,
        tag: crate::TopoTag,
    ) -> Result<()> {
        for i in &cavity.global_elem_ids {
            self.remove_elem(*i)?;
        }
        let ip = self.insert_vertex(edge_center, &tag, new_metric);
        for (face, tag) in filled_cavity.faces() {
            let f = cavity.global_elem(&face);
            assert!(!f.contains_edge(&edg));
            let e = C::from_vertex_and_face(ip, &f);
            self.insert_elem(e, tag)?;
        }
        for (f, _) in cavity.global_tagged_faces() {
            self.remove_tagged_face(f)?;
        }

        for (b, t) in filled_cavity.tagged_faces_boundary_global() {
            self.add_tagged_face(C::FACE::from_vertex_and_face(ip, &b), t)?;
        }
        Ok(())
    }

    /// Extend the cavity until it can be filled
    fn extend_cavity(
        &self,
        dbg: bool,
        edg: Edge<usize>,
        params: &SplitParams,
        cavity: &mut Cavity<D, C, M>,
        edge_center: tmesh::Vertex<D>,
    ) -> bool {
        let mut cavity_extension_ok = false;
        trace_if!(dbg, "Cavity extension {edg:?} - {edge_center:?}");
        for _ in 0..params.max_extensions {
            let mut tmp = Vec::new();
            for (face, tag) in cavity.faces() {
                let f = cavity.global_elem(&face);
                let gf = self.gface(&f);
                let ge = C::GEOM::from_vert_and_face(&edge_center, &gf);
                if ge.vol() <= 0.0 {
                    trace_if!(dbg, "f = {gf:?}, vol < 0");
                    tmp.push((face, tag));
                }
            }

            if tmp.is_empty() {
                cavity_extension_ok = true;
                break;
            }

            let mut failed = false;
            for &(face, tag) in &tmp {
                if !cavity.extend(self, face, tag) {
                    failed = true;
                    break;
                }
            }

            if failed {
                trace_if!(dbg, "failed to extend");
                break;
            }
            trace_if!(dbg, "extended by {} elems", tmp.len());
        }
        cavity_extension_ok
    }

    /// Attempts to split a single edge
    fn try_split<G: Geometry<D>>(
        &mut self,
        dbg: bool,
        edg: Edge<usize>,
        params: &SplitParams,
        geom: &G,
        cavity: &mut Cavity<D, C, M>,
        num_ops: &mut NumSplitOps,
    ) -> Result<()> {
        let l_min = params.min_l_abs;
        let q_min = params.min_q_abs;
        if !self.edges.contains_key(&edg) {
            trace_if!(dbg, "Edge has been removed");
            return Ok(());
        }
        cavity.init_from_edge(edg, self);
        // TODO: move to Cavity?
        let Seed::Edge(local_edg) = cavity.seed else {
            unreachable!()
        };
        let (mut edge_center, new_metric) = cavity.seed_barycenter();

        let edge_tag = match (C::DIM, &cavity.etags[..]) {
            // For triangle meshes, an edge with two identical tags is never
            // a boundary edge whatever his vertices tags are if the two
            // adjacent faces have the same tag.
            (2, [t1, t2]) if t1 == t2 => (2, *t1),
            _ => self.edge_tag(&edg),
        };
        trace_if!(dbg, "Edge tag: {edge_tag:?}");

        // tag < 0 on fixed boundaries
        if edge_tag.1 < 0 {
            trace_if!(dbg, "Cannot split: fixed boundary");
            return Ok(());
        }

        // projection if needed
        trace_if!(dbg, "edge center : {edge_center:?}");
        if edge_tag.0 < D as Dim {
            geom.project(&mut edge_center, &edge_tag);
            trace_if!(dbg, "projected edge center : {edge_center:?}");
        }

        let ftype = FilledCavityType::EdgeCenter((local_edg, edge_center, new_metric));
        let filled_cavity = FilledCavity::new(cavity, ftype);

        // lower the min quality threshold if the min quality in the cavity increases
        let min_q_rel = if edge_tag.0 == C::DIM as Dim {
            params.min_q_rel
        } else {
            params.min_q_rel_bdy
        };
        let q_min = q_min.min(cavity.q_min * min_q_rel);
        trace_if!(dbg, "using q_min = {q_min:.2e}");
        let l_min = l_min.min(cavity.l_min * params.min_l_rel);
        let status = filled_cavity.check(l_min, f64::MAX, q_min);
        trace_if!(
            dbg,
            "status = {status:?}, l_min = {l_min:.2e}, q_min = {q_min:.2e}"
        );
        match status {
            CavityCheckStatus::Ok(_) => {
                if filled_cavity.check_normals(&self.topo, geom, params.max_angle) {
                    trace_if!(dbg, "Edge split");
                    self.edge_split(
                        edg,
                        cavity,
                        &filled_cavity,
                        edge_center,
                        new_metric,
                        edge_tag,
                    )?;
                    num_ops.splits += 1;
                } else {
                    trace_if!(dbg, "Cannot split: would create a non smooth surface");
                    num_ops.fails += 1;
                }
            }
            CavityCheckStatus::Invalid if edge_tag.0 < C::DIM as Dim => {
                if cavity.elems.len() == 1 && self.remove_boundary_element(cavity, geom)? {
                    num_ops.removed += 1;
                    trace_if!(dbg, "Remove one element");
                } else if self.extend_cavity(dbg, edg, params, cavity, edge_center) {
                    let filled_cavity = FilledCavity::new(cavity, ftype);
                    let status = filled_cavity.check(l_min, f64::MAX, q_min);
                    if let CavityCheckStatus::Ok(_) = status {
                        if filled_cavity.check_normals(&self.topo, geom, params.max_angle) {
                            trace_if!(dbg, "Edge split");
                            self.edge_split(
                                edg,
                                cavity,
                                &filled_cavity,
                                edge_center,
                                new_metric,
                                edge_tag,
                            )?;
                            for i in cavity.global_internal_vertices() {
                                let v = self.verts.get(&i).unwrap();
                                assert!(v.els.is_empty());
                                self.verts.remove(&i);
                            }
                            num_ops.splits += 1;
                            num_ops.extended += 1;
                        } else {
                            num_ops.fails += 1;
                            trace_if!(dbg, "Cannot split: would create a non smooth surface");
                        }
                    } else {
                        trace_if!(dbg, "Cannot split: {status:?}");
                        num_ops.fails += 1;
                    }
                } else {
                    num_ops.fails += 1;
                    num_ops.extension_failed += 1;
                }
            }
            _ => num_ops.fails += 1,
        }
        Ok(())
    }

    fn sort_edges_split(&self) -> Vec<(Edge<usize>, Dim, f64)> {
        let func = |e: Edge<usize>| {
            let p0 = self.verts.get(&e.get(0)).unwrap();
            let p1 = self.verts.get(&e.get(1)).unwrap();

            (
                e,
                self.topo.parent(p0.tag, p1.tag).unwrap().0,
                M::edge_length(&p0.vx, &p0.m, &p1.vx, &p1.m),
            )
        };
        let mut edges: Vec<_> = self.edges.keys().copied().map(func).collect();
        edges.sort_by(|(_, d0, l0), (_, d1, l1)| match d0.cmp(d1) {
            Ordering::Less => Ordering::Less,
            Ordering::Equal => l1.partial_cmp(l0).unwrap(),
            Ordering::Greater => Ordering::Greater,
        });
        edges
    }

    /// Loop over the edges and split them if
    /// - their length is larger that `l_0`
    /// - no edge smaller than
    ///   min(1/sqrt(2), max(params.split_min_l_abs, params.collapse_min_l_rel * min(l)))
    /// - no element with a quality lower than
    ///   max(params.collapse_min_q_abs, params.collapse_min_q_rel * min(q)))
    ///
    /// where min(l) and min(q) as the max edge length and min quality over the entire mesh
    pub fn split<G: Geometry<D>>(
        &mut self,
        params: &SplitParams,
        geom: &G,
        debug: bool,
    ) -> Result<u32> {
        debug!("Split edges with length > {:.2e}", params.l);

        let l_min = params.min_l_abs;
        debug!("min. allowed length: {l_min:.2}");
        let q_min = params.min_q_abs;
        debug!("min. allowed quality: {q_min:.2}");

        let mut n_iter = 0;
        let mut cavity = Cavity::default();
        loop {
            n_iter += 1;

            let edges = self.sort_edges_split();

            let mut num_ops = NumSplitOps::default();
            for (edg, _, length) in edges {
                let dbg = self.debug_edge(edg);
                if length > params.l {
                    trace_if!(dbg, "Try to split edge {edg:?}, l = {length}");
                    self.try_split(dbg, edg, params, geom, &mut cavity, &mut num_ops)?;
                } else {
                    trace_if!(dbg, "No need to split edge {edg:?}, l = {length}");
                }
            }

            debug!(
                "Iteration {n_iter}: {} edges split ({} failed)",
                num_ops.splits, num_ops.fails
            );
            if num_ops.extended > 0 || num_ops.extension_failed > 0 {
                debug!(
                    "{} cavity extended, {} extension failed",
                    num_ops.extended, num_ops.extension_failed
                );
            }
            if num_ops.removed > 0 {
                debug!("{} elements removed", num_ops.removed);
            }
            self.stats.push(StepStats::Split(SplitStats::new(
                num_ops.splits,
                num_ops.fails,
                num_ops.removed,
                num_ops.extended,
                self,
            )));
            if num_ops.splits == 0 || n_iter == params.max_iter {
                if debug {
                    self.check().unwrap();
                }
                return Ok(n_iter);
            }
        }
    }
}
