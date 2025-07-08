use super::Remesher;
use crate::{
    Result,
    geometry::Geometry,
    mesh::Elem,
    metric::Metric,
    remesher::{
        cavity::{Cavity, CavityCheckStatus, FilledCavity, FilledCavityType},
        sequential::argsort_edges_increasing_length,
        stats::{CollapseStats, StepStats},
    },
};
use log::{debug, trace};

#[derive(Clone, Debug)]
pub struct CollapseParams {
    // Length below which collapse is applied
    pub l: f64,
    /// Max. number of loops through the mesh edges during the collapse step
    pub max_iter: u32,
    /// Constraint the length of the newly created edges to be < collapse_max_l_rel * max(l) during collapse
    pub max_l_rel: f64,
    /// Constraint the length of the newly created edges to be < collapse_max_l_abs during collapse
    pub max_l_abs: f64,
    /// Constraint the quality of the newly created elements to be > collapse_min_q_rel * min(q) during collapse
    pub min_q_rel: f64,
    /// Constraint the quality of the newly created elements to be > collapse_min_q_abs during collapse
    pub min_q_abs: f64,
    /// Max angle between the normals of the new faces and the geometry (in degrees)
    pub max_angle: f64,
}

impl Default for CollapseParams {
    fn default() -> Self {
        Self {
            l: 0.5_f64.sqrt(),
            max_iter: 1,
            max_l_rel: 1.0,
            max_l_abs: 1.5 * f64::sqrt(2.0),
            min_q_rel: 1.0,
            min_q_abs: 0.5,
            max_angle: 25.0,
        }
    }
}

impl<const D: usize, E: Elem, M: Metric<D>> Remesher<D, E, M> {
    /// Loop over the edges and collapse them if
    /// - their length is smaller that 1/sqrt(2)
    /// - no edge larger than
    ///   max(sqrt(2), min(params.collapse_max_l_abs, params.collapse_max_l_rel * max(l)))
    /// - no new boundary face is created if its normal forms an angle > than
    ///   params.max_angle with the normal of the geometry at the face center
    /// - no element with a quality lower than
    ///   max(params.collapse_min_q_abs, params.collapse_min_q_rel * min(q))
    ///
    /// where max(l) and min(q) as the max edge length and min quality over the entire mesh
    #[allow(clippy::too_many_lines)]
    pub fn collapse<G: Geometry<D>>(
        &mut self,
        params: &CollapseParams,
        geom: &G,
        debug: bool,
    ) -> Result<u32> {
        debug!("Collapse elements");

        let l_max = params.max_l_abs;
        debug!("max. allowed length: {l_max:.2}");
        let q_min = params.min_q_abs;
        debug!("min. allowed quality: {q_min:.2}");

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
                if dims_and_lengths[i_edge].1 < params.l {
                    trace!("Try to collapse edgs {edg:?}");
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
                    let q_min = q_min.min(params.min_q_rel * cavity.q_min);
                    let l_max = l_max.max(params.max_l_rel * cavity.l_max);
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
            if n_collapses == 0 || n_iter == params.max_iter {
                if debug {
                    self.check().unwrap();
                }
                return Ok(n_iter);
            }
        }
    }
}
