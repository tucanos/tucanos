use super::Remesher;
use crate::{
    Dim, Result,
    geometry::Geometry,
    metric::Metric,
    remesher::{
        cavity::{Cavity, CavityCheckStatus, FilledCavity, FilledCavityType, Seed},
        sequential::argsort_edges_decreasing_length,
        stats::{SplitStats, StepStats},
    },
};
use log::{debug, trace};
use tmesh::mesh::{GSimplex, Idx, Simplex};

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
        }
    }
}

impl<T: Idx, const D: usize, C: Simplex<T>, M: Metric<D>> Remesher<T, D, C, M> {
    /// Loop over the edges and split them if
    /// - their length is larger that `l_0`
    /// - no edge smaller than
    ///   min(1/sqrt(2), max(params.split_min_l_abs, params.collapse_min_l_rel * min(l)))
    /// - no element with a quality lower than
    ///   max(params.collapse_min_q_abs, params.collapse_min_q_rel * min(q)))
    ///
    /// where min(l) and min(q) as the max edge length and min quality over the entire mesh
    #[allow(clippy::too_many_lines)]
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
        let mut cavity = Cavity::new();
        loop {
            n_iter += 1;

            let mut edges = Vec::with_capacity(self.edges.len());
            edges.extend(self.edges.keys().copied());

            let mut dims_and_lengths = Vec::with_capacity(self.edges.len());
            dims_and_lengths.extend(self.dims_and_lengths_iter());

            // loop through the edges by increasing dimension and decreasing length
            let indices = argsort_edges_decreasing_length(&dims_and_lengths);

            let mut n_splits = T::ZERO;
            let mut n_fails = T::ZERO;
            let mut n_extended = T::ZERO;
            let mut n_extension_failed = T::ZERO;
            let mut n_removed = T::ZERO;
            for i_edge in indices {
                let edg = edges[i_edge];
                let length = dims_and_lengths[i_edge].1;
                if length > params.l {
                    trace!("Try to split edge {edg:?}, l = {length}");
                    if !self.edges.contains_key(&edg) {
                        continue;
                    }
                    cavity.init_from_edge(edg, self);
                    // TODO: move to Cavity?
                    let Seed::Edge(local_edg) = cavity.seed else {
                        unreachable!()
                    };
                    let (mut edge_center, new_metric) = cavity.seed_barycenter();
                    let tag = self
                        .topo
                        .parent(
                            cavity.tags[local_edg[0].try_into().unwrap()],
                            cavity.tags[local_edg[1].try_into().unwrap()],
                        )
                        .unwrap();

                    // tag < 0 on fixed boundaries
                    if tag.1 < 0 {
                        continue;
                    }

                    // projection if needed
                    if tag.0 < C::DIM as Dim {
                        geom.project(&mut edge_center, &tag);
                    }

                    let ftype = FilledCavityType::EdgeCenter((local_edg, edge_center, new_metric));
                    let filled_cavity = FilledCavity::new(&cavity, ftype);

                    // lower the min quality threshold if the min quality in the cavity increases
                    let q_min = if tag.0 == C::DIM as Dim {
                        q_min.min(cavity.q_min * params.min_q_rel)
                    } else {
                        q_min.min(cavity.q_min * params.min_q_rel_bdy)
                    };
                    let l_min = l_min.min(cavity.l_min * params.min_l_rel);
                    let status = filled_cavity.check(l_min, f64::MAX, q_min);
                    if let CavityCheckStatus::Ok(_) = status {
                        trace!("Edge split");
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
                        n_splits += T::ONE;
                    } else if matches!(status, CavityCheckStatus::Invalid) && tag.0 < C::DIM as Dim
                    {
                        if cavity.elems.len() == 1 {
                            // If the cavity contains one element with two (1 in 2D) tagged faces with the same tag
                            // then we remove the element from the mesh

                            let e = cavity.global_elem(&cavity.elems[0]);
                            let mut tags = vec![0; C::N_FACES];
                            let mut face_tag = 0;
                            let mut n_tagged = 0;
                            let mut same_tags = true;
                            for i in 0..C::N_FACES {
                                let f = e.face(i.try_into().unwrap());
                                let f = f.sorted();
                                let tag = self.tagged_faces.get(&f);
                                if let Some(tag) = tag {
                                    tags[i] = *tag;
                                    n_tagged += 1;
                                    if face_tag == 0 {
                                        face_tag = *tag;
                                    } else if *tag != face_tag {
                                        same_tags = false;
                                    }
                                }
                            }

                            if n_tagged == C::FACE::DIM && same_tags {
                                // remove the element
                                self.remove_elem(cavity.global_elem_ids[0])?;
                                for (i, &t) in tags.iter().enumerate() {
                                    let f = e.face(i.try_into().unwrap());
                                    if t == face_tag {
                                        self.remove_tagged_face(f)?;
                                    } else {
                                        self.add_tagged_face(f, face_tag)?;
                                    }
                                }
                                n_removed += T::ONE;
                            }
                        } else {
                            // Extend the cavity until it can be filled
                            let mut cavity_extension_ok = false;
                            trace!("Cavity extension {edg:?} - {edge_center:?}");
                            for _ in 0..params.max_extensions {
                                let mut tmp = Vec::new();
                                for (face, tag) in cavity.faces() {
                                    let f = cavity.global_elem(&face);
                                    let gf = self.gface(&f);
                                    let ge = C::GEOM::from_vertex_and_face(&edge_center, &gf);
                                    if ge.vol() <= 0.0 {
                                        trace!("f = {gf:?}, vol < 0");
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
                                        // let msh = cavity.to_mesh();
                                        // msh.vtu_writer().export("cavity.vtu")?;
                                        // msh.bdy_vtu_writer().export("cavity_bdy.vtu")?;
                                        failed = true;
                                        break;
                                    }
                                }

                                if failed {
                                    trace!("failed to extend");
                                    break;
                                }
                                trace!("extended by {} elems", tmp.len());
                            }

                            if cavity_extension_ok {
                                let filled_cavity = FilledCavity::new(&cavity, ftype);
                                let status = filled_cavity.check(l_min, f64::MAX, q_min);
                                if let CavityCheckStatus::Ok(_) = status {
                                    trace!("Edge split");
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
                                        self.add_tagged_face(
                                            C::FACE::from_vertex_and_face(ip, &b),
                                            t,
                                        )?;
                                    }
                                    for i in cavity.global_internal_vertices() {
                                        let v = self.verts.get(&i).unwrap();
                                        assert!(v.els.is_empty());
                                        self.verts.remove(&i);
                                    }
                                    n_splits += T::ONE;
                                    n_extended += T::ONE;
                                } else {
                                    trace!("Cannot split: {status:?}");
                                    n_fails += T::ONE;
                                }
                            } else {
                                n_fails += T::ONE;
                                n_extension_failed += T::ONE;
                            }
                        }
                    } else {
                        n_fails += T::ONE;
                    }
                }
            }

            debug!("Iteration {n_iter}: {n_splits} edges split ({n_fails} failed)");
            if n_extended > T::ZERO || n_extension_failed > T::ZERO {
                debug!("{n_extended} cavity extended, {n_extension_failed} extension failed");
            }
            if n_removed > T::ZERO {
                debug!("{n_removed} elements removed");
            }
            self.stats
                .push(StepStats::Split(SplitStats::new(n_splits, n_fails, self)));

            if n_splits == T::ZERO || n_iter == params.max_iter {
                if debug {
                    self.check().unwrap();
                }
                return Ok(n_iter);
            }
        }
    }
}
