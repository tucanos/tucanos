use super::{Remesher, cavity::intersection};
use crate::{
    Dim, Idx, Result,
    geometry::Geometry,
    mesh::{Elem, GElem, Point},
    metric::Metric,
    remesher::{
        cavity::{Cavity, CavityCheckStatus, FilledCavity, FilledCavityType, Seed},
        sequential::argsort_edges_decreasing_length,
        stats::{SplitStats, StepStats},
    },
};
use log::{debug, info, trace};
use rustc_hash::{FxBuildHasher, FxHashSet};
use tmesh::deform::MeshDeformation;

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
    /// Two step split
    pub two_steps: bool,
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
            two_steps: false,
        }
    }
}

pub enum SplitMode {
    All,
    BoundaryOnly,
    NoBoundary,
}

impl<const D: usize, E: Elem, M: Metric<D>> Remesher<D, E, M> {
    /// Loop over the edges and split them if
    /// - their length is larger that `l_0`
    /// - no edge smaller than
    ///   min(1/sqrt(2), max(params.split_min_l_abs, params.collapse_min_l_rel * min(l)))
    /// - no element with a quality lower than
    ///   max(params.collapse_min_q_abs, params.collapse_min_q_rel * min(q)))
    ///
    /// where min(l) and min(q) as the max edge length and min quality over the entire mesh
    ///
    /// If split_mode is `SplitMode::BoundaryOnly` BLABLA
    #[allow(clippy::too_many_lines)]
    pub fn split<G: Geometry<D>>(
        &mut self,
        params: &SplitParams,
        geom: &G,
        debug: bool,
    ) -> Result<u32> {
        if params.two_steps {
            let (n0, verts_to_project) =
                self.split2(params, geom, debug, &SplitMode::BoundaryOnly)?;
            for i_vert in verts_to_project {
                let v = self.verts.get(&i_vert).unwrap();
                let mut p = v.vx;
                let tag = v.tag;
                geom.project(&mut p, &tag);
                let d = p - v.vx;
                assert!(self.deform(i_vert, d, 1));
                break;
            }
            // assert_eq!(verts_to_project.len(), 0, "{verts_to_project:?}");
            let (n1, _) = self.split2(params, geom, debug, &SplitMode::NoBoundary)?;
            Ok(n0 + n1)
        } else {
            let (n1, _) = self.split2(params, geom, debug, &SplitMode::All)?;
            Ok(n1)
        }
    }

    fn deform(&mut self, i_vert: Idx, d: Point<D>, n_steps: Idx) -> bool {
        let v = self.verts.get(&i_vert).unwrap();
        let x = v.vx;
        let d_max = 50.0 * d.norm();

        // Flag the elements that have a face which distance to x is less than d_max
        let mut elems = self.vertex_elements(i_vert).to_vec();
        let mut faces_visited = FxHashSet::with_hasher(FxBuildHasher);

        let mut added_elems = elems.clone();

        let mut bdy_faces = Vec::new();
        loop {
            let mut new_elems = Vec::new();
            for &i_elem in &added_elems {
                let e = self.elems.get(&i_elem).unwrap().el;
                for i_face in 0..E::N_FACES {
                    let face = e.face(i_face).sorted();
                    if faces_visited.insert(face) {
                        let mut f2e = intersection(
                            self.vertex_elements(face[0]),
                            self.vertex_elements(face[1]),
                        );
                        for &i_other in face.iter().skip(2) {
                            f2e = intersection(&f2e, self.vertex_elements(i_other));
                        }
                        if f2e.len() == 1 {
                            bdy_faces.push(face);
                        } else {
                            assert_eq!(f2e.len(), 2);
                            let i_new = if f2e[0] == i_elem { f2e[1] } else { f2e[0] };
                            if !elems.contains(&i_new) {
                                let gf = <<E as Elem>::Geom<D, M> as GElem<D, M>>::Face::from_verts(
                                    face.iter().map(|i| {
                                        let v = self.verts.get(i).unwrap();
                                        (v.vx, v.m)
                                    }),
                                );

                                if gf.distance(&x) < d_max {
                                    println!(
                                        "dist = {:.2e} {:.2e} {:.2e}",
                                        gf.distance(&x),
                                        d_max,
                                        d.norm()
                                    );
                                    new_elems.push(i_new);
                                } else {
                                    bdy_faces.push(face);
                                }
                            }
                        }
                    }
                }
            }
            if new_elems.is_empty() {
                break;
            }
            elems.extend_from_slice(&new_elems);
            added_elems = new_elems;
        }

        let fac = 1.0 / f64::from(n_steps);
        let d_step = fac * d;

        // Get the constraints
        for _ in 0..n_steps {
            let idx_constraints = bdy_faces
                .iter()
                .copied()
                .flatten()
                .collect::<FxHashSet<_>>();
            let constraints = idx_constraints.iter().map(|&i| {
                let v = self.verts.get(&i).unwrap().vx;
                if i == i_vert {
                    (v, d_step)
                } else {
                    (v, Point::<D>::zeros())
                }
            });

            // Compute the deformation
            let deform = MeshDeformation::new(&constraints, Some(1.0));

            // Get the vertex to be deformed
            let idx_verts = elems
                .iter()
                .flat_map(|i| self.elems.get(i).unwrap().el)
                .collect::<FxHashSet<_>>();
            let idx_verts = Vec::from_iter(idx_verts);
            // let old_verts = idx_verts
            //     .iter()
            //     .map(|i| self.verts.get(i).unwrap().vx)
            //     .collect::<Vec<_>>();

            println!("{:?}, {:?}", idx_verts.len(), idx_constraints.len());
            for i in &idx_verts {
                let v = self.verts.get_mut(i).unwrap();
                if *i == i_vert {
                    v.vx += d_step;
                } else if !idx_constraints.contains(i) {
                    println!("{:?}, {:?}", d_step, deform.deform(&v.vx));
                    v.vx += deform.deform(&v.vx);
                }
            }

            // self.to_mesh(false)
            //     .vtu_writer()
            //     .export("local.vtu")
            //     .unwrap();

            for &i_elem in &elems {
                let el = self.elems.get(&i_elem).unwrap().el;
                let ge = <<E as Elem>::Geom<D, M> as GElem<D, M>>::from_verts(el.iter().map(|i| {
                    let v = self.verts.get(i).unwrap();
                    (v.vx, v.m)
                }));
                let q = ge.quality();
                if q < 0.0 {
                    println!("{x:?}");
                }
                // assert!(q > 0.0, "q={q}");
            }

            // Check that the mesh is valid, otherwise revert to the original vertices
            // if elems.iter().any(|&i_elem| {
            //     let e = msh.elem(i_elem);
            //     let ge = msh.gelem(&e);
            //     Cell::<C>::vol(&ge) < 0.
            // }) {
            //     for (&i_vert, &old_pos) in idx_verts.iter().zip(old_verts.iter()) {
            //         msh.set_vert(i_vert, old_pos);
            //     }
            //     return false;
            // }
        }
        true
    }
    /// Loop over the edges and split them if
    /// - their length is larger that `l_0`
    /// - no edge smaller than
    ///   min(1/sqrt(2), max(params.split_min_l_abs, params.collapse_min_l_rel * min(l)))
    /// - no element with a quality lower than
    ///   max(params.collapse_min_q_abs, params.collapse_min_q_rel * min(q)))
    ///
    /// where min(l) and min(q) as the max edge length and min quality over the entire mesh
    ///
    /// If split_mode is `SplitMode::BoundaryOnly` BLABLA
    #[allow(clippy::too_many_lines)]
    pub fn split2<G: Geometry<D>>(
        &mut self,
        params: &SplitParams,
        geom: &G,
        debug: bool,
        split_mode: &SplitMode,
    ) -> Result<(u32, Vec<Idx>)> {
        debug!("Split edges with length > {:.2e}", params.l);

        let l_min = params.min_l_abs;
        debug!("min. allowed length: {l_min:.2}");
        let q_min = params.min_q_abs;
        debug!("min. allowed quality: {q_min:.2}");

        let mut n_iter = 0;
        let mut cavity = Cavity::new();
        let mut verts_to_project = Vec::new();
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
                if length > params.l {
                    trace!("Try to split edge {edg:?}, l = {length}");
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
                    debug_assert_eq!(dims_and_lengths[i_edge].0, tag.0);
                    if tag.0 < E::DIM as Dim && matches!(split_mode, SplitMode::NoBoundary) {
                        continue;
                    }

                    if tag.0 == E::DIM as Dim && matches!(split_mode, SplitMode::BoundaryOnly) {
                        continue;
                    }

                    // tag < 0 on fixed boundaries
                    if tag.1 < 0 {
                        continue;
                    }

                    // projection if needed
                    let unprojected_edge_center = edge_center;
                    if tag.0 < E::DIM as Dim {
                        geom.project(&mut edge_center, &tag);
                    }

                    let ftype = FilledCavityType::EdgeCenter((local_edg, edge_center, new_metric));
                    let filled_cavity = FilledCavity::new(&cavity, ftype);

                    // lower the min quality threshold if the min quality in the cavity increases
                    let q_min = if tag.0 == E::DIM as Dim {
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
                    } else if matches!(status, CavityCheckStatus::Invalid)
                        && matches!(split_mode, SplitMode::BoundaryOnly)
                    {
                        // try with the unprojected vertex
                        let ftype = FilledCavityType::EdgeCenter((
                            local_edg,
                            unprojected_edge_center,
                            new_metric,
                        ));
                        let filled_cavity = FilledCavity::new(&cavity, ftype);

                        // lower the min quality threshold if the min quality in the cavity increases
                        let q_min = q_min.min(cavity.q_min * params.min_q_rel_bdy);
                        let l_min = l_min.min(cavity.l_min * params.min_l_rel);
                        let status = filled_cavity.check(l_min, f64::MAX, q_min);
                        if let CavityCheckStatus::Ok(_) = status {
                            trace!("Edge split");
                            for i in &cavity.global_elem_ids {
                                self.remove_elem(*i)?;
                            }
                            let ip = self.insert_vertex(unprojected_edge_center, &tag, new_metric);
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
                            verts_to_project.push(ip);
                        } else {
                            n_fails += 1;
                        }
                    } else {
                        n_fails += 1;
                    }
                }
            }

            debug!("Iteration {n_iter}: {n_splits} edges split ({n_fails} failed)");
            if matches!(split_mode, SplitMode::BoundaryOnly) {
                info!(
                    "{} boundary vertices could not be projected",
                    verts_to_project.len()
                );
            }
            self.stats
                .push(StepStats::Split(SplitStats::new(n_splits, n_fails, self)));

            if n_splits == 0 || n_iter == params.max_iter {
                if debug {
                    self.check().unwrap();
                }
                return Ok((n_iter, verts_to_project));
            }
        }
    }
}
