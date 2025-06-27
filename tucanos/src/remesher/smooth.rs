use super::Remesher;
use crate::{
    Dim, Idx, Result,
    geometry::Geometry,
    mesh::{AsSliceF64, Elem, GElem, Point},
    metric::Metric,
    min_iter,
    remesher::{
        cavity::{Cavity, CavityCheckStatus, FilledCavity, FilledCavityType, Seed},
        stats::{SmoothStats, StepStats},
    },
};
use log::{debug, trace};
#[cfg(feature = "nlopt")]
use nlopt::{Algorithm, Nlopt, Target};

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
#[derive(Clone, Copy, Debug)]
pub enum SmoothingMethod {
    Laplacian,
    Avro,
    #[cfg(feature = "nlopt")]
    NLOpt,
    Laplacian2,
}

#[derive(Clone, Debug)]
pub struct SmoothParams {
    /// Number of smoothing steps
    pub n_iter: u32,
    /// Type of smoothing used
    pub method: SmoothingMethod,
    /// Smoothing relaxation
    pub relax: Vec<f64>,
    /// Don't smooth vertices that are a local metric minimum
    pub keep_local_minima: bool,
    /// Max angle between the normals of the new faces and the geometry (in degrees)
    pub max_angle: f64,
}

impl Default for SmoothParams {
    fn default() -> Self {
        Self {
            n_iter: 2,
            method: SmoothingMethod::Laplacian,
            relax: vec![0.5, 0.25, 0.125],
            keep_local_minima: false,
            max_angle: 25.0,
        }
    }
}

impl<const D: usize, E: Elem, M: Metric<D>> Remesher<D, E, M> {
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
            if res.contains(&i1) {
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
                trace!("NLOpt: {res:?}");
                if res.unwrap().1 > qmax {
                    qmax = res.unwrap().1;
                    let mut sum = 0.0;
                    for i in (1..E::N_VERTS as usize).rev() {
                        x[i] = x[i - 1];
                        sum += x[i];
                    }
                    x[0] = 1.0 - sum;
                    p0_new = ge.point(&x);
                    trace!("bcoords = {x:?}");
                    trace!("p0_new = {p0_new:?}");
                }
            }
            p0_new
        } else {
            Self::smooth_laplacian(cavity, neighbors)
        }
    }

    fn smooth_iter<G: Geometry<D>>(
        &mut self,
        params: &SmoothParams,
        geom: &G,
        cavity: &mut Cavity<D, E, M>,
        verts: &[Idx],
    ) -> (Idx, Idx, Idx) {
        let (mut n_fails, mut n_min, mut n_smooth) = (0, 0, 0);
        for i0 in verts.iter().copied() {
            trace!("Try to smooth vertex {i0}");
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

            if params.keep_local_minima && is_local_minimum {
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
            let p0_smoothed = match params.method {
                SmoothingMethod::Laplacian => Self::smooth_laplacian(cavity, &neighbors),
                SmoothingMethod::Laplacian2 => Self::smooth_laplacian_2(cavity, &neighbors),
                SmoothingMethod::Avro => Self::smooth_avro(cavity, &neighbors),
                #[cfg(feature = "nlopt")]
                SmoothingMethod::NLOpt => Self::smooth_nlopt(cavity, &neighbors),
            };

            let mut p0_new = Point::<D>::zeros();
            let mut valid = false;

            for omega in params.relax.iter().copied() {
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
                trace!("Smooth, quality would decrease for omega={omega}",);
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
    pub fn smooth<G: Geometry<D>>(
        &mut self,
        params: &SmoothParams,
        geom: &G,
        debug: bool,
    ) -> Result<()> {
        debug!("Smooth vertices");

        // We modify the vertices while iterating over them so we must copy
        // the keys. Apart from going unsafe the only way to avoid this would be
        // to have one RefCell for each VtxInfo but copying self.verts is cheaper.
        let verts = self.verts.keys().copied().collect::<Vec<_>>();

        let mut cavity = Cavity::new();
        for iter in 0..params.n_iter {
            let (n_fails, n_min, n_smooth) = self.smooth_iter(params, geom, &mut cavity, &verts);
            debug!(
                "Iteration {}: {n_smooth} vertices moved, {n_fails} fails, {n_min} local minima",
                iter + 1,
            );
            self.stats
                .push(StepStats::Smooth(SmoothStats::new(n_fails, self)));
        }
        if debug {
            self.check()?;
        }

        Ok(())
    }
}
