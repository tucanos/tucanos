use super::Remesher;
use crate::{
    Dim, Result,
    geometry::Geometry,
    metric::Metric,
    remesher::{
        cavity::{Cavity, CavityCheckStatus, FilledCavity, FilledCavityType, Seed},
        stats::{SmoothStats, StepStats},
    },
};
use log::{debug, trace};
use tmesh::{Vertex, mesh::Simplex};

/// Defines available smoothing algorithms for mesh vertices.
///
/// A set of valid neighbors $`N(i)`$ is built as a subset
/// if the neighbors of vertex $`i`$ that are tagged on the same entity of one of its children
/// The new vertex location $`\tilde v_i`$ is then computed as (where $`||v_j - v_i||_M`$ is the
/// edge length in metric space):
///  - for `Laplacian`
/// ```math
/// \tilde v_i = v_i + \frac{1}{|N(i)|}\sum_{j \in N(i)} (v_j - v_i)
/// ```
///  - for `Laplacian2`
/// ```math
/// \tilde v_i = \frac{\sum_{j \in N(i)} ||v_j - v_i||_M v_j}{\sum_{j \in N(i)} ||v_j - v_i||_M}
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

impl<const D: usize, C: Simplex, M: Metric<D>> Remesher<D, C, M> {
    /// Get the vertices in a vertex cavity usable for smoothing, i.e. with tag that is a children of the cavity vertes
    /// TODO: move to Cavity
    fn get_smoothing_neighbors(&self, cavity: &Cavity<D, C, M>) -> (bool, Vec<usize>) {
        let mut res = Vec::<usize>::with_capacity(cavity.n_verts());
        let Seed::Vertex(i0) = cavity.seed else {
            unreachable!()
        };

        let m0 = &cavity.metrics[i0];
        let t0 = cavity.tags[i0];

        let mut local_minimum = true;
        for i1 in 0..cavity.n_verts() {
            if i1 == i0 {
                continue;
            }
            if res.contains(&i1) {
                continue;
            }
            // check tag
            let t1 = cavity.tags[i1];
            let tag = self.topo.parent(t0, t1);
            if tag.is_none() {
                continue;
            }
            let tag = tag.unwrap();
            if t0.0 != tag.0 || t0.1 != tag.1 {
                continue;
            }

            res.push(i1);

            let m1 = &cavity.metrics[i1];
            if m1.vol() < 1.01 * m0.vol() {
                local_minimum = false;
            }
        }

        (local_minimum, res)
    }

    fn smooth_laplacian(cavity: &Cavity<D, C, M>, neighbors: &[usize]) -> Vertex<D> {
        let Seed::Vertex(i0) = cavity.seed else {
            unreachable!()
        };
        let (p0, _, _) = cavity.vert(i0);
        let mut p0_new = Vertex::<D>::zeros();
        for i1 in neighbors {
            let p1 = &cavity.points[*i1];
            let e = p0 - p1;
            p0_new -= e;
        }
        p0_new /= neighbors.len() as f64;
        p0_new += p0;
        p0_new
    }

    fn smooth_laplacian_2(cavity: &Cavity<D, C, M>, neighbors: &[usize]) -> Vertex<D> {
        let Seed::Vertex(i0) = cavity.seed else {
            unreachable!()
        };
        let (p0, _, m0) = cavity.vert(i0);
        let mut p0_new = Vertex::<D>::zeros();
        let mut w = 0.0;
        for i1 in neighbors {
            let p1 = &cavity.points[*i1];
            let e = p0 - p1;
            let l = m0.length(&e);
            p0_new += l * p1;
            w += l;
        }
        p0_new /= w;
        p0_new
    }

    fn smooth_avro(cavity: &Cavity<D, C, M>, neighbors: &[usize]) -> Vertex<D> {
        let Seed::Vertex(i0) = cavity.seed else {
            unreachable!()
        };
        let (p0, _, m0) = cavity.vert(i0);
        let mut p0_new = Vertex::<D>::zeros();
        for i1 in neighbors {
            let p1 = &cavity.points[*i1];
            let e = p0 - p1;
            let omega = 0.2;
            let l = m0.length(&e);
            let l4 = l.powi(4);
            let fac = omega * (1.0 - l4) * libm::exp(-l4) / l;
            p0_new += fac * e;
        }
        p0_new += p0;
        p0_new
    }

    fn smooth_iter<G: Geometry<D>>(
        &mut self,
        params: &SmoothParams,
        geom: &G,
        cavity: &mut Cavity<D, C, M>,
        verts: &[usize],
    ) -> (usize, usize, usize) {
        let (mut n_fails, mut n_min, mut n_smooth) = (0, 0, 0);
        for i0 in verts.iter().copied() {
            trace!("Try to smooth vertex {i0}");
            cavity.init_from_vertex(i0, self);
            let Seed::Vertex(i0_local) = cavity.seed else {
                unreachable!()
            };
            if cavity.tags[i0_local].0 == 0 {
                continue;
            }

            if cavity.tags[i0_local].1 < 0 {
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

            let p0 = &cavity.points[i0_local];
            let m0 = &cavity.metrics[i0_local];
            let t0 = &cavity.tags[i0_local];

            let p0_smoothed = match params.method {
                SmoothingMethod::Laplacian => Self::smooth_laplacian(cavity, &neighbors),
                SmoothingMethod::Laplacian2 => Self::smooth_laplacian_2(cavity, &neighbors),
                SmoothingMethod::Avro => Self::smooth_avro(cavity, &neighbors),
            };

            let mut p0_new = Vertex::<D>::zeros();
            let mut valid = false;

            for omega in params.relax.iter().copied() {
                p0_new = (1.0 - omega) * p0 + omega * p0_smoothed;

                if t0.0 < D as Dim {
                    geom.project(&mut p0_new, t0);
                }

                trace!(
                    "Smooth, vertex moved by {} -> {p0_new:?}",
                    (p0 - p0_new).norm()
                );

                let ftype = FilledCavityType::MovedVertex((i0_local, p0_new, *m0));
                let filled_cavity = FilledCavity::new(cavity, ftype);

                if !filled_cavity.check_normals(&self.topo, geom, params.max_angle) {
                    trace!("Cannot smooth, would create a non smooth surface");
                    continue;
                }

                if let CavityCheckStatus::Ok(_) = filled_cavity.check(0.0, f64::MAX, cavity.q_min) {
                    valid = true;
                    break;
                }
                trace!("Smooth, quality would decrease for omega={omega}");
            }

            if !valid {
                n_fails += 1;
                trace!("Smooth, no smoothing is valid");
                continue;
            }

            // Smoothing is valid, interpolate the metric at the new vertex location
            let h0_new = cavity.interpolate_metric(&p0_new);

            trace!("Smooth, update vertex");
            {
                let vert = self.verts.get_mut(&i0).unwrap();
                vert.vx = p0_new;
                assert!(h0_new.vol() > 0.0);
                vert.m = h0_new;
            }

            // update the quality; the cavity still holds the old vertex location
            // and metric, so the elements are rebuilt from the updated vertices
            for i_global in &cavity.global_elem_ids {
                let q = self.gelem(&self.elems[i_global].el).quality();
                self.elems.get_mut(i_global).unwrap().q = q;
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

        let mut cavity = Cavity::default();
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

#[cfg(test)]
mod tests {
    use crate::{
        Result,
        geometry::NoGeometry,
        mesh::{MeshTopology, test_meshes::test_mesh_2d},
        metric::IsoMetric,
        remesher::{Remesher, SmoothParams},
    };
    use tmesh::mesh::Mesh;

    #[test]
    fn test_smooth_updates_cached_quality() -> Result<()> {
        let mut mesh = test_mesh_2d().split().split().split();
        mesh.fix()?;

        // perturb the interior vertices so that smoothing moves them
        let flg = mesh.boundary_flag();
        for (k, (v, f)) in mesh.verts_mut().zip(flg).enumerate() {
            if !f {
                v[0] += 0.015 * ((k * 7 % 5) as f64 - 2.0);
                v[1] += 0.015 * ((k * 3 % 5) as f64 - 2.0);
            }
        }

        let h = vec![IsoMetric::<2>::from(0.125); mesh.n_verts()];
        let topo = MeshTopology::new(&mesh);
        let geom = NoGeometry();
        let mut remesher = Remesher::new(&mesh, &topo, &h, &geom)?;
        remesher.smooth(&SmoothParams::default(), &geom, false)?;

        for e in remesher.elems.values() {
            let q = remesher.gelem(&e.el).quality();
            assert!((q - e.q).abs() < 1e-12, "cached quality {} != {q}", e.q);
        }
        Ok(())
    }
}
