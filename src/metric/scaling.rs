use crate::{
    Error, Idx, Result,
    mesh::{Elem, SimplexMesh},
    metric::Metric,
};
use log::{debug, warn};
use rayon::prelude::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator,
    IntoParallelRefMutIterator, ParallelIterator,
};

impl<const D: usize, E: Elem> SimplexMesh<D, E> {
    /// Compute the scaling factor $`\alpha`$ such that complexity of the bounded metric field
    /// equals a target number of elements
    /// ```math
    /// \mathcal C(\mathcal T(\alpha \mathcal M, h_{min}, h_{max})) = N
    /// ```
    /// where the bounded metric is given by
    ///  ```math
    /// \mathcal T(\mathcal M, h_{min}, h_{max}) = \mathcal P ^T \tilde \Lambda \mathcal P
    /// ```
    /// with
    /// ```math
    /// \tilde \Lambda_{ii} = \min(\max(\Lambda_{ii}, h_{max}^{-2}), h_{min}^{-2})
    /// ```
    /// and
    /// ```math
    /// \mathcal M = \mathcal P ^T \Lambda \mathcal P
    pub fn scale_metric_simple<M: Metric<D>>(
        &self,
        m: &[M],
        h_min: f64,
        h_max: f64,
        n_elems: Idx,
        max_iter: Idx,
    ) -> f64 {
        let mut fac = 1.0;
        let mut scale = 1.0;

        let mut sizes: Vec<_> = m.iter().flat_map(Metric::sizes).collect();

        for iter in 0..max_iter {
            sizes.par_iter_mut().for_each(|x| *x *= fac);

            let c = if iter == 0 {
                self.complexity_from_sizes::<M>(&sizes, 0.0, f64::MAX)
            } else {
                self.complexity_from_sizes::<M>(&sizes, h_min, h_max)
            };
            debug!("Iteration {iter}, complexity = {c:.2e}, scale = {scale:.2e}");
            if f64::abs(c - f64::from(n_elems)) < 0.05 * f64::from(n_elems) {
                return scale;
            }
            if iter == max_iter - 1 {
                warn!("Target complexity {n_elems} not reached: complexity {c:.2e}");
                return -1.0;
            }
            fac = f64::powf(f64::from(n_elems) / c, -1. / f64::from(E::DIM));
            scale *= fac;
        }
        -1.0
    }

    fn get_bounded_metric<M: Metric<D>>(
        alpha: f64,
        h_min: f64,
        h_max: f64,
        m: Option<&M>,
        m_f: Option<&M>,
        step: Option<f64>,
        m_i: Option<&M>,
    ) -> M {
        let mut res = M::default();
        if let Some(m) = m {
            res = *m;
            res.scale_with_bounds(alpha, h_min, h_max);
            if let Some(m_f) = m_f {
                res = res.intersect(m_f);
            }
        } else if let Some(m_f) = m_f {
            res = *m_f;
        }
        if let Some(m_i) = m_i {
            res.control_step(m_i, step.unwrap_or(4.0));
        }
        res.scale_with_bounds(1.0, h_min, h_max);
        res
    }

    /// Find the scaling factor $`\alpha`$ such that the complexity
    /// ```math
    /// \mathcal C(\mathcal L(\mathcal T(\alpha \mathcal M, h_{min}, h_{max}) \cap \mathcal M_f, \mathcal M_i, f))
    /// ```
    /// equals a target number of elements. The metric field is modified in-place to
    /// ```math
    /// \mathcal L(\mathcal T(\alpha \mathcal M, h_{min}, h_{max}) \cap \mathcal M_f, \mathcal M_i, f)
    /// ```
    /// An error is returned if $`\mathcal L(\mathcal C(\mathcal M_f), \mathcal M_i, f)`$ is larger than the target
    /// number of elements
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_lines)]
    pub fn scale_metric<M: Metric<D>>(
        &self,
        m: &mut [M],
        h_min: f64,
        h_max: f64,
        n_elems: Idx,
        fixed_m: Option<&[M]>,
        implied_m: Option<&[M]>,
        step: Option<f64>,
        max_iter: Idx,
    ) -> Result<f64> {
        debug!(
            "Scaling the metric (h_min = {h_min}, h_max = {h_max}, n_elems = {n_elems}, max_iter = {max_iter})"
        );
        if fixed_m.is_some() {
            debug!("Using a fixed metric");
        }
        if implied_m.is_some() {
            debug!(
                "Using the implied metric with step = {}",
                step.unwrap_or(4.0)
            );
        }

        let mut scale = if max_iter > 0 {
            self.scale_metric_simple(m, h_min, h_max, n_elems, max_iter)
        } else {
            1.0
        };
        if scale < 0.0 {
            return Err(Error::from("Unable to scale the metric (simple)"));
        }

        if fixed_m.is_some() || implied_m.is_some() {
            let fixed_m = (0..self.n_verts())
                .into_par_iter()
                .map(|i| fixed_m.map(|x| &x[i as usize]));
            let implied_m = (0..self.n_verts())
                .into_par_iter()
                .map(|i| implied_m.map(|x| &x[i as usize]));

            if max_iter > 0 {
                let constrain_m = fixed_m.clone().zip(implied_m.clone()).map(|(m_f, m_i)| {
                    Self::get_bounded_metric(0.0, h_min, h_max, None, m_f, step, m_i)
                });
                let constrain_c = self.complexity_iter(constrain_m, h_min, h_max);

                debug!("Complexity of the constrain metric: {constrain_c}");

                if constrain_c > f64::from(n_elems) {
                    return Err(Error::from(&format!(
                        "The complexity of the constrain metric is {constrain_c:.2e} > n_elems = {n_elems}"
                    )));
                }

                let m_iter = |s: f64| {
                    m.par_iter()
                        .zip(fixed_m.clone())
                        .zip(implied_m.clone())
                        .map(move |((m, m_f), m_i)| {
                            Self::get_bounded_metric(s, h_min, h_max, Some(m), m_f, step, m_i)
                        })
                };

                // Get an upper bound for the bisection
                let mut scale_high = 1.5 * scale;
                for iter in 0..max_iter {
                    let tmp_m = m_iter(scale_high);
                    let c = self.complexity_iter(tmp_m, h_min, h_max);
                    debug!("Iteration {iter}: scale_high = {scale_high:.2e}, complexity = {c:.2e}");

                    if iter == max_iter - 1 {
                        return Err(Error::from("Unable to scale the metric (bisection)"));
                    }

                    if c < f64::from(n_elems) {
                        break;
                    }
                    scale_high *= 1.5;
                }

                // Get an lower bound for the bisection
                let mut scale_low = scale / 1.5;
                for iter in 0..max_iter {
                    let tmp_m = m_iter(scale_low);
                    let c = self.complexity_iter(tmp_m, h_min, h_max);
                    debug!("Iteration {iter}: scale_low = {scale_low:.2e}, complexity = {c:.2e}");

                    if iter == max_iter - 1 {
                        return Err(Error::from("Unable to scale the metric (bisection)"));
                    }

                    if c > f64::from(n_elems) {
                        break;
                    }
                    scale_low /= 1.5;
                }

                // bisection
                for iter in 0..max_iter {
                    scale = 0.5 * (scale_low + scale_high);
                    let tmp_m = m_iter(scale);
                    let c = self.complexity_iter(tmp_m, h_min, h_max);
                    debug!("Iteration {iter}: scale = {scale:.2e}, complexity = {c:.2e}");
                    if f64::abs(c - f64::from(n_elems)) < 0.05 * f64::from(n_elems) {
                        break;
                    }
                    if iter == max_iter - 1 {
                        return Err(Error::from("Unable to scale the metric (bisection)"));
                    }
                    if c < f64::from(n_elems) {
                        scale_high = scale;
                    } else {
                        scale_low = scale;
                    }
                }
            }
            m.par_iter_mut()
                .zip(fixed_m.clone())
                .zip(implied_m.clone())
                .for_each(|((m, m_f), m_i)| {
                    *m = Self::get_bounded_metric(scale, h_min, h_max, Some(m), m_f, step, m_i);
                });
        } else {
            m.iter_mut()
                .for_each(|m| m.scale_with_bounds(scale, h_min, h_max));
        }

        Ok(scale)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        Idx, Result,
        mesh::Point,
        mesh::test_meshes::{test_mesh_2d, test_mesh_3d},
        metric::{AnisoMetric2d, AnisoMetric3d, IsoMetric},
    };

    #[test]
    fn test_cscaling_2d() {
        let mut mesh = test_mesh_2d().split().split();
        mesh.compute_volumes();

        let h = vec![0.1; mesh.n_verts() as usize];
        let mut m: Vec<_> = h.iter().map(|&x| IsoMetric::<2>::from(x)).collect();

        let c0 = mesh
            .scale_metric(&mut m, 0.0, 0.05, 1000, None, None, None, 10)
            .unwrap();
        assert!(c0 > 0.0);
        let c1 = mesh.complexity(&m, 0.0, 0.05);
        assert!(f64::abs(c1 - 1000.) < 100.);
    }

    #[test]
    fn test_scaling_2d_aniso() {
        let mut mesh = test_mesh_2d().split().split();
        mesh.compute_volumes();

        let mfunc = |_p| {
            let v0 = Point::<2>::new(0.5, 0.);
            let v1 = Point::<2>::new(0.0, 4.0);
            AnisoMetric2d::from_sizes(&v0, &v1)
        };

        let mut m: Vec<_> = mesh.verts().map(mfunc).collect();

        let c0 = mesh
            .scale_metric(&mut m, 0.0, 0.05, 1000, None, None, None, 10)
            .unwrap();
        assert!(c0 > 0.0);
        let c1 = mesh.complexity(&m, 0.0, 0.05);
        assert!(f64::abs(c1 - 1000.) < 100.);
    }

    #[test]
    fn test_scaling_3d() -> Result<()> {
        let mut mesh = test_mesh_3d().split().split();
        mesh.compute_volumes();

        let h = vec![0.1; mesh.n_verts() as usize];
        let mut m: Vec<_> = h.iter().map(|&x| IsoMetric::<3>::from(x)).collect();

        let c0 = mesh.scale_metric(&mut m, 0.0, 0.05, 1000, None, None, None, 10);
        assert!(c0.is_err());

        let n_target = (1.0 / f64::powi(0.05, 3) * 15.0) as Idx;
        let c0 = mesh.scale_metric(&mut m, 0.0, 0.05, n_target, None, None, None, 10)?;
        assert!(c0 > 0.0);
        let c1 = mesh.complexity(&m, 0.0, 0.05);
        assert!(f64::abs(c1 - f64::from(n_target)) < 0.1 * f64::from(n_target));

        Ok(())
    }

    #[test]
    fn test_scaling_3d_aniso() -> Result<()> {
        let mut mesh = test_mesh_3d().split().split();
        mesh.compute_volumes();

        let mfunc = |_p| {
            let v0 = Point::<3>::new(0.5, 0., 0.);
            let v1 = Point::<3>::new(0.0, 4.0, 0.);
            let v2 = Point::<3>::new(0.0, 0., 6.0);
            AnisoMetric3d::from_sizes(&v0, &v1, &v2)
        };

        let mut m: Vec<_> = mesh.verts().map(mfunc).collect();

        let c0 = mesh.scale_metric(&mut m, 0.0, 0.05, 1000, None, None, None, 10);
        assert!(c0.is_err());

        let n_target = (1.0 / f64::powi(0.05, 3) * 50.0) as Idx;
        let c0 = mesh.scale_metric(&mut m, 0.0, 0.05, n_target, None, None, None, 10)?;
        assert!(c0 > 0.0);
        let c1 = mesh.complexity(&m, 0.0, 0.05);
        assert!(f64::abs(c1 - f64::from(n_target)) < 0.1 * f64::from(n_target));
        Ok(())
    }

    #[test]
    fn test_scaling_3d_fixed() -> Result<()> {
        let mut mesh = test_mesh_3d().split().split();
        mesh.compute_volumes();

        let h = vec![0.1; mesh.n_verts() as usize];
        let mut m: Vec<_> = h.iter().map(|&x| IsoMetric::<3>::from(x)).collect();
        let fixed_m: Vec<_> = mesh
            .verts()
            .map(|p| IsoMetric::<3>::from(0.1 + p[0] + p[1]))
            .collect();

        let n_target = (1.0 / f64::powi(0.05, 3) * 15.0) as Idx;

        let c0 = mesh.scale_metric(&mut m, 0.0, 0.05, n_target, Some(&fixed_m), None, None, 10)?;
        assert!(c0 > 0.0);
        let c1 = mesh.complexity(&m, 0.0, 0.05);
        assert!(f64::abs(c1 - f64::from(n_target)) < 0.1 * f64::from(n_target));

        Ok(())
    }
}
