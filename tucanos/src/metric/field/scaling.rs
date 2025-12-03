use crate::{
    Error, Result,
    metric::{Metric, MetricField},
};
use log::{debug, warn};
use rayon::prelude::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator,
    IntoParallelRefMutIterator, ParallelIterator,
};
use tmesh::mesh::{Mesh, Simplex};

impl<const D: usize, C: Simplex, M: Mesh<D, C>, T: Metric<D>> MetricField<'_, D, C, M, T> {
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
    pub fn scale_metric_simple(
        &self,
        h_min: f64,
        h_max: f64,
        n_elems: usize,
        max_iter: u32,
    ) -> f64 {
        let mut fac = 1.0;
        let mut scale = 1.0;
        let n_elems = n_elems as f64;

        let mut sizes: Vec<_> = self.metric.iter().flat_map(Metric::sizes).collect();

        for iter in 0..max_iter {
            sizes.par_iter_mut().for_each(|x| *x *= fac);

            let c = if iter == 0 {
                self.complexity_from_sizes(&sizes, 0.0, f64::MAX)
            } else {
                self.complexity_from_sizes(&sizes, h_min, h_max)
            };
            debug!(
                "Iteration {iter}, complexity = {c:.2e} - target = {n_elems:.2e}, scale = {scale:.2e}"
            );
            if f64::abs(c - n_elems) < 0.05 * n_elems {
                return scale;
            }
            if iter == max_iter - 1 {
                warn!("Target complexity {n_elems} not reached: complexity {c:.2e}");
                return -1.0;
            }
            fac = f64::powf(n_elems / c, -1. / C::DIM as f64);
            scale *= fac;
        }
        -1.0
    }

    fn get_bounded_metric(
        alpha: f64,
        h_min: f64,
        h_max: f64,
        m: Option<&T>,
        m_f: Option<&T>,
        step: Option<f64>,
        m_i: Option<&T>,
    ) -> T {
        let mut res = T::default();
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
    #[allow(clippy::too_many_lines)]
    pub fn scale(
        &mut self,
        h_range: (f64, f64),
        n_elems: usize,
        fixed_m: Option<&Self>,
        implied_m: Option<&Self>,
        step: Option<f64>,
        max_iter: u32,
    ) -> Result<f64> {
        let (h_min, h_max) = h_range;
        let fixed_m = fixed_m.map(MetricField::metric);
        let implied_m = implied_m.map(MetricField::metric);

        debug!(
            "Scaling the metric (h_min = {h_min}, h_max = {h_max}, n_elems = {n_elems}, max_iter = {max_iter})"
        );
        if let Some(fixed_m) = fixed_m {
            debug!("Using a fixed metric");
            let c = self.complexity_iter(fixed_m.par_iter().cloned(), h_min, h_max);
            debug!("Complexity of the fixed metric: {c}");
        }
        if let Some(implied_m) = implied_m {
            debug!(
                "Using the implied metric with step = {}",
                step.unwrap_or(4.0)
            );
            let c = self.complexity_iter(implied_m.par_iter().cloned(), h_min, h_max);
            debug!("Complexity of the implied metric: {c}");
        }

        let mut scale = if max_iter > 0 {
            self.scale_metric_simple(h_min, h_max, n_elems, max_iter)
        } else {
            1.0
        };
        if scale < 0.0 {
            return Err(Error::from("Unable to scale the metric (simple)"));
        }

        if fixed_m.is_some() || implied_m.is_some() {
            let fixed_m = (0..self.msh.n_verts())
                .into_par_iter()
                .map(|i| fixed_m.map(|x| &x[i]));
            let implied_m = (0..self.msh.n_verts())
                .into_par_iter()
                .map(|i| implied_m.map(|x| &x[i]));

            if max_iter > 0 {
                let constrain_m = fixed_m.clone().zip(implied_m.clone()).map(|(m_f, m_i)| {
                    Self::get_bounded_metric(0.0, h_min, h_max, None, m_f, step, m_i)
                });
                let constrain_c = self.complexity_iter(constrain_m, h_min, h_max);

                debug!(
                    "Complexity of the constrain metric: {constrain_c}, target complexity: {n_elems}"
                );

                if constrain_c > n_elems as f64 {
                    return Err(Error::from(&format!(
                        "The complexity of the constrain metric is {constrain_c:.2e} > n_elems = {n_elems}"
                    )));
                }

                let m_iter = |s: f64| {
                    self.metric
                        .par_iter()
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

                    if c < n_elems as f64 {
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

                    if c > n_elems as f64 {
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
                    if f64::abs(c - n_elems as f64) < 0.05 * n_elems as f64 {
                        break;
                    }
                    if iter == max_iter - 1 {
                        return Err(Error::from("Unable to scale the metric (bisection)"));
                    }
                    if c < n_elems as f64 {
                        scale_high = scale;
                    } else {
                        scale_low = scale;
                    }
                }
            }
            self.metric
                .par_iter_mut()
                .zip(fixed_m.clone())
                .zip(implied_m.clone())
                .for_each(|((m, m_f), m_i)| {
                    *m = Self::get_bounded_metric(scale, h_min, h_max, Some(m), m_f, step, m_i);
                });
        } else {
            for m in &mut self.metric {
                m.scale_with_bounds(scale, h_min, h_max);
            }
        }

        Ok(scale)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        Result,
        mesh::test_meshes::{test_mesh_2d, test_mesh_3d},
        metric::{AnisoMetric2d, AnisoMetric3d, IsoMetric, MetricField},
    };
    use tmesh::{Vert2d, Vert3d, mesh::Mesh};

    #[test]
    fn test_scaling_2d() {
        let mesh = test_mesh_2d().split().split();

        let h = vec![0.1; mesh.n_verts()];
        let m: Vec<_> = h.iter().map(|&x| IsoMetric::<2>::from(x)).collect();
        let mut m = MetricField::new(&mesh, m);

        let c0 = m.scale((0.0, 0.05), 1000, None, None, None, 10).unwrap();
        assert!(c0 > 0.0);
        let c1 = m.complexity(0.0, 0.05);
        assert!(f64::abs(c1 - 1000.) < 100.);
    }

    #[test]
    fn test_scaling_2d_aniso() {
        let mesh = test_mesh_2d().split().split();

        let mfunc = |_p| {
            let v0 = Vert2d::new(0.5, 0.);
            let v1 = Vert2d::new(0.0, 4.0);
            AnisoMetric2d::from_sizes(&v0, &v1)
        };

        let m: Vec<_> = mesh.verts().map(mfunc).collect();
        let mut m = MetricField::new(&mesh, m);

        let c0 = m.scale((0.0, 0.05), 1000, None, None, None, 10).unwrap();
        assert!(c0 > 0.0);
        let c1 = m.complexity(0.0, 0.05);
        assert!(f64::abs(c1 - 1000.) < 100.);
    }

    #[test]
    fn test_scaling_3d() -> Result<()> {
        let mesh = test_mesh_3d().split().split();

        let h = vec![0.1; mesh.n_verts()];
        let m: Vec<_> = h.iter().map(|&x| IsoMetric::<3>::from(x)).collect();
        let mut m = MetricField::new(&mesh, m);

        let c0 = m.scale((-0.0, 0.05), 1000, None, None, None, 10);
        assert!(c0.is_err());

        let n_target = (1.0 / f64::powi(0.05, 3) * 15.0) as usize;
        let c0 = m.scale((0.0, 0.05), n_target, None, None, None, 10)?;
        assert!(c0 > 0.0);
        let c1 = m.complexity(0.0, 0.05);
        assert!(f64::abs(c1 - n_target as f64) < 0.1 * n_target as f64);

        Ok(())
    }

    #[test]
    fn test_scaling_3d_aniso() -> Result<()> {
        let mesh = test_mesh_3d().split().split();

        let mfunc = |_p| {
            let v0 = Vert3d::new(0.5, 0., 0.);
            let v1 = Vert3d::new(0.0, 4.0, 0.);
            let v2 = Vert3d::new(0.0, 0., 6.0);
            AnisoMetric3d::from_sizes(&v0, &v1, &v2)
        };

        let m: Vec<_> = mesh.verts().map(mfunc).collect();
        let mut m = MetricField::new(&mesh, m);

        let c0 = m.scale((0.0, 0.05), 1000, None, None, None, 10);
        assert!(c0.is_err());

        let n_target = (1.0 / f64::powi(0.05, 3) * 50.0) as usize;
        let c0 = m.scale((0.0, 0.05), n_target, None, None, None, 10)?;
        assert!(c0 > 0.0);
        let c1 = m.complexity(0.0, 0.05);
        assert!(f64::abs(c1 - n_target as f64) < 0.1 * n_target as f64);
        Ok(())
    }

    #[test]
    fn test_scaling_3d_fixed() -> Result<()> {
        let mesh = test_mesh_3d().split().split();

        let h = vec![0.1; mesh.n_verts()];
        let m: Vec<_> = h.iter().map(|&x| IsoMetric::<3>::from(x)).collect();
        let mut m = MetricField::new(&mesh, m);

        let fixed_m: Vec<_> = mesh
            .verts()
            .map(|p| IsoMetric::<3>::from(0.1 + p[0] + p[1]))
            .collect();
        let fixed_m = MetricField::new(&mesh, fixed_m);

        let n_target = (1.0 / f64::powi(0.05, 3) * 15.0) as usize;

        let c0 = m.scale((0.0, 0.05), n_target, Some(&fixed_m), None, None, 10)?;
        assert!(c0 > 0.0);
        let c1 = m.complexity(0.0, 0.05);
        assert!(f64::abs(c1 - n_target as f64) < 0.1 * n_target as f64);

        Ok(())
    }
}
