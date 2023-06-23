use crate::{
    geom_elems::GElem,
    geometry::LinearGeometry,
    mesh::{Point, SimplexMesh},
    metric::{AnisoMetric2d, AnisoMetric3d, Metric},
    topo_elems::{Elem, Tetrahedron, Triangle},
    Error, Idx, Mesh, Result, Tag,
};

use log::{debug, info, warn};
use rustc_hash::FxHashSet;

impl<const D: usize, E: Elem> SimplexMesh<D, E> {
    /// Get the metric information (min/max size, max anisotropy, complexity)
    pub fn metric_info<M: Metric<D>>(&self, m: &[M]) -> (f64, f64, f64, f64) {
        let (h_min, h_max, aniso_max) =
            m.iter()
                .map(|m| m.sizes())
                .fold((f64::MAX, 0.0, 0.0), |(a, b, c), d| {
                    (
                        f64::min(a, d[0]),
                        f64::max(b, d[2]),
                        f64::max(c, d[2] / d[0]),
                    )
                });

        (
            h_min,
            h_max,
            aniso_max,
            self.complexity(m.iter().copied(), 0.0, f64::MAX),
        )
    }

    /// Convert a metric field defined at the element centers (P0) to a field defined at the vertices (P1)
    /// using the interpolation method appropriate for the metric type.
    /// vertex-to-element connectivity and volumes are required
    pub fn elem_data_to_vertex_data_metric<M: Metric<D>>(&self, v: &[M]) -> Result<Vec<M>> {
        debug!("Convert metric element data to vertex data");
        if self.vertex_to_elems.is_none() {
            return Err(Error::from("vertex to element connectivity not computed"));
        }
        if self.elem_vol.is_none() {
            return Err(Error::from("element volumes not computed"));
        }
        if self.vert_vol.is_none() {
            return Err(Error::from("node volumes not computed"));
        }

        let n_elems = self.n_elems() as usize;
        let n_verts = self.n_verts() as usize;
        assert_eq!(v.len() % n_elems, 0);

        let mut res = Vec::with_capacity(n_verts);

        let v2e = self.vertex_to_elems.as_ref().unwrap();
        let elem_vol = self.elem_vol.as_ref().unwrap();
        let node_vol = self.vert_vol.as_ref().unwrap();

        let mut weights = Vec::new();
        let mut metrics = Vec::new();
        for (i_vert, vert_vol) in node_vol.iter().copied().enumerate() {
            let elems = v2e.row(i_vert as Idx);
            let n_elems = elems.len();
            weights.reserve(n_elems);
            weights.clear();
            weights.extend(
                elems
                    .iter()
                    .map(|i| elem_vol[*i as usize] / f64::from(E::N_VERTS) / vert_vol),
            );
            metrics.reserve(n_elems);
            metrics.clear();
            metrics.extend(elems.iter().map(|&i| v[i as usize]));
            let wm = weights.iter().copied().zip(metrics.iter());
            let m = M::interpolate(wm);
            res.push(m);
        }

        Ok(res)
    }

    /// Convert a metric field defined at the element centers (P0) to a field defined at the vertices (P1)
    /// using the interpolation method appropriate for the metric type.
    /// vertex-to-element connectivity and volumes are required
    pub fn vertex_data_to_elem_data_metric<M: Metric<D>>(&self, v: &[M]) -> Result<Vec<M>> {
        debug!("Convert metric vertex data to element data");

        let n_elems = self.n_elems() as usize;
        let n_verts = self.n_verts() as usize;
        assert_eq!(v.len() % n_verts, 0);

        let mut res = Vec::with_capacity(n_elems);

        let mut weights = Vec::new();
        let mut metrics = Vec::new();
        let f = 1. / f64::from(E::N_VERTS);

        for e in self.elems() {
            weights.reserve(n_elems);
            weights.clear();
            weights.resize(E::N_VERTS as usize, f);
            metrics.reserve(n_elems);
            metrics.clear();
            metrics.extend(e.iter().map(|&i| v[i as usize]));
            let wm = weights.iter().copied().zip(metrics.iter());
            let m = M::interpolate(wm);
            res.push(m);
        }
        Ok(res)
    }

    /// Compute the number of elements corresponding to a metric field based on its D characteristic sizes and min/max constraints
    #[must_use]
    fn complexity_from_sizes<M: Metric<D>>(&self, sizes: &[f64], h_min: f64, h_max: f64) -> f64 {
        assert!(self.vert_vol.is_some());
        let vols = self.vert_vol.as_ref().unwrap();

        vols.iter()
            .enumerate()
            .map(|(i, v)| {
                let s = &sizes[D * i..D * (i + 1)];
                let vol = s
                    .iter()
                    .fold(1.0, |a, b| a * f64::min(h_max, f64::max(h_min, *b)));
                v / (E::Geom::<D, M>::IDEAL_VOL * vol)
            })
            .sum::<f64>()
    }

    /// Compute the number of elements corresponding to a metric field taking into account min/max size constraints
    /// The volume, in euclidian space, of an element that is ideal (i.e. equilateral) in metric space is
    /// ```math
    /// v(\mathcal M) = v_\Delta V(M)
    /// ```
    /// where
    /// ```math
    /// v_\Delta \equiv \frac{1}{d!}\sqrt{\frac{d+1}{2^n}}
    /// ```
    /// is the volume of an ideal element in dimension $`d`$ and
    /// $`V(\mathcal M) = \det(\mathcal M)^{-1/2}`$ is the metric volume
    /// The ideal number of elements to fill the domain is therefore
    /// ```math
    /// \mathcal C = \int \frac{1}{v(\mathcal M)} dx = \frac{1}{v_\Delta} \int \sqrt{\det(\mathcal M)} dx
    /// ```
    /// TODO: the complexity is actually $`\int \sqrt{\det(\mathcal M)} dx`$
    pub fn complexity<I: Iterator<Item = M>, M: Metric<D>>(
        &self,
        m: I,
        h_min: f64,
        h_max: f64,
    ) -> f64 {
        let sizes: Vec<_> = m.flat_map(|x| x.sizes()).collect();
        self.complexity_from_sizes::<M>(&sizes, h_min, h_max)
    }

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
            sizes.iter_mut().for_each(|x| *x *= fac);

            let c = if iter == 0 {
                self.complexity_from_sizes::<M>(&sizes, 0.0, f64::MAX)
            } else {
                self.complexity_from_sizes::<M>(&sizes, h_min, h_max)
            };
            debug!("Iteration {}, complexity = {}", iter, c);
            if f64::abs(c - f64::from(n_elems)) < 0.05 * f64::from(n_elems) {
                return scale;
            }
            if iter == max_iter - 1 {
                warn!(
                    "Target complexity {} not reached: complexity {}",
                    n_elems, c
                );
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
            res.limit(m_i, step.unwrap_or(4.0));
        }
        res
        // let mut res = m;
        // res.scale_with_bounds(alpha, h_min, h_max);
        // res = m.intersect(m_f);
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
        info!(
            "Scaling the metric (h_min = {}, h_max = {}, n_elems = {}, max_iter = {})",
            h_min, h_max, n_elems, max_iter
        );
        if fixed_m.is_some() {
            info!("Using a fixed metric");
        }
        if implied_m.is_some() {
            info!(
                "Using the implied metric with step = {}",
                step.unwrap_or(4.0)
            );
        }

        let mut scale = self.scale_metric_simple(m, h_min, h_max, n_elems, max_iter);
        if scale < 0.0 {
            return Err(Error::from("Unable to scale the metric (simple)"));
        }

        if fixed_m.is_some() || implied_m.is_some() {
            let fixed_m = (0..self.n_verts()).map(|i| fixed_m.map(|x| &x[i as usize]));
            let implied_m = (0..self.n_verts()).map(|i| implied_m.map(|x| &x[i as usize]));

            let constrain_m = fixed_m.clone().zip(implied_m.clone()).map(|(m_f, m_i)| {
                Self::get_bounded_metric(0.0, h_min, h_max, None, m_f, step, m_i)
            });
            let constrain_c = self.complexity(constrain_m, h_min, h_max);

            debug!("Complexity of the constrain metric: {}", constrain_c);

            if constrain_c > n_elems as f64 {
                return Err(Error::from(&format!(
                    "The complexity of the constrain metric is {} > n_elems = {}",
                    constrain_c, n_elems
                )));
            }

            let m_iter = |s: f64| {
                m.iter()
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
                let c = self.complexity(tmp_m, h_min, h_max);
                debug!(
                    "Iteration {}: scale_high = {}, complexity = {}",
                    iter, scale_high, c
                );

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
                let c = self.complexity(tmp_m, h_min, h_max);
                debug!(
                    "Iteration {}: scale_low = {}, complexity = {}",
                    iter, scale_low, c
                );

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
                let c = self.complexity(tmp_m, h_min, h_max);
                debug!("Iteration {}: scale = {}, complexity = {}", iter, scale, c);
                if f64::abs(c - f64::from(n_elems)) < 0.05 * f64::from(n_elems) {
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
            m.iter_mut()
                .zip(fixed_m.clone())
                .zip(implied_m.clone())
                .for_each(|((m, m_f), m_i)| {
                    *m = Self::get_bounded_metric(scale, h_min, h_max, Some(m), m_f, step, m_i)
                });
        } else {
            m.iter_mut()
                .for_each(|m| m.scale_with_bounds(scale, h_min, h_max));
        }

        Ok(scale)
    }

    /// Smooth a metric field to avoid numerical artifacts
    /// For each mesh vertex $`i`$, a set a suitable neighbors $`N(i)`$ is built as
    /// a subset of the neighbors of $`i`$ ($`i`$ is included) ignoring the vertices with the metrics with
    /// the smallest and largest metric volume.
    /// The smoothed metric field is then computed as the average (i.e. interpolation
    /// with equal weights) of the metrics in $`N(i)`$
    /// , the metric is replaced by the average
    /// on its neighbors ignoring the metrics with the minimum and maximum volumes
    /// TODO: doc
    pub fn smooth_metric<M: Metric<D>>(&self, m: &[M]) -> Result<Vec<M>> {
        if self.vertex_to_vertices.is_none() {
            return Err(Error::from("vertex to vertex connection not available"));
        }

        info!("Apply metric smoothing");

        let v2v = self.vertex_to_vertices.as_ref().unwrap();
        let mut weights = Vec::new();
        let mut metrics = Vec::new();
        let mut res = Vec::with_capacity(m.len());

        for i_vert in 0..self.n_verts() {
            let m_v = &m[i_vert as usize];
            let vol = m_v.vol();
            let mut min_vol = vol;
            let mut max_vol = vol;
            let mut min_idx = 0;
            let mut max_idx = 0;
            let neighbors = v2v.row(i_vert);
            for i_neigh in neighbors {
                let m_n = &m[*i_neigh as usize];
                let vol = m_n.vol();
                if vol < min_vol {
                    min_vol = vol;
                    min_idx = i_neigh + 1;
                } else if vol > max_vol {
                    max_vol = vol;
                    max_idx = i_neigh + 1;
                }
            }

            weights.clear();
            metrics.clear();
            let n = if min_idx == max_idx {
                neighbors.len()
            } else {
                neighbors.len() - 1
            };
            let w = 1. / n as f64;

            if min_idx != 0 && max_idx != 0 {
                weights.push(w);
                metrics.push(&m[i_vert as usize]);
            }
            for i_neigh in neighbors {
                if min_idx != i_neigh + 1 && max_idx != i_neigh + 1 {
                    weights.push(w);
                    metrics.push(&m[*i_neigh as usize]);
                }
            }

            let m_smooth = M::interpolate(weights.iter().copied().zip(metrics.iter().copied()));
            res.push(m_smooth);
        }

        Ok(res)
    }

    /// Compute the gradation on an edge
    fn edge_gradation<M: Metric<D>>(&self, m: &[M], i0: Idx, i1: Idx) -> f64 {
        let m0 = &m[i0 as usize];
        let m1 = &m[i1 as usize];
        let e = self.vert(i1) - self.vert(i0);
        let l0 = m0.length(&e);
        let l1 = m1.length(&e);
        let a = l0 / l1;
        if f64::abs(a - 1.0) < 1e-3 {
            1.0
        } else {
            let l = l0 * f64::ln(a) / (a - 1.0);
            f64::max(a, 1.0 / a).powf(1. / l).min(100.0)
        }
    }

    /// Compute the maximum metric gradation and the fraction of edges with a gradation
    /// higher than an threshold
    pub fn gradation<M: Metric<D>>(&self, m: &[M], target: f64) -> Result<(f64, f64)> {
        if self.edges.is_none() {
            return Err(Error::from("edges not available"));
        }

        let edges = self.edges.as_ref().unwrap();

        let n_edgs = edges.len() / 2;

        let mut c_max = 0.0;
        let mut n_larger_than_target = 0;
        for edg in edges.chunks(2) {
            let i0 = edg[0];
            let i1 = edg[1];
            let c = self.edge_gradation(m, i0, i1);
            if c > target {
                n_larger_than_target += 1;
            }
            c_max = f64::max(c_max, c);
        }

        Ok((c_max, n_larger_than_target as f64 / n_edgs as f64))
    }

    /// Enforce a maximum gradiation on a metric field
    /// Algorithm taken from "Size gradation control of anisotropic meshes", F. Alauzet, 2010 assuming
    ///  - a linear interpolation on h
    ///  - physical-space-gradation (eq. 10)
    pub fn apply_metric_gradation<M: Metric<D>>(
        &self,
        m: &mut [M],
        beta: f64,
        max_iter: Idx,
    ) -> Result<(Idx, Idx)> {
        if self.edges.is_none() {
            return Err(Error::from("edges not available"));
        }

        info!(
            "Apply metric gradation (beta = {}, max_iter = {})",
            beta, max_iter
        );

        let edges = self.edges.as_ref().unwrap();

        let n_edgs = edges.len() / 2;
        let mut n = 0;
        for iter in 0..max_iter {
            n = 0;
            let mut c = Vec::with_capacity(n_edgs);
            c.extend(
                edges
                    .chunks(2)
                    .map(|edg| self.edge_gradation(m, edg[0], edg[1])),
            );
            // argsort
            let mut indices = Vec::with_capacity(c.len());
            indices.extend(0..c.len());
            indices.sort_by(|j, i| c[*i].partial_cmp(&c[*j]).unwrap());
            for i_edge in indices {
                if c[i_edge] < beta {
                    break;
                }
                let i0 = edges[2 * i_edge] as usize;
                let i1 = edges[2 * i_edge + 1] as usize;

                let e = self.vert(i1 as Idx) - self.vert(i0 as Idx);
                let m0 = m[i0];
                let m1 = m[i1];

                let m01 = m0.span(&e, beta);
                m[i1] = m1.intersect(&m01);

                let m10 = m1.span(&e, beta);
                m[i0] = m0.intersect(&m10);
                if m0.differs_from(&m[i0], 1e-8) || m1.differs_from(&m[i1], 1e-8) {
                    n += 1;
                }
            }
            debug!("Iteration {}, {}/{} edges modified", iter, n, n_edgs);
            if n == 0 {
                break;
            }
        }

        if n > 0 {
            let (c_max, frac_large_gradation) = self.gradation(m, beta)?;
            warn!(
                "gradation: target not achieved: max gradation: {}, {}% of edges have a gradation > {}",
                c_max,
                frac_large_gradation * 100.0,
                beta
            );
        }

        Ok((n, n_edgs as Idx))
    }
}

impl SimplexMesh<3, Tetrahedron> {
    /// Compute the element-implied metric
    pub fn implied_metric(&self) -> Result<Vec<AnisoMetric3d>> {
        let mut implied_metric = Vec::with_capacity(self.n_elems() as usize);
        implied_metric.extend(self.gelems().map(|ge| ge.implied_metric()));
        self.elem_data_to_vertex_data_metric(&implied_metric)
    }

    /// Compute an anisotropic metric based on the boundary curvature
    /// - geom : the geometry on which the curvature is computed
    /// - r_h: the curvature radius to element size ratio
    /// - beta: the mesh gradation
    /// - h_n: the normal size, defined at the boundary vertices
    ///   if <0, the min of the tangential sizes is used
    pub fn curvature_metric(
        &self,
        geom: &LinearGeometry<3, Triangle>,
        r_h: f64,
        beta: f64,
        h_n: Option<&[f64]>,
        h_n_tags: Option<&[Tag]>,
    ) -> Result<Vec<AnisoMetric3d>> {
        info!(
            "Compute the curvature metric with r/h = {} and gradation = {}",
            r_h, beta
        );

        if self.vertex_to_vertices.is_none() {
            return Err(Error::from("vertex to vertices connectivity not available"));
        }

        let (bdy, boundary_vertex_ids) = self.boundary();
        let bdy_tags: FxHashSet<Tag> = bdy.etags().collect();

        // Initialize the metric field
        let hx = Point::<3>::new(1.0, 0.0, 0.0);
        let hy = Point::<3>::new(0.0, 1.0, 0.0);
        let hz = Point::<3>::new(0.0, 0.0, 1.0);
        let m = AnisoMetric3d::from_sizes(&hx, &hy, &hz);
        let n_verts = self.n_verts() as usize;
        let mut curvature_metric = vec![m; n_verts];
        let mut flg = vec![false; n_verts];

        // Set the metric at the boundary vertices
        for tag in bdy_tags {
            let (_, ids, _, _) = bdy.extract(tag);
            let use_h_n = if let Some(h_n_tags) = h_n_tags {
                h_n_tags.iter().any(|&t| t == tag)
            } else {
                false
            };
            ids.iter()
                .map(|&i| (i, boundary_vertex_ids[i as usize]))
                .for_each(|(i_bdy_vert, i_vert)| {
                    let pt = self.vert(i_vert);
                    let (mut u, mut v) = geom.curvature(&pt, tag).unwrap();
                    let hu = 1. / (r_h * u.norm());
                    let hv = 1. / (r_h * v.norm());
                    let mut hn = f64::min(hu, hv);
                    if use_h_n {
                        if let Some(h_n) = h_n {
                            assert!(h_n[i_bdy_vert as usize] > 0.0);
                            hn = h_n[i_bdy_vert as usize].min(hn);
                        }
                    }
                    u.normalize_mut();
                    v.normalize_mut();
                    let n = hn * u.cross(&v);
                    u *= hu;
                    v *= hv;

                    curvature_metric[i_vert as usize] = AnisoMetric3d::from_sizes(&n, &u, &v);
                    flg[i_vert as usize] = true;
                });
        }

        // Extend the metric into the volume
        let mut to_fix = n_verts - boundary_vertex_ids.len();
        debug!("{} / {} internal vertices to fix", to_fix, n_verts);

        let v2v = self.vertex_to_vertices.as_ref().unwrap();

        let mut n_iter = 0;
        loop {
            let mut fixed = Vec::with_capacity(to_fix);
            for (i_vert, pt) in self.verts().enumerate() {
                if !flg[i_vert] {
                    let neighbors = v2v.row(i_vert as Idx);
                    let mut valid_neighbors =
                        neighbors.iter().copied().filter(|&i| flg[i as usize]);
                    if let Some(i) = valid_neighbors.next() {
                        let m_i = curvature_metric[i as usize];
                        let pt_i = self.vert(i);
                        let e = pt - pt_i;
                        let mut m = m_i.span(&e, beta);
                        for i in valid_neighbors {
                            let m_i = curvature_metric[i as usize];
                            let pt_i = self.vert(i);
                            let e = pt - pt_i;
                            let m_i_spanned = m_i.span(&e, beta);
                            m = m.intersect(&m_i_spanned);
                        }
                        curvature_metric[i_vert] = m;
                        to_fix -= 1;
                        fixed.push(i_vert);
                    }
                }
            }
            if fixed.is_empty() {
                // No element was fixed
                if to_fix > 0 {
                    warn!(
                        "stop at iteration {}, {} elements cannot be fixed",
                        n_iter + 1,
                        to_fix
                    );
                }
                break;
            }
            fixed.iter().for_each(|&i| flg[i] = true);
            n_iter += 1;

            debug!(
                "iteration {}: {} / {} vertices remain to be fixed",
                n_iter, to_fix, n_verts
            );
        }

        Ok(curvature_metric)
    }
}

impl SimplexMesh<2, Triangle> {
    /// Compute the element-implied metric
    pub fn implied_metric(&self) -> Result<Vec<AnisoMetric2d>> {
        let mut implied_metric = Vec::with_capacity(self.n_elems() as usize);
        implied_metric.extend(self.gelems().map(|ge| ge.implied_metric()));
        self.elem_data_to_vertex_data_metric(&implied_metric)
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::Matrix3;

    use crate::{
        mesh::Point,
        metric::{AnisoMetric2d, AnisoMetric3d, IsoMetric, Metric},
        min_iter,
        test_meshes::{test_mesh_2d, test_mesh_3d},
        Idx, Mesh, Result,
    };

    #[test]
    fn test_complexity_2d() {
        let mut mesh = test_mesh_2d().split().split();
        mesh.compute_volumes();

        let h = vec![0.1; mesh.n_verts() as usize];
        let mut m: Vec<_> = (0..mesh.n_verts())
            .map(|i| IsoMetric::<2>::from_slice(&h, i))
            .collect();

        let c = mesh.complexity(m.iter().copied(), 0.0, f64::MAX);
        assert!(f64::abs(c - 100. * 4. / f64::sqrt(3.0)) < 1e-6);

        let c = mesh.complexity(m.iter().copied(), 0.0, 0.05);
        assert!(f64::abs(c - 400. * 4. / f64::sqrt(3.0)) < 1e-6);

        let c0 = mesh
            .scale_metric(&mut m, 0.0, 0.05, 1000, None, None, None, 10)
            .unwrap();
        assert!(c0 > 0.0);
        let c1 = mesh.complexity(m.iter().copied(), 0.0, 0.05);
        assert!(f64::abs(c1 - 1000.) < 100.);
    }

    #[test]
    fn test_complexity_2d_aniso() {
        let mut mesh = test_mesh_2d().split().split();
        mesh.compute_volumes();

        let mfunc = |_p| {
            let v0 = Point::<2>::new(0.5, 0.);
            let v1 = Point::<2>::new(0.0, 4.0);
            AnisoMetric2d::from_sizes(&v0, &v1)
        };

        let mut m: Vec<_> = mesh.verts().map(mfunc).collect();

        let c = mesh.complexity(m.iter().copied(), 0.0, f64::MAX);
        assert!(f64::abs(c - 0.5 * 4. / f64::sqrt(3.0)) < 1e-6);

        let c = mesh.complexity(m.iter().copied(), 1., 3.);
        assert!(f64::abs(c - 1.0 / 3.0 * 4. / f64::sqrt(3.0)) < 1e-6);

        let c0 = mesh
            .scale_metric(&mut m, 0.0, 0.05, 1000, None, None, None, 10)
            .unwrap();
        assert!(c0 > 0.0);
        let c1 = mesh.complexity(m.iter().copied(), 0.0, 0.05);
        assert!(f64::abs(c1 - 1000.) < 100.);
    }

    #[test]
    fn test_complexity_3d() -> Result<()> {
        let mut mesh = test_mesh_3d().split().split();
        mesh.compute_volumes();

        let h = vec![0.1; mesh.n_verts() as usize];
        let mut m: Vec<_> = (0..mesh.n_verts())
            .map(|i| IsoMetric::<3>::from_slice(&h, i))
            .collect();

        let c = mesh.complexity(m.iter().copied(), 0.0, f64::MAX);
        assert!(f64::abs(c - 1000. * 6. * f64::sqrt(2.0)) < 1e-6);

        let c = mesh.complexity(m.iter().copied(), 0.0, 0.05);
        assert!(f64::abs(c - 8000. * 6. * f64::sqrt(2.0)) < 1e-6);

        let c0 = mesh.scale_metric(&mut m, 0.0, 0.05, 1000, None, None, None, 10);
        assert!(c0.is_err());

        let n_target = (1.0 / f64::powi(0.05, 3) * 15.0) as Idx;
        let c0 = mesh.scale_metric(&mut m, 0.0, 0.05, n_target, None, None, None, 10)?;
        assert!(c0 > 0.0);
        let c1 = mesh.complexity(m.iter().copied(), 0.0, 0.05);
        assert!(f64::abs(c1 - f64::from(n_target)) < 0.1 * f64::from(n_target));

        Ok(())
    }

    #[test]
    fn test_complexity_3d_aniso() -> Result<()> {
        let mut mesh = test_mesh_3d().split().split();
        mesh.compute_volumes();

        let mfunc = |_p| {
            let v0 = Point::<3>::new(0.5, 0., 0.);
            let v1 = Point::<3>::new(0.0, 4.0, 0.);
            let v2 = Point::<3>::new(0.0, 0., 6.0);
            AnisoMetric3d::from_sizes(&v0, &v1, &v2)
        };

        let mut m: Vec<_> = mesh.verts().map(mfunc).collect();

        let c = mesh.complexity(m.iter().copied(), 0.0, f64::MAX);
        assert!(f64::abs(c - 1.0 / 12.0 * 6. * f64::sqrt(2.0)) < 1e-6);

        let c = mesh.complexity(m.iter().copied(), 1.0, 5.0);
        assert!(f64::abs(c - 1.0 / 20. * 6. * f64::sqrt(2.0)) < 1e-6);

        let c0 = mesh.scale_metric(&mut m, 0.0, 0.05, 1000, None, None, None, 10);
        assert!(c0.is_err());

        let n_target = (1.0 / f64::powi(0.05, 3) * 50.0) as Idx;
        let c0 = mesh.scale_metric(&mut m, 0.0, 0.05, n_target, None, None, None, 10)?;
        assert!(c0 > 0.0);
        let c1 = mesh.complexity(m.iter().copied(), 0.0, 0.05);
        assert!(f64::abs(c1 - f64::from(n_target)) < 0.1 * f64::from(n_target));
        Ok(())
    }

    #[test]
    fn test_complexity_3d_fixed() -> Result<()> {
        let mut mesh = test_mesh_3d().split().split();
        mesh.compute_volumes();

        let h = vec![0.1; mesh.n_verts() as usize];
        let mut m: Vec<_> = (0..mesh.n_verts())
            .map(|i| IsoMetric::<3>::from_slice(&h, i))
            .collect();
        let fixed_m: Vec<_> = mesh
            .verts()
            .map(|p| IsoMetric::<3>::from(0.1 + p[0] + p[1]))
            .collect();

        let n_target = (1.0 / f64::powi(0.05, 3) * 15.0) as Idx;

        let c0 = mesh.scale_metric(&mut m, 0.0, 0.05, n_target, Some(&fixed_m), None, None, 10)?;
        assert!(c0 > 0.0);
        let c1 = mesh.complexity(m.iter().copied(), 0.0, 0.05);
        assert!(f64::abs(c1 - f64::from(n_target)) < 0.1 * f64::from(n_target));

        Ok(())
    }

    #[test]
    fn test_smooth_2d() {
        let mut mesh = test_mesh_2d().split().split();
        mesh.compute_vertex_to_vertices();

        let mut m: Vec<_> = (0..mesh.n_verts())
            .map(|_| IsoMetric::<2>::from(0.1))
            .collect();

        m[2] = IsoMetric::<2>::from(0.01);
        m[5] = IsoMetric::<2>::from(1.);

        let m = mesh.smooth_metric(&m).unwrap();

        let vmin = min_iter(m.iter().map(Metric::vol));
        let vmax = min_iter(m.iter().map(Metric::vol));

        assert!(f64::abs(vmin - 0.01) < 1e-6);
        assert!(f64::abs(vmax - 0.01) < 1e-6);
    }

    #[test]
    fn test_smooth_2d_aniso() {
        let mut mesh = test_mesh_2d().split().split();
        mesh.compute_volumes();

        mesh.compute_vertex_to_vertices();

        let v0 = Point::<2>::new(0.5, 0.);
        let v1 = Point::<2>::new(0.0, 4.0);

        let mut m: Vec<_> = (0..mesh.n_verts())
            .map(|_| AnisoMetric2d::from_sizes(&v0, &v1))
            .collect();

        let v0 = Point::<2>::new(0.05, 0.);
        let v1 = Point::<2>::new(0.0, 4.0);
        m[2] = AnisoMetric2d::from_sizes(&v0, &v1);

        let v0 = Point::<2>::new(0.5, 0.);
        let v1 = Point::<2>::new(0.0, 40.0);
        m[5] = AnisoMetric2d::from_sizes(&v0, &v1);

        let m = mesh.smooth_metric(&m).unwrap();

        let vmin = min_iter(m.iter().map(Metric::vol));
        let vmax = min_iter(m.iter().map(Metric::vol));

        assert!(f64::abs(vmin - 2.0) < 1e-6);
        assert!(f64::abs(vmax - 2.0) < 1e-6);
    }

    #[test]
    fn test_smooth_3d() {
        let mut mesh = test_mesh_3d().split().split();
        mesh.compute_vertex_to_vertices();

        let mut m: Vec<_> = (0..mesh.n_verts())
            .map(|_| IsoMetric::<3>::from(0.1))
            .collect();

        m[2] = IsoMetric::<3>::from(0.01);
        m[5] = IsoMetric::<3>::from(1.);

        let m = mesh.smooth_metric(&m).unwrap();

        let vmin = min_iter(m.iter().map(Metric::vol));
        let vmax = min_iter(m.iter().map(Metric::vol));

        assert!(f64::abs(vmin - 0.001) < 1e-6);
        assert!(f64::abs(vmax - 0.001) < 1e-6);
    }

    #[test]
    fn test_smooth_3d_aniso() {
        let mut mesh = test_mesh_3d().split().split();
        mesh.compute_volumes();

        mesh.compute_vertex_to_vertices();

        let v0 = Point::<3>::new(0.5, 0.0, 0.0);
        let v1 = Point::<3>::new(0.0, 4.0, 0.0);
        let v2 = Point::<3>::new(0.0, 0.0, 0.1);

        let mut m: Vec<_> = (0..mesh.n_verts())
            .map(|_| AnisoMetric3d::from_sizes(&v0, &v1, &v2))
            .collect();

        let v0 = Point::<3>::new(0.05, 0.0, 0.0);
        let v1 = Point::<3>::new(0.0, 4.0, 0.0);
        let v2 = Point::<3>::new(0.0, 0.0, 0.1);
        m[2] = AnisoMetric3d::from_sizes(&v0, &v1, &v2);

        let v0 = Point::<3>::new(0.5, 0.0, 0.0);
        let v1 = Point::<3>::new(0.0, 4.0, 0.0);
        let v2 = Point::<3>::new(0.0, 0.0, 1.0);
        m[5] = AnisoMetric3d::from_sizes(&v0, &v1, &v2);

        let m = mesh.smooth_metric(&m).unwrap();

        let vmin = min_iter(m.iter().map(Metric::vol));
        let vmax = min_iter(m.iter().map(Metric::vol));

        assert!(f64::abs(vmin - 0.2) < 1e-6);
        assert!(f64::abs(vmax - 0.2) < 1e-6);
    }

    #[test]
    fn test_gradation_2d() {
        let mut mesh = test_mesh_2d().split();
        mesh.compute_edges();

        let mut m: Vec<_> = (0..mesh.n_verts())
            .map(|_| IsoMetric::<2>::from(0.1))
            .collect();
        m[0] = IsoMetric::<2>::from(0.0001);

        let beta = 1.2;
        let (c_max, frac_large_c) = mesh.gradation(&m, beta).unwrap();
        assert!(c_max > beta);
        assert!(frac_large_c > 0.0);

        let (n, _n_edgs) = mesh.apply_metric_gradation(&mut m, beta, 10).unwrap();
        assert_eq!(n, 0);

        let (c_max, frac_large_c) = mesh.gradation(&m, beta).unwrap();
        assert!(c_max < 1.001 * beta);
        assert!(frac_large_c < 1e-12);

        let edges = mesh.edges.as_ref().unwrap();
        let n_edgs = edges.len() / 2;
        for i_edge in 0..n_edgs {
            let i0 = edges[2 * i_edge] as usize;
            let i1 = edges[2 * i_edge + 1] as usize;

            let e = mesh.vert(i1 as Idx) - mesh.vert(i0 as Idx);

            let l = m[i0].length(&e);
            let rmax = 1.0 + l * f64::ln(beta);
            assert!(m[i1].h() < 1.0001 * m[i0].h() * rmax);

            let l = m[i1].length(&e);
            let rmax = 1.0 + l * f64::ln(beta);
            assert!(m[i0].h() < 1.0001 * m[i1].h() * rmax);
        }
    }

    #[test]
    fn test_implicit_metric() -> Result<()> {
        let mesh = test_mesh_3d();
        let mut mesh = mesh.split().split().split();

        let h0 = 10.;
        let h1 = 0.1;
        let h2 = 2.;

        let rot = Matrix3::new(
            1. / f64::sqrt(3.),
            1. / f64::sqrt(6.),
            -1. / f64::sqrt(2.),
            1. / f64::sqrt(3.),
            1. / f64::sqrt(6.),
            1. / f64::sqrt(2.),
            1. / f64::sqrt(3.),
            -2. / f64::sqrt(6.),
            0.,
        );

        let mut coords = Vec::with_capacity(mesh.coords.len());
        for mut pt in mesh.verts() {
            pt[0] *= h0;
            pt[1] *= h1;
            pt[2] *= h2;
            pt = rot * pt;
            coords.extend(pt.iter());
        }
        mesh.coords = coords;
        mesh.compute_vertex_to_elems();
        mesh.compute_volumes();

        let m = mesh.implied_metric()?;

        for m_i in m {
            let s = m_i.sizes();

            assert!(s[0] > 0.33 * h1 / 8. && s[0] < 3. * h1 / 8.);
            assert!(s[1] > 0.33 * h2 / 8. && s[1] < 3. * h2 / 8.);
            assert!(s[2] > 0.33 * h0 / 8. && s[2] < 3. * h0 / 8.);
        }

        Ok(())
    }
}
