use crate::{Result, mesh::SimplexMesh, metric::Metric};
use log::debug;
use rayon::{
    prelude::{
        IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator,
        ParallelIterator,
    },
    slice::ParallelSlice,
};
use tmesh::mesh::{GSimplex, Idx, Mesh, Simplex};

impl<T: Idx, const D: usize, C: Simplex<T>> SimplexMesh<T, D, C> {
    pub fn ideal_vol() -> f64 {
        if D == 2 && C::N_VERTS == 3 {
            3.0_f64.sqrt() / 4.0
        } else if D == 3 && C::N_VERTS == 4 {
            1.0 / (6.0 * std::f64::consts::SQRT_2)
        } else {
            unreachable!("Invalid dimension / element type");
        }
    }
    /// Get the quality of element $`K`$, defined as
    /// ```math
    /// q(K) = \beta_d \frac{|K|_{\mathcal M}^{2/d}}{\sum_e ||e||_{\mathcal M}^2}
    /// ```
    /// where
    ///  - $`|K|_{\mathcal M}`$ is the volume element in metric space. It is computed
    ///    on a discrete mesh as the ratio of the volume in physical space to the minimum
    ///    of the volumes of the metrics at each vertex
    ///  - the sum on the denominator is performed over all the edges of the element
    ///  - $`\beta_d`$ is a normalization factor such that $`q = 1`$ of equilateral elements
    ///     - $`\beta_2 = 1 / (6\sqrt{2}) `$
    ///     - $`\beta_3 = \sqrt{3}/ 4 `$
    pub fn quality<M: Metric<D>>(ge: &C::GEOM<D>, m: &C::DATA<M>) -> f64 {
        let m = M::min_metric(m.as_ref());
        let vol = ge.vol();

        let mut l = 0.0;
        for i in 0..C::N_EDGES {
            let e = ge.edge(i);
            l += f64::powi(m.length(&e), 2);
        }
        let l = l / 6.0;
        let vol = vol / m.vol() / Self::ideal_vol();

        f64::powf(vol, 2. / C::DIM as f64) / l
    }

    /// Get the metric information (min/max size, max anisotropy, complexity)
    pub fn metric_info<M: Metric<D>>(&self, m: &[M]) -> (f64, f64, f64, f64) {
        let (h_min, h_max, aniso_max) = m
            .par_iter()
            .map(Metric::sizes)
            .fold(
                || (f64::MAX, 0.0, 0.0),
                |(a, b, c), d| {
                    let dmin = d[0];
                    let dmax = *d.last().unwrap();
                    (
                        f64::min(a, dmin),
                        f64::max(b, dmax),
                        f64::max(c, dmax / dmin),
                    )
                },
            )
            .reduce(
                || (f64::MAX, 0.0, 0.0),
                |a, b| (f64::min(a.0, b.0), f64::max(a.1, b.1), f64::max(a.2, b.2)),
            );

        (h_min, h_max, aniso_max, self.complexity(m, 0.0, f64::MAX))
    }

    /// Get the quality of all the mesh elements using a metric field
    pub fn qualities<M: Metric<D>>(&self, m: &[M]) -> Vec<f64> {
        self.par_elems()
            .map(|e| {
                let ge = self.gelem(&e);
                let m = C::data_from_iter(e.into_iter().map(|i| m[i.try_into().unwrap()]));

                Self::quality(&ge, &m)
            })
            .collect()
    }

    /// Get the lengths of all the edges using a metric field
    pub fn edge_lengths<M: Metric<D>>(&self, m: &[M]) -> Result<Vec<f64>> {
        let edgs = self.get_edges()?;

        Ok(edgs
            .par_iter()
            .map(|&e| {
                M::edge_length(
                    &self.vert(e[0]),
                    &m[e[0].try_into().unwrap()],
                    &self.vert(e[1]),
                    &m[e[1].try_into().unwrap()],
                )
            })
            .collect())
    }

    /// Convert a metric field defined at the element centers (P0) to a field defined at the vertices (P1)
    /// using the interpolation method appropriate for the metric type.
    /// vertex-to-element connectivity and volumes are required
    pub fn elem_data_to_vertex_data_metric<M: Metric<D>>(&self, v: &[M]) -> Result<Vec<M>> {
        debug!("Convert metric element data to vertex data");

        let n_elems = self.n_elems().try_into().unwrap();
        let n_verts = self.n_verts().try_into().unwrap();
        assert_eq!(v.len(), n_elems);

        let mut res = vec![M::default(); n_verts];

        let v2e = self.get_vertex_to_elems()?;
        let elem_vol = self.get_elem_volumes()?;
        let node_vol = self.get_vertex_volumes()?;

        res.par_iter_mut()
            .zip(node_vol.par_iter())
            .enumerate()
            .for_each(|(i_vert, (m_vert, vert_vol))| {
                let elems = v2e.row(i_vert.try_into().unwrap());
                let n_elems = elems.len();
                let mut weights = Vec::with_capacity(n_elems);
                let mut metrics = Vec::with_capacity(n_elems);
                weights.extend(
                    elems
                        .iter()
                        .map(|&i| elem_vol[i.try_into().unwrap()] / C::N_VERTS as f64 / vert_vol),
                );
                metrics.extend(elems.iter().map(|&i| v[i.try_into().unwrap()]));
                let wm = weights.iter().copied().zip(metrics.iter());
                *m_vert = M::interpolate(wm);
            });

        Ok(res)
    }

    /// Convert a metric field defined at the element centers (P0) to a field defined at the vertices (P1)
    /// using the interpolation method appropriate for the metric type.
    /// vertex-to-element connectivity and volumes are required
    pub fn vertex_data_to_elem_data_metric<M: Metric<D>>(&self, v: &[M]) -> Result<Vec<M>> {
        debug!("Convert metric vertex data to element data");

        let n_elems = self.n_elems().try_into().unwrap();
        let n_verts = self.n_verts().try_into().unwrap();
        assert_eq!(v.len(), n_verts);

        let mut res = vec![M::default(); n_elems];

        let f = 1. / C::N_VERTS as f64;

        res.par_iter_mut()
            .zip(self.par_elems())
            .for_each(|(m_elem, e)| {
                let mut weights = Vec::with_capacity(C::N_VERTS.try_into().unwrap());
                let mut metrics = Vec::with_capacity(C::N_VERTS.try_into().unwrap());
                weights.resize(C::N_VERTS.try_into().unwrap(), f);
                metrics.extend(e.into_iter().map(|i| v[i.try_into().unwrap()]));
                let wm = weights.iter().copied().zip(metrics.iter());
                *m_elem = M::interpolate(wm);
            });
        Ok(res)
    }

    /// Compute the number of elements corresponding to a metric field based on its D characteristic sizes and min/max constraints
    #[must_use]
    pub fn complexity_from_sizes<M: Metric<D>>(
        &self,
        sizes: &[f64],
        h_min: f64,
        h_max: f64,
    ) -> f64 {
        let n_verts = self.n_verts().try_into().unwrap();
        assert_eq!(sizes.len(), n_verts * D);

        let vols = self.get_vertex_volumes().unwrap();

        sizes
            .par_chunks(D)
            .zip(vols.par_iter())
            .map(|(s, v)| {
                let vol = s
                    .iter()
                    .fold(1.0, |a, b| a * f64::min(h_max, f64::max(h_min, *b)));
                v / (Self::ideal_vol() * vol)
            })
            .sum::<f64>()
    }

    pub fn complexity_iter<M: Metric<D>, I: IndexedParallelIterator<Item = M>>(
        &self,
        m: I,
        h_min: f64,
        h_max: f64,
    ) -> f64 {
        let n_verts = self.n_verts().try_into().unwrap();
        assert_eq!(m.len(), n_verts);

        let vols = self.get_vertex_volumes().unwrap();

        m.zip(vols.par_iter())
            .map(|(m, v)| {
                let s = m.sizes();
                let vol = s
                    .iter()
                    .fold(1.0, |a, b| a * f64::min(h_max, f64::max(h_min, *b)));
                v / (Self::ideal_vol() * vol)
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
    pub fn complexity<M: Metric<D>>(&self, m: &[M], h_min: f64, h_max: f64) -> f64 {
        self.complexity_iter(m.par_iter().cloned(), h_min, h_max)
    }
}

#[cfg(test)]
mod tests {
    use tmesh::{Vert2d, Vert3d, mesh::Mesh};

    use crate::{
        mesh::test_meshes::{test_mesh_2d, test_mesh_3d},
        metric::{AnisoMetric2d, AnisoMetric3d, IsoMetric},
    };

    #[test]
    fn test_complexity_2d() {
        let mut mesh = test_mesh_2d().split().split();
        mesh.compute_volumes();

        let h = vec![0.1; mesh.n_verts().try_into().unwrap()];
        let m: Vec<_> = h.iter().map(|&x| IsoMetric::<2>::from(x)).collect();

        let c = mesh.complexity(&m, 0.0, f64::MAX);
        assert!(f64::abs(c - 100. * 4. / f64::sqrt(3.0)) < 1e-6);

        let c = mesh.complexity(&m, 0.0, 0.05);
        assert!(f64::abs(c - 400. * 4. / f64::sqrt(3.0)) < 1e-6);
    }

    #[test]
    fn test_complexity_2d_aniso() {
        let mut mesh = test_mesh_2d().split().split();
        mesh.compute_volumes();

        let mfunc = |_p| {
            let v0 = Vert2d::new(0.5, 0.);
            let v1 = Vert2d::new(0.0, 4.0);
            AnisoMetric2d::from_sizes(&v0, &v1)
        };

        let m: Vec<_> = mesh.verts().map(mfunc).collect();

        let c = mesh.complexity(&m, 0.0, f64::MAX);
        assert!(f64::abs(c - 0.5 * 4. / f64::sqrt(3.0)) < 1e-6);

        let c = mesh.complexity(&m, 1., 3.);
        assert!(f64::abs(c - 1.0 / 3.0 * 4. / f64::sqrt(3.0)) < 1e-6);
    }

    #[test]
    fn test_complexity_3d() {
        let mut mesh = test_mesh_3d().split().split();
        mesh.compute_volumes();

        let h = vec![0.1; mesh.n_verts().try_into().unwrap()];
        let m: Vec<_> = h.iter().map(|&x| IsoMetric::<3>::from(x)).collect();

        let c = mesh.complexity(&m, 0.0, f64::MAX);
        assert!(f64::abs(c - 1000. * 6. * f64::sqrt(2.0)) < 1e-6);

        let c = mesh.complexity(&m, 0.0, 0.05);
        assert!(f64::abs(c - 8000. * 6. * f64::sqrt(2.0)) < 1e-6);
    }

    #[test]
    fn test_complexity_3d_aniso() {
        let mut mesh = test_mesh_3d().split().split();
        mesh.compute_volumes();

        let mfunc = |_p| {
            let v0 = Vert3d::new(0.5, 0., 0.);
            let v1 = Vert3d::new(0.0, 4.0, 0.);
            let v2 = Vert3d::new(0.0, 0., 6.0);
            AnisoMetric3d::from_sizes(&v0, &v1, &v2)
        };

        let m: Vec<_> = mesh.verts().map(mfunc).collect();

        let c = mesh.complexity(&m, 0.0, f64::MAX);
        assert!(f64::abs(c - 1.0 / 12.0 * 6. * f64::sqrt(2.0)) < 1e-6);

        let c = mesh.complexity(&m, 1.0, 5.0);
        assert!(f64::abs(c - 1.0 / 20. * 6. * f64::sqrt(2.0)) < 1e-6);
    }
}
