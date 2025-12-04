use crate::metric::{Metric, MetricField};
use rayon::{
    prelude::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator},
    slice::ParallelSlice,
};
use tmesh::mesh::{GSimplex, Mesh, Simplex};

impl<const D: usize, M: Mesh<D>, T: Metric<D>> MetricField<'_, D, M, T> {
    /// Get the metric information (min/max size, max anisotropy, complexity)
    pub fn info(&self) -> (f64, f64, f64, f64) {
        let (h_min, h_max, aniso_max) = self
            .metric
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

        (h_min, h_max, aniso_max, self.complexity(0.0, f64::MAX))
    }

    /// Get the quality of all the mesh elements using a metric field
    #[must_use]
    pub fn qualities(&self) -> Vec<f64> {
        self.msh
            .par_elems()
            .map(|e| T::quality(&self.msh.gelem(&e), e.into_iter().map(|i| self.metric[i])))
            .collect()
    }

    /// Get the lengths of all the edges using a metric field
    #[must_use]
    pub fn edge_lengths(&self) -> Vec<f64> {
        let edgs = self.msh.edges();

        edgs.par_iter()
            .map(|(e, _)| {
                let i0 = e.get(0);
                let i1 = e.get(1);
                T::edge_length(
                    &self.msh.vert(i0),
                    &self.metric[i0],
                    &self.msh.vert(i1),
                    &self.metric[i1],
                )
            })
            .collect()
    }

    /// Compute the number of elements corresponding to a metric field based on its D characteristic sizes and min/max constraints
    #[must_use]
    pub fn complexity_from_sizes(&self, sizes: &[f64], h_min: f64, h_max: f64) -> f64 {
        let n_verts = self.msh.n_verts();
        assert_eq!(sizes.len(), n_verts * D);

        sizes
            .par_chunks(D)
            .zip(self.vols.par_iter())
            .map(|(s, v)| {
                let vol = s
                    .iter()
                    .fold(1.0, |a, b| a * f64::min(h_max, f64::max(h_min, *b)));
                v / (<M::C as Simplex>::GEOM::<D>::ideal_vol() * vol)
            })
            .sum::<f64>()
    }

    pub fn complexity_iter(
        &self,
        m: impl IndexedParallelIterator<Item = T>,
        h_min: f64,
        h_max: f64,
    ) -> f64 {
        let n_verts = self.msh.n_verts();
        assert_eq!(m.len(), n_verts);

        m.zip(self.vols.par_iter())
            .map(|(m, v)| {
                let s = m.sizes();
                let vol = s
                    .iter()
                    .fold(1.0, |a, b| a * f64::min(h_max, f64::max(h_min, *b)));
                v / (<M::C as Simplex>::GEOM::<D>::ideal_vol() * vol)
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
    #[must_use]
    pub fn complexity(&self, h_min: f64, h_max: f64) -> f64 {
        self.complexity_iter(self.metric.par_iter().cloned(), h_min, h_max)
    }
}

#[cfg(test)]
mod tests {
    use tmesh::{Vert2d, Vert3d, mesh::Mesh};

    use crate::{
        mesh::test_meshes::{test_mesh_2d, test_mesh_3d},
        metric::{AnisoMetric2d, AnisoMetric3d, IsoMetric, MetricField},
    };

    #[test]
    fn test_complexity_2d() {
        let mesh = test_mesh_2d().split().split();

        let h = vec![0.1; mesh.n_verts() as usize];
        let m: Vec<_> = h.iter().map(|&x| IsoMetric::<2>::from(x)).collect();

        let m = MetricField::new(&mesh, m);

        let c = m.complexity(0.0, f64::MAX);
        assert!(f64::abs(c - 100. * 4. / f64::sqrt(3.0)) < 1e-6);

        let c = m.complexity(0.0, 0.05);
        assert!(f64::abs(c - 400. * 4. / f64::sqrt(3.0)) < 1e-6);
    }

    #[test]
    fn test_complexity_2d_aniso() {
        let mesh = test_mesh_2d().split().split();

        let mfunc = |_p| {
            let v0 = Vert2d::new(0.5, 0.);
            let v1 = Vert2d::new(0.0, 4.0);
            AnisoMetric2d::from_sizes(&v0, &v1)
        };

        let m: Vec<_> = mesh.verts().map(mfunc).collect();
        let m = MetricField::new(&mesh, m);

        let c = m.complexity(0.0, f64::MAX);
        assert!(f64::abs(c - 0.5 * 4. / f64::sqrt(3.0)) < 1e-6);

        let c = m.complexity(1., 3.);
        assert!(f64::abs(c - 1.0 / 3.0 * 4. / f64::sqrt(3.0)) < 1e-6);
    }

    #[test]
    fn test_complexity_3d() {
        let mesh = test_mesh_3d().split().split();

        let h = vec![0.1; mesh.n_verts() as usize];
        let m: Vec<_> = h.iter().map(|&x| IsoMetric::<3>::from(x)).collect();
        let m = MetricField::new(&mesh, m);

        let c = m.complexity(0.0, f64::MAX);
        assert!(f64::abs(c - 1000. * 6. * f64::sqrt(2.0)) < 1e-6);

        let c = m.complexity(0.0, 0.05);
        assert!(f64::abs(c - 8000. * 6. * f64::sqrt(2.0)) < 1e-6);
    }

    #[test]
    fn test_complexity_3d_aniso() {
        let mesh = test_mesh_3d().split().split();

        let mfunc = |_p| {
            let v0 = Vert3d::new(0.5, 0., 0.);
            let v1 = Vert3d::new(0.0, 4.0, 0.);
            let v2 = Vert3d::new(0.0, 0., 6.0);
            AnisoMetric3d::from_sizes(&v0, &v1, &v2)
        };

        let m: Vec<_> = mesh.verts().map(mfunc).collect();
        let m = MetricField::new(&mesh, m);

        let c = m.complexity(0.0, f64::MAX);
        assert!(f64::abs(c - 1.0 / 12.0 * 6. * f64::sqrt(2.0)) < 1e-6);

        let c = m.complexity(1.0, 5.0);
        assert!(f64::abs(c - 1.0 / 20. * 6. * f64::sqrt(2.0)) < 1e-6);
    }
}
