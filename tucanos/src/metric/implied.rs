use crate::{
    Result,
    mesh::{GTetrahedron, GTriangle, SimplexMesh, Tetrahedron, Triangle},
    metric::{AnisoMetric2d, AnisoMetric3d},
};
use rayon::prelude::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};

use super::{IsoMetric, Metric};

pub trait HasImpliedMetric<const D: usize, M: Metric<D>>: Sized {
    type ImpliedMetricType: Metric<D> + Default + Send + Sync;
    fn calculate_implied_metric(&self) -> Self::ImpliedMetricType;
}

impl HasImpliedMetric<3, IsoMetric<3>> for GTetrahedron<3, IsoMetric<3>> {
    type ImpliedMetricType = AnisoMetric3d;
    fn calculate_implied_metric(&self) -> Self::ImpliedMetricType {
        self.implied_metric()
    }
}

impl HasImpliedMetric<2, IsoMetric<2>> for GTriangle<2, IsoMetric<2>> {
    type ImpliedMetricType = AnisoMetric2d;
    fn calculate_implied_metric(&self) -> Self::ImpliedMetricType {
        self.implied_metric()
    }
}

impl SimplexMesh<3, Tetrahedron> {
    /// Compute the element-implied metric
    pub fn implied_metric(&self) -> Result<Vec<AnisoMetric3d>> {
        let mut implied_metric = vec![AnisoMetric3d::default(); self.n_elems() as usize];

        implied_metric
            .par_iter_mut()
            .zip(self.par_gelems())
            .for_each(|(m, ge)| *m = ge.implied_metric());

        self.elem_data_to_vertex_data_metric(&implied_metric)
    }
}

impl SimplexMesh<2, Triangle> {
    /// Compute the element-implied metric
    pub fn implied_metric(&self) -> Result<Vec<AnisoMetric2d>> {
        let mut implied_metric = vec![AnisoMetric2d::default(); self.n_elems() as usize];

        implied_metric
            .par_iter_mut()
            .zip(self.par_gelems())
            .for_each(|(m, ge)| *m = ge.implied_metric());

        self.elem_data_to_vertex_data_metric(&implied_metric)
    }
}

#[cfg(test)]
mod tests {
    use crate::{Result, mesh::test_meshes::test_mesh_3d, metric::Metric};
    use nalgebra::Matrix3;
    use tmesh::mesh::Mesh;

    #[test]
    fn test_implied_metric() -> Result<()> {
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

        mesh.mut_verts().for_each(|p| {
            p[0] *= h0;
            p[1] *= h1;
            p[2] *= h2;
            *p = rot * (*p);
        });
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
