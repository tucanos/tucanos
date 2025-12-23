use crate::metric::{AnisoMetric, AnisoMetric2d, AnisoMetric3d, Metric, MetricField};
use nalgebra::{Matrix2, Matrix3};
use rayon::prelude::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};
use tmesh::mesh::{GTetrahedron, GTriangle, Mesh, Simplex};

/// Trait of geometric simplices that can compute an implied metric
pub trait ImpliedMetric<T> {
    fn implied_metric(&self) -> T;
}

impl<'a, const D: usize, M, T> MetricField<'a, D, M, T>
where
    M: Mesh<D>,
    T: Metric<D> + Send + Sync + Default + Clone,
    <<M as Mesh<D>>::C as Simplex>::GEOM<D>: ImpliedMetric<T>,
{
    pub fn implied_metric(msh: &'a M) -> Self {
        let mut implied_metric = vec![T::default(); msh.n_elems()];

        implied_metric
            .par_iter_mut()
            .zip(msh.par_gelems())
            .for_each(|(m, ge)| *m = ge.implied_metric());

        Self::from_elem_metric(msh, &implied_metric)
    }
}

const SQRT_3: f64 = 1.732_050_807_568_877_2;
const SQRT_6: f64 = std::f64::consts::SQRT_2 * SQRT_3;

// Jacobian of the reference to equilateral transformation
const TETRA_J_EQ: Matrix3<f64> = Matrix3::new(
    1.0,
    -1. / SQRT_3,
    -1. / SQRT_6,
    0.,
    2. / SQRT_3,
    -1.0 / SQRT_6,
    0.,
    0.,
    3. / SQRT_6,
);

const TRI_J_EQ: Matrix2<f64> = Matrix2::new(1.0, -1. / SQRT_3, 0., 2. / SQRT_3);

impl ImpliedMetric<AnisoMetric2d> for GTriangle<2> {
    /// Compute the implied metric
    /// It can be computed using the Jacobian $`J`$ of the transformation from the
    /// reference unit-length element to the physical element as $`(J J^T)^{-1}`$ .
    /// $`J`$ can be decomposed as the product of
    ///  - the Jacobian $`J_0`$ of the transformation from the reference unit-length
    ///    element to the orthogonal element, stored as `Self::J_EQ`
    ///  - the Jacobian $`J_1`$ of the transformation from the orthogonal element to
    ///    the physical element
    ///
    /// (reference: Ph.D. P. Caplan, p. 35)
    fn implied_metric(&self) -> AnisoMetric2d {
        // Calculate Jacobian J_1 (Physical)
        let j_geo = Matrix2::from_fn(|r, c| self[c + 1][r] - self[0][r]);
        // J = J_1 * J_0 (Equilateral)
        let j = j_geo * TRI_J_EQ;
        let m = j * j.transpose();
        // Metric = (J J^T)^-1
        let m_inv = m.try_inverse().expect("Degenerate element encountered");
        AnisoMetric2d::from_mat(m_inv)
    }
}

impl ImpliedMetric<AnisoMetric3d> for GTetrahedron<3> {
    /// Compute the implied metric
    /// It can be computed using the Jacobian $`J`$ of the transformation from the
    /// reference unit-length element to the physical element as $`(J J^T)^{-1}`$ .
    /// $`J`$ can be decomposed as the product of
    ///  - the Jacobian $`J_0`$ of the transformation from the reference unit-length
    ///    element to the orthogonal element, stored as `Self::J_EQ`
    ///  - the Jacobian $`J_1`$ of the transformation from the orthogonal element to
    ///    the physical element
    ///
    /// (reference: Ph.D. P. Caplan, p. 35)
    fn implied_metric(&self) -> AnisoMetric3d {
        // Calculate Jacobian J_1 (Physical)
        let j_geo = Matrix3::from_fn(|r, c| self[c + 1][r] - self[0][r]);
        // J = J_1 * J_0 (Equilateral)
        let j = j_geo * TETRA_J_EQ;
        let m = j * j.transpose();
        // Metric = (J J^T)^-1
        let m_inv = m.try_inverse().expect("Degenerate element encountered");
        AnisoMetric3d::from_mat(m_inv)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        mesh::test_meshes::test_mesh_3d,
        metric::{Metric, MetricField},
    };
    use nalgebra::Matrix3;
    use tmesh::mesh::Mesh;

    #[test]
    fn test_implied_metric() {
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

        mesh.verts_mut().for_each(|p| {
            p[0] *= h0;
            p[1] *= h1;
            p[2] *= h2;
            *p = rot * (*p);
        });

        let m = MetricField::implied_metric(&mesh);
        let m = m.metric();

        for m_i in m {
            let s = m_i.sizes();

            assert!(s[0] > 0.33 * h1 / 8. && s[0] < 3. * h1 / 8.);
            assert!(s[1] > 0.33 * h2 / 8. && s[1] < 3. * h2 / 8.);
            assert!(s[2] > 0.33 * h0 / 8. && s[2] < 3. * h0 / 8.);
        }
    }
}
