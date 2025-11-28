use crate::metric::{AnisoMetric, AnisoMetric2d, AnisoMetric3d, MetricField};
use nalgebra::{Matrix2, Matrix3};
use rayon::prelude::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};
use tmesh::mesh::{GTetrahedron, GTriangle, Idx, Mesh, Tetrahedron, Triangle};

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

/// Get the jacobian of the transformation from the reference to the current element
#[must_use]
fn tetrahedron_jacobian(ge: &GTetrahedron<3>) -> Matrix3<f64> {
    Matrix3::<f64>::new(
        ge[1][0] - ge[0][0],
        ge[2][0] - ge[0][0],
        ge[3][0] - ge[0][0],
        ge[1][1] - ge[0][1],
        ge[2][1] - ge[0][1],
        ge[3][1] - ge[0][1],
        ge[1][2] - ge[0][2],
        ge[2][2] - ge[0][2],
        ge[3][2] - ge[0][2],
    )
}

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
#[must_use]
pub fn tetrahedron_implied_metric(ge: &GTetrahedron<3>) -> AnisoMetric3d {
    let j = tetrahedron_jacobian(ge) * TETRA_J_EQ;
    let m = j * j.transpose();
    let m = m.try_inverse().unwrap();
    AnisoMetric3d::from_mat(m)
}

impl<'a, T: Idx, M: Mesh<3, C = Tetrahedron<T>>> MetricField<'a, 3, M, AnisoMetric3d> {
    /// Compute the element-implied metric
    pub fn implied_metric_3d(msh: &'a M) -> Self {
        let mut implied_metric = vec![AnisoMetric3d::default(); msh.n_elems()];

        implied_metric
            .par_iter_mut()
            .zip(msh.par_gelems())
            .for_each(|(m, ge)| *m = tetrahedron_implied_metric(&ge));

        Self::from_elem_metric(msh, &implied_metric)
    }
}

const TRI_J_EQ: Matrix2<f64> = Matrix2::new(1.0, -1. / SQRT_3, 0., 2. / SQRT_3);

/// Get the jacobian of the transformation from the reference to the current element
#[must_use]
fn triangle_jacobian(ge: &GTriangle<2>) -> Matrix2<f64> {
    Matrix2::<f64>::new(
        ge[1][0] - ge[0][0],
        ge[2][0] - ge[0][0],
        ge[1][1] - ge[0][1],
        ge[2][1] - ge[0][1],
    )
}

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
#[must_use]
pub fn triangle_implied_metric(ge: &GTriangle<2>) -> AnisoMetric2d {
    let j = triangle_jacobian(ge) * TRI_J_EQ;
    let m = j * j.transpose();
    let m = m.try_inverse().unwrap();
    AnisoMetric2d::from_mat(m)
}

impl<'a, T: Idx, M: Mesh<2, C = Triangle<T>>> MetricField<'a, 2, M, AnisoMetric2d> {
    /// Compute the element-implied metric
    pub fn implied_metric_2d(msh: &'a M) -> Self {
        let mut implied_metric = vec![AnisoMetric2d::default(); msh.n_elems()];

        implied_metric
            .par_iter_mut()
            .zip(msh.par_gelems())
            .for_each(|(m, ge)| *m = triangle_implied_metric(&ge));

        Self::from_elem_metric(msh, &implied_metric)
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

        let m = MetricField::implied_metric_3d(&mesh);
        let m = m.metric();

        for m_i in m {
            let s = m_i.sizes();

            assert!(s[0] > 0.33 * h1 / 8. && s[0] < 3. * h1 / 8.);
            assert!(s[1] > 0.33 * h2 / 8. && s[1] < 3. * h2 / 8.);
            assert!(s[2] > 0.33 * h0 / 8. && s[2] < 3. * h0 / 8.);
        }
    }
}
