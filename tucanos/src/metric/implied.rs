use crate::{
    Result,
    mesh::SimplexMesh,
    metric::{AnisoMetric, AnisoMetric2d, AnisoMetric3d},
};
use nalgebra::{Matrix2, Matrix3};
use rayon::prelude::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};
use tmesh::mesh::{GTetrahedron, GTriangle, Idx, Mesh, Tetrahedron, Triangle};

const SQRT_3: f64 = 1.732_050_807_568_877_2;
const SQRT_6: f64 = std::f64::consts::SQRT_2 * SQRT_3;
impl<T: Idx> SimplexMesh<T, 3, Tetrahedron<T>> {
    const J_EQ: Matrix3<f64> = Matrix3::new(
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
    fn jacobian(ge: &GTetrahedron<3>) -> Matrix3<f64> {
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
    pub fn gelem_implied_metric(ge: &GTetrahedron<3>) -> AnisoMetric3d {
        let j = Self::jacobian(ge) * Self::J_EQ;
        let m = j * j.transpose();
        let m = m.try_inverse().unwrap();
        AnisoMetric3d::from_mat(m)
    }

    /// Compute the element-implied metric
    pub fn implied_metric(&self) -> Result<Vec<AnisoMetric3d>> {
        let mut implied_metric = vec![AnisoMetric3d::default(); self.n_elems().try_into().unwrap()];

        implied_metric
            .par_iter_mut()
            .zip(self.par_gelems())
            .for_each(|(m, ge)| *m = Self::gelem_implied_metric(&ge));

        self.elem_data_to_vertex_data_metric(&implied_metric)
    }
}

impl<T: Idx> SimplexMesh<T, 2, Triangle<T>> {
    const J_EQ: Matrix2<f64> = Matrix2::new(1.0, -1. / SQRT_3, 0., 2. / SQRT_3);

    /// Get the jacobian of the transformation from the reference to the current element
    pub fn jacobian(ge: &GTriangle<2>) -> Matrix2<f64> {
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
    pub fn gelem_implied_metric(ge: &GTriangle<2>) -> AnisoMetric2d {
        let j = Self::jacobian(ge) * Self::J_EQ;
        let m = j * j.transpose();
        let m = m.try_inverse().unwrap();
        AnisoMetric2d::from_mat(m)
    }

    /// Compute the element-implied metric
    pub fn implied_metric(&self) -> Result<Vec<AnisoMetric2d>> {
        let mut implied_metric = vec![AnisoMetric2d::default(); self.n_elems().try_into().unwrap()];

        implied_metric
            .par_iter_mut()
            .zip(self.par_gelems())
            .for_each(|(m, ge)| *m = Self::gelem_implied_metric(&ge));

        self.elem_data_to_vertex_data_metric(&implied_metric)
    }
}

#[cfg(test)]
mod tests {
    use crate::{Result, mesh::test_meshes::test_mesh_3d, metric::Metric};
    use nalgebra::Matrix3;
    use tmesh::mesh::{Mesh, MutMesh};

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

        mesh.verts_mut().for_each(|p| {
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
