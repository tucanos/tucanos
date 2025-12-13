use crate::Result;
use crate::metric::{
    IsoMetric, Metric,
    reduction::{control_step, simultaneous_reduction, step},
};
use crate::{S_MAX, S_MIN, S_RATIO_MAX};
use nalgebra::allocator::Allocator;
use nalgebra::{Const, DefaultAllocator, SMatrix, SVector};
use std::array::IntoIter;
use std::fmt;
use std::fmt::Debug;
use std::fmt::Display;
use std::ops::Index;
use tmesh::Vertex;

pub trait AnisoMetric<const D: usize>: Metric<D> + Index<usize, Output = f64> + Default
where
    Const<D>: nalgebra::ToTypenum + nalgebra::DimSub<nalgebra::U1>,
    DefaultAllocator: Allocator<<Const<D> as nalgebra::DimSub<nalgebra::U1>>::Output>,
{
    const N: usize;

    fn from_iso(iso: &IsoMetric<D>) -> Self;

    fn slice_to_mat(m: &[f64]) -> SMatrix<f64, D, D>;

    fn from_mat_and_vol(mat: SMatrix<f64, D, D>, vol: f64) -> Self;

    fn update_from_mat_and_vol(&mut self, mat: SMatrix<f64, D, D>, vol: f64);

    fn length_sqr(&self, e: &Vertex<D>) -> f64;

    /// Get a matrix representation of the metric
    fn as_mat(&self) -> SMatrix<f64, D, D>;

    fn vol_aniso(&self) -> f64;

    fn bound_eigenvalues(eigs: &mut SVector<f64, D>) {
        let mut s_max: f64 = 0.0;
        eigs.iter_mut().for_each(|s| {
            *s = s.clamp(S_MIN, S_MAX);
            s_max = s_max.max(*s);
        });

        let s_min = s_max / S_RATIO_MAX;
        eigs.iter_mut().for_each(|s| {
            *s = s.max(s_min);
        });
    }

    /// Initialise an isotropic metric from a symmetric matrix $`M`$ as
    /// ```math
    /// \mathcal M = |M|
    /// ```
    ///
    /// NB: A threshold is applied to the eigenvalues of $`\mathcal M`$.
    #[must_use]
    fn from_mat(mat: SMatrix<f64, D, D>) -> Self {
        let mut eig = mat.symmetric_eigen();
        // Ensure that the metric is valid, i.e. that all the eigenvalues are >0
        eig.eigenvalues.iter_mut().for_each(|i| *i = i.abs());
        Self::bound_eigenvalues(&mut eig.eigenvalues);
        let mat = eig.recompose();
        let vol = 1. / eig.eigenvalues.iter().product::<f64>().sqrt();
        debug_assert!(vol > 0.0);
        Self::from_mat_and_vol(mat, vol)
    }

    fn is_diagonal(&self, tol: f64) -> bool {
        let on_diag = (0..D).map(|i| self[i].abs()).sum::<f64>();
        let off_diag = (D..<Self as Metric<D>>::N)
            .map(|i| self[i].abs())
            .sum::<f64>();
        off_diag < 1e10 * f64::MIN_POSITIVE || tol * on_diag > off_diag
    }

    fn is_near_zero(&self, tol: f64) -> bool {
        self.into_iter().map(f64::abs).sum::<f64>() < tol
    }

    fn from_diagonal(s: &[f64]) -> Self;

    fn scale_aniso(&mut self, s: f64);
}

impl Display for AnisoMetric3d {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mat = self.as_mat();
        writeln!(f, "M = {mat:?}")?;

        let eig = mat.symmetric_eigen();
        for i in 0..3 {
            writeln!(
                f,
                "--> h = {}, {:?}",
                1. / eig.eigenvalues[i].sqrt(),
                eig.eigenvectors.row(i).clone()
            )?;
        }

        let vol = 1. / eig.eigenvalues.iter().product::<f64>().sqrt();
        writeln!(f, "vol = {vol}")?;
        Ok(())
    }
}

impl<const D: usize, T: AnisoMetric<D>> Metric<D> for T
where
    Const<D>: nalgebra::ToTypenum + nalgebra::DimSub<nalgebra::U1>,
    DefaultAllocator: Allocator<<Const<D> as nalgebra::DimSub<nalgebra::U1>>::Output>,
{
    const N: usize = <Self as AnisoMetric<D>>::N;

    fn from_slice(m: &[f64]) -> Self {
        let mat = Self::slice_to_mat(m);
        Self::from_mat(mat)
    }

    /// For an anisotropic metric, the metric space length is
    /// ```math
    /// l_\mathcal M(e) =  \sqrt{e^T \mathcal M e}
    /// ```
    fn length(&self, e: &Vertex<D>) -> f64 {
        AnisoMetric::length_sqr(self, e).sqrt()
    }

    fn length_sqr(&self, e: &Vertex<D>) -> f64 {
        AnisoMetric::length_sqr(self, e)
    }

    /// For an anisotropic metric, the volume is
    /// ```math
    /// V(\mathcal M) =  \frac{1}{\sqrt{\det(\mathcal M)}}
    /// ```
    fn vol(&self) -> f64 {
        self.vol_aniso()
    }

    fn check(&self) -> Result<()> {
        let eig = self.as_mat().symmetric_eigen();

        let eps = 1e-8;
        let mut s_max: f64 = 0.0;
        let mut s_min: f64 = S_MAX;

        for s in eig.eigenvalues.iter().copied() {
            assert!(s > (1.0 - eps) * S_MIN, "s < S_MIN");
            assert!(s < (1.0 + eps) * S_MAX, "s > S_MAX");
            s_max = s_max.max(s);
            s_min = s_min.min(s);
        }
        assert!(
            s_max / s_min < (1.0 + eps) * S_RATIO_MAX,
            "aniso > ANISO_MAX"
        );
        Ok(())
    }

    /// In order to ensure that the resulting matrix is >0, we compute the interpolation as
    /// ```math
    /// \mathcal M(\sum \alpha_i v_i) = \exp\left(\sum \alpha _i \ln(\mathcal M(v_i))\right)
    /// ```
    ///
    fn interpolate<'a, I: Iterator<Item = (f64, &'a Self)>>(weights_and_metrics: I) -> Self
    where
        Self: 'a,
    {
        let mut mat = Self::slice_to_mat(&[0.0; 6]);

        for (w, m) in weights_and_metrics {
            let mut eig = m.as_mat().symmetric_eigen();
            eig.eigenvalues
                .iter_mut()
                .for_each(|i| *i = w * libm::log((*i).max(S_MIN)));
            assert!(eig.eigenvalues.iter().all(|&x| f64::is_finite(x)));
            mat += eig.recompose();
        }

        let mut eig = mat.symmetric_eigen();
        eig.eigenvalues.iter_mut().for_each(|i| *i = libm::exp(*i));
        Self::bound_eigenvalues(&mut eig.eigenvalues);
        assert!(
            eig.eigenvalues.iter().all(|&x| f64::is_finite(x)),
            "{:?}",
            eig.eigenvalues
        );
        assert!(eig.eigenvalues.iter().all(|&x| x > 0.0));
        mat = eig.recompose();
        let vol = 1. / eig.eigenvalues.iter().product::<f64>().sqrt();

        Self::from_mat_and_vol(mat, vol)
    }

    /// The sizes associated with metric $`\mathcal M`$ are given by $`\lambda_i ^{-1/2}`$ where
    /// the $`\lambda_i`$ are the eigenvalues of $`\mathcal M`$
    fn sizes(&self) -> [f64; D] {
        let eig = self.as_mat().symmetric_eigen();
        let mut s = [0.; D];
        eig.eigenvalues
            .iter()
            .enumerate()
            .for_each(|(i, e)| s[i] = 1. / e.max(S_MIN).sqrt());
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        s
    }

    fn scale_with_bounds(&mut self, s: f64, h_min: f64, h_max: f64) {
        let s = 1. / (s * s);
        let s_min = 1. / (h_max * h_max);
        let s_max = 1. / (h_min * h_min);

        let mat = self.as_mat();
        let mut eig = mat.symmetric_eigen();
        eig.eigenvalues
            .iter_mut()
            .for_each(|i| *i = f64::min(s_max, f64::max(s_min, s * (*i))));

        let mat = eig.recompose();
        let vol = 1. / eig.eigenvalues.iter().product::<f64>().sqrt();

        self.update_from_mat_and_vol(mat, vol);
    }

    /// The intersection of metrics $`\mathcal M_0`$ and $`\mathcal M_1`$ is obtained
    /// using the simulataneous reduction algorithm
    fn intersect(&self, other: &Self) -> Self {
        let tol = 1e-8;
        if self.is_diagonal(tol) && other.is_diagonal(tol) {
            let mut s = [0.0; D];
            for i in 0..D {
                s[i] = f64::max(self[i], other[i]);
            }
            Self::from_diagonal(&s)
        } else if self.is_near_zero(1e-16) {
            *other
        } else if other.is_near_zero(1e-16) {
            *self
        } else {
            let (res, det) = simultaneous_reduction(self.as_mat(), other.as_mat());
            Self::from_mat_and_vol(res, 1. / det.sqrt())
        }
    }

    /// Span a metric using progression $`\beta`$ using physical-space-gradation
    /// (see "Size gradation control of anisotropic meshes", F. Alauzet, 2010 and
    /// "Feature-based and goal-oriented anisotropic mesh adaptation for RANS
    /// applications in aeronautics and aerospace", F. Alauzet & L. Frazza, 2021
    ///
    fn span(&self, e: &Vertex<D>, beta: f64, t: f64) -> Self {
        let nrm = e.norm();
        let mat = self.as_mat();
        let mut eig = mat.symmetric_eigen();
        let eta_0 = 1. + self.length(e) * f64::ln(beta);
        let eta_0 = eta_0.powf(1.0 - t);
        eig.eigenvalues.iter_mut().for_each(|s| {
            let eta_1 = 1.0 + f64::sqrt(*s) * nrm * f64::ln(beta);
            let eta_1 = eta_1.powf(t);
            let eta = eta_0 * eta_1;
            *s /= eta * eta;
        });
        Self::bound_eigenvalues(&mut eig.eigenvalues);
        let mat = eig.recompose();
        let vol = 1. / eig.eigenvalues.iter().fold(1.0, |v, &e| v * e).sqrt();

        Self::from_mat_and_vol(mat, vol)
    }

    fn differs_from(&self, other: &Self, tol: f64) -> bool {
        self.into_iter()
            .zip(*other)
            .any(|(x, y)| f64::abs(x - y) > tol * x)
    }

    fn step(&self, other: &Self) -> (f64, f64) {
        step(self.as_mat(), other.as_mat())
    }

    fn control_step(&mut self, other: &Self, f: f64) {
        let res = control_step(other.as_mat(), self.as_mat(), f);
        if let Some(res) = res {
            *self = Self::from_mat(res);
        }
    }

    fn scale(&mut self, s: f64) {
        self.scale_aniso(s);
    }
}

/// Anisotropic metric in 2D, represented with 3 scalars $`(x_0,x_1,x_2)`$
/// For the storage, we follow VTK: the symmetric matrix is
/// ```math
/// \begin{bmatrix}
/// x_0 & x_2\\
/// x_2 & x_1\\
/// \end{bmatrix}
/// ```
/// NB: the matrix must be positive definite for the metric to be valid
/// TODO: reuse the eigenvalue solvers ?
#[derive(Clone, Copy, Debug)]
pub struct AnisoMetric2d {
    m: [f64; 3],
    v: f64,
}

impl AnisoMetric2d {
    /// Convert from a matrix to 3 scalars
    fn mat_to_slice(mat: SMatrix<f64, 2, 2>) -> [f64; 3] {
        [mat[(0, 0)], mat[(1, 1)], mat[(0, 1)]]
    }

    /// Create a metric from 2 orthogonal vectors
    /// The length of the vectors will be the characteric length along this direction
    #[must_use]
    pub fn from_sizes(s0: &Vertex<2>, s1: &Vertex<2>) -> Self {
        let n0 = s0.norm();
        let n1 = s1.norm();
        let s0 = s0 / n0;
        let s1 = s1 / n1;
        assert!(s0.dot(&s1) < 1e-12);

        let mut eigvals = SVector::<f64, 2>::new(1. / n0.powi(2), 1. / n1.powi(2));
        Self::bound_eigenvalues(&mut eigvals);
        let eigvals = SMatrix::<f64, 2, 2>::from_diagonal(&eigvals);
        let eigvecs = SMatrix::<f64, 2, 2>::new(s0[0], s0[1], s1[0], s1[1]);
        let mat = eigvals * eigvecs;
        let mat = eigvecs.tr_mul(&mat);

        Self::from_mat(mat)
    }

    #[must_use]
    pub fn from_meshb(x: [f64; 3]) -> Self {
        Self::from_slice(&[x[0], x[2], x[1]])
    }

    #[must_use]
    pub const fn to_meshb(&self) -> [f64; 3] {
        let x = &self.m;
        [x[0], x[2], x[1]]
    }
}

impl Default for AnisoMetric2d {
    fn default() -> Self {
        Self {
            m: [S_MIN, S_MIN, 0.],
            v: (S_MIN.powi(2)),
        }
    }
}

impl AnisoMetric<2> for AnisoMetric2d {
    const N: usize = 3;

    fn slice_to_mat(m: &[f64]) -> SMatrix<f64, 2, 2> {
        SMatrix::<f64, 2, 2>::new(m[0], m[2], m[2], m[1])
    }

    fn from_mat_and_vol(mat: SMatrix<f64, 2, 2>, vol: f64) -> Self {
        Self {
            m: Self::mat_to_slice(mat),
            v: vol,
        }
    }

    fn update_from_mat_and_vol(&mut self, mat: SMatrix<f64, 2, 2>, vol: f64) {
        self.m = Self::mat_to_slice(mat);
        self.v = vol;
    }

    fn as_mat(&self) -> SMatrix<f64, 2, 2> {
        Self::slice_to_mat(&self.m)
    }

    fn from_iso(iso: &IsoMetric<2>) -> Self {
        let s = 1. / (iso.0 * iso.0);
        Self {
            m: [s, s, 0.0],
            v: iso.0.powi(2),
        }
    }

    fn from_diagonal(s: &[f64]) -> Self {
        Self {
            m: [s[0], s[1], 0.0],
            v: 1. / (s[0] * s[1]).sqrt(),
        }
    }

    fn vol_aniso(&self) -> f64 {
        self.v
    }

    fn scale_aniso(&mut self, s: f64) {
        for i in 0..3 {
            self.m[i] *= s;
        }
        self.v /= f64::sqrt(f64::powi(s, 2));
    }

    fn length_sqr(&self, e: &Vertex<2>) -> f64 {
        // In debug mode this implementation is much faster than nalgrebra's matrix-vector
        // multiplication. In release mode it's the same.
        let m = &self.m;
        let e = &e.data.0[0];
        // (m * e).dot(e)
        let p = [m[0] * e[0] + m[2] * e[1], m[2] * e[0] + m[1] * e[1]];
        p[0] * e[0] + p[1] * e[1]
    }
}

impl IntoIterator for AnisoMetric2d {
    type Item = f64;
    type IntoIter = IntoIter<f64, 3>;

    fn into_iter(self) -> Self::IntoIter {
        self.m.into_iter()
    }
}

impl Index<usize> for AnisoMetric2d {
    type Output = f64;

    fn index(&self, index: usize) -> &Self::Output {
        &self.m[index]
    }
}

impl Display for AnisoMetric2d {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mat = self.as_mat();
        writeln!(f, "M = {mat:?}")?;

        let eig = mat.symmetric_eigen();
        for i in 0..2 {
            writeln!(
                f,
                "--> h = {}, {:?}",
                1. / eig.eigenvalues[i].sqrt(),
                eig.eigenvectors.row(i).clone()
            )?;
        }

        let vol = 1. / eig.eigenvalues.iter().product::<f64>().sqrt();
        writeln!(f, "vol = {vol}")?;
        Ok(())
    }
}

/// Anisotropic metric in 3D, represented with 6 scalars $`(x_0,..., x_5)`$
/// For the storage, we follow VTK: the symmetric matrix is
/// ```math
/// \begin{bmatrix}
/// x_0& x_3& x_5\\
/// x_3& x_1& x_4\\
/// x_5& x_4& x_2\\
/// \end{bmatrix}
/// ```
/// NB: the matrix must be positive definite for the metric to be valid
/// TODO: reuse the eigenvalue solvers?
#[derive(Clone, Copy, Debug)]
pub struct AnisoMetric3d {
    m: [f64; 6],
    v: f64,
}

impl AnisoMetric3d {
    /// Convert from a matrix to 6 scalars
    #[must_use]
    pub fn mat_to_slice(mat: SMatrix<f64, 3, 3>) -> [f64; 6] {
        [
            mat[(0, 0)],
            mat[(1, 1)],
            mat[(2, 2)],
            mat[(0, 1)],
            mat[(1, 2)],
            mat[(0, 2)],
        ]
    }

    /// Create a metric from 2 orthogonal vectors
    /// The length of the vectors will be the characteric length along this direction
    #[must_use]
    pub fn from_sizes(s0: &Vertex<3>, s1: &Vertex<3>, s2: &Vertex<3>) -> Self {
        let n0 = s0.norm();
        let n1 = s1.norm();
        let n2 = s2.norm();
        let s0 = s0 / n0;
        let s1 = s1 / n1;
        let s2 = s2 / n2;
        assert!(s0.dot(&s1) < 1e-12);
        assert!(s0.dot(&s2) < 1e-12);
        assert!(s1.dot(&s2) < 1e-12);

        let mut eigvals = SVector::<f64, 3>::new(1. / n0.powi(2), 1. / n1.powi(2), 1. / n2.powi(2));
        Self::bound_eigenvalues(&mut eigvals);
        let eigvals = SMatrix::<f64, 3, 3>::from_diagonal(&eigvals);

        let eigvecs = SMatrix::<f64, 3, 3>::new(
            s0[0], s0[1], s0[2], s1[0], s1[1], s1[2], s2[0], s2[1], s2[2],
        );
        let mat = eigvals * eigvecs;
        let mat = eigvecs.tr_mul(&mat);

        Self::from_mat(mat)
    }

    #[must_use]
    pub fn from_meshb(x: [f64; 6]) -> Self {
        Self::from_slice(&[x[0], x[2], x[5], x[1], x[4], x[3]])
    }

    #[must_use]
    pub const fn to_meshb(&self) -> [f64; 6] {
        let x = &self.m;
        [x[0], x[3], x[1], x[5], x[4], x[2]]
    }
}

impl Default for AnisoMetric3d {
    fn default() -> Self {
        Self {
            m: [S_MIN, S_MIN, S_MIN, 0.0, 0.0, 0.0],
            v: (S_MIN.powi(3)),
        }
    }
}

impl AnisoMetric<3> for AnisoMetric3d {
    const N: usize = 6;

    /// Convert from 6 scalars to a matrix
    fn slice_to_mat(m: &[f64]) -> SMatrix<f64, 3, 3> {
        SMatrix::<f64, 3, 3>::new(m[0], m[3], m[5], m[3], m[1], m[4], m[5], m[4], m[2])
    }

    fn from_mat_and_vol(mat: SMatrix<f64, 3, 3>, vol: f64) -> Self {
        Self {
            m: Self::mat_to_slice(mat),
            v: vol,
        }
    }

    fn update_from_mat_and_vol(&mut self, mat: SMatrix<f64, 3, 3>, vol: f64) {
        self.m = Self::mat_to_slice(mat);
        self.v = vol;
    }

    fn as_mat(&self) -> SMatrix<f64, 3, 3> {
        Self::slice_to_mat(&self.m)
    }

    fn from_iso(iso: &IsoMetric<3>) -> Self {
        let s = 1. / (iso.0 * iso.0);
        Self {
            m: [s, s, s, 0.0, 0.0, 0.0],
            v: iso.0.powi(3),
        }
    }

    fn from_diagonal(s: &[f64]) -> Self {
        Self {
            m: [s[0], s[1], s[2], 0.0, 0.0, 0.0],
            v: 1. / (s[0] * s[1] * s[2]).sqrt(),
        }
    }

    fn vol_aniso(&self) -> f64 {
        self.v
    }

    fn scale_aniso(&mut self, s: f64) {
        for i in 0..6 {
            self.m[i] *= s;
        }
        self.v /= f64::sqrt(f64::powi(s, 3));
    }

    fn length_sqr(&self, e: &Vertex<3>) -> f64 {
        // In debug mode this implementation is much faster than nalgrebra's matrix-vector
        // multiplication. In release mode it's the same.
        let m = &self.m;
        let e = &e.data.0[0];
        // (m * e).dot(e)
        let p = [
            m[0] * e[0] + m[3] * e[1] + m[5] * e[2],
            m[3] * e[0] + m[1] * e[1] + m[4] * e[2],
            m[5] * e[0] + m[4] * e[1] + m[2] * e[2],
        ];
        p[0] * e[0] + p[1] * e[1] + p[2] * e[2]
    }
}

impl IntoIterator for AnisoMetric3d {
    type Item = f64;
    type IntoIter = IntoIter<f64, 6>;

    fn into_iter(self) -> Self::IntoIter {
        self.m.into_iter()
    }
}

impl Index<usize> for AnisoMetric3d {
    type Output = f64;

    fn index(&self, index: usize) -> &Self::Output {
        &self.m[index]
    }
}

#[cfg(test)]
mod tests {
    use super::{AnisoMetric, AnisoMetric2d, AnisoMetric3d, Metric};
    use crate::{Result, S_RATIO_MAX};
    use nalgebra::SMatrix;
    use tmesh::{Vert2d, Vert3d};

    #[test]
    fn test_aniso_2d() -> Result<()> {
        let v0 = Vert2d::new(1.0, 0.);
        let v1 = Vert2d::new(0., 0.1);
        let m = AnisoMetric2d::from_sizes(&v0, &v1);

        m.check()?;
        assert!(f64::abs(m.vol() - 0.1) < 1e-12);

        let e = Vert2d::new(1.0, 0.0);
        assert!(f64::abs(m.length(&e) - 1.0) < 1e-12);

        let e = Vert2d::new(0.0, 1.0);
        assert!(f64::abs(m.length(&e) - 10.) < 1e-12);

        let e = Vert2d::new(1.0, 1.0);
        assert!(f64::abs(m.length(&e) - f64::sqrt(101.)) < 1e-12);

        Ok(())
    }

    #[test]
    fn test_positive_2d() -> Result<()> {
        let m = AnisoMetric2d::from_slice(&[1., 1., 2.]);

        m.check()?;
        assert!(f64::abs(m.vol() - 1.0 / f64::sqrt(3.0)) < 1e-12);

        Ok(())
    }

    #[test]
    fn test_check_2d() -> Result<()> {
        let m = AnisoMetric2d::from_slice(&[1.0, 10.0 * S_RATIO_MAX, 0.0]);

        m.check()?;
        assert!(f64::abs(m[0] - 10.0) < 1e-12);

        Ok(())
    }

    #[test]
    fn test_check_3d() -> Result<()> {
        let m = AnisoMetric3d::from_slice(&[1.0, 2.0, 10.0 * S_RATIO_MAX, 0.0, 0.0, 0.0]);

        m.check()?;
        assert!(f64::abs(m[0] - 10.0) < 1e-12);
        assert!(f64::abs(m[1] - 10.0) < 1e-12);

        Ok(())
    }

    #[test]
    fn test_aniso_3d() -> Result<()> {
        let v0 = Vert3d::new(1.0, 0., 0.);
        let v1 = Vert3d::new(0., 0.1, 0.);
        let v2 = Vert3d::new(0., 0., 0.01);
        let m = AnisoMetric3d::from_sizes(&v0, &v1, &v2);

        m.check()?;
        assert!(f64::abs(m.vol() - 0.001) < 1e-12);

        let e = Vert3d::new(1.0, 0.0, 0.0);
        assert!(f64::abs(m.length(&e) - 1.0) < 1e-12);

        let e = Vert3d::new(0.0, 1.0, 0.);
        assert!(f64::abs(m.length(&e) - 10.) < 1e-12);

        let e = Vert3d::new(0.0, 0.0, 1.0);
        assert!(f64::abs(m.length(&e) - 100.) < 1e-12);

        let e = Vert3d::new(1.0, 1.0, 1.0);
        assert!(f64::abs(m.length(&e) - f64::sqrt(10101.)) < 1e-12);

        let s = m.sizes();
        assert!(f64::abs(s[0] - 0.01) < 1e-12);
        assert!(f64::abs(s[1] - 0.1) < 1e-12);
        assert!(f64::abs(s[2] - 1.) < 1e-12);

        Ok(())
    }

    #[test]
    fn test_positive_3d() -> Result<()> {
        let m = AnisoMetric3d::from_slice(&[1., 3., 6., 2., 4., 3.]);

        m.check()?;
        assert!(f64::abs(m.vol() - 1.0) < 1e-12);

        Ok(())
    }

    #[test]
    fn test_intersection_2d_aniso() {
        let eps = 1e-8;

        for _ in 0..100 {
            let mat_r = SMatrix::<f64, 2, 2>::new_random();
            let mat_a = mat_r.transpose() * mat_r;
            let mat_r = SMatrix::<f64, 2, 2>::new_random();
            let mat_b = mat_r.transpose() * mat_r;

            let met_a = AnisoMetric2d::from_mat(mat_a);
            let met_b = AnisoMetric2d::from_mat(mat_b);
            let met_c = met_a.intersect(&met_b);

            for _ in 0..100 {
                let v = Vert2d::new_random();
                let v = v.normalize();
                let la = met_a.length(&v);
                let lb = met_b.length(&v);
                let lc = met_c.length(&v);
                assert!(lc > (1.0 - eps) * la);
                assert!(lc > (1.0 - eps) * lb);
            }
        }
    }

    #[test]
    fn test_intersection_3d_aniso() {
        let eps = 1e-8;

        for _ in 0..100 {
            let mat_r = SMatrix::<f64, 3, 3>::new_random();
            let mat_a = mat_r.transpose() * mat_r;
            let mat_r = SMatrix::<f64, 3, 3>::new_random();
            let mat_b = mat_r.transpose() * mat_r;

            let met_a = AnisoMetric3d::from_mat(mat_a);
            let met_b = AnisoMetric3d::from_mat(mat_b);
            let met_c = met_a.intersect(&met_b);

            for _ in 0..100 {
                let v = Vert3d::new_random();
                let v = v.normalize();
                let la = met_a.length(&v);
                let lb = met_b.length(&v);
                let lc = met_c.length(&v);
                assert!(lc > (1.0 - eps) * la);
                assert!(lc > (1.0 - eps) * lb);
            }
        }
    }

    #[test]
    fn test_span_2d_aniso() {
        let v0 = Vert2d::new(1.0, 0.);
        let v1 = Vert2d::new(0., 0.1);
        let m = AnisoMetric2d::from_sizes(&v0, &v1);

        let e = Vert2d::new(1.0, 0.);
        let m2 = m.span(&e, 1.2, 1.0);

        let l = m2.length(&e);
        assert!(f64::abs(1. / l - 1.18) < 0.01);

        let m2 = m.span(&e, 2.0, 1.0);

        let l = m2.length(&e);
        assert!(f64::abs(1. / l - 1.69) < 0.01);

        let e = Vert2d::new(0.0, 0.1);
        let m2 = m.span(&e, 1.2, 1.0);

        let l = m2.length(&e);
        assert!(f64::abs(1. / l - 1.18) < 0.01);

        let m2 = m.span(&e, 2.0, 1.0);

        let l = m2.length(&e);
        assert!(f64::abs(1. / l - 1.69) < 0.01);

        let e = Vert2d::new(0.0, 0.2);
        let m2 = m.span(&e, 1.2, 1.0);

        let l = m2.length(&e);
        assert!(f64::abs(1. / l - 0.5 * 1.36) < 0.01);

        let m2 = m.span(&e, 2.0, 1.0);

        let l = m2.length(&e);
        assert!(f64::abs(1. / l - 0.5 * 2.38) < 0.01);
    }

    #[test]
    fn test_span_3d_aniso() {
        let v0 = Vert3d::new(1.0, 0.0, 0.0);
        let v1 = Vert3d::new(0.0, 0.1, 0.0);
        let v2 = Vert3d::new(0.0, 0.0, 0.01);
        let m = AnisoMetric3d::from_sizes(&v0, &v1, &v2);

        let e = Vert3d::new(1.0, 0.0, 0.0);
        let m2 = m.span(&e, 1.2, 1.0);

        let l = m2.length(&e);
        assert!(f64::abs(1. / l - 1.18) < 0.01);

        let m2 = m.span(&e, 2.0, 1.0);

        let l = m2.length(&e);
        assert!(f64::abs(1. / l - 1.69) < 0.01);

        let e = Vert3d::new(0.0, 0.1, 0.0);
        let m2 = m.span(&e, 1.2, 1.0);

        let l = m2.length(&e);
        assert!(f64::abs(1. / l - 1.18) < 0.01);

        let m2 = m.span(&e, 2.0, 1.0);

        let l = m2.length(&e);
        assert!(f64::abs(1. / l - 1.69) < 0.01);

        let e = Vert3d::new(0.0, 0.0, 0.01);
        let m2 = m.span(&e, 1.2, 1.0);

        let l = m2.length(&e);
        assert!(f64::abs(1. / l - 1.18) < 0.01);

        let m2 = m.span(&e, 2.0, 1.0);

        let l = m2.length(&e);
        assert!(f64::abs(1. / l - 1.69) < 0.01);

        let e = Vert3d::new(0.0, 0.2, 0.0);
        let m2 = m.span(&e, 1.2, 1.0);

        let l = m2.length(&e);
        assert!(f64::abs(1. / l - 0.5 * 1.36) < 0.01);

        let m2 = m.span(&e, 2.0, 1.0);

        let l = m2.length(&e);
        assert!(f64::abs(1. / l - 0.5 * 2.38) < 0.01);
    }

    #[test]
    fn test_limit_aniso_2d() {
        let ex = Vert2d::new(1.0, 0.0);
        let ey = Vert2d::new(0.0, 1.0);

        let mut m0 = AnisoMetric2d::from_sizes(&ex, &ey);
        let m1 = AnisoMetric2d::from_sizes(&(10.0 * ex), &(10.0 * ey));

        let (a, b) = m0.step(&m1);
        assert!(f64::abs(a - 0.1) < 1e-12);
        assert!(f64::abs(b - 0.1) < 1e-12);

        let m1 = AnisoMetric2d::from_sizes(&(10.0 * ex), &(0.1 * ey));

        let (a, b) = m0.step(&m1);
        assert!(f64::abs(a - 0.1) < 1e-12);
        assert!(f64::abs(b - 10.0) < 1e-12);

        m0.control_step(&m1, 2.0);

        let (a, b) = m0.step(&m1);
        assert!(f64::abs(a - 0.5) < 1e-12);
        assert!(f64::abs(b - 2.0) < 1e-12);

        assert!(f64::abs(m0.sizes()[0] - 0.2) < 1e-12);
        assert!(f64::abs(m0.sizes()[1] - 5.0) < 1e-12);
    }

    #[test]
    fn test_limit_aniso_3d() {
        let ex = Vert3d::new(1.0, 0.0, 0.0);
        let ey = Vert3d::new(0.0, 1.0, 0.0);
        let ez = Vert3d::new(0.0, 0.0, 1.0);

        let mut m0 = AnisoMetric3d::from_sizes(&ex, &ey, &ez);
        let m1 = AnisoMetric3d::from_sizes(&(10.0 * ex), &(0.1 * ey), &(0.001 * ez));

        let (a, b) = m0.step(&m1);
        assert!(f64::abs(a - 0.1) < 1e-12);
        assert!(f64::abs(b - 1000.0) < 1e-12);

        m0.control_step(&m1, 2.0);
        let (a, b) = m0.step(&m1);
        assert!(f64::abs(a - 0.5) < 1e-12);
        assert!(f64::abs(b - 2.0) < 1e-12);

        assert!(f64::abs(m0.sizes()[0] - 0.002) < 1e-12);
        assert!(f64::abs(m0.sizes()[1] - 0.2) < 1e-12);
        assert!(f64::abs(m0.sizes()[2] - 5.0) < 1e-12);
    }
}
