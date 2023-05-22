use crate::metric_reduction::simultaneous_reduction;
use crate::{mesh::Point, Error, Idx, Result};
use nalgebra::allocator::Allocator;
use nalgebra::{Const, DefaultAllocator, SMatrix};
use std::array::IntoIter;
use std::fmt::Debug;
use std::ops::Index;

/// Metric in D-dimensions (iso or anisotropic)
pub trait Metric<const D: usize>:
    Debug + Clone + Copy + IntoIterator<Item = f64> + Default
{
    const N: usize;
    /// Create a metric from m[start..end]
    fn from_slice(m: &[f64], i: Idx) -> Self;
    /// Check if the metric is valid (i.e. positive)
    fn check(&self) -> Result<()>;
    /// Compute the length of an edge in metric space
    fn length(&self, e: &Point<D>) -> f64;
    /// Compute the volume associated with the metric
    fn vol(&self) -> f64;
    /// Interpolate between different metrics to return a valid metric
    fn interpolate<'a, I: Iterator<Item = (f64, &'a Self)>>(weights_and_metrics: I) -> Self
    where
        Self: 'a;
    /// Return the D characteristic sizes of the metric (sorted)
    fn sizes(&self) -> [f64; D];
    // /// Scale the metric
    // fn scale(&mut self, s: f64);
    /// Scale the metric, applying bounds on the characteristic sizes
    fn scale_with_bounds(&mut self, s: f64, h_min: f64, h_max: f64);
    /// Intersect with another metric, i.e. return the "largest" metric that is both "smaller" that self and other
    #[must_use]
    fn intersect(&self, other: &Self) -> Self;
    /// Span a metric field at location e with a maximum gradation of bets
    #[must_use]
    fn span(&self, e: &Point<D>, beta: f64) -> Self;
    /// Check if metrics are different with a given tolerance
    fn differs_from(&self, other: &Self, tol: f64) -> bool;
    /// Limit a metric so the required sizes between 1/f and f times those required by other
    /// The directions are not changed
    fn limit(&mut self, other: &Self, f: f64);
    /// Compute the length of an edge in metric space, assuming a geometric variation of the metric sizes along the edge
    /// NB: this is consistent with metric interpolation, but a linear variation of the sizes is assumed when it comes to gradation
    fn edge_length(p0: &Point<D>, m0: &Self, p1: &Point<D>, m1: &Self) -> f64 {
        let e = p1 - p0;
        let l0 = m0.length(&e);
        let l1 = m1.length(&e);

        let r = l0 / l1;

        if f64::abs(r - 1.0) > 0.01 {
            l0 * (r - 1.0) / r / f64::ln(r)
        } else {
            l0
        }
    }
    /// Find the metric with the minimum volume
    fn min_metric<'a, I: Iterator<Item = &'a Self>>(mut metrics: I) -> &'a Self {
        let m = metrics.next().unwrap();
        let mut vol = m.vol();
        let mut res = m;
        for m in metrics {
            let volm = m.vol();
            if volm < vol {
                res = m;
                vol = volm;
            }
        }

        res
    }
}

/// Isotropic metric in D dimensions
/// The metric is represented by a single scalar, which represents the characteristic size in all the directions
#[derive(Clone, Copy, Debug)]
pub struct IsoMetric<const D: usize>(f64);

impl<const D: usize> IsoMetric<D> {
    /// Create an `IsoMetric` from size h
    #[must_use]
    pub fn from(h: f64) -> Self {
        Self(h)
    }

    /// Get the size h from a metric
    #[must_use]
    pub fn h(&self) -> f64 {
        self.0
    }
}

impl<const D: usize> Default for IsoMetric<D> {
    fn default() -> Self {
        Self(0.)
    }
}

impl<const D: usize> Metric<D> for IsoMetric<D> {
    const N: usize = 1;

    fn from_slice(m: &[f64], i: Idx) -> Self {
        Self(m[i as usize])
    }

    fn length(&self, e: &Point<D>) -> f64 {
        e.norm() / self.0
    }

    fn vol(&self) -> f64 {
        self.0.powi(D as i32)
    }

    fn check(&self) -> Result<()> {
        if self.0 < 0.0 {
            return Err(Error::from("Negative metric"));
        }
        Ok(())
    }

    fn interpolate<'a, I: Iterator<Item = (f64, &'a Self)>>(weights_and_metrics: I) -> Self {
        // Use linear interpolation
        let res = weights_and_metrics.map(|(w, h)| w * h.0).sum();
        Self(res)
    }

    fn sizes(&self) -> [f64; D] {
        [self.0; D]
    }

    // fn scale(&mut self, s: f64) {
    //     self.0 *= s;
    // }

    fn scale_with_bounds(&mut self, s: f64, h_min: f64, h_max: f64) {
        self.0 = f64::min(h_max, f64::max(h_min, s * self.0));
    }

    fn intersect(&self, other: &Self) -> Self {
        Self(f64::min(self.0, other.0))
    }

    fn span(&self, e: &Point<D>, beta: f64) -> Self {
        // assumption: linear variation of h along e (see "Size gradation control of anisotropic meshes", F. Alauzet, 2010)
        let f = 1. + self.length(e) * f64::ln(beta);
        Self::from(self.0 * f)
    }

    fn differs_from(&self, other: &Self, tol: f64) -> bool {
        f64::abs(self.0 - other.0) > tol * self.0
    }

    fn limit(&mut self, other: &Self, f: f64) {
        self.0 = f64::min(self.0, other.0 * f).max(other.0 / f);
    }

    fn edge_length(p0: &Point<D>, m0: &Self, p1: &Point<D>, m1: &Self) -> f64 {
        let e = p1 - p0;
        let l0 = m0.length(&e);
        let l1 = m1.length(&e);

        let r = l0 / l1;

        if f64::abs(r - 1.0) > 0.01 {
            l0 * (r - 1.0) / r / f64::ln(r)
        } else {
            l0
        }
    }

    fn min_metric<'a, I: Iterator<Item = &'a Self>>(mut metrics: I) -> &'a Self {
        let m = metrics.next().unwrap();
        let mut vol = m.vol();
        let mut res = m;
        for m in metrics {
            let volm = m.vol();
            if volm < vol {
                res = m;
                vol = volm;
            }
        }

        res
    }
}

impl<const D: usize> IntoIterator for IsoMetric<D> {
    type Item = f64;
    type IntoIter = IntoIter<f64, 1>;

    fn into_iter(self) -> Self::IntoIter {
        [self.0].into_iter()
    }
}

pub trait AnisoMetric<const D: usize>: Metric<D> + Index<usize, Output = f64>
where
    Const<D>: nalgebra::ToTypenum,
    Const<D>: nalgebra::DimSub<nalgebra::U1>,
    DefaultAllocator: Allocator<f64, <Const<D> as nalgebra::DimSub<nalgebra::U1>>::Output>,
{
    const N: usize;

    fn slice_to_mat(m: &[f64]) -> SMatrix<f64, D, D>;

    fn from_mat_and_vol(mat: SMatrix<f64, D, D>, vol: f64) -> Self;

    fn update_from_mat_and_vol(&mut self, mat: SMatrix<f64, D, D>, vol: f64);

    /// Get a matrix representation of the metric
    fn as_mat(&self) -> SMatrix<f64, D, D>;

    fn get_vol(&self) -> f64;

    fn from_mat(mat: SMatrix<f64, D, D>) -> Self {
        let mut eig = mat.symmetric_eigen();
        // Ensure that the metric is valid, i.e. that all the eigenvalues are >0
        eig.eigenvalues
            .iter_mut()
            .for_each(|i| *i = f64::max(f64::MIN_POSITIVE, f64::abs(*i)));
        let mat = eig.recompose();
        let vol = 1. / eig.eigenvalues.iter().product::<f64>().sqrt();
        assert!(vol > 0.0);
        Self::from_mat_and_vol(mat, vol)
    }

    fn is_diagonal(&self, tol: f64) -> bool {
        let on_diag = (0..D).map(|i| self[i]).sum::<f64>();
        let off_diag = (D..<Self as Metric<D>>::N).map(|i| self[i]).sum::<f64>();
        tol * on_diag > off_diag
    }

    fn from_diagonal(s: &[f64]) -> Self;
}

impl<const D: usize, T: AnisoMetric<D>> Metric<D> for T
where
    Const<D>: nalgebra::ToTypenum,
    Const<D>: nalgebra::DimSub<nalgebra::U1>,
    DefaultAllocator: Allocator<f64, <Const<D> as nalgebra::DimSub<nalgebra::U1>>::Output>,
{
    const N: usize = <Self as AnisoMetric<D>>::N;

    fn from_slice(m: &[f64], i: Idx) -> Self {
        let start = i as usize * <Self as Metric<D>>::N;
        let end = start + <Self as Metric<D>>::N;

        let mat = Self::slice_to_mat(&m[start..end]);
        Self::from_mat(mat)
    }

    fn check(&self) -> Result<()> {
        let eig = self.as_mat().symmetric_eigen();

        if eig.eigenvalues.iter().any(|&x| x < f64::MIN_POSITIVE) {
            return Err(Error::from("Negative metric"));
        }
        Ok(())
    }

    fn interpolate<'a, I: Iterator<Item = (f64, &'a Self)>>(weights_and_metrics: I) -> Self
    where
        Self: 'a,
    {
        // In order to ensure that the resulting matrix is >0, we compute the interpolation as
        // exp(sum(w_i ln(m_i)))
        let mut mat = Self::slice_to_mat(&[0.0; 6]);

        for (w, m) in weights_and_metrics {
            let mut eig = m.as_mat().symmetric_eigen();
            eig.eigenvalues.iter_mut().for_each(|i| *i = w * (*i).ln());
            mat += eig.recompose();
        }

        let mut eig = mat.symmetric_eigen();
        eig.eigenvalues.iter_mut().for_each(|i| *i = (*i).exp());
        mat = eig.recompose();
        let vol = 1. / eig.eigenvalues.iter().product::<f64>().sqrt();

        Self::from_mat_and_vol(mat, vol)
    }

    fn sizes(&self) -> [f64; D] {
        let eig = self.as_mat().symmetric_eigen();
        let mut s = [0.; D];
        eig.eigenvalues
            .iter()
            .enumerate()
            .for_each(|(i, e)| s[i] = 1. / e.sqrt());

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
        let vol = 1. / eig.eigenvalues.iter().fold(1.0, |v, &e| v * e).sqrt();

        self.update_from_mat_and_vol(mat, vol);
    }

    fn intersect(&self, other: &Self) -> Self {
        let tol = 1e-8;
        if self.is_diagonal(tol) && other.is_diagonal(tol) {
            let mut s = [0.0; D];
            for i in 0..D {
                s[i] = f64::max(self[i], other[i]);
            }
            return Self::from_slice(&s, 0);
        }

        let res = simultaneous_reduction(self.as_mat(), other.as_mat());

        Self::from_mat(res)
    }

    fn span(&self, e: &Point<D>, beta: f64) -> Self {
        // use physical-space-gradation
        let nrm = e.norm();
        let mat = self.as_mat();
        let mut eig = mat.symmetric_eigen();
        eig.eigenvalues.iter_mut().for_each(|s| {
            let eta = 1.0 + f64::sqrt(*s) * nrm * f64::ln(beta);
            *s /= eta * eta;
        });
        let mat = eig.recompose();
        let vol = 1. / eig.eigenvalues.iter().fold(1.0, |v, &e| v * e).sqrt();

        Self::from_mat_and_vol(mat, vol)
    }

    fn differs_from(&self, other: &Self, tol: f64) -> bool {
        self.into_iter()
            .zip(other.into_iter())
            .any(|(x, y)| f64::abs(x - y) > tol * x)
    }

    fn limit(&mut self, other: &Self, f: f64) {
        let f2 = f * f;
        let mat = self.as_mat();
        let mut eig = mat.symmetric_eigen();

        eig.eigenvalues
            .iter_mut()
            .zip(eig.eigenvectors.column_iter())
            .for_each(|(l, v)| {
                let v = v.clone_owned();
                let l_other = other.length(&v).powi(2);
                *l = f64::min(*l, l_other * f2).max(l_other / f2);
            });

        let vol = 1. / eig.eigenvalues.iter().fold(1.0, |v, &e| v * e).sqrt();

        let mat = eig.recompose();
        self.update_from_mat_and_vol(mat, vol);
    }

    fn length(&self, e: &Point<D>) -> f64 {
        (self.as_mat() * e).dot(e).sqrt()
    }

    fn vol(&self) -> f64 {
        self.get_vol()
    }

    // fn scale(&mut self, s: f64) {
    //     todo!()
    // }
}

/// Anisotropic metric in 2D, represented with 3 scalars (x0,x1,x2)
/// For the storage, we follow VTK: the symmetric matrix is
/// x0 x2
/// x2 x1
/// NB: the matrix must be positive definite for the metric to be valid
/// The metric volume 1/sqrt(det(M)) is stored to avoid multiple recomputations
/// TODO: reuse the eigenvalue solvers
#[derive(Clone, Copy, Debug, Default)]
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
    pub fn from_sizes(s0: &Point<2>, s1: &Point<2>) -> Self {
        let n0 = s0.norm();
        let n1 = s1.norm();
        let s0 = s0 / n0;
        let s1 = s1 / n1;
        assert!(s0.dot(&s1) < 1e-12);

        let eigvals = SMatrix::<f64, 2, 2>::new(1. / n0.powi(2), 0., 0., 1. / n1.powi(2));
        let eigvecs = SMatrix::<f64, 2, 2>::new(s0[0], s0[1], s1[0], s1[1]);
        let mat = eigvals * eigvecs;
        let mat = eigvecs.tr_mul(&mat);

        Self::from_mat(mat)
    }

    // fn scale(&mut self, s: f64) {
    //     for i in 0..3 {
    //         self.m[i] *= s;
    //     }
    //     self.v /= f64::sqrt(f64::powi(s, 2));
    // }
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

    fn from_diagonal(s: &[f64]) -> Self {
        Self {
            m: [s[0], s[1], 0.0],
            v: 1. / (s[0] * s[1]).sqrt(),
        }
    }

    fn get_vol(&self) -> f64 {
        self.v
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

/// Anisotropic metric in 3D, represented with 6 scalars (x0,x1,x2,x3,x4,x5)
/// For the storage, we follow VTK: the symmetric matrix is
/// x0 x3 x5
/// x3 x1 x4
/// x5 x4 x2
/// NB: the matrix must be positive definite for the metric to be valid
/// The metric volume 1/sqrt(det(M)) is stored to avoid multiple recomputations
/// TODO: reuse the eigenvalue solvers
#[derive(Clone, Copy, Debug, Default)]
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
    pub fn from_sizes(s0: &Point<3>, s1: &Point<3>, s2: &Point<3>) -> Self {
        let n0 = s0.norm();
        let n1 = s1.norm();
        let n2 = s2.norm();
        let s0 = s0 / n0;
        let s1 = s1 / n1;
        let s2 = s2 / n2;
        assert!(s0.dot(&s1) < 1e-12);
        assert!(s0.dot(&s2) < 1e-12);
        assert!(s1.dot(&s2) < 1e-12);

        let eigvals = SMatrix::<f64, 3, 3>::new(
            1. / n0.powi(2),
            0.,
            0.,
            0.,
            1. / n1.powi(2),
            0.,
            0.,
            0.,
            1. / n2.powi(2),
        );
        let eigvecs = SMatrix::<f64, 3, 3>::new(
            s0[0], s0[1], s0[2], s1[0], s1[1], s1[2], s2[0], s2[1], s2[2],
        );
        let mat = eigvals * eigvecs;
        let mat = eigvecs.tr_mul(&mat);

        Self::from_mat(mat)
    }
}
impl AnisoMetric<3> for AnisoMetric3d {
    const N: usize = 6;

    /// Convert from 6 scalars to a matrix
    #[must_use]
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

    fn from_diagonal(s: &[f64]) -> Self {
        Self {
            m: [s[0], s[1], s[2], 0.0, 0.0, 0.0],
            v: 1. / (s[0] * s[1] * s[2]).sqrt(),
        }
    }

    fn get_vol(&self) -> f64 {
        self.v
    }
}

// fn scale(&mut self, s: f64) {
//     for i in 0..Self::N {
//         self.m[i] *= s;
//     }
//     self.v /= f64::sqrt(f64::powi(s, 3));
// }

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
    use super::{AnisoMetric, AnisoMetric2d, AnisoMetric3d, IsoMetric, Metric};
    use crate::{mesh::Point, Result};
    use nalgebra::{SMatrix, SVector};

    #[test]
    fn test_aniso_2d() -> Result<()> {
        let v0 = Point::<2>::new(1.0, 0.);
        let v1 = Point::<2>::new(0., 0.1);
        let m = AnisoMetric2d::from_sizes(&v0, &v1);

        m.check()?;
        assert!(f64::abs(m.vol() - 0.1) < 1e-12);

        let e = Point::<2>::new(1.0, 0.0);
        assert!(f64::abs(m.length(&e) - 1.0) < 1e-12);

        let e = Point::<2>::new(0.0, 1.0);
        assert!(f64::abs(m.length(&e) - 10.) < 1e-12);

        let e = Point::<2>::new(1.0, 1.0);
        assert!(f64::abs(m.length(&e) - f64::sqrt(101.)) < 1e-12);

        Ok(())
    }

    #[test]
    fn test_positive_2d() -> Result<()> {
        let m = AnisoMetric2d::from_slice(&[1., 1., 2.], 0);

        m.check()?;
        assert!(f64::abs(m.vol() - 1.0 / f64::sqrt(3.0)) < 1e-12);

        Ok(())
    }

    #[test]
    fn test_aniso_3d() -> Result<()> {
        let v0 = Point::<3>::new(1.0, 0., 0.);
        let v1 = Point::<3>::new(0., 0.1, 0.);
        let v2 = Point::<3>::new(0., 0., 0.01);
        let m = AnisoMetric3d::from_sizes(&v0, &v1, &v2);

        m.check()?;
        assert!(f64::abs(m.vol() - 0.001) < 1e-12);

        let e = Point::<3>::new(1.0, 0.0, 0.0);
        assert!(f64::abs(m.length(&e) - 1.0) < 1e-12);

        let e = Point::<3>::new(0.0, 1.0, 0.);
        assert!(f64::abs(m.length(&e) - 10.) < 1e-12);

        let e = Point::<3>::new(0.0, 0.0, 1.0);
        assert!(f64::abs(m.length(&e) - 100.) < 1e-12);

        let e = Point::<3>::new(1.0, 1.0, 1.0);
        assert!(f64::abs(m.length(&e) - f64::sqrt(10101.)) < 1e-12);

        let s = m.sizes();
        assert!(f64::abs(s[0] - 0.01) < 1e-12);
        assert!(f64::abs(s[1] - 0.1) < 1e-12);
        assert!(f64::abs(s[2] - 1.) < 1e-12);

        Ok(())
    }

    #[test]
    fn test_positive_3d() -> Result<()> {
        let m = AnisoMetric3d::from_slice(&[1., 3., 6., 2., 4., 3.], 0);

        m.check()?;
        assert!(f64::abs(m.vol() - 1.0) < 1e-12);

        Ok(())
    }

    #[test]
    fn test_intersection_2d_iso() {
        let eps = 1e-8;

        for _ in 0..100 {
            let vec_r = SVector::<f64, 2>::new_random();

            let met_a = IsoMetric::<2>::from(vec_r[0]);
            let met_b = IsoMetric::<2>::from(vec_r[1]);
            let met_c = met_a.intersect(&met_b);

            for _ in 0..100 {
                let v = Point::<2>::new_random();
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
                let v = Point::<2>::new_random();
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
                let v = Point::<3>::new_random();
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
    fn test_span_2d_iso() {
        let m = IsoMetric::<2>::from(0.1);
        let e = Point::<2>::new(0.0, 0.1);
        let m2 = m.span(&e, 1.2);

        assert!(f64::abs(m2.0 - 0.118) < 0.001);

        let m2 = m.span(&e, 2.0);

        assert!(f64::abs(m2.0 - 0.169) < 0.001);
    }

    #[test]
    fn test_span_2d_aniso() {
        let v0 = Point::<2>::new(1.0, 0.);
        let v1 = Point::<2>::new(0., 0.1);
        let m = AnisoMetric2d::from_sizes(&v0, &v1);

        let e = Point::<2>::new(1.0, 0.);
        let m2 = m.span(&e, 1.2);

        let l = m2.length(&e);
        assert!(f64::abs(1. / l - 1.18) < 0.01);

        let m2 = m.span(&e, 2.0);

        let l = m2.length(&e);
        assert!(f64::abs(1. / l - 1.69) < 0.01);

        let e = Point::<2>::new(0.0, 0.1);
        let m2 = m.span(&e, 1.2);

        let l = m2.length(&e);
        assert!(f64::abs(1. / l - 1.18) < 0.01);

        let m2 = m.span(&e, 2.0);

        let l = m2.length(&e);
        assert!(f64::abs(1. / l - 1.69) < 0.01);

        let e = Point::<2>::new(0.0, 0.2);
        let m2 = m.span(&e, 1.2);

        let l = m2.length(&e);
        assert!(f64::abs(1. / l - 0.5 * 1.36) < 0.01);

        let m2 = m.span(&e, 2.0);

        let l = m2.length(&e);
        assert!(f64::abs(1. / l - 0.5 * 2.38) < 0.01);
    }

    #[test]
    fn test_span_3d_aniso() {
        let v0 = Point::<3>::new(1.0, 0.0, 0.0);
        let v1 = Point::<3>::new(0.0, 0.1, 0.0);
        let v2 = Point::<3>::new(0.0, 0.0, 0.01);
        let m = AnisoMetric3d::from_sizes(&v0, &v1, &v2);

        let e = Point::<3>::new(1.0, 0.0, 0.0);
        let m2 = m.span(&e, 1.2);

        let l = m2.length(&e);
        assert!(f64::abs(1. / l - 1.18) < 0.01);

        let m2 = m.span(&e, 2.0);

        let l = m2.length(&e);
        assert!(f64::abs(1. / l - 1.69) < 0.01);

        let e = Point::<3>::new(0.0, 0.1, 0.0);
        let m2 = m.span(&e, 1.2);

        let l = m2.length(&e);
        assert!(f64::abs(1. / l - 1.18) < 0.01);

        let m2 = m.span(&e, 2.0);

        let l = m2.length(&e);
        assert!(f64::abs(1. / l - 1.69) < 0.01);

        let e = Point::<3>::new(0.0, 0.0, 0.01);
        let m2 = m.span(&e, 1.2);

        let l = m2.length(&e);
        assert!(f64::abs(1. / l - 1.18) < 0.01);

        let m2 = m.span(&e, 2.0);

        let l = m2.length(&e);
        assert!(f64::abs(1. / l - 1.69) < 0.01);

        let e = Point::<3>::new(0.0, 0.2, 0.0);
        let m2 = m.span(&e, 1.2);

        let l = m2.length(&e);
        assert!(f64::abs(1. / l - 0.5 * 1.36) < 0.01);

        let m2 = m.span(&e, 2.0);

        let l = m2.length(&e);
        assert!(f64::abs(1. / l - 0.5 * 2.38) < 0.01);
    }

    #[test]
    fn test_limit_iso() {
        let mut m0 = IsoMetric::<2>::from(1.0);
        let m1 = IsoMetric::<2>::from(10.0);
        m0.limit(&m1, 2.0);
        assert!(f64::abs(m0.h() - 5.0) < 1e-12);

        let mut m0 = IsoMetric::<2>::from(1.0);
        let m1 = IsoMetric::<2>::from(0.1);
        m0.limit(&m1, 2.0);
        assert!(f64::abs(m0.h() - 0.2) < 1e-12);
    }

    #[test]
    fn test_limit_aniso_2d() {
        let ex = Point::<2>::new(1.0, 0.0);
        let ey = Point::<2>::new(0.0, 1.0);

        let mut m0 = AnisoMetric2d::from_sizes(&ex, &ey);
        let m1 = AnisoMetric2d::from_sizes(&(10.0 * ex), &(0.1 * ey));

        m0.limit(&m1, 2.0);

        assert!(f64::abs(m0.sizes()[0] - 0.2) < 1e-12);
        assert!(f64::abs(m0.sizes()[1] - 5.0) < 1e-12);
    }

    #[test]
    fn test_limit_aniso_3d() {
        let ex = Point::<3>::new(1.0, 0.0, 0.0);
        let ey = Point::<3>::new(0.0, 1.0, 0.0);
        let ez = Point::<3>::new(0.0, 0.0, 1.0);

        let mut m0 = AnisoMetric3d::from_sizes(&ex, &ey, &ez);
        let m1 = AnisoMetric3d::from_sizes(&(10.0 * ex), &(0.1 * ey), &(0.001 * ez));

        m0.limit(&m1, 2.0);

        assert!(f64::abs(m0.sizes()[0] - 0.002) < 1e-12);
        assert!(f64::abs(m0.sizes()[1] - 0.2) < 1e-12);
        assert!(f64::abs(m0.sizes()[2] - 5.0) < 1e-12);
    }
}
