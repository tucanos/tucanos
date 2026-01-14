use super::Metric;
use crate::H_MAX;
use crate::{Error, Result};
use std::array::IntoIter;
use std::fmt;
use std::fmt::Debug;
use std::fmt::Display;
use tmesh::Vertex;

/// Isotropic metric in D dimensions
/// The metric is represented by a single scalar, which represents the characteristic size in all the directions
#[derive(Clone, Copy, Debug)]
pub struct IsoMetric<const D: usize>(pub f64);

impl<const D: usize> IsoMetric<D> {
    /// Get the size h from a metric
    #[must_use]
    pub const fn h(&self) -> f64 {
        self.0
    }
}

impl<const D: usize> From<f64> for IsoMetric<D> {
    /// Create an `IsoMetric` from size h
    fn from(h: f64) -> Self {
        Self(h)
    }
}

impl<const D: usize> Default for IsoMetric<D> {
    fn default() -> Self {
        Self(H_MAX)
    }
}

impl<const D: usize> Metric<D> for IsoMetric<D> {
    const N: usize = 1;

    fn from_slice(m: &[f64]) -> Self {
        Self(m[0])
    }

    /// For an isotropic metric, the metric space length is
    /// ```math
    /// l_\mathcal M(e) = \frac{||e||_2}{h}
    /// ```
    fn length(&self, e: &Vertex<D>) -> f64 {
        e.norm() / self.0
    }

    fn length_sqr(&self, e: &Vertex<D>) -> f64 {
        e.norm_squared() / self.0.powi(2)
    }
    /// For an isotropic metric in $`d`$ dimensions, the volume is
    /// ```math
    /// V(\mathcal M) = h^d
    /// ```
    fn vol(&self) -> f64 {
        self.0.powi(D as i32)
    }

    fn check(&self) -> Result<()> {
        if self.0 < 0.0 {
            return Err(Error::from("Negative metric"));
        }
        Ok(())
    }

    /// Linear interpolation is used for isotropic metrics
    /// ```math
    /// h(\sum \alpha_i v_i) = \sum \alpha _i h(v_i)
    /// ```
    ///
    /// NB: this is not consistent with the edge length computation, and different from
    /// what is used for anisotropic metrics
    fn interpolate<'a, I: Iterator<Item = (f64, &'a Self)>>(weights_and_metrics: I) -> Self {
        // Use linear interpolation
        let res = weights_and_metrics.map(|(w, h)| w * h.0).sum();
        Self(res)
    }

    fn sizes(&self) -> [f64; D] {
        [self.0; D]
    }

    fn scale(&mut self, s: f64) {
        self.0 *= s;
    }

    fn scale_with_bounds(&mut self, s: f64, h_min: f64, h_max: f64) {
        self.0 = f64::min(h_max, f64::max(h_min, s * self.0));
    }

    fn intersect(&self, other: &Self) -> Self {
        Self(f64::min(self.0, other.0))
    }

    // assumption: linear variation of h along e (see "Size gradation control of anisotropic meshes", F. Alauzet, 2010)
    fn span(&self, e: &Vertex<D>, beta: f64, _t: f64) -> Self {
        let f = 1. + self.length(e) * f64::ln(beta);
        Self::from(self.0 * f)
    }

    fn differs_from(&self, other: &Self, tol: f64) -> bool {
        f64::abs(self.0 - other.0) > tol * self.0
    }

    fn step(&self, other: &Self) -> (f64, f64) {
        (self.0 / other.0, self.0 / other.0)
    }

    fn control_step(&mut self, other: &Self, f: f64) {
        self.0 = f64::min(self.0, other.0 * f).max(other.0 / f);
    }
}

impl<const D: usize> IntoIterator for IsoMetric<D> {
    type Item = f64;
    type IntoIter = IntoIter<f64, 1>;

    fn into_iter(self) -> Self::IntoIter {
        [self.0].into_iter()
    }
}

impl<const D: usize> Display for IsoMetric<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "h = {:?}", self.0)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::metric::Metric;

    use super::IsoMetric;
    use nalgebra::SVector;
    use tmesh::Vert2d;

    #[test]
    fn test_intersection_2d_iso() {
        let eps = 1e-8;

        for _ in 0..100 {
            let vec_r = SVector::<f64, 2>::new_random();

            let met_a = IsoMetric::<2>::from(vec_r[0]);
            let met_b = IsoMetric::<2>::from(vec_r[1]);
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
    fn test_span_2d_iso() {
        let m = IsoMetric::<2>::from(0.1);
        let e = Vert2d::new(0.0, 0.1);
        let m2 = m.span(&e, 1.2, 1.0);

        assert!(f64::abs(m2.0 - 0.118) < 0.001);

        let m2 = m.span(&e, 2.0, 1.0);

        assert!(f64::abs(m2.0 - 0.169) < 0.001);
    }

    #[test]
    fn test_limit_iso() {
        let mut m0 = IsoMetric::<2>::from(1.0);
        let m1 = IsoMetric::<2>::from(10.0);

        let (a, b) = m0.step(&m1);
        assert!(f64::abs(a - 0.1) < 1e-12);
        assert!(f64::abs(b - 0.1) < 1e-12);

        let (a, b) = m1.step(&m0);
        assert!(f64::abs(a - 10.) < 1e-12);
        assert!(f64::abs(b - 10.) < 1e-12);

        m0.control_step(&m1, 2.0);
        assert!(f64::abs(m0.h() - 5.0) < 1e-12);

        let mut m0 = IsoMetric::<2>::from(1.0);
        let m1 = IsoMetric::<2>::from(0.1);
        m0.control_step(&m1, 2.0);
        assert!(f64::abs(m0.h() - 0.2) < 1e-12);
    }
}
