mod aniso;
mod field;
mod iso;
mod reduction;

use crate::Result;
pub use aniso::{AnisoMetric, AnisoMetric2d, AnisoMetric3d};
pub use field::{
    MetricField,
    implied::{tetrahedron_implied_metric, triangle_implied_metric},
};
pub use iso::IsoMetric;
use tmesh::mesh::{GSimplex, Simplex};

use std::fmt::Debug;
use std::fmt::Display;
use tmesh::Vertex;

/// Metric in D-dimensions (iso or anisotropic)
pub trait Metric<const D: usize>:
    Debug + Clone + Copy + IntoIterator<Item = f64> + Default + Display + Send + Sync
{
    const N: usize;
    /// Create a metric from m
    fn from_slice(m: &[f64]) -> Self;
    /// Check if the metric is valid (i.e. positive)
    fn check(&self) -> Result<()>;
    /// Compute the square of the length of an edge in metric space
    fn length_sqr(&self, e: &Vertex<D>) -> f64;
    /// Compute the length of an edge in metric space
    fn length(&self, e: &Vertex<D>) -> f64;
    /// Compute the volume associated with the metric
    fn vol(&self) -> f64;
    /// Interpolate between different metrics to return a valid metric
    fn interpolate<'a, I: Iterator<Item = (f64, &'a Self)>>(weights_and_metrics: I) -> Self
    where
        Self: 'a;
    /// Return the D characteristic sizes of the metric (sorted)
    fn sizes(&self) -> [f64; D];
    /// Scale the metric
    fn scale(&mut self, s: f64);
    /// Scale the metric, applying bounds on the characteristic sizes
    fn scale_with_bounds(&mut self, s: f64, h_min: f64, h_max: f64);
    /// Intersect with another metric, i.e. return the "largest" metric that is both "smaller" that self and other
    #[must_use]
    fn intersect(&self, other: &Self) -> Self;
    /// Span a metric field at location e with a maximum gradation of bets
    #[must_use]
    fn span(&self, e: &Vertex<D>, beta: f64, t: f64) -> Self;
    /// Check if metrics are different with a given tolerance
    fn differs_from(&self, other: &Self, tol: f64) -> bool;
    /// Compute the step between two metrics, i.e. the min and max of
    /// ```math
    /// \frac{\sqrt{e^T \mathcal M_2 e}}{\sqrt{e^T \mathcal M_1 e}}
    /// ```
    /// over all edges $`e`$.
    fn step(&self, other: &Self) -> (f64, f64);
    /// Limit a metric so the required sizes between 1/f and f times those required by other
    /// The directions are not changed
    fn control_step(&mut self, other: &Self, f: f64);
    /// Compute the length of an edge in metric space, assuming a geometric variation of the metric sizes along the edge
    ///
    /// The length of $`e = v_1 - v_0`$ in metric space is
    /// ```math
    /// l_\mathcal M(e) = ||e||_\mathcal M = \int_e \sqrt{e^T \mathcal M e} ds
    /// ```
    /// Assuming a geometric progression of the size along the edge:
    /// ```math
    /// h(t) = h_0^{1 - t} h_1^t
    /// ```
    /// yields
    /// ```math
    /// l_\mathcal M(e) = l_0 \frac{a - 1} { a \ln(a)}
    /// ```
    /// with $`l_0 = \sqrt{e^T \mathcal M_0 e}`$, $`l_1 = \sqrt{e^T \mathcal M_1 e}`$ and $`a = l_1 / l_0`$
    ///
    /// NB: this is consistent with metric interpolation, but a linear variation of the sizes, $`h(t) = (1 - t) h_0^{1 - t} + th_1`$ is assumed
    /// when it comes to gradation. With this assumtion, the metric-space length would be
    /// ```math
    /// l_\mathcal M(e) = l_0 \frac{\ln(a)} { a  - 1}
    /// ```
    ///
    fn edge_length(p0: &Vertex<D>, m0: &Self, p1: &Vertex<D>, m1: &Self) -> f64 {
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
    fn min_metric(metrics: impl IntoIterator<Item = Self>) -> Self {
        let mut iter = metrics.into_iter();
        let m = iter.next().unwrap();
        let mut vol = m.vol();
        let mut res = m;
        for m in iter {
            let volm = m.vol();
            if volm < vol {
                res = m;
                vol = volm;
            }
        }

        res
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
    fn quality<'a, G: GSimplex<D>>(ge: &'a G, metrics: impl IntoIterator<Item = Self>) -> f64
    where
        Self: 'a,
    {
        let m = Self::min_metric(metrics);

        let vol = ge.vol();
        if vol <= 0.0 {
            return -1.0;
        }

        let l = ge.edges().map(|e| m.length_sqr(&e.as_vec())).sum::<f64>();

        let l = l / G::TOPO::N_EDGES as f64;

        let fac = if G::has_normal() {
            let n = ge.normal().normalize();
            m.length(&n)
        } else {
            assert_eq!(D, G::TOPO::DIM);
            1.0
        };

        let vol = vol / (fac * m.vol() * G::ideal_vol());
        let q = f64::powf(vol, 2. / G::TOPO::DIM as f64) / l;
        assert!(q >= 0.0, "q = {q:.2e}, ge = {ge:?}, vol = {:.2e}", ge.vol());
        assert!(
            q < 1.0 + 1e-8,
            "q = {q:.2e}, ge = {ge:?}, vol = {:.2e}",
            ge.vol()
        );
        q
    }
}

#[derive(Debug)]
pub struct MetricElem<const D: usize, C: Simplex, M: Metric<D>>(
    C::GEOM<D>,
    <C::GEOM<D> as GSimplex<D>>::ARRAY<M>,
);

impl<const D: usize, C: Simplex, M: Metric<D>> MetricElem<D, C, M> {
    pub const fn new(ge: C::GEOM<D>, metrics: <C::GEOM<D> as GSimplex<D>>::ARRAY<M>) -> Self {
        Self(ge, metrics)
    }

    #[allow(clippy::should_implement_trait)]
    pub fn from_iter(iter: impl IntoIterator<Item = (Vertex<D>, M)>) -> Self {
        let mut ge = C::GEOM::<D>::default();
        let mut metrics = <C::GEOM<D> as GSimplex<D>>::ARRAY::default();
        let mut count = 0;
        for (i, (v, m)) in iter.into_iter().enumerate() {
            ge.set(i, v);
            metrics[i] = m;
            count += 1;
        }
        assert_eq!(count, C::N_VERTS);

        Self(ge, metrics)
    }

    pub fn vol(&self) -> f64 {
        self.0.vol()
    }

    pub const fn ge(&self) -> &C::GEOM<D> {
        &self.0
    }

    pub fn quality(&self) -> f64 {
        M::quality(&self.0, self.1)
    }
}
