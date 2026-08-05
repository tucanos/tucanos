use crate::{
    Vertex,
    mesh::{
        Edge, GEdge, GNode, GSimplex, Idx, Node, Simplex,
        elements::{ho_simplex::HOType, quadratures::QUADRATURE_EDGE_6},
    },
};
use std::fmt::Debug;
use std::ops::Index;

/// Edge
#[derive(Default, Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct QuadraticEdge<T: Idx>(pub(crate) [T; 3]);

impl<T: Idx> QuadraticEdge<T> {
    #[must_use]
    pub fn new(i0: usize, i1: usize, i2: usize) -> Self {
        Self([
            i0.try_into().unwrap(),
            i1.try_into().unwrap(),
            i2.try_into().unwrap(),
        ])
    }

    pub fn linear(&self) -> Edge<T> {
        Edge::new(self.0[0].try_into().unwrap(), self.0[1].try_into().unwrap())
    }
}

impl<T: Idx> IntoIterator for QuadraticEdge<T> {
    type Item = usize;
    type IntoIter = std::iter::Map<std::array::IntoIter<T, 3>, fn(T) -> usize>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter().map(|x| x.try_into().unwrap())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct QuadraticGEdge<const D: usize>([Vertex<D>; 3], HOType);

impl<const D: usize> QuadraticGEdge<D> {
    #[must_use]
    pub const fn new(v0: &Vertex<D>, v1: &Vertex<D>, v2: &Vertex<D>, etype: HOType) -> Self {
        Self([*v0, *v1, *v2], etype)
    }

    #[must_use]
    pub fn linear(&self) -> GEdge<D> {
        GEdge::new(&self[0], &self[1])
    }

    fn mapping(&self, bcoords: &[f64; 2]) -> Vertex<D> {
        let [u, v] = bcoords;
        2.0 * u * (u - 0.5) * self[0] + 2.0 * v * (v - 0.5) * self[1] + 4.0 * u * v * self[2]
    }

    fn jac_mapping(&self, bcoords: &[f64; 2]) -> [Vertex<D>; 2] {
        let [u, v] = bcoords;
        [
            (4.0 * u - 1.0) * self[0] + 4.0 * v * self[2],
            (4.0 * v - 1.0) * self[1] + 4.0 * u * self[2],
        ]
    }

    fn hess_mapping(&self, _bcoords: &[f64; 2]) -> [Vertex<D>; 3] {
        [4.0 * self[0], 4.0 * self[1], 4.0 * self[2]]
    }

    /// Curvature at the center of the edge
    #[must_use]
    pub fn curvature(&self) -> Vertex<D> {
        let bcoords = [0.5, 0.5];
        let [g_u, g_v] = self.jac_mapping(&bcoords);
        let g = g_v - g_u;
        let [h_uu, h_vv, h_uv] = self.hess_mapping(&bcoords);
        let h = h_uu + h_vv - 2.0 * h_uv;
        let f = if D == 3 {
            g.cross(&h).norm()
        } else {
            (g[0] * h[1] - g[1] * h[0]).abs()
        };
        let res = f / g.norm_squared().powi(2) * g;
        if res.norm() > 1e-12 {
            res
        } else {
            1e-12 * g.normalize()
        }
    }

    fn bezier(&self) -> Self {
        match self.1 {
            HOType::Lagrange => {
                let p = 0.5 * (4.0 * self[2] - self[0] - self[1]);
                Self([self[0], self[1], p], HOType::Bezier)
            }
            HOType::Bezier => *self,
        }
    }
}

impl<const D: usize> Index<usize> for QuadraticGEdge<D> {
    type Output = Vertex<D>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<const D: usize> IntoIterator for QuadraticGEdge<D> {
    type Item = Vertex<D>;
    type IntoIter = std::array::IntoIter<Self::Item, 3>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<const D: usize> Default for QuadraticGEdge<D> {
    fn default() -> Self {
        Self([Vertex::zeros(); 3], HOType::Lagrange)
    }
}

const QUADRATICEDGE2FACES: [Node<usize>; 2] = [Node([1]), Node([0])];

impl<T: Idx> Simplex for QuadraticEdge<T> {
    type T = T;
    type FACE = Node<T>;
    type GEOM<const D: usize> = QuadraticGEdge<D>;
    const DIM: usize = 1;
    const N_VERTS: usize = 3;
    const N_EDGES: usize = 1;
    const N_FACES: usize = 2;

    fn get(&self, index: usize) -> usize {
        self.0[index].try_into().unwrap()
    }

    fn edge(&self, _i: usize) -> Edge<usize> {
        unreachable!()
    }

    fn face(&self, i: usize) -> Self::FACE {
        Self::FACE::from_iter(QUADRATICEDGE2FACES[i].into_iter().map(|j| self.get(j)))
    }

    fn set(&mut self, i: usize, v: usize) {
        self.0[i] = v.try_into().unwrap();
    }

    fn contains(&self, i: usize) -> bool {
        self.0.contains(&i.try_into().unwrap())
    }

    fn sorted(&self) -> Self {
        if self.0[0] < self.0[1] {
            Self(self.0)
        } else {
            Self([self.0[1], self.0[0], self.0[2]])
        }
    }

    fn is_same(&self, other: &Self) -> bool {
        *self == *other
    }

    fn invert(&mut self) {
        self.0.swap(1, 0);
    }

    fn order() -> u8 {
        2
    }

    fn as_slice(&self) -> &[Self::T] {
        &self.0
    }

    fn from_slice(slice: &[Self::T]) -> Result<Self, std::array::TryFromSliceError> {
        slice.try_into().map(|x| Self(x))
    }
}

impl<const D: usize> GSimplex<D> for QuadraticGEdge<D> {
    const N_VERTS: usize = 3;
    type ARRAY<T: Debug + Default + Clone + Copy> = [T; 2];
    type BCOORDS = Self::ARRAY<f64>;
    type TOPO = QuadraticEdge<usize>;
    type FACE = GNode<D>;

    fn ideal_vol() -> f64 {
        1.0
    }

    fn edge(&self, _i: usize) -> GEdge<D> {
        unreachable!()
    }

    fn face(&self, i: usize) -> Self::FACE {
        Self::FACE::from_iter(QUADRATICEDGE2FACES[i].into_iter().map(|j| self[j]))
    }

    fn set(&mut self, i: usize, v: Vertex<D>) {
        self.0[i] = v;
    }

    fn has_normal() -> bool {
        D == 2
    }

    fn vol(&self) -> f64 {
        self.integrate(|_| 1.0)
    }

    fn integrate<G: Fn(&Self::BCOORDS) -> f64>(&self, f: G) -> f64 {
        let mut res = 0.0;
        for &(weight, v) in &QUADRATURE_EDGE_6 {
            let bcoords = [1.0 - v, v];
            let [mut du, dv] = self.jac_mapping(&bcoords);
            du -= dv;
            res += weight * f(&bcoords) * (du.norm_squared()).sqrt();
        }
        res
    }

    fn normal(&self, bcoords: Option<&Self::BCOORDS>) -> Vertex<D> {
        let bcoords = bcoords.unwrap_or(&[0.5, 0.5]);
        let [du, dv] = self.jac_mapping(bcoords);
        let mut res = Vertex::<D>::zeros();
        res[0] = dv[1] - du[1];
        res[1] = du[0] - dv[0];
        res
    }

    fn radius(&self) -> f64 {
        unreachable!()
    }

    fn center_bcoords() -> Self::BCOORDS {
        [0.5, 0.5]
    }

    fn bcoords(&self, v: &Vertex<D>) -> Self::BCOORDS {
        // With p(t) quadratic in t, the stationary points of |p(t) - v|^2
        // are the real roots of a cubic polynomial that can be computed
        // directly. An iterative minimization is not robust here: the
        // objective has negative curvature wherever v lies beyond the
        // local center of curvature, and the gradient magnitude scales as
        // the squared edge length, so fallback gradient steps stall for
        // small edges.
        let a = 2.0 * self[0] + 2.0 * self[1] - 4.0 * self[2];
        let b = 4.0 * self[2] - 3.0 * self[0] - self[1];
        let e = self[0] - v;
        let (roots, n) = real_cubic_roots(
            2.0 * a.norm_squared(),
            3.0 * a.dot(&b),
            b.norm_squared() + 2.0 * a.dot(&e),
            b.dot(&e),
        );
        let mut t = 0.5;
        let mut dmin = f64::MAX;
        for &r in roots.iter().take(n) {
            let d = (e + r * (b + r * a)).norm_squared();
            if d < dmin {
                dmin = d;
                t = r;
            }
        }
        [1.0 - t, t]
    }

    /// Vertex from barycentric coordinates
    fn vert(&self, bcoords: &Self::BCOORDS) -> Vertex<D> {
        self.mapping(bcoords)
    }

    fn gamma(&self) -> f64 {
        unreachable!()
    }

    fn bounding_box(&self) -> (Vertex<D>, Vertex<D>) {
        match self.1 {
            HOType::Lagrange => self.bezier().bounding_box(),
            HOType::Bezier => self
                .into_iter()
                .skip(1)
                .fold((self[0], self[0]), |mut a, b| {
                    for i in 0..D {
                        a.0[i] = a.0[i].min(b[i]);
                        a.1[i] = a.1[i].max(b[i]);
                    }
                    a
                }),
        }
    }
}

/// Real roots of `c3 x^3 + c2 x^2 + c1 x + c0`, gracefully degrading to the
/// quadratic / linear case when the leading coefficients vanish (e.g. the
/// stationary points of the distance to a straight quadratic edge).
pub(super) fn real_cubic_roots(c3: f64, c2: f64, c1: f64, c0: f64) -> ([f64; 3], usize) {
    let mut roots = [0.0; 3];
    let scale = c3.abs().max(c2.abs()).max(c1.abs()).max(c0.abs());
    if scale == 0.0 {
        return (roots, 0);
    }
    let eps = 1e-12 * scale;

    let n = if c3.abs() < eps {
        if c2.abs() < eps {
            if c1.abs() < eps {
                return (roots, 0);
            }
            roots[0] = -c0 / c1;
            1
        } else {
            let delta = c1 * c1 - 4.0 * c2 * c0;
            if delta < 0.0 {
                return (roots, 0);
            }
            // Citardauq formulation to avoid cancellation
            let q = -0.5 * (c1 + c1.signum() * delta.sqrt());
            roots[0] = q / c2;
            if q.abs() > 0.0 {
                roots[1] = c0 / q;
                2
            } else {
                1
            }
        }
    } else {
        // Depressed cubic y^3 + p y + q = 0 with x = y - c2 / (3 c3)
        let a = c2 / c3;
        let b = c1 / c3;
        let c = c0 / c3;
        let p = b - a * a / 3.0;
        let q = 2.0 * a * a * a / 27.0 - a * b / 3.0 + c;
        let shift = -a / 3.0;
        let delta = 0.25 * q * q + p * p * p / 27.0;
        if delta > 0.0 {
            // One real root (Cardano)
            let s = delta.sqrt();
            roots[0] = shift + (-0.5 * q + s).cbrt() + (-0.5 * q - s).cbrt();
            1
        } else {
            // Three real roots (trigonometric method); delta <= 0 implies p <= 0
            let r = (-p / 3.0).sqrt();
            if r < 1e-30 {
                // p ~ 0 and delta <= 0 imply q ~ 0: triple root
                roots[0] = shift;
                1
            } else {
                let phi = ((3.0 * q) / (2.0 * p * r)).clamp(-1.0, 1.0).acos() / 3.0;
                for (k, root) in roots.iter_mut().enumerate() {
                    *root = shift
                        + 2.0 * r * (phi - 2.0 * std::f64::consts::PI * (k as f64) / 3.0).cos();
                }
                3
            }
        }
    };

    // Newton polish on the original polynomial, keeping a step only if it
    // reduces the residual
    for root in roots.iter_mut().take(n) {
        for _ in 0..2 {
            let g = ((c3 * *root + c2) * *root + c1) * *root + c0;
            let dg = (3.0 * c3 * *root + 2.0 * c2) * *root + c1;
            if dg != 0.0 {
                let cand = *root - g / dg;
                let gc = ((c3 * cand + c2) * cand + c1) * cand + c0;
                if gc.abs() < g.abs() {
                    *root = cand;
                }
            }
        }
    }

    (roots, n)
}

#[cfg(test)]
mod tests {
    use rand::{RngExt, SeedableRng, rngs::StdRng};

    use crate::{
        Vert2d, assert_delta,
        mesh::{GEdge, GSimplex, QuadraticGEdge, elements::ho_simplex::HOType},
    };

    #[test]
    fn test_projection_concave_side() {
        // Points beyond the local center of curvature used to make the
        // Newton-CG iteration stall inside the edge and panic
        let h = 1e-3;
        let p0 = Vert2d::new(0.0, 0.0);
        let p1 = Vert2d::new(h, 0.0);
        let p2 = Vert2d::new(0.5 * h, 0.1 * h);
        let ge = QuadraticGEdge::new(&p0, &p1, &p2, HOType::Lagrange);

        for (x, y) in [(0.55, -3.0e-3), (0.68, -1.0e-2), (0.9, -3.0e-3)] {
            let v = Vert2d::new(x * h, y);
            let bc = ge.bcoords(&v);
            let d = (ge.vert(&bc) - v).norm_squared();
            let d_ref = (0..=10000)
                .map(|i| {
                    let t = f64::from(i) / 1e4;
                    (ge.vert(&[1.0 - t, t]) - v).norm_squared()
                })
                .fold(f64::MAX, f64::min);
            assert!(
                d <= d_ref * (1.0 + 1e-8),
                "d = {d:.16e} > d_ref = {d_ref:.16e}"
            );
        }
    }

    #[test]
    fn test_projection_global_min() {
        // The distance at the computed stationary point cannot exceed the
        // minimal distance sampled on the edge
        let mut rng = StdRng::seed_from_u64(5678);

        for _ in 0..10000 {
            let p0 = Vert2d::from_fn(|_, _| rng.random::<f64>() - 0.5);
            let p1 = Vert2d::from_fn(|_, _| rng.random::<f64>() - 0.5);
            let p2 = Vert2d::from_fn(|_, _| rng.random::<f64>() - 0.5);
            let ge = QuadraticGEdge::new(&p0, &p1, &p2, HOType::Lagrange);
            let v = Vert2d::from_fn(|_, _| 10.0 * (rng.random::<f64>() - 0.5));

            let bc = ge.bcoords(&v);
            let d = (ge.vert(&bc) - v).norm_squared();
            let d_ref = (0..=1000)
                .map(|i| {
                    let t = f64::from(i) / 1e3;
                    (ge.vert(&[1.0 - t, t]) - v).norm_squared()
                })
                .fold(f64::MAX, f64::min);
            assert!(
                d <= d_ref * (1.0 + 1e-8),
                "d = {d:.16e} > d_ref = {d_ref:.16e}"
            );
        }
    }

    #[test]
    fn test_quadratic_edge() {
        let p0 = Vert2d::new(0.0, 0.1);
        let p1 = Vert2d::new(0.2, 1.0);

        let ge = GEdge::new(&p0, &p1);
        let p2 = 0.5 * (p0 + p1);
        let ge2 = QuadraticGEdge::new(&p0, &p1, &p2, HOType::Lagrange);

        let n = ge.normal(None);
        let n2 = ge2.normal(Some(&[0.5, 0.5]));
        assert_delta!((n - n2).norm(), 0.0, 1e-12);

        let v = ge.vol();
        let v2 = ge2.vol();
        assert_delta!(v, v2, 1e-12);

        let p2 = Vert2d::new(0.5, 1.2);
        let ge2 = QuadraticGEdge::new(&p0, &p1, &p2, HOType::Lagrange);

        let n = 100;
        let t = (0..=n)
            .map(|i| f64::from(i) / f64::from(n))
            .collect::<Vec<_>>();

        let mut v = 0.0;
        for tmp in t.windows(2) {
            let p0 = ge2.vert(&[1.0 - tmp[0], tmp[0]]);
            let p1 = ge2.vert(&[1.0 - tmp[1], tmp[1]]);
            let ge = GEdge::new(&p0, &p1);
            v += ge.vol();
        }
        let v2 = ge2.vol();

        assert_delta!(v, v2, 0.05 * v);

        let v = 0.1234;
        let p = ge2.vert(&[1.0 - v, v]);
        let n = ge2.normal(Some(&[1.0 - v, v]));

        for p2 in [p + 0.1 * n, p + n, p + 10.0 * n] {
            let (p3, _) = ge2.project(&p2);
            assert!(
                (p - p3).norm() < 1e-10,
                "distance = {:.2e} > 1e-10",
                (p - p3).norm()
            );
        }
    }
}
