use crate::{
    Vertex,
    mesh::{Edge, GEdge, GSimplex, Simplex, elements::Idx},
};
use nalgebra::{SMatrix, SVector};
use std::fmt::Debug;
use std::ops::Index;

/// Triangle
#[derive(Default, Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Triangle<T: Idx>(pub(crate) [T; 3]);

impl<T: Idx> Triangle<T> {
    #[must_use]
    pub fn new(i0: usize, i1: usize, i2: usize) -> Self {
        Self([
            i0.try_into().unwrap(),
            i1.try_into().unwrap(),
            i2.try_into().unwrap(),
        ])
    }
}

impl<T: Idx> IntoIterator for Triangle<T> {
    type Item = usize;
    type IntoIter = std::iter::Map<std::array::IntoIter<T, 3>, fn(T) -> usize>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter().map(|x| x.try_into().unwrap())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct GTriangle<const D: usize>([Vertex<D>; 3]);

impl<const D: usize> GTriangle<D> {
    #[must_use]
    pub const fn new(v0: &Vertex<D>, v1: &Vertex<D>, v2: &Vertex<D>) -> Self {
        Self([*v0, *v1, *v2])
    }
}

impl<const D: usize> Index<usize> for GTriangle<D> {
    type Output = Vertex<D>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<const D: usize> IntoIterator for GTriangle<D> {
    type Item = Vertex<D>;
    type IntoIter = std::array::IntoIter<Self::Item, 3>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<const D: usize> Default for GTriangle<D> {
    fn default() -> Self {
        Self([Vertex::zeros(); 3])
    }
}

const TRIANGLE2EDGES: [Edge<usize>; 3] = [Edge([0, 1]), Edge([1, 2]), Edge([2, 0])];
const TRIANGLE2FACES: [Edge<usize>; 3] = [Edge([1, 2]), Edge([2, 0]), Edge([0, 1])];

impl<T: Idx> Simplex for Triangle<T> {
    type T = T;
    type FACE = Edge<T>;
    type GEOM<const D: usize> = GTriangle<D>;
    const DIM: usize = 2;
    const N_VERTS: usize = 3;
    const N_EDGES: usize = 3;
    const N_FACES: usize = 3;

    fn get(&self, index: usize) -> usize {
        self.0[index].try_into().unwrap()
    }

    fn edge(&self, i: usize) -> Edge<usize> {
        Edge::from_iter(TRIANGLE2EDGES[i].into_iter().map(|j| self.get(j)))
    }

    fn face(&self, i: usize) -> Self::FACE {
        Self::FACE::from_iter(TRIANGLE2FACES[i].into_iter().map(|j| self.get(j)))
    }

    fn set(&mut self, i: usize, v: usize) {
        self.0[i] = v.try_into().unwrap();
    }

    fn contains(&self, i: usize) -> bool {
        self.0.contains(&i.try_into().unwrap())
    }

    fn sorted(&self) -> Self {
        let mut tmp = *self;
        tmp.0.sort_unstable();
        tmp
    }

    fn is_same(&self, other: &Self) -> bool {
        let [i0, i1, i2] = self.0;
        *other == *self || other.0 == [i1, i2, i0] || other.0 == [i2, i0, i1]
    }

    fn invert(&mut self) {
        self.0.swap(1, 0);
    }
}

impl<const D: usize> GSimplex<D> for GTriangle<D> {
    const N_VERTS: usize = 3;
    type ARRAY<T: Debug + Default + Clone + Copy> = [T; 3];
    type BCOORDS = Self::ARRAY<f64>;
    type TOPO = Triangle<usize>;
    type FACE = GEdge<D>;

    fn ideal_vol() -> f64 {
        3.0_f64.sqrt() / 4.
    }

    fn edge(&self, i: usize) -> GEdge<D> {
        GEdge::from_iter(TRIANGLE2EDGES[i].into_iter().map(|j| self[j]))
    }

    fn face(&self, i: usize) -> Self::FACE {
        Self::FACE::from_iter(TRIANGLE2FACES[i].into_iter().map(|j| self[j]))
    }

    fn set(&mut self, i: usize, v: Vertex<D>) {
        self.0[i] = v;
    }

    fn has_normal() -> bool {
        D == 3
    }

    fn vol(&self) -> f64 {
        if Self::has_normal() {
            self.normal(None).norm()
        } else {
            assert_eq!(D, 2);
            let e1 = self[1] - self[0];
            let e2 = self[2] - self[0];

            0.5 * (e1[0] * e2[1] - e1[1] * e2[0])
        }
    }

    fn integrate<G: Fn(&Self::BCOORDS) -> f64>(&self, f: G) -> f64 {
        let mut res = 0.0;
        for &(weight, v, w) in &super::quadratures::QUADRATURE_TRIANGLE_3 {
            res += weight * f(&[1.0 - v - w, v, w]);
        }
        res * self.vol()
    }

    fn normal(&self, bcoords: Option<&Self::BCOORDS>) -> Vertex<D> {
        if Self::has_normal() {
            let e1 = self[1] - self[0];
            let e2 = self[2] - self[0];
            let n = 0.5 * e1.cross(&e2);
            if bcoords.is_some() { n.normalize() } else { n }
        } else {
            unreachable!()
        }
    }

    fn radius(&self) -> f64 {
        let a = (self[2] - self[1]).norm();
        let b = (self[2] - self[0]).norm();
        let c = (self[1] - self[0]).norm();
        let s = 0.5 * (a + b + c);
        ((s - a) * (s - b) * (s - c) / s).sqrt()
    }

    fn center_bcoords() -> Self::BCOORDS {
        [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
    }

    fn bcoords(&self, v: &Vertex<D>) -> [f64; 3] {
        if D == 2 {
            let a = SMatrix::<f64, 3, 3>::new(
                1.0, 1.0, 1.0, self[0][0], self[1][0], self[2][0], self[0][1], self[1][1],
                self[2][1],
            );
            let b = SVector::<f64, 3>::new(1., v[0], v[1]);
            let decomp = a.lu();
            let x = decomp.solve(&b).unwrap();
            [x[0], x[1], x[2]]
        } else {
            let e0 = self[1] - self[0];
            let e1 = self[2] - self[0];
            let n = e0.cross(&e1);
            let w = v - self[0];
            let nrm = n.norm_squared();
            let gamma = e0.cross(&w).dot(&n) / nrm;
            let beta = w.cross(&e1).dot(&n) / nrm;
            [1.0 - beta - gamma, beta, gamma]
        }
    }

    fn gamma(&self) -> f64 {
        let mut a = self[2] - self[1];
        let mut b = self[0] - self[2];
        let mut c = self[1] - self[0];

        a.normalize_mut();
        b.normalize_mut();
        c.normalize_mut();

        let cross_norm = |e1: &Vertex<D>, e2: &Vertex<D>| {
            if D == 2 {
                (e1[0] * e2[1] - e1[1] * e2[0]).abs()
            } else {
                let n = e1.cross(e2);
                n.norm()
            }
        };
        let sina = cross_norm(&b, &c);
        let sinb = cross_norm(&a, &c);
        let sinc = cross_norm(&a, &b);

        let tmp = sina + sinb + sinc;
        if tmp < 1e-12 {
            0.0
        } else {
            4.0 * sina * sinb * sinc / tmp
        }
    }

    fn bounding_box(&self) -> (Vertex<D>, Vertex<D>) {
        self.into_iter()
            .skip(1)
            .fold((self[0], self[0]), |mut a, b| {
                for i in 0..D {
                    a.0[i] = a.0[i].min(b[i]);
                    a.1[i] = a.1[i].max(b[i]);
                }
                a
            })
    }
}

#[cfg(test)]
mod tests {
    use super::{GTriangle, Triangle};
    use crate::{
        Vert2d, Vert3d, assert_delta,
        mesh::{GSimplex, Simplex},
    };
    use rand::{Rng, SeedableRng, rngs::StdRng, seq::SliceRandom};

    #[test]
    fn test_vol() {
        let v0 = Vert2d::new(0.0, 0.0);
        let v1 = Vert2d::new(1.0, 0.0);
        let v2 = Vert2d::new(0.0, 1.0);
        let ge = GTriangle([v0, v1, v2]);
        assert_delta!(ge.vol(), 0.5, 1e-12);
        let ge = GTriangle([v0, v2, v1]);
        assert_delta!(ge.vol(), -0.5, 1e-12);
    }

    #[test]
    fn test_is_same() {
        let pts = [
            Vert3d::new(0.0, 0.0, 0.0),
            Vert3d::new(1.0, 0.0, 0.0),
            Vert3d::new(0.0, 1.0, 0.0),
            Vert3d::new(0.0, 0.0, 1.0),
        ];

        let gt = |e: &Triangle<usize>| GTriangle([pts[e.get(0)], pts[e.get(1)], pts[e.get(2)]]);
        let e = Triangle([0, 1, 2]);
        let mut o = e;

        let mut rng = StdRng::seed_from_u64(1234);
        for _ in 0..10 {
            o.0.shuffle(&mut rng);
            let is_same = e.is_same(&o);
            let n = gt(&o).normal(None);
            if is_same {
                assert!(n[2] > 0.0);
            } else {
                assert!(n[2] < 0.0);
            }
        }
    }

    #[test]
    fn test_project_triangle() {
        let p0 = Vert3d::new(0.0, 0.0, 0.0);
        let p1 = Vert3d::new(1.0, 0.0, 0.0);
        let p2 = Vert3d::new(0.0, 1.0, 0.0);

        let ge = GTriangle([p0, p1, p2]);

        let p = Vert3d::new(-1.0, -1.0, 1.0);
        let (proj, _) = ge.project(&p);
        assert!((proj - Vert3d::new(0.0, 0.0, 0.0)).norm() < 1e-12);

        let p = Vert3d::new(2.0, -1.0, 1.0);
        let (proj, _) = ge.project(&p);
        assert!((proj - Vert3d::new(1.0, 0.0, 0.0)).norm() < 1e-12);

        let p = Vert3d::new(-2.0, 3.0, -1.0);
        let (proj, _) = ge.project(&p);
        assert!((proj - Vert3d::new(0.0, 1.0, 0.0)).norm() < 1e-12);

        let p = Vert3d::new(0.5, -2.0, -1.0);
        let (proj, _) = ge.project(&p);
        assert!((proj - Vert3d::new(0.5, 0.0, 0.0)).norm() < 1e-12);

        let p = Vert3d::new(-10.0, 0.2, 1.0);
        let (proj, _) = ge.project(&p);
        assert!((proj - Vert3d::new(0.0, 0.2, 0.0)).norm() < 1e-12);

        let p = Vert3d::new(1.2, 0.6, 3.0);
        let (proj, _) = ge.project(&p);
        assert!((proj - Vert3d::new(0.8, 0.2, 0.0)).norm() < 1e-12);

        let p = 0.1 * p0 + 0.2 * p1 + 0.7 * p2;
        let (proj, _) = ge.project(&p);
        assert!((proj - p).norm() < 1e-12);

        let (proj, _) = ge.project(&(p + Vert3d::new(0.0, 0.0, 2.0)));
        assert!((proj - p).norm() < 1e-12);
    }

    #[test]
    fn test_project_triangle_2d() {
        let p0 = Vert2d::new(0.0, 0.0);
        let p1 = Vert2d::new(1.0, 0.0);
        let p2 = Vert2d::new(0.0, 1.0);

        let ge = GTriangle([p0, p1, p2]);

        let p = Vert2d::new(-1.0, -1.0);
        let (proj, _) = ge.project(&p);
        assert!((proj - Vert2d::new(0.0, 0.0)).norm() < 1e-12);

        let p = Vert2d::new(2.0, -1.0);
        let (proj, _) = ge.project(&p);
        assert!((proj - Vert2d::new(1.0, 0.0)).norm() < 1e-12);

        let p = Vert2d::new(-2.0, 3.0);
        let (proj, _) = ge.project(&p);
        assert!((proj - Vert2d::new(0.0, 1.0)).norm() < 1e-12);

        let p = Vert2d::new(0.5, -2.0);
        let (proj, _) = ge.project(&p);
        assert!((proj - Vert2d::new(0.5, 0.0)).norm() < 1e-12);

        let p = Vert2d::new(-10.0, 0.2);
        let (proj, _) = ge.project(&p);
        assert!((proj - Vert2d::new(0.0, 0.2)).norm() < 1e-12);

        let p = Vert2d::new(1.2, 0.6);
        let (proj, _) = ge.project(&p);
        assert!((proj - Vert2d::new(0.8, 0.2)).norm() < 1e-12);

        let p = 0.1 * p0 + 0.2 * p1 + 0.7 * p2;
        let (proj, _) = ge.project(&p);
        assert!((proj - p).norm() < 1e-12);
    }

    #[test]
    fn test_circumcenter_2d() {
        let mut rng = StdRng::seed_from_u64(1234);

        for _ in 0..100 {
            let p0 = Vert2d::from_fn(|_, _| rng.random::<f64>() - 0.5);
            let p1 = Vert2d::from_fn(|_, _| rng.random::<f64>() - 0.5);
            let p2 = Vert2d::from_fn(|_, _| rng.random::<f64>() - 0.5);
            let ge = GTriangle([p0, p1, p2]);
            let bcoords = ge.circumcenter_bcoords();
            let p = ge.vert(&bcoords);
            let l0 = (p0 - p).norm();
            let l1 = (p1 - p).norm();
            let l2 = (p2 - p).norm();
            assert_delta!(l0, l1, 1e-12);
            assert_delta!(l0, l2, 1e-12);
        }
    }

    #[test]
    fn test_circumcenter_3d_tri() {
        let mut rng = StdRng::seed_from_u64(1234);

        for _ in 0..100 {
            let p0 = Vert2d::from_fn(|_, _| rng.random::<f64>() - 0.5);
            let p1 = Vert2d::from_fn(|_, _| rng.random::<f64>() - 0.5);
            let p2 = Vert2d::from_fn(|_, _| rng.random::<f64>() - 0.5);
            let ge = GTriangle([p0, p1, p2]);
            let bcoords = ge.circumcenter_bcoords();
            let p = ge.vert(&bcoords);
            let l0 = (p0 - p).norm();
            let l1 = (p1 - p).norm();
            let l2 = (p2 - p).norm();
            assert_delta!(l0, l1, 1e-12);
            assert_delta!(l0, l2, 1e-12);
        }
    }
}
