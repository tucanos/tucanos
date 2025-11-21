use crate::{
    Vertex,
    mesh::{Edge, GEdge, GSimplex, GTriangle, Simplex, Triangle, elements::Idx},
};
use nalgebra::{SMatrix, SVector};
use std::fmt::Debug;
use std::ops::Index;

/// Tetrahedron
#[derive(Default, Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Tetrahedron<T: Idx>([T; 4]);

impl<T: Idx> Tetrahedron<T> {
    #[must_use]
    pub fn new(i0: usize, i1: usize, i2: usize, i3: usize) -> Self {
        Self([
            i0.try_into().unwrap(),
            i1.try_into().unwrap(),
            i2.try_into().unwrap(),
            i3.try_into().unwrap(),
        ])
    }
}

impl<T: Idx> IntoIterator for Tetrahedron<T> {
    type Item = usize;
    type IntoIter = std::iter::Map<std::array::IntoIter<T, 4>, fn(T) -> usize>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter().map(|x| x.try_into().unwrap())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct GTetrahedron<const D: usize>([Vertex<D>; 4]);

impl<const D: usize> GTetrahedron<D> {
    #[must_use]
    pub const fn new(v0: &Vertex<D>, v1: &Vertex<D>, v2: &Vertex<D>, v3: &Vertex<D>) -> Self {
        Self([*v0, *v1, *v2, *v3])
    }
}

impl<const D: usize> Index<usize> for GTetrahedron<D> {
    type Output = Vertex<D>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<const D: usize> IntoIterator for GTetrahedron<D> {
    type Item = Vertex<D>;
    type IntoIter = std::array::IntoIter<Self::Item, 4>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<const D: usize> Default for GTetrahedron<D> {
    fn default() -> Self {
        Self([Vertex::zeros(); 4])
    }
}

const TETRA2EDGES: [Edge<usize>; 6] = [
    Edge([0, 1]),
    Edge([1, 2]),
    Edge([2, 0]),
    Edge([0, 3]),
    Edge([1, 3]),
    Edge([2, 3]),
];
const TETRA2FACES: [Triangle<usize>; 4] = [
    Triangle([1, 2, 3]),
    Triangle([2, 0, 3]),
    Triangle([0, 1, 3]),
    Triangle([0, 2, 1]),
];

impl<T: Idx> Simplex for Tetrahedron<T> {
    type T = T;
    type FACE = Triangle<T>;
    type GEOM<const D: usize> = GTetrahedron<D>;
    const DIM: usize = 3;
    const N_VERTS: usize = 4;
    const N_EDGES: usize = 6;
    const N_FACES: usize = 4;

    fn get(&self, index: usize) -> usize {
        self.0[index].try_into().unwrap()
    }

    fn edge(&self, i: usize) -> Edge<usize> {
        Edge::from_iter(TETRA2EDGES[i].into_iter().map(|j| self.get(j)))
    }

    fn face(&self, i: usize) -> Self::FACE {
        Self::FACE::from_iter(TETRA2FACES[i].into_iter().map(|j| self.get(j)))
    }

    fn set(&mut self, i: usize, v: usize) {
        self.0[i] = v.try_into().unwrap();
    }

    fn contains(&self, i: usize) -> bool {
        self.0.contains(&i.try_into().unwrap())
    }

    fn quadrature() -> (Vec<f64>, Vec<Vec<f64>>) {
        let weights = vec![0.25, 0.25, 0.25, 0.25];
        let pts = vec![
            vec![0.1381966011250105, 0.1381966011250105, 0.1381966011250105],
            vec![0.5854101966249685, 0.1381966011250105, 0.1381966011250105],
            vec![0.1381966011250105, 0.5854101966249685, 0.1381966011250105],
            vec![0.1381966011250105, 0.1381966011250105, 0.5854101966249685],
        ];
        (weights, pts)
    }

    fn sorted(&self) -> Self {
        let mut tmp = *self;
        tmp.0.sort_unstable();
        tmp
    }

    fn is_same(&self, other: &Self) -> bool {
        let f = self.face(0);
        other
            .0
            .iter()
            .position(|&x| x == self.0[0])
            .is_some_and(|i| f.is_same(&other.face(i)))
    }

    fn invert(&mut self) {
        self.0.swap(1, 0);
    }
}

impl<const D: usize> GSimplex<D> for GTetrahedron<D> {
    const N_VERTS: usize = 4;
    type ARRAY<T: Debug + Default + Clone + Copy> = [T; 4];
    type BCOORDS = Self::ARRAY<f64>;
    type TOPO = Tetrahedron<usize>;
    type FACE = GTriangle<D>;

    fn ideal_vol() -> f64 {
        1.0 / (6.0 * std::f64::consts::SQRT_2)
    }

    fn edge(&self, i: usize) -> GEdge<D> {
        GEdge::from_iter(TETRA2EDGES[i].into_iter().map(|j| self[j]))
    }

    fn face(&self, i: usize) -> Self::FACE {
        Self::FACE::from_iter(TETRA2FACES[i].into_iter().map(|j| self[j]))
    }

    fn set(&mut self, i: usize, v: Vertex<D>) {
        self.0[i] = v;
    }

    fn has_normal() -> bool {
        false
    }

    fn vol(&self) -> f64 {
        let e1 = self[1] - self[0];
        let e2 = self[2] - self[0];
        let e3 = self[3] - self[0];

        e3.dot(&e1.cross(&e2)) / 6.0
    }

    fn normal(&self) -> Vertex<D> {
        unreachable!()
    }

    fn radius(&self) -> f64 {
        let a0 = self.face(0).vol();
        let a1 = self.face(1).vol();
        let a2 = self.face(2).vol();
        let a3 = self.face(3).vol();
        let v = self.vol();
        3.0 * v / (a0 + a1 + a2 + a3)
    }

    fn bcoords(&self, v: &Vertex<D>) -> Self::BCOORDS {
        let a = SMatrix::<f64, 4, 4>::new(
            1.0, 1.0, 1.0, 1.0, self[0][0], self[1][0], self[2][0], self[3][0], self[0][1],
            self[1][1], self[2][1], self[3][1], self[0][2], self[1][2], self[2][2], self[3][2],
        );
        let b = SVector::<f64, 4>::new(1., v[0], v[1], v[2]);
        let decomp = a.lu();
        let x = decomp.solve(&b).unwrap();
        [x[0], x[1], x[2], x[3]]
    }

    fn gamma(&self) -> f64 {
        let vol = self.vol();
        if vol < f64::EPSILON {
            return 0.0;
        }

        let a = self[1] - self[0];
        let b = self[2] - self[0];
        let c = self[3] - self[0];

        let aa = self[3] - self[2];
        let bb = self[3] - self[1];
        let cc = self[2] - self[1];

        let la = a.norm_squared();
        let lb = b.norm_squared();
        let lc = c.norm_squared();
        let laa = aa.norm_squared();
        let lbb = bb.norm_squared();
        let lcc = cc.norm_squared();

        let lalaa = (la * laa).sqrt();
        let lblbb = (lb * lbb).sqrt();
        let lclcc = (lc * lcc).sqrt();

        let tmp = (lalaa + lblbb + lclcc)
            * (lalaa + lblbb - lclcc)
            * (lalaa - lblbb + lclcc)
            * (-lalaa + lblbb + lclcc);

        // This happens when the 4 points are (nearly) co-planar
        // => R is actually undetermined but the quality is (close to) zero
        if tmp < f64::EPSILON {
            return 0.0;
        }

        let r = tmp.sqrt() / 24.0 / vol;

        let s1 = self.face(0).vol();
        let s2 = self.face(1).vol();
        let s3 = self.face(2).vol();
        let s4 = self.face(3).vol();
        let rho = 9.0 * vol / (s1 + s2 + s3 + s4);

        rho / r
    }
}

#[cfg(test)]
mod tests {
    use super::{GTetrahedron, Tetrahedron};
    use crate::{
        Vert3d, assert_delta,
        mesh::{GSimplex, Simplex},
    };
    use rand::{Rng, SeedableRng, rngs::StdRng, seq::SliceRandom};

    #[test]
    fn test_vol() {
        let v0 = Vert3d::new(0.0, 0.0, 0.0);
        let v1 = Vert3d::new(1.0, 0.0, 0.0);
        let v2 = Vert3d::new(0.0, 1.0, 0.0);
        let v3 = Vert3d::new(0.0, 0.0, 1.0);
        let ge = GTetrahedron([v0, v1, v2, v3]);
        assert_delta!(ge.vol(), 1.0 / 6.0, 1e-12);
        let ge = GTetrahedron([v0, v2, v1, v3]);
        assert_delta!(ge.vol(), -1.0 / 6.0, 1e-12);
    }

    #[test]
    fn test_is_same() {
        let pts = [
            Vert3d::new(0.0, 0.0, 0.0),
            Vert3d::new(1.0, 0.0, 0.0),
            Vert3d::new(0.0, 1.0, 0.0),
            Vert3d::new(0.0, 0.0, 1.0),
        ];

        let mut rng = StdRng::seed_from_u64(1234);

        let gt = |e: &Tetrahedron<usize>| {
            GTetrahedron([pts[e.get(0)], pts[e.get(1)], pts[e.get(2)], pts[e.get(3)]])
        };
        let e = Tetrahedron([0, 1, 2, 3]);
        let mut o = e;

        for _ in 0..10 {
            o.0.shuffle(&mut rng);
            let is_same = e.is_same(&o);
            let n = gt(&o).vol();
            if is_same {
                assert!(n > 0.0);
            } else {
                assert!(n < 0.0);
            }
        }
    }

    #[test]
    fn test_circumcenter_3d() {
        let mut rng = StdRng::seed_from_u64(1234);

        for _ in 0..100 {
            let p0 = Vert3d::from_fn(|_, _| rng.random::<f64>() - 0.5);
            let p1 = Vert3d::from_fn(|_, _| rng.random::<f64>() - 0.5);
            let p2 = Vert3d::from_fn(|_, _| rng.random::<f64>() - 0.5);
            let p3 = Vert3d::from_fn(|_, _| rng.random::<f64>() - 0.5);
            let ge = GTetrahedron([p0, p1, p2, p3]);
            let bcoords = ge.circumcenter_bcoords();
            let p = ge.vert(&bcoords);
            let l0 = (p0 - p).norm();
            let l1 = (p1 - p).norm();
            let l2 = (p2 - p).norm();
            let l3 = (p3 - p).norm();
            assert_delta!(l0, l1, 1e-12);
            assert_delta!(l0, l2, 1e-12);
            assert_delta!(l0, l3, 1e-12);
        }
    }
}
