use crate::{
    Vertex,
    mesh::{GNode, GSimplex, Node, Simplex, elements::Idx},
};
use std::fmt::Debug;
use std::ops::Index;

/// Edge
#[derive(Default, Clone, Copy, PartialEq, Eq, Hash, Debug, Ord, PartialOrd)]
pub struct Edge<T: Idx>(pub(crate) [T; 2]);

impl<T: Idx> Edge<T> {
    #[must_use]
    pub fn new(i0: usize, i1: usize) -> Self {
        Self([i0.try_into().unwrap(), i1.try_into().unwrap()])
    }
}

impl<T: Idx> IntoIterator for Edge<T> {
    type Item = usize;
    type IntoIter = std::iter::Map<std::array::IntoIter<T, 2>, fn(T) -> usize>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter().map(|x| x.try_into().unwrap())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct GEdge<const D: usize>([Vertex<D>; 2]);

impl<const D: usize> GEdge<D> {
    #[must_use]
    pub const fn new(v0: &Vertex<D>, v1: &Vertex<D>) -> Self {
        Self([*v0, *v1])
    }

    #[must_use]
    pub fn as_vec(&self) -> Vertex<D> {
        self.0[1] - self.0[0]
    }
}

impl<const D: usize> Index<usize> for GEdge<D> {
    type Output = Vertex<D>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<const D: usize> IntoIterator for GEdge<D> {
    type Item = Vertex<D>;
    type IntoIter = std::array::IntoIter<Self::Item, 2>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<const D: usize> Default for GEdge<D> {
    fn default() -> Self {
        Self([Vertex::zeros(); 2])
    }
}

const EDGE2EDGES: [Edge<usize>; 1] = [Edge([0, 1])];
const EDGE2FACES: [Node<usize>; 2] = [Node([1]), Node([0])];

impl<T: Idx> Simplex for Edge<T> {
    type T = T;
    type FACE = Node<T>;
    type GEOM<const D: usize> = GEdge<D>;
    const DIM: usize = 1;
    const N_VERTS: usize = 2;
    const N_EDGES: usize = 1;
    const N_FACES: usize = 2;

    fn get(&self, index: usize) -> usize {
        self.0[index].try_into().unwrap()
    }

    fn edge(&self, i: usize) -> Edge<usize> {
        Edge::from_iter(EDGE2EDGES[i].into_iter().map(|j| self.get(j)))
    }

    fn face(&self, i: usize) -> Self::FACE {
        Self::FACE::from_iter(EDGE2FACES[i].into_iter().map(|j| self.get(j)))
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
        *self == *other
    }

    fn invert(&mut self) {
        self.0.swap(1, 0);
    }
}

impl<const D: usize> GSimplex<D> for GEdge<D> {
    const N_VERTS: usize = 2;
    type ARRAY<T: Debug + Default + Clone + Copy> = [T; 2];
    type BCOORDS = Self::ARRAY<f64>;
    type TOPO = Edge<usize>;
    type FACE = GNode<D>;

    fn ideal_vol() -> f64 {
        1.0
    }

    fn edge(&self, i: usize) -> Self {
        Self::from_iter(EDGE2EDGES[i].into_iter().map(|j| self[j]))
    }

    fn face(&self, i: usize) -> Self::FACE {
        Self::FACE::from_iter(EDGE2FACES[i].into_iter().map(|j| self[j]))
    }

    fn set(&mut self, i: usize, v: Vertex<D>) {
        self.0[i] = v;
    }

    fn has_normal() -> bool {
        D == 2
    }

    fn vol(&self) -> f64 {
        (self[1] - self[0]).norm()
    }

    fn integrate<G: Fn(&Self::BCOORDS) -> f64>(&self, f: G) -> f64 {
        let mut res = 0.0;
        for &(weight, v) in &super::quadratures::QUADRATURE_EDGE_3 {
            res += weight * f(&[1.0 - v, v]);
        }
        res * self.vol()
    }

    fn normal(&self, bcoords: Option<&Self::BCOORDS>) -> Vertex<D> {
        if Self::has_normal() {
            let n =
                Vertex::<D>::from_column_slice(&[self[1][1] - self[0][1], self[0][0] - self[1][0]]);
            if bcoords.is_some() { n.normalize() } else { n }
        } else {
            unreachable!()
        }
    }

    fn radius(&self) -> f64 {
        0.5 * (self[1] - self[0]).norm()
    }

    fn center_bcoords() -> Self::BCOORDS {
        [0.5, 0.5]
    }

    fn bcoords(&self, v: &Vertex<D>) -> Self::BCOORDS {
        let ab = self[1] - self[0];
        let ap = v - self[0];

        let ab_squared_magnitude = ab.norm_squared();
        let t = ap.dot(&ab) / ab_squared_magnitude;
        [1.0 - t, t]
    }

    fn gamma(&self) -> f64 {
        1.0
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
    use rand::{RngExt, SeedableRng, rngs::StdRng};

    use super::{Edge, GEdge};
    use crate::{
        Vert2d, assert_delta,
        mesh::{GSimplex, Simplex},
    };

    #[test]
    fn test_vol() {
        let v0 = Vert2d::new(0.0, 0.0);
        let v1 = Vert2d::new(0.5, 0.0);
        let ge = GEdge([v0, v1]);
        assert_delta!(ge.vol(), 0.5, 1e-12);
        let ge = GEdge([v1, v0]);
        assert_delta!(ge.vol(), 0.5, 1e-12);
    }

    #[test]
    fn test_is_same() {
        let e = Edge([10, 12]);
        let o = Edge([10, 10]);
        assert!(!e.is_same(&o));
        let o = Edge([12, 10]);
        assert!(!e.is_same(&o));
        let o = Edge([10, 12]);
        assert!(e.is_same(&o));
    }

    #[test]
    fn test_circumcenter_2d_edg() {
        let mut rng = StdRng::seed_from_u64(1234);

        for _ in 0..100 {
            let p0 = Vert2d::from_fn(|_, _| rng.random::<f64>() - 0.5);
            let p1 = Vert2d::from_fn(|_, _| rng.random::<f64>() - 0.5);
            let ge = GEdge([p0, p1]);
            let bcoords = ge.circumcenter_bcoords();
            let p = ge.vert(&bcoords);
            let l0 = (p0 - p).norm();
            let l1 = (p1 - p).norm();
            assert_delta!(l0, l1, 1e-12);
            assert_delta!(bcoords[0], 0.5, 1e-12);
            assert_delta!(bcoords[1], 0.5, 1e-12);
        }
    }

    #[test]
    fn test_bb() {
        let v0 = Vert2d::new(-1.0, 0.0);
        let v1 = Vert2d::new(1.0, -1.0);
        let ge = GEdge([v0, v1]);
        let (bb0, bb1) = ge.bounding_box();

        assert_delta!((bb0 - Vert2d::new(-1.0, -1.0)).norm(), 0., 1e-12);
        assert_delta!((bb1 - Vert2d::new(1.0, 0.0)).norm(), 0., 1e-12);
    }
}
