//! Simplex elements
use crate::{Edge, Node, Tetrahedron, Triangle, Vertex};

/// Simplex elements
pub trait Simplex<const C: usize>: Sized {
    /// Get the edges for the simplex `(0, .., C-1)`
    fn edges() -> Vec<Edge>;

    /// Get the volume of a simplex
    fn vol<const D: usize>(v: [&Vertex<D>; C]) -> f64;

    /// Check if a normal can be computed in D dimensions
    fn has_normal<const D: usize>() -> bool {
        D == C
    }

    /// Normal to the vertex
    fn normal<const D: usize>(v: [&Vertex<D>; C]) -> Vertex<D>;

    /// Radius (=diameter of the inner circle / sphere)
    fn radius<const D: usize>(v: [&Vertex<D>; C]) -> f64;

    /// Get a quadrature (weights and points)
    fn quadrature() -> (Vec<f64>, Vec<Vec<f64>>);

    /// Sort the vertex indices
    fn sorted(&self) -> Self;

    /// Check of two elements are the same (allowing circular permutation)
    fn is_same(&self, other: &Self) -> bool;
}

fn is_circ_perm<const N: usize>(a: &[usize; N], b: &[usize; N]) -> bool {
    let mut tmp = *b;
    for i in 0..N {
        if tmp == *a {
            return true;
        }
        if i != N - 1 {
            tmp.rotate_right(1);
        }
    }
    false
}

impl Simplex<1> for Node {
    fn edges() -> Vec<Edge> {
        unreachable!()
    }

    fn vol<const D: usize>(_v: [&Vertex<D>; 1]) -> f64 {
        unreachable!()
    }

    fn normal<const D: usize>(_v: [&Vertex<D>; 1]) -> Vertex<D> {
        unreachable!()
    }

    fn radius<const D: usize>(_v: [&Vertex<D>; 1]) -> f64 {
        unreachable!()
    }

    fn quadrature() -> (Vec<f64>, Vec<Vec<f64>>) {
        unreachable!()
    }

    fn sorted(&self) -> Self {
        *self
    }

    fn is_same(&self, other: &Self) -> bool {
        self[0] == other[0]
    }
}

pub(crate) const EDGE_FACES: [Node; 2] = [[0], [1]];

impl Simplex<2> for Edge {
    fn edges() -> Vec<Edge> {
        vec![[0, 1]]
    }

    fn vol<const D: usize>(v: [&Vertex<D>; 2]) -> f64 {
        (v[1] - v[0]).norm()
    }

    fn normal<const D: usize>(v: [&Vertex<D>; 2]) -> Vertex<D> {
        if Self::has_normal::<D>() {
            Vertex::<D>::from_column_slice(&[v[1][1] - v[0][1], v[0][0] - v[1][0]])
        } else {
            unreachable!()
        }
    }

    fn radius<const D: usize>(v: [&Vertex<D>; 2]) -> f64 {
        0.5 * (v[1] - v[0]).norm()
    }

    fn quadrature() -> (Vec<f64>, Vec<Vec<f64>>) {
        let weights = vec![5.0 / 18.0, 8.0 / 18.0, 5.0 / 18.0];
        let pts = vec![
            vec![0.5 - 0.5 * (3.0_f64 / 5.0).sqrt()],
            vec![0.5],
            vec![0.5 + 0.5 * (3.0_f64 / 5.0).sqrt()],
        ];
        (weights, pts)
    }

    fn sorted(&self) -> Self {
        let mut tmp = *self;
        tmp.sort();
        tmp
    }

    fn is_same(&self, other: &Self) -> bool {
        is_circ_perm(self, other)
    }
}

pub(crate) const TRIANGLE_FACES: [Edge; 3] = [[0, 1], [1, 2], [2, 0]];

impl Simplex<3> for Triangle {
    fn edges() -> Vec<Edge> {
        vec![[0, 1], [1, 2], [2, 0]]
    }

    fn vol<const D: usize>(v: [&Vertex<D>; 3]) -> f64 {
        if Self::has_normal::<D>() {
            Self::normal(v).norm()
        } else {
            assert_eq!(D, 2);
            let e1 = v[1] - v[0];
            let e2 = v[2] - v[0];

            0.5 * (e1[0] * e2[1] - e1[1] * e2[0])
        }
    }

    fn normal<const D: usize>(v: [&Vertex<D>; 3]) -> Vertex<D> {
        if Self::has_normal::<D>() {
            let e1 = v[1] - v[0];
            let e2 = v[2] - v[0];
            0.5 * e1.cross(&e2)
        } else {
            unreachable!()
        }
    }

    fn radius<const D: usize>(v: [&Vertex<D>; 3]) -> f64 {
        let a = (v[2] - v[1]).norm();
        let b = (v[2] - v[0]).norm();
        let c = (v[1] - v[0]).norm();
        let s = 0.5 * (a + b + c);
        ((s - a) * (s - b) * (s - c) / s).sqrt()
    }

    fn quadrature() -> (Vec<f64>, Vec<Vec<f64>>) {
        let weights = vec![1. / 3., 1. / 3., 1. / 3.];
        let pts = vec![
            vec![2. / 3., 1. / 6.],
            vec![1. / 6., 2. / 3.],
            vec![1. / 6., 1. / 6.],
        ];
        (weights, pts)
    }

    fn sorted(&self) -> Self {
        let mut tmp = *self;
        tmp.sort();
        tmp
    }

    fn is_same(&self, other: &Self) -> bool {
        is_circ_perm(self, other)
    }
}

pub(crate) const TETRA_FACES: [Triangle; 4] = [[1, 2, 3], [2, 0, 3], [0, 1, 3], [0, 2, 1]];
impl Simplex<4> for Tetrahedron {
    fn edges() -> Vec<Edge> {
        vec![[0, 1], [1, 2], [2, 0], [0, 3], [1, 3], [2, 3]]
    }

    fn vol<const D: usize>(v: [&Vertex<D>; 4]) -> f64 {
        let e1 = v[1] - v[0];
        let e2 = v[2] - v[0];
        let e3 = v[3] - v[0];

        e3.dot(&e1.cross(&e2)) / 6.0
    }

    fn normal<const D: usize>(_v: [&Vertex<D>; 4]) -> Vertex<D> {
        unreachable!()
    }

    fn radius<const D: usize>(_v: [&Vertex<D>; 4]) -> f64 {
        unimplemented!()
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
        tmp.sort();
        tmp
    }

    fn is_same(&self, other: &Self) -> bool {
        is_circ_perm(self, other)
    }
}
