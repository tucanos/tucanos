//! Simplex elements
use super::{Edge, Node, Tetrahedron, Triangle, twovec};
use crate::Vertex;
use crate::mesh::GNode;
use nalgebra::{SMatrix, SVector};
use rustc_hash::FxHashMap;
use std::array::IntoIter;
use std::fmt::Debug;
use std::ops::{Index, IndexMut};

pub trait Cell:
    Sized
    + Index<usize, Output = usize>
    + IndexMut<usize, Output = usize>
    + IntoIterator<Item = usize>
    + Default
    + Debug
    + Send
    + Sync
    + Copy
    + Clone
{
    type GEOM<const D: usize>: GCell<D>;
    fn from_slice(s: &[usize]) -> Self;
    fn from_iter<I: Iterator<Item = usize>>(iter: I) -> Self;
    fn to_slice(&self) -> &[usize];
    fn contains(&self, i: usize) -> bool;
}

pub trait GCell<const D: usize>:
    Sized
    + Index<usize, Output = Vertex<D>>
    + IntoIterator<Item = Vertex<D>>
    + Default
    + Debug
    + Send
    + Sync
    + Copy
    + Clone
{
    fn from_slice(s: &[Vertex<D>]) -> Self;
    fn to_slice(&self) -> &[Vertex<D>];
}

/// Simplex elements
pub trait Simplex: Cell
where
    <Self as Simplex>::FACE: 'static,
{
    type FACE: Simplex;
    const DIM: usize;
    const N_EDGES: usize;
    const N_FACES: usize;
    const EDGES: &'static [Edge];
    const FACES: &'static [Self::FACE];

    /// Get the i-th edge for the current simplex
    fn edge(&self, i: usize) -> Edge {
        Edge::from_iter(Self::EDGES[i].into_iter().map(|j| self[j]))
    }

    /// Get the i-th face for the current simplex
    fn face<const F: usize>(&self, i: usize) -> Self::FACE {
        Self::FACE::from_iter(Self::FACES[i].into_iter().map(|j| self[j]))
    }

    /// Check if a normal can be computed in D dimensions
    #[must_use]
    fn has_normal<const D: usize>() -> bool {
        D == Self::DIM + 1
    }

    /// Get a quadrature (weights and points)
    fn quadrature() -> (Vec<f64>, Vec<Vec<f64>>);

    /// Sort the vertex indices
    #[must_use]
    fn sorted(&self) -> Self;

    /// Check of two elements are the same (allowing circular permutation)
    fn is_same(&self, other: &Self) -> bool;

    /// Invert the element
    fn invert(&mut self);
}

pub trait GSimplex<const D: usize>: GCell<D> {
    type BCOORDS: IntoIterator<Item = f64> + Debug + Clone + Copy;

    /// Get the volume of a simplex
    fn vol(&self) -> f64;
    /// Normal to the vertex
    fn normal(&self) -> Vertex<D>;

    /// Radius (=diameter of the inner circle / sphere)
    fn radius(&self) -> f64;

    /// Barycentric coordinates
    fn bcoords(&self, v: &Vertex<D>) -> Self::BCOORDS;

    /// Vertex from barycentric coordinates
    fn vert(&self, bcoords: &Self::BCOORDS) -> Vertex<D>;

    /// Gamma quality measure, ratio of inscribed radius to circumradius
    /// normalized to be between 0 and 1
    fn gamma(&self) -> f64;

    /// Project a point on the simplex
    fn project(&self, v: &Vertex<D>) -> Vertex<D>;

    /// Try to project a point inside the simplex
    #[must_use]
    fn project_inside(&self, v: &Vertex<D>) -> Option<Vertex<D>> {
        let bcoords = self.bcoords(v);
        let p = self.vert(&bcoords);
        if bcoords.into_iter().all(|x| x > 0.0) {
            Some(p)
        } else {
            None
        }
    }

    /// Distance from a point to the simplex
    #[must_use]
    fn distance(&self, v: &Vertex<D>) -> f64 {
        let pt = self.project(v);
        (v - pt).norm()
    }
}

const NODE2EDGES: [Edge; 0] = [];
const NODE2FACES: [Node; 1] = [Node([0])];

impl Index<usize> for Node {
    type Output = usize;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl IndexMut<usize> for Node {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl IntoIterator for Node {
    type Item = usize;
    type IntoIter = IntoIter<usize, 1>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<const D: usize> Index<usize> for GNode<D> {
    type Output = Vertex<D>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<const D: usize> IndexMut<usize> for GNode<D> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl<const D: usize> IntoIterator for GNode<D> {
    type Item = Vertex<D>;
    type IntoIter = IntoIter<Vertex<D>, 1>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl Cell for Node {
    type GEOM<const D: usize> = GNode<D>;

    fn from_slice(s: &[usize]) -> Self {
        Self(s.try_into().unwrap())
    }

    fn from_iter<I: Iterator<Item = usize>>(iter: I) -> Self {
        let mut res = Self::default();
        for (i, j) in iter.enumerate() {
            res.0[i] = j;
        }
        res
    }

    fn to_slice(&self) -> &[usize] {
        &self.0
    }

    fn contains(&self, i: usize) -> bool {
        self.0.contains(&i)
    }
}

impl<const D: usize> GCell<D> for GNode<D> {
    fn from_slice(s: &[Vertex<D>]) -> Self {
        Self(s.try_into().unwrap())
    }

    fn to_slice(&self) -> &[Vertex<D>] {
        &self.0
    }
}

impl Simplex for Node {
    type FACE = Node;
    const DIM: usize = 0;
    const N_EDGES: usize = 0;
    const N_FACES: usize = 1;
    const EDGES: &'static [Edge] = &NODE2EDGES;
    const FACES: &'static [Self::FACE] = &NODE2FACES;

    fn quadrature() -> (Vec<f64>, Vec<Vec<f64>>) {
        unreachable!()
    }

    fn sorted(&self) -> Self {
        *self
    }

    fn is_same(&self, other: &Self) -> bool {
        self[0] == other[0]
    }

    fn invert(&mut self) {}
}

impl<const D: usize> GSimplex<D> for GNode<D> {
    type BCOORDS = [f64; 1];

    fn vol(&self) -> f64 {
        unreachable!()
    }

    fn normal(&self) -> Vertex<D> {
        unreachable!()
    }

    fn radius(&self) -> f64 {
        unreachable!()
    }

    fn bcoords(&self, _v: &Vertex<D>) -> Self::BCOORDS {
        unreachable!()
    }

    fn vert(&self, _bcoords: &Self::BCOORDS) -> Vertex<D> {
        unreachable!()
    }

    fn gamma(&self) -> f64 {
        unreachable!()
    }

    fn project(&self, _v: &Vertex<D>) -> Vertex<D> {
        unreachable!()
    }
}

pub const EDGE_FACES: [Node; 2] = [[0], [1]];

impl Simplex<2> for Edge {
    fn edges() -> Vec<Edge> {
        vec![[0, 1]]
    }

    fn edge(&self, i: usize) -> Edge {
        match i {
            0 => [self[0], self[1]],
            _ => unreachable!(),
        }
    }

    fn faces<const F: usize>() -> Vec<Face<F>> {
        debug_assert_eq!(F, 1);
        EDGE_FACES
            .iter()
            .map(|x| x.as_slice().try_into().unwrap())
            .collect()
    }

    fn face<const F: usize>(&self, i: usize) -> Face<F> {
        debug_assert_eq!(F, 1);
        let [i0] = EDGE_FACES[i];
        [self[i0]].as_slice().try_into().unwrap()
    }

    fn vol<const D: usize>(v: &[Vertex<D>; 2]) -> f64 {
        (v[1] - v[0]).norm()
    }

    fn normal<const D: usize>(v: &[Vertex<D>; 2]) -> Vertex<D> {
        if Self::has_normal::<D>() {
            Vertex::<D>::from_column_slice(&[v[1][1] - v[0][1], v[0][0] - v[1][0]])
        } else {
            unreachable!()
        }
    }

    fn radius<const D: usize>(v: &[Vertex<D>; 2]) -> f64 {
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
        tmp.sort_unstable();
        tmp
    }

    fn is_same(&self, other: &Self) -> bool {
        *self == *other
    }

    fn invert(&mut self) {
        self.swap(1, 0);
    }

    fn bcoords<const D: usize>(ge: &[Vertex<D>; 2], v: &Vertex<D>) -> [f64; 2] {
        let ab = ge[1] - ge[0];
        let ap = v - ge[0];

        let ab_squared_magnitude = ab.norm_squared();
        let t = ap.dot(&ab) / ab_squared_magnitude;
        [1.0 - t, t]
    }

    fn vert<const D: usize>(ge: &[Vertex<D>; 2], bcoords: [f64; 2]) -> Vertex<D> {
        bcoords[0] * ge[0] + bcoords[1] * ge[1]
    }

    fn gamma<const D: usize>(_ge: &[Vertex<D>; 2]) -> f64 {
        1.0
    }

    fn project<const D: usize>(ge: &[Vertex<D>; 2], v: &Vertex<D>) -> Vertex<D> {
        Self::project_inside(ge, v).map_or_else(
            || {
                let p0 = Node::project(&[ge[0]], v);
                let d0 = (v - p0).norm_squared();
                let p1 = Node::project(&[ge[1]], v);
                let d1 = (v - p1).norm_squared();
                if d0 < d1 { p0 } else { p1 }
            },
            |pt| pt,
        )
    }
}

pub const TRIANGLE_FACES: [Edge; 3] = [[0, 1], [1, 2], [2, 0]];

impl Simplex<3> for Triangle {
    fn edges() -> Vec<Edge> {
        vec![[0, 1], [1, 2], [2, 0]]
    }

    fn edge(&self, i: usize) -> Edge {
        match i {
            0 => [self[0], self[1]],
            1 => [self[1], self[2]],
            2 => [self[2], self[0]],
            _ => unreachable!(),
        }
    }

    fn faces<const F: usize>() -> Vec<Face<F>> {
        debug_assert_eq!(F, 2);
        TRIANGLE_FACES
            .iter()
            .map(|x| x.as_slice().try_into().unwrap())
            .collect()
    }

    fn face<const F: usize>(&self, i: usize) -> Face<F> {
        debug_assert_eq!(F, 2);
        let [i0, i1] = TRIANGLE_FACES[i];
        [self[i0], self[i1]].as_slice().try_into().unwrap()
    }

    fn vol<const D: usize>(v: &[Vertex<D>; 3]) -> f64 {
        if Self::has_normal::<D>() {
            Self::normal(v).norm()
        } else {
            assert_eq!(D, 2);
            let e1 = v[1] - v[0];
            let e2 = v[2] - v[0];

            0.5 * (e1[0] * e2[1] - e1[1] * e2[0])
        }
    }

    fn normal<const D: usize>(v: &[Vertex<D>; 3]) -> Vertex<D> {
        if Self::has_normal::<D>() {
            let e1 = v[1] - v[0];
            let e2 = v[2] - v[0];
            0.5 * e1.cross(&e2)
        } else {
            unreachable!()
        }
    }

    fn radius<const D: usize>(v: &[Vertex<D>; 3]) -> f64 {
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
        tmp.sort_unstable();
        tmp
    }

    fn is_same(&self, other: &Self) -> bool {
        let [i0, i1, i2] = *self;
        *other == *self || *other == [i1, i2, i0] || *other == [i2, i0, i1]
    }

    fn invert(&mut self) {
        self.swap(1, 0);
    }

    fn bcoords<const D: usize>(ge: &[Vertex<D>; 3], v: &Vertex<D>) -> [f64; 3] {
        if D == 2 {
            let a = SMatrix::<f64, 3, 3>::new(
                1.0, 1.0, 1.0, ge[0][0], ge[1][0], ge[2][0], ge[0][1], ge[1][1], ge[2][1],
            );
            let b = SVector::<f64, 3>::new(1., v[0], v[1]);
            let decomp = a.lu();
            let x = decomp.solve(&b).unwrap();
            [x[0], x[1], x[2]]
        } else {
            let e0 = ge[1] - ge[0];
            let e1 = ge[2] - ge[0];
            let n = e0.cross(&e1);
            let w = v - ge[0];
            let nrm = n.norm_squared();
            let gamma = e0.cross(&w).dot(&n) / nrm;
            let beta = w.cross(&e1).dot(&n) / nrm;
            [1.0 - beta - gamma, beta, gamma]
        }
    }

    fn vert<const D: usize>(ge: &[Vertex<D>; 3], bcoords: [f64; 3]) -> Vertex<D> {
        ge[0] * bcoords[0] + ge[1] * bcoords[1] + ge[2] * bcoords[2]
    }

    fn gamma<const D: usize>(ge: &[Vertex<D>; 3]) -> f64 {
        let mut a = ge[2] - ge[1];
        let mut b = ge[0] - ge[2];
        let mut c = ge[1] - ge[0];

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

    fn project<const D: usize>(ge: &[Vertex<D>; 3], v: &Vertex<D>) -> Vertex<D> {
        Self::project_inside(ge, v).map_or_else(
            || {
                let p0 = Edge::project(&[ge[TRIANGLE_FACES[0][0]], ge[TRIANGLE_FACES[0][1]]], v);
                let d0 = (v - p0).norm_squared();
                let p1 = Edge::project(&[ge[TRIANGLE_FACES[1][0]], ge[TRIANGLE_FACES[1][1]]], v);
                let d1 = (v - p1).norm_squared();
                let p2 = Edge::project(&[ge[TRIANGLE_FACES[2][0]], ge[TRIANGLE_FACES[2][1]]], v);
                let d2 = (v - p2).norm_squared();
                if d0 < d1 && d0 < d2 {
                    p0
                } else if d1 < d2 {
                    p1
                } else {
                    p2
                }
            },
            |pt| pt,
        )
    }
}

pub const TETRA_FACES: [Triangle; 4] = [[1, 2, 3], [2, 0, 3], [0, 1, 3], [0, 2, 1]];
impl Simplex<4> for Tetrahedron {
    fn edges() -> Vec<Edge> {
        vec![[0, 1], [1, 2], [2, 0], [0, 3], [1, 3], [2, 3]]
    }

    fn edge(&self, i: usize) -> Edge {
        match i {
            0 => [self[0], self[1]],
            1 => [self[1], self[2]],
            2 => [self[2], self[0]],
            3 => [self[0], self[3]],
            4 => [self[1], self[3]],
            5 => [self[2], self[3]],
            _ => unreachable!(),
        }
    }

    fn faces<const F: usize>() -> Vec<Face<F>> {
        debug_assert_eq!(F, 3);
        TETRA_FACES
            .iter()
            .map(|x| x.as_slice().try_into().unwrap())
            .collect()
    }

    fn face<const F: usize>(&self, i: usize) -> Face<F> {
        debug_assert_eq!(F, 3);
        let [i0, i1, i2] = TETRA_FACES[i];
        [self[i0], self[i1], self[i2]]
            .as_slice()
            .try_into()
            .unwrap()
    }

    fn vol<const D: usize>(v: &[Vertex<D>; 4]) -> f64 {
        let e1 = v[1] - v[0];
        let e2 = v[2] - v[0];
        let e3 = v[3] - v[0];

        e3.dot(&e1.cross(&e2)) / 6.0
    }

    fn normal<const D: usize>(_v: &[Vertex<D>; 4]) -> Vertex<D> {
        unreachable!()
    }

    fn radius<const D: usize>(v: &[Vertex<D>; 4]) -> f64 {
        let a0 = Triangle::vol(&[v[0], v[1], v[2]]);
        let a1 = Triangle::vol(&[v[0], v[1], v[3]]);
        let a2 = Triangle::vol(&[v[1], v[2], v[3]]);
        let a3 = Triangle::vol(&[v[2], v[0], v[3]]);
        let v = Self::vol(v);
        3.0 * v / (a0 + a1 + a2 + a3)
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
        tmp.sort_unstable();
        tmp
    }

    fn is_same(&self, other: &Self) -> bool {
        let f = [self[1], self[2], self[3]];
        other.iter().position(|&x| x == self[0]).is_some_and(|i| {
            let o = [
                other[TETRA_FACES[i][0]],
                other[TETRA_FACES[i][1]],
                other[TETRA_FACES[i][2]],
            ];
            f.is_same(&o)
        })
    }

    fn invert(&mut self) {
        self.swap(1, 0);
    }

    fn bcoords<const D: usize>(ge: &[Vertex<D>; 4], v: &Vertex<D>) -> [f64; 4] {
        let a = SMatrix::<f64, 4, 4>::new(
            1.0, 1.0, 1.0, 1.0, ge[0][0], ge[1][0], ge[2][0], ge[3][0], ge[0][1], ge[1][1],
            ge[2][1], ge[3][1], ge[0][2], ge[1][2], ge[2][2], ge[3][2],
        );
        let b = SVector::<f64, 4>::new(1., v[0], v[1], v[2]);
        let decomp = a.lu();
        let x = decomp.solve(&b).unwrap();
        [x[0], x[1], x[2], x[3]]
    }

    fn vert<const D: usize>(ge: &[Vertex<D>; 4], bcoords: [f64; 4]) -> Vertex<D> {
        ge[0] * bcoords[0] + ge[1] * bcoords[1] + ge[2] * bcoords[2] + ge[3] * bcoords[3]
    }

    fn gamma<const D: usize>(ge: &[Vertex<D>; 4]) -> f64 {
        let vol = Self::vol(ge);
        if vol < f64::EPSILON {
            return 0.0;
        }

        let a = ge[1] - ge[0];
        let b = ge[2] - ge[0];
        let c = ge[3] - ge[0];

        let aa = ge[3] - ge[2];
        let bb = ge[3] - ge[1];
        let cc = ge[2] - ge[1];

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

        let s1 = Face::<3>::vol(&[
            ge[TETRA_FACES[0][0]],
            ge[TETRA_FACES[0][1]],
            ge[TETRA_FACES[0][2]],
        ]);
        let s2 = Face::<3>::vol(&[
            ge[TETRA_FACES[1][0]],
            ge[TETRA_FACES[1][1]],
            ge[TETRA_FACES[1][2]],
        ]);
        let s3 = Face::<3>::vol(&[
            ge[TETRA_FACES[2][0]],
            ge[TETRA_FACES[2][1]],
            ge[TETRA_FACES[2][2]],
        ]);
        let s4 = Face::<3>::vol(&[
            ge[TETRA_FACES[3][0]],
            ge[TETRA_FACES[3][1]],
            ge[TETRA_FACES[3][2]],
        ]);
        let rho = 9.0 * vol / (s1 + s2 + s3 + s4);

        rho / r
    }

    fn project<const D: usize>(ge: &[Vertex<D>; 4], v: &Vertex<D>) -> Vertex<D> {
        Self::project_inside(ge, v).map_or_else(
            || {
                let p0 = Triangle::project(
                    &[
                        ge[TETRA_FACES[0][0]],
                        ge[TETRA_FACES[0][1]],
                        ge[TETRA_FACES[0][2]],
                    ],
                    v,
                );
                let d0 = (v - p0).norm_squared();
                let p1 = Triangle::project(
                    &[
                        ge[TETRA_FACES[1][0]],
                        ge[TETRA_FACES[1][1]],
                        ge[TETRA_FACES[1][2]],
                    ],
                    v,
                );
                let d1 = (v - p1).norm_squared();
                let p2 = Triangle::project(
                    &[
                        ge[TETRA_FACES[2][0]],
                        ge[TETRA_FACES[2][1]],
                        ge[TETRA_FACES[2][2]],
                    ],
                    v,
                );
                let d2 = (v - p2).norm_squared();
                let p3 = Triangle::project(
                    &[
                        ge[TETRA_FACES[3][0]],
                        ge[TETRA_FACES[3][1]],
                        ge[TETRA_FACES[3][2]],
                    ],
                    v,
                );
                let d3 = (v - p3).norm_squared();
                if d0 < d1 && d0 < d2 && d0 < d3 {
                    p0
                } else if d1 < d2 && d1 < d3 {
                    p1
                } else if d2 < d3 {
                    p2
                } else {
                    p3
                }
            },
            |pt| pt,
        )
    }
}

/// Compute a `FxHashMap` that maps face-to-vertex connectivity (sorted) to a vector of element indices
#[must_use]
pub fn get_face_to_elem<
    'a,
    const C: usize,
    const F: usize,
    I: ExactSizeIterator<Item = &'a Cell<C>>,
>(
    elems: I,
) -> FxHashMap<Face<F>, twovec::Vec<usize>>
where
    Cell<C>: Simplex<C>,
{
    let mut map: FxHashMap<Face<F>, twovec::Vec<usize>> = FxHashMap::default();
    for (i_elem, elem) in elems.enumerate() {
        for i_face in 0..Cell::<C>::n_faces() {
            let mut f = elem.face(i_face);
            f.sort_unstable();
            let n = map.get_mut(&f);
            if let Some(n) = n {
                n.push(i_elem);
            } else {
                map.insert(f, twovec::Vec::with_single(i_elem));
            }
        }
    }

    map
}

#[cfg(test)]
mod tests {
    use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};

    use crate::{
        Vert2d, Vert3d,
        mesh::{Simplex, Tetrahedron, Triangle},
    };

    #[test]
    fn test_is_same() {
        let e = [10, 12];
        let o = [10, 10];
        assert!(!e.is_same(&o));
        let o = [12, 10];
        assert!(!e.is_same(&o));
        let o = [10, 12];
        assert!(e.is_same(&o));

        let pts = [
            Vert3d::new(0.0, 0.0, 0.0),
            Vert3d::new(1.0, 0.0, 0.0),
            Vert3d::new(0.0, 1.0, 0.0),
            Vert3d::new(0.0, 0.0, 1.0),
        ];

        let gt = |e: &Triangle| [pts[e[0]], pts[e[1]], pts[e[2]]];
        let e = [0, 1, 2];
        let mut o = e;

        let mut rng = StdRng::seed_from_u64(1234);
        for _ in 0..10 {
            o.shuffle(&mut rng);
            let is_same = e.is_same(&o);
            let n = Triangle::normal(&gt(&o));
            if is_same {
                assert!(n[2] > 0.0);
            } else {
                assert!(n[2] < 0.0);
            }
        }

        let gt = |e: &Tetrahedron| [pts[e[0]], pts[e[1]], pts[e[2]], pts[e[3]]];
        let e = [0, 1, 2, 3];
        let mut o = e;

        for _ in 0..10 {
            o.shuffle(&mut rng);
            let is_same = e.is_same(&o);
            let n = Tetrahedron::vol(&gt(&o));
            if is_same {
                assert!(n > 0.0);
            } else {
                assert!(n < 0.0);
            }
        }
    }

    #[test]
    fn test_project_triangle() {
        let p0 = Vert3d::new(0.0, 0.0, 0.0);
        let p1 = Vert3d::new(1.0, 0.0, 0.0);
        let p2 = Vert3d::new(0.0, 1.0, 0.0);

        let ge = [p0, p1, p2];

        let p = Vert3d::new(-1.0, -1.0, 1.0);
        let proj = Triangle::project(&ge, &p);
        assert!((proj - Vert3d::new(0.0, 0.0, 0.0)).norm() < 1e-12);

        let p = Vert3d::new(2.0, -1.0, 1.0);
        let proj = Triangle::project(&ge, &p);
        assert!((proj - Vert3d::new(1.0, 0.0, 0.0)).norm() < 1e-12);

        let p = Vert3d::new(-2.0, 3.0, -1.0);
        let proj = Triangle::project(&ge, &p);
        assert!((proj - Vert3d::new(0.0, 1.0, 0.0)).norm() < 1e-12);

        let p = Vert3d::new(0.5, -2.0, -1.0);
        let proj = Triangle::project(&ge, &p);
        assert!((proj - Vert3d::new(0.5, 0.0, 0.0)).norm() < 1e-12);

        let p = Vert3d::new(-10.0, 0.2, 1.0);
        let proj = Triangle::project(&ge, &p);
        assert!((proj - Vert3d::new(0.0, 0.2, 0.0)).norm() < 1e-12);

        let p = Vert3d::new(1.2, 0.6, 3.0);
        let proj = Triangle::project(&ge, &p);
        assert!((proj - Vert3d::new(0.8, 0.2, 0.0)).norm() < 1e-12);

        let p = 0.1 * p0 + 0.2 * p1 + 0.7 * p2;
        let proj = Triangle::project(&ge, &p);
        assert!((proj - p).norm() < 1e-12);

        let proj = Triangle::project(&ge, &(p + Vert3d::new(0.0, 0.0, 2.0)));
        assert!((proj - p).norm() < 1e-12);
    }

    #[test]
    fn test_project_triangle_2d() {
        let p0 = Vert2d::new(0.0, 0.0);
        let p1 = Vert2d::new(1.0, 0.0);
        let p2 = Vert2d::new(0.0, 1.0);

        let ge = [p0, p1, p2];

        let p = Vert2d::new(-1.0, -1.0);
        let proj = Triangle::project(&ge, &p);
        assert!((proj - Vert2d::new(0.0, 0.0)).norm() < 1e-12);

        let p = Vert2d::new(2.0, -1.0);
        let proj = Triangle::project(&ge, &p);
        assert!((proj - Vert2d::new(1.0, 0.0)).norm() < 1e-12);

        let p = Vert2d::new(-2.0, 3.0);
        let proj = Triangle::project(&ge, &p);
        assert!((proj - Vert2d::new(0.0, 1.0)).norm() < 1e-12);

        let p = Vert2d::new(0.5, -2.0);
        let proj = Triangle::project(&ge, &p);
        assert!((proj - Vert2d::new(0.5, 0.0)).norm() < 1e-12);

        let p = Vert2d::new(-10.0, 0.2);
        let proj = Triangle::project(&ge, &p);
        assert!((proj - Vert2d::new(0.0, 0.2)).norm() < 1e-12);

        let p = Vert2d::new(1.2, 0.6);
        let proj = Triangle::project(&ge, &p);
        assert!((proj - Vert2d::new(0.8, 0.2)).norm() < 1e-12);

        let p = 0.1 * p0 + 0.2 * p1 + 0.7 * p2;
        let proj = Triangle::project(&ge, &p);
        assert!((proj - p).norm() < 1e-12);
    }
}
