//! Simplex elements
use super::{Edge, Node, Tetrahedron, Triangle, twovec};
use crate::Vertex;
use crate::mesh::{GEdge, GNode, GTetrahedron, GTriangle};
use nalgebra::{SMatrix, SVector};
use rustc_hash::FxHashMap;
use std::fmt::Debug;
use std::hash::Hash;
use std::ops::{Index, IndexMut};

/// Simplex elements
pub trait Simplex:
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
    + Hash
    + Eq
    + PartialEq
where
    <Self as Simplex>::FACE: 'static,
{
    type FACE: Simplex;
    type GEOM<const D: usize>: GSimplex<D>;
    const DIM: usize;
    const N_VERTS: usize;
    const N_EDGES: usize;
    const N_FACES: usize;
    const EDGES: &'static [Edge];
    const FACES: &'static [Self::FACE];

    fn from_other<C: Simplex>(other: C) -> Self {
        assert_eq!(Self::DIM, C::DIM);
        assert_eq!(Self::N_VERTS, C::N_VERTS);
        Self::from_iter(other.into_iter())
    }

    fn as_slice(&self) -> &[usize];

    fn from_iter<I: Iterator<Item = usize>>(iter: I) -> Self {
        let mut res = Self::default();
        let mut count = 0;
        for (i, j) in iter.enumerate() {
            assert!(i < Self::N_VERTS);
            res[i] = j;
            count += 1;
        }
        assert_eq!(count, Self::N_VERTS);
        res
    }

    fn contains(&self, i: usize) -> bool {
        self.as_slice().contains(&i)
    }

    /// Get the i-th edge for the current simplex
    fn edge(&self, i: usize) -> Edge {
        Edge::from_iter(Self::EDGES[i].into_iter().map(|j| self[j]))
    }

    /// Get an iterator over the edges of the current simplex
    fn edges(&self) -> impl ExactSizeIterator<Item = Edge> {
        (0..Self::N_EDGES).map(|i| self.edge(i))
    }

    /// Get the i-th face for the current simplex
    fn face(&self, i: usize) -> Self::FACE {
        Self::FACE::from_iter(Self::FACES[i].into_iter().map(|j| self[j]))
    }

    /// Get an iterator over the faces of the current simplex
    fn faces(&self) -> impl ExactSizeIterator<Item = Self::FACE> {
        (0..Self::N_FACES).map(|i| self.face(i))
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

pub trait GSimplex<const D: usize>:
    Sized
    + Index<usize, Output = Vertex<D>>
    + IndexMut<usize, Output = Vertex<D>>
    + IntoIterator<Item = Vertex<D>>
    + Default
    + Debug
    + Send
    + Sync
    + Copy
    + Clone
{
    type BCOORDS: IntoIterator<Item = f64>
        + Debug
        + Clone
        + Copy
        + Default
        + Index<usize, Output = f64>
        + IndexMut<usize, Output = f64>;
    type TOPO: Simplex;
    type FACE: GSimplex<D>;
    const N_VERTS: usize;

    fn as_slice(&self) -> &[Vertex<D>];
    fn from_iter<I: Iterator<Item = Vertex<D>>>(iter: I) -> Self {
        let mut res = Self::default();
        let mut count = 0;
        for (i, j) in iter.enumerate() {
            assert!(i < Self::N_VERTS);
            res[i] = j;
            count += 1;
        }
        assert_eq!(count, Self::N_VERTS);
        res
    }

    /// Get the i-th face for the current simplex
    fn face(&self, i: usize) -> Self::FACE {
        Self::FACE::from_iter(Self::TOPO::FACES[i].into_iter().map(|j| self[j]))
    }

    /// Check if a normal can be computed in D dimensions
    #[must_use]
    fn has_normal() -> bool;

    /// Get the volume of a simplex
    fn vol(&self) -> f64;

    /// Normal to the vertex
    fn normal(&self) -> Vertex<D>;

    /// Radius (=diameter of the inner circle / sphere)
    fn radius(&self) -> f64;

    /// Barycentric coordinates
    fn bcoords(&self, v: &Vertex<D>) -> Self::BCOORDS;

    /// Vertex from barycentric coordinates
    fn vert(&self, bcoords: &Self::BCOORDS) -> Vertex<D> {
        bcoords.into_iter().zip(*self).map(|(w, v)| w * v).sum()
    }

    /// Center
    fn center(&self) -> Vertex<D> {
        let res = self.into_iter().sum::<Vertex<D>>();
        (1.0 / Self::N_VERTS as f64) * res
    }

    /// Gamma quality measure, ratio of inscribed radius to circumradius
    /// normalized to be between 0 and 1
    fn gamma(&self) -> f64;

    /// Project a point on the simplex
    fn project(&self, v: &Vertex<D>) -> Vertex<D> {
        self.project_inside(v).map_or_else(
            || {
                let mut p = self.face(0).project(v);
                let mut d = (v - p).norm_squared();
                for j in 1..Self::TOPO::N_FACES {
                    let p1 = self.face(j).project(v);
                    let d1 = (v - p1).norm_squared();
                    if d1 < d {
                        d = d1;
                        p = p1;
                    }
                }
                p
            },
            |pt| pt,
        )
    }

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

impl<const D: usize> Default for GNode<D> {
    fn default() -> Self {
        Self([Vertex::zeros(); 1])
    }
}

const NODE2EDGES: [Edge; 0] = [];
const NODE2FACES: [Node; 1] = [Node([0])];

impl Simplex for Node {
    #[allow(clippy::use_self)]
    type FACE = Node;
    type GEOM<const D: usize> = GNode<D>;
    const DIM: usize = 0;
    const N_VERTS: usize = 1;
    const N_EDGES: usize = 0;
    const N_FACES: usize = 1;
    const EDGES: &'static [Edge] = &NODE2EDGES;
    const FACES: &'static [Self::FACE] = &NODE2FACES;

    fn as_slice(&self) -> &[usize] {
        &self.0
    }

    fn contains(&self, i: usize) -> bool {
        self.0.contains(&i)
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

    fn invert(&mut self) {}
}

impl<const D: usize> GSimplex<D> for GNode<D> {
    const N_VERTS: usize = 1;
    type BCOORDS = [f64; 1];
    type TOPO = Node;
    #[allow(clippy::use_self)]
    type FACE = GNode<D>;

    fn as_slice(&self) -> &[Vertex<D>] {
        &self.0
    }

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

    fn has_normal() -> bool {
        unreachable!()
    }

    fn project(&self, _v: &Vertex<D>) -> Vertex<D> {
        self.0[0]
    }

    fn project_inside(&self, _v: &Vertex<D>) -> Option<Vertex<D>> {
        Some(self.0[0])
    }
}

impl<const D: usize> Default for GEdge<D> {
    fn default() -> Self {
        Self([Vertex::zeros(); 2])
    }
}

const EDGE2EDGES: [Edge; 1] = [Edge([0, 1])];
const EDGE2FACES: [Node; 2] = [Node([0]), Node([1])];

impl Simplex for Edge {
    type FACE = Node;
    type GEOM<const D: usize> = GEdge<D>;
    const DIM: usize = 1;
    const N_VERTS: usize = 2;
    const N_EDGES: usize = 1;
    const N_FACES: usize = 2;
    #[allow(clippy::use_self)]
    const EDGES: &'static [Edge] = &EDGE2EDGES;
    const FACES: &'static [Self::FACE] = &EDGE2FACES;

    fn as_slice(&self) -> &[usize] {
        &self.0
    }

    fn contains(&self, i: usize) -> bool {
        self.0.contains(&i)
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
    type BCOORDS = [f64; 2];
    type TOPO = Edge;
    type FACE = GNode<D>;
    const N_VERTS: usize = 2;

    fn as_slice(&self) -> &[Vertex<D>] {
        self.0.as_slice()
    }

    fn has_normal() -> bool {
        D == 2
    }

    fn vol(&self) -> f64 {
        (self[1] - self[0]).norm()
    }

    fn normal(&self) -> Vertex<D> {
        if Self::has_normal() {
            Vertex::<D>::from_column_slice(&[self[1][1] - self[0][1], self[0][0] - self[1][0]])
        } else {
            unreachable!()
        }
    }

    fn radius(&self) -> f64 {
        0.5 * (self[1] - self[0]).norm()
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
}

impl<const D: usize> Default for GTriangle<D> {
    fn default() -> Self {
        Self([Vertex::zeros(); 3])
    }
}

const TRIANGLE2EDGES: [Edge; 3] = [Edge([0, 1]), Edge([1, 2]), Edge([2, 0])];
const TRIANGLE2FACES: [Edge; 3] = [Edge([0, 1]), Edge([1, 2]), Edge([2, 0])];

impl Simplex for Triangle {
    type FACE = Edge;
    type GEOM<const D: usize> = GTriangle<D>;
    const DIM: usize = 2;
    const N_VERTS: usize = 3;
    const N_EDGES: usize = 3;
    const N_FACES: usize = 3;
    const EDGES: &'static [Edge] = &TRIANGLE2EDGES;
    const FACES: &'static [Self::FACE] = &TRIANGLE2FACES;

    fn as_slice(&self) -> &[usize] {
        &self.0
    }

    fn contains(&self, i: usize) -> bool {
        self.0.contains(&i)
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
    type BCOORDS = [f64; 3];
    type TOPO = Triangle;
    type FACE = GEdge<D>;
    const N_VERTS: usize = 3;

    fn as_slice(&self) -> &[Vertex<D>] {
        self.0.as_slice()
    }

    fn has_normal() -> bool {
        D == 3
    }

    fn vol(&self) -> f64 {
        if Self::has_normal() {
            self.normal().norm()
        } else {
            assert_eq!(D, 2);
            let e1 = self[1] - self[0];
            let e2 = self[2] - self[0];

            0.5 * (e1[0] * e2[1] - e1[1] * e2[0])
        }
    }

    fn normal(&self) -> Vertex<D> {
        if Self::has_normal() {
            let e1 = self[1] - self[0];
            let e2 = self[2] - self[0];
            0.5 * e1.cross(&e2)
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
}

impl<const D: usize> Default for GTetrahedron<D> {
    fn default() -> Self {
        Self([Vertex::zeros(); 4])
    }
}

const TETRA2EDGES: [Edge; 6] = [
    Edge([0, 1]),
    Edge([1, 2]),
    Edge([2, 0]),
    Edge([0, 3]),
    Edge([1, 3]),
    Edge([2, 3]),
];
const TETRA2FACES: [Triangle; 4] = [
    Triangle([1, 2, 3]),
    Triangle([2, 0, 3]),
    Triangle([0, 1, 3]),
    Triangle([0, 2, 1]),
];

impl Simplex for Tetrahedron {
    type FACE = Triangle;
    type GEOM<const D: usize> = GTetrahedron<D>;
    const DIM: usize = 3;
    const N_VERTS: usize = 4;
    const N_EDGES: usize = 6;
    const N_FACES: usize = 4;
    const EDGES: &'static [Edge] = &TETRA2EDGES;
    const FACES: &'static [Self::FACE] = &TETRA2FACES;

    fn as_slice(&self) -> &[usize] {
        &self.0
    }

    fn contains(&self, i: usize) -> bool {
        self.0.contains(&i)
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
            .position(|&x| x == self[0])
            .is_some_and(|i| f.is_same(&other.face(i)))
    }

    fn invert(&mut self) {
        self.0.swap(1, 0);
    }
}

impl<const D: usize> GSimplex<D> for GTetrahedron<D> {
    type BCOORDS = [f64; 4];
    type TOPO = Tetrahedron;
    type FACE = GTriangle<D>;
    const N_VERTS: usize = 4;

    fn as_slice(&self) -> &[Vertex<D>] {
        self.0.as_slice()
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

/// Compute a `FxHashMap` that maps face-to-vertex connectivity (sorted) to a vector of element indices
#[must_use]
pub fn get_face_to_elem<'a, C: Simplex + 'a, I: ExactSizeIterator<Item = &'a C>>(
    elems: I,
) -> FxHashMap<C::FACE, twovec::Vec<usize>> {
    let mut map: FxHashMap<C::FACE, twovec::Vec<usize>> = FxHashMap::default();
    for (i_elem, elem) in elems.enumerate() {
        for i_face in 0..C::N_FACES {
            let f = elem.face(i_face).sorted();
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
        mesh::{
            Edge, GTetrahedron, GTriangle, Simplex, Tetrahedron, Triangle, simplices::GSimplex,
        },
    };

    #[test]
    fn test_is_same() {
        let e = Edge([10, 12]);
        let o = Edge([10, 10]);
        assert!(!e.is_same(&o));
        let o = Edge([12, 10]);
        assert!(!e.is_same(&o));
        let o = Edge([10, 12]);
        assert!(e.is_same(&o));

        let pts = [
            Vert3d::new(0.0, 0.0, 0.0),
            Vert3d::new(1.0, 0.0, 0.0),
            Vert3d::new(0.0, 1.0, 0.0),
            Vert3d::new(0.0, 0.0, 1.0),
        ];

        let gt = |e: &Triangle| GTriangle([pts[e[0]], pts[e[1]], pts[e[2]]]);
        let e = Triangle([0, 1, 2]);
        let mut o = e;

        let mut rng = StdRng::seed_from_u64(1234);
        for _ in 0..10 {
            o.0.shuffle(&mut rng);
            let is_same = e.is_same(&o);
            let n = gt(&o).normal();
            if is_same {
                assert!(n[2] > 0.0);
            } else {
                assert!(n[2] < 0.0);
            }
        }

        let gt = |e: &Tetrahedron| GTetrahedron([pts[e[0]], pts[e[1]], pts[e[2]], pts[e[3]]]);
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
    fn test_project_triangle() {
        let p0 = Vert3d::new(0.0, 0.0, 0.0);
        let p1 = Vert3d::new(1.0, 0.0, 0.0);
        let p2 = Vert3d::new(0.0, 1.0, 0.0);

        let ge = GTriangle([p0, p1, p2]);

        let p = Vert3d::new(-1.0, -1.0, 1.0);
        let proj = ge.project(&p);
        assert!((proj - Vert3d::new(0.0, 0.0, 0.0)).norm() < 1e-12);

        let p = Vert3d::new(2.0, -1.0, 1.0);
        let proj = ge.project(&p);
        assert!((proj - Vert3d::new(1.0, 0.0, 0.0)).norm() < 1e-12);

        let p = Vert3d::new(-2.0, 3.0, -1.0);
        let proj = ge.project(&p);
        assert!((proj - Vert3d::new(0.0, 1.0, 0.0)).norm() < 1e-12);

        let p = Vert3d::new(0.5, -2.0, -1.0);
        let proj = ge.project(&p);
        assert!((proj - Vert3d::new(0.5, 0.0, 0.0)).norm() < 1e-12);

        let p = Vert3d::new(-10.0, 0.2, 1.0);
        let proj = ge.project(&p);
        assert!((proj - Vert3d::new(0.0, 0.2, 0.0)).norm() < 1e-12);

        let p = Vert3d::new(1.2, 0.6, 3.0);
        let proj = ge.project(&p);
        assert!((proj - Vert3d::new(0.8, 0.2, 0.0)).norm() < 1e-12);

        let p = 0.1 * p0 + 0.2 * p1 + 0.7 * p2;
        let proj = ge.project(&p);
        assert!((proj - p).norm() < 1e-12);

        let proj = ge.project(&(p + Vert3d::new(0.0, 0.0, 2.0)));
        assert!((proj - p).norm() < 1e-12);
    }

    #[test]
    fn test_project_triangle_2d() {
        let p0 = Vert2d::new(0.0, 0.0);
        let p1 = Vert2d::new(1.0, 0.0);
        let p2 = Vert2d::new(0.0, 1.0);

        let ge = GTriangle([p0, p1, p2]);

        let p = Vert2d::new(-1.0, -1.0);
        let proj = ge.project(&p);
        assert!((proj - Vert2d::new(0.0, 0.0)).norm() < 1e-12);

        let p = Vert2d::new(2.0, -1.0);
        let proj = ge.project(&p);
        assert!((proj - Vert2d::new(1.0, 0.0)).norm() < 1e-12);

        let p = Vert2d::new(-2.0, 3.0);
        let proj = ge.project(&p);
        assert!((proj - Vert2d::new(0.0, 1.0)).norm() < 1e-12);

        let p = Vert2d::new(0.5, -2.0);
        let proj = ge.project(&p);
        assert!((proj - Vert2d::new(0.5, 0.0)).norm() < 1e-12);

        let p = Vert2d::new(-10.0, 0.2);
        let proj = ge.project(&p);
        assert!((proj - Vert2d::new(0.0, 0.2)).norm() < 1e-12);

        let p = Vert2d::new(1.2, 0.6);
        let proj = ge.project(&p);
        assert!((proj - Vert2d::new(0.8, 0.2)).norm() < 1e-12);

        let p = 0.1 * p0 + 0.2 * p1 + 0.7 * p2;
        let proj = ge.project(&p);
        assert!((proj - p).norm() < 1e-12);
    }
}
