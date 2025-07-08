//! Simplex elements
use super::{Cell, Edge, Face, Node, Tetrahedron, Triangle, twovec};
use crate::Vertex;
use nalgebra::{SMatrix, SVector};
use rustc_hash::FxHashMap;

/// Simplex elements
pub trait Simplex<const C: usize>: Sized {
    /// Get the dimension
    #[must_use]
    fn dim() -> usize {
        C - 1
    }

    /// Get the number of edges
    #[must_use]
    fn n_edges() -> usize {
        C * (C - 1) / 2
    }

    /// Get the number of faces
    #[must_use]
    fn n_faces() -> usize {
        C
    }

    /// Get the edges for the simplex `(0, .., C-1)`
    fn edges() -> Vec<Edge>;

    /// Get the i-th edge for the current simplex
    fn edge(&self, i: usize) -> Edge;

    /// Get the faces for the simplex `(0, .., C-1)`
    fn faces<const F: usize>() -> Vec<Face<F>>;

    /// Get the i-th face for the current simplex
    fn face<const F: usize>(&self, i: usize) -> Face<F>;

    /// Get the volume of a simplex
    fn vol<const D: usize>(v: &[Vertex<D>; C]) -> f64;

    /// Check if a normal can be computed in D dimensions
    #[must_use]
    fn has_normal<const D: usize>() -> bool {
        D == C
    }

    /// Normal to the vertex
    fn normal<const D: usize>(v: &[Vertex<D>; C]) -> Vertex<D>;

    /// Radius (=diameter of the inner circle / sphere)
    fn radius<const D: usize>(v: &[Vertex<D>; C]) -> f64;

    /// Get a quadrature (weights and points)
    fn quadrature() -> (Vec<f64>, Vec<Vec<f64>>);

    /// Sort the vertex indices
    #[must_use]
    fn sorted(&self) -> Self;

    /// Check of two elements are the same (allowing circular permutation)
    fn is_same(&self, other: &Self) -> bool;

    /// Invert the element
    fn invert(&mut self);

    /// Barycentric coordinates
    fn bcoords<const D: usize>(ge: &[Vertex<D>; C], v: &Vertex<D>) -> [f64; C];

    /// Gamma quality measure, ratio of inscribed radius to circumradius
    /// normalized to be between 0 and 1
    fn gamma<const D: usize>(ge: &[Vertex<D>; C]) -> f64;
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

impl Simplex<0> for [usize; 0] {
    fn edges() -> Vec<Edge> {
        unreachable!()
    }

    fn edge(&self, _i: usize) -> Edge {
        unreachable!()
    }

    fn faces<const F: usize>() -> Vec<Face<F>> {
        unreachable!()
    }

    fn face<const F: usize>(&self, _i: usize) -> Face<F> {
        unreachable!()
    }

    fn vol<const D: usize>(_v: &[Vertex<D>; 0]) -> f64 {
        unreachable!()
    }

    fn normal<const D: usize>(_v: &[Vertex<D>; 0]) -> Vertex<D> {
        unreachable!()
    }

    fn radius<const D: usize>(_v: &[Vertex<D>; 0]) -> f64 {
        unreachable!()
    }

    fn quadrature() -> (Vec<f64>, Vec<Vec<f64>>) {
        unreachable!()
    }

    fn sorted(&self) -> Self {
        unreachable!()
    }

    fn is_same(&self, _other: &Self) -> bool {
        unreachable!()
    }

    fn invert(&mut self) {}

    fn bcoords<const D: usize>(_ge: &[Vertex<D>; 0], _v: &Vertex<D>) -> [f64; 0] {
        unreachable!()
    }

    fn gamma<const D: usize>(_ge: &[Vertex<D>; 0]) -> f64 {
        unreachable!()
    }
}

impl Simplex<1> for Node {
    fn edges() -> Vec<Edge> {
        unreachable!()
    }

    fn edge(&self, _i: usize) -> Edge {
        unreachable!()
    }

    fn faces<const F: usize>() -> Vec<Face<F>> {
        unreachable!()
    }

    fn face<const F: usize>(&self, _i: usize) -> Face<F> {
        unreachable!()
    }

    fn vol<const D: usize>(_v: &[Vertex<D>; 1]) -> f64 {
        unreachable!()
    }

    fn normal<const D: usize>(_v: &[Vertex<D>; 1]) -> Vertex<D> {
        unreachable!()
    }

    fn radius<const D: usize>(_v: &[Vertex<D>; 1]) -> f64 {
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

    fn invert(&mut self) {}

    fn bcoords<const D: usize>(_ge: &[Vertex<D>; 1], _v: &Vertex<D>) -> [f64; 1] {
        unreachable!()
    }

    fn gamma<const D: usize>(_ge: &[Vertex<D>; 1]) -> f64 {
        1.0
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
        is_circ_perm(self, other)
    }

    fn invert(&mut self) {
        self.swap(1, 0);
    }

    fn bcoords<const D: usize>(_ge: &[Vertex<D>; 2], _v: &Vertex<D>) -> [f64; 2] {
        unreachable!()
    }

    fn gamma<const D: usize>(_ge: &[Vertex<D>; 2]) -> f64 {
        1.0
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
        is_circ_perm(self, other)
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
            let u = ge[1] - ge[0];
            let v = ge[2] - ge[0];
            let n = u.cross(&v);
            let w = v - ge[0];
            let nrm = n.norm_squared();
            let gamma = u.cross(&w).dot(&n) / nrm;
            let beta = w.cross(&v).dot(&n) / nrm;
            [1.0 - beta - gamma, beta, gamma]
        }
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

    fn radius<const D: usize>(_v: &[Vertex<D>; 4]) -> f64 {
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
        tmp.sort_unstable();
        tmp
    }

    fn is_same(&self, other: &Self) -> bool {
        is_circ_perm(self, other)
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
