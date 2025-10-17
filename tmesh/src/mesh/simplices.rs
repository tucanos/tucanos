//! Simplex elements
use super::{Cell, Edge, Face, Node, Tetrahedron, Triangle, twovec};
use crate::{Vertex, mesh::QuadraticEdge};
use argmin::{
    core::{CostFunction, Executor, Gradient, Hessian},
    solver::{linesearch::MoreThuenteLineSearch, newton::NewtonCG},
};
use core::f64;
use nalgebra::{SMatrix, SVector};
use rustc_hash::FxHashMap;
use std::fmt::Debug;

/// Simplex elements
pub trait Simplex<const C: usize, const O: usize>: Sized {
    type BCOORDS: IntoIterator<Item = f64> + Debug + Clone + Copy;

    /// Get the dimension
    fn dim() -> usize;

    /// Get the number of edges
    fn n_edges() -> usize;

    /// Get the number of faces
    fn n_faces() -> usize;

    /// Get the edges for the simplex `(0, .., C-1)`
    fn edges() -> Vec<Edge>;

    /// Get the i-th edge for the current simplex
    fn edge(&self, i: usize) -> Edge;

    /// Get the faces for the simplex `(0, .., C-1)`
    fn faces<const F: usize>() -> Vec<Face<F>>;

    /// Get the i-th face for the current simplex
    fn face<const F: usize>(&self, i: usize) -> Face<F>;

    /// Check if a normal can be computed in D dimensions
    #[must_use]
    fn has_normal<const D: usize>() -> bool {
        D == Self::dim() + 1
    }

    /// Check of two elements are the same (allowing circular permutation)
    fn is_same(&self, other: &Self) -> bool;

    /// Invert the element
    fn invert(&mut self);

    /// Sort the vertex indices
    #[must_use]
    fn sorted(&self) -> Self;

    /// Get the volume of a simplex
    fn vol<const D: usize>(v: &[Vertex<D>; C]) -> f64;

    /// Barycentric coordinates
    fn bcoords<const D: usize>(ge: &[Vertex<D>; C], v: &Vertex<D>) -> Self::BCOORDS;

    /// Vertex from barycentric coordinates
    fn vert<const D: usize>(ge: &[Vertex<D>; C], bcoords: &Self::BCOORDS) -> Vertex<D>;

    /// Project a point on the simplex
    fn project<const D: usize>(ge: &[Vertex<D>; C], v: &Vertex<D>) -> Vertex<D>;

    /// Try to project a point inside the simplex
    #[must_use]
    fn project_inside<const D: usize>(ge: &[Vertex<D>; C], v: &Vertex<D>) -> Option<Vertex<D>> {
        let bcoords = Self::bcoords(ge, v);
        if bcoords.into_iter().all(|x| x > 0.0) {
            Some(Self::vert(ge, &bcoords))
        } else {
            None
        }
    }

    /// Distance from a point to the simplex
    #[must_use]
    fn distance<const D: usize>(ge: &[Vertex<D>; C], v: &Vertex<D>) -> f64 {
        let pt = Self::project::<D>(ge, v);
        (v - pt).norm()
    }

    /// Get a quadrature (weights and points)
    fn integrate<const D: usize, G: Fn(&Self::BCOORDS) -> f64>(ge: &[Vertex<D>; C], f: G) -> f64;

    /// Barycentric coordinates of the center
    fn center() -> Self::BCOORDS;

    /// Normal to the cell
    fn center_normal<const D: usize>(v: &[Vertex<D>; C]) -> Vertex<D> {
        Self::normal(v, &Self::center())
    }

    /// Normal to the cell
    fn normal<const D: usize>(v: &[Vertex<D>; C], bcoords: &Self::BCOORDS) -> Vertex<D>;

    /// Radius (=diameter of the inner circle / sphere)
    fn radius<const D: usize>(v: &[Vertex<D>; C]) -> f64;

    /// Gamma quality measure, ratio of inscribed radius to circumradius
    /// normalized to be between 0 and 1
    fn gamma<const D: usize>(ge: &[Vertex<D>; C]) -> f64;
}

trait HighOrderLagrangeSimplex<const C: usize, const O: usize>: Simplex<C, O> {
    fn mapping<const D: usize>(v: &[Vertex<D>; C], bcoords: &Self::BCOORDS) -> Vertex<D>;

    fn jac_mapping<const D: usize, const N: usize>(
        ge: &[Vertex<D>; 3],
        bcoords: &Self::BCOORDS,
    ) -> [Vertex<D>; N];

    fn hess_mapping<const D: usize, const N: usize>(
        ge: &[Vertex<D>; 3],
        bcoords: &Self::BCOORDS,
    ) -> [Vertex<D>; N];

    #[allow(dead_code)]
    fn to_bezier<const D: usize>(v: &[Vertex<D>; C]) -> [Vertex<D>; C];
}

impl Simplex<0, 1> for [usize; 0] {
    type BCOORDS = [f64; 0];

    fn dim() -> usize {
        unreachable!()
    }

    fn n_edges() -> usize {
        unreachable!()
    }

    fn n_faces() -> usize {
        unreachable!()
    }

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

    fn integrate<const D: usize, G: Fn(&Self::BCOORDS) -> f64>(_ge: &[Vertex<D>; 0], _f: G) -> f64 {
        unreachable!()
    }

    fn sorted(&self) -> Self {
        unreachable!()
    }

    fn is_same(&self, _other: &Self) -> bool {
        unreachable!()
    }

    fn invert(&mut self) {}

    fn bcoords<const D: usize>(_ge: &[Vertex<D>; 0], _v: &Vertex<D>) -> Self::BCOORDS {
        unreachable!()
    }

    fn vert<const D: usize>(_ge: &[Vertex<D>; 0], _bcoords: &Self::BCOORDS) -> Vertex<D> {
        unreachable!()
    }

    fn project<const D: usize>(_ge: &[Vertex<D>; 0], _v: &Vertex<D>) -> Vertex<D> {
        unreachable!()
    }

    fn center() -> Self::BCOORDS {
        unreachable!()
    }

    fn normal<const D: usize>(_v: &[Vertex<D>; 0], _bcoords: &Self::BCOORDS) -> Vertex<D> {
        unreachable!()
    }

    fn radius<const D: usize>(_v: &[Vertex<D>; 0]) -> f64 {
        unreachable!()
    }

    fn gamma<const D: usize>(_ge: &[Vertex<D>; 0]) -> f64 {
        unreachable!()
    }
}

impl Simplex<1, 1> for Node {
    type BCOORDS = [f64; 1];
    fn dim() -> usize {
        unreachable!()
    }

    fn n_edges() -> usize {
        unreachable!()
    }

    fn n_faces() -> usize {
        unreachable!()
    }

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

    fn integrate<const D: usize, G: Fn(&Self::BCOORDS) -> f64>(_ge: &[Vertex<D>; 1], _f: G) -> f64 {
        unreachable!()
    }

    fn sorted(&self) -> Self {
        *self
    }

    fn is_same(&self, other: &Self) -> bool {
        self[0] == other[0]
    }

    fn invert(&mut self) {}

    fn center() -> Self::BCOORDS {
        unreachable!()
    }
    fn bcoords<const D: usize>(_ge: &[Vertex<D>; 1], _v: &Vertex<D>) -> Self::BCOORDS {
        unreachable!()
    }

    fn vert<const D: usize>(_ge: &[Vertex<D>; 1], _bcoords: &Self::BCOORDS) -> Vertex<D> {
        unreachable!()
    }

    fn project<const D: usize>(ge: &[Vertex<D>; 1], _v: &Vertex<D>) -> Vertex<D> {
        ge[0]
    }

    fn normal<const D: usize>(_v: &[Vertex<D>; 1], _bcoords: &Self::BCOORDS) -> Vertex<D> {
        unreachable!()
    }

    fn radius<const D: usize>(_v: &[Vertex<D>; 1]) -> f64 {
        unreachable!()
    }

    fn gamma<const D: usize>(_ge: &[Vertex<D>; 1]) -> f64 {
        unreachable!()
    }
}

pub const EDGE_FACES: [Node; 2] = [[0], [1]];

impl Simplex<2, 1> for Edge {
    type BCOORDS = [f64; 2];
    fn dim() -> usize {
        1
    }

    fn n_edges() -> usize {
        1
    }

    fn n_faces() -> usize {
        2
    }

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

    fn integrate<const D: usize, G: Fn(&Self::BCOORDS) -> f64>(ge: &[Vertex<D>; 2], f: G) -> f64 {
        let (w, v) = (5.0 / 18.0, 0.5 - 0.5 * (3.0_f64 / 5.0).sqrt());
        let mut res = w * f(&[1.0 - v, v]);
        let (w, v) = (8.0 / 18.0, 0.5);
        res += w * f(&[1.0 - v, v]);
        let (w, v) = (5.0 / 18.0, 0.5 + 0.5 * (3.0_f64 / 5.0).sqrt());
        res += w * f(&[1.0 - v, v]);
        res * Self::vol(ge)
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

    fn center() -> Self::BCOORDS {
        [0.5, 0.5]
    }

    fn bcoords<const D: usize>(ge: &[Vertex<D>; 2], v: &Vertex<D>) -> Self::BCOORDS {
        let ab = ge[1] - ge[0];
        let ap = v - ge[0];

        let ab_squared_magnitude = ab.norm_squared();
        let t = ap.dot(&ab) / ab_squared_magnitude;
        [1.0 - t, t]
    }

    fn vert<const D: usize>(ge: &[Vertex<D>; 2], bcoords: &Self::BCOORDS) -> Vertex<D> {
        bcoords[0] * ge[0] + bcoords[1] * ge[1]
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

    fn center_normal<const D: usize>(v: &[Vertex<D>; 2]) -> Vertex<D> {
        if Self::has_normal::<D>() {
            Vertex::<D>::from_column_slice(&[v[1][1] - v[0][1], v[0][0] - v[1][0]])
        } else {
            unreachable!()
        }
    }

    fn normal<const D: usize>(v: &[Vertex<D>; 2], _bcoords: &Self::BCOORDS) -> Vertex<D> {
        Self::center_normal(v)
    }

    fn radius<const D: usize>(v: &[Vertex<D>; 2]) -> f64 {
        0.5 * (v[1] - v[0]).norm()
    }

    fn gamma<const D: usize>(_ge: &[Vertex<D>; 2]) -> f64 {
        1.0
    }
}

pub const TRIANGLE_FACES: [Edge; 3] = [[0, 1], [1, 2], [2, 0]];

impl Simplex<3, 1> for Triangle {
    type BCOORDS = [f64; 3];

    fn dim() -> usize {
        2
    }

    fn n_edges() -> usize {
        3
    }

    fn n_faces() -> usize {
        3
    }

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
        if <Self as Simplex<3, 1>>::has_normal::<D>() {
            <Self as Simplex<3, 1>>::center_normal(v).norm()
        } else {
            assert_eq!(D, 2);
            let e1 = v[1] - v[0];
            let e2 = v[2] - v[0];

            0.5 * (e1[0] * e2[1] - e1[1] * e2[0])
        }
    }

    fn integrate<const D: usize, G: Fn(&Self::BCOORDS) -> f64>(ge: &[Vertex<D>; 3], f: G) -> f64 {
        let (weight, v, w) = (1.0 / 3.0, 2.0 / 3.0, 1.0 / 6.0);
        let mut res = weight * f(&[1.0 - v - w, v, w]);
        let (weight, v, w) = (1.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0);
        res += weight * f(&[1.0 - v - w, v, w]);
        let (weight, v, w) = (1.0 / 3.0, 1.0 / 6.0, 2.0 / 3.0);
        res += weight * f(&[1.0 - v - w, v, w]);
        res * <Self as Simplex<3, 1>>::vol(ge)
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

    fn center() -> Self::BCOORDS {
        [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
    }

    fn bcoords<const D: usize>(ge: &[Vertex<D>; 3], v: &Vertex<D>) -> Self::BCOORDS {
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

    fn vert<const D: usize>(ge: &[Vertex<D>; 3], bcoords: &Self::BCOORDS) -> Vertex<D> {
        ge[0] * bcoords[0] + ge[1] * bcoords[1] + ge[2] * bcoords[2]
    }

    fn project<const D: usize>(ge: &[Vertex<D>; 3], v: &Vertex<D>) -> Vertex<D> {
        <Self as Simplex<3, 1>>::project_inside(ge, v).map_or_else(
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

    fn center_normal<const D: usize>(v: &[Vertex<D>; 3]) -> Vertex<D> {
        if <Self as Simplex<3, 1>>::has_normal::<D>() {
            let e1 = v[1] - v[0];
            let e2 = v[2] - v[0];
            0.5 * e1.cross(&e2)
        } else {
            unreachable!()
        }
    }

    fn normal<const D: usize>(v: &[Vertex<D>; 3], _bcoords: &Self::BCOORDS) -> Vertex<D> {
        <Self as Simplex<3, 1>>::center_normal(v)
    }

    fn radius<const D: usize>(v: &[Vertex<D>; 3]) -> f64 {
        let a = (v[2] - v[1]).norm();
        let b = (v[2] - v[0]).norm();
        let c = (v[1] - v[0]).norm();
        let s = 0.5 * (a + b + c);
        ((s - a) * (s - b) * (s - c) / s).sqrt()
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
impl Simplex<4, 1> for Tetrahedron {
    type BCOORDS = [f64; 4];
    fn dim() -> usize {
        3
    }

    fn n_edges() -> usize {
        6
    }

    fn n_faces() -> usize {
        4
    }

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

    fn integrate<const D: usize, G: Fn(&Self::BCOORDS) -> f64>(ge: &[Vertex<D>; 4], f: G) -> f64 {
        let (weight, v, w, t) = (
            0.25,
            0.1381966011250105,
            0.1381966011250105,
            0.1381966011250105,
        );
        let mut res = weight * f(&[1.0 - v - w - t, v, w, t]);
        let (weight, v, w, t) = (
            0.25,
            0.5854101966249685,
            0.1381966011250105,
            0.1381966011250105,
        );
        res += weight * f(&[1.0 - v - w - t, v, w, t]);
        let (weight, v, w, t) = (
            0.25,
            0.1381966011250105,
            0.5854101966249685,
            0.1381966011250105,
        );
        res += weight * f(&[1.0 - v - w - t, v, w, t]);
        let (weight, v, w, t) = (
            0.25,
            0.1381966011250105,
            0.1381966011250105,
            0.5854101966249685,
        );
        res += weight * f(&[1.0 - v - w - t, v, w, t]);
        res * Self::vol(ge)
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
            <Triangle as Simplex<3, 1>>::is_same(&f, &o)
        })
    }

    fn invert(&mut self) {
        self.swap(1, 0);
    }

    fn center() -> Self::BCOORDS {
        [0.25, 0.25, 0.25, 0.25]
    }

    fn bcoords<const D: usize>(ge: &[Vertex<D>; 4], v: &Vertex<D>) -> Self::BCOORDS {
        let a = SMatrix::<f64, 4, 4>::new(
            1.0, 1.0, 1.0, 1.0, ge[0][0], ge[1][0], ge[2][0], ge[3][0], ge[0][1], ge[1][1],
            ge[2][1], ge[3][1], ge[0][2], ge[1][2], ge[2][2], ge[3][2],
        );
        let b = SVector::<f64, 4>::new(1., v[0], v[1], v[2]);
        let decomp = a.lu();
        let x = decomp.solve(&b).unwrap();
        [x[0], x[1], x[2], x[3]]
    }

    fn vert<const D: usize>(ge: &[Vertex<D>; 4], bcoords: &Self::BCOORDS) -> Vertex<D> {
        ge[0] * bcoords[0] + ge[1] * bcoords[1] + ge[2] * bcoords[2] + ge[3] * bcoords[3]
    }

    fn project<const D: usize>(ge: &[Vertex<D>; 4], v: &Vertex<D>) -> Vertex<D> {
        Self::project_inside(ge, v).map_or_else(
            || {
                let p0 = <Triangle as Simplex<3, 1>>::project(
                    &[
                        ge[TETRA_FACES[0][0]],
                        ge[TETRA_FACES[0][1]],
                        ge[TETRA_FACES[0][2]],
                    ],
                    v,
                );
                let d0 = (v - p0).norm_squared();
                let p1 = <Triangle as Simplex<3, 1>>::project(
                    &[
                        ge[TETRA_FACES[1][0]],
                        ge[TETRA_FACES[1][1]],
                        ge[TETRA_FACES[1][2]],
                    ],
                    v,
                );
                let d1 = (v - p1).norm_squared();
                let p2 = <Triangle as Simplex<3, 1>>::project(
                    &[
                        ge[TETRA_FACES[2][0]],
                        ge[TETRA_FACES[2][1]],
                        ge[TETRA_FACES[2][2]],
                    ],
                    v,
                );
                let d2 = (v - p2).norm_squared();
                let p3 = <Triangle as Simplex<3, 1>>::project(
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

    fn center_normal<const D: usize>(_v: &[Vertex<D>; 4]) -> Vertex<D> {
        unreachable!()
    }

    fn normal<const D: usize>(_v: &[Vertex<D>; 4], _bcoords: &Self::BCOORDS) -> Vertex<D> {
        unreachable!()
    }

    fn radius<const D: usize>(v: &[Vertex<D>; 4]) -> f64 {
        let a0 = <Triangle as Simplex<3, 1>>::vol(&[v[0], v[1], v[2]]);
        let a1 = <Triangle as Simplex<3, 1>>::vol(&[v[0], v[1], v[3]]);
        let a2 = <Triangle as Simplex<3, 1>>::vol(&[v[1], v[2], v[3]]);
        let a3 = <Triangle as Simplex<3, 1>>::vol(&[v[2], v[0], v[3]]);
        let v = Self::vol(v);
        3.0 * v / (a0 + a1 + a2 + a3)
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

        let s1 = <Triangle as Simplex<3, 1>>::vol(&[
            ge[TETRA_FACES[0][0]],
            ge[TETRA_FACES[0][1]],
            ge[TETRA_FACES[0][2]],
        ]);
        let s2 = <Triangle as Simplex<3, 1>>::vol(&[
            ge[TETRA_FACES[1][0]],
            ge[TETRA_FACES[1][1]],
            ge[TETRA_FACES[1][2]],
        ]);
        let s3 = <Triangle as Simplex<3, 1>>::vol(&[
            ge[TETRA_FACES[2][0]],
            ge[TETRA_FACES[2][1]],
            ge[TETRA_FACES[2][2]],
        ]);
        let s4 = <Triangle as Simplex<3, 1>>::vol(&[
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
    const O: usize,
    I: ExactSizeIterator<Item = &'a Cell<C>>,
>(
    elems: I,
) -> FxHashMap<Face<F>, twovec::Vec<usize>>
where
    Cell<C>: Simplex<C, O>,
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

impl HighOrderLagrangeSimplex<3, 2> for QuadraticEdge {
    fn mapping<const D: usize>(ge: &[Vertex<D>; 3], bcoords: &Self::BCOORDS) -> Vertex<D> {
        let [u, v] = bcoords;
        2.0 * u * (u - 0.5) * ge[0] + 2.0 * v * (v - 0.5) * ge[1] + 4.0 * u * v * ge[2]
    }

    fn jac_mapping<const D: usize, const N: usize>(
        ge: &[Vertex<D>; 3],
        bcoords: &Self::BCOORDS,
    ) -> [Vertex<D>; N] {
        assert_eq!(N, 2);
        let [u, v] = bcoords;
        let mut res = [Vertex::zeros(); N];
        res[0] = (4.0 * u - 1.0) * ge[0] + 4.0 * v * ge[2];
        res[1] = (4.0 * v - 1.0) * ge[1] + 4.0 * u * ge[2];
        res
    }

    fn hess_mapping<const D: usize, const N: usize>(
        ge: &[Vertex<D>; 3],
        _bcoords: &Self::BCOORDS,
    ) -> [Vertex<D>; N] {
        assert_eq!(N, 3);
        let mut res = [Vertex::zeros(); N];
        res[0] = 4.0 * ge[0];
        res[1] = 4.0 * ge[1];
        res[2] = 4.0 * ge[2];
        res
    }

    fn to_bezier<const D: usize>(v: &[Vertex<D>; 3]) -> [Vertex<D>; 3] {
        let p = 0.5 * (4.0 * v[2] - v[0] - v[1]);
        [v[0], v[1], p]
    }
}

struct QuadraticEdgeProjection<'a, const D: usize> {
    v: &'a Vertex<D>,
    ge: &'a [Vertex<D>; 3],
}

impl<const D: usize> CostFunction for QuadraticEdgeProjection<'_, D> {
    type Param = f64;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, argmin::core::Error> {
        let uv = [1.0 - *param, *param];
        let dx = self.v - QuadraticEdge::mapping(self.ge, &uv);
        Ok(dx.norm_squared())
    }
}

impl<const D: usize> Gradient for QuadraticEdgeProjection<'_, D> {
    type Param = f64;
    type Gradient = f64;

    fn gradient(&self, param: &Self::Param) -> Result<Self::Gradient, argmin::core::Error> {
        let uv = [1.0 - *param, *param];
        let dx = self.v - QuadraticEdge::mapping(self.ge, &uv);
        let [du, dv] = QuadraticEdge::jac_mapping(self.ge, &uv);
        Ok(-2.0 * dx.dot(&(dv - du)))
    }
}

impl<const D: usize> Hessian for QuadraticEdgeProjection<'_, D> {
    type Param = f64;
    type Hessian = f64;

    fn hessian(&self, param: &Self::Param) -> Result<Self::Hessian, argmin::core::Error> {
        let uv = [1.0 - *param, *param];
        let dx = self.v - QuadraticEdge::mapping(self.ge, &uv);
        let [du, dv] = QuadraticEdge::jac_mapping(self.ge, &uv);
        let [duu, dvv, duv] = QuadraticEdge::hess_mapping(self.ge, &uv);
        Ok(-2.0 * (dx.dot(&(duu + dvv - 2.0 * duv)) - (dv - du).norm_squared()))
    }
}

impl Simplex<3, 2> for QuadraticEdge {
    type BCOORDS = [f64; 2];

    fn dim() -> usize {
        1
    }

    fn n_edges() -> usize {
        1
    }

    fn n_faces() -> usize {
        2
    }

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

    fn vol<const D: usize>(v: &[Vertex<D>; 3]) -> f64 {
        <Self as Simplex<3, 2>>::integrate(v, |_| 1.0)
    }

    fn integrate<const D: usize, G: Fn(&Self::BCOORDS) -> f64>(ge: &[Vertex<D>; 3], f: G) -> f64 {
        let (w, v) = (0.173927422568727, 0.069431844202974);
        let bcoords = [1.0 - v, v];
        let [mut du, dv] = Self::jac_mapping(ge, &bcoords);
        du -= dv;
        let mut res = w * f(&bcoords) * (du.norm_squared()).sqrt();
        let (w, v) = (0.326072577431273, 0.330009478207572);
        let bcoords = [1.0 - v, v];
        let [mut du, dv] = Self::jac_mapping(ge, &bcoords);
        du -= dv;
        res += w * f(&bcoords) * (du.norm_squared()).sqrt();
        let (w, v) = (0.326072577431273, 0.669990521792428);
        let bcoords = [1.0 - v, v];
        let [mut du, dv] = Self::jac_mapping(ge, &bcoords);
        du -= dv;
        res += w * f(&bcoords) * (du.norm_squared()).sqrt();
        let (w, v) = (0.173927422568727, 0.930568155797026);
        let bcoords = [1.0 - v, v];
        let [mut du, dv] = Self::jac_mapping(ge, &bcoords);
        du -= dv;
        res += w * f(&bcoords) * (du.norm_squared()).sqrt();
        res
    }

    fn sorted(&self) -> Self {
        let mut tmp = *self;
        tmp[0..2].sort_unstable();
        tmp
    }

    fn is_same(&self, other: &Self) -> bool {
        *self == *other
    }

    fn invert(&mut self) {
        self.swap(1, 0);
    }

    fn center() -> Self::BCOORDS {
        [0.5, 0.5]
    }

    fn bcoords<const D: usize>(ge: &[Vertex<D>; 3], v: &Vertex<D>) -> Self::BCOORDS {
        let uv = <Edge as Simplex<2, 1>>::bcoords(&[ge[0], ge[1]], v);

        let linesearch = MoreThuenteLineSearch::new();
        let solver = NewtonCG::new(linesearch);

        if let Ok(res) = Executor::new(QuadraticEdgeProjection { v, ge }, solver)
            .configure(|state| state.param(uv[1]).max_iters(100))
            .run()
        {
            let v = res.state.best_param.unwrap();
            [1.0 - v, v]
        } else {
            [f64::NAN, f64::NAN]
        }
    }

    fn vert<const D: usize>(ge: &[Vertex<D>; 3], bcoords: &Self::BCOORDS) -> Vertex<D> {
        Self::mapping(ge, bcoords)
    }

    fn project<const D: usize>(ge: &[Vertex<D>; 3], v: &Vertex<D>) -> Vertex<D> {
        <Self as Simplex<3, 2>>::project_inside(ge, v).map_or_else(
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

    fn normal<const D: usize>(v: &[Vertex<D>; 3], bcoords: &Self::BCOORDS) -> Vertex<D> {
        if <Self as Simplex<3, 2>>::has_normal::<D>() {
            let [du, dv] = Self::jac_mapping(v, bcoords);
            let mut res = Vertex::<D>::zeros();
            res[0] = dv[1] - du[1];
            res[1] = du[0] - dv[0];
            res
        } else {
            unreachable!()
        }
    }

    fn radius<const D: usize>(_v: &[Vertex<D>; 3]) -> f64 {
        unimplemented!()
    }

    fn gamma<const D: usize>(_ge: &[Vertex<D>; 3]) -> f64 {
        unimplemented!()
    }
}

// pub const QUADRATIC_TRIANGLE_FACES: [QuadraticEdge; 3] = [[0, 1, 3], [1, 2, 4], [2, 0, 5]];

// impl Simplex<6, 2> for QuadraticTriangle {
//     type BCOORDS = [f64; 3];

//     fn dim() -> usize {
//         2
//     }

//     fn n_edges() -> usize {
//         3
//     }

//     fn n_faces() -> usize {
//         3
//     }

//     fn edges() -> Vec<Edge> {
//         vec![[0, 1], [1, 2], [2, 0]]
//     }

//     fn edge(&self, i: usize) -> Edge {
//         match i {
//             0 => [self[0], self[1]],
//             1 => [self[1], self[2]],
//             2 => [self[2], self[0]],
//             _ => unreachable!(),
//         }
//     }

//     fn faces<const F: usize>() -> Vec<Face<F>> {
//         debug_assert_eq!(F, 3);
//         QUADRATIC_TRIANGLE_FACES
//             .iter()
//             .map(|x| x.as_slice().try_into().unwrap())
//             .collect()
//     }

//     fn face<const F: usize>(&self, i: usize) -> Face<F> {
//         debug_assert_eq!(F, 3);
//         let [i0, i1, i2] = QUADRATIC_TRIANGLE_FACES[i];
//         [self[i0], self[i1], self[i2]]
//             .as_slice()
//             .try_into()
//             .unwrap()
//     }

//     fn vol<const D: usize>(_v: &[Vertex<D>; 6]) -> f64 {
//         todo!()
//     }

//     fn quadrature() -> (Vec<f64>, Vec<Vec<f64>>) {
//         let weights = vec![1. / 3., 1. / 3., 1. / 3.];
//         let pts = vec![
//             vec![2. / 3., 1. / 6.],
//             vec![1. / 6., 2. / 3.],
//             vec![1. / 6., 1. / 6.],
//         ];
//         (weights, pts)
//     }

//     fn sorted(&self) -> Self {
//         todo!()
//     }

//     fn is_same(&self, other: &Self) -> bool {
//         let [i0, i1, i2] = self[0..3] else {
//             unreachable!()
//         };
//         other[0..3] == [i0, i1, i2] || other[0..3] == [i1, i2, i0] || other[0..3] == [i2, i0, i1]
//     }

//     fn invert(&mut self) {
//         self.swap(1, 0);
//     }

//     fn center() -> Self::BCOORDS {
//         [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
//     }

//     fn bcoords<const D: usize>(ge: &[Vertex<D>; 3], v: &Vertex<D>) -> Self::BCOORDS {
//         let uvw = <Edge as Simplex<2, 1>::bcoords(&[ge[0], ge[1]], v);

//     }

//     fn vert<const D: usize>(ge: &[Vertex<D>; 3], bcoords: &Self::BCOORDS) -> Vertex<D> {
//         ge[0] * bcoords[0] + ge[1] * bcoords[1] + ge[2] * bcoords[2]
//     }

//     fn project<const D: usize>(ge: &[Vertex<D>; 3], v: &Vertex<D>) -> Vertex<D> {
//         <Self as Simplex<3, 1>>::project_inside(ge, v).map_or_else(
//             || {
//                 let p0 = Edge::project(&[ge[TRIANGLE_FACES[0][0]], ge[TRIANGLE_FACES[0][1]]], v);
//                 let d0 = (v - p0).norm_squared();
//                 let p1 = Edge::project(&[ge[TRIANGLE_FACES[1][0]], ge[TRIANGLE_FACES[1][1]]], v);
//                 let d1 = (v - p1).norm_squared();
//                 let p2 = Edge::project(&[ge[TRIANGLE_FACES[2][0]], ge[TRIANGLE_FACES[2][1]]], v);
//                 let d2 = (v - p2).norm_squared();
//                 if d0 < d1 && d0 < d2 {
//                     p0
//                 } else if d1 < d2 {
//                     p1
//                 } else {
//                     p2
//                 }
//             },
//             |pt| pt,
//         )
//     }

//     fn center_normal<const D: usize>(v: &[Vertex<D>; 3]) -> Vertex<D> {
//         if <Self as Simplex<3, 1>>::has_normal::<D>() {
//             let e1 = v[1] - v[0];
//             let e2 = v[2] - v[0];
//             0.5 * e1.cross(&e2)
//         } else {
//             unreachable!()
//         }
//     }

//     fn normal<const D: usize>(v: &[Vertex<D>; 3], _bcoords: &Self::BCOORDS) -> Vertex<D> {
//         <Self as Simplex<3, 1>>::center_normal(v)
//     }

//     fn radius<const D: usize>(v: &[Vertex<D>; 3]) -> f64 {
//         let a = (v[2] - v[1]).norm();
//         let b = (v[2] - v[0]).norm();
//         let c = (v[1] - v[0]).norm();
//         let s = 0.5 * (a + b + c);
//         ((s - a) * (s - b) * (s - c) / s).sqrt()
//     }

//     fn gamma<const D: usize>(ge: &[Vertex<D>; 3]) -> f64 {
//         let mut a = ge[2] - ge[1];
//         let mut b = ge[0] - ge[2];
//         let mut c = ge[1] - ge[0];

//         a.normalize_mut();
//         b.normalize_mut();
//         c.normalize_mut();

//         let cross_norm = |e1: &Vertex<D>, e2: &Vertex<D>| {
//             if D == 2 {
//                 (e1[0] * e2[1] - e1[1] * e2[0]).abs()
//             } else {
//                 let n = e1.cross(e2);
//                 n.norm()
//             }
//         };
//         let sina = cross_norm(&b, &c);
//         let sinb = cross_norm(&a, &c);
//         let sinc = cross_norm(&a, &b);

//         let tmp = sina + sinb + sinc;
//         if tmp < 1e-12 {
//             0.0
//         } else {
//             4.0 * sina * sinb * sinc / tmp
//         }
//     }
// }
#[cfg(test)]
mod tests {
    use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};

    use crate::{
        Vert2d, Vert3d,
        mesh::{Edge, QuadraticEdge, Simplex, Tetrahedron, Triangle},
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
            let is_same = <Triangle as Simplex<3, 1>>::is_same(&e, &o);
            let n = <Triangle as Simplex<3, 1>>::center_normal(&gt(&o));
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
    fn test_quadratic_edge() {
        let p0 = Vert2d::new(0.0, 0.1);
        let p1 = Vert2d::new(0.2, 1.0);

        let ge = [p0, p1];
        let ge2 = [p0, p1, 0.5 * (p0 + p1)];

        let n = Edge::center_normal(&ge);
        let n2 = <QuadraticEdge as Simplex<3, 2>>::center_normal(&ge2);
        println!("{n:?} {n2:?}");
        assert!((n - n2).norm() < 1e-12);

        let v = Edge::vol(&ge);
        let v2 = <QuadraticEdge as Simplex<3, 2>>::vol(&ge2);
        assert!((v - v2).abs() < 1e-12);

        let p2 = Vert2d::new(0.5, 1.2);

        let ge2 = [p0, p1, p2];
        let n = 100;
        let t = (0..=n)
            .map(|i| f64::from(i) / f64::from(n))
            .collect::<Vec<_>>();

        let mut v = 0.0;
        for tmp in t.windows(2) {
            let p0 = <QuadraticEdge as Simplex<3, 2>>::vert(&ge2, &[1.0 - tmp[0], tmp[0]]);
            let p1 = <QuadraticEdge as Simplex<3, 2>>::vert(&ge2, &[1.0 - tmp[1], tmp[1]]);
            let e = [p0, p1];
            v += Edge::vol(&e);
        }
        let v2 = <QuadraticEdge as Simplex<3, 2>>::vol(&ge2);

        assert!((v - v2).abs() < 0.05 * v);

        let v = 0.1234;
        let p = <QuadraticEdge as Simplex<3, 2>>::vert(&ge2, &[1.0 - v, v]);
        let n = <QuadraticEdge as Simplex<3, 2>>::normal(&ge2, &[1.0 - v, v]);

        for p2 in [p + 0.1 * n, p + n, p + 10.0 * n] {
            let p3 = <QuadraticEdge as Simplex<3, 2>>::project(&ge2, &p2);
            assert!((p - p3).norm() < 1e-12);
        }
    }

    #[test]
    fn test_project_triangle() {
        let p0 = Vert3d::new(0.0, 0.0, 0.0);
        let p1 = Vert3d::new(1.0, 0.0, 0.0);
        let p2 = Vert3d::new(0.0, 1.0, 0.0);

        let ge = [p0, p1, p2];

        let p = Vert3d::new(-1.0, -1.0, 1.0);
        let proj = <Triangle as Simplex<3, 1>>::project(&ge, &p);
        assert!((proj - Vert3d::new(0.0, 0.0, 0.0)).norm() < 1e-12);

        let p = Vert3d::new(2.0, -1.0, 1.0);
        let proj = <Triangle as Simplex<3, 1>>::project(&ge, &p);
        assert!((proj - Vert3d::new(1.0, 0.0, 0.0)).norm() < 1e-12);

        let p = Vert3d::new(-2.0, 3.0, -1.0);
        let proj = <Triangle as Simplex<3, 1>>::project(&ge, &p);
        assert!((proj - Vert3d::new(0.0, 1.0, 0.0)).norm() < 1e-12);

        let p = Vert3d::new(0.5, -2.0, -1.0);
        let proj = <Triangle as Simplex<3, 1>>::project(&ge, &p);
        assert!((proj - Vert3d::new(0.5, 0.0, 0.0)).norm() < 1e-12);

        let p = Vert3d::new(-10.0, 0.2, 1.0);
        let proj = <Triangle as Simplex<3, 1>>::project(&ge, &p);
        assert!((proj - Vert3d::new(0.0, 0.2, 0.0)).norm() < 1e-12);

        let p = Vert3d::new(1.2, 0.6, 3.0);
        let proj = <Triangle as Simplex<3, 1>>::project(&ge, &p);
        assert!((proj - Vert3d::new(0.8, 0.2, 0.0)).norm() < 1e-12);

        let p = 0.1 * p0 + 0.2 * p1 + 0.7 * p2;
        let proj = <Triangle as Simplex<3, 1>>::project(&ge, &p);
        assert!((proj - p).norm() < 1e-12);

        let proj = <Triangle as Simplex<3, 1>>::project(&ge, &(p + Vert3d::new(0.0, 0.0, 2.0)));
        assert!((proj - p).norm() < 1e-12);
    }

    #[test]
    fn test_project_triangle_2d() {
        let p0 = Vert2d::new(0.0, 0.0);
        let p1 = Vert2d::new(1.0, 0.0);
        let p2 = Vert2d::new(0.0, 1.0);

        let ge = [p0, p1, p2];

        let p = Vert2d::new(-1.0, -1.0);
        let proj = <Triangle as Simplex<3, 1>>::project(&ge, &p);
        assert!((proj - Vert2d::new(0.0, 0.0)).norm() < 1e-12);

        let p = Vert2d::new(2.0, -1.0);
        let proj = <Triangle as Simplex<3, 1>>::project(&ge, &p);
        assert!((proj - Vert2d::new(1.0, 0.0)).norm() < 1e-12);

        let p = Vert2d::new(-2.0, 3.0);
        let proj = <Triangle as Simplex<3, 1>>::project(&ge, &p);
        assert!((proj - Vert2d::new(0.0, 1.0)).norm() < 1e-12);

        let p = Vert2d::new(0.5, -2.0);
        let proj = <Triangle as Simplex<3, 1>>::project(&ge, &p);
        assert!((proj - Vert2d::new(0.5, 0.0)).norm() < 1e-12);

        let p = Vert2d::new(-10.0, 0.2);
        let proj = <Triangle as Simplex<3, 1>>::project(&ge, &p);
        assert!((proj - Vert2d::new(0.0, 0.2)).norm() < 1e-12);

        let p = Vert2d::new(1.2, 0.6);
        let proj = <Triangle as Simplex<3, 1>>::project(&ge, &p);
        assert!((proj - Vert2d::new(0.8, 0.2)).norm() < 1e-12);

        let p = 0.1 * p0 + 0.2 * p1 + 0.7 * p2;
        let proj = <Triangle as Simplex<3, 1>>::project(&ge, &p);
        assert!((proj - p).norm() < 1e-12);
    }
}
