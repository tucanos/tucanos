//! Simplex elements
use crate::Vertex;
use crate::mesh::elements::{Idx, twovec};
use crate::mesh::{Edge, GEdge};
use nalgebra::{DMatrix, DVector};
use rustc_hash::FxHashMap;
use std::fmt::Debug;
use std::hash::Hash;
use std::ops::{Index, IndexMut};

/// Simplex elements
pub trait Simplex:
    Sized
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
    + 'static
{
    type T: Idx;
    type FACE: Simplex<T = Self::T>;
    type GEOM<const D: usize>: GSimplex<D>;
    const DIM: usize;
    const N_VERTS: usize;
    const N_EDGES: usize;
    const N_FACES: usize;

    fn set(&mut self, i: usize, v: usize);

    fn get(&self, i: usize) -> usize;

    fn set_from_iter(&mut self, iter: impl IntoIterator<Item = usize>) {
        let mut count = 0;
        for (i, j) in iter.into_iter().enumerate() {
            assert!(i < Self::N_VERTS);
            self.set(i, j);
            count += 1;
        }
        assert_eq!(count, Self::N_VERTS);
    }

    fn from_iter(iter: impl IntoIterator<Item = usize>) -> Self {
        let mut res = Self::default();
        res.set_from_iter(iter);
        res
    }

    fn from_vert_and_face(i: usize, f: &Self::FACE) -> Self {
        let mut res = Self::default();
        res.set(0, i);
        for (i, j) in f.into_iter().enumerate() {
            res.set(i + 1, j);
        }
        res
    }

    /// Get the local index of vertex i in element e
    fn vertex_index(&self, i: usize) -> Option<usize> {
        self.into_iter().position(|j| j == i)
    }

    /// Create an element from a vertex Id and the opposite face
    fn from_vertex_and_face(i: usize, f: &Self::FACE) -> Self {
        let mut e = Self::default();
        e.set(0, i);
        for i in 1..Self::N_VERTS {
            e.set(i, f.get(i - 1));
        }
        e
    }

    /// Check if an element contains a vertex
    fn contains(&self, i: usize) -> bool;

    /// Check if an element contains an edge
    fn contains_edge(&self, edg: &Edge<usize>) -> bool {
        self.contains(edg.get(0)) && self.contains(edg.get(1))
    }

    /// Get the i-th edge for the current simplex
    fn edge(&self, i: usize) -> Edge<usize>;

    /// Get an iterator over the edges of the current simplex
    fn edges(&self) -> impl ExactSizeIterator<Item = Edge<usize>> {
        (0..Self::N_EDGES).map(|i| self.edge(i))
    }

    /// Get the i-th face for the current simplex. Convention: face #i does not contain vertex i-th vertex
    fn face(&self, i: usize) -> Self::FACE;

    /// Get an iterator over the faces of the current simplex
    fn faces(&self) -> impl ExactSizeIterator<Item = Self::FACE> {
        (0..Self::N_FACES).map(|i| self.face(i))
    }

    /// Sort the vertex indices
    #[must_use]
    fn sorted(&self) -> Self;

    /// Check of two elements are the same (allowing circular permutation)
    fn is_same(&self, other: &Self) -> bool;

    /// Invert the element
    fn invert(&mut self);

    /// Is the simplex linear?
    #[must_use]
    fn order() -> u8 {
        1
    }
}

pub trait GSimplex<const D: usize>:
    Sized
    + Index<usize, Output = Vertex<D>>
    + IntoIterator<Item = Vertex<D>>
    + Default
    + Debug
    + Send
    + Sync
    + Copy
    + Clone
    + 'static
{
    type ARRAY<T: Debug + Default + Clone + Copy>: IntoIterator<Item = T>
        + Debug
        + Clone
        + Copy
        + Default
        + Index<usize, Output = T>
        + IndexMut<usize, Output = T>;
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

    fn ideal_vol() -> f64;

    fn set(&mut self, i: usize, v: Vertex<D>);

    fn from_iter(iter: impl IntoIterator<Item = Vertex<D>>) -> Self {
        let mut res = Self::default();
        let mut count = 0;
        for (i, j) in iter.into_iter().enumerate() {
            assert!(i < Self::N_VERTS);
            res.set(i, j);
            count += 1;
        }
        assert_eq!(count, Self::N_VERTS);
        res
    }

    fn from_vert_and_face(c: &Vertex<D>, f: &Self::FACE) -> Self {
        let mut res = Self::default();
        res.set(0, *c);
        for (i, v) in f.into_iter().enumerate() {
            res.set(i + 1, v);
        }
        res
    }

    /// Get the i-th edge for the current simplex
    fn edge(&self, i: usize) -> GEdge<D>;

    /// Get an iterator over the edges of the current simplex
    fn edges(&self) -> impl ExactSizeIterator<Item = GEdge<D>> {
        (0..Self::TOPO::N_EDGES).map(|i| self.edge(i))
    }

    /// Get the i-th face for the current simplex. Convention: face #i does not contain vertex i-th vertex
    fn face(&self, i: usize) -> Self::FACE;

    /// Get an iterator over the faces of the current simplex
    fn faces(&self) -> impl ExactSizeIterator<Item = Self::FACE> {
        (0..Self::TOPO::N_FACES).map(|i| self.face(i))
    }

    /// Check if a normal can be computed in D dimensions
    #[must_use]
    fn has_normal() -> bool;

    /// Integration
    fn integrate<G: Fn(&Self::BCOORDS) -> f64>(&self, f: G) -> f64;

    /// Get the volume of a simplex
    fn vol(&self) -> f64;

    /// Normal to the simplex
    ///   - if `bcoords.is_some()` at a given location (norm = 1)
    ///   - if `bcoords.is_none()` integrated over the element
    fn normal(&self, bcoords: Option<&Self::BCOORDS>) -> Vertex<D>;

    /// Radius (=diameter of the inner circle / sphere)
    fn radius(&self) -> f64;

    /// Barycentric coordinates of the center
    fn center_bcoords() -> Self::BCOORDS;

    /// Center
    fn center(&self) -> Vertex<D> {
        self.vert(&Self::center_bcoords())
    }

    /// Barycentric coordinates
    fn bcoords(&self, v: &Vertex<D>) -> Self::BCOORDS;

    /// Vertex from barycentric coordinates
    fn vert(&self, bcoords: &Self::BCOORDS) -> Vertex<D> {
        bcoords.into_iter().zip(*self).map(|(w, v)| w * v).sum()
    }

    /// Gamma quality measure, ratio of inscribed radius to circumradius
    /// normalized to be between 0 and 1
    fn gamma(&self) -> f64;

    /// Project a point on the simplex
    fn project(&self, v: &Vertex<D>) -> (Vertex<D>, bool) {
        let (bcoords, p) = self.project_inside(v);

        p.map_or_else(
            || {
                let mut p = Vertex::zeros();
                let mut d = f64::MAX;
                let mut ok = false;
                for (x, f) in bcoords.into_iter().zip(self.faces()) {
                    if x <= 0.0 {
                        let (p1, _) = f.project(v);
                        let d1 = (v - p1).norm_squared();
                        if d1 < d {
                            d = d1;
                            p = p1;
                            ok = true;
                        }
                    }
                }
                debug_assert!(ok, "bcoords = {bcoords:?}");
                (p, false)
            },
            |pt| (pt, true),
        )
    }

    /// Try to project a point inside the simplex
    #[must_use]
    fn project_inside(&self, v: &Vertex<D>) -> (Self::BCOORDS, Option<Vertex<D>>) {
        let bcoords = self.bcoords(v);
        if bcoords.into_iter().all(|x| x > 0.0) {
            let p = self.vert(&bcoords);
            (bcoords, Some(p))
        } else {
            (bcoords, None)
        }
    }

    /// Distance from a point to the simplex
    #[must_use]
    fn distance(&self, v: &Vertex<D>) -> f64 {
        let (pt, _) = self.project(v);
        (v - pt).norm()
    }

    /// Get the barycentric coordinates of the circumcenter
    #[must_use]
    fn circumcenter_bcoords(&self) -> Self::BCOORDS {
        assert!(Self::N_VERTS <= D + 1);

        let mut a = DMatrix::<f64>::zeros(Self::N_VERTS + 1, Self::N_VERTS + 1);
        let mut b = DVector::<f64>::zeros(Self::N_VERTS + 1);

        for i in 0..Self::N_VERTS {
            for j in i..Self::N_VERTS {
                a[(Self::N_VERTS + 1) * i + j] = 2.0 * self[i].dot(&self[j]);
                a[(Self::N_VERTS + 1) * j + i] = a[(Self::N_VERTS + 1) * i + j];
            }
            b[i] = self[i].dot(&self[i]);
        }
        b[Self::N_VERTS] = 1.0;
        let j = Self::N_VERTS;
        for i in 0..Self::N_VERTS {
            a[(Self::N_VERTS + 1) * i + j] = 1.0;
            a[(Self::N_VERTS + 1) * j + i] = 1.0;
        }

        a.lu().solve_mut(&mut b);

        let mut res = Self::BCOORDS::default();
        for (i, &v) in b.iter().take(Self::N_VERTS).enumerate() {
            res[i] = v;
        }
        res
    }

    /// Get the bounding box
    fn bounding_box(&self) -> (Vertex<D>, Vertex<D>);
}

/// Compute a `FxHashMap` that maps face-to-vertex connectivity (sorted) to a vector of element indices
#[must_use]
pub fn get_face_to_elem<C: Simplex>(
    elems: impl IntoIterator<Item = C, IntoIter: ExactSizeIterator>,
) -> FxHashMap<C::FACE, twovec::Vec<usize>> {
    let mut map: FxHashMap<C::FACE, twovec::Vec<usize>> = FxHashMap::default();
    for (i_elem, elem) in elems.into_iter().enumerate() {
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
