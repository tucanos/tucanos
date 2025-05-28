use super::geom_quadratic_elems::{GQuadraticElem, GQuadraticEdge, GQuadraticTriangle, GVertex};
use super::twovec;
use crate::metric::Metric;
use crate::Idx;
use core::hash::Hash;
use core::slice::Iter;
use rustc_hash::FxHashMap;
use std::fmt::Debug;
use std::ops::Index;
use std::{array::IntoIter, ops::IndexMut};
pub trait QuadraticElem:
    Clone
    + Copy
    + Eq
    + PartialEq
    + Hash
    + IntoIterator<Item = Idx>
    + Index<usize, Output = Idx>
    + IndexMut<usize, Output = Idx>
    + Default
    + Debug
    + Send
    + Sync
{
    /// Number of vertices in the element
    const N_VERTS: Idx;
    /// Number of faces in the element
    const N_FACES: Idx;
    /// Number of edes in the element
    const N_EDGES: Idx;
    /// Element dimension
    const DIM: Idx;
    /// Element name (in .xdmf files)
    const NAME: &'static str;
    /// Type for the element faces
    type Face: QuadraticElem;
    /// Type for the element geometry
    type GeomQuadratic<M: Metric<3>>: GQuadraticElem<M>;
    /// Iterate through the element's vertices
    fn iter(&self) -> Iter<Idx>;
    /// Create from a slice containing the element connectivity
    fn from_slice(s: &[Idx]) -> Self;
    /// Create from a iterator containing the element connectivity
    fn from_iter<I: Iterator<Item = Idx>>(s: I) -> Self;
    fn face(&self, i: Idx) -> Self::Face;
    /// Sort the vertices by increasing order (can make the geometric element invalid !)
    fn sort(&mut self);

    #[must_use]
    fn sorted(&self) -> Self {
        let mut r = *self;
        r.sort();
        r
    }
}

/// Quadratic Triangle
#[derive(Clone, Copy, Hash, Debug, Eq, PartialEq, Default)]
pub struct QuadraticTriangle([Idx; 6]);

impl QuadraticTriangle {
    #[must_use]
    pub const fn new(i0: Idx, i1: Idx, i2: Idx, i3: Idx, i4: Idx, i5: Idx) -> Self {
        Self([i0, i1, i2, i3, i4, i5])
    }
    /// Get the i-th vertex index
    #[must_use]
    pub fn index(&self, idx: usize) -> Idx {
        self.0[idx]
    }
}

impl QuadraticElem for QuadraticTriangle {
    const N_VERTS: Idx = 6;
    const N_FACES: Idx = 3;
    const N_EDGES: Idx = 3;
    const DIM: Idx = 3;
    const NAME: &'static str = "QuadraticTriangle";
    type Face = QuadraticEdge;
    type GeomQuadratic<M: Metric<3>> = GQuadraticTriangle<M>;

    fn iter(&self) -> Iter<Idx> {
        self.0.iter()
    }

    fn from_slice(s: &[Idx]) -> Self {
        let mut res = Self([0; 6]);
        res.0.clone_from_slice(s);
        res
    }

    fn from_iter<I: Iterator<Item = Idx>>(mut s: I) -> Self {
        let mut res = Self([0; 6]);
        for i in 0..6 {
            res.0[i] = s.next().unwrap();
        }
        res
    }

    #[inline]
    fn face(&self, i: Idx) -> Self::Face {
        debug_assert!(i < Self::N_FACES);
        match i {
            0 => QuadraticEdge([self.0[0], self.0[1], self.0[3]]),
            1 => QuadraticEdge([self.0[1], self.0[2], self.0[4]]),
            2 => QuadraticEdge([self.0[2], self.0[0], self.0[5]]),
            _ => QuadraticEdge([0, 0, 0]),
        }
    }

    fn sort(&mut self) {
        self.0.sort_unstable();
    }
}

impl IntoIterator for QuadraticTriangle {
    type Item = Idx;
    type IntoIter = IntoIter<Idx, 6>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl Index<usize> for QuadraticTriangle {
    type Output = Idx;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl IndexMut<usize> for QuadraticTriangle {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

/// Quadratic Edge
#[derive(Clone, Copy, Hash, Debug, Eq, PartialEq, Default)]
pub struct QuadraticEdge([Idx; 3]);

impl QuadraticEdge {
    #[must_use]
    pub const fn new(i0: Idx, i1: Idx, i2: Idx) -> Self {
        Self([i0, i1, i2])
    }
}

impl QuadraticElem for QuadraticEdge {
    const N_VERTS: Idx = 3;
    const N_FACES: Idx = 3;
    const N_EDGES: Idx = 1;
    const DIM: Idx = 3;
    const NAME: &'static str = "QuadraticPolyline";
    type Face = Vertex;
    type GeomQuadratic<M: Metric<3>> = GQuadraticEdge<M>;

    fn iter(&self) -> Iter<Idx> {
        self.0.iter()
    }

    fn from_slice(s: &[Idx]) -> Self {
        let mut res = Self([0; 3]);
        res.0.clone_from_slice(s);
        res
    }

    fn from_iter<I: Iterator<Item = Idx>>(mut s: I) -> Self {
        let mut res = Self([0; 3]);
        for i in 0..3 {
            res.0[i] = s.next().unwrap();
        }
        res
    }

    #[inline]
    fn face(&self, i: Idx) -> Self::Face {
        debug_assert!(i < Self::N_FACES);
        match i {
            0 => Vertex([self.0[0]]),
            1 => Vertex([self.0[1]]),
            2 => Vertex([self.0[2]]),
            _ => Vertex([0]),
        }
    }

    fn sort(&mut self) {
        self.0.sort_unstable();
    }
}

impl IntoIterator for QuadraticEdge {
    type Item = Idx;
    type IntoIter = IntoIter<Idx, 3>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl Index<usize> for QuadraticEdge {
    type Output = Idx;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl IndexMut<usize> for QuadraticEdge {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}
/// Vertex
/// The Vertex edges and afaces cannot be computed
#[derive(Clone, Copy, Hash, Debug, Eq, PartialEq, Default)]
pub struct Vertex([Idx; 1]);

impl QuadraticElem for Vertex {
    const N_VERTS: Idx = 1;
    const N_FACES: Idx = 0;
    const N_EDGES: Idx = 0;
    const DIM: Idx = 0;
    const NAME: &'static str = "Polyvertex";
    type Face = Self;
    type GeomQuadratic<M: Metric<3>> = GVertex<M>;

    fn iter(&self) -> Iter<Idx> {
        self.0.iter()
    }

    fn from_slice(s: &[Idx]) -> Self {
        let mut res = Self([0; 1]);
        res.0.clone_from_slice(s);
        res
    }

    fn from_iter<I: Iterator<Item = Idx>>(mut s: I) -> Self {
        let mut res = Self([0; 1]);
        for i in 0..1 {
            res.0[i] = s.next().unwrap();
        }
        res
    }

    fn face(&self, _i: Idx) -> Self::Face {
        unreachable!();
    }

    fn sort(&mut self) {
        self.0.sort_unstable();
    }
}

impl IntoIterator for Vertex {
    type Item = Idx;
    type IntoIter = IntoIter<Idx, 1>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl Index<usize> for Vertex {
    type Output = Idx;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl IndexMut<usize> for Vertex {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

/// Compute a `FxHashMap` that maps face-to-vertex connectivity (sorted) to a vector of element indices
#[must_use]
pub fn get_face_to_elem_quadratic<QE: QuadraticElem, I: Iterator<Item = QE>>(
    elems: I,
) -> FxHashMap<QE::Face, twovec::Vec<Idx>> {
    let mut map: FxHashMap<QE::Face, twovec::Vec<Idx>> = FxHashMap::default();
    for (i_elem, elem) in elems.enumerate() {
        for i_face in 0..QE::N_FACES {
            let mut f = elem.face(i_face);
            f.sort();
            let n = map.get_mut(&f);
            if let Some(n) = n {
                n.push(i_elem as Idx);
            } else {
                map.insert(f, twovec::Vec::with_single(i_elem as Idx));
            }
        }
    }

    map
}
