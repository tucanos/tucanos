use super::{
    geom_elems::{GEdge, GElem, GTetrahedron, GTriangle, GVertex},
    twovec,
};
use crate::Idx;
use crate::metric::Metric;
use core::hash::Hash;
use core::slice::Iter;
use rustc_hash::FxHashMap;
use std::fmt::Debug;
use std::ops::Index;
use std::{array::IntoIter, ops::IndexMut};

/// Topological elements (i.e. element-to-vertex connectivity)
/// Only usable for simplices
pub trait Elem:
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
    type Face: Elem;
    /// Type for the element geometry
    type Geom<const D: usize, M: Metric<D>>: GElem<D, M>;
    /// Iterate through the element's vertices
    fn iter(&self) -> Iter<Idx>;
    /// Create from a slice containing the element connectivity
    fn from_slice(s: &[Idx]) -> Self;
    /// Create from a iterator containing the element connectivity
    fn from_iter<I: Iterator<Item = Idx>>(s: I) -> Self;
    /// Sort the vertices by increasing order (can make the geometric element invalid !)
    fn sort(&mut self);
    /// Get the i-th face
    fn face(&self, i: Idx) -> Self::Face;
    /// Get the i-the edge
    fn edge(&self, i: Idx) -> [Idx; 2];
    /// Get the local index of vertex i in element e
    fn vertex_index(&self, i: Idx) -> Idx {
        for j in 0..(Self::N_VERTS as usize) {
            if self[j] == i {
                return j as Idx;
            }
        }
        unreachable!();
    }
    /// Create an element from a vertex Id and the opposite face
    fn from_vertex_and_face(i: Idx, f: &Self::Face) -> Self {
        let mut e = Self::default();
        e[0] = i;
        for i in 1..(Self::N_VERTS as usize) {
            e[i] = f[i - 1];
        }
        e
    }
    /// Check if an element contains a vertex
    fn contains_vertex(&self, i: Idx) -> bool {
        self.iter().any(|x| *x == i)
    }
    /// Check if an element contains an edge
    fn contains_edge(&self, edg: [Idx; 2]) -> bool {
        self.contains_vertex(edg[0]) && self.contains_vertex(edg[1])
    }

    #[must_use]
    fn sorted(&self) -> Self {
        let mut r = *self;
        r.sort();
        r
    }

    #[must_use]
    fn sorted_edge(&self, i: Idx) -> [Idx; 2] {
        let mut e = self.edge(i);
        e.sort_unstable();
        e
    }
}

/// Tetrahedron
#[derive(Clone, Copy, Hash, Debug, Eq, PartialEq, Default)]
pub struct Tetrahedron([Idx; 4]);

impl Tetrahedron {
    #[must_use]
    pub const fn new(i0: Idx, i1: Idx, i2: Idx, i3: Idx) -> Self {
        Self([i0, i1, i2, i3])
    }
}

impl Elem for Tetrahedron {
    const N_VERTS: Idx = 4;
    const N_FACES: Idx = 4;
    const N_EDGES: Idx = 6;
    const DIM: Idx = 3;
    const NAME: &'static str = "Tetrahedron";

    type Face = Triangle;
    type Geom<const D: usize, M: Metric<D>> = GTetrahedron<D, M>;

    fn iter(&self) -> Iter<Idx> {
        self.0.iter()
    }

    fn from_slice(s: &[Idx]) -> Self {
        let mut res = Self([0; 4]);
        res.0.clone_from_slice(s);
        res
    }

    fn from_iter<I: Iterator<Item = Idx>>(mut s: I) -> Self {
        let mut res = Self([0; 4]);
        for i in 0..4 {
            res.0[i] = s.next().unwrap();
        }
        res
    }

    fn sort(&mut self) {
        self.0.sort_unstable();
    }

    #[inline]
    fn face(&self, i: Idx) -> Self::Face {
        debug_assert!(i < Self::N_FACES);
        match i {
            0 => Triangle([self.0[1], self.0[2], self.0[3]]),
            1 => Triangle([self.0[2], self.0[0], self.0[3]]),
            2 => Triangle([self.0[0], self.0[1], self.0[3]]),
            3 => Triangle([self.0[0], self.0[2], self.0[1]]),
            _ => Triangle([0, 0, 0]),
        }
    }

    fn edge(&self, i: Idx) -> [Idx; 2] {
        debug_assert!(i < Self::N_EDGES);
        match i {
            0 => [self.0[0], self.0[1]],
            1 => [self.0[1], self.0[2]],
            2 => [self.0[2], self.0[0]],
            3 => [self.0[0], self.0[3]],
            4 => [self.0[1], self.0[3]],
            5 => [self.0[2], self.0[3]],
            _ => [0, 0],
        }
    }
}

impl IntoIterator for Tetrahedron {
    type Item = Idx;
    type IntoIter = IntoIter<Idx, 4>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl Index<usize> for Tetrahedron {
    type Output = Idx;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl IndexMut<usize> for Tetrahedron {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

/// Triangle
#[derive(Clone, Copy, Hash, Debug, Eq, PartialEq, Default)]
pub struct Triangle([Idx; 3]);

impl Triangle {
    #[must_use]
    pub const fn new(i0: Idx, i1: Idx, i2: Idx) -> Self {
        Self([i0, i1, i2])
    }
}

impl Elem for Triangle {
    const N_VERTS: Idx = 3;
    const N_FACES: Idx = 3;
    const N_EDGES: Idx = 3;
    const DIM: Idx = 2;
    const NAME: &'static str = "Triangle";
    type Face = Edge;
    type Geom<const D: usize, M: Metric<D>> = GTriangle<D, M>;

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

    fn sort(&mut self) {
        self.0.sort_unstable();
    }

    #[inline]
    fn face(&self, i: Idx) -> Self::Face {
        debug_assert!(i < Self::N_FACES);
        match i {
            0 => Edge([self.0[0], self.0[1]]),
            1 => Edge([self.0[1], self.0[2]]),
            2 => Edge([self.0[2], self.0[0]]),
            _ => Edge([0, 0]),
        }
    }

    fn edge(&self, i: Idx) -> [Idx; 2] {
        assert!(i < Self::N_EDGES);
        match i {
            0 => [self.0[0], self.0[1]],
            1 => [self.0[1], self.0[2]],
            2 => [self.0[2], self.0[0]],
            _ => [0, 0],
        }
    }
}

impl IntoIterator for Triangle {
    type Item = Idx;
    type IntoIter = IntoIter<Idx, 3>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}
impl Index<usize> for Triangle {
    type Output = Idx;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}
impl IndexMut<usize> for Triangle {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

/// Edge
#[derive(Clone, Copy, Hash, Debug, Eq, PartialEq, Default)]
pub struct Edge([Idx; 2]);

impl Edge {
    #[must_use]
    pub const fn new(i0: Idx, i1: Idx) -> Self {
        Self([i0, i1])
    }
}

impl Elem for Edge {
    const N_VERTS: Idx = 2;
    const N_FACES: Idx = 2;
    const N_EDGES: Idx = 1;
    const DIM: Idx = 1;
    const NAME: &'static str = "Polyline";
    type Face = Vertex;
    type Geom<const D: usize, M: Metric<D>> = GEdge<D, M>;

    fn iter(&self) -> Iter<Idx> {
        self.0.iter()
    }

    fn from_slice(s: &[Idx]) -> Self {
        let mut res = Self([0; 2]);
        res.0.clone_from_slice(s);
        res
    }

    fn from_iter<I: Iterator<Item = Idx>>(mut s: I) -> Self {
        let mut res = Self([0; 2]);
        for i in 0..2 {
            res.0[i] = s.next().unwrap();
        }
        res
    }

    fn sort(&mut self) {
        self.0.sort_unstable();
    }

    #[inline]
    fn face(&self, i: Idx) -> Self::Face {
        debug_assert!(i < Self::N_FACES);
        match i {
            0 => Vertex([self.0[0]]),
            1 => Vertex([self.0[1]]),
            _ => Vertex([0]),
        }
    }

    fn edge(&self, i: Idx) -> [Idx; 2] {
        assert!(i < Self::N_EDGES);
        match i {
            0 => self.0,
            _ => [0, 0],
        }
    }
}

impl IntoIterator for Edge {
    type Item = Idx;
    type IntoIter = IntoIter<Idx, 2>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}
impl Index<usize> for Edge {
    type Output = Idx;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl IndexMut<usize> for Edge {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

/// Vertex
/// The Vertex edges and afaces cannot be computed
#[derive(Clone, Copy, Hash, Debug, Eq, PartialEq, Default)]
pub struct Vertex([Idx; 1]);

impl Elem for Vertex {
    const N_VERTS: Idx = 1;
    const N_FACES: Idx = 0;
    const N_EDGES: Idx = 0;
    const DIM: Idx = 0;
    const NAME: &'static str = "Polyvertex";
    type Face = Self;
    type Geom<const D: usize, M: Metric<D>> = GVertex<D, M>;

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

    fn sort(&mut self) {
        self.0.sort_unstable();
    }

    fn face(&self, _i: Idx) -> Self::Face {
        unreachable!();
    }

    fn edge(&self, _i: Idx) -> [Idx; 2] {
        unreachable!();
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
pub fn get_face_to_elem<E: Elem, I: Iterator<Item = E>>(
    elems: I,
) -> FxHashMap<E::Face, twovec::Vec<Idx>> {
    let mut map: FxHashMap<E::Face, twovec::Vec<Idx>> = FxHashMap::default();
    for (i_elem, elem) in elems.enumerate() {
        for i_face in 0..E::N_FACES {
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

#[cfg(test)]
mod tests {
    use super::{Edge, Idx, Triangle, get_face_to_elem};

    #[test]
    fn test_2d() {
        let elems = [Triangle([0, 1, 2]), Triangle([0, 2, 3])];
        let faces = get_face_to_elem(elems.iter().copied());
        assert_eq!(faces.len(), 5);

        let face = Edge([0 as Idx, 1 as Idx]);
        let f2e = faces.get(&face).unwrap();
        assert_eq!(f2e.len(), 1);
        assert_eq!(f2e[0], 0);

        let face = Edge([0 as Idx, 2 as Idx]);
        let f2e = faces.get(&face).unwrap();
        assert_eq!(f2e.len(), 2);
        assert_eq!(f2e[0], 0);
        assert_eq!(f2e[1], 1);

        let face = Edge([0 as Idx, 3 as Idx]);
        let f2e = faces.get(&face).unwrap();
        assert_eq!(f2e.len(), 1);
        assert_eq!(f2e[0], 1);
    }
}
