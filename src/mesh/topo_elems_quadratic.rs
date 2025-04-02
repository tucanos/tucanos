use super::geom_quad_elems::{GQuadElem, GQuadraticEdge, GQuadraticTriangle, GVertex};
use crate::metric::Metric;
use crate::Idx;
use core::hash::Hash;
use std::fmt::Debug;

pub trait QuadraticElem:
    Clone + Copy + Eq + PartialEq + Hash + Default + Debug + Send + Sync
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
    type Geom<M: Metric<3>>: GQuadElem<M>;
    /// Create from a slice containing the element connectivity
    fn from_slice(s: &[Idx]) -> Self;
}

/// Quadratic Triangle
#[derive(Clone, Copy, Hash, Debug, Eq, PartialEq, Default)]
pub struct QuadraticTriangle([Idx; 6]);

impl QuadraticTriangle {
    #[must_use]
    pub const fn new(i0: Idx, i1: Idx, i2: Idx, i3: Idx, i4: Idx, i5: Idx) -> Self {
        Self([i0, i1, i2, i3, i4, i5])
    }
}

impl QuadraticElem for QuadraticTriangle {
    const N_VERTS: Idx = 6;
    const N_FACES: Idx = 3;
    const N_EDGES: Idx = 3;
    const DIM: Idx = 3;
    const NAME: &'static str = "QuadraticTriangle";
    type Face = QuadraticEdge;
    type Geom<M: Metric<3>> = GQuadraticTriangle<M>;

    fn from_slice(s: &[Idx]) -> Self {
        let mut res = Self([0; 6]);
        res.0.clone_from_slice(s);
        res
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
    type Geom<M: Metric<3>> = GQuadraticEdge<M>;

    fn from_slice(s: &[Idx]) -> Self {
        let mut res = Self([0; 3]);
        res.0.clone_from_slice(s);
        res
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
    type Geom<M: Metric<3>> = GVertex<M>;

    fn from_slice(s: &[Idx]) -> Self {
        let mut res = Self([0; 1]);
        res.0.clone_from_slice(s);
        res
    }
}
