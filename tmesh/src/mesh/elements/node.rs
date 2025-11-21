use crate::{
    Vertex,
    mesh::{Edge, GEdge, GSimplex, Simplex, elements::Idx},
};
use std::fmt::Debug;
use std::ops::Index;

/// Node
#[derive(Default, Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Node<T: Idx>(pub(crate) [T; 1]);

impl<T: Idx> Node<T> {
    #[must_use]
    pub fn new(i0: usize) -> Self {
        Self([i0.try_into().unwrap()])
    }
}

impl<T: Idx> IntoIterator for Node<T> {
    type Item = usize;
    type IntoIter = std::iter::Map<std::array::IntoIter<T, 1>, fn(T) -> usize>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter().map(|x| x.try_into().unwrap())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct GNode<const D: usize>([Vertex<D>; 1]);

impl<const D: usize> GNode<D> {
    #[must_use]
    pub const fn new(v0: &Vertex<D>) -> Self {
        Self([*v0])
    }
}

impl<const D: usize> Index<usize> for GNode<D> {
    type Output = Vertex<D>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<const D: usize> IntoIterator for GNode<D> {
    type Item = Vertex<D>;
    type IntoIter = std::array::IntoIter<Self::Item, 1>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<const D: usize> Default for GNode<D> {
    fn default() -> Self {
        Self([Vertex::zeros(); 1])
    }
}

const NODE2EDGES: [Edge<usize>; 0] = [];
const NODE2FACES: [Node<usize>; 1] = [Node([0])];

impl<T: Idx> Simplex for Node<T> {
    type T = T;
    type FACE = Self;
    type GEOM<const D: usize> = GNode<D>;
    const DIM: usize = 0;
    const N_VERTS: usize = 1;
    const N_EDGES: usize = 0;
    const N_FACES: usize = 1;

    fn get(&self, index: usize) -> usize {
        self.0[index].try_into().unwrap()
    }

    fn edge(&self, i: usize) -> Edge<usize> {
        Edge::from_iter(NODE2EDGES[i].into_iter().map(|j| self.get(j)))
    }

    fn face(&self, i: usize) -> Self::FACE {
        Self::FACE::from_iter(NODE2FACES[i].into_iter().map(|j| self.get(j)))
    }

    fn set(&mut self, i: usize, v: usize) {
        self.0[i] = v.try_into().unwrap();
    }

    fn contains(&self, i: usize) -> bool {
        self.0.contains(&i.try_into().unwrap())
    }

    fn quadrature() -> (Vec<f64>, Vec<Vec<f64>>) {
        unreachable!()
    }

    fn sorted(&self) -> Self {
        *self
    }

    fn is_same(&self, other: &Self) -> bool {
        self.0[0] == other.0[0]
    }

    fn invert(&mut self) {}
}

impl<const D: usize> GSimplex<D> for GNode<D> {
    const N_VERTS: usize = 1;
    type ARRAY<T: Debug + Default + Clone + Copy> = [T; 1];
    type BCOORDS = Self::ARRAY<f64>;
    type TOPO = Node<usize>;
    type FACE = Self;

    fn ideal_vol() -> f64 {
        1.0
    }

    fn edge(&self, i: usize) -> GEdge<D> {
        GEdge::from_iter(NODE2EDGES[i].into_iter().map(|j| self[j]))
    }

    fn face(&self, i: usize) -> Self::FACE {
        Self::FACE::from_iter(NODE2FACES[i].into_iter().map(|j| self[j]))
    }

    fn set(&mut self, i: usize, v: Vertex<D>) {
        self.0[i] = v;
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
