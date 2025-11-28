//! Indices to efficiently locate the nearest vertices or elements
use crate::{
    Vertex,
    mesh::Mesh,
    spatialindex::{parry_2d::ObjectIndex2d, parry_3d::ObjectIndex3d},
};

mod parry_2d;
mod parry_3d;

/// Point index based on `kdtree`
pub struct PointIndex<const D: usize> {
    tree: kdtree::KdTree<f64, usize, [f64; D]>,
}

impl<const D: usize> PointIndex<D> {
    /// Create a PointIndex from vertices
    pub fn new(verts: impl ExactSizeIterator<Item = Vertex<D>>) -> Self {
        assert!(verts.len() > 0);
        let mut tree = kdtree::KdTree::new(D);
        for (i, pt) in verts.enumerate() {
            tree.add(pt.as_slice().try_into().unwrap(), i).unwrap();
        }
        Self { tree }
    }

    /// Get the index of the nearest point & the distance
    #[must_use]
    pub fn nearest_vert(&self, pt: &Vertex<D>) -> (usize, f64) {
        let r = self
            .tree
            .nearest(pt.as_slice(), 1, &kdtree::distance::squared_euclidean)
            .unwrap()[0];
        (*r.1, r.0)
    }
}

enum ObjectIndexNd<const D: usize, M: Mesh<D>> {
    ObjectIndex2d(ObjectIndex2d<D, M>),
    ObjectIndex3d(ObjectIndex3d<D, M>),
}

pub struct ObjectIndex<const D: usize, M: Mesh<D>>(ObjectIndexNd<D, M>);

impl<const D: usize, M: Mesh<D>> ObjectIndex<D, M> {
    pub fn new(mesh: M) -> Self {
        Self(match D {
            2 => ObjectIndexNd::ObjectIndex2d(ObjectIndex2d::new(mesh)),
            3 => ObjectIndexNd::ObjectIndex3d(ObjectIndex3d::new(mesh)),
            _ => unimplemented!(),
        })
    }

    pub const fn mesh(&self) -> &M {
        match &self.0 {
            ObjectIndexNd::ObjectIndex2d(index) => index.mesh(),
            ObjectIndexNd::ObjectIndex3d(index) => index.mesh(),
        }
    }

    /// Get the index of the nearest element
    #[must_use]
    pub fn nearest_elem(&self, pt: &Vertex<D>) -> usize {
        match &self.0 {
            ObjectIndexNd::ObjectIndex2d(index) => index.nearest_elem(pt),
            ObjectIndexNd::ObjectIndex3d(index) => index.nearest_elem(pt),
        }
    }

    /// Project a point onto the nearest element
    #[must_use]
    pub fn project(&self, pt: &Vertex<D>) -> (f64, Vertex<D>) {
        match &self.0 {
            ObjectIndexNd::ObjectIndex2d(index) => index.project(pt),
            ObjectIndexNd::ObjectIndex3d(index) => index.project(pt),
        }
    }
}
// pub use parry::ObjectIndex;
// pub use parry_3d::ObjectIndex3d;
// #[cfg(test)]
// mod tests {
//     use nalgebra::SVector;
//     use rand::{Rng, SeedableRng, rngs::StdRng};
//     use std::f64::consts::PI;

//     use crate::{
//         Vert2d, Vert3d,
//         mesh::{BoundaryMesh2d, BoundaryMesh3d, Edge, Triangle},
//         spatialindex::ObjectIndex,
//     };

// }
