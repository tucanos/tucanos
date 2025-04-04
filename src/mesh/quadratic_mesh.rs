use super::vector::VectorQuadratic;
use crate::mesh::topo_elems_quadratic::QuadraticElem; // Adjusted the path to `crate::mesh`
use crate::{mesh::Point, Idx, Tag};
use log::debug;

#[derive(Debug, Clone)]

pub struct QuadraticMesh<QE: QuadraticElem> {
    verts: VectorQuadratic<Point<3>>,
    tris: VectorQuadratic<QE>,
    tri_tags: VectorQuadratic<Tag>,
    edgs: VectorQuadratic<QE::Face>,
    edg_tags: VectorQuadratic<Tag>,
}

pub struct QuadraticMesh2 {
    verts: Vec<Point<3>>,
    tris: Vec<[Idx; 6]>,
    tri_tags: Vec<Tag>,
    edgs: Vec<[Idx; 3]>,
    edg_tags: Vec<Tag>,
}

impl<QE: QuadraticElem> QuadraticMesh<QE> {
    #[must_use]
    pub fn new_with_vector(
        verts: VectorQuadratic<Point<3>>,
        tris: VectorQuadratic<QE>,
        tri_tags: VectorQuadratic<Tag>,
        edgs: VectorQuadratic<QE::Face>,
        edg_tags: VectorQuadratic<Tag>,
    ) -> Self {
        debug!(
            "Create a Quadratic mesh with {} {}D vertices / {} {} / {} {}",
            verts.len(),
            3,
            tris.len(),
            QE::NAME,
            edgs.len(),
            QE::Face::NAME
        );
        Self {
            verts,
            tris,
            tri_tags,
            edgs,
            edg_tags,
        }
    }

    /// Create a new `QuadraticMesh`. The extra connectivity information is not built.
    #[must_use]
    pub fn new(
        verts: Vec<Point<3>>,
        tris: Vec<QE>,
        tri_tags: Vec<Tag>,
        edgs: Vec<QE::Face>,
        edg_tags: Vec<Tag>,
    ) -> Self {
        Self::new_with_vector(
            verts.into(),
            tris.into(),
            tri_tags.into(),
            edgs.into(),
            edg_tags.into(),
        )
    }

    /// Get the number of vertices
    #[must_use]
    pub fn n_verts(&self) -> Idx {
        self.verts.len() as Idx
    }

    /// Get the number of edges
    #[must_use]
    pub fn n_edges(&self) -> Idx {
        self.edgs.len() as Idx
    }

    /// Get the number of triangles
    #[must_use]
    pub fn n_tris(&self) -> Idx {
        self.tris.len() as Idx
    }

    /// Get the i-th vertex
    #[must_use]
    pub fn vert(&self, idx: Idx) -> Point<3> {
        self.verts.index(idx)
    }

    /// Get the i-th edge
    #[must_use]
    pub fn edge(&self, idx: Idx) -> QE::Face {
        self.edgs.index(idx)
    }

    /// Get the i-th triangle
    #[must_use]
    pub fn tri(&self, idx: Idx) -> QE {
        self.tris.index(idx)
    }

    /// Get the i-th edge tag
    #[must_use]
    pub fn edgetag(&self, idx: Idx) -> Tag {
        self.edg_tags.index(idx)
    }

    /// Get the i-th triangle tag
    #[must_use]
    pub fn tritag(&self, idx: Idx) -> Tag {
        self.tri_tags.index(idx)
    }

    /// Get an iterator through the triangles
    #[must_use]
    pub fn tris(&self) -> impl ExactSizeIterator<Item = QE> + '_ {
        self.tris.iter()
    }

    /// Get an iterator through the vertices
    #[must_use]
    pub fn verts(&self) -> impl ExactSizeIterator<Item = Point<3>> + '_ {
        self.verts.iter()
    }

    /// Get an iterator through the triangle tags
    #[must_use]
    pub fn tritags(&self) -> impl ExactSizeIterator<Item = Tag> + '_ {
        self.tri_tags.iter()
    }
}

impl QuadraticMesh<QuadraticTriangle> {
    /// Create a QuadraticMesh from a SimplexMesh by adding midpoints on each edge
    pub fn from_simplex_mesh(mesh: &SimplexMesh<2, Triangle>) -> Self {
        let mut verts = mesh.verts().collect::<Vec<_>>();
        let mut tris = Vec::new();
        let mut tri_tags = Vec::new();
        let mut edgs = Vec::new();
        let mut edg_tags = Vec::new();

        let mut edge_to_midpoint = std::collections::HashMap::new();

        for (tri, tag) in mesh.elems().zip(mesh.etags()) {
            let mut quad_tri = [0; 6];
            for (i, &v) in tri.iter().enumerate() {
                quad_tri[i] = v;
            }

            for i in 0..3 {
                let edge = if tri[i] < tri[(i + 1) % 3] {
                    (tri[i], tri[(i + 1) % 3])
                } else {
                    (tri[(i + 1) % 3], tri[i])
                };

                let midpoint_idx = *edge_to_midpoint.entry(edge).or_insert_with(|| {
                    let midpoint = (mesh.vert(edge.0) + mesh.vert(edge.1)) / 2.0;
                    verts.push(midpoint);
                    (verts.len() - 1) as Idx
                });

                quad_tri[3 + i] = midpoint_idx;
            }

            tris.push(QuadraticTriangle::from_slice(&quad_tri));
            tri_tags.push(tag);
        }

        for (edge, &midpoint_idx) in &edge_to_midpoint {
            edgs.push(QuadraticEdge::new(edge.0, edge.1, midpoint_idx));
            edg_tags.push(0); // Default tag for edges
        }

        Self::new(verts, tris, tri_tags, edgs, edg_tags)
    }
}

#[cfg(test)]
mod tests {
    use crate::mesh::test_meshes::test_mesh_2d_quadratic;
    use crate::mesh::{Point, QuadraticEdge, QuadraticTriangle};
    #[test]
    fn test_2d_quadratic() {
        let mesh = test_mesh_2d_quadratic();

        assert_eq!(mesh.n_verts(), 6);
        assert_eq!(mesh.n_tris(), 1);
        assert_eq!(mesh.n_edges(), 3);

        assert_eq!(mesh.vert(0), Point::<3>::new(0., 0., 0.));
        assert_eq!(mesh.tri(0), QuadraticTriangle::new(0, 1, 2, 3, 4, 5));
        assert_eq!(mesh.edge(0), QuadraticEdge::new(0, 1, 3));
    }
}
