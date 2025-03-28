use crate::{mesh::Point, Idx, Tag};

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct QuadraticMesh {
    verts: Vec<Point<3>>,
    tris: Vec<[Idx; 6]>,
    tri_tags: Vec<Tag>,
    edgs: Vec<[Idx; 3]>,
    edg_tags: Vec<Tag>,
}

impl QuadraticMesh {
    #[must_use]
    pub fn new(
        verts: Vec<Point<3>>,
        tris: Vec<[Idx; 6]>,
        tri_tags: Vec<Tag>,
        edgs: Vec<[Idx; 3]>,
        edg_tags: Vec<Tag>,
    ) -> Self {
        Self {
            verts,
            tris,
            tri_tags,
            edgs,
            edg_tags,
        }
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
    pub fn vert(&self, idx: Idx) -> Option<Point<3>> {
        self.verts.get(idx as usize).copied()
    }

    /// Get the i-th edge
    #[must_use]
    pub fn edge(&self, idx: Idx) -> Option<[Idx; 3]> {
        self.edgs.get(idx as usize).copied()
    }

    /// Get the i-th triangle
    #[must_use]
    pub fn tri(&self, idx: Idx) -> Option<[Idx; 6]> {
        self.tris.get(idx as usize).copied()
    }

    /// Get the i-th edge tag
    #[must_use]
    pub fn edgetag(&self, idx: Idx) -> Option<Tag> {
        self.edg_tags.get(idx as usize).copied()
    }

    /// Get the i-th triangle tag
    #[must_use]
    pub fn tritag(&self, idx: Idx) -> Option<Tag> {
        self.tri_tags.get(idx as usize).copied()
    }

    /// Get an iterator through the vertices
    pub fn verts(&self) -> impl Iterator<Item = &Point<3>> {
        self.verts.iter()
    }

    /// Get an iterator through the triangles
    pub fn tris(&self) -> impl Iterator<Item = &[Idx; 6]> {
        self.tris.iter()
    }

    /// Get an iterator through the triangles tag
    pub fn tri_tags(&self) -> impl Iterator<Item = &Tag> {
        self.tri_tags.iter()
    }
}

#[cfg(test)]
mod tests {
    use crate::mesh::Point;

    use super::QuadraticMesh;

    #[test]
    fn test_new() {
        let verts = vec![
            Point::<3>::new(0., 0., 0.),
            Point::<3>::new(1., 0., 0.),
            Point::<3>::new(0., 1., 0.),
            Point::<3>::new(0.5, 0.5, 0.),
            Point::<3>::new(0.5, 0., 0.),
            Point::<3>::new(0., 0.5, 0.),
        ];
        let tris = vec![[0, 1, 2, 3, 4, 5]];
        let tri_tags = vec![1];
        let edgs = vec![[0, 3, 1], [1, 4, 2], [2, 5, 0]];
        let edg_tags = vec![1, 2, 3];
        let test_mesh = QuadraticMesh::new(verts, tris, tri_tags, edgs, edg_tags);

        assert_eq!(test_mesh.n_verts(), 6);
        assert_eq!(test_mesh.n_tris(), 1);

        if let Some(vert) = test_mesh.vert(0) {
            assert_eq!(vert, Point::<3>::new(0., 0., 0.));
        }

        if let Some(edge) = test_mesh.edge(0) {
            assert_eq!(edge, [0, 3, 1]);
        }

        if let Some(tri) = test_mesh.tri(0) {
            assert_eq!(tri, [0, 1, 2, 3, 4, 5]);
        }

        if let Some(tag) = test_mesh.edgetag(0) {
            assert_eq!(tag, 1);
        }
        if let Some(tag) = test_mesh.tritag(0) {
            assert_eq!(tag, 1);
        }

        let verts_iter: Vec<_> = test_mesh.verts().collect();
        assert_eq!(verts_iter.len(), 6);

        let tris_iter: Vec<_> = test_mesh.tris().collect();
        assert_eq!(tris_iter.len(), 1);

        let tri_tags_iter: Vec<_> = test_mesh.tri_tags().collect();
        assert_eq!(tri_tags_iter.len(), 1);
    }
}
