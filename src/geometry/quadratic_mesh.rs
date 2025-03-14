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
        let _mesh = QuadraticMesh::new(verts, tris, tri_tags, edgs, edg_tags);
    }
}
