use crate::{Result, Tag, mesh::SimplexMesh};
use log::debug;
use std::collections::HashMap;
use tmesh::{
    Vertex,
    graph::CSRGraph,
    mesh::{GSimplex, Idx, Mesh, MutMesh, Simplex},
    spatialindex::ObjectIndex,
};

impl<T: Idx, const D: usize, C: Simplex<T>> SimplexMesh<T, D, C> {
    /// Automatically tag the (surface) mesh elements as follows
    ///   - A graph is built such that two elements are connected if :
    ///       - they share a face that has only belongs to these two elements
    ///       - the angle between the normals to these elements is lower than a threshold
    ///   - The new element tag is the index of the connected component of this graph
    ///     to which the element belongs
    pub fn autotag(&mut self, angle_deg: f64) -> Result<HashMap<Tag, Vec<Tag>>> {
        assert_eq!(D - 1, C::DIM as usize);

        let f2e = self.get_face_to_elems()?;
        let threshold = angle_deg.to_radians().cos();

        let mut e2e = Vec::with_capacity(f2e.len());
        for elems in f2e.values() {
            if elems.len() == 2 {
                let n0 = self.gelem(&self.elem(elems[0])).normal().normalize();
                let n1 = self.gelem(&self.elem(elems[1])).normal().normalize();
                if n0.dot(&n1) > threshold {
                    e2e.push([elems[0], elems[1]]);
                }
            }
        }

        let components = CSRGraph::from_edges(e2e.iter().copied(), None).connected_components()?;

        let mut new_tags: HashMap<Tag, Vec<Tag>> = HashMap::new();
        let mut tags = HashMap::new();
        let mut next = 1;
        self.etags_mut().zip(components).for_each(|(t, c)| {
            if let Some(new_tag) = tags.get(&(*t, c)) {
                *t = *new_tag;
            } else {
                tags.insert((*t, c), next);
                if let Some(x) = new_tags.get_mut(t) {
                    x.push(next);
                } else {
                    new_tags.insert(*t, vec![next]);
                }
                *t = next;
                next += 1;
            }
        });
        Ok(new_tags)
    }

    /// Automatically tag the mesh faces applying `autotag` to the mesh boundary
    pub fn autotag_bdy(&mut self, angle_deg: f64) -> Result<HashMap<Tag, Vec<Tag>>> {
        assert_eq!(D, C::DIM as usize);

        let mut bdy = self.boundary::<SimplexMesh<T, D, C::FACE>>().0;
        bdy.compute_face_to_elems();
        let new_tags = bdy.autotag(angle_deg)?;

        self.ftags_mut()
            .zip(bdy.etags())
            .for_each(|(t, new_t)| *t = new_t);

        Ok(new_tags)
    }

    /// Transfer the tag information to another mesh.
    /// For each element or face in `mesh` (depending on the dimension), its tag is updated
    /// to the tag of the element of `self` onto which the element center is projected.
    pub fn transfer_tags<C2: Simplex<T>>(
        &self,
        tree: &ObjectIndex<D>,
        mesh: &mut SimplexMesh<T, D, C2>,
    ) -> Result<()> {
        let get_tag = |pt: &Vertex<D>| {
            let idx = tree.nearest_elem(pt);
            self.etag(idx.try_into().unwrap())
        };

        if C2::DIM == C::DIM {
            debug!("Computing the mesh element tags");
            let tags = mesh
                .gelems()
                .map(|ge| get_tag(&ge.center()))
                .collect::<Vec<_>>();
            mesh.etags_mut().zip(tags).for_each(|(t, new_t)| *t = new_t);
        } else if C2::DIM == C::DIM + 1 {
            debug!("Computing the mesh face tags");
            let tags = mesh
                .gfaces()
                .map(|gf| get_tag(&gf.center()))
                .collect::<Vec<_>>();
            mesh.ftags_mut().zip(tags).for_each(|(t, new_t)| *t = new_t);
        } else {
            panic!("Invalid mesh element dimension");
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use tmesh::{
        mesh::{Mesh, MutMesh},
        spatialindex::ObjectIndex,
    };

    use crate::mesh::{
        SimplexMesh,
        test_meshes::{test_mesh_2d, test_mesh_3d},
    };
    use std::collections::HashMap;

    #[test]
    fn test_square() {
        let mut mesh = test_mesh_2d().split();
        mesh.ftags_mut().for_each(|t| *t = 1);

        let new_tags = mesh.autotag_bdy(30.0).unwrap();
        assert_eq!(new_tags.len(), 1);
        assert_eq!(*new_tags.get(&1).unwrap(), vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_square_2() {
        let mut mesh = test_mesh_2d().split();
        let tmp = mesh.faces().collect::<Vec<_>>();
        mesh.ftags_mut().zip(tmp).for_each(|(t, f)| {
            if f[0] == 0 || f[1] == 0 {
                *t = 1;
            } else {
                *t = 2;
            }
        });

        let new_tags = mesh.autotag_bdy(30.0).unwrap();
        assert_eq!(new_tags.len(), 2);
        assert_eq!(*new_tags.get(&1).unwrap(), vec![1, 6]);
        assert_eq!(*new_tags.get(&2).unwrap(), vec![2, 3, 4, 5]);
    }

    #[test]
    fn test_cube() {
        let mut mesh = test_mesh_3d().split();
        let tmp = mesh.faces().collect::<Vec<_>>();
        mesh.ftags_mut().zip(tmp).for_each(|(t, f)| {
            if f[0] == 0 || f[1] == 0 || f[2] == 1 {
                *t = 1;
            } else {
                *t = 2;
            }
        });

        let new_tags = mesh.autotag_bdy(30.0).unwrap();
        assert_eq!(new_tags.len(), 2);
        assert_eq!(*new_tags.get(&1).unwrap(), vec![1, 4, 8]);
        assert_eq!(*new_tags.get(&2).unwrap(), vec![2, 3, 5, 6, 7, 9]);
    }

    #[test]
    fn test_cube_geom() {
        let mut mesh = test_mesh_3d().split();
        let tmp = mesh.faces().collect::<Vec<_>>();
        mesh.ftags_mut().zip(tmp).for_each(|(t, _)| *t = 1);

        let mut bdy = mesh.boundary::<SimplexMesh<_, _, _>>().0.split();
        bdy.compute_face_to_elems();
        let new_tags = bdy.autotag(30.0).unwrap();

        assert_eq!(new_tags.len(), 1);
        assert_eq!(*new_tags.get(&1).unwrap(), vec![1, 2, 3, 4, 5, 6]);

        let tree = ObjectIndex::new(&bdy);
        bdy.transfer_tags(&tree, &mut mesh).unwrap();

        let mut res = HashMap::new();
        for t in mesh.ftags() {
            if let Some(c) = res.get_mut(&t) {
                *c += 1;
            } else {
                res.insert(t, 1);
            }
        }
        assert_eq!(res.len(), 6);
        for t in 1..7 {
            assert_eq!(*res.get(&t).unwrap(), 8);
        }
    }
}
