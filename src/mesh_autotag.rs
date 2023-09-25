use std::collections::HashMap;

use crate::{
    geom_elems::GElem,
    graph::{CSRGraph, ConnectedComponents},
    mesh::SimplexMesh,
    topo_elems::Elem,
    Result, Tag,
};

impl<const D: usize, E: Elem> SimplexMesh<D, E> {
    /// Automatically tag the (surface) mesh elements as follows
    ///   - A graph is built such that two elements are connected if :
    ///       - they share a face that has only belongs to these two elements
    ///       - the angle between the normals to these elements is lower than a threshold
    ///   - The new element tag is the index of the connected component of this graph
    ///     to which the element belongs
    pub fn autotag(&mut self, angle_deg: f64) -> Result<HashMap<Tag, Vec<Tag>>> {
        assert_eq!(D - 1, E::DIM as usize);

        let f2e = self.get_face_to_elems()?;
        let threshold = angle_deg.to_radians().cos();

        let mut e2e = Vec::with_capacity(f2e.len());
        for (_, elems) in f2e.iter() {
            if elems.len() == 2 {
                let n0 = self.gelem(self.elem(elems[0])).normal();
                let n1 = self.gelem(self.elem(elems[1])).normal();
                if n0.dot(&n1) > threshold {
                    e2e.push([elems[0], elems[1]])
                }
            }
        }

        let e2e = CSRGraph::new(&e2e);

        let cc = ConnectedComponents::new(&e2e);
        let components = cc.tags();

        let mut new_tags: HashMap<Tag, Vec<Tag>> = HashMap::new();
        let mut tags = HashMap::new();
        let mut next = 1;
        self.mut_etags().zip(components).for_each(|(t, &c)| {
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
        assert_eq!(D, E::DIM as usize);

        let mut bdy = self.boundary().0;
        bdy.compute_face_to_elems();
        let new_tags = bdy.autotag(angle_deg)?;

        self.mut_ftags()
            .zip(bdy.etags())
            .for_each(|(t, new_t)| *t = new_t);

        Ok(new_tags)
    }
}

#[cfg(test)]
mod tests {
    use crate::test_meshes::{test_mesh_2d, test_mesh_3d};

    #[test]
    fn test_square() {
        let mut mesh = test_mesh_2d().split();
        mesh.mut_ftags().for_each(|t| *t = 1);

        let new_tags = mesh.autotag_bdy(30.0).unwrap();
        assert_eq!(new_tags.len(), 1);
        assert_eq!(*new_tags.get(&1).unwrap(), vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_square_2() {
        let mut mesh = test_mesh_2d().split();
        let tmp = mesh.faces().collect::<Vec<_>>();
        mesh.mut_ftags().zip(tmp).for_each(|(t, f)| {
            if f[0] == 0 || f[1] == 0 {
                *t = 1
            } else {
                *t = 2
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
        mesh.mut_ftags().zip(tmp).for_each(|(t, f)| {
            if f[0] == 0 || f[1] == 0 || f[2] == 1 {
                *t = 1
            } else {
                *t = 2
            }
        });

        let new_tags = mesh.autotag_bdy(30.0).unwrap();
        assert_eq!(new_tags.len(), 2);
        assert_eq!(*new_tags.get(&1).unwrap(), vec![1, 4, 8]);
        assert_eq!(*new_tags.get(&2).unwrap(), vec![2, 3, 5, 6, 7, 9]);

        mesh.boundary().0.write_vtk("test.vtu", None, None).unwrap();
    }
}
