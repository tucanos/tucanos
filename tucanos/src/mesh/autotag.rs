use crate::{Result, Tag};
use log::debug;
use std::collections::HashMap;
use tmesh::{
    Vertex,
    graph::CSRGraph,
    mesh::{GSimplex, GenericMesh, Mesh, Simplex},
    spatialindex::ObjectIndex,
};

/// Automatically tag the (surface) mesh elements as follows
///   - A graph is built such that two elements are connected if :
///       - they share a face that has only belongs to these two elements
///       - the angle between the normals to these elements is lower than a threshold
///   - The new element tag is the index of the connected component of this graph
///     to which the element belongs
pub fn autotag<const D: usize, M: Mesh<D>>(
    msh: &mut M,
    angle_deg: f64,
) -> Result<HashMap<Tag, Vec<Tag>>> {
    assert_eq!(D - 1, M::C::DIM);

    let faces = msh.all_faces();
    let threshold = angle_deg.to_radians().cos();

    let mut e2e = Vec::with_capacity(faces.len());
    for elems in faces.values() {
        if elems[1] != usize::MAX && elems[2] != usize::MAX {
            let n0 = msh.gelem(&msh.elem(elems[1])).normal().normalize();
            let n1 = msh.gelem(&msh.elem(elems[2])).normal().normalize();
            if n0.dot(&n1) > threshold {
                e2e.push([elems[1], elems[2]]);
            }
        }
    }

    let components = CSRGraph::from_edges(e2e.iter().copied(), None).connected_components()?;

    let mut new_tags: HashMap<Tag, Vec<Tag>> = HashMap::new();
    let mut tags = HashMap::new();
    let mut next = 1;
    msh.etags_mut().zip(components).for_each(|(t, c)| {
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
pub fn autotag_bdy<const D: usize, M: Mesh<D>>(
    msh: &mut M,
    angle_deg: f64,
) -> Result<HashMap<Tag, Vec<Tag>>> {
    assert_eq!(D, <M::C as Simplex>::DIM);

    let mut bdy = msh.boundary::<GenericMesh<D, <M::C as Simplex>::FACE>>().0;
    let new_tags = autotag(&mut bdy, angle_deg)?;

    msh.ftags_mut()
        .zip(bdy.etags())
        .for_each(|(t, new_t)| *t = new_t);

    Ok(new_tags)
}

/// Transfer the tag information to another mesh.
/// For each element or face in `mesh` (depending on the dimension), its tag is updated
/// to the tag of the element of `self` onto which the element center is projected.
pub fn transfer_tags<const D: usize, M: Mesh<D>, M2: Mesh<D>>(
    msh: &M,
    tree: &ObjectIndex<D>,
    mesh: &mut M2,
) {
    let get_tag = |pt: &Vertex<D>| {
        let idx = tree.nearest_elem(pt);
        msh.etag(idx)
    };

    if <M2::C as Simplex>::DIM == <M::C as Simplex>::DIM {
        debug!("Computing the mesh element tags");
        let tags = mesh
            .gelems()
            .map(|ge| get_tag(&ge.center()))
            .collect::<Vec<_>>();
        mesh.etags_mut().zip(tags).for_each(|(t, new_t)| *t = new_t);
    } else if <M2::C as Simplex>::DIM == <M::C as Simplex>::DIM + 1 {
        debug!("Computing the mesh face tags");
        let tags = mesh
            .gfaces()
            .map(|gf| get_tag(&gf.center()))
            .collect::<Vec<_>>();
        mesh.ftags_mut().zip(tags).for_each(|(t, new_t)| *t = new_t);
    } else {
        panic!("Invalid mesh element dimension");
    }
}

#[cfg(test)]
mod tests {
    use crate::mesh::{
        autotag::{autotag, autotag_bdy, transfer_tags},
        test_meshes::{test_mesh_2d, test_mesh_3d},
    };
    use std::collections::HashMap;
    use tmesh::{
        mesh::{BoundaryMesh3d, Mesh, Simplex},
        spatialindex::ObjectIndex,
    };

    #[test]
    fn test_square() {
        let mut mesh = test_mesh_2d().split();
        mesh.ftags_mut().for_each(|t| *t = 1);

        let new_tags = autotag_bdy(&mut mesh, 30.0).unwrap();
        assert_eq!(new_tags.len(), 1);
        assert_eq!(*new_tags.get(&1).unwrap(), vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_square_2() {
        let mut mesh = test_mesh_2d().split();
        let tmp = mesh.faces().collect::<Vec<_>>();
        mesh.ftags_mut().zip(tmp).for_each(|(t, f)| {
            if f.get(0) == 0 || f.get(1) == 0 {
                *t = 1;
            } else {
                *t = 2;
            }
        });

        let new_tags = autotag_bdy(&mut mesh, 30.0).unwrap();
        assert_eq!(new_tags.len(), 2);
        assert_eq!(*new_tags.get(&1).unwrap(), vec![1, 6]);
        assert_eq!(*new_tags.get(&2).unwrap(), vec![2, 3, 4, 5]);
    }

    #[test]
    fn test_cube() {
        let mut mesh = test_mesh_3d().split();
        let tmp = mesh.faces().collect::<Vec<_>>();
        mesh.ftags_mut().zip(tmp).for_each(|(t, f)| {
            if f.get(0) == 0 || f.get(1) == 0 || f.get(2) == 1 {
                *t = 1;
            } else {
                *t = 2;
            }
        });

        let new_tags = autotag_bdy(&mut mesh, 30.0).unwrap();
        assert_eq!(new_tags.len(), 2);
        assert_eq!(*new_tags.get(&1).unwrap(), vec![1, 4, 8]);
        assert_eq!(*new_tags.get(&2).unwrap(), vec![2, 3, 5, 6, 7, 9]);
    }

    #[test]
    fn test_cube_geom() {
        let mut mesh = test_mesh_3d().split();
        let tmp = mesh.faces().collect::<Vec<_>>();
        mesh.ftags_mut().zip(tmp).for_each(|(t, _)| *t = 1);

        let mut bdy = mesh.boundary::<BoundaryMesh3d>().0.split();
        let new_tags = autotag(&mut bdy, 30.0).unwrap();

        assert_eq!(new_tags.len(), 1);
        assert_eq!(*new_tags.get(&1).unwrap(), vec![1, 2, 3, 4, 5, 6]);

        let tree = ObjectIndex::new(&bdy);
        transfer_tags(&bdy, &tree, &mut mesh);

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
