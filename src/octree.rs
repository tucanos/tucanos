use crate::{
    mesh::{Point, SimplexMesh},
    topo_elems::Elem,
    Idx, Mesh,
};
use log::info;
use marechal_libol_sys::{LolFreeOctree, LolGetNearest, LolNewOctree, LolProjectVertex, TypTag};
use std::{
    ffi::c_int,
    ptr::{null, null_mut},
};

#[derive(Debug, Clone)]
pub struct Octree {
    tree: i64,
    etype: c_int,
    #[allow(dead_code)]
    coords: Vec<f64>,
    #[allow(dead_code)]
    elems: Vec<c_int>,
}

///  Octree to identify the element closest to a vertex and perform projection if needed
///  This is an interface to `libOL`
impl Octree {
    /// Build an octree from a `SimplexMesh`
    /// The coordinates and element connectivity are copied
    pub fn new<const D: usize, E: Elem>(mesh: &SimplexMesh<D, E>) -> Self {
        let mut coords = Vec::with_capacity(3 * mesh.n_verts() as usize);
        if D == 3 {
            for e in &mesh.coords {
                coords.push(*e);
            }
        } else {
            for p in mesh.verts() {
                for j in 0..D {
                    coords.push(p[j]);
                }
                for _j in D..3 {
                    coords.push(0.);
                }
            }
        }

        let mut elems = Vec::with_capacity(mesh.elems.len());
        for e in &mesh.elems {
            elems.push(*e as c_int);
        }

        match E::N_VERTS {
            2 => {
                info!(
                    "create an octree with {} vertices and {} edges",
                    mesh.n_verts(),
                    mesh.n_elems()
                );
                let tree = unsafe {
                    LolNewOctree(
                        mesh.n_verts() as c_int,
                        coords.as_ptr(),
                        coords[3..].as_ptr(),
                        mesh.n_elems() as c_int,
                        elems.as_ptr(),
                        elems[2..].as_ptr(),
                        0,
                        null(),
                        null(),
                        0,
                        null(),
                        null(),
                        0,
                        null(),
                        null(),
                        0,
                        null(),
                        null(),
                        0,
                        null(),
                        null(),
                        0,
                        null(),
                        null(),
                        0,
                        1,
                    )
                };
                Self {
                    tree,
                    etype: TypTag::LolTypEdg as c_int,
                    coords,
                    elems,
                }
            }
            3 => {
                info!(
                    "create an octree with {} vertices and {} triangles",
                    mesh.n_verts(),
                    mesh.n_elems()
                );
                let tree = unsafe {
                    LolNewOctree(
                        mesh.n_verts() as c_int,
                        coords.as_ptr(),
                        coords[3..].as_ptr(),
                        0,
                        null(),
                        null(),
                        mesh.n_elems() as c_int,
                        elems.as_ptr(),
                        elems[3..].as_ptr(),
                        0,
                        null(),
                        null(),
                        0,
                        null(),
                        null(),
                        0,
                        null(),
                        null(),
                        0,
                        null(),
                        null(),
                        0,
                        null(),
                        null(),
                        0,
                        1,
                    )
                };
                Self {
                    tree,
                    etype: TypTag::LolTypTri as c_int,
                    coords,
                    elems,
                }
            }
            4 => {
                info!(
                    "create an octree with {} vertices and {} tetrahedra",
                    mesh.n_verts(),
                    mesh.n_elems()
                );
                let tree = unsafe {
                    LolNewOctree(
                        mesh.n_verts() as c_int,
                        coords.as_ptr(),
                        coords[3..].as_ptr(),
                        0, // Edges
                        null(),
                        null(),
                        0, // Triangles
                        null(),
                        null(),
                        0, // Quads
                        null(),
                        null(),
                        mesh.n_elems() as c_int, // Tets
                        elems.as_ptr(),
                        elems[4..].as_ptr(),
                        0, // Pyr
                        null(),
                        null(),
                        0, // Pri
                        null(),
                        null(),
                        0, // Hex
                        null(),
                        null(),
                        0,
                        1,
                    )
                };
                Self {
                    tree,
                    etype: TypTag::LolTypTet as c_int,
                    coords,
                    elems,
                }
            }
            _ => unreachable!(),
        }
    }

    /// Find the nearest element for a given vertex
    pub fn nearest<const D: usize>(&self, pt: &Point<D>) -> Idx {
        let mut pt = match D {
            2 => Point::<3>::new(pt[0], pt[1], 0.0),
            3 => Point::<3>::new(pt[0], pt[1], pt[2]),
            _ => unreachable!(),
        };

        let mut min_dis: f64 = -1.0;

        let idx = unsafe {
            LolGetNearest(
                self.tree,
                self.etype,
                pt.as_mut_ptr(),
                &mut min_dis,
                0.0,
                None,
                null_mut(),
                0,
            )
        };
        idx as Idx
    }

    /// Project a vertex onto the nearest element
    pub fn project<const D: usize>(&self, pt: &Point<D>) -> (f64, Point<D>) {
        let mut pt = match D {
            2 => Point::<3>::new(pt[0], pt[1], 0.0),
            3 => Point::<3>::new(pt[0], pt[1], pt[2]),
            _ => unreachable!(),
        };

        let mut tmp = Point::<3>::zeros();
        let mut min_dis: f64 = -1.0;

        unsafe {
            let idx = LolGetNearest(
                self.tree,
                self.etype,
                pt.as_mut_ptr(),
                &mut min_dis,
                0.0,
                None,
                null_mut(),
                0,
            );

            LolProjectVertex(
                self.tree,
                pt.as_mut_ptr(),
                self.etype,
                idx,
                tmp.as_mut_ptr(),
                0,
            );
        }

        let mut res = Point::<D>::zeros();
        for i in 0..D {
            res[i] = tmp[i];
        }
        (min_dis, res)
    }
}

impl Drop for Octree {
    fn drop(&mut self) {
        unsafe {
            LolFreeOctree(self.tree);
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        mesh::{Point, SimplexMesh},
        topo_elems::Edge,
        Idx, Tag,
    };

    use super::Octree;

    #[test]
    fn test_edgs() {
        let coords: Vec<f64> = vec![0., 0., 1., 0., 1., 1., 0., 1.];
        let elems: Vec<Idx> = vec![0, 1, 1, 2, 2, 3, 3, 0];
        let etags: Vec<Tag> = vec![1, 2];
        let faces: Vec<Idx> = vec![];
        let ftags: Vec<Tag> = vec![];

        let mesh = SimplexMesh::<2, Edge>::new(coords, elems, etags, faces, ftags);

        let tree = Octree::new(&mesh);

        let pt = Point::<2>::new(1.5, 0.25);
        let (dist, p) = tree.project(&pt);
        assert!(f64::abs(dist - 0.5) < 1e-12);
        assert!((p - Point::<2>::new(1., 0.25)).norm() < 1e-12);
    }
}
