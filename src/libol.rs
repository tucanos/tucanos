use crate::{
    mesh::{Point, SimplexMesh},
    spatialindex::{ObjectIndex, PointIndex},
    topo_elems::Elem,
    Idx,
};
use log::info;
use marechal_libol_sys::{LolFreeOctree, LolGetNearest, LolNewOctree, LolProjectVertex, TypTag};
use std::{
    ffi::c_int,
    ptr::{null, null_mut},
};

#[derive(Debug)]
pub struct Octree {
    tree: i64,
    etype: c_int,
    #[allow(dead_code)]
    coords: Vec<f64>,
    #[allow(dead_code)]
    elems: Vec<c_int>,
}

fn ptr_or_null(slice: &[c_int], offset: usize) -> *const c_int {
    if slice.is_empty() {
        null()
    } else {
        slice[offset..].as_ptr()
    }
}

fn lol_new_octree(coords: &[f64], edges: &[c_int], trias: &[c_int], tetras: &[c_int]) -> i64 {
    unsafe {
        LolNewOctree(
            (coords.len() / 3).try_into().unwrap(),
            coords.as_ptr(),
            coords[3..].as_ptr(),
            (edges.len() / 2).try_into().unwrap(),
            edges.as_ptr(),
            ptr_or_null(edges, 2),
            (trias.len() / 3).try_into().unwrap(),
            trias.as_ptr(),
            ptr_or_null(trias, 3),
            0, // Quads
            null(),
            null(),
            (tetras.len() / 4).try_into().unwrap(),
            tetras.as_ptr(),
            ptr_or_null(tetras, 4),
            0, // Pyr
            null(),
            null(),
            0, // Prism
            null(),
            null(),
            0, // Hexa
            null(),
            null(),
            0,
            1,
        )
    }
}

///  Octree to identify the element closest to a vertex and perform projection if needed
///  This is an interface to `libOL`
impl Octree {
    /// Build an octree from a `SimplexMesh`
    /// The coordinates and element connectivity are copied
    fn new_impl<const D: usize, E: Elem>(mesh: &SimplexMesh<D, E>) -> Self {
        let mut coords = Vec::with_capacity(3 * mesh.n_verts() as usize);
        for p in mesh.verts() {
            coords.extend(p.iter());
            coords.extend(&[0.; 3][D..3]);
        }
        let elems: Vec<_> = mesh.elems().flatten().map(|i| i as c_int).collect();
        let (nv, ne) = (mesh.n_verts(), mesh.n_elems());
        let (tree, etype) = match E::N_VERTS {
            2 => {
                info!("create an octree with {nv} vertices and {ne} edges");
                (lol_new_octree(&coords, &elems, &[], &[]), TypTag::LolTypEdg)
            }
            3 => {
                info!("create an octree with {nv} vertices and {ne} triangles");
                (lol_new_octree(&coords, &[], &elems, &[]), TypTag::LolTypTri)
            }
            4 => {
                info!("create an octree with {nv} vertices and {ne} tetrahedra");
                (lol_new_octree(&coords, &[], &[], &elems), TypTag::LolTypTet)
            }
            _ => unreachable!(),
        };
        Self {
            tree,
            etype: etype as c_int,
            coords,
            elems,
        }
    }
}

impl Drop for Octree {
    fn drop(&mut self) {
        unsafe {
            LolFreeOctree(self.tree);
        }
    }
}

impl<const D: usize> PointIndex<D> for Octree {
    fn nearest_vertex(&self, pt: &Point<D>) -> Idx {
        let mut pt = match D {
            2 => Point::<3>::new(pt[0], pt[1], 0.0),
            3 => Point::<3>::new(pt[0], pt[1], pt[2]),
            _ => unreachable!(),
        };

        let mut min_dis: f64 = -1.0;

        let idx = unsafe {
            LolGetNearest(
                self.tree,
                TypTag::LolTypVer as i32,
                pt.as_mut_ptr(),
                &mut min_dis,
                0.0,
                None,
                null_mut(),
                0,
            )
        };
        idx.try_into().unwrap()
    }

    fn new<E: Elem>(mesh: &SimplexMesh<D, E>) -> Self {
        Self::new_impl(mesh)
    }
}

impl<const D: usize> ObjectIndex<D> for Octree {
    fn new<E: Elem>(mesh: &SimplexMesh<D, E>) -> Self {
        Self::new_impl(mesh)
    }
    /// Find the nearest element for a given vertex
    fn nearest(&self, pt: &Point<D>) -> Idx {
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
        idx.try_into().unwrap()
    }
    /// Project a vertex onto the nearest element
    fn project(&self, pt: &Point<D>) -> (f64, Point<D>) {
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
