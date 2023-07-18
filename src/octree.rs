use crate::{
    mesh::{Point, SimplexMesh},
    topo_elems::Elem,
    Idx,
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
        let mut coords = Vec::with_capacity(mesh.n_verts() as usize);
        for p in mesh.verts() {
            for j in 0..D {
                coords.push(p[j]);
            }
            for _j in D..3 {
                coords.push(0.);
            }
        }

        let elems: Vec<c_int> = mesh.elems().flatten().map(|i| i as c_int).collect();

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

    pub fn nearest_vertex<const D: usize>(&self, pt: &Point<D>) -> Idx {
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
    use nalgebra::SVector;
    use rand::{rngs::StdRng, Rng, SeedableRng};
    use std::f64::consts::PI;

    use crate::{
        mesh::{Point, SimplexMesh},
        topo_elems::{Edge, Elem, Triangle},
        Tag,
    };

    use super::Octree;

    #[test]
    fn test_edgs() {
        let coords = vec![
            Point::<2>::new(0., 0.),
            Point::<2>::new(1., 0.),
            Point::<2>::new(1., 1.),
            Point::<2>::new(0., 1.),
        ];
        let elems: Vec<_> = vec![
            Edge::from_slice(&[0, 1]),
            Edge::from_slice(&[1, 2]),
            Edge::from_slice(&[2, 3]),
            Edge::from_slice(&[3, 0]),
        ];
        let etags: Vec<Tag> = vec![1, 2, 3, 4];
        let faces: Vec<_> = Vec::new();
        let ftags: Vec<Tag> = Vec::new();

        let mesh = SimplexMesh::new(coords, elems, etags, faces, ftags);

        let tree = Octree::new(&mesh);

        let pt = Point::<2>::new(1.5, 0.25);
        let (dist, p) = tree.project(&pt);
        assert!(f64::abs(dist - 0.5) < 1e-12);
        assert!((p - Point::<2>::new(1., 0.25)).norm() < 1e-12);
    }

    #[test]
    fn test_proj() {
        let r_in = 1.;
        let r_out = 1000.;
        let n = 100;
        let dim = 3;

        let mut coords = Vec::with_capacity(2 * (n as usize) * dim);
        for i in 0..n {
            let theta = 2.0 * PI * i as f64 / n as f64;
            coords.push(Point::<3>::new(
                r_in * f64::cos(theta),
                r_in * f64::sin(theta),
                0.0,
            ));
            coords.push(Point::<3>::new(
                r_out * f64::cos(theta),
                r_out * f64::sin(theta),
                0.0,
            ));
        }

        let mut tris = Vec::with_capacity(2 * (n as usize));
        for i in 0..n - 1 {
            tris.push(Triangle::from_slice(&[2 * i, 2 * i + 1, 2 * i + 2]));
            tris.push(Triangle::from_slice(&[2 * i + 2, 2 * i + 1, 2 * i + 3]));
        }
        tris.push(Triangle::from_slice(&[2 * n - 2, 2 * n - 1, 0]));
        tris.push(Triangle::from_slice(&[0, 2 * n - 1, 1]));

        let tri_tags = vec![1; 2 * (n as usize)];

        let msh = SimplexMesh::<3, Triangle>::new(coords, tris, tri_tags, Vec::new(), Vec::new());
        // msh.write_vtk("dbg.vtu", None, None);

        let tree = Octree::new(&msh);
        let pt = Point::<3>::new(-360., -105., 0.);
        assert_eq!(tree.nearest(&pt), 109);

        let pt = Point::<3>::new(41.905, -7.933, 0.);
        assert_eq!(tree.nearest(&pt), 194);

        let pt = Point::<3>::new(977.405_622_304_933_2, -193.219_725_123_763_82, 0.);
        assert_eq!(tree.nearest(&pt), 193);
        let (d, _) = tree.project(&pt);
        assert!(f64::abs(d) < 1e-12, "{d} vs 0");

        let pt = Point::<3>::new(732.254_535_699_460_3, 628.314_474_637_604_1, 0.);
        assert_eq!(tree.nearest(&pt), 23);
        let (d, _) = tree.project(&pt);
        assert!(f64::abs(d) < 1e-12, "{d} vs 0");

        let pt = Point::<3>::new(41.905_036_870_164_33, -7.932_967_693_525_678, 0.);
        assert_eq!(tree.nearest(&pt), 194);
        let (d, _) = tree.project(&pt);
        assert!(f64::abs(d) < 1e-12, "{d} vs 0");

        let mut rng = StdRng::seed_from_u64(0);
        for _ in 0..10000 {
            let tmp = SVector::<f64, 3>::from_fn(|_, _| rng.gen());
            let theta = 2.0 * PI * tmp[0];
            let r = r_in + tmp[1] * (r_out * 0.999 - r_in);
            let x = r * f64::cos(theta);
            let y = r * f64::sin(theta);
            let z = r_out * (tmp[2] - 0.5);
            let pt = Point::<3>::new(x, y, z);
            let (d, pt_proj) = tree.project(&pt);
            println!("{pt:?} -> {pt_proj:?}, {d}");
            assert!(f64::abs(d - z.abs()) < 1e-12);
            assert!(f64::abs(pt_proj[0] - x) < 1e-12);
            assert!(f64::abs(pt_proj[1] - y) < 1e-12);
            assert!(f64::abs(pt_proj[2] - 0.0) < 1e-12);
        }
    }
}
