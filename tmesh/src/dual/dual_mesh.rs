//! Dual meshes
//! Both and edge-based representation (used for cell-vertex FV schemes)
//!  - edges of the original mesh, and the normals (scaled by the area) of the faces built around
//!    the edge
//!  - boundary faces
//!
//! and an explicit polygonal meshes (`PolyMesh<D>, where all the faces are of type `Face<F>`)
//! are built
use super::PolyMesh;
use crate::{
    Error, Result, Tag, Vertex,
    mesh::{Cell, Face, Mesh, Simplex, cell_center},
};
use nalgebra::{DMatrix, DVector};
use rayon::prelude::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use rustc_hash::{FxBuildHasher, FxHashMap, FxHashSet};

/// Types of dual cells
#[derive(Clone, Copy, Debug)]
pub enum DualType {
    /// Classical median cells, going through edge, face and element geometric centers
    Median,
    /// Barth cells, as described in
    /// T. Barth, Aspects of unstructured grids and finite-volume solvers for the Euler
    /// and Navier-Stokes equations, Technical report 787, AGARD, 1992
    Barth,
    /// Modified Barth cells: if a barycentric coordinates of the center given by Barth
    /// cells is smaller that a threshold, it is clipped to 0
    ThresholdBarth(f64),
}

#[derive(Clone, Copy, Debug)]
pub enum DualCellCenter<const D: usize, const F: usize> {
    Vertex(Vertex<D>),
    Face(Face<F>),
}

/// Dual of a `Mesh<D, C, F>`
pub trait DualMesh<const D: usize, const C: usize, const F: usize>: PolyMesh<D>
where
    Cell<C>: Simplex<C>,
    Cell<F>: Simplex<F>,
{
    /// Compute the dual of `mesh`
    fn new<M: Mesh<D, C, F>>(msh: &M, t: DualType) -> Self;

    /// Display element `i`
    fn print_elem_info(&self, i: usize) {
        println!("Dual element {i}");
        self.par_edges_and_normals()
            .filter(|&([i0, i1], _)| i0 == i || i1 == i)
            .for_each(|(e, n)| {
                println!(
                    "  edges: {} - {} : n = {:?} (nrm = {})",
                    e[0],
                    e[1],
                    n,
                    n.norm()
                );
            });
        self.par_boundary_faces()
            .filter(|&(i0, _, _)| i0 == i)
            .for_each(|(_, tag, n)| {
                println!(
                    "  boundary face: tag = {},: n = {:?} (nrm = {})",
                    tag,
                    n,
                    n.norm()
                );
            });
        println!("  vol: {:.2e}", self.vol(i));
    }

    /// Get the vertex coordinates for face `f`
    fn gface(&self, f: &[usize; F]) -> [Vertex<D>; F] {
        let mut res = [self.vert(0); F];
        for (j, &k) in f.iter().enumerate() {
            res[j] = self.vert(k);
        }
        res
    }

    /// Parallel iterator over the faces vertex coordinates
    fn par_gfaces(&self) -> impl IndexedParallelIterator<Item = [Vertex<D>; F]> + '_ {
        self.par_faces().map(|f| {
            let f = f.try_into().unwrap();
            self.gface(&f)
        })
    }

    /// Sequential iterator over the faces vertex coordinates
    fn gfaces(&self) -> impl ExactSizeIterator<Item = [Vertex<D>; F]> + '_ {
        self.faces().map(|f| {
            let f = f.try_into().unwrap();
            self.gface(&f)
        })
    }

    /// Number of edges
    fn n_edges(&self) -> usize;

    /// Get the `i`th edge
    fn edge(&self, i: usize) -> [usize; 2];

    /// Parallel itertator over the edges
    fn par_edges(&self) -> impl IndexedParallelIterator<Item = [usize; 2]> + '_ + Clone {
        (0..self.n_edges()).into_par_iter().map(|i| self.edge(i))
    }

    /// Sequential iterator over the edges
    fn edges(&self) -> impl ExactSizeIterator<Item = [usize; 2]> + '_ + Clone {
        (0..self.n_edges()).map(|i| self.edge(i))
    }

    /// Get the normal associated with the `i`th edge
    fn edge_normal(&self, i: usize) -> Vertex<D>;

    /// Parallel iterator over the edges and edge normals
    fn par_edges_and_normals(
        &self,
    ) -> impl IndexedParallelIterator<Item = ([usize; 2], Vertex<D>)> + '_ {
        (0..self.n_edges())
            .into_par_iter()
            .map(|i| (self.edge(i), self.edge_normal(i)))
    }

    /// Sequential iterator over the edges and edge normals
    fn edges_and_normals(&self) -> impl ExactSizeIterator<Item = ([usize; 2], Vertex<D>)> + '_ {
        (0..self.n_edges()).map(|i| (self.edge(i), self.edge_normal(i)))
    }

    /// Number of boundary faces
    fn n_boundary_faces(&self) -> usize;

    /// Parallel iterator over the boundary faces
    fn par_boundary_faces(
        &self,
    ) -> impl IndexedParallelIterator<Item = (usize, Tag, Vertex<D>)> + '_;

    /// Sequential iterator over the boundary faces
    fn boundary_faces(&self) -> impl ExactSizeIterator<Item = (usize, Tag, Vertex<D>)> + '_;

    /// Get the volume of the `i`th cell
    fn vol(&self, i: usize) -> f64 {
        self.elem(i)
            .iter()
            .map(|&(i, orient)| {
                let mut f: [usize; F] = self.face(i).try_into().unwrap();
                if !orient {
                    f.swap(0, 1);
                }
                self.gface(&f)
            })
            .map(|gf| cell_center(&gf).dot(&Face::<F>::normal(&gf)))
            .sum::<f64>()
            / D as f64
    }

    /// Parallel iterator over cell volumes
    fn par_vols(&self) -> impl IndexedParallelIterator<Item = f64> + '_ {
        (0..self.n_elems()).into_par_iter().map(|i| self.vol(i))
    }

    /// Sequential iterator over cell volumes
    fn vols(&self) -> impl ExactSizeIterator<Item = f64> + '_ {
        (0..self.n_elems()).map(|i| self.vol(i))
    }

    /// Check if polygonal cell `e` is closed
    fn is_closed(&self, e: &[(usize, bool)]) -> bool {
        let mut res = [0.0; D];

        e.iter()
            .map(|&(i, orient)| {
                let mut f: [usize; F] = self.face(i).try_into().unwrap();
                if !orient {
                    f.swap(0, 1);
                }
                self.gface(&f)
            })
            .for_each(|gf| {
                let n = Face::<F>::normal(&gf);
                res.iter_mut().zip(n.iter()).for_each(|(x, y)| *x += y);
            });
        res.iter().map(|x| x.abs()).sum::<f64>() < 1e-10
    }

    /// Check the validity of the dual mesh
    ///  - consistent number of faces and faces tags
    ///  - consistent face to vertex connectivity
    ///  - consistent element to face connectivity
    ///  - closed elements
    ///  - unique faces
    fn check(&self) -> Result<()> {
        // lengths
        if self.par_faces().len() != self.par_ftags().len() {
            return Err(Error::from("Inconsistent sizes (faces)"));
        }

        // indices
        if self
            .par_faces()
            .any(|f| f.iter().any(|&i| i >= self.n_verts()))
        {
            return Err(Error::from("Inconsistent indices (faces)"));
        }

        // faces
        if self
            .par_elems()
            .any(|e| e.iter().any(|&x| x.0 >= self.n_faces()))
        {
            return Err(Error::from("Inconsistent indices (elems)"));
        }

        // closed elements
        for (i, (e, v)) in self.elems().zip(self.vols()).enumerate() {
            if v < 0.0 {
                return Err(Error::from(&format!(
                    "Element {i} invalid: vol={v} < 0  ({e:?})"
                )));
            }
            if !self.is_closed(e) {
                return Err(Error::from(&format!("Element {i} not closed ({e:?})")));
            }
        }

        // all faces appear only once
        let mut faces = FxHashMap::with_hasher(FxBuildHasher);
        for (i, f) in self.faces().enumerate() {
            assert_eq!(f.len(), F);
            let mut res = [0; F];
            res.iter_mut().zip(f.iter()).for_each(|(x, y)| *x = *y);
            res.sort_unstable();
            if let std::collections::hash_map::Entry::Vacant(e) = faces.entry(res) {
                e.insert(i);
            } else {
                let j = *faces.get(&res).unwrap();
                return Err(Error::from(&format!(
                    "Face {i} ({f:?}) = face {j} ({:?})",
                    self.face(j)
                )));
            }
        }

        // faces appear at most once in 1 or 2 elements
        let mut flg = vec![0; self.n_faces()];
        for (i_elem, e) in self.elems().enumerate() {
            let mut tmp = FxHashSet::with_capacity_and_hasher(e.len(), FxBuildHasher);
            for &(i, sgn) in e {
                if tmp.contains(&i) {
                    return Err(Error::from(&format!(
                        "Face {i} appears multiple times in element {i_elem}"
                    )));
                }
                tmp.insert(i);
                if flg[i] == 0 {
                    if sgn {
                        flg[i] = 1;
                    } else {
                        flg[i] = -1;
                    }
                } else if flg[i] == 1 {
                    if sgn {
                        return Err(Error::from(&format!("Face {i} appears twice with True")));
                    }
                    flg[i] = 2;
                } else if flg[i] == -1 {
                    if !sgn {
                        return Err(Error::from(&format!("Face {i} appears twice with False")));
                    }
                    flg[i] = 2;
                } else if flg[i] == 2 {
                    return Err(Error::from(&format!("Face {i} appears 3 times")));
                } else {
                    unreachable!()
                }
            }
        }

        Ok(())
    }

    /// Return a `Mesh<D, C2, F2>` containing the faces such that `filter(tag)` is true.
    fn extract_faces<const C2: usize, const F2: usize, M: Mesh<D, C2, F2>, G: Fn(Tag) -> bool>(
        &self,
        filter: G,
    ) -> (M, Vec<usize>)
    where
        Cell<C2>: Simplex<C2>,
        Cell<F2>: Simplex<F2>,
    {
        assert_eq!(C2, C - 1);
        assert_eq!(F2, F - 1);
        let mut new_ids = vec![usize::MAX; self.n_verts()];
        let mut vert_ids = Vec::new();
        let mut next = 0;

        let n_faces = self
            .faces()
            .zip(self.ftags())
            .filter(|(_, t)| filter(*t))
            .map(|(f, _)| {
                for &i in f {
                    if new_ids[i] == usize::MAX {
                        new_ids[i] = next;
                        vert_ids.push(i);
                        next += 1;
                    }
                }
            })
            .count();
        let n_verts = next;

        let mut verts = vec![Vertex::<D>::zeros(); n_verts];
        let mut faces = Vec::with_capacity(n_faces);
        let mut ftags = Vec::with_capacity(n_faces);

        new_ids
            .iter()
            .enumerate()
            .filter(|&(_, j)| *j != usize::MAX)
            .for_each(|(i, &j)| verts[j] = self.vert(i));
        self.faces()
            .zip(self.ftags())
            .filter(|(f, _)| f.iter().all(|&i| new_ids[i] != usize::MAX))
            .for_each(|(f, t)| {
                faces.push(std::array::from_fn(|i| new_ids[f[i]]));
                ftags.push(t);
            });

        let mut res = M::empty();
        res.add_verts(verts.iter().copied());
        res.add_elems(faces.iter().copied(), ftags.iter().copied());

        (res, vert_ids)
    }

    /// Return a `Mesh<D, C2, F2>` containing all the boundary faces.
    fn boundary<const C2: usize, const F2: usize, M: Mesh<D, C2, F2>>(&self) -> (M, Vec<usize>)
    where
        Cell<C2>: Simplex<C2>,
        Cell<F2>: Simplex<F2>,
    {
        self.extract_faces(|t| t > 0)
    }
}

/// Get the barycentric coordinates of the circumcenter
pub fn circumcenter_bcoords<const D: usize, const C: usize>(v: &[Vertex<D>; C]) -> [f64; C] {
    assert!(C <= D + 1);

    let mut a = DMatrix::<f64>::zeros(C + 1, C + 1);
    let mut b = DVector::<f64>::zeros(C + 1);

    for i in 0..C {
        for j in i..C {
            a[(C + 1) * i + j] = 2.0 * v[i].dot(&v[j]);
            a[(C + 1) * j + i] = a[(C + 1) * i + j];
        }
        b[i] = v[i].dot(&v[i]);
    }
    b[C] = 1.0;
    let j = C;
    for i in 0..C {
        a[(C + 1) * i + j] = 1.0;
        a[(C + 1) * j + i] = 1.0;
    }

    a.lu().solve_mut(&mut b);

    let mut res = [0.0; C];
    for (i, &v) in b.iter().take(C).enumerate() {
        res[i] = v;
    }
    res
}

#[cfg(test)]
mod tests {
    use crate::{Vert2d, Vert3d, assert_delta, dual::circumcenter_bcoords, mesh::cell_vertex};
    use rand::{Rng, SeedableRng, rngs::StdRng};

    #[test]
    fn test_circumcenter_2d() {
        let mut rng = StdRng::seed_from_u64(1234);

        for _ in 0..100 {
            let p0 = Vert2d::from_fn(|_, _| rng.random::<f64>() - 0.5);
            let p1 = Vert2d::from_fn(|_, _| rng.random::<f64>() - 0.5);
            let p2 = Vert2d::from_fn(|_, _| rng.random::<f64>() - 0.5);
            let ge = [p0, p1, p2];
            let bcoords = circumcenter_bcoords(&ge);
            let p = cell_vertex(&ge, bcoords);
            let l0 = (p0 - p).norm();
            let l1 = (p1 - p).norm();
            let l2 = (p2 - p).norm();
            assert_delta!(l0, l1, 1e-12);
            assert_delta!(l0, l2, 1e-12);
        }
    }

    #[test]
    fn test_circumcenter_2d_edg() {
        let mut rng = StdRng::seed_from_u64(1234);

        for _ in 0..100 {
            let p0 = Vert2d::from_fn(|_, _| rng.random::<f64>() - 0.5);
            let p1 = Vert2d::from_fn(|_, _| rng.random::<f64>() - 0.5);
            let ge = [p0, p1];
            let bcoords = circumcenter_bcoords(&ge);
            let p = cell_vertex(&ge, bcoords);
            let l0 = (p0 - p).norm();
            let l1 = (p1 - p).norm();
            assert_delta!(l0, l1, 1e-12);
            assert_delta!(bcoords[0], 0.5, 1e-12);
            assert_delta!(bcoords[1], 0.5, 1e-12);
        }
    }

    #[test]
    fn test_circumcenter_3d() {
        let mut rng = StdRng::seed_from_u64(1234);

        for _ in 0..100 {
            let p0 = Vert3d::from_fn(|_, _| rng.random::<f64>() - 0.5);
            let p1 = Vert3d::from_fn(|_, _| rng.random::<f64>() - 0.5);
            let p2 = Vert3d::from_fn(|_, _| rng.random::<f64>() - 0.5);
            let p3 = Vert3d::from_fn(|_, _| rng.random::<f64>() - 0.5);
            let ge = [p0, p1, p2, p3];
            let bcoords = circumcenter_bcoords(&ge);
            let p = cell_vertex(&ge, bcoords);
            let l0 = (p0 - p).norm();
            let l1 = (p1 - p).norm();
            let l2 = (p2 - p).norm();
            let l3 = (p3 - p).norm();
            assert_delta!(l0, l1, 1e-12);
            assert_delta!(l0, l2, 1e-12);
            assert_delta!(l0, l3, 1e-12);
        }
    }

    #[test]
    fn test_circumcenter_3d_tri() {
        let mut rng = StdRng::seed_from_u64(1234);

        for _ in 0..100 {
            let p0 = Vert2d::from_fn(|_, _| rng.random::<f64>() - 0.5);
            let p1 = Vert2d::from_fn(|_, _| rng.random::<f64>() - 0.5);
            let p2 = Vert2d::from_fn(|_, _| rng.random::<f64>() - 0.5);
            let ge = [p0, p1, p2];
            let bcoords = circumcenter_bcoords(&ge);
            let p = cell_vertex(&ge, bcoords);
            let l0 = (p0 - p).norm();
            let l1 = (p1 - p).norm();
            let l2 = (p2 - p).norm();
            assert_delta!(l0, l1, 1e-12);
            assert_delta!(l0, l2, 1e-12);
        }
    }
}
