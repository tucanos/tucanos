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
    mesh::{Edge, GSimplex, Idx, Mesh, Simplex},
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
pub enum DualCellCenter<T: Idx, const D: usize, C: Simplex<T>> {
    Vertex(Vertex<D>),
    Face(C::FACE),
}

/// Dual of a `Mesh<D, C, F>`
pub trait DualMesh<T: Idx, const D: usize, C: Simplex<T>>: PolyMesh<T, D> {
    /// Compute the dual of `mesh`
    fn new<M: Mesh<T, D, C>>(msh: &M, t: DualType) -> Self;

    /// Display element `i`
    fn print_elem_info(&self, i: T) {
        println!("Dual element {i}");
        self.par_edges_and_normals()
            .filter(|&(e, _)| e[0] == i || e[1] == i)
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
    fn gface(&self, f: &C::FACE) -> <C::FACE as Simplex<T>>::GEOM<D> {
        <C::FACE as Simplex<T>>::GEOM::from_iter(f.into_iter().map(|i| self.vert(i)))
    }

    /// Parallel iterator over the faces vertex coordinates
    fn par_gfaces(
        &self,
    ) -> impl IndexedParallelIterator<Item = <C::FACE as Simplex<T>>::GEOM<D>> + '_ {
        self.par_faces().map(|f| {
            let f = C::FACE::from_iter(f.iter().copied());
            self.gface(&f)
        })
    }

    /// Sequential iterator over the faces vertex coordinates
    fn gfaces(&self) -> impl ExactSizeIterator<Item = <C::FACE as Simplex<T>>::GEOM<D>> + '_ {
        self.faces().map(|f| {
            let f = C::FACE::from_iter(f.iter().copied());
            self.gface(&f)
        })
    }

    /// Number of edges
    fn n_edges(&self) -> T;

    /// Get the `i`th edge
    fn edge(&self, i: T) -> Edge<T>;

    /// Parallel itertator over the edges
    fn par_edges(&self) -> impl IndexedParallelIterator<Item = Edge<T>> + '_ + Clone {
        (0..self.n_edges().try_into().unwrap())
            .into_par_iter()
            .map(|i| self.edge(i.try_into().unwrap()))
    }

    /// Sequential iterator over the edges
    fn edges(&self) -> impl ExactSizeIterator<Item = Edge<T>> + '_ + Clone {
        (0..self.n_edges().try_into().unwrap()).map(|i| self.edge(i.try_into().unwrap()))
    }

    /// Get the normal associated with the `i`th edge
    fn edge_normal(&self, i: T) -> Vertex<D>;

    /// Parallel iterator over the edges and edge normals
    fn par_edges_and_normals(
        &self,
    ) -> impl IndexedParallelIterator<Item = (Edge<T>, Vertex<D>)> + '_ {
        (0..self.n_edges().try_into().unwrap())
            .into_par_iter()
            .map(|i| {
                (
                    self.edge(i.try_into().unwrap()),
                    self.edge_normal(i.try_into().unwrap()),
                )
            })
    }

    /// Sequential iterator over the edges and edge normals
    fn edges_and_normals(&self) -> impl ExactSizeIterator<Item = (Edge<T>, Vertex<D>)> + '_ {
        (0..self.n_edges().try_into().unwrap()).map(|i| {
            (
                self.edge(i.try_into().unwrap()),
                self.edge_normal(i.try_into().unwrap()),
            )
        })
    }

    /// Number of boundary faces
    fn n_boundary_faces(&self) -> T;

    /// Parallel iterator over the boundary faces
    fn par_boundary_faces(&self) -> impl IndexedParallelIterator<Item = (T, Tag, Vertex<D>)> + '_;

    /// Sequential iterator over the boundary faces
    fn boundary_faces(&self) -> impl ExactSizeIterator<Item = (T, Tag, Vertex<D>)> + '_;

    /// Get the volume of the `i`th cell
    fn vol(&self, i: T) -> f64 {
        self.elem(i)
            .iter()
            .map(|&(i, orient)| {
                let mut f = C::FACE::from_iter(self.face(i).iter().copied());
                if !orient {
                    f.invert();
                }
                self.gface(&f)
            })
            .map(|gf| gf.center().dot(&gf.normal()))
            .sum::<f64>()
            / D as f64
    }

    /// Parallel iterator over cell volumes
    fn par_vols(&self) -> impl IndexedParallelIterator<Item = f64> + '_ {
        (0..self.n_elems().try_into().unwrap())
            .into_par_iter()
            .map(|i| self.vol(i.try_into().unwrap()))
    }

    /// Sequential iterator over cell volumes
    fn vols(&self) -> impl ExactSizeIterator<Item = f64> + '_ {
        (0..self.n_elems().try_into().unwrap()).map(|i| self.vol(i.try_into().unwrap()))
    }

    /// Check if polygonal cell `e` is closed
    fn is_closed(&self, e: &[(T, bool)]) -> bool {
        let mut res = [0.0; D];

        e.iter()
            .map(|&(i, orient)| {
                let mut f = C::FACE::from_iter(self.face(i).iter().copied());
                if !orient {
                    f.invert();
                }
                self.gface(&f)
            })
            .for_each(|gf| {
                let n = gf.normal();
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
            assert_eq!(f.len(), C::FACE::N_VERTS);
            let res = C::FACE::from_iter(f.iter().copied()).sorted();
            if let std::collections::hash_map::Entry::Vacant(e) = faces.entry(res) {
                e.insert(i.try_into().unwrap());
            } else {
                let j = *faces.get(&res).unwrap();
                return Err(Error::from(&format!(
                    "Face {i} ({f:?}) = face {j} ({:?})",
                    self.face(j)
                )));
            }
        }

        // faces appear at most once in 1 or 2 elements
        let mut flg = vec![0; self.n_faces().try_into().unwrap()];
        for (i_elem, e) in self.elems().enumerate() {
            let mut tmp = FxHashSet::with_capacity_and_hasher(e.len(), FxBuildHasher);
            for &(i, sgn) in e {
                if tmp.contains(&i) {
                    return Err(Error::from(&format!(
                        "Face {i} appears multiple times in element {i_elem}"
                    )));
                }
                tmp.insert(i);
                let i = i.try_into().unwrap();
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
    fn extract_faces<M: Mesh<T, D, C::FACE>, G: Fn(Tag) -> bool>(&self, filter: G) -> (M, Vec<T>) {
        // find out if faces needs to be inverted (boundary faces will only be seen once)
        let mut flg = vec![true; self.n_faces().try_into().unwrap()];
        self.elems()
            .flatten()
            .for_each(|&(i, x)| flg[i.try_into().unwrap()] = x);

        let mut new_ids = vec![T::MAX; self.n_verts().try_into().unwrap()];
        let mut vert_ids = Vec::new();
        let mut next = T::ZERO;

        let n_faces = self
            .faces()
            .zip(self.ftags())
            .filter(|(_, t)| filter(*t))
            .map(|(f, _)| {
                for &i in f {
                    if new_ids[i.try_into().unwrap()] == T::MAX {
                        new_ids[i.try_into().unwrap()] = next;
                        vert_ids.push(i);
                        next += T::ONE;
                    }
                }
            })
            .count();
        let n_verts = next;

        let mut verts = vec![Vertex::<D>::zeros(); n_verts.try_into().unwrap()];
        let mut faces = Vec::with_capacity(n_faces);
        let mut ftags = Vec::with_capacity(n_faces);

        new_ids
            .iter()
            .enumerate()
            .filter(|&(_, j)| *j != T::MAX)
            .for_each(|(i, &j)| verts[j.try_into().unwrap()] = self.vert(i.try_into().unwrap()));
        self.faces()
            .zip(self.ftags())
            .zip(flg.iter())
            .filter(|((f, _), _)| f.iter().all(|&i| new_ids[i.try_into().unwrap()] != T::MAX))
            .for_each(|((f, t), &invert)| {
                let mut f = C::FACE::from_iter(f.iter().map(|&i| new_ids[i.try_into().unwrap()]));
                if invert {
                    f.invert();
                }
                faces.push(f);
                ftags.push(t);
            });

        let mut res = M::empty();
        res.add_verts(verts.iter().copied());
        res.add_elems(faces.iter().copied(), ftags.iter().copied());

        (res, vert_ids)
    }

    /// Return a `Mesh<D, C2, F2>` containing all the boundary faces.
    fn boundary<M: Mesh<T, D, C::FACE>>(&self) -> (M, Vec<T>) {
        self.extract_faces(|t| t > 0)
    }
}

/// Get the barycentric coordinates of the circumcenter
pub fn circumcenter_bcoords<const D: usize, G: GSimplex<D>>(v: &G) -> G::BCOORDS {
    assert!(G::N_VERTS <= D + 1);

    let mut a = DMatrix::<f64>::zeros(G::N_VERTS + 1, G::N_VERTS + 1);
    let mut b = DVector::<f64>::zeros(G::N_VERTS + 1);

    for i in 0..G::N_VERTS {
        for j in i..G::N_VERTS {
            a[(G::N_VERTS + 1) * i + j] = 2.0 * v[i].dot(&v[j]);
            a[(G::N_VERTS + 1) * j + i] = a[(G::N_VERTS + 1) * i + j];
        }
        b[i] = v[i].dot(&v[i]);
    }
    b[G::N_VERTS] = 1.0;
    let j = G::N_VERTS;
    for i in 0..G::N_VERTS {
        a[(G::N_VERTS + 1) * i + j] = 1.0;
        a[(G::N_VERTS + 1) * j + i] = 1.0;
    }

    a.lu().solve_mut(&mut b);

    let mut res = G::BCOORDS::default();
    for (i, &v) in b.iter().take(G::N_VERTS).enumerate() {
        res[i] = v;
    }
    res
}

#[cfg(test)]
mod tests {
    use crate::{
        Vert2d, Vert3d, assert_delta,
        dual::circumcenter_bcoords,
        mesh::{GEdge, GSimplex, GTetrahedron, GTriangle},
    };
    use rand::{Rng, SeedableRng, rngs::StdRng};

    #[test]
    fn test_circumcenter_2d() {
        let mut rng = StdRng::seed_from_u64(1234);

        for _ in 0..100 {
            let p0 = Vert2d::from_fn(|_, _| rng.random::<f64>() - 0.5);
            let p1 = Vert2d::from_fn(|_, _| rng.random::<f64>() - 0.5);
            let p2 = Vert2d::from_fn(|_, _| rng.random::<f64>() - 0.5);
            let ge = GTriangle::from([p0, p1, p2]);
            let bcoords = circumcenter_bcoords(&ge);
            let p = ge.vert(&bcoords);
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
            let ge = GEdge::from([p0, p1]);
            let bcoords = circumcenter_bcoords(&ge);
            let p = ge.vert(&bcoords);
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
            let ge = GTetrahedron::from([p0, p1, p2, p3]);
            let bcoords = circumcenter_bcoords(&ge);
            let p = ge.vert(&bcoords);
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
            let ge = GTriangle::from([p0, p1, p2]);
            let bcoords = circumcenter_bcoords(&ge);
            let p = ge.vert(&bcoords);
            let l0 = (p0 - p).norm();
            let l1 = (p1 - p).norm();
            let l2 = (p2 - p).norm();
            assert_delta!(l0, l1, 1e-12);
            assert_delta!(l0, l2, 1e-12);
        }
    }
}
