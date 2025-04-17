use crate::{
    mesh::{cell_center, Mesh},
    poly_mesh::PolyMesh,
    simplices::Simplex,
    Cell, Error, Face, Result, Tag, Vertex,
};
use nalgebra::{DMatrix, DVector};
use rayon::prelude::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

#[derive(Clone, Copy, Debug)]
pub enum DualType {
    Median,
    Barth,
}

pub trait DualMesh<const D: usize, const C: usize, const F: usize>: PolyMesh<D>
where
    Cell<C>: Simplex<C>,
    Cell<F>: Simplex<F>,
{
    fn new<M: Mesh<D, C, F>>(msh: &M, t: DualType) -> Self;

    fn print_elem_info(&self, i: usize) {
        println!("Dual element {i}");
        self.edges_and_normals()
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
        self.boundary_faces()
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
    fn gface(&self, f: &[usize; F]) -> [&Vertex<D>; F] {
        let mut res = [self.vert(0); F];
        for (j, &k) in f.iter().enumerate() {
            res[j] = self.vert(k);
        }
        res
    }
    fn gfaces(&self) -> impl IndexedParallelIterator<Item = [&Vertex<D>; F]> + '_ {
        self.faces().map(|f| {
            let f = f.try_into().unwrap();
            self.gface(&f)
        })
    }
    fn seq_gfaces(&self) -> impl ExactSizeIterator<Item = [&Vertex<D>; F]> + '_ {
        self.seq_faces().map(|f| {
            let f = f.try_into().unwrap();
            self.gface(&f)
        })
    }
    fn n_edges(&self) -> usize;
    fn edge(&self, i: usize) -> [usize; 2];
    fn edges(&self) -> impl IndexedParallelIterator<Item = [usize; 2]> + '_ + Clone {
        (0..self.n_edges()).into_par_iter().map(|i| self.edge(i))
    }
    fn seq_edges(&self) -> impl ExactSizeIterator<Item = [usize; 2]> + '_ + Clone {
        (0..self.n_edges()).map(|i| self.edge(i))
    }
    fn edge_normal(&self, i: usize) -> Vertex<D>;
    fn edges_and_normals(
        &self,
    ) -> impl IndexedParallelIterator<Item = ([usize; 2], Vertex<D>)> + '_ {
        (0..self.n_edges())
            .into_par_iter()
            .map(|i| (self.edge(i), self.edge_normal(i)))
    }
    fn seq_edges_and_normals(&self) -> impl ExactSizeIterator<Item = ([usize; 2], Vertex<D>)> + '_ {
        (0..self.n_edges()).map(|i| (self.edge(i), self.edge_normal(i)))
    }
    fn n_boundary_faces(&self) -> usize;
    fn boundary_faces(&self) -> impl IndexedParallelIterator<Item = (usize, Tag, Vertex<D>)> + '_;
    fn seq_boundary_faces(&self) -> impl ExactSizeIterator<Item = (usize, Tag, Vertex<D>)> + '_;
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
            .map(|gf| cell_center(gf).dot(&Face::<F>::normal(gf)))
            .sum::<f64>()
            / D as f64
    }
    fn vols(&self) -> impl IndexedParallelIterator<Item = f64> + '_ {
        (0..self.n_elems()).into_par_iter().map(|i| self.vol(i))
    }
    fn seq_vols(&self) -> impl ExactSizeIterator<Item = f64> + '_ {
        (0..self.n_elems()).map(|i| self.vol(i))
    }
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
                let n = Face::<F>::normal(gf);
                res.iter_mut().zip(n.iter()).for_each(|(x, y)| *x += y)
            });
        res.iter().map(|x| x.abs()).sum::<f64>() < 1e-10
    }
    fn check(&self) -> Result<()> {
        // lengths
        if self.faces().len() != self.ftags().len() {
            return Err(Error::from("Inconsistent sizes (faces)"));
        }

        // indices
        if self.faces().any(|f| f.iter().any(|&i| i >= self.n_verts())) {
            return Err(Error::from("Inconsistent indices (faces)"));
        }

        // faces
        if self
            .elems()
            .any(|e| e.iter().any(|&x| x.0 >= self.n_faces()))
        {
            return Err(Error::from("Inconsistent indices (elems)"));
        }

        // closed elements
        for (e, v) in self.seq_elems().zip(self.seq_vols()) {
            if v < 0.0 {
                return Err(Error::from(&format!("Volume of {e:?} = {v} < 0")));
            }
            if !self.is_closed(e) {
                return Err(Error::from(&format!("Element {e:?} not closed")));
            }
        }

        Ok(())
    }

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
            .seq_faces()
            .zip(self.seq_ftags())
            .filter(|(_, t)| filter(*t))
            .map(|(f, _)| {
                f.iter().for_each(|&i| {
                    if new_ids[i] == usize::MAX {
                        new_ids[i] = next;
                        vert_ids.push(i);
                        next += 1;
                    }
                })
            })
            .count();
        let n_verts = next;

        let mut verts = vec![Vertex::<D>::zeros(); n_verts];
        let mut faces = Vec::with_capacity(n_faces);
        let mut ftags = Vec::with_capacity(n_faces);

        new_ids
            .iter()
            .enumerate()
            .filter(|(_, &j)| j != usize::MAX)
            .for_each(|(i, &j)| verts[j] = *self.vert(i));
        self.seq_faces()
            .zip(self.seq_ftags())
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

    fn boundary<const C2: usize, const F2: usize, M: Mesh<D, C2, F2>>(&self) -> (M, Vec<usize>)
    where
        Cell<C2>: Simplex<C2>,
        Cell<F2>: Simplex<F2>,
    {
        self.extract_faces(|t| t > 0)
    }
}

pub fn circumcenter_bcoords<const D: usize, const C: usize>(v: [&Vertex<D>; C]) -> [f64; C] {
    assert!(C <= D + 1);

    let mut a = DMatrix::<f64>::zeros(C + 1, C + 1);
    let mut b = DVector::<f64>::zeros(C + 1);

    for i in 0..C {
        for j in i..C {
            a[(C + 1) * i + j] = 2.0 * v[i].dot(v[j]);
            a[(C + 1) * j + i] = a[(C + 1) * i + j];
        }
        b[i] = v[i].dot(v[i]);
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
