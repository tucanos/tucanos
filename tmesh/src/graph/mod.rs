//! Basic graphs to compute and store connectivities that can not be stored in
//! simple 2d arrays
use crate::{Error, Result, mesh::Idx};
use rustc_hash::{FxBuildHasher, FxHashMap};
use std::fmt::Debug;

/// Compute the indices that would sort `data`
fn argsort<T: Idx>(data: &[T]) -> Vec<T> {
    let mut indices = (0..data.len())
        .map(|x| x.try_into().unwrap())
        .collect::<Vec<T>>();
    indices.sort_by_key(|&i| &data[i.try_into().unwrap()]);
    indices
}

/// Renumber the vertices in order to have contininuous indices, and return he map from old to nex indices
#[allow(dead_code)]
#[must_use]
pub fn reindex<T: Idx, const N: usize>(elems: &[[T; N]]) -> (Vec<[T; N]>, FxHashMap<T, T>) {
    let mut map = FxHashMap::with_hasher(FxBuildHasher);
    let mut next: T = T::ZERO;
    for i in elems.iter().copied().flatten() {
        if let std::collections::hash_map::Entry::Vacant(e) = map.entry(i) {
            e.insert(next);
            next += T::ONE;
        }
    }

    let new_elems = elems
        .iter()
        .map(|&e| {
            let mut res = e;
            for x in &mut res {
                *x = *map.get(x).unwrap();
            }
            res
        })
        .collect();

    (new_elems, map)
}

/// CSR representation of a graph
#[derive(Debug, Default, Clone)]
pub struct CSRGraph<T: Idx = usize> {
    ptr: Vec<T>,
    indices: Vec<T>,
    values: Option<Vec<T>>,
    m: T,
}

impl<T: Idx> CSRGraph<T> {
    fn set_ptr<E: IntoIterator<Item = T> + Copy, I: ExactSizeIterator<Item = E> + Clone>(
        elems: I,
        n_verts: Option<T>,
    ) -> Self
    where
        <T as TryInto<T>>::Error: Debug,
    {
        let nv = n_verts.unwrap_or_else(|| {
            elems
                .clone()
                .flatten()
                .filter(|&i| i != T::MAX)
                .max()
                .unwrap_or_else(|| T::ZERO)
                + T::ONE
        });
        let n = elems.clone().flatten().filter(|&i| i != T::MAX).count();

        let mut res = Self {
            ptr: vec![T::ZERO; nv.try_into().unwrap() + 1],
            indices: vec![T::MAX; n],
            values: None,
            m: T::ZERO,
        };

        for i in elems.flatten().filter(|&i| i != T::MAX) {
            res.ptr[i.try_into().unwrap() + 1] += T::ONE;
        }

        for i in 0..nv.try_into().unwrap() {
            let tmp = res.ptr[i];
            res.ptr[i + 1] += tmp;
        }

        res
    }

    /// Sort the indices for every vertex in the graph
    pub fn sort(&mut self) {
        let n = self.ptr.len() - 1;
        for i in 0..n {
            let start = self.ptr[i].try_into().unwrap();
            let end = self.ptr[i + 1].try_into().unwrap();
            self.indices[start..end].sort_unstable();
            for j in start + 1..end {
                assert_ne!(self.indices[j], self.indices[j - 1]);
            }
        }
    }

    /// Create a new graph explicitely
    #[must_use]
    pub fn new(ptr: Vec<T>, indices: Vec<T>) -> Self {
        let m = indices.iter().copied().max().unwrap();
        let mut res = Self {
            ptr,
            indices,
            values: None,
            m,
        };
        res.sort();
        res
    }

    /// Create a graph from edges
    pub fn from_edges<I: ExactSizeIterator<Item = [T; 2]> + Clone>(
        edgs: I,
        n_verts: Option<T>,
    ) -> Self {
        let mut res = Self::set_ptr(edgs.clone(), n_verts);
        res.m = res.n();

        for [i0, i1] in edgs {
            assert!(i0 != T::MAX);
            assert!(i1 != T::MAX);
            let mut ok = false;
            for j in res.ptr[i0.try_into().unwrap()].try_into().unwrap()
                ..res.ptr[i0.try_into().unwrap() + 1].try_into().unwrap()
            {
                if res.indices[j] == T::MAX {
                    res.indices[j] = i1;
                    ok = true;
                    break;
                }
            }
            assert!(ok);
            let mut ok = false;
            for j in res.ptr[i1.try_into().unwrap()].try_into().unwrap()
                ..res.ptr[i1.try_into().unwrap() + 1].try_into().unwrap()
            {
                if res.indices[j] == T::MAX {
                    res.indices[j] = i0;
                    ok = true;
                    break;
                }
            }
            assert!(ok);
        }
        res.sort();
        res
    }

    /// Compute the vertex to element connectivity from an element to vertex connectivity
    pub fn transpose<E: IntoIterator<Item = T> + Copy, I: ExactSizeIterator<Item = E> + Clone>(
        elems: I,
        n_verts: Option<T>,
    ) -> Self {
        let mut res = Self::set_ptr(elems.clone(), n_verts);
        res.m = elems.len().try_into().unwrap();
        let mut values = vec![T::ZERO; res.indices.len()];

        for (i, e) in elems.enumerate() {
            for (v, i_vert) in e.into_iter().filter(|&i| i != T::MAX).enumerate() {
                let start = res.ptr[i_vert.try_into().unwrap()].try_into().unwrap();
                let end = res.ptr[i_vert.try_into().unwrap() + 1].try_into().unwrap();
                let mut ok = false;
                #[allow(clippy::needless_range_loop)]
                for j in start..end {
                    if res.indices[j] == T::MAX {
                        res.indices[j] = i.try_into().unwrap();
                        values[j] = v.try_into().unwrap();
                        ok = true;
                        break;
                    }
                }
                assert!(ok);
            }
        }
        res.sort();
        res.values = Some(values);
        res
    }

    /// Number of vertices
    #[must_use]
    pub fn n(&self) -> T {
        (self.ptr.len() - 1).try_into().unwrap()
    }

    /// Number of columns
    #[must_use]
    pub const fn m(&self) -> T {
        self.m
    }

    /// Number of edges
    #[must_use]
    pub fn n_edges(&self) -> T {
        self.indices.len().try_into().unwrap()
    }

    /// Get the neighbors of the `i`th vertex
    #[must_use]
    pub fn row(&self, i: T) -> &[T] {
        let start = self.ptr[i.try_into().unwrap()].try_into().unwrap();
        let end = self.ptr[i.try_into().unwrap() + 1].try_into().unwrap();
        &self.indices[start..end]
    }

    /// Get the indices corresponding to the `i`th vertex
    #[must_use]
    pub fn row_ptr(&self, i: T) -> impl ExactSizeIterator<Item = usize> {
        let start = self.ptr[i.try_into().unwrap()].try_into().unwrap();
        let end = self.ptr[i.try_into().unwrap() + 1].try_into().unwrap();
        start..end
    }

    /// Sequential iterator over the rows
    #[must_use]
    pub fn rows(&self) -> impl ExactSizeIterator<Item = &[T]> {
        (0..self.n().try_into().unwrap()).map(|i| self.row(i.try_into().unwrap()))
    }

    /// Sequential iterator over the rows and values
    #[must_use]
    pub fn row_and_values(&self, i: T) -> (&[T], &[T]) {
        let start = self.ptr[i.try_into().unwrap()].try_into().unwrap();
        let end = self.ptr[i.try_into().unwrap() + 1].try_into().unwrap();
        (
            &self.indices[start..end],
            &self.values.as_ref().unwrap()[start..end],
        )
    }

    #[must_use]
    fn node_degrees(&self) -> Vec<T> {
        let mut res = Vec::with_capacity(self.n().try_into().unwrap());
        for (i_row, row) in self.rows().enumerate() {
            let mut n = row.len();
            if row.contains(&i_row.try_into().unwrap()) {
                n += 1;
            }
            res.push(n.try_into().unwrap());
        }
        res
    }

    /// Compute the Reverse Cuthill McKee ordering
    #[must_use]
    pub fn reverse_cuthill_mckee(&self) -> Vec<T> {
        // strongly inspired from scipy
        let mut order = vec![T::ZERO; self.n().try_into().unwrap()];
        let degree = self.node_degrees();
        let inds = argsort(&degree);
        let mut flg = vec![true; self.n().try_into().unwrap()];
        let rev_inds = argsort(&inds);
        let mut tmp_degrees =
            vec![T::ZERO; degree.iter().copied().max().unwrap().try_into().unwrap()];
        let mut n = 0;

        for idx in 0..self.n().try_into().unwrap() {
            if flg[idx] {
                let seed = inds[idx];
                order[n] = seed;
                n += 1;
                flg[rev_inds[seed.try_into().unwrap()].try_into().unwrap()] = false;
                let mut level_start = n - 1;
                let mut level_end = n;

                while level_start < level_end {
                    for level in level_start..level_end {
                        let i = order[level];
                        let n_old = n;

                        for &j in self.row(i) {
                            if flg[rev_inds[j.try_into().unwrap()].try_into().unwrap()] {
                                flg[rev_inds[j.try_into().unwrap()].try_into().unwrap()] = false;
                                order[n] = j;
                                n += 1;
                            }
                        }

                        let mut level_len = 0;
                        for k in n_old..n {
                            tmp_degrees[level_len] = degree[order[k].try_into().unwrap()];
                            level_len += 1;
                        }

                        for k in 1..level_len {
                            let tmp = tmp_degrees[k];
                            let tmp2 = order[n_old + k];
                            let mut l = k;
                            while l > 0 && tmp < tmp_degrees[l - 1] {
                                tmp_degrees[l] = tmp_degrees[l - 1];
                                order[n_old + l] = order[n_old + l - 1];
                                l -= 1;
                            }
                            tmp_degrees[l] = tmp;
                            order[n_old + l] = tmp2;
                        }
                    }

                    level_start = level_end;
                    level_end = n;
                }
            }

            if n == self.n().try_into().unwrap() {
                break;
            }
        }

        // return reversed order for RCM ordering
        order.iter().rev().copied().collect()
    }

    /// Compute the connected components
    pub fn connected_components(&self) -> Result<Vec<T>> {
        Ok(ConnectedComponents::new(self)?.vtag)
    }

    /// Extract a sub-graph
    #[must_use]
    pub fn subgraph<I: Iterator<Item = T>>(&self, ids: I) -> Self {
        let mut new_ids = vec![T::MAX; self.n().try_into().unwrap()];
        let mut m = 0;
        for (i, j) in ids.enumerate() {
            new_ids[j.try_into().unwrap()] = i.try_into().unwrap();
            m += 1;
        }
        let mut ptr = vec![T::ZERO];
        let mut indices = Vec::new();
        let mut values = Vec::new();
        for (old_i, &new_i) in new_ids.iter().enumerate() {
            if new_i != T::MAX {
                for k in self.row_ptr(old_i.try_into().unwrap()) {
                    let old_j = self.indices[k];
                    let new_j = new_ids[old_j.try_into().unwrap()];
                    if new_j != T::MAX {
                        indices.push(new_j);
                        if let Some(v) = &self.values {
                            values.push(v[k]);
                        }
                    }
                }
                ptr.push(indices.len().try_into().unwrap());
            }
        }
        let values = if self.values.is_none() {
            None
        } else {
            Some(values)
        };
        Self {
            ptr,
            indices,
            values,
            m: m.try_into().unwrap(),
        }
    }
}

/// Connected components of a graph
#[derive(Debug)]
struct ConnectedComponents<T: Idx> {
    pub vtag: Vec<T>,
}

impl<T: Idx> ConnectedComponents<T> {
    /// Compute the connected components of a CSR graph
    pub fn new(g: &CSRGraph<T>) -> Result<Self> {
        assert_eq!(g.n(), g.m());

        let mut res = Self {
            vtag: vec![T::MAX; g.n().try_into().unwrap()],
        };
        res.compute(g)?;
        Ok(res)
    }

    fn compute_from(&mut self, g: &CSRGraph<T>, starts: &[T], component: T) {
        let mut next_starts = Vec::new();
        for &start in starts {
            self.vtag[start.try_into().unwrap()] = component;
            for i in g.row(start).iter().copied() {
                if self.vtag[i.try_into().unwrap()] == T::MAX {
                    self.vtag[i.try_into().unwrap()] = component;
                    next_starts.push(i);
                }
            }
        }
        if !starts.is_empty() {
            self.compute_from(g, &next_starts, component);
        }
    }

    fn compute(&mut self, g: &CSRGraph<T>) -> Result<()> {
        let mut start = 0;
        let mut component = T::ZERO;
        while start < g.n().try_into().unwrap() {
            self.compute_from(g, &[start.try_into().unwrap()], component);
            while start < g.n().try_into().unwrap() && self.vtag[start] < T::MAX {
                start += 1;
            }
            component += T::ONE;
            if component == T::MAX {
                return Err(Error::from("too many connected components"));
            }
            assert!(component < T::MAX);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::graph::{CSRGraph, reindex};

    #[test]
    fn test_reindex() {
        let g: [[u32; 2]; 4] = [[0, 1], [1, 2], [2, 0], [5, 6]];
        let (new_elems, _) = reindex(&g);
        assert_eq!(new_elems.iter().copied().flatten().max().unwrap(), 4);
        assert_eq!(new_elems.len(), 4);
    }

    #[test]
    fn test_csr_edges() {
        let g: [[u32; 2]; 4] = [[0, 1], [1, 2], [2, 0], [3, 4]];
        let g = CSRGraph::from_edges(g.iter().copied(), None);
        assert_eq!(g.n(), 5);
        assert_eq!(g.m(), 5);
        assert_eq!(g.n_edges(), 8);
        let edgs = g.row(0);
        assert_eq!(*edgs, [1, 2]);
        let edgs = g.row(1);
        assert_eq!(*edgs, [0, 2]);
        let edgs = g.row(2);
        assert_eq!(*edgs, [0, 1]);
        let edgs = g.row(3);
        assert_eq!(*edgs, [4]);
        let edgs = g.row(4);
        assert_eq!(*edgs, [3]);
    }

    #[test]
    fn test_csr_triangles() {
        let g = [[0_u32, 1, 2], [0, 2, 3]];

        let g = CSRGraph::transpose(g.iter().copied(), None);
        assert_eq!(g.n(), 4);
        assert_eq!(g.m(), 2);

        let edgs = g.row(0);
        assert_eq!(*edgs, [0, 1]);
        let edgs = g.row(1);
        assert_eq!(*edgs, [0]);
        let edgs = g.row(2);
        assert_eq!(*edgs, [0, 1]);
        let edgs = g.row(3);
        assert_eq!(*edgs, [1]);
    }

    #[test]
    fn test_csr_v2e() {
        let mut edgs = vec![[0, 1], [1, 2], [2, 3], [3, 0], [0, 2]];
        let bdy = [0, 1, 1, 2, 2, 3, 3, 0];
        let bdy = bdy
            .iter()
            .copied()
            .map(|i| [i, usize::MAX])
            .collect::<Vec<_>>();
        edgs.extend(bdy.iter());

        let g = CSRGraph::transpose(edgs.iter().copied(), None);
        assert_eq!(g.n(), 4);
        assert_eq!(g.m(), 13);

        let (edgs, vals) = g.row_and_values(0);
        assert_eq!(*edgs, [0, 3, 4, 5, 12]);
        assert_eq!(*vals, [0, 1, 0, 0, 0]);

        let (edgs, vals) = g.row_and_values(1);
        assert_eq!(*edgs, [0, 1, 6, 7]);
        assert_eq!(*vals, [1, 0, 0, 0]);
    }

    #[test]
    fn test_rcm() {
        let graph = CSRGraph::<usize>::new(
            vec![0, 1, 2, 5, 5, 7, 8, 8, 9, 10, 12],
            vec![1, 4, 2, 3, 8, 2, 9, 2, 5, 6, 6, 7],
        );
        let ids = graph.reverse_cuthill_mckee();
        // computed using the scipy code and forcing stable sort
        assert_eq!(ids, [5, 8, 7, 2, 9, 4, 1, 0, 6, 3]);
    }

    #[test]
    fn test_cc() {
        let g = [[0_usize, 1], [1, 2], [2, 0], [3, 4]];
        let g = CSRGraph::from_edges(g.iter().copied(), None);
        let cc = g.connected_components().unwrap();
        assert_eq!(cc, [0, 0, 0, 1, 1]);
    }
}
