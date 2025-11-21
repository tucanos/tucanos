//! Basic graphs to compute and store connectivities that can not be stored in
//! simple 2d arrays
use crate::{Error, Result};
use rustc_hash::{FxBuildHasher, FxHashMap};
use std::fmt::Debug;

/// Compute the indices that would sort `data`
fn argsort(data: &[usize]) -> Vec<usize> {
    let mut indices = (0..data.len()).collect::<Vec<_>>();
    indices.sort_by_key(|&i| &data[i]);
    indices
}

/// Renumber the vertices in order to have contininuous indices, and return he map from old to nex indices
#[allow(dead_code)]
#[must_use]
pub fn reindex<const N: usize>(elems: &[[usize; N]]) -> (Vec<[usize; N]>, FxHashMap<usize, usize>) {
    let mut map = FxHashMap::with_hasher(FxBuildHasher);
    let mut next = 0;
    for i in elems.iter().copied().flatten() {
        if let std::collections::hash_map::Entry::Vacant(e) = map.entry(i) {
            e.insert(next);
            next += 1;
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
pub struct CSRGraph {
    ptr: Vec<usize>,
    indices: Vec<usize>,
    values: Option<Vec<usize>>,
    m: usize,
}

impl CSRGraph {
    fn set_ptr<T: TryInto<usize> + Ord + Debug>(
        elems: impl ExactSizeIterator<Item = impl IntoIterator<Item = T> + Copy> + Clone,
        n_verts: Option<usize>,
    ) -> Self
    where
        <T as TryInto<usize>>::Error: Debug,
    {
        let nv = n_verts.unwrap_or_else(|| {
            elems
                .clone()
                .flatten()
                .map(|i| i.try_into().unwrap())
                .filter(|&i| i != usize::MAX)
                .max()
                .unwrap_or(0)
                + 1
        });
        let n = elems
            .clone()
            .flatten()
            .map(|i| i.try_into().unwrap())
            .filter(|&i| i != usize::MAX)
            .count();

        let mut res = Self {
            ptr: vec![0; nv + 1],
            indices: vec![usize::MAX; n],
            values: None,
            m: 0,
        };

        for i in elems
            .flatten()
            .map(|i| i.try_into().unwrap())
            .filter(|&i| i != usize::MAX)
        {
            res.ptr[i + 1] += 1;
        }

        for i in 0..nv {
            res.ptr[i + 1] += res.ptr[i];
        }

        res
    }

    /// Sort the indices for every vertex in the graph
    pub fn sort(&mut self) {
        let n = self.ptr.len() - 1;
        for i in 0..n {
            let start = self.ptr[i];
            let end = self.ptr[i + 1];
            self.indices[start..end].sort_unstable();
            for j in start + 1..end {
                assert_ne!(self.indices[j], self.indices[j - 1]);
            }
        }
    }

    /// Create a new graph explicitely
    #[must_use]
    pub fn new(ptr: Vec<usize>, indices: Vec<usize>) -> Self {
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
    pub fn from_edges<T: TryInto<usize> + Ord + Debug + Copy>(
        edgs: impl ExactSizeIterator<Item = [T; 2]> + Clone,
        n_verts: Option<usize>,
    ) -> Self
    where
        <T as TryInto<usize>>::Error: Debug,
    {
        let mut res = Self::set_ptr(edgs.clone(), n_verts);
        res.m = res.n();

        for e in edgs {
            let i0 = e[0].try_into().unwrap();
            let i1 = e[1].try_into().unwrap();
            assert!(i0 != usize::MAX);
            assert!(i1 != usize::MAX);
            let mut ok = false;
            for j in res.ptr[i0]..res.ptr[i0 + 1] {
                if res.indices[j] == usize::MAX {
                    res.indices[j] = i1;
                    ok = true;
                    break;
                }
            }
            assert!(ok);
            let mut ok = false;
            for j in res.ptr[i1]..res.ptr[i1 + 1] {
                if res.indices[j] == usize::MAX {
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
    pub fn transpose<T: TryInto<usize> + Ord + Debug>(
        elems: impl ExactSizeIterator<Item = impl IntoIterator<Item = T> + Copy> + Clone,
        n_verts: Option<usize>,
    ) -> Self
    where
        <T as TryInto<usize>>::Error: Debug,
    {
        let mut res = Self::set_ptr(elems.clone(), n_verts);
        res.m = elems.len();
        let mut values = vec![0; res.indices.len()];

        for (i, e) in elems.enumerate() {
            for (v, i_vert) in e
                .into_iter()
                .map(|i| i.try_into().unwrap())
                .filter(|&i| i != usize::MAX)
                .enumerate()
            {
                let start = res.ptr[i_vert];
                let end = res.ptr[i_vert + 1];
                let mut ok = false;
                #[allow(clippy::needless_range_loop)]
                for j in start..end {
                    if res.indices[j] == usize::MAX {
                        res.indices[j] = i;
                        values[j] = v;
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
    pub const fn n(&self) -> usize {
        self.ptr.len() - 1
    }

    /// Number of columns
    #[must_use]
    pub const fn m(&self) -> usize {
        self.m
    }

    /// Number of edges
    #[must_use]
    pub const fn n_edges(&self) -> usize {
        self.indices.len()
    }

    /// Get the neighbors of the `i`th vertex
    #[must_use]
    pub fn row(&self, i: usize) -> &[usize] {
        let start = self.ptr[i];
        let end = self.ptr[i + 1];
        &self.indices[start..end]
    }

    /// Get the indices corresponding to the `i`th vertex
    #[must_use]
    pub fn row_ptr(&self, i: usize) -> impl ExactSizeIterator<Item = usize> {
        let start = self.ptr[i];
        let end = self.ptr[i + 1];
        start..end
    }

    /// Sequential iterator over the rows
    #[must_use]
    pub fn rows(&self) -> impl ExactSizeIterator<Item = &[usize]> {
        (0..self.n()).map(|i| self.row(i))
    }

    /// Sequential iterator over the rows and values
    #[must_use]
    pub fn row_and_values(&self, i: usize) -> (&[usize], &[usize]) {
        let start = self.ptr[i];
        let end = self.ptr[i + 1];
        (
            &self.indices[start..end],
            &self.values.as_ref().unwrap()[start..end],
        )
    }

    #[must_use]
    fn node_degrees(&self) -> Vec<usize> {
        let mut res = Vec::with_capacity(self.n());
        for (i_row, row) in self.rows().enumerate() {
            let mut n = row.len();
            if row.contains(&i_row) {
                n += 1;
            }
            res.push(n);
        }
        res
    }

    /// Compute the Reverse Cuthill McKee ordering
    #[must_use]
    pub fn reverse_cuthill_mckee(&self) -> Vec<usize> {
        // strongly inspired from scipy
        let mut order = vec![0; self.n()];
        let degree = self.node_degrees();
        let inds = argsort(&degree);
        let mut flg = vec![true; self.n()];
        let rev_inds = argsort(&inds);
        let mut tmp_degrees = vec![0; degree.iter().copied().max().unwrap()];
        let mut n = 0;

        for idx in 0..self.n() {
            if flg[idx] {
                let seed = inds[idx];
                order[n] = seed;
                n += 1;
                flg[rev_inds[seed]] = false;
                let mut level_start = n - 1;
                let mut level_end = n;

                while level_start < level_end {
                    for level in level_start..level_end {
                        let i = order[level];
                        let n_old = n;

                        for &j in self.row(i) {
                            if flg[rev_inds[j]] {
                                flg[rev_inds[j]] = false;
                                order[n] = j;
                                n += 1;
                            }
                        }

                        let mut level_len = 0;
                        for k in n_old..n {
                            tmp_degrees[level_len] = degree[order[k]];
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

            if n == self.n() {
                break;
            }
        }

        // return reversed order for RCM ordering
        order.iter().rev().copied().collect()
    }

    /// Compute the connected components
    pub fn connected_components(&self) -> Result<Vec<usize>> {
        Ok(ConnectedComponents::new(self)?.vtag)
    }

    /// Extract a sub-graph
    #[must_use]
    pub fn subgraph(&self, ids: impl IntoIterator<Item = usize>) -> Self {
        let mut new_ids = vec![usize::MAX; self.n()];
        let mut m = 0;
        for (i, j) in ids.into_iter().enumerate() {
            new_ids[j] = i;
            m += 1;
        }
        let mut ptr = vec![0];
        let mut indices = Vec::new();
        let mut values = Vec::new();
        for (old_i, &new_i) in new_ids.iter().enumerate() {
            if new_i != usize::MAX {
                for k in self.row_ptr(old_i) {
                    let old_j = self.indices[k];
                    let new_j = new_ids[old_j];
                    if new_j != usize::MAX {
                        indices.push(new_j);
                        if let Some(v) = &self.values {
                            values.push(v[k]);
                        }
                    }
                }
                ptr.push(indices.len());
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
            m,
        }
    }
}

/// Connected components of a graph
#[derive(Debug)]
struct ConnectedComponents {
    pub vtag: Vec<usize>,
}

impl ConnectedComponents {
    /// Compute the connected components of a CSR graph
    pub fn new(g: &CSRGraph) -> Result<Self> {
        assert_eq!(g.n(), g.m());

        let mut res = Self {
            vtag: vec![usize::MAX; g.n()],
        };
        res.compute(g)?;
        Ok(res)
    }

    fn compute_from(&mut self, g: &CSRGraph, starts: &[usize], component: usize) {
        let mut next_starts = Vec::new();
        for &start in starts {
            self.vtag[start] = component;
            for i in g.row(start).iter().copied() {
                if self.vtag[i] == usize::MAX {
                    self.vtag[i] = component;
                    next_starts.push(i);
                }
            }
        }
        if !starts.is_empty() {
            self.compute_from(g, &next_starts, component);
        }
    }

    fn compute(&mut self, g: &CSRGraph) -> Result<()> {
        let mut start = 0;
        let mut component = 0;
        while start < g.n() {
            self.compute_from(g, &[start], component);
            while start < g.n() && self.vtag[start] < usize::MAX {
                start += 1;
            }
            component += 1;
            if component == usize::MAX {
                return Err(Error::from("too many connected components"));
            }
            assert!(component < usize::MAX);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::graph::{CSRGraph, reindex};

    #[test]
    fn test_reindex() {
        let g = [[0, 1], [1, 2], [2, 0], [5, 6]];
        let (new_elems, _) = reindex(&g);
        assert_eq!(new_elems.iter().copied().flatten().max().unwrap(), 4);
        assert_eq!(new_elems.len(), 4);
    }

    #[test]
    fn test_csr_edges() {
        let g = [[0, 1], [1, 2], [2, 0], [3, 4]];
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
        let g = [[0_i16, 1_i16, 2_i16], [0_i16, 2_i16, 3_i16]];

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
        let graph = CSRGraph::new(
            vec![0, 1, 2, 5, 5, 7, 8, 8, 9, 10, 12],
            vec![1, 4, 2, 3, 8, 2, 9, 2, 5, 6, 6, 7],
        );
        let ids = graph.reverse_cuthill_mckee();
        // computed using the scipy code and forcing stable sort
        assert_eq!(ids, [5, 8, 7, 2, 9, 4, 1, 0, 6, 3]);
    }

    #[test]
    fn test_cc() {
        let g = [[0, 1], [1, 2], [2, 0], [3, 4]];
        let g = CSRGraph::from_edges(g.iter().copied(), None);
        let cc = g.connected_components().unwrap();
        assert_eq!(cc, [0, 0, 0, 1, 1]);
    }
}
