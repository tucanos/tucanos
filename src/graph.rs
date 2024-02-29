use crate::{topo_elems::Elem, Idx};
use rustc_hash::FxHashMap;
use std::{collections::hash_map::Entry, hash::BuildHasherDefault};

// pub trait Connectivity {
//     fn elem(&self, i: Idx) -> &[Idx];
//     fn n_elems(&self) -> Idx;
//     fn max_node(&self) -> Idx;
//     fn n(&self) -> Idx;
// }

// #[derive(Debug, Default, Clone)]
// pub struct ElemGraph<E: Elem> {
//     pub n: Idx,
//     pub elems: Vec<E>,
// }

// impl ElemGraph<E: Elem> {
//     #[must_use]
//     pub fn new(n: Idx, n_elems: Idx) -> Self {
//         Self {
//             n,
//             elems: Vec::with_capacity((n * n_elems) as usize),
//         }
//     }

//     pub fn from<I: Iterator<Item = Idx>>(n: Idx, els: I) -> Self {
//         let elems = els.collect::<Vec<_>>();
//         debug_assert_eq!(elems.len() % n as usize, 0);
//         Self { n, elems }
//     }

//     pub fn add_elem(&mut self, e: &[Idx]) {
//         debug_assert_eq!(e.len(), self.n as usize);
//         self.elems.extend(e.iter());
//     }

//     pub fn reindex(&mut self) -> FxHashMap<Idx, Idx> {
//         let mut map = FxHashMap::with_hasher(BuildHasherDefault::default());
//         let mut next = 0;
//         for e in &mut self.elems {
//             if let Some(i0) = map.get(e) {
//                 *e = *i0;
//             } else {
//                 map.insert(*e, next);
//                 *e = next;
//                 next += 1;
//             }
//         }
//         map
//     }
// }

// impl Connectivity for ElemGraph {
//     fn elem(&self, i: Idx) -> &[Idx] {
//         let start = (self.n * i) as usize;
//         let end = start + self.n as usize;
//         &self.elems[start..end]
//     }

//     fn n_elems(&self) -> Idx {
//         debug_assert_eq!(self.elems.len() % self.n as usize, 0);
//         (self.elems.len() / self.n as usize) as Idx
//     }

//     fn max_node(&self) -> Idx {
//         self.elems.iter().copied().max().unwrap_or(0)
//     }

//     fn n(&self) -> Idx {
//         self.n
//     }
// }

// #[derive(Debug)]
// pub struct ElemGraphInterface<'a> {
//     n: Idx,
//     elems: &'a [Idx],
// }

// impl<'a> ElemGraphInterface<'a> {
//     #[must_use]
//     pub fn new(n: Idx, elems: &'a [Idx]) -> Self {
//         Self { n, elems }
//     }
// }

// impl<'a> Connectivity for ElemGraphInterface<'a> {
//     fn elem(&self, i: Idx) -> &[Idx] {
//         let start = (self.n * i) as usize;
//         let end = start + self.n as usize;
//         &self.elems[start..end]
//     }

//     fn n_elems(&self) -> Idx {
//         debug_assert_eq!(self.elems.len() % self.n as usize, 0);
//         (self.elems.len() / self.n as usize) as Idx
//     }

//     fn max_node(&self) -> Idx {
//         self.elems.iter().copied().max().unwrap()
//     }

//     fn n(&self) -> Idx {
//         self.n
//     }
// }

/// Renumber the vertices in order to have contininuous indices, and return he map from old to nex indices
pub fn reindex<E: Elem>(elems: &[E]) -> (Vec<E>, FxHashMap<Idx, Idx>) {
    let mut map = FxHashMap::with_hasher(BuildHasherDefault::default());
    let mut next = 0 as Idx;
    for i in elems.iter().copied().flatten() {
        if let Entry::Vacant(e) = map.entry(i) {
            e.insert(next);
            next += 1;
        }
    }

    let new_elems = elems
        .iter()
        .map(|e| E::from_iter(e.iter().map(|&i| *map.get(&i).unwrap())))
        .collect();

    (new_elems, map)
}

#[derive(Debug, Default, Clone)]
pub struct CSRGraph {
    pub ptr: Vec<Idx>,
    pub indices: Vec<Idx>,
    pub m: Idx,
}

impl CSRGraph {
    fn set_ptr<E: IntoIterator<Item = Idx> + Copy>(elems: &[E]) -> Self {
        let nv = elems.iter().copied().flatten().max().unwrap_or(0) as usize + 1;
        let n = elems.iter().copied().flatten().count();

        let mut res = Self {
            ptr: vec![0; nv + 1],
            indices: vec![Idx::MAX; n],
            m: 0,
        };

        for i in elems.iter().copied().flatten() {
            res.ptr[i as usize + 1] += 1;
        }

        for i in 0..nv {
            res.ptr[i + 1] += res.ptr[i];
        }

        res
    }

    pub fn sort(&mut self) {
        let n = self.ptr.len() - 1;
        for i in 0..n {
            let start = self.ptr[i] as usize;
            let end = self.ptr[i + 1] as usize;
            self.indices[start..end].sort_unstable();
        }
    }

    #[must_use]
    pub fn new(edgs: &[[Idx; 2]]) -> Self {
        let mut res = Self::set_ptr(edgs);
        res.m = res.n();

        for e in edgs {
            let i0 = e[0] as usize;
            let i1 = e[1] as usize;
            let mut ok = false;
            for j in res.ptr[i0]..res.ptr[i0 + 1] {
                if res.indices[j as usize] == Idx::MAX {
                    res.indices[j as usize] = i1 as Idx;
                    ok = true;
                    break;
                }
            }
            assert!(ok);
            let mut ok = false;
            for j in res.ptr[i1]..res.ptr[i1 + 1] {
                if res.indices[j as usize] == Idx::MAX {
                    res.indices[j as usize] = i0 as Idx;
                    ok = true;
                    break;
                }
            }
            assert!(ok);
        }
        res.sort();
        res
    }

    pub fn transpose<E: Elem>(elems: &[E]) -> Self {
        let mut res = Self::set_ptr(elems);
        res.m = elems.len() as Idx;

        for (i, e) in elems.iter().enumerate() {
            for i_vert in e.iter().copied() {
                let start = res.ptr[i_vert as usize];
                let end = res.ptr[i_vert as usize + 1];
                let mut ok = false;
                for j in start..end {
                    if res.indices[j as usize] == Idx::MAX {
                        res.indices[j as usize] = i as Idx;
                        ok = true;
                        break;
                    }
                }
                assert!(ok);
            }
        }
        res.sort();
        res
    }

    #[must_use]
    pub fn n(&self) -> Idx {
        self.ptr.len() as Idx - 1
    }

    #[must_use]
    pub const fn m(&self) -> Idx {
        self.m
    }

    #[must_use]
    pub fn n_edges(&self) -> Idx {
        self.indices.len() as Idx
    }

    #[must_use]
    pub fn row(&self, i: Idx) -> &[Idx] {
        let start = self.ptr[i as usize] as usize;
        let end = self.ptr[i as usize + 1] as usize;
        &self.indices[start..end]
    }
}

#[derive(Debug)]
pub struct ConnectedComponents {
    vtag: Vec<u16>,
}

impl ConnectedComponents {
    #[must_use]
    pub fn new(g: &CSRGraph) -> Self {
        assert_eq!(g.n(), g.m());

        let mut res = Self {
            vtag: vec![u16::MAX; g.n() as usize],
        };
        res.compute(g);
        res
    }

    #[must_use]
    pub fn tags(&self) -> &[u16] {
        &self.vtag
    }

    fn compute_from(&mut self, g: &CSRGraph, start: Idx, component: u16) {
        for i in g.row(start).iter().copied() {
            if self.vtag[i as usize] == u16::MAX {
                self.vtag[i as usize] = component;
                self.compute_from(g, i, component);
            }
        }
    }

    fn compute(&mut self, g: &CSRGraph) {
        let mut start = 0;
        let mut component = 0;
        while start < g.n() {
            self.compute_from(g, start, component);
            while start < g.n() && self.vtag[start as usize] != u16::MAX {
                start += 1;
            }
            component += 1;
            assert!(component < u16::MAX);
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::topo_elems::{Edge, Triangle};

    use super::*;

    #[test]
    fn test_reindex() {
        let g = vec![
            Edge::from_slice(&[0, 1]),
            Edge::from_slice(&[1, 2]),
            Edge::from_slice(&[2, 0]),
            Edge::from_slice(&[5, 6]),
        ];
        let (new_elems, _) = reindex(&g);
        assert_eq!(new_elems.iter().copied().flatten().max().unwrap(), 4);
        assert_eq!(new_elems.len(), 4);
    }

    #[test]
    fn test_csr_edges() {
        let g = vec![[0, 1], [1, 2], [2, 0], [3, 4]];
        let g = CSRGraph::new(&g);
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
        let g = vec![
            Triangle::from_slice(&[0, 1, 2]),
            Triangle::from_slice(&[0, 2, 3]),
        ];

        let g = CSRGraph::transpose(&g);
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
    fn test_cc() {
        let g = vec![[0, 1], [1, 2], [2, 0], [3, 4]];
        let g = CSRGraph::new(&g);
        let cc = ConnectedComponents::new(&g);
        assert_eq!(cc.tags(), [0, 0, 0, 1, 1]);
    }
}
